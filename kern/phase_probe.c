#include "phase_probe.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint64_t *observations;
static CUdeviceptr mapped[PHASE_PROBE_DIES];
static phase_probe_descriptor descriptors[PHASE_PROBE_DIES][PHASE_PROBE_PHASES];
static size_t descriptor_count[PHASE_PROBE_DIES];
static unsigned die_count;
static int segment_active;
static uint64_t request_id;
static uint32_t segment_first, segment_end;

int phase_probe_decode(uint64_t tag, const phase_probe_descriptor *program,
                       size_t count, uint32_t first, uint32_t end,
                       phase_probe_record *out) {
    if (!tag || !program || !out || !count || count > PHASE_PROBE_PHASES || first >= end)
        return -1;
    uint32_t low = (uint32_t)tag, encoded_index = low >> 5;
    uint32_t mode = low & 31u, token = (uint32_t)(tag >> 32);
    if (!encoded_index || encoded_index > count || token < first || token >= end ||
        (mode != 0 && mode != 8 && mode != 24)) return -1;
    uint32_t primary = encoded_index - 1, physical = primary;
    if (primary && (program[primary - 1].flags & 8u)) return -1;
    uint32_t flags = program[primary].flags & 24u;
    if (mode == 24) {
        if (flags != 24 || primary + 1 >= count || token + 1u >= end) return -1;
    } else if (mode == 8) {
        if (flags != 8 || primary + 1 >= count || token + 1u >= end) return -1;
    } else if (flags == 8) {
        if (primary + 1 >= count || token + 1u >= end) return -1;
        physical++; token++;
    } else if (flags != 0) return -1;
    *out = (phase_probe_record){
        .base_token = (uint32_t)(tag >> 32), .token = token,
        .primary_index = primary, .physical_index = physical, .mode = mode,
        .descriptor = program[physical]
    };
    return 0;
}

CUresult phase_probe_create(const CUcontext *contexts, unsigned count) {
    if (observations || !contexts || !count || count > PHASE_PROBE_DIES)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult result = cuCtxSetCurrent(contexts[0]);
    if (result != CUDA_SUCCESS) return result;
    result = cuMemHostAlloc((void **)&observations,
                           PHASE_PROBE_DIES * PHASE_PROBE_CTAS * sizeof *observations,
                           CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE);
    if (result != CUDA_SUCCESS) return result;
    die_count = count;
    memset(observations, 0, PHASE_PROBE_DIES * PHASE_PROBE_CTAS * sizeof *observations);
    for (unsigned die = 0; die < count; die++) {
        result = cuCtxSetCurrent(contexts[die]);
        if (result != CUDA_SUCCESS) return result;
        result = cuMemHostGetDevicePointer(&mapped[die],
                    observations + die * PHASE_PROBE_CTAS, 0);
        if (result != CUDA_SUCCESS) return result;
    }
    return cuCtxSetCurrent(contexts[0]);
}

void phase_probe_destroy(void) {
    if (observations && segment_active)
        phase_probe_fault(CUDA_ERROR_ILLEGAL_STATE, "destroy_active_phase_probe", 0);
    if (observations) {
        CUresult result = cuMemFreeHost(observations);
        if (result != CUDA_SUCCESS)
            fprintf(stderr, "PHASE_PROBE_CLEANUP error=%d\n", (int)result);
    }
    observations = NULL; die_count = 0; segment_active = 0;
    memset(mapped, 0, sizeof mapped);
    memset(descriptor_count, 0, sizeof descriptor_count);
}

int phase_probe_enabled(void) { return observations != NULL; }
CUdeviceptr phase_probe_pointer(unsigned die) {
    return die < die_count ? mapped[die] : 0;
}
void phase_probe_upload_begin(void) { segment_active = 0; }

int phase_probe_program(unsigned die, const prog_t *program) {
    if (!observations) return 0;
    if (die >= die_count || !program || program->n < 1 ||
        (unsigned)program->n > PHASE_PROBE_PHASES || !program->p) return -1;
    descriptor_count[die] = (size_t)program->n;
    for (int i = 0; i < program->n; i++)
        descriptors[die][i] = (phase_probe_descriptor){
            program->p[i].w[0], program->p[i].w[1], program->p[i].w[30]
        };
    return 0;
}

void phase_probe_launch(uint64_t request, uint32_t first, uint32_t end) {
    if (!observations) return;
    if (first >= end) {
        fprintf(stderr, "PHASE_PROBE_INVALID_SEGMENT first=%u end=%u\n", first, end);
        fflush(NULL); _Exit(71);
    }
    request_id = request; segment_first = first; segment_end = end;
    memset(observations, 0, PHASE_PROBE_DIES * PHASE_PROBE_CTAS * sizeof *observations);
    segment_active = 1;
}

uint64_t phase_probe_observe(unsigned die, unsigned cta) {
    if (!observations || die >= die_count || cta >= PHASE_PROBE_CTAS) return 0;
    return __atomic_load_n(observations + die * PHASE_PROBE_CTAS + cta, __ATOMIC_RELAXED);
}

static void dump(CUresult error, const char *expression, int line) {
    fprintf(stderr, "PHASE_PROBE_BEGIN request=%" PRIu64 " active=%d segment_first=%u segment_end=%u window_first=%u window_end=%u error=%d expression=%s line=%d last_observation_only=1\n",
            request_id, segment_active, segment_first, segment_end,
            PHASE_PROBE_TOKEN_FIRST, PHASE_PROBE_TOKEN_END,
            (int)error, expression, line);
    for (unsigned die = 0; die < die_count; die++) {
        for (unsigned cta = 0; cta < PHASE_PROBE_CTAS; cta++) {
            uint64_t tag = phase_probe_observe(die, cta);
            phase_probe_record record;
            int valid = segment_active && !phase_probe_decode(tag, descriptors[die],
                             descriptor_count[die], segment_first, segment_end, &record);
            valid = valid && record.base_token >= PHASE_PROBE_TOKEN_FIRST &&
                             record.base_token < PHASE_PROBE_TOKEN_END;
            if (!valid) {
                fprintf(stderr, "PHASE_PROBE die=%u cta=%u raw=%016" PRIx64 " valid=0\n", die, cta, tag);
                continue;
            }
            const char *name = record.descriptor.opcode < OP_COUNT
                ? die_opname[record.descriptor.opcode] : "unknown";
            fprintf(stderr, "PHASE_PROBE die=%u cta=%u raw=%016" PRIx64 " valid=1 base_token=%u token=%u primary_phase=%u physical_phase=%u mode=%u opcode=%u name=%s units=%u flags=%u\n",
                    die, cta, tag, record.base_token, record.token, record.primary_index,
                    record.physical_index, record.mode, record.descriptor.opcode, name,
                    record.descriptor.units, record.descriptor.flags);
        }
    }
    fprintf(stderr, "PHASE_PROBE_END last_observation_only=1\n");
}

void phase_probe_done(void) {
    if (!observations) return;
    unsigned valid = 0, observed = 0;
    unsigned expected = 0;
    for (unsigned die = 0; die < die_count; die++) {
        int paired = 0;
        for (size_t phase = 0; phase < descriptor_count[die]; phase++)
            paired |= (descriptors[die][phase].flags & 8u) != 0;
        uint64_t first = segment_first > PHASE_PROBE_TOKEN_FIRST
                       ? segment_first : PHASE_PROBE_TOKEN_FIRST;
        /* A paired launch advances its kernel base by two, so a one-token
         * overlap at the lower boundary may still contain no recorded base. */
        if (paired && ((first - segment_first) & 1u)) first++;
        int in_window = first < segment_end && first < PHASE_PROBE_TOKEN_END;
        expected += in_window ? PHASE_PROBE_CTAS : 0;
        for (unsigned cta = 0; cta < PHASE_PROBE_CTAS; cta++) {
            phase_probe_record record;
            uint64_t tag = phase_probe_observe(die, cta);
            observed += tag != 0;
            valid += !phase_probe_decode(tag, descriptors[die],
                        descriptor_count[die], segment_first, segment_end, &record) &&
                     record.base_token >= PHASE_PROBE_TOKEN_FIRST &&
                     record.base_token < PHASE_PROBE_TOKEN_END;
        }
    }
    if (valid != expected || observed != expected) {
        dump(CUDA_SUCCESS, "missing_or_invalid_probe_records", 0);
        fflush(NULL); _Exit(71);
    }
    fprintf(stderr, "PHASE_PROBE_SEGMENT request=%" PRIu64 " first=%u end=%u window_first=%u window_end=%u valid=%u expected=%u last_observation_only=1\n",
            request_id, segment_first, segment_end, PHASE_PROBE_TOKEN_FIRST,
            PHASE_PROBE_TOKEN_END, valid, expected);
    segment_active = 0;
}

void phase_probe_fault(CUresult error, const char *expression, int line) {
    if (!observations) return;
    /* No CUDA call here: error 700 makes later CUDA operations unusable in
     * this process. Host snapshots may be stale; report that limit explicitly. */
    dump(error, expression, line);
    fflush(NULL); _Exit(70);
}

void phase_probe_limit(int line) {
    if (!observations) return;
    /* This is the kernel's self-limit, not a CUDA API error code. */
    dump(CUDA_SUCCESS, "persistent_kernel_time_limit", line);
    fflush(NULL); _Exit(72);
}
