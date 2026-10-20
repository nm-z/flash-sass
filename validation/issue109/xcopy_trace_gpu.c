/* Exercise the diagnostic XCOPY in the real persistent die kernel. The cubin
 * passed on argv[1] must have validation/issue109/xcopy_trace.sass loaded as
 * the xcopy opcode. */
#include "prog.h"

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);

#define CK(x) do { \
    CUresult r_ = (x); \
    if (r_ != CUDA_SUCCESS) { \
        const char *s_ = "?"; \
        cuGetErrorString(r_, &s_); \
        fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); \
        exit(2); \
    } \
} while (0)

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

enum {
    COPY_BYTES = 40960,
    TRACE_POSITIONS = 16,
    GUARD_BYTES = 64,
    CTL_BYTES = 8192,
    CTL_WORDS = CTL_BYTES / 4,
    POS_WORD = 6
};

#define POS_MASK (1u << POS_WORD)

typedef struct {
    unsigned pos;
    const uint8_t *src;
} expected_copy;

static CUdeviceptr device_alloc(size_t bytes) {
    CUdeviceptr p;
    CK(cuMemAlloc(&p, bytes));
    return p;
}

static void add_positioned_copy(prog_t *p, unsigned token_offset,
                                CUdeviceptr src, CUdeviceptr dst) {
    p->loop = 1;
    p->token_offset = token_offset;
    phase_t *x = prog_add(p, OP_XCOPY, 10);
    ph_ptr(x, 0x140, src);
    ph_ptr(x, 0x148, dst);
    ph_tokv(p, x, 0x150, 0, 0);
}

static int launch(CUfunction fn, CUdeviceptr ctl, const prog_t *p,
                  unsigned start, unsigned end) {
    uint32_t host_ctl[CTL_WORDS] = {0};
    host_ctl[2] = start;
    host_ctl[9] = end - start;
    CK(cuMemcpyHtoD(ctl, host_ctl, sizeof host_ctl));

    CUdeviceptr dprog = device_alloc((size_t)p->n * sizeof(phase_t));
    CK(cuMemcpyHtoD(dprog, p->p, (size_t)p->n * sizeof(phase_t)));
    unsigned nph = (unsigned)p->n, ntok = end;
    void *args[] = { &dprog, &nph, &ctl, &ntok };
    CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(host_ctl, ctl, sizeof host_ctl));
    CK(cuMemFree(dprog));
    if (host_ctl[2] != end) {
        fprintf(stderr, "token loop stopped at %u; expected %u\n", host_ctl[2], end);
        return 1;
    }
    return 0;
}

static int check_trace(CUdeviceptr allocation, size_t total,
                       const expected_copy *copies, unsigned count,
                       const char *label) {
    uint8_t *got = malloc(total);
    if (!got) { fprintf(stderr, "readback allocation failed\n"); return 1; }
    CK(cuMemcpyDtoH(got, allocation, total));
    int bad = 0;
    for (size_t i = 0; i < total; i++) {
        uint8_t want = 0xa5;
        if (i >= (size_t)GUARD_BYTES && i < total - (size_t)GUARD_BYTES) {
            size_t trace_off = i - GUARD_BYTES;
            unsigned pos = (unsigned)(trace_off / COPY_BYTES);
            size_t in_slot = trace_off % COPY_BYTES;
            for (unsigned c = 0; c < count; c++) {
                if (copies[c].pos == pos) {
                    want = copies[c].src[in_slot];
                    break;
                }
            }
        }
        if (got[i] != want) {
            if (!bad) fprintf(stderr, "%s byte %zu: got %02x expected %02x\n",
                              label, i, got[i], want);
            bad++;
        }
    }
    free(got);
    return bad;
}

static void make_input(uint8_t *p, unsigned seed) {
    for (size_t i = 0; i < (size_t)COPY_BYTES; i++)
        p[i] = (uint8_t)(i * (17u + seed * 12u) + 31u * seed + 5u);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s diagnostic-die.cubin\n", argv[0]);
        return 2;
    }
    CK(cuInit(0));
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    CK(cuDeviceGet(&dev, 0));
    CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
    CK(cuModuleLoad(&mod, argv[1]));
    CK(cuModuleGetFunction(&fn, mod, "die"));
    int resident = 0;
    CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident, fn, 256, 0));
    CHECK(resident >= 1);

    const size_t source_total = GUARD_BYTES + (size_t)COPY_BYTES + GUARD_BYTES;
    const size_t trace_span = (size_t)TRACE_POSITIONS * COPY_BYTES;
    const size_t trace_total = GUARD_BYTES + trace_span + GUARD_BYTES;
    uint8_t *host_src[2] = { malloc(COPY_BYTES), malloc(COPY_BYTES) };
    CHECK(host_src[0] != NULL && host_src[1] != NULL);
    make_input(host_src[0], 1);
    make_input(host_src[1], 2);

    CUdeviceptr src_alloc[2] = { device_alloc(source_total), device_alloc(source_total) };
    CUdeviceptr src[2] = { src_alloc[0] + GUARD_BYTES, src_alloc[1] + GUARD_BYTES };
    CUdeviceptr trace = device_alloc(trace_total);
    CUdeviceptr dst = trace + GUARD_BYTES;
    CUdeviceptr ctl = device_alloc(CTL_BYTES);
    for (unsigned i = 0; i < 2u; i++) {
        CK(cuMemsetD8(src_alloc[i], 0x5a, source_total));
        CK(cuMemcpyHtoD(src[i], host_src[i], COPY_BYTES));
    }

    prog_t legacy, a, b, pair;
    prog_init(&legacy); prog_init(&a); prog_init(&b); prog_init(&pair);
    phase_t *old = prog_add(&legacy, OP_XCOPY, 10);
    ph_ptr(old, 0x140, src[0]);
    ph_ptr(old, 0x148, dst);
    CHECK(old->w[POS_WORD] == 0 && old->w[31] == 0);

    add_positioned_copy(&a, 0, src[0], dst);
    add_positioned_copy(&b, 1, src[1], dst);
    CHECK(a.p[0].w[31] == POS_MASK && b.p[0].w[31] == POS_MASK);
    CHECK(prog_pair(&pair, &a, &b) == 0);
    CHECK(pair.n == 2 && (pair.p[0].w[30] & 8u) != 0);

    int fails = 0;
    CK(cuMemsetD8(trace, 0xa5, trace_total));
    fails += launch(fn, ctl, &legacy, 0, 1);
    expected_copy zero = { 0, host_src[0] };
    fails += check_trace(trace, trace_total, &zero, 1, "legacy position 0");
    printf("xcopy default position=0 status=%s\n", fails ? "FAIL" : "PASS");

    const unsigned starts[] = { 0, 7, 14 };
    for (size_t i = 0; i < sizeof starts / sizeof starts[0]; i++) {
        unsigned start = starts[i];
        CHECK(start + 1 < (unsigned)TRACE_POSITIONS);
        CK(cuMemsetD8(trace, 0xa5, trace_total));
        int bad = launch(fn, ctl, &pair, start, start + 2);
        expected_copy copies[2] = {
            { start, host_src[0] }, { start + 1, host_src[1] }
        };
        bad += check_trace(trace, trace_total, copies, 2, "paired positions");
        printf("xcopy pair positions=%u,%u status=%s\n", start, start + 1,
               bad ? "FAIL" : "PASS");
        fails += bad;
    }

    printf("xcopy_trace_gpu: default + paired slots 0/1, 7/8, 14/15; %s\n",
           fails ? "FAIL" : "PASS");
    prog_free(&legacy); prog_free(&a); prog_free(&b); prog_free(&pair);
    CK(cuMemFree(src_alloc[0])); CK(cuMemFree(src_alloc[1]));
    CK(cuMemFree(trace)); CK(cuMemFree(ctl));
    free(host_src[0]); free(host_src[1]);
    CK(cuModuleUnload(mod));
    CK(cuCtxDestroy(ctx));
    return fails ? 1 : 0;
}
