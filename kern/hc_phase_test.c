/*
 * Persistent HC regression checks on one physical GPU.
 *
 *   hc_phase_test --baseline old-die.cubin [--candidate die.cubin]
 *
 * HCWSUM: independent fixed-order C sum + hc_write_gpu, n tails around
 * 160/256-column boundaries, logical die counts 1..6, separate/in-place SIN.
 * BF16HCX/HCRX/HCRWX: full-output differential against the baseline module.
 * HCRWX SOUT also checks hc_write_gpu independently.
 *
 * Six destination slots are separate local allocations. This tests fanout
 * addressing, not physical PCIe transfers or six-GPU synchronization.
 * Descriptor exchange flags are zero: dispatch, bodies and grid barriers run,
 * but this does not test peer publication/wait protocols.
 *
 * All buffers have prefix/suffix canaries; output holes retain poison.
 * Every case is repeated with a different poison and fresh frozen inputs.
 *
 * Acquire the shared validation lock and check GPU/process availability before
 * invoking this program. It does not acquire host locks or touch other engines.
 */
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "hc.h"
#include "dieops.h"

CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);

#define CK(call) do {                                                \
    CUresult result_ = (call);                                        \
    if (result_ != CUDA_SUCCESS) {                                    \
        const char *message_ = "?";                                   \
        cuGetErrorString(result_, &message_);                         \
        fprintf(stderr, "%s:%d: %s: %s\n",                            \
                __FILE__, __LINE__, #call, message_);                 \
        exit(1);                                                     \
    }                                                                \
} while (0)

enum { GUARD_WORDS = 32, WIDTH = 2560, STREAM = 10240 };
static const uint32_t guard_value = 0x5a91ed37u;
static uint32_t poison = 0x7fc05301u;
static uint32_t rng = 53001u;
static CUdeviceptr program_device, control_device;

typedef struct {
    CUdeviceptr base, data;
    size_t words;
    const char *name;
} buffer;

static void *allocate(size_t bytes)
{
    void *p = malloc(bytes);
    if (!p) {
        fprintf(stderr, "malloc failed for %zu bytes\n", bytes);
        exit(2);
    }
    return p;
}

static float sample(void)
{
    rng = rng * 1664525u + 1013904223u;
    return ((int)((rng >> 8) & 65535u) - 32768) / 32768.f;
}

static buffer device_buffer(size_t words, const char *name)
{
    buffer b;
    b.words = words;
    b.name = name;
    CK(cuMemAlloc(&b.base, 4 * (words + 2 * GUARD_WORDS)));
    b.data = b.base + 4 * GUARD_WORDS;
    return b;
}

static void seed(buffer b, const void *data)
{
    CK(cuMemsetD32(b.base, guard_value, b.words + 2 * GUARD_WORDS));
    if (data)
        CK(cuMemcpyHtoD(b.data, data, 4 * b.words));
    else
        CK(cuMemsetD32(b.data, poison, b.words));
}

static uint32_t *capture(buffer b)
{
    uint32_t *p = allocate(4 * (b.words + 2 * GUARD_WORDS));
    CK(cuMemcpyDtoH(p, b.base, 4 * (b.words + 2 * GUARD_WORDS)));
    return p;
}

/* expected describes the entire payload, including untouched interior holes. */
static int check(buffer b, const void *expected)
{
    uint32_t *got = capture(b);
    size_t bad = 0, guards = 0;
    for (size_t i = 0; i < GUARD_WORDS; i++) {
        guards += got[i] != guard_value;
        guards += got[GUARD_WORDS + b.words + i] != guard_value;
    }
    if (expected) {
        for (size_t i = 0; i < b.words; i++) {
            uint32_t want;
            memcpy(&want, (const unsigned char *)expected + 4 * i, 4);
            if (got[GUARD_WORDS + i] != want) {
                if (bad < 3)
                    printf("  %s[%zu]: got=%08x want=%08x\n",
                           b.name, i, got[GUARD_WORDS + i], want);
                bad++;
            }
        }
    }
    if (bad || guards)
        printf("  %s: mismatches=%zu guard_errors=%zu\n",
               b.name, bad, guards);
    free(got);
    return bad != 0 || guards != 0;
}

/* A baseline and candidate that both fail to write must not pass differential. */
static int check_layout(buffer b, size_t begin, size_t count)
{
    uint32_t *got = capture(b);
    size_t bad = 0;
    for (size_t i = 0; i < b.words; i++) {
        uint32_t bits = got[GUARD_WORDS + i];
        if (i >= begin && i - begin < count) {
            float value;
            memcpy(&value, &bits, 4);
            bad += bits == poison || !isfinite(value);
        } else {
            bad += bits != poison;
        }
    }
    if (bad)
        printf("  %s: missing/nonfinite writes or modified holes=%zu\n",
               b.name, bad);
    free(got);
    return (bad != 0) | check(b, NULL);
}

static void release(buffer b)
{
    CK(cuMemFree(b.base));
}

static void ptr(uint32_t *d, unsigned address, CUdeviceptr p)
{
    unsigned i = 2 + (address - 0x140) / 4;
    d[i] = (uint32_t)p;
    d[i + 1] = (uint32_t)(p >> 32);
}

static void word(uint32_t *d, unsigned address, uint32_t value)
{
    d[2 + (address - 0x140) / 4] = value;
}

static void real_word(uint32_t *d, unsigned address, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, 4);
    word(d, address, bits);
}

static void launch(CUfunction fn, const uint32_t descriptor[32])
{
    unsigned phases = 1, tokens = 1;
    uint32_t control[32] = {0};
    control[4] = 1; /* One physical participant; no exchange flags are set. */
    CK(cuMemsetD8(control_device, 0, 8192));
    CK(cuMemcpyHtoD(control_device, control, sizeof control));
    CK(cuMemcpyHtoD(program_device, descriptor, 128));
    void *args[] = {
        &program_device, &phases, &control_device, &tokens
    };
    CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, args, NULL));
    CK(cuCtxSynchronize());
}

static int sum_case(CUfunction baseline, CUfunction candidate,
                    unsigned n, unsigned dies, int inplace, unsigned round)
{
    const unsigned stride = WIDTH + 32;
    float inj[4] = {-80.f, -0.f, 0.f, 80.f};
    float *parts = allocate(4 * 6 * stride);
    float *sin = allocate(4 * STREAM);
    float *sum = allocate(4 * WIDTH);
    float *updated = allocate(4 * STREAM);
    uint32_t *expected_s = allocate(4 * STREAM);
    uint32_t *expected_out = allocate(4 * WIDTH);
    static const float cancellation[6] = {
        0x1p24f, 1.f, -0x1p24f, 0.5f, -0.25f, 0.125f
    };
    if (round) {
        inj[0] = 3.75f;
        inj[1] = -8.5f;
        inj[2] = 0.03125f;
        inj[3] = -0.0625f;
    }
    for (unsigned d = 0; d < 6; d++)
        for (unsigned i = 0; i < stride; i++)
            parts[d * stride + i] =
                (i & 1) ? sample() * (float)(d + 1) : cancellation[d];
    for (unsigned i = 0; i < STREAM; i++)
        sin[i] = sample() * 2.f;
    for (unsigned i = 0; i < WIDTH; i++) {
        float value = parts[i];
        for (unsigned d = 1; d < dies; d++)
            value = value + parts[d * stride + i];
        sum[i] = value;
        expected_out[i] = poison;
    }
    memcpy(updated, sin, 4 * STREAM);
    hc_write_gpu(inj, sum, updated);
    for (unsigned i = 0; i < STREAM; i++) {
        expected_s[i] = poison;
        if (inplace)
            memcpy(expected_s + i, sin + i, 4);
    }
    for (unsigned i = 0; i < n; i++) {
        memcpy(expected_out + i, sum + i, 4);
        for (unsigned lane = 0; lane < 4; lane++)
            memcpy(expected_s + lane * WIDTH + i,
                   updated + lane * WIDTH + i, 4);
    }

    buffer di = device_buffer(4, "inj");
    buffer dp = device_buffer(6 * stride, "parts");
    buffer ds = device_buffer(STREAM, "SIN");
    buffer dy = device_buffer(STREAM, "HCWSUM.S");
    buffer dout = device_buffer(WIDTH, "HCWSUM.OUT");
    int failures = 0;
    for (unsigned version = 0; version < 2; version++) {
        seed(di, inj);
        seed(dp, parts);
        seed(ds, sin);
        seed(dy, inplace ? sin : NULL);
        seed(dout, NULL);
        uint32_t d[32] = {0};
        unsigned columns = version ? 160 : 256;
        d[0] = OP_HCWSUM;
        d[1] = (n + columns - 1) / columns;
        ptr(d, 0x140, di.data);
        ptr(d, 0x148, dp.data);
        ptr(d, 0x150, dy.data);
        word(d, 0x158, n);
        word(d, 0x15c, 4 * stride);
        word(d, 0x164, dies);
        ptr(d, 0x168, dout.data);
        ptr(d, 0x170, inplace ? dy.data : ds.data);
        launch(version ? candidate : baseline, d);
        failures += check(dy, expected_s);
        failures += check(dout, expected_out);
        failures += check(di, inj);
        failures += check(dp, parts);
        failures += check(ds, sin);
    }
    printf("hcwsum n=%u dies=%u inplace=%d round=%u: %s\n",
           n, dies, inplace, round, failures ? "FAIL" : "BIT-EXACT");
    release(dout); release(dy); release(ds); release(dp); release(di);
    free(expected_out); free(expected_s); free(updated);
    free(sum); free(sin); free(parts);
    return failures != 0;
}

static int xout_case(CUfunction baseline, CUfunction candidate,
                     unsigned op, const char *name,
                     unsigned rows, unsigned units, unsigned round)
{
    const unsigned offset = round ? 54 : 1;
    int rms = op != OP_BF16HCX;
    int write_stream = op == OP_BF16HCRWX;
    size_t weight_elems = (size_t)rows * STREAM;
    uint16_t *weights = allocate(2 * weight_elems);
    float *stream = allocate(4 * STREAM);
    float *norm = allocate(4 * STREAM);
    float *y = allocate(4 * WIDTH);
    float *written = allocate(4 * STREAM);
    float inj[4] = {-7.f, -0.f, 0.f, 5.f};
    for (size_t i = 0; i < weight_elems; i++) {
        float value = sample() * 0.125f;
        uint32_t bits;
        memcpy(&bits, &value, 4);
        weights[i] = (uint16_t)(bits >> 16);
    }
    for (unsigned i = 0; i < STREAM; i++) {
        stream[i] = sample() * 2.f;
        norm[i] = 1.f + sample() * 0.25f;
    }
    for (unsigned i = 0; i < WIDTH; i++)
        y[i] = sample();
    memcpy(written, stream, 4 * STREAM);
    hc_write_gpu(inj, y, written);

    buffer dw = device_buffer(weight_elems / 2, "W");
    buffer dx = device_buffer(STREAM, "S");
    buffer dn = device_buffer(STREAM, "norm");
    buffer dy = device_buffer(WIDTH, "Y");
    buffer di = device_buffer(4, "inj");
    buffer dt = device_buffer(12, "destination table");
    buffer outputs[8];
    outputs[0] = device_buffer(STREAM, "XH");
    outputs[1] = device_buffer(STREAM, "SOUT");
    CUdeviceptr addresses[6];
    for (unsigned i = 0; i < 6; i++) {
        outputs[2 + i] = device_buffer(offset + rows + 32, "XOUT slot");
        addresses[i] = outputs[2 + i].data;
    }
    uint32_t *saved[8] = {0};
    int failures = 0;
    for (unsigned version = 0; version < 2; version++) {
        seed(dw, weights); seed(dx, stream); seed(dn, norm);
        seed(dy, y); seed(di, inj); seed(dt, addresses);
        for (unsigned i = 0; i < 8; i++)
            seed(outputs[i], NULL);
        uint32_t d[32] = {0};
        d[0] = op;
        d[1] = units;
        ptr(d, 0x140, dw.data);
        ptr(d, 0x148, dx.data);
        ptr(d, 0x150, dt.data);
        word(d, 0x158, rows);
        ptr(d, 0x160, outputs[1].data);
        word(d, 0x168, 4 * offset);
        ptr(d, 0x178, outputs[0].data);
        ptr(d, 0x190, dn.data);
        real_word(d, 0x198, 1e-6f);
        ptr(d, 0x1a0, dy.data);
        ptr(d, 0x1a8, di.data);
        launch(version ? candidate : baseline, d);

        failures += check_layout(outputs[0], 0, rms ? STREAM : 0);
        failures += check_layout(outputs[1], 0, write_stream ? STREAM : 0);
        if (write_stream)
            failures += check(outputs[1], written);
        for (unsigned i = 2; i < 8; i++)
            failures += check_layout(outputs[i], offset, rows);
        for (unsigned i = 0; i < 8; i++) {
            if (!version)
                saved[i] = capture(outputs[i]);
            else
                failures += check(outputs[i], saved[i] + GUARD_WORDS);
        }
        failures += check(dw, weights);
        failures += check(dx, stream);
        failures += check(dn, norm);
        failures += check(dy, y);
        failures += check(di, inj);
        failures += check(dt, addresses);
    }
    printf("%s rows=%u units=%u offset=%u round=%u: %s\n",
           name, rows, units, offset, round,
           failures ? "FAIL" : "BIT-EXACT");
    for (unsigned i = 0; i < 8; i++) {
        free(saved[i]);
        release(outputs[i]);
    }
    release(dt); release(di); release(dy);
    release(dn); release(dx); release(dw);
    free(written); free(y); free(norm); free(stream); free(weights);
    return failures != 0;
}

int main(int argc, char **argv)
{
    const char *baseline_path = NULL, *candidate_path = "die.cubin";
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--baseline") && i + 1 < argc)
            baseline_path = argv[++i];
        else if (!strcmp(argv[i], "--candidate") && i + 1 < argc)
            candidate_path = argv[++i];
        else {
            fprintf(stderr,
                    "usage: %s --baseline cubin [--candidate cubin]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!baseline_path) {
        fprintf(stderr, "--baseline is required\n");
        return 2;
    }
    CK(cuInit(0));
    CUdevice device;
    CUcontext context;
    CUmodule baseline_module, candidate_module;
    CUfunction baseline, candidate;
    CK(cuDeviceGet(&device, 0));
    CK(cuCtxCreate_v2(&context, CU_CTX_SCHED_SPIN, device));
    char name[128];
    int major, minor, sms;
    CK(cuDeviceGetName(name, sizeof name, device));
    CK(cuDeviceGetAttribute(&major,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                           device));
    CK(cuDeviceGetAttribute(&minor,
                           CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                           device));
    CK(cuDeviceGetAttribute(&sms,
                           CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    printf("GPU: %s sm_%d%d SMs=%d; local fanout only\n",
           name, major, minor, sms);
    if (major != 5 || minor != 2 || sms < 16) {
        fprintf(stderr, "this test requires sm_52 with at least 16 SMs\n");
        CK(cuCtxDestroy(context));
        return 2;
    }
    printf("baseline=%s\ncandidate=%s\n", baseline_path, candidate_path);
    CK(cuModuleLoad(&baseline_module, baseline_path));
    CK(cuModuleGetFunction(&baseline, baseline_module, "die"));
    CK(cuModuleLoad(&candidate_module, candidate_path));
    CK(cuModuleGetFunction(&candidate, candidate_module, "die"));
    int occupancy;
    CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(
        &occupancy, candidate, 256, 0));
    if (occupancy < 1) {
        fprintf(stderr, "candidate cannot keep the grid resident\n");
        return 2;
    }
    CK(cuMemAlloc(&program_device, 128));
    CK(cuMemAlloc(&control_device, 8192));

    const unsigned sizes[] = {
        1, 31, 33, 159, 160, 161, 255, 257, 2559, 2560
    };
    const unsigned rms_rows[] = {1, 53, 54, 55, 137, 324};
    const unsigned plain_rows[] = {1, 137, 513};
    const unsigned plain_units[] = {1, 16, 33};
    int failures = 0;
    unsigned cases = 0;
    for (unsigned round = 0; round < 2; round++) {
        poison = 0x7fc05301u + round;
        for (unsigned dies = 1; dies <= 6; dies++) {
            for (unsigned j = 0; j < sizeof sizes / sizeof sizes[0]; j++) {
                if (dies != 6 && sizes[j] != WIDTH)
                    continue;
                for (int inplace = 0; inplace <= 1; inplace++) {
                    failures += sum_case(baseline, candidate, sizes[j],
                                         dies, inplace, round);
                    cases++;
                }
            }
        }
        for (unsigned j = 0; j < sizeof rms_rows / sizeof rms_rows[0]; j++) {
            failures += xout_case(baseline, candidate,
                                  OP_BF16HCRX, "bf16hcrx",
                                  rms_rows[j], 16, round);
            failures += xout_case(baseline, candidate,
                                  OP_BF16HCRWX, "bf16hcrwx",
                                  rms_rows[j], 16, round);
            cases += 2;
        }
        for (unsigned j = 0; j < sizeof plain_rows / sizeof plain_rows[0]; j++)
            for (unsigned k = 0;
                 k < sizeof plain_units / sizeof plain_units[0]; k++) {
                failures += xout_case(baseline, candidate,
                                      OP_BF16HCX, "bf16hcx",
                                      plain_rows[j], plain_units[k], round);
                cases++;
            }
    }
    CK(cuMemFree(control_device));
    CK(cuMemFree(program_device));
    CK(cuModuleUnload(candidate_module));
    CK(cuModuleUnload(baseline_module));
    CK(cuCtxDestroy(context));
    printf("persistent HC: %u cases, %d failed; "
           "single-GPU correctness, not throughput acceptance\n",
           cases, failures);
    return failures ? 3 : 0;
}
