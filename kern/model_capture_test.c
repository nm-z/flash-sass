/* CPU readback mock: exercises capture publication and vocabulary layout only. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { MAXD = 2, MAXB = 48 };
static int nd = 2, model_qsa = 1, model_host = 0;
static unsigned model_budget = 512;
static int ctx[MAXD], mock_read_failure;
typedef struct { uintptr_t hlo; } dn_bufs_t;
typedef struct { int o0, on; } split_t;
static int cuCtxSetCurrent(int context) { (void)context; return 0; }
static int cuMemcpyDtoH(void *out, uintptr_t in, size_t bytes) {
    if (mock_read_failure) return 43;
    memcpy(out, (const void *)in, bytes);
    return 0;
}
#define CK(call) do { if ((call) != 0) { fputs("mock readback failure\n", stderr); exit(4); } } while (0)
#include "model_capture.h"

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const char *mode = argv[2];
    if (strcmp(mode, "disabled") && setenv("MODEL_CAPTURE", argv[1], 1)) return 2;
    if (setenv("MODEL_IDS", "mock-ids", 1) || setenv("MODEL_PLE", "mock-ple", 1)) return 2;
    model_capture_check(0, 48, 1, 1, 0);
    float *values = malloc(248320 * sizeof(*values));
    if (!values) return 2;
    for (unsigned i = 0; i < 248320; i++) values[i] = (float)i;
    dn_bufs_t bufs[MAXD] = {{(uintptr_t)values}, {(uintptr_t)(values + 124160)}};
    split_t sp[MAXD][MAXB] = {{{0,124160}}, {{124160,124160}}};
    uint32_t ids[2] = {10, 11};
    if (!strcmp(mode, "overlap")) sp[1][0].o0 = 0;
    if (!strcmp(mode, "missing")) sp[1][0].on--;
    if (!strcmp(mode, "out-of-range")) sp[1][0].o0 = 248320;
    if (!strcmp(mode, "skip")) {
        model_capture_record(1, 2, ids, bufs, sp);
        return 0;
    }
    model_capture_record(0, 2, ids, bufs, sp);
    if (!strcmp(mode, "interrupted")) return 7;
    if (!strcmp(mode, "duplicate")) model_capture_record(0, 2, ids, bufs, sp);
    if (!strcmp(mode, "readback-failure")) mock_read_failure = 1;
    for (unsigned i = 0; i < 248320; i++) values[i] = (float)i + 1.0f;
    model_capture_record(1, 2, ids, bufs, sp);
    free(values);
    return 0;
}
