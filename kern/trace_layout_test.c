#define _POSIX_C_SOURCE 200809L
/* Exercise the actual trace image with empty opcode bodies, local grid barriers,
 * and six distinct peer-flag banks on one GPU. This is an ABI/dispatch check,
 * not whole-model parity or a six-device timing measurement. */
#include <cuda.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dieops.h"
CUresult CUDAAPI cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);

#define CK(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r, &s); fprintf(stderr, "%s: %s\n", #x, s); return 2; } } while (0)
#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); return 1; } } while (0)
enum { PHASES = 3, TOKENS = 2, WORDS = 16, GUARD = 128 };
static void ptr(uint32_t *h, unsigned word, CUdeviceptr p) { h[word] = (uint32_t)p; h[word + 1] = (uint32_t)(p >> 32); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 3 || (strcmp(argv[2], "1") && strcmp(argv[2], "16"))) {
        fprintf(stderr, "usage: %s TRACE_CUBIN CTAS(1|16)\n", argv[0]); return 2;
    }
    unsigned ctas = (unsigned)strtoul(argv[2], NULL, 10);
    CK(cuInit(0));
    CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_YIELD, dev));
    char name[128]; CK(cuDeviceGetName(name, sizeof name, dev));
    CK(cuModuleLoad(&mod, argv[1])); CK(cuModuleGetFunction(&fn, mod, "die"));
    uint32_t program[PHASES][32] = {{0}};
    program[0][0] = OP_EMBROW;
    program[1][0] = OP_XCOPY; program[1][30] = 1;
    program[2][0] = OP_IDXSEL; program[2][30] = 2;
    CUdeviceptr dp, ctl, flags, buffer;
    size_t bytes = (size_t)TOKENS * PHASES * ctas * WORDS * 8;
    CK(cuMemAlloc(&dp, sizeof program)); CK(cuMemAlloc(&ctl, 0x200));
    CK(cuMemAlloc(&flags, 6 * 16 * 4)); CK(cuMemAlloc(&buffer, bytes + GUARD * 2));
    CK(cuMemcpyHtoD(dp, program, sizeof program));
    unsigned char *copy = malloc(bytes + GUARD * 2); REQUIRE(copy != NULL);
    void *probe; CUstream poll;
    CK(cuMemAllocHost(&probe, bytes + GUARD * 2 + 0x200));
    CK(cuStreamCreate(&poll, CU_STREAM_NON_BLOCKING));
    for (unsigned pass = 0; pass < 3; pass++) {
        uint32_t header[128] = {0}; header[4] = 6;
        for (unsigned e = 0; e < 6; e++) ptr(header, 16 + 2 * e, flags + 64 * e);
        ptr(header, 28, flags);
        ptr(header, 0xd0 / 4, buffer + GUARD);
        header[0xd8 / 4] = 0; header[0xdc / 4] = TOKENS;
        header[0xe0 / 4] = pass == 1 ? (ctas == 1 ? 16 : 1) : ctas;
        header[0xe4 / 4] = pass == 2 ? 8 : WORDS;
        CK(cuMemsetD8(buffer, 0xa5, bytes + GUARD * 2));
        CK(cuMemsetD8(buffer + GUARD, 0, bytes)); CK(cuMemsetD8(flags, 0, 6 * 16 * 4));
        CK(cuMemcpyHtoD(ctl, header, sizeof header));
        unsigned nph = PHASES, ntok = TOKENS;
        void *params[] = {&dp, &nph, &ctl, &ntok};
        CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
        unsigned pending = 0;
        for (; pending < 200; pending++) {
            CUresult q = cuStreamQuery(0);
            if (q == CUDA_SUCCESS) break;
            if (q != CUDA_ERROR_NOT_READY) CK(q);
            struct timespec nap = {0, 10000000}; nanosleep(&nap, NULL);
        }
        if (pending == 200) {
            CK(cuMemcpyDtoHAsync(probe, ctl, 0x200, poll)); CK(cuStreamSynchronize(poll));
            uint32_t *h = probe;
            fprintf(stderr, "FAIL bounded launch: count=%u exchange=%u grid=%u arrivals=", h[2], h[8], h[1]);
            for (unsigned c = 0; c < 16; c++) fprintf(stderr, "%s%u", c ? "," : "", h[32 + c]);
            fputc('\n', stderr);
            CK(cuMemcpyDtoHAsync(probe, buffer + GUARD, bytes, poll)); CK(cuStreamSynchronize(poll));
            uint64_t *r = probe;
            for (unsigned p = 0; p < PHASES; p++) {
                fprintf(stderr, "phase=%u", p);
                for (unsigned k = 0; k < 12; k++) fprintf(stderr, " z%u=%" PRIu64, k, r[p * ctas * WORDS + k]);
                fputc('\n', stderr);
            }
            return 3;
        }
        CK(cuMemcpyDtoH(copy, buffer, bytes + GUARD * 2));
        for (unsigned i = 0; i < GUARD; i++) REQUIRE(copy[i] == 0xa5 && copy[GUARD + bytes + i] == 0xa5);
        CK(cuMemcpyDtoH(header, ctl, sizeof header));
        printf("pass=%u count=%u end=%u exchange=%u grid=%u\n", pass, header[2], header[5], header[8], header[1]);
        REQUIRE(header[2] == TOKENS);
        if (pass) {
            for (size_t i = 0; i < bytes; i++) REQUIRE(copy[GUARD + i] == 0);
            printf("PASS %s mismatch: no trace writes, canaries intact\n", pass == 1 ? "CTA mode" : "record stride");
            continue;
        }
        uint64_t *z = (uint64_t *)(copy + GUARD);
        printf("device=%s ctas=%u records=%u words=%u\n", name, ctas, TOKENS * PHASES * ctas, WORDS);
        for (unsigned t = 0; t < TOKENS; t++) for (unsigned c = 0; c < ctas; c++) {
            uint64_t sum = 0, previous = 0, first = 0, publication = 0;
            for (unsigned p = 0; p < PHASES; p++) {
                uint64_t *r = z + ((t * PHASES + p) * ctas + c) * WORDS;
                REQUIRE(r[0] && r[0] <= r[1] && r[1] <= r[2] && r[2] <= r[8] && r[8] <= r[3] && r[3] <= r[4]);
                REQUIRE(r[5] == 0 && r[6] == 0);
                REQUIRE(r[11] == (p == 2 ? t + 1 : t));
                if (p == 1) REQUIRE(r[9] >= r[3] && r[10] >= r[9] && r[10] <= r[4]);
                else REQUIRE(r[9] == 0 && r[10] == 0);
                if (p == PHASES - 1) { REQUIRE(r[7] >= r[4]); publication = r[7]; sum += r[7] - r[4]; }
                else REQUIRE(r[7] == 0);
                if (previous) { REQUIRE(r[0] >= previous); sum += r[0] - previous; }
                else first = r[0];
                sum += (r[1] - r[0]) + (r[2] - r[1]) + (r[8] - r[2]) + (r[3] - r[8]) + (r[4] - r[3]);
                previous = r[4];
                printf("sample t=%u cta=%u phase=%u generation=%" PRIu64 " dispatch_ns=%" PRIu64 " body_ns=%" PRIu64 "\n", t, c, p, r[11], r[8] - r[2], r[3] - r[8]);
            }
            REQUIRE(sum == publication - first);
        }
        printf("PASS dispatch/body boundaries, generation, peer-publication brackets, phase zero, per-token reconciliation\n");
    }
    free(copy); CK(cuStreamDestroy(poll)); CK(cuMemFreeHost(probe)); CK(cuMemFree(buffer)); CK(cuMemFree(flags)); CK(cuMemFree(ctl)); CK(cuMemFree(dp)); CK(cuModuleUnload(mod)); CK(cuCtxDestroy(ctx));
    return 0;
}
