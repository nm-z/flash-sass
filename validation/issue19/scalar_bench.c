/* Successful-call overhead of the real checked scalar KV API.
 * No GPU access. Build with the same ref/qsa.c and ref/fmath.c as the range test.
 * Compare outputs outside timed regions; preserve every raw paired sample. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fenv.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "qsa.h"

enum { GROUPS = 64, COMPONENTS = 32, SAMPLES = 12, WARMUP = 4096 };
static float input[GROUPS][COMPONENTS];
static int8_t expected_q[GROUPS][COMPONENTS], result_q[GROUPS][COMPONENTS];
static uint16_t expected_s[GROUPS], result_s[GROUPS];

static uint64_t clock_ns(clockid_t clock) {
    struct timespec value;
    if (clock_gettime(clock, &value)) { perror("clock_gettime"); exit(1); }
    return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}

static uint64_t checksum(void) {
    uint64_t hash = UINT64_C(14695981039346656037);
    for (unsigned g = 0; g < GROUPS; g++) {
        for (unsigned i = 0; i < COMPONENTS; i++)
            hash = (hash ^ (uint8_t)result_q[g][i]) * UINT64_C(1099511628211);
        hash = (hash ^ (uint8_t)result_s[g]) * UINT64_C(1099511628211);
        hash = (hash ^ (uint8_t)(result_s[g] >> 8)) * UINT64_C(1099511628211);
    }
    return hash;
}

static void verify(void) {
    if (memcmp(expected_q, result_q, sizeof expected_q) ||
        memcmp(expected_s, result_s, sizeof expected_s)) {
        fputs("FAIL code or scale mismatch\n", stderr); exit(1);
    }
}

typedef struct { uint64_t cpu_ns, wall_ns, output_fnv; } measurement;

static measurement run(unsigned iterations, int checked) {
    unsigned rejected = 0;
    uint64_t wall_start = clock_ns(CLOCK_MONOTONIC);
    uint64_t cpu_start = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    if (checked) {
        for (unsigned i = 0; i < iterations; i++) {
            unsigned g = i % GROUPS;
            rejected |= (unsigned)kv_quant32_checked(input[g], result_q[g], &result_s[g]);
        }
    } else {
        for (unsigned i = 0; i < iterations; i++) {
            unsigned g = i % GROUPS;
            kv_quant32_gpu(input[g], result_q[g], &result_s[g]);
        }
    }
    uint64_t cpu_end = clock_ns(CLOCK_THREAD_CPUTIME_ID);
    uint64_t wall_end = clock_ns(CLOCK_MONOTONIC);
    if (rejected) { fputs("FAIL valid input was rejected\n", stderr); exit(1); }
    verify();
    measurement value = {cpu_end - cpu_start, wall_end - wall_start, checksum()};
    return value;
}

int main(int argc, char **argv) {
    unsigned iterations = 262144;
    if (argc > 2) { fputs("Usage: scalar_bench [iterations]\n", stderr); return 2; }
    if (argc == 2) {
        char *end;
        errno = 0;
        unsigned long value = strtoul(argv[1], &end, 10);
        if (errno || !argv[1][0] || *end || value < GROUPS ||
            value > 100000000 || value % GROUPS) {
            fputs("Iterations must be a multiple of 64, from 64 to 100000000\n", stderr); return 2;
        }
        iterations = (unsigned)value;
    }
    if (fesetround(FE_TONEAREST) || fegetround() != FE_TONEAREST) {
        fputs("Cannot establish nearest-even rounding\n", stderr); return 1;
    }
    for (unsigned g = 0; g < GROUPS; g++) {
        for (unsigned i = 0; i < COMPONENTS; i++) {
            int value = (int)((g * 37u + i * 19u) % 251u) - 125;
            input[g][i] = i % 7u ? (float)value * (1.0f / 64.0f) : 0.0f;
        }
        kv_quant32_gpu(input[g], expected_q[g], &expected_s[g]);
        if (kv_quant32_checked(input[g], result_q[g], &result_s[g])) {
            fputs("FAIL preflight rejected valid input\n", stderr); return 1;
        }
    }
    verify();
    printf("metadata samples=%d groups=%d components=%d iterations=%u warmup_iterations=%d "
           "rounding=nearest_even primary_clock=thread_cpu secondary_clock=monotonic_wall\n",
           SAMPLES, GROUPS, COMPONENTS, iterations, WARMUP);
    for (unsigned round = 0; round < 2; round++) {
        (void)run(WARMUP, (int)(round & 1u));
        (void)run(WARMUP, (int)((round & 1u) ^ 1u));
    }
    for (unsigned sample = 0; sample < SAMPLES; sample++) {
        measurement value[2];
        unsigned first = sample & 1u;
        value[first] = run(iterations, (int)first);
        value[first ^ 1u] = run(iterations, (int)(first ^ 1u));
        printf("sample=%u order=%s unchecked_cpu_ns=%" PRIu64 " checked_cpu_ns=%" PRIu64
               " unchecked_wall_ns=%" PRIu64 " checked_wall_ns=%" PRIu64
               " unchecked_fnv=%016" PRIx64 " checked_fnv=%016" PRIx64 " outputs=PASS\n",
               sample, first ? "checked,unchecked" : "unchecked,checked",
               value[0].cpu_ns, value[1].cpu_ns, value[0].wall_ns, value[1].wall_ns,
               value[0].output_fnv, value[1].output_fnv);
    }
    printf("PASS paired_samples=%d timed_calls_per_variant=%" PRIu64
           " timed_output_checks=%d checked_calls_all_succeeded=1\n",
           SAMPLES, (uint64_t)iterations * SAMPLES, 2 * SAMPLES);
    return 0;
}
