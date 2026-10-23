#ifndef QUANT_RANGE_CASES_H
#define QUANT_RANGE_CASES_H

#include <stdint.h>
#include <string.h>

/* Fixed binary32 inputs and independently specified binary16 scales.
 * 127 * float(1/127) = 1 - 2^-28 in exact arithmetic. The last input
 * whose rounded FP32 scale is the half-zero midpoint is 0x367e0001,
 * and its adjacent successor produces the first nonzero stored scale.
 * FP32 operands/products are normal; half subnormals must be preserved.
 * Each fixture has both sign placements and mixed zero channels.
 * The final pair crosses unbounded overflow; clipping keeps both finite. */
typedef struct {
    const char *name;
    uint32_t input_bits;
    uint16_t scale_bits;
} quant_range_case;

static const quant_range_case quant_range_cases[] = {
    { "zero",                  0x00000000u, 0x0000u },
    { "underflow-witness",      0x358637bdu, 0x0000u },
    { "ordinary-one",           0x3f800000u, 0x2008u },
    { "clip-predecessor",       0x4afde03fu, 0x7bffu },
    { "clip-limit",             0x4afde040u, 0x7bffu },
    { "clip-successor",         0x4afde041u, 0x7bffu },
    { "overflow-witness",       0x4b189680u, 0x7bffu },
    { "maximum-finite",         0x7f7fffffu, 0x7bffu },
    { "last-zero-scale",        0x367e0001u, 0x0000u },
    { "first-nonzero-scale",    0x367e0002u, 0x0001u },
    { "last-scale-0001",        0x373e7fffu, 0x0001u },
    { "first-scale-0002",       0x373e8000u, 0x0002u },
    { "last-subnormal-scale",   0x3bfde03fu, 0x03ffu },
    { "first-normal-scale",     0x3bfde040u, 0x0400u },
    { "last-scale-0400",        0x3bfe1fc1u, 0x0400u },
    { "first-scale-0401",       0x3bfe1fc2u, 0x0401u },
    { "last-scale-3c00",        0x42fe1fc1u, 0x3c00u },
    { "first-scale-3c01",       0x42fe1fc2u, 0x3c01u },
    { "last-scale-7bfe",        0x4afdd060u, 0x7bfeu },
    { "first-scale-7bff",       0x4afdd061u, 0x7bffu },
    { "unbounded-last-finite",  0x4afdf01fu, 0x7bffu },
    { "unbounded-first-inf",    0x4afdf020u, 0x7bffu }
};
enum {
    QUANT_RANGE_CASE_COUNT = sizeof quant_range_cases / sizeof quant_range_cases[0],
    QUANT_RANGE_SIGNED_COUNT = 2 * QUANT_RANGE_CASE_COUNT
};
static inline uint32_t quant_float_bits(float x) {
    uint32_t u; memcpy(&u, &x, sizeof u); return u;
}
static inline float quant_float_from_bits(uint32_t u) {
    float x; memcpy(&x, &u, sizeof x); return x;
}
static inline float quant_range_amplitude(unsigned fixture) {
    uint32_t bits = quant_range_cases[fixture / 2].input_bits;
    if (fixture & 1u) bits |= 0x80000000u;
    return quant_float_from_bits(bits);
}
static inline void quant_range_group(float *x, float a) {
    for (unsigned i = 0; i < 32; ++i)
        x[i] = i % 3 == 0 ? 0.f : (i & 1u) ? -a : a;
}
#endif
