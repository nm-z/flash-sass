/* Reference decoders for the formats in the Flash-Next GSQ-RCO file, the int8
 * activation quantiser, the Q2_0 row tile the GPU reads, and the Q2_0 GEMV
 * computed in exactly the GPU's accumulation order. */
#ifndef FLASH_QUANT_H
#define FLASH_QUANT_H
#include <stdint.h>

float fp16_to_fp32(uint16_t h);
float bf16_to_fp32(uint16_t b);

/* Decode n elements (a whole number of blocks) of `type` from src into dst.
 * Returns 0, or -1 for an unsupported type. */
int dequant(uint32_t type, const void *src, float *dst, uint64_t n);

/* Per-row int8 quantisation: scale = amax/127, q = lrintf(x/scale).
 * sum32[h] receives the sum of the 32 codes of half-block h (n/32 entries). */
float quant_act_int8(const float *x, int8_t *q, int32_t *sum32, int n);

/* Q2_0 row tile: [fp16 d x B, padded to 16 B][16 B qs x B], rows padded to 16 B.
 * Returns the row stride in bytes. dst may be NULL to query the stride. */
int q2_0_tile_stride(int K);
int repack_q2_0(const uint8_t *src, int rows, int K, uint8_t *dst);

/* lanes per row the GPU uses for a Q2_0 matvec of width K (5 half-blocks per lane) */
int q2_0_lanes_per_row(int K);

/* y[r] for r < rows, computed as the GPU does: per lane a sequential FMA chain
 * over its 5 half-blocks, a SHFL-butterfly tree across the row's lanes, then
 * times xscale. Bit-identical to the kernel by construction. */
void gemv_q2_0_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y);

/* plain fp32 dot of the dequantised row with the dequantised activations, for sanity */
void gemv_q2_0_float(const uint8_t *src, int rows, int K, const float *x, float *y);


/* v2 layout (lane = row): the per-block (64) activation sums, negated, as the
 * kernel's XMAD addend wants them. nsum64 has K/64 entries. */
void act_nsum64(const int8_t *q, int32_t *nsum64, int n);

/* y[r] computed in the v2 kernel's order: one lane per row, blocks 0..B-1 in
 * sequence, acc = fma(fp32(d_b), float(dot_b + nsum64[b]), acc), then * xscale. */
void gemv_q2_0_ref_v2(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *nsum64, float xscale, float *y);

/* BF16 GEMV in the kernel's order: L lanes per row, lane l owns 8-element chunks
 * at 8 (l + L j); one fp32 FMA chain per lane over its chunks in element order,
 * then a SHFL butterfly over the L lanes. x is fp32, W is bf16 rows of K. */
int bf16_lanes_per_row(int K);
void gemv_bf16_ref(const uint16_t *W, int rows, int K, const float *x, float *y);

/* Q4_K sub-block-major row tile: per block 16 B header {fp16 d, fp16 dmin, u8 scales[12]}
 * for all B blocks, then per block 8 x 16 B of nibbles, sub-block j (elements
 * 32j..32j+31 of the block) in bytes 16j.., element 2i in the low nibble of byte i,
 * 2i+1 in the high nibble. Row stride 144 B per block. */
int q4_k_tile_stride(int K);
int repack_q4_k(const uint8_t *src, int rows, int K, uint8_t *dst);
int q4_k_lanes_per_row(int K);   /* 2560 -> 16 lanes x 5 sub-blocks, 6144 -> 32 x 6 */
/* 6-bit scale and min of sub-block j from a block's 12 scale bytes */
void q4_k_scale_min(const uint8_t *sc, int j, int *scale, int *min);
/* GPU order: lane l owns sub-blocks s = l + L i; per sub-block dot (int), t1 = (float)dot *
 * (float)sc, acc = fma(d, t1, acc), acc = fma(-dmin, (float)(min * sum32[s]), acc);
 * butterfly over L lanes; * xscale. */
void gemv_q4_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y);

/* Q5_K tile: Q4_K tile followed by per block 8 x 4 B high-bit words (bit e of word j =
 * high bit of element e of sub-block j). Row stride 176 B per block. */
int q5_k_tile_stride(int K);
int repack_q5_k(const uint8_t *src, int rows, int K, uint8_t *dst);
/* GPU order as gemv_q4_k_ref with dot = sum (q4 + 16 h) a */
void gemv_q5_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y);

#endif
