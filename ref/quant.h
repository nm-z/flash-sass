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

/* Q3_K row tile for L lanes per row (q4_k_lanes_per_row), Q = L/8 lane groups:
 * [prefix: Q x 16 B, lane group q holds fp16 d of blocks Q i + q for i = 0..B/Q-1]
 * [per block 112 B: 16 code words (group g of 16 elements: element e at bits
 *  2 (e & 7) of half e >> 3), 16 half-words of inverted high bits (bit e = !h_e),
 *  16 int8 scales (sc - 32)]. Element value = (code - 4 !h) * sc * d. */
int q3_k_tile_stride(int K);
int repack_q3_k(const uint8_t *src, int rows, int K, uint8_t *dst);
/* GPU order: lane (q = l >> 3, jj = l & 7) owns groups 2jj, 2jj+1 of blocks Q i + q;
 * per block sumi = sc0 dot0 + sc1 dot1 (exact integers), acc = fma(d, (float)sumi,
 * acc), then the SHFL butterfly and * xscale. */
void gemv_q3_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, float xscale, float *y);

/* Q6_K row tile: [prefix: Q x 16 B of fp16 d as for Q3_K] [per block 208 B: 8 x 16 B
 * low nibbles (sub-block j, element 2i low / 2i+1 high), 8 x 8 B high bits (sub-block
 * j, word w = elements 16w.. as 2-bit fields at bits 2 (e & 7) of half e >> 3),
 * 16 int8 scales]. Element value = d * sc[g] * (q6 - 32). */
int q6_k_tile_stride(int K);
int repack_q6_k(const uint8_t *src, int rows, int K, uint8_t *dst);
/* per-16 activation sums (n/16 entries) */
void act_sum16(const int8_t *q, int32_t *sum16, int n);
/* GPU order: lane (q, jj) owns sub-block jj of blocks Q i + q; per block
 * sumi = sc0 dot0 + sc1 dot1 with dot_g = sum (q6 - 32) a over the 16-element group
 * (exact), acc = fma(d, (float)sumi, acc), then the butterfly and * xscale. */
void gemv_q6_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum16, float xscale, float *y);

/* IQ4_XS row tile: [prefix: Q x 16 B of fp16 d as for Q3_K] [nibbles: block b,
 * sub-block j at 128 b + 16 j, element 2i low / 2i+1 high] [scales: int8 (ls - 32)
 * at 128 B + 8 b + j]. Element value = d * (ls - 32) * kvalues[q]. */
int iq4_xs_tile_stride(int K);
int repack_iq4_xs(const uint8_t *src, int rows, int K, uint8_t *dst);
int iq4_kvalue(int q);
/* GPU order: lane (q, jj) owns sub-block jj of blocks Q i + q; per block
 * dot = sum kvalues[q] a (exact), sumi = sc * dot, acc = fma(d, (float)sumi, acc). */
void gemv_iq4_xs_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, float xscale, float *y);

/* Block-32 formats at K = 640 (ffn_down_shexp): 4 lanes per row, lane l owns blocks
 * l + 4 i, i = 0..4. Row tile: [prefix 48 B: fp16 d of the 20 blocks][data: per block
 * 16 B nibbles (element 2i low / 2i+1 high) for Q4_0, Q5_0, IQ4_NL or 32 B int8 for
 * Q8_0][Q5_0 only: 20 x 4 B high-bit words, bit e = element e]. */
int b32_tile_stride(uint32_t type, int K);
int repack_b32(uint32_t type, const uint8_t *src, int rows, int K, uint8_t *dst);
/* element value of the tile (for checks): d * (q - 8 | q - 16 | q | kvalues[q]) */
float b32_tile_value(uint32_t type, const uint8_t *tile, int K, int r, int e);
/* GPU order: per lane dot_b = sum (q - off) a (exact), acc = fma(d_b, (float)dot_b, acc)
 * over its 5 blocks, a 4-lane butterfly, then * xscale. */
void gemv_b32_ref(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y);

/* The GPU activation quantiser (kern/actq.sass), bit for bit: amax = max |x|,
 * r = 1/amax (IEEE), inv = r * 127 (0 if amax == 0), q = lrintf(x * inv) as int16,
 * sum32/sum16 = per-32/per-16 sums of the codes. Returns xscale = amax * (1/127). */
float quant_act_gpu(const float *x, int n, int16_t *q16, int32_t *sum32, int32_t *sum16);

/* The GPU RMSNorm + quantiser (kern/normq.sass), bit for bit, n = 2560: lane l
 * (32 lanes) sums x^2 over the 4-element chunks at 4 (l + 32 j) with one fmaf chain,
 * SHFL butterfly (xor 16, 8, 4, 2, 1), m = ss * (1/2560), v = m + eps, s = sqrtf(v),
 * r = 1/s, y_i = (x_i * r) * w_i, then quant_act_gpu on y. Returns xscale. */
float rmsnorm_quant_gpu(const float *x, const float *w, int n, float eps, float *y, int16_t *q16, int32_t *sum32, int32_t *sum16);

/* The GPU fp32 RMSNorm (kern/rmsnorm.sass), bit for bit, n = 2560 or 10240:
 * per-lane fmaf chains over the chunks at 4 (l + 32 j), SHFL butterfly,
 * m = ss * (1/n), v = m + eps, s = sqrtf(v), r = 1/s, y_i = (x_i * r) * w_i. */
void rmsnorm_gpu(const float *x, const float *w, int n, float eps, float *y);

/* Per-32 activation quantiser (kern/actq.sass, GPU order): for each 32-element
 * group g: gmax = max(max |x|, 2^-100), r = 1/gmax (IEEE), inv = r * 127,
 * q = lrintf(x * inv) (int16), as[g] = gmax * (1/127); sum16/sum32 of the codes. */
void quant_act_gpu32(const float *x, int n, int16_t *q16, int32_t *sum32, int32_t *sum16, float *as);

/* RMSNorm + per-32 quantiser (kern/normq.sass): rmsnorm_gpu then quant_act_gpu32. */
void rmsnorm_quant_gpu32(const float *x, const float *w, int n, float eps, float *y, int16_t *q16, int32_t *sum32, int32_t *sum16, float *as);
