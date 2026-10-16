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

#endif
