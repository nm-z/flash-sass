/* sample.h: bit-exact twins of kern/topk.sass and kern/sample.sass (see their headers). */
#ifndef FLASH_SAMPLE_H
#define FLASH_SAMPLE_H
#include <stdint.h>

#define SAMPLE_MAXK 32

/* The K largest of x[0..n) by (value descending, index ascending) into idx[0..32) / val[0..32):
 * ranks >= K hold (0xffffffff, -inf). K <= 32, n <= 49152. */
void topk_gpu(const float *x, unsigned n, int K, uint32_t *idx, float *val);

/* The sampler over ndies per-die lists (idx[d][32], val[d][32] as topk_gpu makes them; the index
 * is within the die's rows, bases[d] is the die's first row): global top K, top_p on the
 * temperature-1 probabilities of those K, then temperature (invT = 1 / temp) and the draw u in
 * [0, 1). Returns the token. */
uint32_t sample_gpu(const uint32_t idx[][32], const float val[][32], const uint32_t *bases, int ndies, int K, float topp, float invT, float u);

#endif
