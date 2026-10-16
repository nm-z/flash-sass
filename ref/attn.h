/* Attention pieces exactly as the kernels compute them. */
#ifndef FLASH_ATTN_H
#define FLASH_ATTN_H
/* kern/attnqk.sass: one warp per head, 8 dims per lane (lane l: dims 8l..8l+7).
 *   ss = per-lane fmaf chain over its 8 dims, xor 16..1 butterfly; m = ss * (1/256);
 *   v = m + eps; s = sqrtf(v); r = 1/s; y_d = (x_d * r) * w_d;
 *   RoPE on d < 64, pairs (p, p+32), p < 32:
 *     a = y_p * cos_p;      y'_p      = fmaf(y_{p+32}, -sin_p, a)
 *     b = y_{p+32} * cos_p; y'_{p+32} = fmaf(y_p, sin_p, b)
 * cs/sn hold cos/sin of position t for the 32 pairs. */
void qk_norm_rope_gpu(const float *x, const float *w, float eps, const float *cs, const float *sn, float *y);
/* the cos/sin table entry for position t, pair p: angle = t * base^(-2p/64) in double, rounded to float */
void rope_table_gpu(unsigned t, float base, float *cs, float *sn);
#endif
