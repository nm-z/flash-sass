/* Attention pieces exactly as the kernels compute them. */
#ifndef FLASH_ATTN_H
#define FLASH_ATTN_H
#include <stddef.h>
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

/* K cache layout (written by attnqk, read by attnpart): blocks of 32 positions,
 * [pos/32][kv][64 dim-chunks][32 keys][4 dims] fp32, so a warp whose lanes hold 32
 * consecutive keys reads one 4-dim chunk with a single 512-byte LDG.128. The V cache
 * stays position-major [pos][kv][256]. */
static inline size_t attn_kidx(unsigned pos, int kv, int d) {
	return (((size_t)(pos >> 5) * 128 + kv * 64 + (d >> 2)) * 128) + (pos & 31) * 4 + (d & 3);
}
#define ATTN_KC_FLOATS(npos) ((size_t)(((npos) + 31) / 32) * 32 * 512)

/* kern/attnpart.sass + kern/attnmerge.sass: decode attention over n keys for the 24
 * query heads (GQA: head i uses KV head i / 12). 32 CTAs of 256 threads: CTA b has
 * chunk c = b & 7, head half hh = (b >> 3) & 1, kv = b >> 4 and computes partials
 * for heads 12 kv + 6 hh + 0..5 over keys [c CK, (c+1) CK), CK = ceil(n/8192) * 1024,
 * in tiles of 1024 keys:
 *   scores: thread t = 32 w + l holds keys tb + 256 s + t, s = 0..3:
 *     a = 0; a = fmaf(q[d], k[d], a) for d = 0..255; a *= 0.0625; -inf for keys >= n
 *   tile max per head over the 1024 keys (exact); m_new = max(m, tile max), m starts
 *     at -1e30; scale = exp(m - m_new); p = keys < n ? exp(a - m_new) : 0
 *   tile den: per thread ((p_s0 + p_s1) + p_s2) + p_s3, xor 16..1 butterfly over the
 *     warp, warps summed in order 0..7; den = fmaf(den, scale, tile den)
 *   V: 4 streams r = 0..3 (keys tb + 4 k + r, k in order), each acc_r[h][d] *= scale then
 *     acc_r = fmaf(p, V[j][d], acc_r) for j < n; chunk result (acc_0 + acc_1) + (acc_2 + acc_3)
 * Merge over the 8 chunks in order: M = max_c m_c; e_c = exp(m_c - M); den = pd_0 e_0 then
 * fmaf; out[d] = (po_0[d] e_0 then fmaf) * rcp(den); result = out * sigmoid(gate),
 * gate_i[d] = gate[512 i + 256 + d] (second half of attn_q's head block).
 * kc is in the blocked layout above; vc is [pos][2][256]. */
#define ATTN_NCH 8
#define ATTN_MINIT (-1e30f)
void attn_decode_gpu(const float *qo, const float *kc, const float *vc, unsigned n, const float *gate, float *out);
/* the partials themselves, for debugging: pm/pd [24][8], po [24][8][256] */
void attn_partials_gpu(const float *qo, const float *kc, const float *vc, unsigned n, float *pm, float *pd, float *po);
void attn_merge_gpu(const float *pm, const float *pd, const float *po, const float *gate, float *out);
#endif
