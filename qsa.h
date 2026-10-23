/* QSA: the sparse attention of the model (llama.cpp's qwen4exp "QSA" path) as the kernels compute it.
 *
 * Every attention layer keeps an int8 KV cache (blocks of 32 values with an fp16 scale) and an indexer key
 * pool. For the token at position t (rank t in the sequence), with nv = (t + 1) / 4 complete blocks of 4
 * tokens and tail = (t + 1) % 4 tokens of the unfinished block:
 *   indexer q  = RoPE_t(RMS128(W_iq z, 4 heads) * q_norm)            (norm + weight, then RoPE on dims 0..63)
 *   indexer k  = W_ik z (raw, 128), kept in a ring of the last 4; a block that the token completes is pooled:
 *                mean of its 4 raw keys, RMS128 * k_norm, RoPE at the position of its first token, stored as fp16
 *   score[b]   = sum_h relu(q_h . pool[b])   for the nv complete blocks
 *   selection  = the min(nv, TOPBLK) highest-scoring blocks (equal scores: the lower block first), in ascending
 *                order, then the tail block; attention runs over every key of those blocks
 * With nv <= TOPBLK nothing is dropped: the selection is every block, so the attended keys are 0..t.
 * kern/idxqk.sass, idxscore.sass, idxsel.sass, kvgather.sass, attnqkq.sass, attnpartn.sass are bit-exact to these. */
#ifndef FLASH_QSA_H
#define FLASH_QSA_H
#include <stdint.h>
#include <stddef.h>

#define QSA_POOL 4
#define QSA_TOPBLK 512            /* indexer.top_k (2048) / QSA_POOL */
#define QSA_NIDXH 4               /* indexer heads */
#define QSA_IDIM 128              /* indexer head size */

uint16_t qsa_f2h(float f);        /* fp32 -> fp16, round to nearest even, subnormals kept (F2F.F16.F32) */
float qsa_h2f(uint16_t h);

/* the KV cache of one layer: token t, kv head h, dim d: q at (t * 2 + h) * 256 + d, scale at (t * 2 + h) * 8 + d / 32 */
void kv_quant32_gpu(const float *x, int8_t *q, uint16_t *s16);   /* 32 values: gmax = max(max|x|, 2^-100); inv = rcp(gmax) * 127; q = rint(x inv); s = f16(gmax / 127) */
float kv_deq_gpu(int8_t q, uint16_t s16);                         /* (float)q * f32(s16) */

/* the number of keys attention covers at position t, and the attended keys of the gather */
unsigned qsa_nkeys(unsigned t, unsigned nblk);
/* the compact K cache (attn_kidx layout, blocked by 32 keys) and V cache ([slot][2][256]) of the selected blocks sel[0..ns) (ignored
 * when nv <= nblk: the blocks are 0..nv) followed by the tail block; returns the key count */
unsigned qsa_gather_gpu(const int8_t *kq, const uint16_t *ks, const int8_t *vq, const uint16_t *vs, const uint32_t *sel, unsigned t, unsigned nblk, float *kc, float *vc);

/* indexer: y = RoPE(RMS128(x) * w) as the warp computes it (lane l owns dims 4l..4l+3); cs/sn: the 32 cos/sin of the position */
void idx_norm_rope_gpu(const float *x, const float *w, float eps, const float *cs, const float *sn, float *y);
/* the pooled key of the 4 raw keys raw[4][128] (cs/sn of the first token's position), as fp16 */
void idx_pool_gpu(const float *raw, const float *kn, float eps, const float *cs, const float *sn, uint16_t *out);
/* scores of nv pooled keys (pool[nv][128] fp16) for iq[4][128]: per head a fmaf chain over the 128 dims, relu, summed over the heads in order */
void idx_scores_gpu(const float *iq, const uint16_t *pool, unsigned nv, float *score);
/* the k = min(nv, nblk) best blocks (score descending, index ascending), as ascending indices in sel; returns k */
unsigned idx_select_gpu(const float *score, unsigned nv, unsigned nblk, uint32_t *sel);
#endif
