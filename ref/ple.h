/* PLE (n-gram) pieces exactly as the kernels compute them. */
#ifndef FLASH_PLE_H
#define FLASH_PLE_H
#include <stddef.h>
#include <stdint.h>

/* Row hash of per_layer_token_embd (GGUF qwen4exp.ple.*, Flash-Next values):
 *   ngram 3, 8 heads per order -> 16 rows per token.
 *   context[back] = ids[pos - back] if back <= pos and no id in ids[pos-back, pos)
 *                   is an end id, else `absent`
 *   order n = 2, 3: mixed_n = XOR_{i < n} context[i] * mult[i]   (wrapping u64)
 *   head h in [(n-2) 8, (n-1) 8): row_h = offsets[h] + mixed_n % vocab[h]
 * (cx-flash RowHash::rows_at.) */
#define PLE_NGRAM 3
#define PLE_HEADS 16
#define PLE_ROW_VALUES 160
extern const uint64_t ple_multipliers[PLE_NGRAM];
extern const uint64_t ple_head_offsets[PLE_HEADS];
extern const uint64_t ple_head_vocab[PLE_HEADS];
void ple_rows(const uint32_t *ids, size_t pos, uint32_t absent, const uint32_t *ends, int nends, uint64_t rows[PLE_HEADS]);
/* same with explicit constants (the test uses a small table) */
void ple_rows_c(const uint32_t *ids, size_t pos, uint32_t absent, const uint32_t *ends, int nends, const uint64_t *mult, const uint64_t *off, const uint64_t *voc, uint64_t rows[PLE_HEADS]);

/* kern/plegate.sass: per-lane gate of the PLE (ref: cx-flash lower_ple).
 *   s_l = sum over c of KN[l][c] QN[l][c] in the kernel's order: thread t (0..255)
 *   chains c = t + 256 k (k = 0..9, FMUL then fmaf), xor 16..1 butterfly per warp,
 *   the 8 warp sums added in order; s' = s_l * (float)(1/sqrt(2560));
 *   mag = max(|s'|, 1e-6); root = sqrtf(mag); sign = (s' > 0) - (0 > s');
 *   gate_l = sigmoid(sign * root); G[l][c] = gate_l * V[c]. */
void ple_gate_gpu(const float *kn, const float *qn, const float *v, float *g, float *gates);

/* kern/pleconv.sass: dilated causal depthwise conv (kernel 4, dilation 3) over the
 * normalised gated value, SiLU, residual and stream add (cx-flash lower_ple +
 * dconv_forward_body: tap k multiplies x(t - (3 - k) 3), positions before 0 read 0).
 *   y = w[c][0] x_{t-9}[c]; y = fmaf(w[c][1], x_{t-6}[c], y); fmaf(w[c][2], x_{t-3}); fmaf(w[c][3], x_t)
 *   s[c] = s[c] + (g[c] + silu(y)); then x_t is kept in state[t mod 10]. w is fp32 (from F16). */
void ple_conv_gpu(const float *x, float state[10][10240], const float *w, unsigned t, const float *g, float *s);
#endif
