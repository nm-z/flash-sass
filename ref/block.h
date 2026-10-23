/* Whole transformer blocks composed from the bit-exact primitive twins, in the
 * exact phase order of kern/prog.c (same buffers, same op sequence), so a die
 * program can be checked bit for bit before any tolerance comparison. */
#ifndef FLASH_BLOCK_H
#define FLASH_BLOCK_H
#include <stdint.h>

/* host images of the block's tensors, in the device layouts (hold's cache files) */
/* merged rows as the holder lays them out: hc_*_dj = down ++ inject (324 x 10240),
 * ffn_gate_inp_x = router ++ shared gate (513 x 2560), ssm_ab = alpha ++ beta (96 x
 * 2560) with ssm_ab_bias = dt_bias ++ 48 zeros */
typedef struct {
	const float *hc_attn_norm; const uint16_t *hc_attn_dj, *hc_attn_up;
	const float *hc_ffn_norm;  const uint16_t *hc_ffn_dj, *hc_ffn_up;
	const uint8_t *attn_qkv, *attn_gate, *ssm_out; uint32_t attn_qkv_type, attn_gate_type, ssm_out_type;
	const uint16_t *ssm_ab; const float *ssm_ab_bias, *ssm_conv1d, *ssm_a, *ssm_norm;
	const uint16_t *ffn_gate_inp_x;
	const uint8_t *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
	const uint8_t *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp; uint32_t ffn_gate_shexp_type, ffn_up_shexp_type, ffn_down_shexp_type;
	/* full-attention blocks */
	const uint8_t *attn_q, *attn_k, *attn_v, *attn_output; uint32_t attn_q_type, attn_k_type, attn_v_type, attn_output_type;
	const float *attn_q_norm, *attn_k_norm;
	/* block 1's PLE: Q2_0 tiles of ple_key (10240 x 2560), BF16 ple_value (2560 x 2560), norms, conv taps as fp32 [10240][4];
	 * the n-gram table (IQ4_NL rows of 90 bytes, host memory) and the row-hash constants */
	const uint8_t *ple_key; const uint16_t *ple_value; const float *ple_norm_key, *ple_norm_query, *ple_norm_conv, *ple_conv_w;
	const uint8_t *ple_tab; uint64_t ple_mult[3], ple_off[16], ple_voc[16]; uint32_t ple_absent, ple_end;
	/* the head and the embedding */
	const float *output_hc_norm; const uint16_t *output_hc_down, *output_hc_up; const uint8_t *output, *token_embd;
} dn_hweights_t;

/* inj = down + 320 (and t + 320, copied), b = a + 48, sh = r[512] */
typedef struct {
	float S[10240];
	float xh[10240], down[324], t[324], up[10240], z[2560];
	int16_t zq[2560]; int32_t zs32[80], zs16[160]; float zas[80];
	float qkv[10240], gate[6144], a[96], u[10240], o[6144], on[6144];
	int16_t oq[6144]; int32_t os32[192], os16[384]; float oas[192];
	float y[2560];
	float r[513], s, w[10]; unsigned idx[10];
	float g[7040], gu[7040];
	int16_t hq[7040]; int32_t hs32[220], hs16[440]; float has[220];
	float part[6][2560];         /* each die's MoE partial (expert split), out = fixed-order sum */
	float out[2560];
	float conv_state[4][10240];
	float *dn_state;             /* 48 x 128 x 128 */
	/* attention: qg = 24 x [q | gate], caches in ref/attn.h layouts (zero-filled, sized by the caller) */
	float qg[12288], k[512], v[512], qo[6144], aout[6144], cs[32], sn[32];
	float *kc, *vc;
	/* PLE: g, its quantisation, K (10240) and kn, qn, V (2560), G (10240), gates, conv input, ring */
	float pg[2560]; int16_t pgq[2560]; int32_t pgs32[80], pgs16[160]; float pgas[80];
	float pK[10240], kn[10240], qn[10240], pV[2560], pG[10240], pgates[4], px[10240];
	float ple_state[10][10240];
	const uint32_t *ids;
	/* head: logits (248320, caller-allocated), the per-die argmax partials and the token */
	float *logits, dmax[6], tokmax; unsigned didx[6], tok;
} dn_hbufs_t;
/* S = the four lanes of token_embd[id] (raw Q3_K rows of 1100 bytes) */
void embed_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned id);
/* the head: output_hc read gate (no inject) -> z -> logits -> argmax; the argmax is taken per die's row range
 * ([248320 d / n, ...), first maximum) and combined in die order (greater, or equal with a lower index) */
void head_gpu(const dn_hweights_t *w, dn_hbufs_t *b, float eps, int ndies);
/* S += PLE(S, the n-gram rows of position t) — the prefix of block 1 (ref/ple.h pieces) */
void ple_prefix_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps);

/* y = W x for any supported weight type (tile layout), with the quantised x */
void gemv_ref_any(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *s32, const int32_t *s16, const float *as, float *y);
/* the die of n that owns expert e, and the die that runs block blk's shared expert
 * (the same arithmetic as kern/prog.h split_of: experts [512 d / n, 512 (d+1) / n), shexp on blk % n) */
int split_expert_owner(int e, int n);
/* one block over ndies dies: the MoE partials are summed per owning die in slot order
 * (the shared expert on die blk % ndies), then across dies in order */
void deltanet_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps, int ndies, int blk, int stage);   /* stage 1: stop after the DeltaNet hyper-block */
extern float *moe_dbg_shd;   /* debugging (chunk_test): when set, the MoE half stores the shared expert's down output (2560) there */
void attn_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps, int ndies, int blk, int stage);       /* the full-attention block at position t */
#define BLK_IS_ATTN(blk) (((blk) & 3) == 3)
#endif
