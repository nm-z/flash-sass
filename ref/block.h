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
} dn_hbufs_t;

/* y = W x for any supported weight type (tile layout), with the quantised x */
void gemv_ref_any(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *s32, const int32_t *s16, const float *as, float *y);
/* the die of n that owns expert e, and the die that runs block blk's shared expert
 * (the same arithmetic as kern/prog.h split_of: experts [512 d / n, 512 (d+1) / n), shexp on blk % n) */
int split_expert_owner(int e, int n);
/* one block over ndies dies: the MoE partials are summed per owning die in slot order
 * (the shared expert on die blk % ndies), then across dies in order */
void deltanet_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps, int ndies, int blk, int stage);   /* stage 1: stop after the DeltaNet hyper-block */
#endif
