/* Whole transformer blocks composed from the bit-exact primitive twins, in the
 * exact phase order of kern/prog.c (same buffers, same op sequence), so a die
 * program can be checked bit for bit before any tolerance comparison. */
#ifndef FLASH_BLOCK_H
#define FLASH_BLOCK_H
#include <stdint.h>

/* host images of the block's tensors, in the device layouts (hold's cache files) */
typedef struct {
	const float *hc_attn_norm; const uint16_t *hc_attn_down, *hc_attn_up, *hc_attn_inject;
	const float *hc_ffn_norm;  const uint16_t *hc_ffn_down, *hc_ffn_up, *hc_ffn_inject;
	const uint8_t *attn_qkv, *attn_gate, *ssm_out; uint32_t attn_qkv_type, attn_gate_type, ssm_out_type;
	const uint16_t *ssm_alpha, *ssm_beta; const float *ssm_dt_bias, *ssm_conv1d, *ssm_a, *ssm_norm;
	const uint16_t *ffn_gate_inp, *ffn_gate_inp_shexp;
	const uint8_t *ffn_gate_exps, *ffn_up_exps, *ffn_down_exps;
	const uint8_t *ffn_gate_shexp, *ffn_up_shexp, *ffn_down_shexp; uint32_t ffn_gate_shexp_type, ffn_up_shexp_type, ffn_down_shexp_type;
} dn_hweights_t;

typedef struct {
	float S[10240];
	float xh[10240], down[320], inj[4], t[320], up[10240], z[2560];
	int16_t zq[2560]; int32_t zs32[80], zs16[160]; float zas[80];
	float qkv[10240], gate[6144], a[48], b[48], u[10240], o[6144], on[6144];
	int16_t oq[6144]; int32_t os32[192], os16[384]; float oas[192];
	float y[2560];
	float r[512], sh, s, w[10]; unsigned idx[10];
	float g[7040], gu[7040];
	int16_t hq[7040]; int32_t hs32[220], hs16[440]; float has[220];
	float out[2560];
	float conv_state[4][10240];
	float *dn_state;             /* 48 x 128 x 128 */
} dn_hbufs_t;

/* y = W x for any supported weight type (tile layout), with the quantised x */
void gemv_ref_any(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *s32, const int32_t *s16, const float *as, float *y);
void deltanet_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps);
#endif
