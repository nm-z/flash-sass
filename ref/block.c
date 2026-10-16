#include "block.h"
#include "quant.h"
#include "gguf.h"
#include "fmath.h"
#include "hc.h"
#include "deltanet.h"
#include "router.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void gemv_ref_any(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *s32, const int32_t *s16, const float *as, float *y) {
	switch (type) {
	case GG_Q2_0: gemv_q2_0_ref(tile, rows, K, xq, s32, as, y); break;
	case GG_Q4_K: gemv_q4_k_ref(tile, rows, K, xq, s32, as, y); break;
	case GG_Q5_K: gemv_q5_k_ref(tile, rows, K, xq, s32, as, y); break;
	case GG_Q3_K: gemv_q3_k_ref(tile, rows, K, xq, as, y); break;
	case GG_Q6_K: gemv_q6_k_ref(tile, rows, K, xq, s16, as, y); break;
	case GG_IQ4_XS: gemv_iq4_xs_ref(tile, rows, K, xq, as, y); break;
	case GG_Q4_0: case GG_Q5_0: case GG_Q8_0: case GG_IQ4_NL: gemv_b32_ref(type, tile, rows, K, xq, s32, as, y); break;
	default: fprintf(stderr, "gemv_ref_any: type %u\n", type); exit(1);
	}
}

static void q16to8(const int16_t *q16, int8_t *q8, int n) { for (int i = 0; i < n; i++) q8[i] = (int8_t)q16[i]; }

/* a 10240-wide BF16 matrix as four K=2560 column slices, y = (((p0) + p1) + p2) + p3
 * (kern/gemv_bf16.sass with GEMV_ROWSTRIDE=20480: the first slice stores, the rest add) */
static void bf16_sliced(const uint16_t *W, int rows, const float *x, float *y) {
	uint16_t *tmp = malloc((size_t)rows * 2560 * 2);
	float *p = malloc(4 * rows);
	for (int s = 0; s < 4; s++) {
		for (int r = 0; r < rows; r++) memcpy(tmp + (size_t)r * 2560, W + (size_t)r * 10240 + 2560 * s, 5120);
		gemv_bf16_ref(tmp, rows, 2560, x + 2560 * s, p);
		for (int r = 0; r < rows; r++) y[r] = s ? y[r] + p[r] : p[r];
	}
	free(tmp); free(p);
}

static void hc_read(const float *S, const float *norm, const uint16_t *down, const uint16_t *up, const uint16_t *inj, dn_hbufs_t *b, float eps) {
	for (int l = 0; l < 4; l++) rmsnorm_gpu(S + 2560 * l, norm + 2560 * l, 2560, eps, b->xh + 2560 * l);
	bf16_sliced(down, 320, b->xh, b->down);
	bf16_sliced(inj, 4, b->xh, b->inj);
	hc_silu_gpu(b->down, b->t);
	gemv_bf16_ref(up, 10240, 320, b->t, b->up);
	hc_mix_gpu(b->up, b->xh, b->z);
	quant_act_gpu32(b->z, 2560, b->zq, b->zs32, b->zs16, b->zas);
}

void deltanet_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps) {
	int8_t *zq8 = malloc(2560), *oq8 = malloc(6144), *hq8 = malloc(7040);
	float *p = malloc(4 * 2560);
	/* ---- Gated DeltaNet hyper-block ---- */
	hc_read(b->S, w->hc_attn_norm, w->hc_attn_down, w->hc_attn_up, w->hc_attn_inject, b, eps);
	q16to8(b->zq, zq8, 2560);
	gemv_ref_any(w->attn_qkv_type, w->attn_qkv, 10240, 2560, zq8, b->zs32, b->zs16, b->zas, b->qkv);
	gemv_ref_any(w->attn_gate_type, w->attn_gate, 6144, 2560, zq8, b->zs32, b->zs16, b->zas, b->gate);
	gemv_bf16_ref(w->ssm_alpha, 48, 2560, b->z, b->a);
	for (int h = 0; h < 48; h++) b->a[h] = b->a[h] + w->ssm_dt_bias[h];
	gemv_bf16_ref(w->ssm_beta, 48, 2560, b->z, b->b);
	conv1d_silu_gpu(b->qkv, &b->conv_state[0][0], w->ssm_conv1d, t, b->u, 10240);
	for (int h = 0; h < 48; h++)
		deltanet_step_gpu(b->dn_state + (size_t)h * 16384, b->u + 128 * (h / 3), b->u + 2048 + 128 * (h / 3), b->u + 4096 + 128 * h, b->a[h], b->b[h], w->ssm_a[h], eps, b->o + 128 * h);
	dnnorm_gpu(b->o, w->ssm_norm, b->gate, eps, b->on);
	quant_act_gpu32(b->on, 6144, b->oq, b->os32, b->os16, b->oas);
	q16to8(b->oq, oq8, 6144);
	gemv_ref_any(w->ssm_out_type, w->ssm_out, 2560, 6144, oq8, b->os32, b->os16, b->oas, b->y);
	hc_write_gpu(b->inj, b->y, b->S);
	/* ---- MoE hyper-block ---- */
	hc_read(b->S, w->hc_ffn_norm, w->hc_ffn_down, w->hc_ffn_up, w->hc_ffn_inject, b, eps);
	q16to8(b->zq, zq8, 2560);
	gemv_bf16_ref(w->ffn_gate_inp, 512, 2560, b->z, b->r);
	gemv_bf16_ref(w->ffn_gate_inp_shexp, 1, 2560, b->z, &b->sh);
	router_gpu(b->r, b->sh, b->idx, b->w, &b->s);
	int stg = q2_0_tile_stride(2560), std = q2_0_tile_stride(640);
	for (int e = 0; e < 10; e++) {
		gemv_q2_0_ref(w->ffn_gate_exps + (size_t)b->idx[e] * stg * 640, 640, 2560, zq8, b->zs32, b->zas, b->g + 640 * e);
		gemv_q2_0_ref(w->ffn_up_exps + (size_t)b->idx[e] * stg * 640, 640, 2560, zq8, b->zs32, b->zas, b->gu + 640 * e);
	}
	gemv_ref_any(w->ffn_gate_shexp_type, w->ffn_gate_shexp, 640, 2560, zq8, b->zs32, b->zs16, b->zas, b->g + 6400);
	gemv_ref_any(w->ffn_up_shexp_type, w->ffn_up_shexp, 640, 2560, zq8, b->zs32, b->zs16, b->zas, b->gu + 6400);
	for (int e = 0; e < 11; e++) {
		float x[640];
		for (int i = 0; i < 640; i++) x[i] = silu_gpu(b->g[640 * e + i]) * b->gu[640 * e + i];
		quant_act_gpu32(x, 640, b->hq + 640 * e, b->hs32 + 20 * e, b->hs16 + 40 * e, b->has + 20 * e);
	}
	q16to8(b->hq, hq8, 7040);
	for (int e = 0; e < 10; e++) {
		gemv_q2_0_ref(w->ffn_down_exps + (size_t)b->idx[e] * std * 2560, 2560, 640, hq8 + 640 * e, b->hs32 + 20 * e, b->has + 20 * e, p);
		for (int r = 0; r < 2560; r++) b->out[r] = e ? fmaf(b->w[e], p[r], b->out[r]) : b->w[0] * p[r];
	}
	gemv_ref_any(w->ffn_down_shexp_type, w->ffn_down_shexp, 2560, 640, hq8 + 6400, b->hs32 + 200, b->hs16 + 400, b->has + 200, p);
	for (int r = 0; r < 2560; r++) b->out[r] = fmaf(b->s, p[r], b->out[r]);
	hc_write_gpu(b->inj, b->out, b->S);
	free(zq8); free(oq8); free(hq8); free(p);
}
