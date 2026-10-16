#include "block.h"
#include "quant.h"
#include "gguf.h"
#include "fmath.h"
#include "hc.h"
#include "deltanet.h"
#include "router.h"
#include "attn.h"
#include "ple.h"
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

static void hc_read(const float *S, const float *norm, const uint16_t *dj, const uint16_t *up, dn_hbufs_t *b, float eps) {
	for (int l = 0; l < 4; l++) rmsnorm_gpu(S + 2560 * l, norm + 2560 * l, 2560, eps, b->xh + 2560 * l);
	bf16_sliced(dj, 324, b->xh, b->down);
	hc_silu_gpu(b->down, b->t);
	for (int i = 0; i < 4; i++) b->t[320 + i] = b->down[320 + i];   /* inj rides behind t (hcsilu copies [320, 324)) */
	gemv_bf16_ref(up, 10240, 320, b->t, b->up);
	hc_mix_gpu(b->up, b->xh, b->z);
	quant_act_gpu32(b->z, 2560, b->zq, b->zs32, b->zs16, b->zas);
}

static void moe_half(const dn_hweights_t *w, dn_hbufs_t *b, float eps, int ndies, int blk, int8_t *zq8, int8_t *hq8, float *p);

int split_expert_owner(int e, int n) {
	for (int d = 0; d < n; d++) if (e >= 512 * d / n && e < 512 * (d + 1) / n) return d;
	return -1;
}

void deltanet_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps, int ndies, int blk, int stage) {
	int8_t *zq8 = malloc(2560), *oq8 = malloc(6144), *hq8 = malloc(7040);
	float *p = malloc(4 * 2560);
	const float *inj = b->t + 320, *bb = b->a + 48;
	/* ---- Gated DeltaNet hyper-block ---- */
	hc_read(b->S, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps);
	q16to8(b->zq, zq8, 2560);
	gemv_ref_any(w->attn_qkv_type, w->attn_qkv, 10240, 2560, zq8, b->zs32, b->zs16, b->zas, b->qkv);
	gemv_ref_any(w->attn_gate_type, w->attn_gate, 6144, 2560, zq8, b->zs32, b->zs16, b->zas, b->gate);
	gemv_bf16_ref(w->ssm_ab, 96, 2560, b->z, b->a);           /* alpha ++ beta; + dt_bias ++ zeros */
	for (int h = 0; h < 96; h++) b->a[h] = b->a[h] + w->ssm_ab_bias[h];
	conv1d_silu_gpu(b->qkv, &b->conv_state[0][0], w->ssm_conv1d, t, b->u, 10240);
	for (int h = 0; h < 48; h++)
		deltanet_step_gpu(b->dn_state + (size_t)h * 16384, b->u + 128 * (h / 3), b->u + 2048 + 128 * (h / 3), b->u + 4096 + 128 * h, b->a[h], bb[h], w->ssm_a[h], eps, b->o + 128 * h);
	dnnorm_gpu(b->o, w->ssm_norm, b->gate, eps, b->on);
	quant_act_gpu32(b->on, 6144, b->oq, b->os32, b->os16, b->oas);
	q16to8(b->oq, oq8, 6144);
	gemv_ref_any(w->ssm_out_type, w->ssm_out, 2560, 6144, oq8, b->os32, b->os16, b->oas, b->y);
	hc_write_gpu(inj, b->y, b->S);
	if (stage != 1) moe_half(w, b, eps, ndies, blk, zq8, hq8, p);
	free(zq8); free(oq8); free(hq8); free(p);
}

void embed_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned id) {
	float row[2560];
	dequant(GG_Q3_K, w->token_embd + (size_t)id * 1100, row, 2560);
	for (int l = 0; l < 4; l++) memcpy(b->S + 2560 * l, row, sizeof row);
}

void head_gpu(const dn_hweights_t *w, dn_hbufs_t *b, float eps, int ndies) {
	int8_t zq8[2560];
	for (int l = 0; l < 4; l++) rmsnorm_gpu(b->S + 2560 * l, w->output_hc_norm + 2560 * l, 2560, eps, b->xh + 2560 * l);
	bf16_sliced(w->output_hc_down, 320, b->xh, b->down);
	hc_silu_gpu(b->down, b->t);
	gemv_bf16_ref(w->output_hc_up, 10240, 320, b->t, b->up);
	hc_mix_gpu(b->up, b->xh, b->z);
	quant_act_gpu32(b->z, 2560, b->zq, b->zs32, b->zs16, b->zas);
	q16to8(b->zq, zq8, 2560);
	gemv_ref_any(GG_Q5_K, w->output, 248320, 2560, zq8, b->zs32, b->zs16, b->zas, b->logits);
	for (int d = 0; d < ndies; d++) {
		unsigned o0 = 248320u * d / ndies, o1 = 248320u * (d + 1) / ndies, im = o0; float m = b->logits[o0];
		for (unsigned i = o0 + 1; i < o1; i++) if (b->logits[i] > m) { m = b->logits[i]; im = i; }
		b->dmax[d] = m; b->didx[d] = im;
	}
	b->tok = b->didx[0]; b->tokmax = b->dmax[0];
	for (int d = 1; d < ndies; d++) if (b->dmax[d] > b->tokmax || (b->dmax[d] == b->tokmax && b->didx[d] < b->tok)) { b->tokmax = b->dmax[d]; b->tok = b->didx[d]; }
}

void ple_prefix_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps) {
	uint64_t rows[16];
	int8_t gq8[2560];
	ple_rows_c(b->ids, t, w->ple_absent, &w->ple_end, 1, w->ple_mult, w->ple_off, w->ple_voc, rows);
	for (int h = 0; h < 16; h++) dequant(GG_IQ4_NL, w->ple_tab + rows[h] * 90, b->pg + 160 * h, 160);
	quant_act_gpu32(b->pg, 2560, b->pgq, b->pgs32, b->pgs16, b->pgas);
	q16to8(b->pgq, gq8, 2560);
	gemv_q2_0_ref(w->ple_key, 10240, 2560, gq8, b->pgs32, b->pgas, b->pK);
	for (int l = 0; l < 4; l++) {
		rmsnorm_gpu(b->pK + 2560 * l, w->ple_norm_key + 2560 * l, 2560, eps, b->kn + 2560 * l);
		rmsnorm_gpu(b->S + 2560 * l, w->ple_norm_query + 2560 * l, 2560, eps, b->qn + 2560 * l);
	}
	gemv_bf16_ref(w->ple_value, 2560, 2560, b->pg, b->pV);
	ple_gate_gpu(b->kn, b->qn, b->pV, b->pG, b->pgates);
	for (int l = 0; l < 4; l++) rmsnorm_gpu(b->pG + 2560 * l, w->ple_norm_conv + 2560 * l, 2560, eps, b->px + 2560 * l);
	ple_conv_gpu(b->px, b->ple_state, w->ple_conv_w, t, b->pG, b->S);
}

/* the full-attention block: hc read; q/k/v projections; per-head RMS + RoPE on q and k,
 * the KV cache gets position t; decode attention over the t+1 keys with the output
 * gate (ref/attn.h, as the attnqk / attnpart / attnmerge kernels); attn_output; hc
 * write; then the MoE hyper-block */
void attn_block_gpu(const dn_hweights_t *w, dn_hbufs_t *b, unsigned t, float eps, int ndies, int blk, int stage) {
	int8_t *zq8 = malloc(2560), *oq8 = malloc(6144), *hq8 = malloc(7040);
	float *p = malloc(4 * 2560), ktmp[256];
	const float *inj = b->t + 320;
	hc_read(b->S, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps);
	q16to8(b->zq, zq8, 2560);
	gemv_ref_any(w->attn_q_type, w->attn_q, 12288, 2560, zq8, b->zs32, b->zs16, b->zas, b->qg);
	gemv_ref_any(w->attn_k_type, w->attn_k, 512, 2560, zq8, b->zs32, b->zs16, b->zas, b->k);
	gemv_ref_any(w->attn_v_type, w->attn_v, 512, 2560, zq8, b->zs32, b->zs16, b->zas, b->v);
	rope_table_gpu(t, 1e7f, b->cs, b->sn);
	for (int h = 0; h < 24; h++) qk_norm_rope_gpu(b->qg + 512 * h, w->attn_q_norm, eps, b->cs, b->sn, b->qo + 256 * h);
	for (int kv = 0; kv < 2; kv++) {
		qk_norm_rope_gpu(b->k + 256 * kv, w->attn_k_norm, eps, b->cs, b->sn, ktmp);
		for (int d = 0; d < 256; d++) { b->kc[attn_kidx(t, kv, d)] = ktmp[d]; b->vc[(size_t)t * 512 + 256 * kv + d] = b->v[256 * kv + d]; }
	}
	attn_decode_gpu(b->qo, b->kc, b->vc, t + 1, b->qg, b->aout);
	quant_act_gpu32(b->aout, 6144, b->oq, b->os32, b->os16, b->oas);
	q16to8(b->oq, oq8, 6144);
	gemv_ref_any(w->attn_output_type, w->attn_output, 2560, 6144, oq8, b->os32, b->os16, b->oas, b->y);
	hc_write_gpu(inj, b->y, b->S);
	if (stage != 1) moe_half(w, b, eps, ndies, blk, zq8, hq8, p);
	free(zq8); free(oq8); free(hq8); free(p);
}

/* ---- MoE hyper-block ---- */
static void moe_half(const dn_hweights_t *w, dn_hbufs_t *b, float eps, int ndies, int blk, int8_t *zq8, int8_t *hq8, float *p) {
	const float *inj = b->t + 320;
	hc_read(b->S, w->hc_ffn_norm, w->hc_ffn_dj, w->hc_ffn_up, b, eps);
	q16to8(b->zq, zq8, 2560);
	gemv_bf16_ref(w->ffn_gate_inp_x, 513, 2560, b->z, b->r);  /* router ++ shared-expert gate */
	router_gpu(b->r, b->r[512], b->idx, b->w, &b->s);
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
	/* each die accumulates the routed experts it owns, in slot order, starting from w p
	 * (a die without one keeps zeros); the shared expert joins its die's partial last */
	int first[6] = { 1, 1, 1, 1, 1, 1 };
	for (int d = 0; d < ndies; d++) for (int r = 0; r < 2560; r++) b->part[d][r] = 0.f;
	for (int e = 0; e < 10; e++) {
		int d = split_expert_owner((int)b->idx[e], ndies);
		gemv_q2_0_ref(w->ffn_down_exps + (size_t)b->idx[e] * std * 2560, 2560, 640, hq8 + 640 * e, b->hs32 + 20 * e, b->has + 20 * e, p);
		for (int r = 0; r < 2560; r++) b->part[d][r] = first[d] ? b->w[e] * p[r] : fmaf(b->w[e], p[r], b->part[d][r]);
		first[d] = 0;
	}
	gemv_ref_any(w->ffn_down_shexp_type, w->ffn_down_shexp, 2560, 640, hq8 + 6400, b->hs32 + 200, b->hs16 + 400, b->has + 200, p);
	float *ps = b->part[blk % ndies];
	for (int r = 0; r < 2560; r++) ps[r] = fmaf(b->s, p[r], ps[r]);
	for (int r = 0; r < 2560; r++) { float acc = b->part[0][r]; for (int d = 1; d < ndies; d++) acc = acc + b->part[d][r]; b->out[r] = acc; }
	hc_write_gpu(inj, b->out, b->S);
}
