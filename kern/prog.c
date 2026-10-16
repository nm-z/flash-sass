#include "prog.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void prog_init(prog_t *P) { P->cap = 64; P->n = 0; P->p = malloc(sizeof(phase_t) * P->cap); }
void prog_free(prog_t *P) { free(P->p); P->p = NULL; P->n = P->cap = 0; }
phase_t *prog_add(prog_t *P, int op, unsigned units) {
	if (P->n == P->cap) { P->cap *= 2; P->p = realloc(P->p, sizeof(phase_t) * P->cap); }
	phase_t *p = &P->p[P->n++];
	memset(p, 0, sizeof *p);
	p->w[0] = (uint32_t)op; p->w[1] = units;
	return p;
}
void ph_ptr(phase_t *p, int addr, CUdeviceptr v) { int w = 2 + (addr - 0x140) / 4; p->w[w] = (uint32_t)v; p->w[w + 1] = (uint32_t)(v >> 32); }
void ph_u32(phase_t *p, int addr, uint32_t v) { p->w[2 + (addr - 0x140) / 4] = v; }
void ph_f32(phase_t *p, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); ph_u32(p, addr, u); }

void ph_rms(prog_t *P, int op, CUdeviceptr X, CUdeviceptr W, CUdeviceptr Y, float eps, unsigned vectors) {
	phase_t *p = prog_add(P, op, vectors);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, Y); ph_f32(p, 0x158, eps);
}
void ph_bf16(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Y, unsigned rows, CUdeviceptr bias) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, Y); ph_u32(p, 0x158, rows); ph_ptr(p, 0x160, bias);
}
void ph_hc(prog_t *P, int op, CUdeviceptr P0, CUdeviceptr P1, CUdeviceptr P2, unsigned n, unsigned units) {
	phase_t *p = prog_add(P, op, units);
	ph_ptr(p, 0x140, P0); ph_ptr(p, 0x148, P1); ph_ptr(p, 0x150, P2); ph_u32(p, 0x158, n);
}
void ph_actq(prog_t *P, int op, CUdeviceptr X, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors) {
	phase_t *p = prog_add(P, op, vectors);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, Q); ph_ptr(p, 0x150, S32); ph_ptr(p, 0x158, S16); ph_ptr(p, 0x160, AS); ph_u32(p, 0x168, n);
}
void ph_ffnact(prog_t *P, CUdeviceptr G, CUdeviceptr U, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors) {
	phase_t *p = prog_add(P, OP_FFNACT, vectors);
	ph_ptr(p, 0x140, G); ph_ptr(p, 0x148, U); ph_ptr(p, 0x150, Q); ph_ptr(p, 0x158, S32); ph_ptr(p, 0x160, S16); ph_ptr(p, 0x168, AS); ph_u32(p, 0x170, n);
}
void ph_gemvk(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr AS) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f); ph_ptr(p, 0x170, AS);
}
void ph_q2(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS, CUdeviceptr WE) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS); ph_ptr(p, 0x178, WE);
}
void ph_b32(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS);
}
void ph_conv1d(prog_t *P, CUdeviceptr X, CUdeviceptr STATE, CUdeviceptr W, CUdeviceptr U, unsigned t, unsigned n) {
	phase_t *p = prog_add(P, OP_CONV1D, n / 1024);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, STATE); ph_ptr(p, 0x150, W); ph_ptr(p, 0x158, U); ph_u32(p, 0x160, t); ph_u32(p, 0x164, n);
}
void ph_dnstep(prog_t *P, CUdeviceptr S, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr A, CUdeviceptr B, CUdeviceptr SSMA, CUdeviceptr O, float eps) {
	phase_t *p = prog_add(P, OP_DNSTEP, 48);
	ph_ptr(p, 0x140, S); ph_ptr(p, 0x148, Q); ph_ptr(p, 0x150, K); ph_ptr(p, 0x158, V); ph_ptr(p, 0x160, A); ph_ptr(p, 0x168, B);
	ph_ptr(p, 0x170, SSMA); ph_ptr(p, 0x178, O); ph_f32(p, 0x180, eps);
}
void ph_dnnorm(prog_t *P, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, CUdeviceptr Y, float eps) {
	phase_t *p = prog_add(P, OP_DNNORM, 48);
	ph_ptr(p, 0x140, O); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, G); ph_ptr(p, 0x158, Y); ph_f32(p, 0x160, eps);
}
void ph_router(prog_t *P, CUdeviceptr R, CUdeviceptr SH, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr S) {
	phase_t *p = prog_add(P, OP_ROUTER, 1);
	ph_ptr(p, 0x140, R); ph_ptr(p, 0x148, SH); ph_ptr(p, 0x150, IDX); ph_ptr(p, 0x158, W); ph_ptr(p, 0x160, S);
}

int gemv_op(uint32_t type, int K, int acc) {
	switch (type) {
	case GG_Q2_0: return K == 2560 ? OP_Q2K2560 : acc ? OP_Q2ACC640 : OP_Q2K640;
	case GG_Q4_K: return K == 2560 ? OP_Q4KK2560 : OP_Q4KK6144;
	case GG_Q5_K: return K == 2560 ? OP_Q5KK2560 : OP_Q5KK6144;
	case GG_Q3_K: return K == 2560 ? OP_Q3KK2560 : OP_Q3KK6144;
	case GG_Q6_K: return K == 2560 ? OP_Q6KK2560 : 0;
	case GG_IQ4_XS: return K == 2560 ? OP_IQ4K2560 : OP_IQ4K6144;
	case GG_Q4_0: return acc ? OP_B32Q40ACC : OP_B32Q40;
	case GG_Q5_0: return acc ? OP_B32Q50ACC : OP_B32Q50;
	case GG_Q8_0: return acc ? OP_B32Q80ACC : OP_B32Q80;
	case GG_IQ4_NL: return acc ? OP_B32IQ4NLACC : OP_B32IQ4NL;
	default: return 0;
	}
}
int gemv_op_is_q2(int op) { return op == OP_Q2K2560 || op == OP_Q2K640 || op == OP_Q2ACC640 || op == OP_Q2MOEG || op == OP_Q2MOED; }
int gemv_op_is_b32(int op) { return op == OP_B32Q40 || op == OP_B32Q50 || op == OP_B32Q80 || op == OP_B32IQ4NL || op == OP_B32Q40ACC || op == OP_B32Q50ACC || op == OP_B32Q80ACC || op == OP_B32IQ4NLACC; }

/* a GEMV phase by weight type: y = W x for the quantised x (codes, sums, scales) */
static void ph_gemv_any(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr Xq, CUdeviceptr S32, CUdeviceptr Y, unsigned rows, CUdeviceptr AS, int acc, CUdeviceptr scale) {
	int op = gemv_op(type, K, acc);
	if (!op) { fprintf(stderr, "prog: no GEMV op for type %u K %d\n", type, K); exit(1); }
	if (gemv_op_is_q2(op)) ph_q2(P, op, W, Xq, S32, Y, rows, 0, AS, scale);
	else if (gemv_op_is_b32(op)) ph_b32(P, op, W, Xq, S32, Y, rows, scale, AS);
	else ph_gemvk(P, op, W, Xq, S32, Y, rows, AS);
}

/* hyper-connection read: xh = per-lane RMS(S) * norm; down (320) and inject (4) as
 * four K=2560 slices of the 10240-wide BF16 rows; t = SiLU(0.25 down); up = W_up t;
 * z = mix(up, xh); z quantised */
static void ph_hc_read(prog_t *P, CUdeviceptr S, CUdeviceptr norm, CUdeviceptr down, CUdeviceptr up, CUdeviceptr inject, const dn_bufs_t *b, float eps) {
	ph_rms(P, OP_RMS2560, S, norm, b->xh, eps, 4);
	for (int s = 0; s < 4; s++) {
		int op = s ? OP_BF16HC : OP_BF16HC0;
		ph_bf16(P, op, down + 5120 * s, b->xh + 2560 * 4 * s, b->down, 320, 0);
		ph_bf16(P, op, inject + 5120 * s, b->xh + 2560 * 4 * s, b->inj, 4, 0);
	}
	ph_hc(P, OP_HCSILU, b->down, b->t, 0, 320, 2);
	ph_bf16(P, OP_BF16K320, up, b->t, b->up, 10240, 0);
	ph_hc(P, OP_HCMIX, b->up, b->xh, b->z, 2560, 10);
	ph_actq(P, OP_ACTQ2560, b->z, b->zq, b->zs32, b->zs16, b->zas, 2560, 1);
}

void prog_deltanet_block(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	/* ---- attention-side hyper-block: Gated DeltaNet ---- */
	ph_hc_read(P, b->S, w->hc_attn_norm, w->hc_attn_down, w->hc_attn_up, w->hc_attn_inject, b, eps);
	ph_gemv_any(P, w->attn_qkv_type, 2560, w->attn_qkv, b->zq, b->zs32, b->qkv, 10240, b->zas, 0, 0);
	ph_gemv_any(P, w->attn_gate_type, 2560, w->attn_gate, b->zq, b->zs32, b->gate, 6144, b->zas, 0, 0);
	ph_bf16(P, OP_BF16BIAS, w->ssm_alpha, b->z, b->a, 48, w->ssm_dt_bias);
	ph_bf16(P, OP_BF16K2560, w->ssm_beta, b->z, b->b, 48, 0);
	ph_conv1d(P, b->qkv, b->conv_state, w->ssm_conv1d, b->u, t, 10240);
	ph_dnstep(P, b->dn_state, b->u, b->u + 2048 * 4, b->u + 4096 * 4, b->a, b->b, w->ssm_a, b->o, eps);
	ph_dnnorm(P, b->o, w->ssm_norm, b->gate, b->on, eps);
	ph_actq(P, OP_ACTQ6144, b->on, b->oq, b->os32, b->os16, b->oas, 6144, 1);
	ph_gemv_any(P, w->ssm_out_type, 6144, w->ssm_out, b->oq, b->os32, b->y, 2560, b->oas, 0, 0);
	ph_hc(P, OP_HCWRITE, b->inj, b->y, b->S, 2560, 10);
	/* ---- FFN hyper-block: MoE ---- */
	ph_hc_read(P, b->S, w->hc_ffn_norm, w->hc_ffn_down, w->hc_ffn_up, w->hc_ffn_inject, b, eps);
	ph_bf16(P, OP_BF16K2560, w->ffn_gate_inp, b->z, b->r, 512, 0);
	ph_bf16(P, OP_BF16K2560, w->ffn_gate_inp_shexp, b->z, b->sh, 1, 0);
	ph_router(P, b->r, b->sh, b->idx, b->w, b->s);
	ph_q2(P, OP_Q2MOEG, w->ffn_gate_exps, b->zq, b->zs32, b->g, 640, b->idx, b->zas, 0);
	ph_q2(P, OP_Q2MOEG, w->ffn_up_exps, b->zq, b->zs32, b->gu, 640, b->idx, b->zas, 0);
	ph_gemv_any(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp, b->zq, b->zs32, b->g + 6400 * 4, 640, b->zas, 0, 0);
	ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->gu + 6400 * 4, 640, b->zas, 0, 0);
	ph_ffnact(P, b->g, b->gu, b->hq, b->hs32, b->hs16, b->has, 640, 11);
	ph_q2(P, OP_Q2MOED, w->ffn_down_exps, b->hq, b->hs32, b->out, 2560, b->idx, b->has, b->w);
	ph_gemv_any(P, w->ffn_down_shexp_type, 640, w->ffn_down_shexp, b->hq + 10 * 640 * 2, b->hs32 + 10 * 20 * 4, b->out, 2560, b->has + 10 * 20 * 4, 1, b->s);
	ph_hc(P, OP_HCWRITE, b->inj, b->out, b->S, 2560, 10);
}
