#include "prog.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void prog_init(prog_t *P) { P->cap = 64; P->n = 0; P->xj = 0; P->p = malloc(sizeof(phase_t) * P->cap); }
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
void ph_hc(prog_t *P, int op, CUdeviceptr P0, CUdeviceptr P1, CUdeviceptr P2, unsigned n, unsigned us, unsigned units) {
	phase_t *p = prog_add(P, op, units);
	ph_ptr(p, 0x140, P0); ph_ptr(p, 0x148, P1); ph_ptr(p, 0x150, P2); ph_u32(p, 0x158, n); ph_u32(p, 0x15c, us);
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
void ph_q2(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS, CUdeviceptr WE, CUdeviceptr NE) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS); ph_ptr(p, 0x178, WE); ph_ptr(p, 0x180, NE);
}
void ph_b32(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS);
}
void ph_conv1d(prog_t *P, CUdeviceptr X, CUdeviceptr STATE, CUdeviceptr W, CUdeviceptr U, unsigned t, unsigned n) {
	phase_t *p = prog_add(P, OP_CONV1D, (n + 1023) / 1024);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, STATE); ph_ptr(p, 0x150, W); ph_ptr(p, 0x158, U); ph_u32(p, 0x160, t); ph_u32(p, 0x164, n);
}
void ph_dnstep(prog_t *P, CUdeviceptr S, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr A, CUdeviceptr B, CUdeviceptr SSMA, CUdeviceptr O, float eps, unsigned hb, unsigned kb, unsigned heads) {
	phase_t *p = prog_add(P, OP_DNSTEP, heads);
	ph_ptr(p, 0x140, S); ph_ptr(p, 0x148, Q); ph_ptr(p, 0x150, K); ph_ptr(p, 0x158, V); ph_ptr(p, 0x160, A); ph_ptr(p, 0x168, B);
	ph_ptr(p, 0x170, SSMA); ph_ptr(p, 0x178, O); ph_f32(p, 0x180, eps); ph_u32(p, 0x184, hb); ph_u32(p, 0x188, kb);
}
void ph_dnnorm(prog_t *P, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, CUdeviceptr Y, float eps, unsigned heads) {
	phase_t *p = prog_add(P, OP_DNNORM, heads);
	ph_ptr(p, 0x140, O); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, G); ph_ptr(p, 0x158, Y); ph_f32(p, 0x160, eps);
}
void ph_router(prog_t *P, CUdeviceptr R, CUdeviceptr SH, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr S) {
	phase_t *p = prog_add(P, OP_ROUTER, 1);
	ph_ptr(p, 0x140, R); ph_ptr(p, 0x148, SH); ph_ptr(p, 0x150, IDX); ph_ptr(p, 0x158, W); ph_ptr(p, 0x160, S);
}
void ph_moesel(prog_t *P, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr MAP, CUdeviceptr LIDX, CUdeviceptr LW, CUdeviceptr NE, CUdeviceptr PART) {
	phase_t *p = prog_add(P, OP_MOESEL, 1);
	ph_ptr(p, 0x140, IDX); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, MAP); ph_ptr(p, 0x158, LIDX); ph_ptr(p, 0x160, LW); ph_ptr(p, 0x168, NE); ph_ptr(p, 0x170, PART);
}
void ph_sumn(prog_t *P, CUdeviceptr SRC, unsigned stride, CUdeviceptr OUT, unsigned n, unsigned ndies) {
	phase_t *p = prog_add(P, OP_SUMN, (n + 255) / 256);
	ph_ptr(p, 0x140, SRC); ph_u32(p, 0x148, stride); ph_ptr(p, 0x150, OUT); ph_u32(p, 0x158, n); ph_u32(p, 0x15c, ndies);
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
int gemv_op_is_q2(int op) { return op == OP_Q2K2560 || op == OP_Q2K640 || op == OP_Q2ACC640 || op == OP_Q2MOEG || op == OP_Q2MOED || op == OP_Q2MOEGN || op == OP_Q2MOEDN; }
int gemv_op_is_b32(int op) { return op == OP_B32Q40 || op == OP_B32Q50 || op == OP_B32Q80 || op == OP_B32IQ4NL || op == OP_B32Q40ACC || op == OP_B32Q50ACC || op == OP_B32Q80ACC || op == OP_B32IQ4NLACC; }

void ph_gemv_any(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr Xq, CUdeviceptr S32, CUdeviceptr Y, unsigned rows, CUdeviceptr AS, int acc, CUdeviceptr scale) {
	int op = gemv_op(type, K, acc);
	if (!op) { fprintf(stderr, "prog: no GEMV op for type %u K %d\n", type, K); exit(1); }
	if (gemv_op_is_q2(op)) ph_q2(P, op, W, Xq, S32, Y, rows, 0, AS, scale, 0);
	else if (gemv_op_is_b32(op)) ph_b32(P, op, W, Xq, S32, Y, rows, scale, AS);
	else ph_gemvk(P, op, W, Xq, S32, Y, rows, AS);
}

/* ---- exchange ---- */
CUdeviceptr ph_xchg(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr src, unsigned nbytes, unsigned myoff, unsigned total) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_XSEND, 16);
	ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, nbytes); ph_u32(p, 0x154, myoff); ph_u32(p, 0x158, slot); ph_u32(p, 0x15c, (unsigned)ndies); ph_ptr(p, 0x160, x->xgen);
	p = prog_add(P, OP_XWAIT, 16);
	ph_ptr(p, 0x140, x->xtab); ph_u32(p, 0x148, (unsigned)me); ph_u32(p, 0x14c, (unsigned)ndies); ph_ptr(p, 0x150, x->xgen); ph_u32(p, 0x158, slot); ph_u32(p, 0x15c, total);
	P->xj++;
	return x->inbox + slot;
}

/* ---- the split ---- */
void split_of(split_t *s, int d, int n, int blk) {
	s->d = d; s->n = n;
	s->dj0 = 324 * d / n;  s->djn = 324 * (d + 1) / n - s->dj0;
	s->cb0 = 80 * d / n;   s->cbn = 80 * (d + 1) / n - s->cb0;
	s->h0 = 48 * d / n;    s->hn = 48 * (d + 1) / n - s->h0;
	s->klo = s->h0 / 3;    s->khi = (s->h0 + s->hn - 1) / 3; s->nk = s->khi - s->klo + 1;
	s->y0 = 2560 * d / n;  s->yn = 2560 * (d + 1) / n - s->y0;
	s->e0 = 512 * d / n;   s->en = 512 * (d + 1) / n - s->e0;
	s->shexp = (blk % n) == d;
}

/* hyper-connection read: xh = per-lane RMS(S) * norm (whole, on every die); this die's
 * rows of down ++ inject, gathered; t = SiLU(0.25 down) ++ inj; this die's rows of
 * up (the 4 lanes of its z columns); z columns mixed locally, gathered; z quantised.
 * Returns the gathered z (an inbox slot). */
static CUdeviceptr ph_hc_read(prog_t *P, const split_t *s, const xchg_t *x, CUdeviceptr norm, CUdeviceptr dj, CUdeviceptr up, const dn_bufs_t *b, float eps) {
	unsigned cols = 32u * (unsigned)s->cbn;
	ph_rms(P, OP_RMS2560, b->S, norm, b->xh, eps, 4);
	ph_bf16(P, OP_BF16HC, dj, b->xh, b->ldown, (unsigned)s->djn, 0);
	CUdeviceptr DOWN = ph_xchg(P, x, s->d, s->n, b->ldown, 4u * (unsigned)s->djn, 4u * (unsigned)s->dj0, 4 * 324);
	ph_hc(P, OP_HCSILU, DOWN, b->t, 0, 324, 320, 2);
	ph_bf16(P, OP_BF16K320, up, b->t, b->lup, 4 * cols, 0);
	ph_hc(P, OP_HCMIX, b->lup, b->xh + 4u * 32u * (unsigned)s->cb0, b->lz, cols, 4 * cols, (cols + 255) / 256);
	CUdeviceptr Z = ph_xchg(P, x, s->d, s->n, b->lz, 4 * cols, 4u * 32u * (unsigned)s->cb0, 4 * 2560);
	ph_actq(P, OP_ACTQ2560, Z, b->zq, b->zs32, b->zs16, b->zas, 2560, 1);
	return Z;
}

void prog_deltanet_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	unsigned nloc = SPLIT_NLOC(s), hrows = 128u * (unsigned)s->hn;
	CUdeviceptr inj = b->t + 320 * 4;
	/* ---- attention-side hyper-block: Gated DeltaNet ---- */
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps);
	ph_gemv_any(P, w->attn_qkv_type, 2560, w->attn_qkv, b->zq, b->zs32, b->lqkv, nloc, b->zas, 0, 0);
	ph_gemv_any(P, w->attn_gate_type, 2560, w->attn_gate, b->zq, b->zs32, b->lgate, hrows, b->zas, 0, 0);
	ph_bf16(P, OP_BF16BIAS, w->ssm_ab, Z, b->la, 2u * (unsigned)s->hn, w->ssm_ab_bias);
	ph_conv1d(P, b->lqkv, b->conv_state, w->ssm_conv1d, b->lu, t, nloc);
	ph_dnstep(P, b->dn_state, b->lu, b->lu + 512u * (unsigned)s->nk, b->lu + 1024u * (unsigned)s->nk, b->la, b->la + 4u * (unsigned)s->hn, w->ssm_a, b->lo, eps, (unsigned)s->h0, (unsigned)s->klo, (unsigned)s->hn);
	ph_dnnorm(P, b->lo, w->ssm_norm, b->lgate, b->lon, eps, (unsigned)s->hn);
	CUdeviceptr ON = ph_xchg(P, x, s->d, s->n, b->lon, 4 * hrows, 4u * 128u * (unsigned)s->h0, 4 * 6144);
	ph_actq(P, OP_ACTQ6144, ON, b->oq, b->os32, b->os16, b->oas, 6144, 1);
	ph_gemv_any(P, w->ssm_out_type, 6144, w->ssm_out, b->oq, b->os32, b->ly, (unsigned)s->yn, b->oas, 0, 0);
	CUdeviceptr Y = ph_xchg(P, x, s->d, s->n, b->ly, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	ph_hc(P, OP_HCWRITE, inj, Y, b->S, 2560, 0, 10);
	/* ---- FFN hyper-block: MoE (expert split: this die's routed experts, partials reduced) ---- */
	Z = ph_hc_read(P, s, x, w->hc_ffn_norm, w->hc_ffn_dj, w->hc_ffn_up, b, eps);
	ph_bf16(P, OP_BF16K2560, w->ffn_gate_inp_x, Z, b->r, 513, 0);
	ph_router(P, b->r, b->r + 512 * 4, b->idx, b->w, b->s);
	ph_moesel(P, b->idx, b->w, w->emap, b->lidx, b->lw, b->ne, b->part);
	ph_q2(P, OP_Q2MOEGN, w->ffn_gate_exps, b->zq, b->zs32, b->g, 640, b->lidx, b->zas, 0, b->ne);
	ph_q2(P, OP_Q2MOEGN, w->ffn_up_exps, b->zq, b->zs32, b->gu, 640, b->lidx, b->zas, 0, b->ne);
	if (s->shexp) {
		ph_gemv_any(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp, b->zq, b->zs32, b->g + 6400 * 4, 640, b->zas, 0, 0);
		ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->gu + 6400 * 4, 640, b->zas, 0, 0);
	}
	ph_ffnact(P, b->g, b->gu, b->hq, b->hs32, b->hs16, b->has, 640, s->shexp ? 11 : 10);
	ph_q2(P, OP_Q2MOEDN, w->ffn_down_exps, b->hq, b->hs32, b->part, 2560, b->lidx, b->has, b->lw, b->ne);
	if (s->shexp) ph_gemv_any(P, w->ffn_down_shexp_type, 640, w->ffn_down_shexp, b->hq + 10 * 640 * 2, b->hs32 + 10 * 20 * 4, b->part, 2560, b->has + 10 * 20 * 4, 1, b->s);
	CUdeviceptr PARTS = ph_xchg(P, x, s->d, s->n, b->part, 2560 * 4, 2560u * 4u * (unsigned)s->d, 2560u * 4u * (unsigned)s->n);
	ph_sumn(P, PARTS, 2560 * 4, b->out, 2560, (unsigned)s->n);
	ph_hc(P, OP_HCWRITE, inj, b->out, b->S, 2560, 0, 10);
}
