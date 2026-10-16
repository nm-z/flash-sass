#include "prog.h"
#include "split.h"
#include "gguf.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void prog_init(prog_t *P) { P->cap = 64; P->n = 0; P->xj = 0; P->loop = 0; P->pending_wait = 0; P->p = malloc(sizeof(phase_t) * P->cap); }
void prog_free(prog_t *P) { free(P->p); P->p = NULL; P->n = P->cap = 0; }
phase_t *prog_add(prog_t *P, int op, unsigned units) {
	if (P->n == P->cap) { P->cap *= 2; P->p = realloc(P->p, sizeof(phase_t) * P->cap); }
	phase_t *p = &P->p[P->n++];
	memset(p, 0, sizeof *p);
	p->w[0] = (uint32_t)op; p->w[1] = units;
	if (P->pending_wait) { p->w[30] |= 2; P->pending_wait = 0; }   /* wait for the exchange before running */
	return p;
}
void ph_ptr(phase_t *p, int addr, CUdeviceptr v) { int w = 2 + (addr - 0x140) / 4; p->w[w] = (uint32_t)v; p->w[w + 1] = (uint32_t)(v >> 32); }
void ph_u32(phase_t *p, int addr, uint32_t v) { p->w[2 + (addr - 0x140) / 4] = v; }
void ph_f32(phase_t *p, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); ph_u32(p, addr, u); }
void ph_tokv(prog_t *P, phase_t *p, int addr, uint32_t base, uint32_t value) {
	if (P->loop) { ph_u32(p, addr, base); p->w[31] |= 1u << (2 + (addr - 0x140) / 4); } else ph_u32(p, addr, value);   /* descriptor word 2 + i */
}

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
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, STATE); ph_ptr(p, 0x150, W); ph_ptr(p, 0x158, U); ph_tokv(P, p, 0x160, 0, t); ph_u32(p, 0x164, n);
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
	case GG_Q6_K: return K == 2560 ? OP_Q6KK2560 : getenv("FAKE_Q6K") ? OP_Q5KK6144 : 0;   /* FAKE_Q6K: timing runs only (wrong numbers) until the K=6144 Q6_K kernel exists */
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

void ph_gemv_any(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr Xq, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr Y, unsigned rows, CUdeviceptr AS, int acc, CUdeviceptr scale) {
	if (type == GG_Q6_K && K == 6144) {   /* four K = 1536 column slices (the holder's layout), y = ((p0 + p1) + p2) + p3 */
		size_t sl = (size_t)rows * (size_t)q6_k_tile_stride(1536);
		for (unsigned s = 0; s < 4; s++) ph_gemvk(P, s ? OP_Q6KK1536ACC : OP_Q6KK1536, W + sl * s, Xq + 2u * 1536u * s, S16 + 4u * 96u * s, Y, rows, AS + 4u * 48u * s);
		return;
	}
	int op = gemv_op(type, K, acc);
	if (!op) { fprintf(stderr, "prog: no GEMV op for type %u K %d\n", type, K); exit(1); }
	if (gemv_op_is_q2(op)) ph_q2(P, op, W, Xq, S32, Y, rows, 0, AS, scale, 0);
	else if (gemv_op_is_b32(op)) ph_b32(P, op, W, Xq, S32, Y, rows, scale, AS);
	else ph_gemvk(P, op, W, Xq, type == GG_Q6_K ? S16 : S32, Y, rows, AS);   /* Q6_K takes the per-16 sums */
}

/* ---- exchange ---- */
CUdeviceptr ph_xchg(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr src, unsigned nbytes, unsigned myoff, unsigned total) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_XSEND, 16);
	ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, nbytes); ph_u32(p, 0x154, myoff); ph_u32(p, 0x158, slot); ph_u32(p, 0x15c, (unsigned)ndies); ph_ptr(p, 0x160, x->xgen); ph_u32(p, 0x168, (unsigned)me);
	p->w[30] |= 1;             /* a send (every CTA publishes its flag itself; the bit is informational) ... */
	P->pending_wait = 1;       /* ... and the next phase polls for every die's and CTA's flag first */
	(void)total;
	P->xj++;
	return x->inbox + slot;
}

void prog_token_end(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr any) {
	if (P->loop && (P->xj & 1)) ph_xchg(P, x, me, ndies, any, 0, 0, 16);
}

/* ---- the split ---- */
void split_of(split_t *s, int d, int n, int blk) {
	s->d = d; s->n = n;
	s->dj0 = split_bound(324, d, n);  s->djn = split_bound(324, d + 1, n) - s->dj0;
	s->cb0 = split_bound(80, d, n);   s->cbn = split_bound(80, d + 1, n) - s->cb0;
	s->h0 = split_bound(48, d, n);    s->hn = split_bound(48, d + 1, n) - s->h0;
	s->klo = s->h0 / 3;    s->khi = (s->h0 + s->hn - 1) / 3; s->nk = s->khi - s->klo + 1;
	s->y0 = split_bound(2560, d, n);  s->yn = split_bound(2560, d + 1, n) - s->y0;
	s->pk0 = split_bound(10240, d, n); s->pkn = split_bound(10240, d + 1, n) - s->pk0;
	s->o0 = split_bound(248320, d, n); s->on = split_bound(248320, d + 1, n) - s->o0;
	s->hd0 = split_bound(320, d, n);  s->hdn = split_bound(320, d + 1, n) - s->hd0;
	s->e0 = split_bound(512, d, n);   s->en = split_bound(512, d + 1, n) - s->e0;
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
	if (getenv("PROG_DUP")) ph_bf16(P, OP_BF16HC, dj, b->xh, b->ldown, (unsigned)s->djn, 0);   /* timing experiment: the same phase twice (warm code) */
	CUdeviceptr DOWN = ph_xchg(P, x, s->d, s->n, b->ldown, 4u * (unsigned)s->djn, 4u * (unsigned)s->dj0, 4 * 324);
	ph_hc(P, OP_HCSILU, DOWN, b->t, 0, 324, 320, 2);
	ph_bf16(P, OP_BF16K320, up, b->t, b->lup, 4 * cols, 0);
	ph_hc(P, OP_HCMIX, b->lup, b->xh + 4u * 32u * (unsigned)s->cb0, b->lz, cols, 4 * cols, (cols + 255) / 256);
	CUdeviceptr Z = ph_xchg(P, x, s->d, s->n, b->lz, 4 * cols, 4u * 32u * (unsigned)s->cb0, 4 * 2560);
	ph_actq(P, OP_ACTQ256, Z, b->zq, b->zs32, b->zs16, b->zas, 256, 10);
	return Z;
}

/* the MoE hyper-block (expert split: this die's routed experts, partials reduced) */
static void ph_moe_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, float eps) {
	CUdeviceptr inj = b->t + 320 * 4;
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_ffn_norm, w->hc_ffn_dj, w->hc_ffn_up, b, eps);
	ph_bf16(P, OP_BF16K2560, w->ffn_gate_inp_x, Z, b->r, 513, 0);
	ph_router(P, b->r, b->r + 512 * 4, b->idx, b->w, b->s);
	ph_moesel(P, b->idx, b->w, w->emap, b->lidx, b->lw, b->ne, b->part);
	ph_q2(P, OP_Q2MOEGN, w->ffn_gate_exps, b->zq, b->zs32, b->g, 640, b->lidx, b->zas, 0, b->ne);
	ph_q2(P, OP_Q2MOEGN, w->ffn_up_exps, b->zq, b->zs32, b->gu, 640, b->lidx, b->zas, 0, b->ne);
	if (s->shexp) {
		ph_gemv_any(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp, b->zq, b->zs32, b->zs16, b->g + 6400 * 4, 640, b->zas, 0, 0);
		ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->zs16, b->gu + 6400 * 4, 640, b->zas, 0, 0);
		if (getenv("PROG_DUP")) ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->zs16, b->gu + 6400 * 4, 640, b->zas, 0, 0);
	}
	ph_ffnact(P, b->g, b->gu, b->hq, b->hs32, b->hs16, b->has, 640, s->shexp ? 11 : 10);
	ph_q2(P, OP_Q2MOEDN, w->ffn_down_exps, b->hq, b->hs32, b->part, 2560, b->lidx, b->has, b->lw, b->ne);
	if (s->shexp) ph_gemv_any(P, w->ffn_down_shexp_type, 640, w->ffn_down_shexp, b->hq + 10 * 640 * 2, b->hs32 + 10 * 20 * 4, b->hs16 + 10 * 40 * 4, b->part, 2560, b->has + 10 * 20 * 4, 1, b->s);
	CUdeviceptr PARTS = ph_xchg(P, x, s->d, s->n, b->part, 2560 * 4, 2560u * 4u * (unsigned)s->d, 2560u * 4u * (unsigned)s->n);
	ph_sumn(P, PARTS, 2560 * 4, b->out, 2560, (unsigned)s->n);
	ph_hc(P, OP_HCWRITE, inj, b->out, b->S, 2560, 0, 10);
}

/* attention proper on die 0 (v1): q/k/v GEMVs, per-head norm + RoPE + cache append,
 * flash-decoding partials over the t+1 keys, merge with the output gate */
static void ph_attn_core(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	ph_gemv_any(P, w->attn_q_type, 2560, w->attn_q, b->zq, b->zs32, b->zs16, b->qg, 12288, b->zas, 0, 0);
	ph_gemv_any(P, w->attn_k_type, 2560, w->attn_k, b->zq, b->zs32, b->zs16, b->k, 512, b->zas, 0, 0);
	ph_gemv_any(P, w->attn_v_type, 2560, w->attn_v, b->zq, b->zs32, b->zs16, b->v, 512, b->zas, 0, 0);
	phase_t *p = prog_add(P, OP_ATTNQK, 26);
	ph_ptr(p, 0x140, b->qg); ph_ptr(p, 0x148, b->k); ph_ptr(p, 0x150, b->v); ph_ptr(p, 0x158, w->attn_q_norm); ph_ptr(p, 0x160, w->attn_k_norm);
	ph_ptr(p, 0x168, b->rope_cs); ph_ptr(p, 0x170, b->rope_sn); ph_ptr(p, 0x178, b->qo); ph_ptr(p, 0x180, b->kc); ph_ptr(p, 0x188, b->vc);
	ph_tokv(P, p, 0x190, 0, t); ph_f32(p, 0x194, eps);
	p = prog_add(P, OP_ATTNPART, 32);
	ph_ptr(p, 0x140, b->qo); ph_ptr(p, 0x148, b->kc); ph_ptr(p, 0x150, b->vc); ph_ptr(p, 0x158, b->pm); ph_ptr(p, 0x160, b->pd); ph_ptr(p, 0x168, b->po); ph_tokv(P, p, 0x170, 1, t + 1);
	p = prog_add(P, OP_ATTNMERGE, 24);
	ph_ptr(p, 0x140, b->pm); ph_ptr(p, 0x148, b->pd); ph_ptr(p, 0x150, b->po); ph_ptr(p, 0x158, b->qg); ph_ptr(p, 0x160, b->aout);
}

void prog_attn_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	CUdeviceptr inj = b->t + 320 * 4;
	ph_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps);
	if (s->d == attn_die()) ph_attn_core(P, w, b, t, eps);
	CUdeviceptr AOUT = ph_xchg(P, x, s->d, s->n, b->aout, s->d == attn_die() ? 6144 * 4 : 0, 0, 6144 * 4);
	ph_actq(P, OP_ACTQ256, AOUT, b->oq, b->os32, b->os16, b->oas, 256, 24);
	ph_gemv_any(P, w->attn_output_type, 6144, w->attn_output, b->oq, b->os32, b->os16, b->ly, (unsigned)s->yn, b->oas, 0, 0);
	CUdeviceptr Y = ph_xchg(P, x, s->d, s->n, b->ly, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	ph_hc(P, OP_HCWRITE, inj, Y, b->S, 2560, 0, 10);
	ph_moe_half(P, s, x, w, b, eps);
}

void prog_embed(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t) {
	phase_t *p = prog_add(P, OP_EMBROW, 10);
	ph_ptr(p, 0x140, w->token_embd); ph_ptr(p, 0x148, b->S); ph_tokv(P, p, 0x150, 0, t); ph_ptr(p, 0x158, b->ids);
}

void prog_head(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, float eps, unsigned t, unsigned nprompt) {
	unsigned cols = 32u * (unsigned)s->cbn;
	/* the read gate without an inject: 320 down rows, nothing copied behind t */
	ph_rms(P, OP_RMS2560, b->S, w->output_hc_norm, b->xh, eps, 4);
	ph_bf16(P, OP_BF16HC, w->output_hc_down, b->xh, b->ldown, (unsigned)s->hdn, 0);
	CUdeviceptr DOWN = ph_xchg(P, x, s->d, s->n, b->ldown, 4u * (unsigned)s->hdn, 4u * (unsigned)s->hd0, 4 * 320);
	ph_hc(P, OP_HCSILU, DOWN, b->t, 0, 320, 320, 2);
	ph_bf16(P, OP_BF16K320, w->output_hc_up, b->t, b->lup, 4 * cols, 0);
	ph_hc(P, OP_HCMIX, b->lup, b->xh + 4u * 32u * (unsigned)s->cb0, b->lz, cols, 4 * cols, (cols + 255) / 256);
	CUdeviceptr Z = ph_xchg(P, x, s->d, s->n, b->lz, 4 * cols, 4u * 32u * (unsigned)s->cb0, 4 * 2560);
	ph_actq(P, OP_ACTQ256, Z, b->zq, b->zs32, b->zs16, b->zas, 256, 10);
	ph_gemv_any(P, GG_Q5_K, 2560, w->output, b->zq, b->zs32, b->zs16, b->hlo, (unsigned)s->on, b->zas, 0, 0);
	phase_t *p = prog_add(P, OP_ARGMAX, 16);
	ph_ptr(p, 0x140, b->hlo); ph_ptr(p, 0x148, b->amo); ph_ptr(p, 0x150, b->apart); ph_ptr(p, 0x158, b->acnt); ph_u32(p, 0x160, (unsigned)s->on); ph_u32(p, 0x164, 16);
	CUdeviceptr PAIRS = ph_xchg(P, x, s->d, s->n, b->amo, 8, 8u * (unsigned)s->d, 8u * (unsigned)s->n);
	p = prog_add(P, OP_AMAX, 1);
	ph_ptr(p, 0x140, PAIRS); ph_ptr(p, 0x148, w->obases); ph_u32(p, 0x150, (unsigned)s->n); ph_ptr(p, 0x158, b->tokout);
	ph_ptr(p, 0x160, b->ids); ph_ptr(p, 0x168, b->toks); ph_tokv(P, p, 0x170, 0, t); ph_u32(p, 0x174, nprompt);
}

void prog_ple_prefix(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	phase_t *p = prog_add(P, OP_PLEROWS, 1);   /* every die hashes and gathers the 16 rows itself */
	ph_ptr(p, 0x140, b->ids); ph_ptr(p, 0x148, b->ple_pc); ph_ptr(p, 0x150, b->ple_tab); ph_ptr(p, 0x158, b->pg); ph_ptr(p, 0x160, b->prows); ph_tokv(P, p, 0x168, 0, t);
	ph_actq(P, OP_ACTQ256, b->pg, b->pgq, b->pgs32, b->pgs16, b->pgas, 256, 10);
	ph_gemv_any(P, GG_Q2_0, 2560, w->ple_key, b->pgq, b->pgs32, b->pgs16, b->lk, (unsigned)s->pkn, b->pgas, 0, 0);
	CUdeviceptr K = ph_xchg(P, x, s->d, s->n, b->lk, 4u * (unsigned)s->pkn, 4u * (unsigned)s->pk0, 4 * 10240);
	ph_rms(P, OP_RMS2560, K, w->ple_norm_key, b->kn, eps, 4);
	ph_rms(P, OP_RMS2560, b->S, w->ple_norm_query, b->qn, eps, 4);
	ph_bf16(P, OP_BF16K2560, w->ple_value, b->pg, b->lv, (unsigned)s->yn, 0);
	CUdeviceptr V = ph_xchg(P, x, s->d, s->n, b->lv, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	p = prog_add(P, OP_PLEGATE, 4);
	ph_ptr(p, 0x140, b->kn); ph_ptr(p, 0x148, b->qn); ph_ptr(p, 0x150, V); ph_ptr(p, 0x158, b->pG); ph_ptr(p, 0x160, b->pgates);
	ph_rms(P, OP_RMS2560, b->pG, w->ple_norm_conv, b->px, eps, 4);
	p = prog_add(P, OP_PLECONV, 10);
	ph_ptr(p, 0x140, b->px); ph_ptr(p, 0x148, b->ple_state); ph_ptr(p, 0x150, w->ple_conv1d); ph_ptr(p, 0x158, b->pG); ph_ptr(p, 0x160, b->S); ph_tokv(P, p, 0x168, 0, t);
}

void prog_deltanet_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	unsigned nloc = SPLIT_NLOC(s), hrows = 128u * (unsigned)s->hn;
	CUdeviceptr inj = b->t + 320 * 4;
	/* ---- attention-side hyper-block: Gated DeltaNet ---- */
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps);
	ph_gemv_any(P, w->attn_qkv_type, 2560, w->attn_qkv, b->zq, b->zs32, b->zs16, b->lqkv, nloc, b->zas, 0, 0);
	ph_gemv_any(P, w->attn_gate_type, 2560, w->attn_gate, b->zq, b->zs32, b->zs16, b->lgate, hrows, b->zas, 0, 0);
	ph_bf16(P, OP_BF16BIAS, w->ssm_ab, Z, b->la, 2u * (unsigned)s->hn, w->ssm_ab_bias);
	ph_conv1d(P, b->lqkv, b->conv_state, w->ssm_conv1d, b->lu, t, nloc);
	ph_dnstep(P, b->dn_state, b->lu, b->lu + 512u * (unsigned)s->nk, b->lu + 1024u * (unsigned)s->nk, b->la, b->la + 4u * (unsigned)s->hn, w->ssm_a, b->lo, eps, (unsigned)s->h0, (unsigned)s->klo, (unsigned)s->hn);
	ph_dnnorm(P, b->lo, w->ssm_norm, b->lgate, b->lon, eps, (unsigned)s->hn);
	CUdeviceptr ON = ph_xchg(P, x, s->d, s->n, b->lon, 4 * hrows, 4u * 128u * (unsigned)s->h0, 4 * 6144);
	ph_actq(P, OP_ACTQ256, ON, b->oq, b->os32, b->os16, b->oas, 256, 24);
	ph_gemv_any(P, w->ssm_out_type, 6144, w->ssm_out, b->oq, b->os32, b->os16, b->ly, (unsigned)s->yn, b->oas, 0, 0);
	CUdeviceptr Y = ph_xchg(P, x, s->d, s->n, b->ly, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	ph_hc(P, OP_HCWRITE, inj, Y, b->S, 2560, 0, 10);
	ph_moe_half(P, s, x, w, b, eps);
}
