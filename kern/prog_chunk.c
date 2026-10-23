#include "prog_chunk.h"
#include "gguf.h"
#include "quant.h"
#include "split.h"
#include <stdio.h>
#include <stdlib.h>

static unsigned lg2(unsigned v) { unsigned l = 0; while ((1u << l) < v) l++; return l; }

/* the GEMM ops' descriptor (gen_gemm.lua): 0x140 W, 0x148 Xt, 0x150 C, 0x158 rows, 0x15c npos, 0x160 K, 0x164 ldx, 0x170 S, 0x178 crow,
 * 0x17c cstride, 0x180 srow, 0x184 sstride; the holder's tiles (ref/quant.c repack_*), placed as gemm_die_test places them */
void ph_gemm(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr X, CUdeviceptr C, unsigned rows, unsigned npos, unsigned ldx) {
	int op, stride;
	unsigned coff = 0, cstride = 16, sstride = 2, Q = (unsigned)q4_k_lanes_per_row(K) / 8u;
	switch (type) {
	case GG_BF16:   op = OP_GEMM_BF16;  stride = 2 * K; break;
	case GG_Q4_K:   op = OP_GEMM_Q4K;   stride = q4_k_tile_stride(K); break;
	case GG_Q5_K:   op = OP_GEMM_Q5K;   stride = q5_k_tile_stride(K); break;
	case GG_Q2_0:   op = OP_GEMM_Q2;    stride = q2_0_tile_stride(K); coff = (unsigned)stride - 16u * (unsigned)(K / 64); break;
	case GG_Q3_K:   op = OP_GEMM_Q3;    stride = q3_k_tile_stride(K); coff = 16u * Q; cstride = 112; sstride = Q == 4 ? 2 : 1; break;
	case GG_IQ4_XS: op = OP_GEMM_IQ4;   stride = iq4_xs_tile_stride(K); coff = 16u * Q; sstride = Q == 4 ? 2 : 1; break;
	case GG_Q6_K:   op = OP_GEMM_Q6K;   stride = q6_k_tile_stride(K); coff = 16u * Q; sstride = Q == 4 ? 2 : 1;
		if (K == 6144) { fprintf(stderr, "ph_gemm: Q6_K K=6144 is sliced (ph_gemm_k6144)\n"); exit(1); }
		break;
	case GG_Q4_0:   op = OP_GEMM_Q40;   stride = b32_tile_stride(type, K); break;
	case GG_Q5_0:   op = OP_GEMM_Q50;   stride = b32_tile_stride(type, K); break;
	case GG_IQ4_NL: op = OP_GEMM_IQ4NL; stride = b32_tile_stride(type, K); break;
	case GG_Q8_0:   op = OP_GEMM_Q80;   stride = b32_tile_stride(type, K); break;
	default: fprintf(stderr, "ph_gemm: no GEMM for type %u\n", type); exit(1);
	}
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W + coff); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, C);
	ph_u32(p, 0x158, rows); ph_u32(p, 0x15c, npos); ph_u32(p, 0x160, (unsigned)K); ph_u32(p, 0x164, ldx);
	ph_ptr(p, 0x170, W); ph_u32(p, 0x178, (unsigned)stride); ph_u32(p, 0x17c, cstride); ph_u32(p, 0x180, (unsigned)stride); ph_u32(p, 0x184, sstride);
}

/* C = W X for K = 6144: a Q6_K weight is held as four K = 1536 column slices (kern/hold.c: slice s of the rows at
 * W + s rows stride(1536)), each a GEMM of X's rows 1536 s .. into SC (scratch [4 rows][ldx]), then C = their sum */
static void ph_gemm_k6144(prog_t *P, uint32_t type, CUdeviceptr W, CUdeviceptr X, CUdeviceptr C, unsigned rows, unsigned npos, unsigned ldx, CUdeviceptr SC) {
	if (type != GG_Q6_K) { ph_gemm(P, type, 6144, W, X, C, rows, npos, ldx); return; }
	if (!SC) { fprintf(stderr, "Q6_K K=6144 GEMM without its scratch\n"); exit(1); }
	unsigned long long row = 4ull * ldx, st = (unsigned long long)q6_k_tile_stride(1536);
	for (int s = 0; s < 4; s++) {
		ph_gemm(P, GG_Q6_K, 1536, W + (unsigned long long)s * rows * st, X + 1536ull * s * row, SC + (unsigned long long)s * rows * row, rows, npos, ldx);
		if (s < 3) prog_nobar(P);
	}
	unsigned n4 = rows * ldx / 4u;
	phase_t *p = prog_add(P, OP_SLAB_SUM, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, SC); ph_ptr(p, 0x148, C); ph_u32(p, 0x150, n4); ph_u32(p, 0x154, 4); ph_u32(p, 0x158, rows); ph_u32(p, 0x15c, lg2(ldx));
}

/* C = W X for a BF16 weight with few rows (few 128-row tiles: SMs idle): K split into S segments (S tiles fill the 16 SMs) whose
 * products land in c->Q6T, then C = their sum. Other types, or no room, as one GEMM. */
static void ph_gemm_ks(prog_t *P, const chunk_bufs_t *c, uint32_t type, int K, CUdeviceptr W, CUdeviceptr X, CUdeviceptr C, unsigned rows, unsigned npos) {
	unsigned ldx = c->ldx, tiles = (rows + 127u) / 128u * (ldx / 128u), S = 1;
	if (type == GG_BF16 && c->Q6T) while (tiles * S < 16u && S < 16u && (unsigned)K % (32u * S) == 0 && 2u * S * rows <= c->q6t_rows) S *= 2;
	if (S == 1) { ph_gemm(P, type, K, W, X, C, rows, npos, ldx); return; }
	unsigned Ks = (unsigned)K / S;
	unsigned long long row = 4ull * ldx;
	/* one phase: the GEMM's tiles of slice s (K rows Ks s ..) read W + 2 Ks s, Xt + row Ks s and write Q6T + row rows s */
	phase_t *p = prog_add(P, OP_GEMM_BF16, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, c->Q6T);
	ph_u32(p, 0x158, rows); ph_u32(p, 0x15c, npos); ph_u32(p, 0x160, Ks); ph_u32(p, 0x164, ldx);
	ph_ptr(p, 0x170, W); ph_u32(p, 0x178, 2u * (unsigned)K); ph_u32(p, 0x17c, 16); ph_u32(p, 0x180, 2u * (unsigned)K); ph_u32(p, 0x184, 2);
	ph_u32(p, 0x188, S); ph_u32(p, 0x18c, 2u * Ks); ph_u32(p, 0x190, (unsigned)(row * Ks)); ph_u32(p, 0x194, (unsigned)(row * rows));
	unsigned n4 = rows * ldx / 4u;
	p = prog_add(P, OP_SLAB_SUM, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, c->Q6T); ph_ptr(p, 0x148, C); ph_u32(p, 0x150, n4); ph_u32(p, 0x154, S); ph_u32(p, 0x158, rows); ph_u32(p, 0x15c, lg2(ldx));
}

/* the fp16 exchange's routes on cards of two dies (die 2k, 2k + 1): source s sends to die t directly when they share a card or
 * a parity, or t has no partner; otherwise (x->relay) t's partner gets it and t reads it there (slab_uh). One copy of each
 * slice crosses to each other card instead of two. */
static int xh_direct(const xchg_t *x, int s, int t) {
	return !x->relay || s == t || (s >> 1) == (t >> 1) || (s & 1) == (t & 1) || (t ^ 1) >= x->xnd;
}

/* after every die's program is built: each fp16 exchange's unpack gets the whole gathered extent and the byte ranges of the
 * relayed sources (read from the partner's inbox), from all dies' records of it */
void prog_chunk_xh_patch(prog_t *P, const xchg_t *x, int ndies) {
	for (int t = 0; t < ndies; t++) if (P[t].nxr != P[0].nxr) { fprintf(stderr, "chunk exchange: die %d has %d fp16 exchanges, die 0 %d\n", t, P[t].nxr, P[0].nxr); exit(1); }
	for (int j = 0; j < P[0].nxr; j++) {
		unsigned long long end = 0;
		for (int e = 0; e < ndies; e++) { unsigned long long b = (unsigned long long)P[e].xr[3 * j + 1] + P[e].xr[3 * j + 2]; if (b > end) end = b; }
		if (getenv("FLASH_CHUNK_OPS")) {   /* debugging: the sources of exchange j must tile the extent without overlap */
			for (int e = 0; e < ndies; e++) for (int f = e + 1; f < ndies; f++) {
				unsigned long long ae = P[e].xr[3 * j + 1], be = ae + P[e].xr[3 * j + 2], af = P[f].xr[3 * j + 1], bf = af + P[f].xr[3 * j + 2];
				if (P[e].xr[3 * j + 2] && P[f].xr[3 * j + 2] && ae < bf && af < be) fprintf(stderr, "chunk exchange %d: die %d [%llu, %llu) overlaps die %d [%llu, %llu)\n", j, e, ae, be, f, af, bf);
			}
			fprintf(stderr, "chunk exchange %d: extent %llu, phases", j, end);
			for (int e = 0; e < ndies; e++) fprintf(stderr, " %u:%u+%u", P[e].xr[3 * j], P[e].xr[3 * j + 1], P[e].xr[3 * j + 2]);
			fputc('\n', stderr);
		}
		for (int t = 0; t < ndies; t++) {
			phase_t *p = &P[t].p[P[t].xr[3 * j]];
			unsigned n16 = (unsigned)(end / 16u), r[4] = { 0 }, nr = 0;
			for (int s = 0; s < ndies; s++) {
				if (xh_direct(&x[t], s, t) || !P[s].xr[3 * j + 2]) continue;
				if (nr == 4) { fprintf(stderr, "chunk exchange: more than two relayed sources for die %d\n", t); exit(1); }
				r[nr++] = P[s].xr[3 * j + 1] / 16u; r[nr++] = (P[s].xr[3 * j + 1] + P[s].xr[3 * j + 2]) / 16u;
			}
			if (nr && !x[t].pinbox) { fprintf(stderr, "chunk exchange: die %d relays without a partner inbox\n", t); exit(1); }
			p->w[1] = n16 ? (n16 + 1023u) / 1024u : 1u;
			ph_u32(p, 0x158, n16); ph_u32(p, 0x15c, r[0]); ph_u32(p, 0x160, r[1]); ph_u32(p, 0x164, r[2]); ph_u32(p, 0x168, r[3]);
		}
	}
}

/* gather slab rows: this die's nbytes at src land at myoff of the next chunk slot of every die, the next phase waits. With
 * x->unp (and not raw), the rows travel as fp16 (slab_xh) and the next phase unpacks the slot to f32 (slab_uh, which then is
 * the result); raw: the bytes as they are (slab_xs / xsend: the attention's codes) */
static CUdeviceptr cxchg_raw(prog_t *P, const xchg_t *x, const split_t *s, CUdeviceptr src, unsigned long long nbytes, unsigned long long myoff, unsigned cslot, int raw) {
	unsigned slot = (unsigned)(P->xj & 1) * cslot;
	if (nbytes > 0xffffffffull || myoff + nbytes > cslot) { fprintf(stderr, "chunk exchange of %llu bytes at %llu does not fit a %u-byte slot\n", nbytes, myoff, cslot); exit(1); }
	phase_t *p;
	if (!raw && x->unp[0] && !(nbytes % 32) && !(myoff % 32) && !(slot % 16)) {
		unsigned n16 = (unsigned)(nbytes / 32), dmask = 0, par = (unsigned)(P->xj & 1);
		for (int t = 0; t < x->xnd; t++) if (xh_direct(x, x->xd, t)) dmask |= 1u << t;
		p = prog_add(P, OP_SLAB_XH, n16 ? (n16 + 255u) / 256u : 1u);
		ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, n16); ph_u32(p, 0x154, (unsigned)(myoff / 2)); ph_u32(p, 0x158, slot);
		ph_u32(p, 0x15c, dmask);
		p->w[30] |= 1; P->pending_wait = 1; P->xj++;
		if (P->nxr == P->xrcap) { P->xrcap = P->xrcap ? 2 * P->xrcap : 64; P->xr = realloc(P->xr, sizeof(unsigned) * 3 * (size_t)P->xrcap); }
		/* a phase (and a grid barrier) between the send's wait and the unpack: a unpack that follows the wait directly lost lines 1 and 3 of 128 B of its second 16 B store in lanes 8-15 / 24-31 of each warp (CHUNK_NOXHBAR: without) */
		if (!getenv("CHUNK_NOXHBAR")) { p = prog_add(P, OP_SLAB_ZERO, 1); ph_ptr(p, 0x140, x->unp[par]); ph_u32(p, 0x148, 0); }
		p = prog_add(P, OP_SLAB_UH, 1);   /* the extent and the relayed ranges: prog_chunk_xh_patch */
		ph_ptr(p, 0x140, x->inbox + slot); ph_ptr(p, 0x148, x->pinbox ? x->pinbox + slot : 0); ph_ptr(p, 0x150, x->unp[par]);
		P->xr[3 * P->nxr] = (unsigned)(P->n - 1); P->xr[3 * P->nxr + 1] = (unsigned)myoff; P->xr[3 * P->nxr + 2] = (unsigned)nbytes; P->nxr++;
		return x->unp[par];
	}
	if (!(nbytes % 16) && !(myoff % 16) && !(slot % 16) && !getenv("CHUNK_XSEND4")) {   /* 16 B words (slab_xs); CHUNK_XSEND4: the 4 B xsend */
		unsigned n16 = (unsigned)(nbytes / 16);
		p = prog_add(P, OP_SLAB_XS, n16 ? (n16 + 255u) / 256u : 1u);
		ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, n16); ph_u32(p, 0x154, (unsigned)myoff); ph_u32(p, 0x158, slot);
		ph_u32(p, 0x15c, (unsigned)s->n);
	} else {
		p = prog_add(P, OP_XSEND, 16);
		ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, (unsigned)nbytes); ph_u32(p, 0x154, (unsigned)myoff); ph_u32(p, 0x158, slot);
		ph_u32(p, 0x15c, (unsigned)s->n); ph_ptr(p, 0x160, x->xgen); ph_u32(p, 0x168, (unsigned)s->d);
	}
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}
static CUdeviceptr cxchg(prog_t *P, const xchg_t *x, const split_t *s, CUdeviceptr src, unsigned long long nbytes, unsigned long long myoff, unsigned cslot) {
	return cxchg_raw(P, x, s, src, nbytes, myoff, cslot, 0);
}

static void ph_slab_rms(prog_t *P, CUdeviceptr X, CUdeviceptr W, CUdeviceptr Y, CUdeviceptr G, unsigned nvec, unsigned vlen, unsigned wvs, float eps, unsigned ldx) {
	unsigned lgg = lg2(ldx / 32u);
	phase_t *p = prog_add(P, OP_SLAB_RMS, nvec << lgg);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, Y); ph_ptr(p, 0x158, G);
	ph_u32(p, 0x160, ldx); ph_u32(p, 0x164, vlen / 8u); ph_f32(p, 0x168, 1.0f / (float)vlen); ph_f32(p, 0x16c, eps); ph_u32(p, 0x170, lgg); ph_u32(p, 0x174, wvs);
}

/* the hc read of a chunk: XH = RMS(S) norm, this die's down ++ inject rows gathered, T = SiLU(0.25 down) ++ inject,
 * this die's up rows, its z columns mixed and gathered; returns the gathered Z [2560][ldx] (a chunk slot) */
static CUdeviceptr chunk_hc_read(prog_t *P, const split_t *s, const xchg_t *x, CUdeviceptr norm, CUdeviceptr dj, CUdeviceptr up, const chunk_bufs_t *c, float eps, unsigned npos, unsigned cslot) {
	unsigned ldx = c->ldx, lgx = lg2(ldx), cols = 32u * (unsigned)s->cbn, c0 = 32u * (unsigned)s->cb0;
	unsigned long long row = 4ull * ldx;
	ph_slab_rms(P, c->S, norm, c->XH, 0, 4, 2560, 2560, eps, ldx);
	ph_gemm_ks(P, c, GG_BF16, 10240, dj, c->XH, c->LDN, (unsigned)s->djn, npos);
	CUdeviceptr DOWN = cxchg(P, x, s, c->LDN, row * (unsigned)s->djn, row * (unsigned)s->dj0, cslot);
	unsigned n4 = 324u * ldx / 4u;
	phase_t *p = prog_add(P, OP_SLAB_SILU, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, DOWN); ph_ptr(p, 0x148, c->T); ph_u32(p, 0x150, n4); ph_u32(p, 0x154, 320u * ldx / 4u);
	ph_gemm_ks(P, c, GG_BF16, 320, up, c->T, c->LUP, 4u * cols, npos);
	p = prog_add(P, OP_SLAB_MIX, cols * ldx / 1024u);
	ph_ptr(p, 0x140, c->LUP); ph_ptr(p, 0x148, c->XH); ph_ptr(p, 0x150, c->LZ - row * c0); ph_u32(p, 0x158, lgx); ph_u32(p, 0x15c, cols); ph_u32(p, 0x160, c0);
	return cxchg(P, x, s, c->LZ, row * cols, row * c0, cslot);
}

void prog_chunk_deltanet_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                              unsigned t0, unsigned npos, float eps, unsigned cslot) {
	unsigned ldx = c->ldx, lgx = lg2(ldx), nloc = SPLIT_NLOC(s), hn = (unsigned)s->hn, hrows = 128u * hn, nk = (unsigned)s->nk;
	unsigned long long row = 4ull * ldx;
	CUdeviceptr Z = chunk_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, c, eps, npos, cslot);
	ph_gemm(P, w->attn_qkv_type, 2560, w->attn_qkv, Z, c->LQKV, nloc, npos, ldx);
	prog_nobar(P);
	ph_gemm(P, w->attn_gate_type, 2560, w->attn_gate, Z, c->LGATE, hrows, npos, ldx);
	prog_nobar(P);   /* qkv, gate and alpha / beta all read Z only */
	ph_gemm_ks(P, c, GG_BF16, 2560, w->ssm_ab, Z, c->LA, 2u * hn, npos);
	unsigned n4 = 2u * hn * ldx / 4u;
	phase_t *p = prog_add(P, OP_SLAB_BIAS, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, c->LA); ph_ptr(p, 0x148, w->ssm_ab_bias); ph_u32(p, 0x150, n4); ph_u32(p, 0x154, lgx);
	p = prog_add(P, OP_SLAB_CONV, nloc * ldx / 1024u);
	ph_ptr(p, 0x140, c->LQKV); ph_ptr(p, 0x148, b->conv_state); ph_ptr(p, 0x150, w->ssm_conv1d); ph_ptr(p, 0x158, c->LU);
	ph_u32(p, 0x160, t0); ph_u32(p, 0x164, nloc); ph_u32(p, 0x168, lgx); ph_u32(p, 0x16c, npos);
	p = prog_add(P, OP_DNSTEPC, hn);   /* q rows 128 nk, k rows 128 nk, v rows 128 hn of LU; alpha rows hn, beta rows hn of LA */
	ph_ptr(p, 0x140, b->dn_state); ph_ptr(p, 0x148, c->LU); ph_ptr(p, 0x150, c->LU + row * 128u * nk); ph_ptr(p, 0x158, c->LU + row * 256u * nk);
	ph_ptr(p, 0x160, c->LA); ph_ptr(p, 0x168, c->LA + row * hn); ph_ptr(p, 0x170, w->ssm_a); ph_ptr(p, 0x178, c->LO);
	ph_f32(p, 0x180, eps); ph_u32(p, 0x184, (unsigned)s->h0); ph_u32(p, 0x188, (unsigned)s->klo); ph_u32(p, 0x18c, ldx); ph_u32(p, 0x190, npos);
	ph_slab_rms(P, c->LO, w->ssm_norm, c->LON, c->LGATE, hn, 128, 0, eps, ldx);
	CUdeviceptr ON = cxchg(P, x, s, c->LON, row * hrows, row * 128u * (unsigned)s->h0, cslot);
	ph_gemm_k6144(P, w->ssm_out_type, w->ssm_out, ON, c->LY, (unsigned)s->yn, npos, ldx, c->Q6T);
	CUdeviceptr Y = cxchg(P, x, s, c->LY, row * (unsigned)s->yn, row * (unsigned)s->y0, cslot);
	p = prog_add(P, OP_SLAB_WRITE, 10240u * ldx / 1024u);   /* S += 2 sigmoid(0.25 inj) Y, inj = T's rows 320..323 */
	ph_ptr(p, 0x140, c->S); ph_ptr(p, 0x148, Y); ph_ptr(p, 0x150, c->T); ph_u32(p, 0x158, lgx);
}

static void ph_elem(prog_t *P, int op, CUdeviceptr a, CUdeviceptr b, CUdeviceptr y, unsigned n4) {
	phase_t *p = prog_add(P, op, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, a); ph_ptr(p, 0x148, b); ph_ptr(p, 0x150, y); ph_u32(p, 0x158, n4);
}

void prog_chunk_moe_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                         unsigned npos, float eps, unsigned cslot) {
	(void)b;
	unsigned ldx = c->ldx, lgx = lg2(ldx), npairs = 10u * npos;
	unsigned long long row = 4ull * ldx;
	CUdeviceptr Z = chunk_hc_read(P, s, x, w->hc_ffn_norm, w->hc_ffn_dj, w->hc_ffn_up, c, eps, npos, cslot);
	phase_t *p = prog_add(P, OP_SLAB_TR, 2560u * ldx / 1024u);
	ph_ptr(p, 0x140, Z); ph_ptr(p, 0x148, c->ZT); ph_u32(p, 0x150, 2560); ph_u32(p, 0x154, lgx);
	prog_nobar(P);   /* the router GEMM reads Z too */
	ph_gemm_ks(P, c, GG_BF16, 2560, w->ffn_gate_inp_x, Z, c->R, 513, npos);
	p = prog_add(P, OP_SLAB_ROUTER, ldx / 32u);
	ph_ptr(p, 0x140, c->R); ph_ptr(p, 0x148, c->IDX); ph_ptr(p, 0x150, c->RW); ph_ptr(p, 0x158, c->SW); ph_u32(p, 0x160, ldx);
	/* the routed units: the die's own (w), or (a test with fewer dies than the unit assignment) several dies' lists in turn */
	unsigned nv = c->nvu ? c->nvu : 1;
	for (unsigned v = 0; v < nv; v++) {
		CUdeviceptr ug = c->nvu ? c->vu_g[v] : w->ffn_gate_exps, uu = c->nvu ? c->vu_u[v] : w->ffn_up_exps;
		CUdeviceptr ud = c->nvu ? c->vu_d[v] : w->ffn_down_exps, em = c->nvu ? c->vu_emap[v] : w->emap;
		p = prog_add(P, OP_SLAB_GROUP, 1);
		ph_ptr(p, 0x140, c->IDX); ph_ptr(p, 0x148, c->RW); ph_ptr(p, 0x150, em); ph_ptr(p, 0x158, c->LP); ph_ptr(p, 0x160, c->LW);
		ph_ptr(p, 0x168, c->ITEMS); ph_ptr(p, 0x170, c->NITEMS); ph_u32(p, 0x178, npairs);
		if (getenv("CHUNK_GU1")) for (int gu = 0; gu < 2; gu++) {   /* (CHUNK_GU1) the units' gate rows, then their up rows */
			p = prog_add(P, OP_EBATCH_GU, 16);
			ph_ptr(p, 0x140, gu ? uu : ug); ph_ptr(p, 0x148, c->ZT); ph_ptr(p, 0x150, gu ? c->U : c->G);
			ph_ptr(p, 0x158, c->LP); ph_ptr(p, 0x160, c->ITEMS); ph_ptr(p, 0x168, c->NITEMS);
			if (!gu) prog_nobar(P);
		} else {   /* gate and up rows in one pass over the activations */
			p = prog_add(P, OP_EBATCH_GU2, 16);
			ph_ptr(p, 0x140, ug); ph_ptr(p, 0x148, c->ZT); ph_ptr(p, 0x150, c->G); ph_ptr(p, 0x158, c->LP); ph_ptr(p, 0x160, c->ITEMS);
			ph_ptr(p, 0x168, c->NITEMS); ph_ptr(p, 0x170, uu); ph_ptr(p, 0x178, c->U);
		}
		ph_elem(P, OP_SLAB_SWIGLU, c->G, c->U, c->H, npairs * 192u / 4u);
		if (!v) {
			prog_nobar(P);   /* the partial is not read by the activation */
			p = prog_add(P, OP_SLAB_ZERO, (2560u * ldx / 4u + 255u) / 256u);   /* PART: int32 fixed point (ebatch dn), value 2^20 */
			ph_ptr(p, 0x140, c->PART); ph_u32(p, 0x148, 2560u * ldx / 4u);
		}
		p = prog_add(P, OP_EBATCH_DN, 16);
		ph_ptr(p, 0x140, ud); ph_ptr(p, 0x148, c->H); ph_ptr(p, 0x150, c->PART); ph_ptr(p, 0x158, c->LP); ph_ptr(p, 0x160, c->LW);
		ph_ptr(p, 0x168, c->ITEMS); ph_ptr(p, 0x170, c->NITEMS); ph_u32(p, 0x178, ldx);
	}
	if (w->ffn_gate_shexp_s && s->shn) {   /* the shared expert split over the dies: this die's shn intermediate rows; its down columns' partial into PART */
		ph_gemm(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp_s, Z, c->SG, (unsigned)s->shn, npos, ldx);
		prog_nobar(P);
		ph_gemm(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp_s, Z, c->SU, (unsigned)s->shn, npos, ldx);
		ph_elem(P, OP_SLAB_SWIGLU, c->SG, c->SU, c->SH, (unsigned)s->shn * ldx / 4u);
		ph_gemm(P, w->ffn_down_shexp_type, s->shn, w->ffn_down_shexp_s, c->SH, c->SD, 2560, npos, ldx);
		p = prog_add(P, OP_SLAB_SAXPY, (2560u * ldx / 4u + 255u) / 256u);
		ph_ptr(p, 0x140, c->PART); ph_ptr(p, 0x148, c->SD); ph_ptr(p, 0x150, c->SW); ph_u32(p, 0x158, 2560u * ldx / 4u); ph_u32(p, 0x15c, lgx);
	} else if (s->shexp) {   /* the shared expert on its die: SD = down(SiLU(gate z) up z), PART += sigmoid(gate logit) SD */
		ph_gemm(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp, Z, c->SG, 640, npos, ldx);
		prog_nobar(P);
		ph_gemm(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, Z, c->SU, 640, npos, ldx);
		ph_elem(P, OP_SLAB_SWIGLU, c->SG, c->SU, c->SH, 640u * ldx / 4u);
		ph_gemm(P, w->ffn_down_shexp_type, 640, w->ffn_down_shexp, c->SH, c->SD, 2560, npos, ldx);
		p = prog_add(P, OP_SLAB_SAXPY, (2560u * ldx / 4u + 255u) / 256u);
		ph_ptr(p, 0x140, c->PART); ph_ptr(p, 0x148, c->SD); ph_ptr(p, 0x150, c->SW); ph_u32(p, 0x158, 2560u * ldx / 4u); ph_u32(p, 0x15c, lgx);
	}
	/* the reduce-scatter: every die's partial rows to their owners (a send), the owners' sums, gathered */
	unsigned slot = (unsigned)(P->xj & 1) * cslot, maxr = 0;   /* maxr: the largest owner's rows, the parts' row stride in the owner's slot */
	for (int e = 0; e < s->n; e++) { unsigned r = split_bound(2560, e + 1, s->n) - split_bound(2560, e, s->n); if (r > maxr) maxr = r; }
	if (row * maxr * (unsigned)s->n > cslot) { fprintf(stderr, "chunk reduce-scatter does not fit a slot\n"); exit(1); }
	p = prog_add(P, OP_SLAB_RS, 2560u * ldx / 4u / 256u);
	ph_ptr(p, 0x140, c->PART); ph_ptr(p, 0x148, x->xtab); ph_u32(p, 0x150, slot); ph_u32(p, 0x154, (unsigned)s->d); ph_u32(p, 0x158, maxr); ph_u32(p, 0x15c, lgx);
	for (int e = 1; e <= 5; e++) ph_u32(p, 0x160 + 4 * (e - 1), split_bound(2560, e, s->n));
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	unsigned n4 = (unsigned)s->yn * ldx / 4u;
	p = prog_add(P, OP_SLAB_SUM, (n4 + 255u) / 256u);
	ph_ptr(p, 0x140, x->inbox + slot); ph_ptr(p, 0x148, c->LY); ph_u32(p, 0x150, n4); ph_u32(p, 0x154, (unsigned)s->n); ph_u32(p, 0x158, maxr); ph_u32(p, 0x15c, lgx);
	CUdeviceptr Y = cxchg(P, x, s, c->LY, row * (unsigned)s->yn, row * (unsigned)s->y0, cslot);
	p = prog_add(P, OP_SLAB_WRITE, 10240u * ldx / 1024u);   /* S += 2 sigmoid(0.25 inj) Y, inj = T's rows 320..323 (this half's hc read) */
	ph_ptr(p, 0x140, c->S); ph_ptr(p, 0x148, Y); ph_ptr(p, 0x150, c->T); ph_u32(p, 0x158, lgx);
}

void prog_chunk_attn_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                          const chunk_attn_t *a, unsigned t0, unsigned npos, float eps, unsigned cslot) {
	unsigned ldx = c->ldx, lgx = lg2(ldx);
	unsigned long long row = 4ull * ldx;
	int owner = b->kq != 0;
	if (w->attn_q_rows % 512 || w->attn_q_first % 512) { fprintf(stderr, "chunk attention: the Q split is not head-aligned (rows %u from %u)\n", w->attn_q_rows, w->attn_q_first); exit(1); }
	unsigned nh = w->attn_q_rows / 512, h0 = w->attn_q_first / 512;
	if (t0 + npos > 32768) { fprintf(stderr, "chunk attention: positions past 32768\n"); exit(1); }
	CUdeviceptr Z = chunk_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, c, eps, npos, cslot);
	if (nh) { ph_gemm(P, w->attn_q_type, 2560, w->attn_q, Z, c->LQ, 512u * nh, npos, ldx); if (owner) prog_nobar(P); }
	CUdeviceptr CKV = c->CB, CKS = c->CB + 1024ull * npos, CSEL = c->CB + 1152ull * npos;
	if (owner) {   /* K, V and the indexer of z; then the cache append, the indexer, the selections */
		ph_gemm(P, w->attn_k_type, 2560, w->attn_k, Z, c->LK, 512, npos, ldx);
		prog_nobar(P);
		ph_gemm(P, w->attn_v_type, 2560, w->attn_v, Z, c->LV, 512, npos, ldx);
		prog_nobar(P);
		ph_gemm_ks(P, c, GG_BF16, 2560, w->indexer_x, Z, c->LI, 640, npos);
		phase_t *p = prog_add(P, OP_CATTN_KV, npos);
		ph_ptr(p, 0x140, c->LK); ph_ptr(p, 0x148, c->LV); ph_ptr(p, 0x150, w->attn_k_norm); ph_ptr(p, 0x158, c->CS); ph_ptr(p, 0x160, c->SN);
		ph_ptr(p, 0x168, b->kq); ph_ptr(p, 0x170, b->ks); ph_ptr(p, 0x178, b->vq); ph_ptr(p, 0x180, b->vs); ph_ptr(p, 0x188, CKV); ph_ptr(p, 0x190, CKS);
		ph_u32(p, 0x198, t0); ph_u32(p, 0x19c, lgx); ph_f32(p, 0x1a0, eps);
		prog_nobar(P);   /* the indexer reads LI only */
		p = prog_add(P, OP_CATTN_IDX, npos);
		ph_ptr(p, 0x140, c->LI); ph_ptr(p, 0x148, w->indexer_qn); ph_ptr(p, 0x150, w->indexer_kn); ph_ptr(p, 0x158, c->CS); ph_ptr(p, 0x160, c->SN);
		ph_ptr(p, 0x168, c->IQ); ph_ptr(p, 0x170, b->ring); ph_ptr(p, 0x178, b->pkt); ph_ptr(p, 0x180, a->POOL);
		ph_u32(p, 0x188, t0); ph_u32(p, 0x18c, npos); ph_u32(p, 0x190, lgx); ph_f32(p, 0x194, eps);
		if ((t0 + npos) / 4 > a->nblk) {   /* some position selects */
			p = prog_add(P, OP_CATTN_SEL, npos);
			ph_ptr(p, 0x140, c->IQ); ph_ptr(p, 0x148, a->POOL); ph_ptr(p, 0x150, CSEL); ph_u32(p, 0x158, t0); ph_u32(p, 0x15c, a->nblk);
		}
	}
	/* the owner's chunk K / V and selections to every die (Z stays in the other slot: this one is free) */
	CUdeviceptr IN = cxchg_raw(P, x, s, c->CB, owner ? CHUNK_CB_BYTES(npos) : 0, 0, cslot, 1);
	if (nh) {
		for (int hk = 0; hk < 2; hk++) {   /* the KV heads of this die's query heads */
			if (h0 + nh <= 12u * hk || h0 >= 12u * (hk + 1)) continue;
			if (!a->R[hk]) { fprintf(stderr, "chunk attention: no prompt cache for KV head %d\n", hk); exit(1); }
			phase_t *p = prog_add(P, OP_CATTN_APP, (npos + 3) / 4);
			ph_ptr(p, 0x140, IN); ph_ptr(p, 0x148, IN + 1024ull * npos); ph_ptr(p, 0x150, a->R[hk]); ph_ptr(p, 0x158, a->RS[hk]);
			ph_u32(p, 0x160, (unsigned)hk); ph_u32(p, 0x164, t0); ph_u32(p, 0x168, npos);
			prog_nobar(P);
		}
		P->p[P->n - 1].w[30] &= ~4u;   /* attention reads what the appends wrote */
		if (getenv("CHUNK_ATTN_PRE")) {   /* debugging: one more grid barrier (a phase that zeroes 16 B of the scratch) before the attention */
			phase_t *z = prog_add(P, OP_SLAB_ZERO, 1); ph_ptr(z, 0x140, c->Q6T); ph_u32(z, 0x148, 1);
		}
		/* debugging: CHUNK_ATTN_REV runs the head runs last to first, CHUNK_ATTN_TWICE runs the first run again after a barrier */
		unsigned rl[8], rn[8], nr = 0;
		for (unsigned lh = 0; lh < nh && nr < 8;) {   /* runs of at most 6 heads of one KV head */
			unsigned hk = (h0 + lh) / 12, len = 12u * (hk + 1) - (h0 + lh);
			if (len > nh - lh) len = nh - lh;
			if (len > 6) len = 6;
			rl[nr] = lh; rn[nr++] = len; lh += len;
		}
		int rev = getenv("CHUNK_ATTN_REV") != NULL, twice = getenv("CHUNK_ATTN_TWICE") != NULL;
		for (unsigned i = 0; i < nr + (twice ? 1u : 0u); i++) {
			unsigned j = i < nr ? (rev ? nr - 1 - i : i) : (rev ? nr - 1 : 0), lh = rl[j], len = rn[j], hk = (h0 + lh) / 12;
			phase_t *p = prog_add(P, OP_CATTN_ATTN, npos);
			ph_ptr(p, 0x140, c->LQ); ph_ptr(p, 0x148, w->attn_q_norm); ph_ptr(p, 0x150, c->CS); ph_ptr(p, 0x158, c->SN); ph_ptr(p, 0x160, a->R[hk]); ph_ptr(p, 0x168, a->RS[hk]);
			ph_ptr(p, 0x170, IN + 1152ull * npos); ph_ptr(p, 0x178, c->LAO); ph_u32(p, 0x180, lh); ph_u32(p, 0x184, len); ph_u32(p, 0x188, t0); ph_u32(p, 0x18c, a->nblk);
			ph_u32(p, 0x190, lgx); ph_f32(p, 0x194, eps);
			if (i + 1 < nr && !getenv("CHUNK_ATTN_BAR")) prog_nobar(P);   /* CHUNK_ATTN_BAR (debugging): a grid barrier between the runs */
		}
	}
	CUdeviceptr AO = cxchg(P, x, s, c->LAO, row * 256u * nh, row * 256u * h0, cslot);
	ph_gemm_k6144(P, w->attn_output_type, w->attn_output, AO, c->LY, (unsigned)s->yn, npos, ldx, c->Q6T);
	CUdeviceptr Y = cxchg(P, x, s, c->LY, row * (unsigned)s->yn, row * (unsigned)s->y0, cslot);
	phase_t *p = prog_add(P, OP_SLAB_WRITE, 10240u * ldx / 1024u);   /* S += 2 sigmoid(0.25 inj) Y */
	ph_ptr(p, 0x140, c->S); ph_ptr(p, 0x148, Y); ph_ptr(p, 0x150, c->T); ph_u32(p, 0x158, lgx);
}

void prog_chunk_ple_prefix(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const chunk_bufs_t *c,
                           CUdeviceptr RIN, CUdeviceptr ROUT, unsigned t0, unsigned npos, float eps, unsigned cslot) {
	unsigned ldx = c->ldx, lgx = lg2(ldx), pk0 = (unsigned)s->pk0, pkn = (unsigned)s->pkn;
	unsigned long long row = 4ull * ldx;
	CUdeviceptr KN = c->XH, QN = c->PL1, G = c->PL2, PX = c->XH;
	ph_gemm(P, GG_Q2_0, 2560, w->ple_key, c->PG, c->LPK, pkn, npos, ldx);
	/* K gathered in two halves of 5120 rows (a slot holds 6144); each half normalised before the next exchange is sent (every
	   die has read a slot before any die sends into it again) */
	for (unsigned half = 0; half < 2; half++) {
		unsigned lo = 5120u * half, r0 = pk0 > lo ? pk0 : lo, r1 = pk0 + pkn < lo + 5120u ? pk0 + pkn : lo + 5120u, n = r1 > r0 ? r1 - r0 : 0;
		CUdeviceptr KH = cxchg(P, x, s, c->LPK + (n ? (r0 - pk0) * row : 0), n * row, (n ? r0 - lo : 0) * row, cslot);
		ph_slab_rms(P, KH, w->ple_norm_key + 4ull * lo, KN + lo * row, 0, 2, 2560, 2560, eps, ldx);
	}
	prog_nobar(P);   /* the query norm reads S only */
	ph_slab_rms(P, c->S, w->ple_norm_query, QN, 0, 4, 2560, 2560, eps, ldx);
	prog_nobar(P);
	ph_gemm_ks(P, c, GG_BF16, 2560, w->ple_value, c->PG, c->LY, (unsigned)s->yn, npos);
	CUdeviceptr V = cxchg(P, x, s, c->LY, row * (unsigned)s->yn, row * (unsigned)s->y0, cslot);
	phase_t *p = prog_add(P, OP_SLAB_PGATE, 4u * ldx / 32u);
	ph_ptr(p, 0x140, KN); ph_ptr(p, 0x148, QN); ph_ptr(p, 0x150, V); ph_ptr(p, 0x158, G); ph_u32(p, 0x160, lgx); ph_u32(p, 0x164, lgx - 5);
	ph_slab_rms(P, G, w->ple_norm_conv, PX, 0, 4, 2560, 2560, eps, ldx);
	p = prog_add(P, OP_SLAB_PCONV, 40u * ldx);
	ph_ptr(p, 0x140, PX); ph_ptr(p, 0x148, RIN); ph_ptr(p, 0x150, ROUT); ph_ptr(p, 0x158, w->ple_conv1d); ph_ptr(p, 0x160, G); ph_ptr(p, 0x168, c->S);
	ph_u32(p, 0x170, t0); ph_u32(p, 0x174, t0 % 10u); ph_u32(p, 0x178, npos); ph_u32(p, 0x17c, lgx);
}
