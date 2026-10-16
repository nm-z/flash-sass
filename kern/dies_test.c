/* dies_test: one Gated DeltaNet block split over n dies (expert split + row split),
 * each die running its own persistent-kernel program with the exchange ops, against
 * the C block twin, bit for bit, over T tokens with the states carried.
 *
 *   dies_test <fullcache> <blk> <T> <manifest0> [manifest1 ...]
 *
 * fullcache: a holder's tile cache of the whole block (hold ... cache=DIR without
 * split), read by the twin; manifest_i: the running holder of die i's slice
 * (hold ... split=i/n). n = number of manifests (1..6). Env: DIES_NPH caps the
 * program (a prefix on every die), DIES_PROF prints die 0's per-phase times. */
#define _POSIX_C_SOURCE 200809L
#include "block.h"
#include "hold.h"
#include "prog.h"
#include "gguf.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
static uint32_t rs = 909;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#define MAXD 6
static int nd, blk;
static const char *fullcache;
static CUcontext ctx[MAXD];
static hold_arena ar[MAXD];
static char tname[128];
static const char *tn(const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static void *hostimg(const char *suffix) {
	char fn[512]; snprintf(fn, sizeof fn, "%s/%s", fullcache, tn(suffix));
	FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	void *p = malloc((size_t)n);
	if (fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", fn); exit(1); }
	fclose(f);
	return p;
}
static CUdeviceptr dptr(int d, const char *suffix) { CUdeviceptr p = hold_ptr(&ar[d], tn(suffix)); if (!p) { fprintf(stderr, "die %d: no %s\n", d, tname); exit(1); } return p; }
static uint32_t ttype(int d, const char *suffix) { const hold_tensor *t = hold_find(&ar[d], tn(suffix)); if (!t) { fprintf(stderr, "no %s\n", tname); exit(1); } return t->type; }
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }

static int cmpbuf(const char *what, const void *ref, CUdeviceptr dev, size_t bytes, int elemsz) {
	void *got = malloc(bytes);
	CK(cuMemcpyDtoH(got, dev, bytes));
	size_t n = bytes / elemsz; int bad = 0; size_t first = 0;
	for (size_t i = 0; i < n; i++) if (memcmp((const char *)ref + i * elemsz, (const char *)got + i * elemsz, elemsz)) { if (!bad) first = i; bad++; }
	if (bad) {
		printf("  %-10s MISMATCH %d of %zu (first at %zu", what, bad, n, first);
		if (elemsz == 4) printf(": gpu %.9g ref %.9g", ((float *)got)[first], ((const float *)ref)[first]);
		printf(")\n");
	if (bad && getenv("DIES_HIST")) { int per = atoi(getenv("DIES_HIST")); printf("   per-%d:", per); for (size_t c = 0; c < (n + per - 1) / per; c++) { int k = 0; for (size_t i = c * per; i < n && i < (c + 1) * per; i++) if (memcmp((const char *)ref + i * elemsz, (const char *)got + i * elemsz, elemsz)) k++; if (k) printf(" %zu:%d", c, k); } printf("\n"); }
	} else printf("  %-10s exact (%zu)\n", what, n);
	free(got);
	return bad;
}

int main(int argc, char **argv) {
	if (argc < 5) { fprintf(stderr, "usage: dies_test <fullcache> <blk> <T> <manifest0> [manifest1 ...]\n"); return 2; }
	fullcache = argv[1]; blk = atoi(argv[2]); int T = atoi(argv[3]); nd = argc - 4; float eps = 1e-6f;
	if (nd > MAXD) { fprintf(stderr, "at most %d dies\n", MAXD); return 2; }
	CK(cuInit(0));
	CUmodule mod[MAXD]; CUfunction fn[MAXD];
	int dev_of[MAXD];
	for (int d = 0; d < nd; d++) {
		FILE *f = fopen(argv[4 + d], "r"); size_t bytes; int die;
		if (!f || fscanf(f, "arena %zu die %d", &bytes, &die) != 2) { fprintf(stderr, "bad manifest %s\n", argv[4 + d]); return 1; }
		fclose(f);
		dev_of[d] = die;
		CUdevice dev; CK(cuDeviceGet(&dev, die));
		CK(cuCtxCreate_v2(&ctx[d], CU_CTX_SCHED_SPIN, dev));
		if (hold_open(&ar[d], argv[4 + d])) return 1;
		CK(cuModuleLoad(&mod[d], "die.cubin")); CK(cuModuleGetFunction(&fn[d], mod[d], "die"));
	}
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int e = 0; e < nd; e++) if (e != d) CK(cuCtxEnablePeerAccess(ctx[e], 0));
	}
	/* twin: whole-block weights from the full tile cache */
	dn_hweights_t hw; memset(&hw, 0, sizeof hw);
	hw.hc_attn_norm = hostimg("hc_attn_norm.weight"); hw.hc_attn_dj = hostimg("hc_attn_dj.weight"); hw.hc_attn_up = hostimg("hc_attn_up.weight");
	hw.hc_ffn_norm = hostimg("hc_ffn_norm.weight"); hw.hc_ffn_dj = hostimg("hc_ffn_dj.weight"); hw.hc_ffn_up = hostimg("hc_ffn_up.weight");
	hw.attn_qkv = hostimg("attn_qkv.weight"); hw.attn_qkv_type = ttype(0, "attn_qkv.weight");
	hw.attn_gate = hostimg("attn_gate.weight"); hw.attn_gate_type = ttype(0, "attn_gate.weight");
	hw.ssm_out = hostimg("ssm_out.weight"); hw.ssm_out_type = ttype(0, "ssm_out.weight");
	hw.ssm_ab = hostimg("ssm_ab.weight"); hw.ssm_ab_bias = hostimg("ssm_ab.bias"); hw.ssm_conv1d = hostimg("ssm_conv1d.weight");
	hw.ssm_a = hostimg("ssm_a"); hw.ssm_norm = hostimg("ssm_norm.weight");
	hw.ffn_gate_inp_x = hostimg("ffn_gate_inp_x.weight");
	hw.ffn_gate_exps = hostimg("ffn_gate_exps.weight"); hw.ffn_up_exps = hostimg("ffn_up_exps.weight"); hw.ffn_down_exps = hostimg("ffn_down_exps.weight");
	hw.ffn_gate_shexp = hostimg("ffn_gate_shexp.weight"); hw.ffn_gate_shexp_type = ttype(0, "ffn_gate_shexp.weight");
	hw.ffn_up_shexp = hostimg("ffn_up_shexp.weight"); hw.ffn_up_shexp_type = ttype(0, "ffn_up_shexp.weight");
	hw.ffn_down_shexp = hostimg("ffn_down_shexp.weight"); hw.ffn_down_shexp_type = ttype(0, "ffn_down_shexp.weight");
	dn_hbufs_t *hb = calloc(1, sizeof *hb); hb->dn_state = calloc(48 * 16384, 4);
	for (int i = 0; i < 10240; i++) hb->S[i] = 0.5f * frand();
	/* per die: weights, buffers, exchange state */
	dn_weights_t dw[MAXD]; dn_bufs_t db[MAXD]; xchg_t xc[MAXD]; split_t sp[MAXD];
	CUdeviceptr dPROG[MAXD], dCTL[MAXD];
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		split_of(&sp[d], d, nd, blk);
		const split_t *s = &sp[d];
		dn_weights_t *w = &dw[d]; memset(w, 0, sizeof *w);
		w->hc_attn_norm = dptr(d, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, "hc_attn_up.weight");
		w->hc_ffn_norm = dptr(d, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, "hc_ffn_up.weight");
		w->attn_qkv = dptr(d, "attn_qkv.weight"); w->attn_qkv_type = hw.attn_qkv_type;
		w->attn_gate = dptr(d, "attn_gate.weight"); w->attn_gate_type = hw.attn_gate_type;
		w->ssm_out = dptr(d, "ssm_out.weight"); w->ssm_out_type = hw.ssm_out_type;
		w->ssm_ab = dptr(d, "ssm_ab.weight"); w->ssm_ab_bias = dptr(d, "ssm_ab.bias");
		w->ssm_conv1d = dptr(d, "ssm_conv1d.weight"); w->ssm_a = dptr(d, "ssm_a"); w->ssm_norm = dptr(d, "ssm_norm.weight");
		w->ffn_gate_inp_x = dptr(d, "ffn_gate_inp_x.weight"); w->emap = dptr(d, "emap");
		w->ffn_gate_exps = dptr(d, "ffn_gate_exps.weight"); w->ffn_up_exps = dptr(d, "ffn_up_exps.weight"); w->ffn_down_exps = dptr(d, "ffn_down_exps.weight");
		w->ffn_gate_shexp = dptr(d, "ffn_gate_shexp.weight"); w->ffn_gate_shexp_type = hw.ffn_gate_shexp_type;
		w->ffn_up_shexp = dptr(d, "ffn_up_shexp.weight"); w->ffn_up_shexp_type = hw.ffn_up_shexp_type;
		w->ffn_down_shexp = dptr(d, "ffn_down_shexp.weight"); w->ffn_down_shexp_type = hw.ffn_down_shexp_type;
		unsigned nloc = SPLIT_NLOC(s), cols = 32u * (unsigned)s->cbn, hrows = 128u * (unsigned)s->hn;
		dn_bufs_t *b = &db[d];
		b->S = dalloc(4 * 10240); b->xh = dalloc(4 * 10240); b->t = dalloc(4 * 324);
		b->ldown = dalloc(4u * (unsigned)s->djn); b->lup = dalloc(4 * 4 * cols); b->lz = dalloc(4 * cols);
		b->zq = dalloc(2 * 2560); b->zs32 = dalloc(4 * 80); b->zs16 = dalloc(4 * 160); b->zas = dalloc(4 * 80);
		b->lqkv = dalloc(4 * nloc); b->lgate = dalloc(4 * hrows); b->la = dalloc(4 * 2u * (unsigned)s->hn); b->lu = dalloc(4 * nloc); b->lo = dalloc(4 * hrows); b->lon = dalloc(4 * hrows);
		b->oq = dalloc(2 * 6144); b->os32 = dalloc(4 * 192); b->os16 = dalloc(4 * 384); b->oas = dalloc(4 * 192);
		b->ly = dalloc(4u * (unsigned)s->yn); b->r = dalloc(4 * 513); b->idx = dalloc(64); b->w = dalloc(64); b->s = dalloc(16);
		b->lidx = dalloc(64); b->lw = dalloc(64); b->ne = dalloc(16); b->part = dalloc(4 * 2560);
		b->g = dalloc(4 * 7040); b->gu = dalloc(4 * 7040); b->hq = dalloc(2 * 7040); b->hs32 = dalloc(4 * 220); b->hs16 = dalloc(4 * 440); b->has = dalloc(4 * 220);
		b->out = dalloc(4 * 2560); b->conv_state = dalloc(4 * 4 * nloc); b->dn_state = dalloc((size_t)s->hn * 16384 * 4);
		CK(cuMemcpyHtoD(b->S, hb->S, 4 * 10240));
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * XSLOT); x->flags = dalloc(32); x->xgen = dalloc(16); x->xtab = dalloc(128);
		uint32_t fl[6]; for (int e = 0; e < 6; e++) fl[e] = e < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		dPROG[d] = dalloc(128 * 512); dCTL[d] = dalloc(0x40 + 4 * 512);
	}
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < nd; d++) { tab[d] = (uint64_t)xc[d].inbox; tab[6 + d] = (uint64_t)xc[d].flags; }
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab)); }
	printf("%d die(s):", nd);
	for (int d = 0; d < nd; d++) printf(" [%d: gpu %d dj %d+%d cb %d+%d h %d+%d k %d..%d y %d+%d e %d+%d%s]", d, dev_of[d], sp[d].dj0, sp[d].djn, sp[d].cb0, sp[d].cbn, sp[d].h0, sp[d].hn, sp[d].klo, sp[d].khi, sp[d].y0, sp[d].yn, sp[d].e0, sp[d].en, sp[d].shexp ? " shexp" : "");
	printf("\n");
	int fails = 0;
	for (int t = 0; t < T; t++) {
		double t0 = now();
		deltanet_block_gpu(&hw, hb, (unsigned)t, eps, nd, blk, getenv("DIES_STAGE") ? atoi(getenv("DIES_STAGE")) : 0);
		double t1 = now();
		prog_t P[MAXD]; unsigned nph[MAXD];
		for (int d = 0; d < nd; d++) {
			prog_init(&P[d]);
			prog_deltanet_block(&P[d], &sp[d], &xc[d], &dw[d], &db[d], (unsigned)t, eps);
			nph[d] = (unsigned)P[d].n;
			if (getenv("DIES_NPH") && (unsigned)atoi(getenv("DIES_NPH")) < nph[d]) nph[d] = (unsigned)atoi(getenv("DIES_NPH"));
			CK(cuCtxSetCurrent(ctx[d]));
			CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
		}
		double t2 = now();
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			void *params[] = { &dPROG[d], &nph[d], &dCTL[d] };
			CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		}
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
		double t3 = now();
		printf("token %d: %u phases; twin %.2f s, dies %.3f ms\n", t, nph[0], t1 - t0, (t3 - t2) * 1e3);
		int bad = 0;
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			char what[32];
			printf(" die %d\n", d);
			const split_t *s = &sp[d];
			bad += cmpbuf("xh", hb->xh, db[d].xh, 4 * 10240, 4);
			bad += cmpbuf("ldown", hb->down + s->dj0, db[d].ldown, 4 * s->djn, 4);
			if (getenv("DIES_DUMP")) {   /* the two inbox slots against the twin's down */
				float sl[8]; CK(cuMemcpyDtoH(sl, xc[d].inbox, 16)); CK(cuMemcpyDtoH(sl + 4, xc[d].inbox + XSLOT, 16));
				printf("  slot0 %.9g %.9g %.9g %.9g  slot1 %.9g %.9g %.9g %.9g  down %.9g %.9g %.9g %.9g\n", sl[0], sl[1], sl[2], sl[3], sl[4], sl[5], sl[6], sl[7], hb->down[0], hb->down[1], hb->down[2], hb->down[3]);
				float tt[8], ib[8], lu[12];
				CK(cuMemcpyDtoH(tt, db[d].t + 316 * 4, 32)); CK(cuMemcpyDtoH(ib, xc[d].inbox + 316 * 4, 32)); CK(cuMemcpyDtoH(lu, db[d].lup + 36 * 4, 48));
				printf("  t[316..324) gpu"); for (int i = 0; i < 8; i++) printf(" %.6g", tt[i]); printf("  twin"); for (int i = 0; i < 8; i++) printf(" %.6g", hb->t[316 + i]); printf("\n");
				printf("  slot0[316..324)"); for (int i = 0; i < 8; i++) printf(" %.6g", ib[i]); printf("  twin down"); for (int i = 0; i < 8; i++) printf(" %.6g", hb->down[316 + i]); printf("\n");
				printf("  lup[36..48) gpu"); for (int i = 0; i < 12; i++) printf(" %.6g", lu[i]); printf("  twin"); for (int i = 0; i < 12; i++) printf(" %.6g", hb->up[36 + i]); printf("\n");
			}
			snprintf(what, sizeof what, "t"); bad += cmpbuf(what, hb->t, db[d].t, 4 * 324, 4);
			for (int l = 0; l < 4; l++) { snprintf(what, sizeof what, "lup%d", l); bad += cmpbuf(what, hb->up + 2560 * l + 32 * s->cb0, db[d].lup + 4u * 32u * (unsigned)(s->cbn * l), 4u * 32u * (unsigned)s->cbn, 4); }
			bad += cmpbuf("lz", hb->z + 32 * s->cb0, db[d].lz, 4u * 32u * (unsigned)s->cbn, 4);
			bad += cmpbuf("zq", hb->zq, db[d].zq, 2 * 2560, 2);
			{   /* attention-side locals (not overwritten by the FFN half) */
				unsigned nk = (unsigned)s->nk, hn = (unsigned)s->hn;
				bad += cmpbuf("lqkv.q", hb->qkv + 128 * s->klo, db[d].lqkv, 4 * 128 * nk, 4);
				bad += cmpbuf("lqkv.k", hb->qkv + 2048 + 128 * s->klo, db[d].lqkv + 4 * 128 * nk, 4 * 128 * nk, 4);
				bad += cmpbuf("lqkv.v", hb->qkv + 4096 + 128 * s->h0, db[d].lqkv + 4 * 256 * nk, 4 * 128 * hn, 4);
				bad += cmpbuf("lgate", hb->gate + 128 * s->h0, db[d].lgate, 4 * 128 * hn, 4);
				bad += cmpbuf("la", hb->a + s->h0, db[d].la, 4 * hn, 4);
				bad += cmpbuf("lb", hb->a + 48 + s->h0, db[d].la + 4 * hn, 4 * hn, 4);
				bad += cmpbuf("lu.q", hb->u + 128 * s->klo, db[d].lu, 4 * 128 * nk, 4);
				bad += cmpbuf("lu.v", hb->u + 4096 + 128 * s->h0, db[d].lu + 4 * 256 * nk, 4 * 128 * hn, 4);
				bad += cmpbuf("lo", hb->o + 128 * s->h0, db[d].lo, 4 * 128 * hn, 4);
				bad += cmpbuf("lon", hb->on + 128 * s->h0, db[d].lon, 4 * 128 * hn, 4);
				bad += cmpbuf("oq", hb->oq, db[d].oq, 2 * 6144, 2);
				bad += cmpbuf("ly", hb->y + s->y0, db[d].ly, 4u * (unsigned)s->yn, 4);
			}
			bad += cmpbuf("r", hb->r, db[d].r, 4 * 513, 4);
			bad += cmpbuf("idx", hb->idx, db[d].idx, 40, 4);
			bad += cmpbuf("w", hb->w, db[d].w, 40, 4);
			bad += cmpbuf("part", hb->part[d], db[d].part, 4 * 2560, 4);
			if (getenv("DIES_DUMP")) for (int sdie = 0; sdie < nd; sdie++) { char nm[16]; snprintf(nm, sizeof nm, "parts.%d", sdie); bad += cmpbuf(nm, hb->part[sdie], xc[d].inbox + 10240u * (unsigned)sdie, 4 * 2560, 4); }   /* the gathered copies (slot 0 at the end) */
			bad += cmpbuf("out", hb->out, db[d].out, 4 * 2560, 4);
			bad += cmpbuf("S", hb->S, db[d].S, 4 * 10240, 4);
		}
		if (bad) fails++;
		if (t == T - 1 && getenv("DIES_PROF")) {
			CK(cuCtxSetCurrent(ctx[0]));
			uint32_t *ts = malloc(4 * (16 + nph[0])); CK(cuMemcpyDtoH(ts, dCTL[0], 4 * (16 + nph[0])));
			unsigned byop[DIE_NOPS] = {0}; unsigned prev = ts[15];
			for (unsigned i = 0; i < nph[0]; i++) {
				unsigned dd = ts[16 + i] - prev; prev = ts[16 + i];
				byop[P[0].p[i].w[0]] += dd;
				printf("   %2u %-10s units %5u %7.1f\n", i, die_opname[P[0].p[i].w[0]], P[0].p[i].w[1], dd / 1e3);
			}
			printf("  by op:");
			for (int o = 0; o < DIE_NOPS; o++) if (byop[o]) printf(" %s %.0f", die_opname[o], byop[o] / 1e3);
			printf("\n  total %.1f us\n", (ts[16 + nph[0] - 1] - ts[15]) / 1e3);
			free(ts);
		}
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); hold_close(&ar[d]); }
	return fails ? 3 : 0;
}
