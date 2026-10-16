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
#include "attn.h"
#include "ple.h"
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
/* IEEE half -> float (exact) */
static float h2f(uint16_t h) {
	uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
	if (e == 0) { if (!m) u = s; else { int sh = 0; while (!(m & 0x400)) { m <<= 1; sh++; } m &= 0x3ff; u = s | ((uint32_t)(113 - sh) << 23) | (m << 13); } }
	else if (e == 31) u = s | 0x7f800000u | (m << 13);
	else u = s | ((e + 112) << 23) | (m << 13);
	float f; memcpy(&f, &u, 4); return f;
}

#define MAXD 6
static int nd, blk;
static const char *fullcache;
static CUcontext ctx[MAXD];
static hold_arena ar[MAXD];
static char tname[128];
static const char *tn(const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static void *hostimg_raw(const char *name) {
	char fn[512]; snprintf(fn, sizeof fn, "%s/%s", fullcache, name);
	FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	void *p = malloc((size_t)n);
	if (fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", fn); exit(1); }
	fclose(f);
	return p;
}
static void *hostimg(const char *suffix) { return hostimg_raw(tn(suffix)); }
static CUdeviceptr dptr_raw(int d, const char *name) { CUdeviceptr p = hold_ptr(&ar[d], name); if (!p) { fprintf(stderr, "die %d: no %s\n", d, name); exit(1); } return p; }
static CUdeviceptr dptr(int d, const char *suffix) { return dptr_raw(d, tn(suffix)); }
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
	if (argc < 5) { fprintf(stderr, "usage: dies_test <fullcache> <blk|head|embed> <T> <manifest0> [manifest1 ...]\n"); return 2; }
	fullcache = argv[1]; const char *whatarg = argv[2]; int T = atoi(argv[3]); nd = argc - 4; float eps = 1e-6f;
	int mode = !strcmp(whatarg, "head") ? 1 : !strcmp(whatarg, "embed") ? 2 : 0;   /* 0: a block */
	blk = mode ? 0 : atoi(whatarg);
	if (nd > MAXD) { fprintf(stderr, "at most %d dies\n", MAXD); return 2; }
	int is_attn = mode == 0 && BLK_IS_ATTN(blk), has_ple = mode == 0 && blk == 1;
	unsigned npos = ((unsigned)T + 8192 + 64 + 31) / 32 * 32;   /* cache positions: chunk round-up plus the loops' stray prefetches */
	size_t kcf = ATTN_KC_FLOATS(npos), vcf = (size_t)npos * 512;
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
	if (mode == 1) { hw.output_hc_norm = hostimg_raw("output_hc_norm.weight"); hw.output_hc_down = hostimg_raw("output_hc_down.weight"); hw.output_hc_up = hostimg_raw("output_hc_up.weight"); hw.output = hostimg_raw("output.weight"); }
	else if (mode == 2) hw.token_embd = hostimg_raw("token_embd.weight");
	if (mode == 0) {
	hw.hc_attn_norm = hostimg("hc_attn_norm.weight"); hw.hc_attn_dj = hostimg("hc_attn_dj.weight"); hw.hc_attn_up = hostimg("hc_attn_up.weight");
	hw.hc_ffn_norm = hostimg("hc_ffn_norm.weight"); hw.hc_ffn_dj = hostimg("hc_ffn_dj.weight"); hw.hc_ffn_up = hostimg("hc_ffn_up.weight");
	if (!is_attn) {
		hw.attn_qkv = hostimg("attn_qkv.weight"); hw.attn_qkv_type = ttype(0, "attn_qkv.weight");
		hw.attn_gate = hostimg("attn_gate.weight"); hw.attn_gate_type = ttype(0, "attn_gate.weight");
		hw.ssm_out = hostimg("ssm_out.weight"); hw.ssm_out_type = ttype(0, "ssm_out.weight");
		hw.ssm_ab = hostimg("ssm_ab.weight"); hw.ssm_ab_bias = hostimg("ssm_ab.bias"); hw.ssm_conv1d = hostimg("ssm_conv1d.weight");
		hw.ssm_a = hostimg("ssm_a"); hw.ssm_norm = hostimg("ssm_norm.weight");
	} else {
		hw.attn_q = hostimg("attn_q.weight"); hw.attn_q_type = ttype(0, "attn_q.weight");
		hw.attn_k = hostimg("attn_k.weight"); hw.attn_k_type = ttype(0, "attn_k.weight");
		hw.attn_v = hostimg("attn_v.weight"); hw.attn_v_type = ttype(0, "attn_v.weight");
		hw.attn_output = hostimg("attn_output.weight"); hw.attn_output_type = ttype(0, "attn_output.weight");
		hw.attn_q_norm = hostimg("attn_q_norm.weight"); hw.attn_k_norm = hostimg("attn_k_norm.weight");
	}
	hw.ffn_gate_inp_x = hostimg("ffn_gate_inp_x.weight");
	hw.ffn_gate_exps = hostimg("ffn_gate_exps.weight"); hw.ffn_up_exps = hostimg("ffn_up_exps.weight"); hw.ffn_down_exps = hostimg("ffn_down_exps.weight");
	hw.ffn_gate_shexp = hostimg("ffn_gate_shexp.weight"); hw.ffn_gate_shexp_type = ttype(0, "ffn_gate_shexp.weight");
	hw.ffn_up_shexp = hostimg("ffn_up_shexp.weight"); hw.ffn_up_shexp_type = ttype(0, "ffn_up_shexp.weight");
	hw.ffn_down_shexp = hostimg("ffn_down_shexp.weight"); hw.ffn_down_shexp_type = ttype(0, "ffn_down_shexp.weight");
	}
	dn_hbufs_t *hb = calloc(1, sizeof *hb); hb->dn_state = calloc(48 * 16384, 4);
	if (is_attn) { hb->kc = calloc(kcf, 4); hb->vc = calloc(vcf, 4); }
	if (mode == 1) hb->logits = malloc(4 * 248320);
	/* PLE: a synthetic n-gram table (the real one is 29 GB) in host-mapped memory, scaled-down
	 * head offsets / vocabularies, the model's multipliers; a token sequence with ends planted */
	const size_t PLE_R = 1u << 20; uint8_t *ptab = NULL; uint32_t *ids = NULL; uint8_t ppc[160]; float *pconv = NULL;
	if (has_ple) {
		hw.ple_key = hostimg("ple_key.weight"); hw.ple_value = hostimg("ple_value.weight");
		hw.ple_norm_key = hostimg("ple_norm_key.weight"); hw.ple_norm_query = hostimg("ple_norm_query.weight"); hw.ple_norm_conv = hostimg("ple_norm_conv.weight");
		const uint16_t *cw = hostimg("ple_conv1d.weight");
		pconv = malloc(4 * 10240 * 4);
		for (int i = 0; i < 10240 * 4; i++) pconv[i] = h2f(cw[i]);
		hw.ple_conv_w = pconv;
		CK(cuMemHostAlloc((void **)&ptab, PLE_R * 90, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
		for (size_t r = 0; r < PLE_R; r++)
			for (int k = 0; k < 5; k++) {   /* 5 blocks of fp16 scale + 16 nibble bytes */
				uint16_t dd = (uint16_t)(0x3000 + (rs >> 16) % 0x1000); frand();   /* a sane fp16 scale */
				memcpy(ptab + r * 90 + 18 * k, &dd, 2);
				for (int j = 2; j < 18; j++) { ptab[r * 90 + 18 * k + j] = (uint8_t)(rs >> 24); frand(); }
			}
		hw.ple_tab = ptab;
		memcpy(hw.ple_mult, ple_multipliers, sizeof hw.ple_mult);
		for (int h = 0; h < 16; h++) { hw.ple_off[h] = (uint64_t)h * 65000; hw.ple_voc[h] = 60000 + 13 * h; }
		hw.ple_absent = 248044; hw.ple_end = 248046;
		memset(ppc, 0, sizeof ppc); memcpy(ppc, hw.ple_mult, 24);
		for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)hw.ple_off[h], v32 = (uint32_t)hw.ple_voc[h]; memcpy(ppc + 24 + 4 * h, &o32, 4); memcpy(ppc + 88 + 4 * h, &v32, 4); }
		memcpy(ppc + 152, &hw.ple_absent, 4); memcpy(ppc + 156, &hw.ple_end, 4);
		ids = malloc(4 * (size_t)(T + 1));
		for (int i = 0; i <= T; i++) { ids[i] = (uint32_t)((rs >> 8) % 248000); frand(); }
		if (T > 5) ids[5] = hw.ple_end;
		hb->ids = ids;
	}
	for (int i = 0; i < 10240; i++) hb->S[i] = 0.5f * frand();
	/* RoPE tables for positions [0, T) as the twin computes them */
	float *rcs = malloc(4 * 32 * T), *rsn = malloc(4 * 32 * T);
	for (int t = 0; t < T; t++) rope_table_gpu((unsigned)t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	/* per die: weights, buffers, exchange state */
	dn_weights_t dw[MAXD]; dn_bufs_t db[MAXD]; xchg_t xc[MAXD]; split_t sp[MAXD];
	CUdeviceptr dPROG[MAXD], dCTL[MAXD];
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		split_of(&sp[d], d, nd, blk);
		const split_t *s = &sp[d];
		dn_weights_t *w = &dw[d]; memset(w, 0, sizeof *w);
		if (mode == 1) {
			w->output_hc_norm = dptr_raw(d, "output_hc_norm.weight"); w->output_hc_down = dptr_raw(d, "output_hc_down.weight"); w->output_hc_up = dptr_raw(d, "output_hc_up.weight"); w->output = dptr_raw(d, "output.weight");
			uint32_t ob[6]; for (int e = 0; e < 6; e++) ob[e] = 248320u * (unsigned)e / (unsigned)nd;
			w->obases = dalloc(24); CK(cuMemcpyHtoD(w->obases, ob, 24));
		} else if (mode == 2) w->token_embd = dptr_raw(d, "token_embd.weight");
		if (mode == 0) {
		w->hc_attn_norm = dptr(d, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, "hc_attn_up.weight");
		w->hc_ffn_norm = dptr(d, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, "hc_ffn_up.weight");
		if (!is_attn) {
			w->attn_qkv = dptr(d, "attn_qkv.weight"); w->attn_qkv_type = hw.attn_qkv_type;
			w->attn_gate = dptr(d, "attn_gate.weight"); w->attn_gate_type = hw.attn_gate_type;
			w->ssm_out = dptr(d, "ssm_out.weight"); w->ssm_out_type = hw.ssm_out_type;
			w->ssm_ab = dptr(d, "ssm_ab.weight"); w->ssm_ab_bias = dptr(d, "ssm_ab.bias");
			w->ssm_conv1d = dptr(d, "ssm_conv1d.weight"); w->ssm_a = dptr(d, "ssm_a"); w->ssm_norm = dptr(d, "ssm_norm.weight");
		} else {
			w->attn_q = dptr(d, "attn_q.weight"); w->attn_q_type = hw.attn_q_type;
			w->attn_k = dptr(d, "attn_k.weight"); w->attn_k_type = hw.attn_k_type;
			w->attn_v = dptr(d, "attn_v.weight"); w->attn_v_type = hw.attn_v_type;
			w->attn_output = dptr(d, "attn_output.weight"); w->attn_output_type = hw.attn_output_type;
			w->attn_q_norm = dptr(d, "attn_q_norm.weight"); w->attn_k_norm = dptr(d, "attn_k_norm.weight");
		}
		w->ffn_gate_inp_x = dptr(d, "ffn_gate_inp_x.weight"); w->emap = dptr(d, "emap");
		w->ffn_gate_exps = dptr(d, "ffn_gate_exps.weight"); w->ffn_up_exps = dptr(d, "ffn_up_exps.weight"); w->ffn_down_exps = dptr(d, "ffn_down_exps.weight");
		w->ffn_gate_shexp = dptr(d, "ffn_gate_shexp.weight"); w->ffn_gate_shexp_type = hw.ffn_gate_shexp_type;
		w->ffn_up_shexp = dptr(d, "ffn_up_shexp.weight"); w->ffn_up_shexp_type = hw.ffn_up_shexp_type;
		w->ffn_down_shexp = dptr(d, "ffn_down_shexp.weight"); w->ffn_down_shexp_type = hw.ffn_down_shexp_type;
		}
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
		if (has_ple) {
			w->ple_key = dptr(d, "ple_key.weight"); w->ple_value = dptr(d, "ple_value.weight");
			w->ple_norm_key = dptr(d, "ple_norm_key.weight"); w->ple_norm_query = dptr(d, "ple_norm_query.weight"); w->ple_norm_conv = dptr(d, "ple_norm_conv.weight");
			w->ple_conv1d = dptr(d, "ple_conv1d.weight");
			b->ids = dalloc(4 * (size_t)(T + 1)); CK(cuMemcpyHtoD(b->ids, ids, 4 * (size_t)(T + 1)));
			b->ple_pc = dalloc(160); CK(cuMemcpyHtoD(b->ple_pc, ppc, 160));
			CK(cuMemHostGetDevicePointer(&b->ple_tab, ptab, 0));
			b->pg = dalloc(4 * 2560); b->pgq = dalloc(2 * 2560); b->pgs32 = dalloc(4 * 80); b->pgs16 = dalloc(4 * 160); b->pgas = dalloc(4 * 80);
			b->lk = dalloc(4u * (unsigned)s->pkn); b->kn = dalloc(4 * 10240); b->qn = dalloc(4 * 10240); b->lv = dalloc(4u * (unsigned)s->yn);
			b->pG = dalloc(4 * 10240); b->pgates = dalloc(16); b->px = dalloc(4 * 10240); b->ple_state = dalloc(4 * 10 * 10240); b->prows = dalloc(64);
		}
		if (is_attn) {
			b->qg = dalloc(4 * 12288); b->k = dalloc(4 * 512); b->v = dalloc(4 * 512); b->qo = dalloc(4 * 6144); b->aout = dalloc(4 * 6144);
			b->pm = dalloc(4 * 24 * 8); b->pd = dalloc(4 * 24 * 8); b->po = dalloc(4 * 24 * 8 * 256);
			b->kc = dalloc(d == 0 ? kcf * 4 : 256); b->vc = dalloc(d == 0 ? vcf * 4 : 256);
			b->rope_cs = dalloc(4 * 32 * T); b->rope_sn = dalloc(4 * 32 * T);
			CK(cuMemcpyHtoD(b->rope_cs, rcs, 4 * 32 * T)); CK(cuMemcpyHtoD(b->rope_sn, rsn, 4 * 32 * T));
		}
		if (mode == 1) { b->hlo = dalloc(4u * (unsigned)s->on); b->amo = dalloc(8); b->apart = dalloc(16 * 8); b->acnt = dalloc(4); b->tokout = dalloc(8); b->toks = dalloc(4 * (size_t)T + 16); }
		if (mode && !b->ids) b->ids = dalloc(4 * (size_t)T + 16);
		CK(cuMemcpyHtoD(b->S, hb->S, 4 * 10240));
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * XSLOT); x->flags = dalloc(512); x->xtab = dalloc(128);   /* flags: u32[6 dies][16 CTAs] */
		uint32_t fl[96]; for (int e = 0; e < 96; e++) fl[e] = e / 16 < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		dPROG[d] = dalloc(128 * 512); dCTL[d] = dalloc(0x100 + 4 * 512);
		x->xgen = dCTL[d] + 0x20;   /* the exchange count lives in ctl[8] between launches */
	}
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < nd; d++) { tab[d] = (uint64_t)xc[d].inbox; tab[6 + d] = (uint64_t)xc[d].flags; }
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab));
		/* the framework's exchange header in ctl: [3] me, [4] ndies, [16 + 2e] = &flags_e[me][0], [28] = my flags */
		uint32_t hdr[32] = { 0 }; hdr[3] = (uint32_t)d; hdr[4] = (uint32_t)nd;
		for (int e = 0; e < nd; e++) { uint64_t a = (uint64_t)xc[e].flags + 64u * (unsigned)d; memcpy(&hdr[16 + 2 * e], &a, 8); }
		uint64_t mf = (uint64_t)xc[d].flags; memcpy(&hdr[28], &mf, 8);
		CK(cuMemcpyHtoD(dCTL[d], hdr, sizeof hdr));
	}
	printf("%d die(s):", nd);
	for (int d = 0; d < nd; d++) printf(" [%d: gpu %d dj %d+%d cb %d+%d h %d+%d k %d..%d y %d+%d e %d+%d%s]", d, dev_of[d], sp[d].dj0, sp[d].djn, sp[d].cb0, sp[d].cbn, sp[d].h0, sp[d].hn, sp[d].klo, sp[d].khi, sp[d].y0, sp[d].yn, sp[d].e0, sp[d].en, sp[d].shexp ? " shexp" : "");
	printf("\n");
	int fails = 0;
	for (int t = 0; t < T; t++) {
		double t0 = now();
		if (getenv("DIES_TRACE")) fprintf(stderr, "[token %d] twin...\n", t);
		int stage = getenv("DIES_STAGE") ? atoi(getenv("DIES_STAGE")) : 0, notwin = getenv("DIES_NOTWIN") != NULL;   /* timing only */
		unsigned id = 0;
		if (mode == 1) {   /* a fresh random stream per token */
			for (int i = 0; i < 10240; i++) hb->S[i] = 0.5f * frand();
			for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].S, hb->S, 4 * 10240)); }
		}
		if (mode == 2) { id = (uint32_t)((rs >> 8) % 248320u); frand(); }
		if (!notwin) {
			if (mode == 1) head_gpu(&hw, hb, eps, nd);
			else if (mode == 2) embed_gpu(&hw, hb, id);
			else {
				if (has_ple) ple_prefix_gpu(&hw, hb, (unsigned)t, eps);
				if (is_attn) attn_block_gpu(&hw, hb, (unsigned)t, eps, nd, blk, stage); else deltanet_block_gpu(&hw, hb, (unsigned)t, eps, nd, blk, stage);
			}
		}
		double t1 = now();
		prog_t P[MAXD]; unsigned nph[MAXD];
		if (getenv("DIES_TRACE")) fprintf(stderr, "[token %d] twin done, building programs\n", t);
		for (int d = 0; d < nd; d++) {
			prog_init(&P[d]);
			if (mode == 1) prog_head(&P[d], &sp[d], &xc[d], &dw[d], &db[d], eps, (unsigned)t, 1u << 30);
			else if (mode == 2) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].ids + 4u * (unsigned)t, &id, 4)); prog_embed(&P[d], &dw[d], &db[d], (unsigned)t); }
			else {
				if (has_ple) prog_ple_prefix(&P[d], &sp[d], &xc[d], &dw[d], &db[d], (unsigned)t, eps);
				if (is_attn) prog_attn_block(&P[d], &sp[d], &xc[d], &dw[d], &db[d], (unsigned)t, eps);
				else prog_deltanet_block(&P[d], &sp[d], &xc[d], &dw[d], &db[d], (unsigned)t, eps);
			}
			nph[d] = (unsigned)P[d].n;
			if (getenv("DIES_NPH") && (unsigned)atoi(getenv("DIES_NPH")) < nph[d]) nph[d] = (unsigned)atoi(getenv("DIES_NPH"));
			CK(cuCtxSetCurrent(ctx[d]));
			CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
		}
		double t2 = now();
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			unsigned ntok = 1; void *params[] = { &dPROG[d], &nph[d], &dCTL[d], &ntok };
			CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		}
		if (getenv("DIES_TRACE")) fprintf(stderr, "[token %d] launched, waiting\n", t);
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
		if (getenv("DIES_TRACE")) fprintf(stderr, "[token %d] done\n", t);
		double t3 = now();
		printf("token %d: %u phases; twin %.2f s, dies %.3f ms\n", t, nph[0], t1 - t0, (t3 - t2) * 1e3);
		int bad = 0;
		if ((t3 - t2) > 1.5e-3 && getenv("DIES_STALL")) {   /* which die stalled where: the two longest phases per die */
			for (int d = 0; d < nd; d++) {
				CK(cuCtxSetCurrent(ctx[d]));
				uint32_t *ts = malloc(4 * (64 + nph[d])); CK(cuMemcpyDtoH(ts, dCTL[d], 4 * (64 + nph[d])));
				unsigned prev = ts[15], b1 = 0, i1 = 0, b2 = 0, i2 = 0;
				for (unsigned i = 0; i < nph[d]; i++) { unsigned dd = ts[64 + i] - prev; prev = ts[64 + i]; if (dd > b1) { b2 = b1; i2 = i1; b1 = dd; i1 = i; } else if (dd > b2) { b2 = dd; i2 = i; } }
				printf("   stall die %d: total %.0f us; phase %u %s %.0f us, phase %u %s %.0f us; launch->phase0 %.0f\n", d, (ts[64 + nph[d] - 1] - ts[15]) / 1e3, i1, die_opname[P[d].p[i1].w[0]], b1 / 1e3, i2, die_opname[P[d].p[i2].w[0]], b2 / 1e3, (ts[64] - ts[15]) / 1e3);
				free(ts);
			}
		}
		if (notwin) { for (int d = 0; d < nd; d++) prog_free(&P[d]); continue; }
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			char what[32];
			printf(" die %d\n", d);
			const split_t *s = &sp[d];
			if (mode == 1) {
				bad += cmpbuf("hlo", hb->logits + s->o0, db[d].hlo, 4u * (unsigned)s->on, 4);
				uint32_t am[2] = { hb->didx[d] - (unsigned)s->o0, 0 }; memcpy(&am[1], &hb->dmax[d], 4);
				bad += cmpbuf("amo", am, db[d].amo, 8, 4);
				uint32_t tk[2] = { hb->tok, 0 }; memcpy(&tk[1], &hb->tokmax, 4);
				bad += cmpbuf("tok", tk, db[d].tokout, 8, 4);
				continue;
			}
			if (mode == 2) { bad += cmpbuf("S", hb->S, db[d].S, 4 * 10240, 4); continue; }
			if (has_ple) {
				bad += cmpbuf("pg", hb->pg, db[d].pg, 4 * 2560, 4);
				bad += cmpbuf("lk", hb->pK + s->pk0, db[d].lk, 4u * (unsigned)s->pkn, 4);
				bad += cmpbuf("kn", hb->kn, db[d].kn, 4 * 10240, 4);
				bad += cmpbuf("qn", hb->qn, db[d].qn, 4 * 10240, 4);
				bad += cmpbuf("lv", hb->pV + s->y0, db[d].lv, 4u * (unsigned)s->yn, 4);
				bad += cmpbuf("pG", hb->pG, db[d].pG, 4 * 10240, 4);
				bad += cmpbuf("pgates", hb->pgates, db[d].pgates, 16, 4);
				bad += cmpbuf("px", hb->px, db[d].px, 4 * 10240, 4);
			}
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
			if (is_attn) {
				if (d == 0) {
					bad += cmpbuf("qg", hb->qg, db[d].qg, 4 * 12288, 4);
					bad += cmpbuf("k", hb->k, db[d].k, 4 * 512, 4);
					bad += cmpbuf("v", hb->v, db[d].v, 4 * 512, 4);
					bad += cmpbuf("qo", hb->qo, db[d].qo, 4 * 6144, 4);
					bad += cmpbuf("aout", hb->aout, db[d].aout, 4 * 6144, 4);
				}
				bad += cmpbuf("oq", hb->oq, db[d].oq, 2 * 6144, 2);
				bad += cmpbuf("ly", hb->y + s->y0, db[d].ly, 4u * (unsigned)s->yn, 4);
			} else {   /* attention-side locals (not overwritten by the FFN half) */
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
			uint32_t *ts = malloc(4 * (64 + nph[0])); CK(cuMemcpyDtoH(ts, dCTL[0], 4 * (64 + nph[0])));
			unsigned byop[DIE_NOPS] = {0}; unsigned prev = ts[15];
			for (unsigned i = 0; i < nph[0]; i++) {
				unsigned dd = ts[64 + i] - prev; prev = ts[64 + i];
				byop[P[0].p[i].w[0]] += dd;
				printf("   %2u %-10s units %5u %7.1f\n", i, die_opname[P[0].p[i].w[0]], P[0].p[i].w[1], dd / 1e3);
			}
			printf("  by op:");
			for (int o = 0; o < DIE_NOPS; o++) if (byop[o]) printf(" %s %.0f", die_opname[o], byop[o] / 1e3);
			printf("\n  total %.1f us\n", (ts[64 + nph[0] - 1] - ts[15]) / 1e3);
			free(ts);
		}
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); hold_close(&ar[d]); }
	return fails ? 3 : 0;
}
