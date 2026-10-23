/* block_test: one Gated DeltaNet block through the persistent kernel's program
 * (prog_deltanet_block) against the C block twin, bit for bit, over T tokens with
 * the conv / DeltaNet states carried. Weights come from a running `hold` (arena)
 * and its tile cache (host copies). usage: block_test <manifest> <cachedir> <blk> [T] */
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

static const char *cachedir;
static hold_arena ar;
static char tname[128];
static const char *tn(int blk, const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static void *hostimg(const char *name) {
	const hold_tensor *t = hold_find(&ar, name);
	if (!t) { fprintf(stderr, "no tensor %s in the manifest\n", name); exit(1); }
	char fn[512]; snprintf(fn, sizeof fn, "%s/%s", cachedir, name);
	FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
	void *p = malloc(t->bytes);
	if (fread(p, 1, t->bytes, f) != t->bytes) { fprintf(stderr, "short read %s\n", fn); exit(1); }
	fclose(f);
	return p;
}
static CUdeviceptr dptr(const char *name) { CUdeviceptr p = hold_ptr(&ar, name); if (!p) { fprintf(stderr, "no %s\n", name); exit(1); } return p; }
static uint32_t ttype(const char *name) { return hold_find(&ar, name)->type; }
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }

static int cmpbuf(const char *what, const void *ref, CUdeviceptr dev, size_t bytes, int elemsz) {
	void *got = malloc(bytes);
	CK(cuMemcpyDtoH(got, dev, bytes));
	size_t n = bytes / elemsz; int bad = 0; size_t first = 0;
	for (size_t i = 0; i < n; i++) if (memcmp((const char *)ref + i * elemsz, (const char *)got + i * elemsz, elemsz)) { if (!bad) first = i; bad++; }
	if (bad) {
		printf("  %-8s MISMATCH %d of %zu (first at %zu", what, bad, n, first);
		if (elemsz == 4) printf(": gpu %.9g ref %.9g", ((float *)got)[first], ((const float *)ref)[first]);
		printf(")\n");
		if (getenv("BLOCK_HIST")) { int per = atoi(getenv("BLOCK_HIST")); printf("   per-%d:", per); for (size_t c = 0; c < (n + per - 1) / per; c++) { int k = 0; for (size_t i = c * per; i < n && i < (c + 1) * per; i++) if (memcmp((const char *)ref + i * elemsz, (const char *)got + i * elemsz, elemsz)) k++; if (k) printf(" %zu:%d", c, k); } printf("\n"); }
	} else printf("  %-8s exact (%zu)\n", what, n);
	free(got);
	return bad;
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: block_test <manifest> <cachedir> <blk> [T]\n"); return 2; }
	cachedir = argv[2]; int blk = atoi(argv[3]); int T = argc > 4 ? atoi(argv[4]) : 2; float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	if (hold_open(&ar, argv[1])) return 1;
	CK(cuModuleLoad(&mod, "die.cubin")); CK(cuModuleGetFunction(&fn, mod, "die"));
	/* host weights */
	dn_hweights_t hw; memset(&hw, 0, sizeof hw);
	hw.hc_attn_norm = hostimg(tn(blk, "hc_attn_norm.weight")); hw.hc_attn_dj = hostimg(tn(blk, "hc_attn_dj.weight")); hw.hc_attn_up = hostimg(tn(blk, "hc_attn_up.weight"));
	hw.hc_ffn_norm = hostimg(tn(blk, "hc_ffn_norm.weight")); hw.hc_ffn_dj = hostimg(tn(blk, "hc_ffn_dj.weight")); hw.hc_ffn_up = hostimg(tn(blk, "hc_ffn_up.weight"));
	hw.attn_qkv = hostimg(tn(blk, "attn_qkv.weight")); hw.attn_qkv_type = ttype(tname);
	hw.attn_gate = hostimg(tn(blk, "attn_gate.weight")); hw.attn_gate_type = ttype(tname);
	hw.ssm_out = hostimg(tn(blk, "ssm_out.weight")); hw.ssm_out_type = ttype(tname);
	hw.ssm_ab = hostimg(tn(blk, "ssm_ab.weight")); hw.ssm_ab_bias = hostimg(tn(blk, "ssm_ab.bias")); hw.ssm_conv1d = hostimg(tn(blk, "ssm_conv1d.weight"));
	hw.ssm_a = hostimg(tn(blk, "ssm_a")); hw.ssm_norm = hostimg(tn(blk, "ssm_norm.weight"));
	hw.ffn_gate_inp_x = hostimg(tn(blk, "ffn_gate_inp_x.weight"));
	hw.ffn_gate_exps = hostimg(tn(blk, "ffn_gate_exps.weight")); hw.ffn_up_exps = hostimg(tn(blk, "ffn_up_exps.weight")); hw.ffn_down_exps = hostimg(tn(blk, "ffn_down_exps.weight"));
	hw.ffn_gate_shexp = hostimg(tn(blk, "ffn_gate_shexp.weight")); hw.ffn_gate_shexp_type = ttype(tname);
	hw.ffn_up_shexp = hostimg(tn(blk, "ffn_up_shexp.weight")); hw.ffn_up_shexp_type = ttype(tname);
	hw.ffn_down_shexp = hostimg(tn(blk, "ffn_down_shexp.weight")); hw.ffn_down_shexp_type = ttype(tname);
	/* device weights */
	dn_weights_t dw; memset(&dw, 0, sizeof dw);
	dw.hc_attn_norm = dptr(tn(blk, "hc_attn_norm.weight")); dw.hc_attn_dj = dptr(tn(blk, "hc_attn_dj.weight")); dw.hc_attn_up = dptr(tn(blk, "hc_attn_up.weight"));
	dw.hc_ffn_norm = dptr(tn(blk, "hc_ffn_norm.weight")); dw.hc_ffn_dj = dptr(tn(blk, "hc_ffn_dj.weight")); dw.hc_ffn_up = dptr(tn(blk, "hc_ffn_up.weight"));
	dw.attn_qkv = dptr(tn(blk, "attn_qkv.weight")); dw.attn_qkv_type = hw.attn_qkv_type;
	dw.attn_gate = dptr(tn(blk, "attn_gate.weight")); dw.attn_gate_type = hw.attn_gate_type;
	dw.ssm_out = dptr(tn(blk, "ssm_out.weight")); dw.ssm_out_type = hw.ssm_out_type;
	dw.ssm_ab = dptr(tn(blk, "ssm_ab.weight")); dw.ssm_ab_bias = dptr(tn(blk, "ssm_ab.bias"));
	dw.ssm_conv1d = dptr(tn(blk, "ssm_conv1d.weight")); dw.ssm_a = dptr(tn(blk, "ssm_a")); dw.ssm_norm = dptr(tn(blk, "ssm_norm.weight"));
	dw.ffn_gate_inp_x = dptr(tn(blk, "ffn_gate_inp_x.weight"));
	dw.ffn_gate_exps = dptr(tn(blk, "ffn_gate_exps.weight")); dw.ffn_up_exps = dptr(tn(blk, "ffn_up_exps.weight")); dw.ffn_down_exps = dptr(tn(blk, "ffn_down_exps.weight"));
	dw.ffn_gate_shexp = dptr(tn(blk, "ffn_gate_shexp.weight")); dw.ffn_gate_shexp_type = hw.ffn_gate_shexp_type;
	dw.ffn_up_shexp = dptr(tn(blk, "ffn_up_shexp.weight")); dw.ffn_up_shexp_type = hw.ffn_up_shexp_type;
	dw.ffn_down_shexp = dptr(tn(blk, "ffn_down_shexp.weight")); dw.ffn_down_shexp_type = hw.ffn_down_shexp_type;
	/* buffers */
	dn_hbufs_t *hb = calloc(1, sizeof *hb); hb->dn_state = calloc(48 * 16384, 4);
	dn_bufs_t db;
	db.S = dalloc(4 * 10240); db.xh = dalloc(4 * 10240); db.down = dalloc(4 * 324); db.t = dalloc(4 * 320); db.up = dalloc(4 * 10240); db.z = dalloc(4 * 2560);
	db.zq = dalloc(2 * 2560); db.zs32 = dalloc(4 * 80); db.zs16 = dalloc(4 * 160); db.zas = dalloc(4 * 80);
	db.qkv = dalloc(4 * 10240); db.gate = dalloc(4 * 6144); db.a = dalloc(4 * 96); db.u = dalloc(4 * 10240); db.o = dalloc(4 * 6144); db.on = dalloc(4 * 6144);
	db.oq = dalloc(2 * 6144); db.os32 = dalloc(4 * 192); db.os16 = dalloc(4 * 384); db.oas = dalloc(4 * 192);
	db.y = dalloc(4 * 2560); db.r = dalloc(4 * 513); db.idx = dalloc(64); db.w = dalloc(64); db.s = dalloc(16);
	db.g = dalloc(4 * 7040); db.gu = dalloc(4 * 7040); db.hq = dalloc(2 * 7040); db.hs32 = dalloc(4 * 220); db.hs16 = dalloc(4 * 440); db.has = dalloc(4 * 220);
	db.out = dalloc(4 * 2560); db.conv_state = dalloc(4 * 4 * 10240); db.dn_state = dalloc((size_t)48 * 16384 * 4);
	for (int i = 0; i < 10240; i++) hb->S[i] = 0.5f * frand();
	CK(cuMemcpyHtoD(db.S, hb->S, 4 * 10240));
	CUdeviceptr dPROG = dalloc(128 * 128), dCTL = dalloc(0x40 + 4 * 512);
	void *params[] = { &dPROG, NULL, &dCTL };
	int fails = 0;
	for (int t = 0; t < T; t++) {
		double t0 = now();
		deltanet_block_gpu(&hw, hb, (unsigned)t, eps);
		double t1 = now();
		prog_t P; prog_init(&P);
		prog_deltanet_block(&P, &dw, &db, (unsigned)t, eps);
		unsigned nph = (unsigned)P.n;
		if (getenv("BLOCK_NPH") && (unsigned)atoi(getenv("BLOCK_NPH")) < nph) nph = (unsigned)atoi(getenv("BLOCK_NPH"));   /* run a prefix of the program */
		CK(cuMemcpyHtoD(dPROG, P.p, sizeof(phase_t) * P.n));
		params[1] = &nph;
		double t2 = now();
		CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double t3 = now();
		printf("token %d: %u phases; twin %.2f s, die %.3f ms\n", t, nph, t1 - t0, (t3 - t2) * 1e3);
		int bad = 0;
		bad += cmpbuf("qkv", hb->qkv, db.qkv, 4 * 10240, 4);
		bad += cmpbuf("gate", hb->gate, db.gate, 4 * 6144, 4);
		bad += cmpbuf("down", hb->down, db.down, 4 * 324, 4);
		bad += cmpbuf("ab", hb->a, db.a, 4 * 96, 4);
		bad += cmpbuf("u", hb->u, db.u, 4 * 10240, 4);
		bad += cmpbuf("o", hb->o, db.o, 4 * 6144, 4);
		bad += cmpbuf("on", hb->on, db.on, 4 * 6144, 4);
		bad += cmpbuf("y", hb->y, db.y, 4 * 2560, 4);
		bad += cmpbuf("z(ffn)", hb->z, db.z, 4 * 2560, 4);
		bad += cmpbuf("r", hb->r, db.r, 4 * 513, 4);
		bad += cmpbuf("idx", hb->idx, db.idx, 40, 4);
		bad += cmpbuf("w", hb->w, db.w, 40, 4);
		bad += cmpbuf("g", hb->g, db.g, 4 * 7040, 4);
		bad += cmpbuf("gu", hb->gu, db.gu, 4 * 7040, 4);
		bad += cmpbuf("hq", hb->hq, db.hq, 2 * 7040, 2);
		bad += cmpbuf("out", hb->out, db.out, 4 * 2560, 4);
		bad += cmpbuf("S", hb->S, db.S, 4 * 10240, 4);
		if (bad) fails++;
		if (t == T - 1 && getenv("BLOCK_PROF")) {   /* per-phase time from the die's %globaltimer stamps */
			uint32_t *ts = malloc(4 * (16 + nph)); CK(cuMemcpyDtoH(ts, dCTL, 4 * (16 + nph)));
			if (getenv("BLOCK_RAWCTL")) { printf("  ctl:"); for (int i = 0; i < 24; i++) printf(" %u", ts[i]); printf("\n"); }
			unsigned byop[DIE_NOPS] = {0}; unsigned prev = ts[15];
			printf("  phase timing (us): launch->phase0 %.1f\n", (ts[16] - prev) / 1e3);
			for (unsigned i = 0; i < nph; i++) {
				unsigned d = ts[16 + i] - prev; prev = ts[16 + i];
				byop[P.p[i].w[0]] += d;
				printf("   %2u %-10s units %5u %7.1f\n", i, die_opname[P.p[i].w[0]], P.p[i].w[1], d / 1e3);
			}
			printf("  by op:");
			for (int o = 0; o < DIE_NOPS; o++) if (byop[o]) printf(" %s %.0f", die_opname[o], byop[o] / 1e3);
			printf("\n  total %.1f us\n", (ts[16 + nph - 1] - ts[15]) / 1e3);
			free(ts);
		}
		prog_free(&P);
	}
	hold_close(&ar);
	return fails ? 3 : 0;
}
