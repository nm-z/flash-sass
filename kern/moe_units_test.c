/* moe_units_test: the K-split MoE ops of the unit layout (kern/hold2.c) on one die against C references:
 * moesel2 (unit list + zeroed partial), q2moegnu (gate / up rows of the listed 64-row units), ffnact128
 * (SiLU(g) * u quantised per 32 in 128-element vectors) and q2downu (the units' 64-column down blocks
 * weighted and summed on top of a partial, every row stored into its owner's inbox region).
 * The references mirror the kernels' integer dots and fp32 operation order; results are compared
 * bit for bit, and the largest relative error is reported as well.
 * Run on one GPU through gpuq: CUDA_VISIBLE_DEVICES=6 DIE_CUBIN=die_units.cubin ./moe_units_test */
#define _POSIX_C_SOURCE 200809L
#include "dieops.h"
#include "quant.h"
#include "fmath.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
#include "test_guards.h"

typedef struct { uint32_t w[32]; } desc_t;
static void d_init(desc_t *d, int op, unsigned units) { memset(d, 0, sizeof *d); d->w[0] = (uint32_t)op; d->w[1] = units; }
static void d_ptr(desc_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w + 1] = (uint32_t)(p >> 32); }
static void d_u32(desc_t *d, int addr, uint32_t v) { d->w[2 + (addr - 0x140) / 4] = v; }
static void d_f32(desc_t *d, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); d_u32(d, addr, u); }

static CUmodule mod; static CUfunction fn; static CUdeviceptr dPROG, dCTL;
static CUdeviceptr dalloc(size_t n) { return test_guard_alloc("moe", n, 0); }
static void run(const desc_t *prog, unsigned nph) {
	CK(cuMemcpyHtoD(dPROG, prog, sizeof(desc_t) * nph));
	unsigned t = 0; CK(cuMemcpyHtoD(dCTL + 8, &t, 4));
	unsigned ntok = 1; void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
}

static uint32_t rs = 12345;   /* MOE_SEED overrides */
static uint32_t rnd(void) { rs ^= rs << 13; rs ^= rs >> 17; rs ^= rs << 5; return rs; }
static float frand(void) { return (float)(rnd() >> 8) / 8388608.f - 1.f; }
static uint16_t f2h(float f) {   /* round-to-nearest-even fp16 for test data */
	uint32_t x; memcpy(&x, &f, 4); uint32_t s = (x >> 16) & 0x8000; int e = (int)((x >> 23) & 0xff) - 127 + 15; uint32_t m = x & 0x7fffff;
	if (e <= 0) return (uint16_t)s; if (e >= 31) return (uint16_t)(s | 0x7c00);
	uint32_t h = s | ((uint32_t)e << 10) | (m >> 13); uint32_t rest = m & 0x1fff; if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) h++; return (uint16_t)h;
}

#define K 2560
#define NSLOT 40
#define NEXP 32
#define MAXU 32

int main(void) {
	if (getenv("MOE_SEED")) rs = (uint32_t)strtoul(getenv("MOE_SEED"), NULL, 10) | 1u;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	dPROG = dalloc(128 * 16); dCTL = dalloc(8192);
	CK(cuMemsetD8(dCTL, 0, 8192));
	/* ---- the die's units: experts' units are consecutive slots, as hold2 lays them out ---- */
	uint32_t map[NEXP + 1]; memset(map, 0, sizeof map);
	int slot = 0;
	for (int e = 0; e < NEXP && slot < NSLOT; e++) { int n = (int)(rnd() % 4); if (slot + n > NSLOT) n = NSLOT - slot; map[e] = ((uint32_t)slot << 8) | (uint32_t)n; slot += n; }
	uint32_t idx[10]; float w[10];
	{ int used[NEXP] = { 0 }; for (int r = 0; r < 10; r++) { uint32_t e; do e = rnd() % NEXP; while (used[e]); used[e] = 1; idx[r] = e; w[r] = 0.05f + 0.1f * fabsf(frand()); } }
	/* reference list */
	uint32_t rl[MAXU]; float rw[MAXU]; int ne = 0;
	for (int r = 0; r < 10; r++) { uint32_t m = map[idx[r]]; for (uint32_t i = 0; i < (m & 0xff); i++) { rl[ne] = (m >> 8) + i; rw[ne] = w[r]; ne++; } }
	printf("unit list: %d entries\n", ne);
	/* ---- weights: gate / up units (64 rows x K Q2_0 tiles), down units (2560 x 16 B codes + 2560 fp16) ---- */
	int st = q2_0_tile_stride(K);
	size_t graw = (size_t)NSLOT * 64 * (K / 64) * 18;
	uint8_t *gsrc = malloc(graw), *usrc = malloc(graw), *gt = malloc((size_t)NSLOT * 64 * st), *ut = malloc((size_t)NSLOT * 64 * st);
	for (int pass = 0; pass < 2; pass++) {
		uint8_t *s = pass ? usrc : gsrc;
		for (size_t b = 0; b < graw / 18; b++) { uint16_t h = f2h(0.02f + 0.03f * fabsf(frand())); memcpy(s + 18 * b, &h, 2); for (int j = 0; j < 16; j++) s[18 * b + 2 + j] = (uint8_t)(rnd() & 0xff) & 0xbf; }   /* codes 0..2 mostly: Q2_0 never emits 3 */
	}
	repack_q2_0(gsrc, NSLOT * 64, K, gt); repack_q2_0(usrc, NSLOT * 64, K, ut);
	size_t ub = 2560 * 18;
	uint8_t *down = malloc((size_t)NSLOT * ub);
	for (int s2 = 0; s2 < NSLOT; s2++) {
		uint8_t *u = down + (size_t)s2 * ub;
		for (int r = 0; r < 2560; r++) { for (int j = 0; j < 16; j++) u[16 * r + j] = (uint8_t)(rnd() & 0xff) & 0xbf; uint16_t h = f2h(0.01f + 0.02f * fabsf(frand())); memcpy(u + 2560 * 16 + 2 * r, &h, 2); }
	}
	/* ---- activations z (quantised per 32) ---- */
	float z[K]; for (int i = 0; i < K; i++) z[i] = 2.f * frand();
	int16_t zq[K]; int32_t zs32[K / 32], zs16[K / 16]; float zas[K / 32];
	quant_act_gpu32(z, K, zq, zs32, zs16, zas);
	/* ---- device buffers ---- */
	CUdeviceptr dIDX = dalloc(64), dW = dalloc(64), dMAP = dalloc(sizeof map), dLIDX = dalloc(4 * MAXU), dLW = dalloc(4 * MAXU), dNE = dalloc(16), dPART = dalloc(4 * 2560);
	CUdeviceptr dG = dalloc((size_t)4 * 64 * MAXU), dGU = dalloc((size_t)4 * 64 * MAXU), dHQ = dalloc((size_t)2 * 64 * MAXU), dHS32 = dalloc(4 * 2 * MAXU), dHS16 = dalloc(4 * 4 * MAXU), dHAS = dalloc(4 * 2 * MAXU);
	/* the gather GEMV reads whole CTA row iterations past a unit (the holder keeps 4 MB of slack): keep 256 rows of slack */
	CUdeviceptr dGT = dalloc((size_t)(NSLOT * 64 + 256) * st), dUT = dalloc((size_t)(NSLOT * 64 + 256) * st), dDOWN = dalloc((size_t)NSLOT * ub);
	CUdeviceptr dZQ = dalloc(2 * K), dZS32 = dalloc(4 * K / 32), dZAS = dalloc(4 * K / 32);
	CUdeviceptr dIN[6], dTAB = dalloc(8 * 12);
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { dIN[d] = dalloc(4 * 6 * 2560); CK(cuMemsetD32(dIN[d], 0x7fc00000u, 6 * 2560)); tab[d] = dIN[d]; }
	CK(cuMemcpyHtoD(dTAB, tab, sizeof tab));
	CK(cuMemcpyHtoD(dIDX, idx, sizeof idx)); CK(cuMemcpyHtoD(dW, w, sizeof w)); CK(cuMemcpyHtoD(dMAP, map, sizeof map));
	CK(cuMemsetD32(dLIDX, 0xffffffffu, MAXU)); CK(cuMemsetD32(dLW, 0x7fc00000u, MAXU)); CK(cuMemsetD32(dNE, 0xffffffffu, 1));
	CK(cuMemsetD32(dPART, 0x7fc00000u, 2560));
	CK(cuMemcpyHtoD(dGT, gt, (size_t)NSLOT * 64 * st)); CK(cuMemcpyHtoD(dUT, ut, (size_t)NSLOT * 64 * st)); CK(cuMemcpyHtoD(dDOWN, down, (size_t)NSLOT * ub));
	CK(cuMemcpyHtoD(dZQ, zq, sizeof zq)); CK(cuMemcpyHtoD(dZS32, zs32, sizeof zs32)); CK(cuMemcpyHtoD(dZAS, zas, sizeof zas));
	int fails = 0;
	/* ---- launch 1: moesel2 ---- */
	desc_t prog[8];
	d_init(&prog[0], OP_MOESEL2, 1);
	d_ptr(&prog[0], 0x140, dIDX); d_ptr(&prog[0], 0x148, dW); d_ptr(&prog[0], 0x150, dMAP); d_ptr(&prog[0], 0x158, dLIDX); d_ptr(&prog[0], 0x160, dLW); d_ptr(&prog[0], 0x168, dNE); d_ptr(&prog[0], 0x170, dPART);
	run(prog, 1);
	uint32_t gl[MAXU], gne; float glw[MAXU], gpart[2560];
	CK(cuMemcpyDtoH(gl, dLIDX, sizeof gl)); CK(cuMemcpyDtoH(glw, dLW, sizeof glw)); CK(cuMemcpyDtoH(&gne, dNE, 4)); CK(cuMemcpyDtoH(gpart, dPART, sizeof gpart));
	int bad = gne != (uint32_t)ne;
	for (int k = 0; k < ne && !bad; k++) bad = gl[k] != rl[k] || memcmp(&glw[k], &rw[k], 4);
	for (int r = 0; r < 2560 && !bad; r++) bad = gpart[r] != 0.f;
	printf("moesel2: %s (ne %u, expected %d)\n", bad ? "MISMATCH" : "BIT-EXACT", gne, ne);
	fails += bad;
	/* a partial as the shared expert would leave it */
	float part[2560]; for (int r = 0; r < 2560; r++) part[r] = 0.1f * frand();
	CK(cuMemcpyHtoD(dPART, part, sizeof part));
	/* ---- launch 2: gate, up, act, down ---- */
	const int me = 2; const uint32_t bnd[5] = { 427, 854, 1281, 1708, 2135 };
	d_init(&prog[0], OP_Q2MOEGNU, 16);
	d_ptr(&prog[0], 0x140, dGT); d_ptr(&prog[0], 0x148, dZQ); d_ptr(&prog[0], 0x150, dZS32); d_ptr(&prog[0], 0x158, dG); d_u32(&prog[0], 0x160, 64); d_f32(&prog[0], 0x164, 1.f);
	d_ptr(&prog[0], 0x168, dLIDX); d_ptr(&prog[0], 0x170, dZAS); d_ptr(&prog[0], 0x178, 0); d_ptr(&prog[0], 0x180, dNE);
	prog[0].w[30] |= 4;   /* gate and up write disjoint outputs */
	prog[1] = prog[0]; d_ptr(&prog[1], 0x140, dUT); d_ptr(&prog[1], 0x158, dGU); prog[1].w[30] = 0;
	d_init(&prog[2], OP_FFNACT128, MAXU / 2);
	d_ptr(&prog[2], 0x140, dG); d_ptr(&prog[2], 0x148, dGU); d_ptr(&prog[2], 0x150, dHQ); d_ptr(&prog[2], 0x158, dHS32); d_ptr(&prog[2], 0x160, dHS16); d_ptr(&prog[2], 0x168, dHAS); d_u32(&prog[2], 0x170, 128);
	d_init(&prog[3], OP_Q2DOWNU, 16);
	d_ptr(&prog[3], 0x140, dDOWN); d_ptr(&prog[3], 0x148, dHQ); d_ptr(&prog[3], 0x150, dHS32); d_ptr(&prog[3], 0x158, dTAB); d_u32(&prog[3], 0x160, 2560); d_u32(&prog[3], 0x164, 4u * 2560u * me);
	d_ptr(&prog[3], 0x168, dLIDX); d_ptr(&prog[3], 0x170, dHAS); d_ptr(&prog[3], 0x178, dLW); d_ptr(&prog[3], 0x180, dNE); d_ptr(&prog[3], 0x188, dPART);
	for (int i = 0; i < 5; i++) d_u32(&prog[3], 0x190 + 4 * i, bnd[i]);
	/* MOE_STEP=1: one op per launch (finds a faulting op) */
	if (getenv("MOE_STEP")) { for (int i = 0; i < 4; i++) { run(prog + i, 1); printf("  op %d ok\n", i); fflush(stdout); } } else run(prog, 4);
	/* ---- references ---- */
	int8_t zq8[K]; for (int i = 0; i < K; i++) zq8[i] = (int8_t)zq[i];
	float *g = malloc(4 * 64 * MAXU), *gu = malloc(4 * 64 * MAXU);
	for (int k = 0; k < ne; k++) {
		gemv_q2_0_ref(gt + (size_t)rl[k] * 64 * st, 64, K, zq8, zs32, zas, g + 64 * k);
		gemv_q2_0_ref(ut + (size_t)rl[k] * 64 * st, 64, K, zq8, zs32, zas, gu + 64 * k);
	}
	float *gg = malloc(4 * 64 * MAXU), *ggu = malloc(4 * 64 * MAXU);
	CK(cuMemcpyDtoH(gg, dG, 4 * 64 * MAXU)); CK(cuMemcpyDtoH(ggu, dGU, 4 * 64 * MAXU));
	bad = memcmp(g, gg, 4 * 64 * ne) || memcmp(gu, ggu, 4 * 64 * ne);
	printf("q2moegnu: %s\n", bad ? "MISMATCH" : "BIT-EXACT"); fails += bad;
	/* the activation of the GPU's g / gu (so a gate/up difference does not propagate) */
	int nv = (ne + 1) / 2;
	int16_t *hq = malloc(2 * 128 * nv); int32_t *hs32 = malloc(4 * 4 * nv), *hs16 = malloc(4 * 8 * nv); float *has = malloc(4 * 4 * nv);
	for (int v = 0; v < nv; v++) {
		float x[128]; for (int i = 0; i < 128; i++) x[i] = silu_gpu(gg[128 * v + i]) * ggu[128 * v + i];
		quant_act_gpu32(x, 128, hq + 128 * v, hs32 + 4 * v, hs16 + 8 * v, has + 4 * v);
	}
	int16_t *ghq = malloc(2 * 128 * nv); int32_t *ghs32 = malloc(4 * 4 * nv); float *ghas = malloc(4 * 4 * nv);
	CK(cuMemcpyDtoH(ghq, dHQ, 2 * 128 * nv)); CK(cuMemcpyDtoH(ghs32, dHS32, 4 * 4 * nv)); CK(cuMemcpyDtoH(ghas, dHAS, 4 * 4 * nv));
	/* entries past ne in the last vector are garbage on both sides: compare the listed units only */
	bad = memcmp(hq, ghq, 2 * 64 * ne) || memcmp(hs32, ghs32, 4 * 2 * ne) || memcmp(has, ghas, 4 * 2 * ne);
	printf("ffnact128: %s\n", bad ? "MISMATCH" : "BIT-EXACT"); fails += bad;
	/* down: y[r] = part[r] + sum_k lw_k * (fma(as1, d dot1, as0 * (d dot0))) */
	float y[2560], maxrel = 0.f; int ybad = 0, shown = 0;
	for (int r = 0; r < 2560; r++) {
		float acc = part[r];
		for (int k = 0; k < ne; k++) {
			const uint8_t *u = down + (size_t)rl[k] * ub;
			float d; { uint16_t h; memcpy(&h, u + 2560 * 16 + 2 * r, 2); d = fp16_to_fp32(h); }
			float accf = 0.f;
			for (int h = 0; h < 2; h++) {
				int32_t dot = 0;
				for (int j = 0; j < 32; j++) { int e = 32 * h + j; int q = (u[16 * r + (e >> 2)] >> (2 * (e & 3))) & 3; dot += q * ghq[64 * k + e]; }
				dot -= ghs32[2 * k + h];
				float uu = d * (float)dot;
				accf = h ? fmaf(ghas[2 * k + 1], uu, accf) : ghas[2 * k] * uu;
			}
			acc = fmaf(glw[k], accf, acc);
		}
		y[r] = acc;
		int own = 0; for (int i = 0; i < 5; i++) own += r >= (int)bnd[i];
		float got; CK(cuMemcpyDtoH(&got, dIN[own] + 4 * ((size_t)me * 2560 + r), 4));
		if (memcmp(&got, &acc, 4)) { ybad++; if (shown++ < 5) printf("  row %d (owner %d): gpu %.9g ref %.9g\n", r, own, got, acc); }
		float rel = fabsf(got - acc) / (fabsf(acc) + 1e-6f); if (rel > maxrel) maxrel = rel;
	}
	/* nothing outside the owners' rows of this die's region */
	int stray = 0;
	for (int d = 0; d < 6; d++) {
		float *in = malloc(4 * 6 * 2560); CK(cuMemcpyDtoH(in, dIN[d], 4 * 6 * 2560));
		for (int i = 0; i < 6 * 2560; i++) { int src = i / 2560, r = i % 2560, own = 0; for (int b = 0; b < 5; b++) own += r >= (int)bnd[b]; uint32_t bits; memcpy(&bits, &in[i], 4); int expect_written = src == me && own == d; if (!expect_written && bits != 0x7fc00000u) stray++; }
		free(in);
	}
	printf("q2downu: %s (%d rows differ, max rel %.3g, %d stray writes)\n", ybad || stray ? "MISMATCH" : "BIT-EXACT", ybad, maxrel, stray);
	fails += ybad != 0 || stray != 0;
	if (test_guard_check()) { printf("guards: FAIL\n"); fails++; }
	printf("%s\n", fails ? "FAIL" : "PASS");
	return fails ? 3 : 0;
}
