/* die_test: the persistent kernel framework on a 5-phase program (embrow, plegate,
 * actq, ffnact, argmax), each phase checked against its C twin; the program is run
 * twice so the barrier state is exercised across launches.
 * DIE_EXCHANGE_TEST=1 instead checks exchange publication across 64 launches;
 * DIE_HCUP_TEST=1 checks fused HC up against the C twins with guarded buffers.
 * FLASH_CUBIN selects the persistent image for every mode.
 * DIE_LAUNCH_WIDTH=1 selects the explicit scalar ABI; 0 retains legacy scalar. */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include "gguf.h"
#include "fmath.h"
#include "ple.h"
#include "hc.h"
#include "split.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 101;
static uint32_t launch_width;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

#include "prog.h"
#include "test_guards.h"
static void desc_init(uint32_t *d, int op, unsigned units) { memset(d, 0, 128); d[0] = op; d[1] = units; }
static void desc_ptr(uint32_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d[w] = (uint32_t)p; d[w + 1] = (uint32_t)(p >> 32); }
static void desc_u32(uint32_t *d, int addr, uint32_t v) { d[2 + (addr - 0x140) / 4] = v; }

static int cmpf(const char *what, const float *ref, const float *got, size_t n) {
	int bad = 0;
	for (size_t i = 0; i < n; i++) if (memcmp(&ref[i], &got[i], 4)) { if (bad < 3) printf("  %s[%zu]: gpu %.9g ref %.9g\n", what, i, got[i], ref[i]); bad++; }
	printf("  %-8s %s (%d of %zu differ)\n", what, bad ? "MISMATCH" : "BIT-EXACT", bad, n);
	return bad;
}

/* One physical GPU, six logical destinations. The absent publishers' flags stay
 * at UINT32_MAX; all six destinations publish into this GPU's die-0 flag row. */
static int exchange_test(CUfunction fn) {
	enum { NCTA = 16, NDIE = 6, WORDS = NCTA * 256, LAUNCHES = 64 };
	const unsigned chunks[] = { 1, 2, 3, 7, 16, 31, 64, 127 };
	const uint32_t exchange_seed = 65521, generation_seed = 7;
	uint32_t ctl[64] = { 0 }, flags[NDIE * NCTA], src[WORDS], got[WORDS];
	uint32_t prog[2][32];
	CUdeviceptr dctl, dflags, dsrc, dtab, dprog, inbox[NDIE];
	CK(cuMemAlloc(&dctl, 8192)); CK(cuMemsetD8(dctl, 0, 8192));
	CK(cuMemAlloc(&dflags, sizeof flags)); CK(cuMemAlloc(&dsrc, sizeof src));
	CK(cuMemAlloc(&dtab, 12 * sizeof(uint64_t))); CK(cuMemAlloc(&dprog, sizeof prog));
	uint64_t tab[12];
	for (unsigned e = 0; e < NDIE; e++) {
		CK(cuMemAlloc(&inbox[e], sizeof src));
		tab[e] = inbox[e]; tab[NDIE + e] = dflags;
		uint64_t row = dflags;
		memcpy(&ctl[16 + 2 * e], &row, sizeof row);
	}
	for (unsigned i = 0; i < NDIE * NCTA; i++) flags[i] = i < NCTA ? exchange_seed : UINT32_MAX;
	ctl[1] = generation_seed; ctl[3] = 0; ctl[4] = NDIE; ctl[8] = exchange_seed;
	ctl[9] = launch_width;
	for (unsigned i = 0; i < NCTA; i++) ctl[32 + i] = generation_seed;
	uint64_t own_flags = dflags; memcpy(&ctl[28], &own_flags, sizeof own_flags);
	CK(cuMemcpyHtoD(dctl, ctl, sizeof ctl)); CK(cuMemcpyHtoD(dflags, flags, sizeof flags));
	CK(cuMemcpyHtoD(dtab, tab, sizeof tab));
	desc_init(prog[0], OP_XSEND, NCTA);
	desc_ptr(prog[0], 0x140, dtab); desc_ptr(prog[0], 0x148, dsrc);
	desc_u32(prog[0], 0x150, sizeof src); desc_u32(prog[0], 0x15c, NDIE);
	desc_ptr(prog[0], 0x160, dctl + 0x20); prog[0][30] = 1;
	desc_init(prog[1], OP_EMBROW, 0); prog[1][30] = 2;
	CK(cuMemcpyHtoD(dprog, prog, sizeof prog));
	unsigned nph = 2, ntok = 0;
	void *params[] = { &dprog, &nph, &dctl, &ntok };
	for (unsigned run = 0; run < LAUNCHES; run++) {
		unsigned count = chunks[run % (sizeof chunks / sizeof chunks[0])];
		ntok += count;
		for (unsigned i = 0; i < WORDS; i++) src[i] = 0x5a000000u ^ (run * 65537u + i);
		CK(cuMemcpyHtoD(dsrc, src, sizeof src));
		for (unsigned e = 0; e < NDIE; e++) CK(cuMemsetD32(inbox[e], 0xdeadbeefu, WORDS));
		/* Preserve both the mirror and every published flag between launches. */
		CK(cuLaunchKernel(fn, NCTA, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ctl, dctl, sizeof ctl)); CK(cuMemcpyDtoH(flags, dflags, sizeof flags));
		uint32_t expected = exchange_seed + ntok, generation = generation_seed + 2 * ntok;
		unsigned bad = ctl[8] != expected || ctl[2] != ntok || ctl[1] != generation || ctl[11] != 0;
		for (unsigned i = 0; i < NDIE * NCTA; i++) {
			uint32_t want = i < NCTA ? expected : UINT32_MAX;
			if (flags[i] != want) {
				fprintf(stderr, "exchange run %u flag[%u]=%u expected %u\n", run, i, flags[i], want);
				bad++;
			}
		}
		for (unsigned i = 0; i < NCTA; i++) bad += ctl[32 + i] != generation;
		for (unsigned e = 0; e < NDIE; e++) {
			CK(cuMemcpyDtoH(got, inbox[e], sizeof got));
			if (memcmp(src, got, sizeof src)) { fprintf(stderr, "exchange run %u inbox %u differs\n", run, e); bad++; }
		}
		printf("{\"case\":\"exchange_counter_mirror\",\"launch\":%u,\"chunk_tokens\":%u,\"tokens\":%u,\"exchange\":%u,\"expected_exchange\":%u,\"generation\":%u,\"expected_generation\":%u,\"failures\":%u}\n",
			run, count, ctl[2], ctl[8], expected, ctl[1], generation, bad);
		if (bad) return 3;
	}
	for (unsigned e = 0; e < NDIE; e++) CK(cuMemFree(inbox[e]));
	CK(cuMemFree(dprog)); CK(cuMemFree(dtab)); CK(cuMemFree(dsrc)); CK(cuMemFree(dflags)); CK(cuMemFree(dctl));
	printf("{\"case\":\"exchange_counter_mirror\",\"status\":\"passed\",\"launches\":%u,\"tokens\":%u,\"published_flags\":%u,\"destinations\":%u}\n", LAUNCHES, ntok, NCTA, NDIE);
	return 0;
}

static CUdeviceptr hcup_copy(const char *name, const void *src, size_t bytes) {
	CUdeviceptr p = test_guard_alloc(name, bytes, 0);
	if (src) CK(cuMemcpyHtoD(p, src, bytes));
	return p;
}
static int hcup_unchanged(const char *name, CUdeviceptr p, const void *ref, size_t bytes) {
	void *got = malloc(bytes); if (!got) exit(2);
	CK(cuMemcpyDtoH(got, p, bytes));
	int bad = memcmp(got, ref, bytes) != 0; free(got);
	if (bad) fprintf(stderr, "HCUP modified input %s\n", name);
	return bad;
}
static int hcup_test(CUfunction fn) {
	enum { COLS = 2560, K = 320, NDIE = 6, NCTA = 16 };
	if (!getenv("SPLIT_SHARES")) setenv("SPLIT_SHARES", "112,97,81,116,97,97", 1);
	uint16_t *weights = malloc((size_t)4 * COLS * K * 2);
	float *up = malloc(4 * COLS * 4), *xh = malloc(4 * COLS * 4), *mixed = malloc(COLS * 4);
	float *got = malloc(2 * XSLOT), *peer_ref = malloc(2 * XSLOT);
	if (!weights || !up || !xh || !mixed || !got || !peer_ref) exit(2);
	for (size_t i = 0; i < (size_t)4 * COLS * K; i++) {
		float w = .125f * frand(); uint32_t bits; memcpy(&bits, &w, 4); weights[i] = (uint16_t)(bits >> 16);
	}
	const float edge[] = { -352.f, -348.f, -64.f, -4.f, -0.f, 0.f, .000244140625f, 4.f, 64.f, 348.f, 352.f };
	const uint32_t inject[] = { 0x7fc12345u, 0x80000000u, 0x3f800001u, 0xc2800000u };
	unsigned cases = 0; int fails = 0;
	for (unsigned pattern = 0; pattern < 3; pattern++) {
		float down[324], t[324];
		for (unsigned i = 0; i < K; i++) {
			if (!pattern) down[i] = 40.f * frand();
			else if (pattern == 1) down[i] = (i < 256 ? -1.f : 1.f) * (.125f + (float)(i % 67));
			else down[i] = edge[(7 * i + i / 64) % (sizeof edge / sizeof edge[0])];
		}
		memcpy(down + K, inject, sizeof inject);
		hc_silu_gpu(down, t); memcpy(t + K, down + K, sizeof inject);
		for (unsigned i = 0; i < 4 * COLS; i++) xh[i] = 2.f * frand();
		gemv_bf16_ref(weights, 4 * COLS, K, t, up);
		hc_mix_gpu(up, xh, mixed);
		for (unsigned die = 0; die < NDIE; die++) {
			unsigned base = 32 * split_bound(80, (int)die, NDIE);
			unsigned cols = 32 * split_bound(80, (int)die + 1, NDIE) - base;
			if (!cols) { fputs("HCUP split has no columns\n", stderr); exit(2); }
			size_t wbytes = (size_t)4 * cols * K * 2, ubytes = (size_t)4 * cols * 4;
			uint16_t *local_w = malloc(wbytes); float *local_up = malloc(ubytes);
			if (!local_w || !local_up) exit(2);
			for (unsigned lane = 0; lane < 4; lane++) {
				memcpy(local_w + (size_t)lane * cols * K, weights + (size_t)(lane * COLS + base) * K, (size_t)cols * K * 2);
				memcpy(local_up + lane * cols, up + lane * COLS + base, (size_t)cols * 4);
			}
			for (unsigned variant = 0; variant < 4; variant++) {
				unsigned tn = variant & 1 ? 324 : 320, slot = (variant >> 1) * XSLOT;
				uint32_t ctl[64] = {0}, flags[NDIE * NCTA], prog[1][32];
				CUdeviceptr dW = hcup_copy("HCUP weights", local_w, wbytes);
				CUdeviceptr dD = hcup_copy("HCUP down", down, tn * 4), dXH = hcup_copy("HCUP XH", xh, 4 * COLS * 4);
				CUdeviceptr dU = hcup_copy("HCUP up rows", NULL, ubytes), dT = hcup_copy("HCUP T", NULL, tn * 4);
				CUdeviceptr dCTL = hcup_copy("HCUP control", NULL, 8192), inbox[NDIE];
				uint64_t tab[12];
				for (unsigned i = 0; i < NDIE * NCTA; i++) flags[i] = i / NCTA == die ? 17u : UINT32_MAX;
				CUdeviceptr dF = hcup_copy("HCUP flags", flags, sizeof flags);
				for (unsigned e = 0; e < NDIE; e++) {
					inbox[e] = hcup_copy("HCUP peer inbox", NULL, 2 * XSLOT);
					CK(cuMemsetD32(inbox[e], 0x7fc5a5a5u, 2 * XSLOT / 4));
					tab[e] = inbox[e]; tab[NDIE + e] = dF;
					uint64_t row = dF + die * NCTA * 4; memcpy(&ctl[16 + 2 * e], &row, 8);
				}
				for (unsigned i = 0; i < 2 * XSLOT / 4; i++) { uint32_t bits = 0x7fc5a5a5u; memcpy(peer_ref + i, &bits, 4); }
				memcpy((char *)peer_ref + slot + base * 4, mixed + base, cols * 4);
				uint64_t own = dF; memcpy(&ctl[28], &own, 8); ctl[3] = die; ctl[4] = NDIE; ctl[8] = 17; ctl[9] = launch_width;
				CK(cuMemcpyHtoD(dCTL, ctl, sizeof ctl));
				CUdeviceptr dTAB = hcup_copy("HCUP peer table", tab, sizeof tab);
				desc_init(prog[0], OP_BF16HCUP, NCTA);
				desc_ptr(prog[0], 0x140, dW); desc_ptr(prog[0], 0x148, dD); desc_ptr(prog[0], 0x150, dU);
				desc_u32(prog[0], 0x158, 4 * cols); desc_u32(prog[0], 0x168, slot + base * 4);
				desc_ptr(prog[0], 0x170, dTAB); desc_ptr(prog[0], 0x178, dXH + base * 4);
				desc_ptr(prog[0], 0x180, dT); desc_u32(prog[0], 0x188, tn); prog[0][30] = 1;
				CUdeviceptr dP = hcup_copy("HCUP program", prog, sizeof prog);
				CK(cuMemsetD32(dU, 0x7fc5a5a5u, 4 * cols)); CK(cuMemsetD32(dT, 0x7fc5a5a5u, tn));
				unsigned nph = 1, ntok = 1; void *args[] = { &dP, &nph, &dCTL, &ntok };
				CK(cuLaunchKernel(fn, NCTA, 1, 1, 256, 1, 1, 0, 0, args, NULL)); CK(cuCtxSynchronize());
				printf("HCUP pattern=%u die=%u cols=%u base=%u tn=%u slot=%u shares=%s\n", pattern, die, cols, base, tn, slot, getenv("SPLIT_SHARES"));
				int bad = 0; CK(cuMemcpyDtoH(got, dT, tn * 4)); bad += cmpf("HCUP T", t, got, tn);
				CK(cuMemcpyDtoH(got, dU, ubytes)); bad += cmpf("HCUP up", local_up, got, 4 * cols);
				for (unsigned e = 0; e < NDIE; e++) { CK(cuMemcpyDtoH(got, inbox[e], 2 * XSLOT)); bad += cmpf("HCUP peer", peer_ref, got, 2 * XSLOT / 4); }
				bad += hcup_unchanged("weights", dW, local_w, wbytes) + hcup_unchanged("down", dD, down, tn * 4);
				bad += hcup_unchanged("XH", dXH, xh, 4 * COLS * 4) + hcup_unchanged("table", dTAB, tab, sizeof tab);
				bad += hcup_unchanged("program", dP, prog, sizeof prog) + test_guard_check();
				CK(cuMemcpyDtoH(ctl, dCTL, sizeof ctl)); CK(cuMemcpyDtoH(flags, dF, sizeof flags));
				bad += ctl[2] != 1 || ctl[1] != 1 || ctl[8] != 18 || ctl[11] != 0;
				for (unsigned i = 0; i < NDIE * NCTA; i++) bad += flags[i] != (i / NCTA == die ? 18u : UINT32_MAX);
				fails += bad; cases++; test_guard_free();
			}
			free(local_w); free(local_up);
		}
	}
	printf("HCUP cases=%u failures=%d reference=hc_silu_gpu,gemv_bf16_ref,hc_mix_gpu exact_bits=yes\n", cases, fails);
	free(weights); free(up); free(xh); free(mixed); free(got); free(peer_ref);
	return fails ? 3 : 0;
}

int main(void) {
	const char *width = getenv("DIE_LAUNCH_WIDTH");
	if (width && strcmp(width, "0") && strcmp(width, "1")) {
		fprintf(stderr, "DIE_LAUNCH_WIDTH must be 0 or 1\n"); return 2;
	}
	launch_width = width && !strcmp(width, "1");
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	const char *cubin = getenv("FLASH_CUBIN");
	CK(cuModuleLoad(&mod, cubin ? cubin : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	int nregs, maxb;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	printf("die: %d registers, %d CTAs/SM\n", nregs, maxb);
	if (maxb < 1) { printf("kernel cannot be resident\n"); return 2; }
	if (getenv("DIE_HCUP_TEST")) return hcup_test(fn);
	if (getenv("DIE_EXCHANGE_TEST")) return exchange_test(fn);

	/* ---- phase 0: embrow ---- */
	const int R = 64; unsigned id = 17;
	uint8_t *tab = malloc((size_t)R * 1100);
	for (size_t i = 0; i < (size_t)R * 1100; i++) tab[i] = (uint8_t)urand();
	for (int r = 0; r < R; r++) for (int b = 0; b < 10; b++) { uint16_t h = (uint16_t)(0x2c00 + (urand() % 0x1000)); memcpy(tab + (size_t)r * 1100 + 110 * b + 108, &h, 2); }
	float *emb_ref = malloc(4 * 2560), *emb_got = malloc(4 * 10240);
	dequant(GG_Q3_K, tab + (size_t)id * 1100, emb_ref, 2560);
	CUdeviceptr dE, dS; CK(cuMemAlloc(&dE, (size_t)R * 1100)); CK(cuMemAlloc(&dS, 4 * 10240));
	CK(cuMemcpyHtoD(dE, tab, (size_t)R * 1100));
	/* ---- phase 1: plegate ---- */
	float *kn = malloc(4 * 10240), *qn = malloc(4 * 10240), *v = malloc(4 * 2560), *g_ref = malloc(4 * 10240), *g_got = malloc(4 * 10240), gates_ref[4], gates_got[4];
	for (int i = 0; i < 10240; i++) { kn[i] = frand(); qn[i] = frand(); }
	for (int i = 0; i < 2560; i++) v[i] = 3.f * frand();
	ple_gate_gpu(kn, qn, v, g_ref, gates_ref);
	CUdeviceptr dKN, dQN, dV, dG, dGT;
	CK(cuMemAlloc(&dKN, 4 * 10240)); CK(cuMemAlloc(&dQN, 4 * 10240)); CK(cuMemAlloc(&dV, 4 * 2560)); CK(cuMemAlloc(&dG, 4 * 10240)); CK(cuMemAlloc(&dGT, 16));
	CK(cuMemcpyHtoD(dKN, kn, 4 * 10240)); CK(cuMemcpyHtoD(dQN, qn, 4 * 10240)); CK(cuMemcpyHtoD(dV, v, 4 * 2560));
	/* ---- phase 2: actq (2560) ---- */
	const int n1 = 2560;
	float *x1 = malloc(4 * n1), *as1 = malloc(4 * n1 / 32), *as1g = malloc(4 * n1 / 32);
	int16_t *q1 = malloc(2 * n1), *q1g = malloc(2 * n1);
	int32_t *s32_1 = malloc(4 * n1 / 32), *s16_1 = malloc(4 * n1 / 16), *s32_1g = malloc(4 * n1 / 32), *s16_1g = malloc(4 * n1 / 16);
	for (int i = 0; i < n1; i++) x1[i] = 5.f * frand();
	quant_act_gpu32(x1, n1, q1, s32_1, s16_1, as1);
	CUdeviceptr dX1, dQ1, dS32_1, dS16_1, dAS1;
	CK(cuMemAlloc(&dX1, 4 * n1)); CK(cuMemAlloc(&dQ1, 2 * n1)); CK(cuMemAlloc(&dS32_1, 4 * n1 / 32)); CK(cuMemAlloc(&dS16_1, 4 * n1 / 16)); CK(cuMemAlloc(&dAS1, 4 * n1 / 32));
	CK(cuMemcpyHtoD(dX1, x1, 4 * n1));
	/* ---- phase 3: ffnact (11 x 640) ---- */
	const int n2 = 640, E = 11; size_t N2 = (size_t)n2 * E;
	float *gg = malloc(4 * N2), *uu = malloc(4 * N2), *xx = malloc(4 * N2), *as2 = malloc(4 * N2 / 32), *as2g = malloc(4 * N2 / 32);
	int16_t *q2 = malloc(2 * N2), *q2g = malloc(2 * N2);
	int32_t *s32_2 = malloc(4 * N2 / 32), *s16_2 = malloc(4 * N2 / 16), *s32_2g = malloc(4 * N2 / 32), *s16_2g = malloc(4 * N2 / 16);
	for (size_t i = 0; i < N2; i++) { gg[i] = 4.f * frand(); uu[i] = 2.f * frand(); xx[i] = silu_gpu(gg[i]) * uu[i]; }
	for (int e = 0; e < E; e++) quant_act_gpu32(xx + (size_t)e * n2, n2, q2 + (size_t)e * n2, s32_2 + (size_t)e * n2 / 32, s16_2 + (size_t)e * n2 / 16, as2 + (size_t)e * n2 / 32);
	CUdeviceptr dGG, dUU, dQ2, dS32_2, dS16_2, dAS2;
	CK(cuMemAlloc(&dGG, 4 * N2)); CK(cuMemAlloc(&dUU, 4 * N2)); CK(cuMemAlloc(&dQ2, 2 * N2)); CK(cuMemAlloc(&dS32_2, 4 * N2 / 32)); CK(cuMemAlloc(&dS16_2, 4 * N2 / 16)); CK(cuMemAlloc(&dAS2, 4 * N2 / 32));
	CK(cuMemcpyHtoD(dGG, gg, 4 * N2)); CK(cuMemcpyHtoD(dUU, uu, 4 * N2));
	/* ---- phase 4: argmax ---- */
	const unsigned NL = 248320; float *lg = malloc(4 * NL);
	for (unsigned i = 0; i < NL; i++) lg[i] = 10.f * frand();
	unsigned where = urand() % NL; lg[where] = 20.f; lg[(where + 5) % NL] = 20.f;
	unsigned am_ref = 0; for (unsigned i = 1; i < NL; i++) if (lg[i] > lg[am_ref]) am_ref = i;
	CUdeviceptr dL, dOUT, dPART, dCNT;
	CK(cuMemAlloc(&dL, 4 * NL)); CK(cuMemAlloc(&dOUT, 8)); CK(cuMemAlloc(&dPART, 8 * 16)); CK(cuMemAlloc(&dCNT, 4));
	CK(cuMemcpyHtoD(dL, lg, 4 * NL)); CK(cuMemsetD32(dCNT, 0, 1));

	/* ---- the program ---- */
	uint32_t prog[5][32];
	CUdeviceptr dIDS; CK(cuMemAlloc(&dIDS, 16)); CK(cuMemcpyHtoD(dIDS, &id, 4));   /* embrow: the row is IDS[pos] */
	desc_init(prog[0], OP_EMBROW, 10); desc_ptr(prog[0], 0x140, dE); desc_ptr(prog[0], 0x148, dS); desc_u32(prog[0], 0x150, 0); desc_ptr(prog[0], 0x158, dIDS);
	desc_init(prog[1], OP_PLEGATE, 4); desc_ptr(prog[1], 0x140, dKN); desc_ptr(prog[1], 0x148, dQN); desc_ptr(prog[1], 0x150, dV); desc_ptr(prog[1], 0x158, dG); desc_ptr(prog[1], 0x160, dGT);
	desc_init(prog[2], OP_ACTQ2560, 1); desc_ptr(prog[2], 0x140, dX1); desc_ptr(prog[2], 0x148, dQ1); desc_ptr(prog[2], 0x150, dS32_1); desc_ptr(prog[2], 0x158, dS16_1); desc_ptr(prog[2], 0x160, dAS1); desc_u32(prog[2], 0x168, n1);
	desc_init(prog[3], OP_FFNACT, E); desc_ptr(prog[3], 0x140, dGG); desc_ptr(prog[3], 0x148, dUU); desc_ptr(prog[3], 0x150, dQ2); desc_ptr(prog[3], 0x158, dS32_2); desc_ptr(prog[3], 0x160, dS16_2); desc_ptr(prog[3], 0x168, dAS2); desc_u32(prog[3], 0x170, n2);
	desc_init(prog[4], OP_ARGMAX, 16); desc_ptr(prog[4], 0x140, dL); desc_ptr(prog[4], 0x148, dOUT); desc_ptr(prog[4], 0x150, dPART); desc_ptr(prog[4], 0x158, dCNT); desc_u32(prog[4], 0x160, NL); desc_u32(prog[4], 0x164, 16);
	if (getenv("DIE_OP0")) prog[0][0] = (uint32_t)atoi(getenv("DIE_OP0"));
	if (getenv("DIE_UNITS0")) prog[0][1] = (uint32_t)atoi(getenv("DIE_UNITS0"));
	if (getenv("DIE_ONLY_ACTQ")) memcpy(prog[0], prog[2], 128);
	CUdeviceptr dPROG, dCTL;
	CK(cuMemAlloc(&dPROG, sizeof prog)); CK(cuMemAlloc(&dCTL, 8192)); CK(cuMemsetD8(dCTL, 0, 8192));
	CK(cuMemcpyHtoD(dPROG, prog, sizeof prog)); CK(cuMemsetD32(dCTL, 0, 32));
	CK(cuMemcpyHtoD(dCTL + 0x24, &launch_width, sizeof launch_width));
	unsigned nph = getenv("DIE_NPH") ? (unsigned)atoi(getenv("DIE_NPH")) : getenv("DIE_ONLY_ACTQ") ? 1 : 5;
	{ float mx = 0; int am = -1; for (int i = 1216; i < 1248; i++) if (fabsf(x1[i]) > mx) { mx = fabsf(x1[i]); am = i; }
	  printf("group 38: max |x| %.9g at %d (lane %d of chunk 9, 8-lane half %s)\n", mx, am, (am - 1152) / 4, ((am - 1152) / 4) & 4 ? "20-23" : "16-19");
	  float m4[4] = { 0, 0, 0, 0 }; for (int i = 1216; i < 1248; i++) { int q = ((i - 1152) / 4 - 16) / 2; if (fabsf(x1[i]) > m4[q]) m4[q] = fabsf(x1[i]); }
	  printf("  lane-pair maxima: %.9g %.9g %.9g %.9g\n", m4[0], m4[1], m4[2], m4[3]); }
	unsigned ntok = 1; void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	int fails = 0;
	for (int run = 0; run < 2; run++) {
		CK(cuMemsetD32(dS, 0x7fc00000u, 10240)); CK(cuMemsetD32(dG, 0x7fc00000u, 10240)); CK(cuMemsetD32(dGT, 0x7fc00000u, 4));
		CK(cuMemsetD32(dQ1, 0x7fff7fffu, n1 / 2)); CK(cuMemsetD32(dAS1, 0x7fc00000u, n1 / 32));
		CK(cuMemsetD32(dQ2, 0x7fff7fffu, N2 / 2)); CK(cuMemsetD32(dAS2, 0x7fc00000u, N2 / 32));
		CK(cuMemsetD32(dOUT, 0xffffffffu, 2));
		CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		printf("run %d:\n", run);
		CK(cuMemcpyDtoH(emb_got, dS, 4 * 10240));
		for (int l = 0; l < 4; l++) { char nm[16]; snprintf(nm, sizeof nm, "emb.l%d", l); fails += cmpf(nm, emb_ref, emb_got + 2560 * l, 2560); }
		CK(cuMemcpyDtoH(g_got, dG, 4 * 10240)); CK(cuMemcpyDtoH(gates_got, dGT, 16));
		fails += cmpf("gates", gates_ref, gates_got, 4); fails += cmpf("G", g_ref, g_got, 10240);
		CK(cuMemcpyDtoH(q1g, dQ1, 2 * n1)); CK(cuMemcpyDtoH(s32_1g, dS32_1, 4 * n1 / 32)); CK(cuMemcpyDtoH(s16_1g, dS16_1, 4 * n1 / 16)); CK(cuMemcpyDtoH(as1g, dAS1, 4 * n1 / 32));
		int bq = 0, b32 = 0, b16 = 0, bs = 0;
		for (int i = 0; i < n1; i++) if (q1[i] != q1g[i]) { if (bq < 2) printf("  actq q[%d]: gpu %d ref %d\n", i, q1g[i], q1[i]); bq++; }
		for (int i = 0; i < n1 / 32; i++) { if (s32_1[i] != s32_1g[i]) b32++; if (memcmp(&as1[i], &as1g[i], 4)) { if (bs < 2) printf("  actq as[%d]: gpu %.9g ref %.9g\n", i, as1g[i], as1[i]); bs++; } }
		for (int i = 0; i < n1 / 16; i++) if (s16_1[i] != s16_1g[i]) b16++;
		int b = bq || b32 || b16 || bs;
		printf("  actq     %s (q %d, s32 %d, s16 %d, as %d differ)\n", b ? "MISMATCH" : "BIT-EXACT", bq, b32, b16, bs); fails += b;
		CK(cuMemcpyDtoH(q2g, dQ2, 2 * N2)); CK(cuMemcpyDtoH(s32_2g, dS32_2, 4 * N2 / 32)); CK(cuMemcpyDtoH(s16_2g, dS16_2, 4 * N2 / 16)); CK(cuMemcpyDtoH(as2g, dAS2, 4 * N2 / 32));
		b = memcmp(q2, q2g, 2 * N2) || memcmp(s32_2, s32_2g, 4 * N2 / 32) || memcmp(s16_2, s16_2g, 4 * N2 / 16) || memcmp(as2, as2g, 4 * N2 / 32);
		printf("  ffnact   %s\n", b ? "MISMATCH" : "BIT-EXACT"); fails += b;
		unsigned out[2]; CK(cuMemcpyDtoH(out, dOUT, 8));
		printf("  argmax   gpu %u ref %u -> %s\n", out[0], am_ref, out[0] == am_ref ? "OK" : "MISMATCH"); fails += out[0] != am_ref;
		unsigned ctl[2]; CK(cuMemcpyDtoH(ctl, dCTL, 8));
		printf("  barrier  ctr %u gen %u\n", ctl[0], ctl[1]);
	}
	return fails ? 3 : 0;
}
