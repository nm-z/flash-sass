/* sample_test: topk.sass + sample.sass inside the persistent kernel on one die. A program of six
 * TOPK phases (six synthetic "dies" worth of logits, N each) and one SAMPLE phase runs once per
 * trial; the lists and the token are compared bit for bit with ref/sample.c. */
#define _POSIX_C_SOURCE 200809L
#include "sample.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
#include "dieops.h"

static uint32_t rs = 12345;
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (float)(urand() & 0xffff) / 32768.f - 1.f; }
static void desc_init(uint32_t *d, int op, unsigned units) { memset(d, 0, 128); d[0] = (uint32_t)op; d[1] = units; }
static void desc_ptr(uint32_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d[w] = (uint32_t)p; d[w + 1] = (uint32_t)(p >> 32); }
static void desc_u32(uint32_t *d, int addr, uint32_t v) { d[2 + (addr - 0x140) / 4] = v; }
static void desc_f32(uint32_t *d, int addr, float v) { memcpy(&d[2 + (addr - 0x140) / 4], &v, 4); }

int main(int argc, char **argv) {
	int trials = argc > 1 ? atoi(argv[1]) : 60;
	const unsigned N = 41387, ND = 6, NPH = ND + 1;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "die.cubin")); CK(cuModuleGetFunction(&fn, mod, "die"));
	float *lg = malloc(4 * (size_t)N * ND);
	CUdeviceptr dX, dPAIRS, dPART, dCNT, dBASES, dRND, dIDS, dTOKS, dEND, dPROG, dCTL;
	CK(cuMemAlloc(&dX, 4 * (size_t)N * ND)); CK(cuMemAlloc(&dPAIRS, 256 * ND)); CK(cuMemAlloc(&dPART, 16 * 256)); CK(cuMemAlloc(&dCNT, 4));
	CK(cuMemAlloc(&dBASES, 4 * ND)); CK(cuMemAlloc(&dRND, 4 * 4096)); CK(cuMemAlloc(&dIDS, 4 * 4096)); CK(cuMemAlloc(&dTOKS, 4 * 4096)); CK(cuMemAlloc(&dEND, 4));
	CK(cuMemAlloc(&dPROG, 128 * 16)); CK(cuMemAlloc(&dCTL, 8192)); CK(cuMemsetD8(dCTL, 0, 8192)); CK(cuMemsetD32(dCNT, 0, 1));
	uint32_t bases[6]; for (unsigned d = 0; d < ND; d++) bases[d] = d * N;
	CK(cuMemcpyHtoD(dBASES, bases, 4 * ND));
	const int Ks[] = { 1, 2, 5, 20, 32, 20, 3 }; const float tps[] = { 1.0f, 0.95f, 0.5f, 0.05f }; const float its[] = { 1.0f, 0.5f, 2.0f, 1.25f };
	int fails = 0, ends = 0;
	for (int t = 0; t < trials; t++) {
		int kind = t % 5, K = Ks[(t / 5) % 7], K2 = K; float topp = tps[(t / 3) % 4], invT = its[(t / 2) % 4];
		if (t < 4) { K2 = 20; topp = 0.95f; invT = 1.0f; }
		K = K2;
		for (size_t i = 0; i < (size_t)N * ND; i++) {
			switch (kind) {
			case 0: lg[i] = 10.f * frand(); break;
			case 1: lg[i] = frand(); break;
			case 2: lg[i] = 1.5f * (float)(urand() % 4); break;
			case 3: lg[i] = 0.01f * frand(); break;
			default: lg[i] = 6.f * frand(); break;
			}
		}
		if (kind == 1) for (unsigned d = 0; d < ND; d++) { lg[(size_t)d * N + urand() % N] = 30.f - d; lg[(size_t)d * N + urand() % N] = 30.f - d; }
		if (kind == 4) for (unsigned d = 0; d < ND; d++) { lg[(size_t)d * N + 7 * (d + 1)] = 9.f; lg[(size_t)d * N + N - 1 - d] = 9.f; }
		float u = (float)(urand() & 0xffffff) / 16777216.f;
		CK(cuMemcpyHtoD(dX, lg, 4 * (size_t)N * ND));
		CK(cuMemcpyHtoD(dRND + 4 * t, &u, 4));
		/* twin */
		uint32_t ti[6][32]; float tv[6][32];
		for (unsigned d = 0; d < ND; d++) topk_gpu(lg + (size_t)d * N, N, K, ti[d], tv[d]);
		uint32_t want = sample_gpu(ti, tv, bases, (int)ND, K, topp, invT, u);
		uint32_t stop0 = (t % 4 == 0) ? want : 0xffffffffu;
		/* program */
		uint32_t prog[NPH][32];
		for (unsigned d = 0; d < ND; d++) {
			desc_init(prog[d], OP_TOPK, 16);
			desc_ptr(prog[d], 0x140, dX + 4 * (CUdeviceptr)d * N); desc_ptr(prog[d], 0x148, dPAIRS + 256 * d); desc_ptr(prog[d], 0x150, dPART); desc_ptr(prog[d], 0x158, dCNT);
			desc_u32(prog[d], 0x160, N); desc_u32(prog[d], 0x164, 16); desc_u32(prog[d], 0x168, (uint32_t)K);
		}
		desc_init(prog[ND], OP_SAMPLE, 1);
		desc_ptr(prog[ND], 0x140, dPAIRS); desc_ptr(prog[ND], 0x148, dBASES); desc_u32(prog[ND], 0x150, ND); desc_u32(prog[ND], 0x154, (uint32_t)K);
		desc_f32(prog[ND], 0x158, topp); desc_f32(prog[ND], 0x15c, invT);
		desc_ptr(prog[ND], 0x160, dIDS); desc_ptr(prog[ND], 0x168, dTOKS); desc_u32(prog[ND], 0x170, (uint32_t)t); desc_u32(prog[ND], 0x174, 0);
		desc_u32(prog[ND], 0x178, stop0); desc_u32(prog[ND], 0x17c, 0xffffffffu); desc_ptr(prog[ND], 0x180, dEND); desc_ptr(prog[ND], 0x188, dRND);
		CK(cuMemcpyHtoD(dPROG, prog, sizeof prog)); CK(cuMemsetD32(dCTL, 0, 32)); CK(cuMemsetD32(dEND, 0, 1));
		CK(cuMemsetD32(dPAIRS, 0xdeadbeefu, 64 * ND));
		unsigned nph = NPH, ntok = 1; void *params[] = { &dPROG, &nph, &dCTL, &ntok };
		CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		uint32_t pairs[6][64]; CK(cuMemcpyDtoH(pairs, dPAIRS, 256 * ND));
		int badl = 0;
		for (unsigned d = 0; d < ND; d++)
			for (int r = 0; r < 32; r++) {
				uint32_t wv; memcpy(&wv, &tv[d][r], 4);
				if (pairs[d][2 * r] != ti[d][r] || pairs[d][2 * r + 1] != wv) { if (badl < 3) printf("  t%d die %u rank %d: gpu (%u, %08x) ref (%u, %08x)\n", t, d, r, pairs[d][2 * r], pairs[d][2 * r + 1], ti[d][r], wv); badl++; }
			}
		uint32_t tok = 0, tnext = 0, endf = 0;
		CK(cuMemcpyDtoH(&tok, dTOKS + 4 * t, 4)); CK(cuMemcpyDtoH(&tnext, dIDS + 4 * (t + 1), 4)); CK(cuMemcpyDtoH(&endf, dEND, 4));
		int badt = tok != want || tnext != want, bade = endf != (stop0 == want ? 0x80000000u : 0u);
		ends += endf != 0;
		printf("trial %2d: kind %d K %2d topp %.2f invT %.2f u %.4f -> token %u (twin %u): lists %s, token %s, end %s\n", t, kind, K, topp, invT, u, tok, want,
		       badl ? "MISMATCH" : "ok", badt ? "MISMATCH" : "ok", bade ? "MISMATCH" : "ok");
		fails += (badl != 0) + badt + bade;
	}
	printf("%s: %d trials, %d end flags, %d failures\n", fails ? "FAIL" : "PASS", trials, ends, fails);
	return fails != 0;
}
