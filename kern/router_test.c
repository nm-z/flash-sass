/* router_test: the MoE router kernel against router_gpu, bit for bit, on random logits (with ties). */
#define _POSIX_C_SOURCE 200809L
#include "router.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 37;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "router.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "router"));
	CUdeviceptr dR, dSH, dI, dW, dS;
	CK(cuMemAlloc(&dR, 2048)); CK(cuMemAlloc(&dSH, 64)); CK(cuMemAlloc(&dI, 64)); CK(cuMemAlloc(&dW, 64)); CK(cuMemAlloc(&dS, 64));
	float r[512], w[10], wg[10], s, sg, sh;
	unsigned idx[10], idxg[10];
	int fails = 0;
	const char *ntr = getenv("ROUTER_TRIALS"); int trials = ntr ? atoi(ntr) : 6; if (trials < 6) trials = 6;
	for (int trial = 0; trial < trials; trial++) {
		for (int i = 0; i < 512; i++) r[i] = 5.f * frand();
		if (trial % 6 >= 3) { for (int i = 0; i < 512; i++) r[i] = (float)((int)(r[i] * 2)) * 0.5f; }   /* many ties */
		sh = 3.f * frand();
		router_gpu(r, sh, idx, w, &s);
		CK(cuMemcpyHtoD(dR, r, 2048)); CK(cuMemcpyHtoD(dSH, &sh, 4));
		CK(cuMemsetD32(dI, 0xffffffffu, 16)); CK(cuMemsetD32(dW, 0x7fc00000u, 16)); CK(cuMemsetD32(dS, 0x7fc00000u, 16));
		void *params[] = { &dR, &dSH, &dI, &dW, &dS };
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(idxg, dI, 40)); CK(cuMemcpyDtoH(wg, dW, 40)); CK(cuMemcpyDtoH(&sg, dS, 4));
		int bad = memcmp(idx, idxg, 40) != 0 || memcmp(w, wg, 40) != 0 || memcmp(&s, &sg, 4) != 0;
		if (bad || trial < 6) printf("trial %d%s: %s  idx", trial, trial % 6 >= 3 ? " (ties)" : "", bad ? "MISMATCH" : "BIT-EXACT");
		if (bad || trial < 6) { for (int j = 0; j < 10; j++) printf(" %u%s", idxg[j], idxg[j] == idx[j] ? "" : "!");
		printf("  w0 %.6g/%.6g s %.6g/%.6g\n", wg[0], w[0], sg, s); }
		if (bad) fails++;
	}
	printf("%d trials, %d mismatches\n", trials, fails);
	return fails ? 3 : 0;
}
