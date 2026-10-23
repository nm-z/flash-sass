/* pleconv_test: PLE dilated conv + SiLU + residual + stream add over a sequence of
 * positions (ring state carried on the device) against ple_conv_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "ple.h"
#include "quant.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 31;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static uint16_t f2h(float f) {       /* round-to-nearest fp16 for test weights (normal range only) */
	uint32_t u; memcpy(&u, &f, 4);
	uint32_t s = (u >> 16) & 0x8000; int e = (int)((u >> 23) & 0xff) - 127 + 15; uint32_t m = u & 0x7fffff;
	if (e <= 0) return (uint16_t)s;
	uint32_t r = (m >> 13) + ((m >> 12) & 1);
	return (uint16_t)(s | (((uint32_t)e << 10) + r));
}

int main(void) {
	const int n = 10240, T = 14;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "pleconv.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "pleconv"));
	float *x = malloc(4 * n), *g = malloc(4 * n), *s = malloc(4 * n), *sg = malloc(4 * n), *w = malloc(4 * n * 4);
	uint16_t *wh = malloc(2 * n * 4);
	float (*state)[10240] = calloc(10, 4 * n);
	for (int i = 0; i < 4 * n; i++) { float f = 0.5f * frand(); wh[i] = f2h(f); w[i] = fp16_to_fp32(wh[i]); }
	for (int i = 0; i < n; i++) s[i] = 2.f * frand();
	CUdeviceptr dX, dST, dW, dG, dS;
	CK(cuMemAlloc(&dX, 4 * n)); CK(cuMemAlloc(&dST, 10 * 4 * n)); CK(cuMemAlloc(&dW, 2 * n * 4)); CK(cuMemAlloc(&dG, 4 * n)); CK(cuMemAlloc(&dS, 4 * n));
	CK(cuMemsetD32(dST, 0x7fc00000u, 10 * n));       /* unread slots may hold anything */
	CK(cuMemcpyHtoD(dW, wh, 2 * n * 4)); CK(cuMemcpyHtoD(dS, s, 4 * n));
	int fails = 0;
	for (unsigned t = 0; t < (unsigned)T; t++) {
		for (int i = 0; i < n; i++) { x[i] = 3.f * frand(); g[i] = frand(); }
		ple_conv_gpu(x, state, w, t, g, s);
		CK(cuMemcpyHtoD(dX, x, 4 * n)); CK(cuMemcpyHtoD(dG, g, 4 * n));
		void *params[] = { &dX, &dST, &dW, &dG, &dS, &t };
		CK(cuLaunchKernel(fn, n / 1024, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(sg, dS, 4 * n));
		{ static float ring[10][10240]; CK(cuMemcpyDtoH(ring, dST, 10 * 4 * n)); int bs = 0, nanslots = 0;
		  for (int i = 0; i < n; i++) if (memcmp(&ring[t % 10][i], &x[i], 4)) bs++;
		  for (int k = 0; k < 10; k++) if (ring[k][0] != ring[k][0]) nanslots |= 1 << k;
		  printf("  ring slot %u: %d of %d differ from x(t); NaN slots mask 0x%03x\n", t % 10, bs, n, nanslots); }
		int bad = 0;
		for (int i = 0; i < n; i++) if (memcmp(&s[i], &sg[i], 4)) { if (bad < 3) printf("  t %u S[%d]: gpu %.9g ref %.9g\n", t, i, sg[i], s[i]); bad++; }
		printf("t=%2u: %s (%d of %d differ)\n", t, bad ? "MISMATCH" : "BIT-EXACT", bad, n);
		if (bad) fails++;
	}
	return fails ? 3 : 0;
}
