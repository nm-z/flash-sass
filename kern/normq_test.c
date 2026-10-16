/*
 * normq_test: the fused RMSNorm + quantiser kernel against rmsnorm_quant_gpu, bit
 * for bit (normalised vector, codes, sums, scale) on random inputs.
 */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 11;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	int n = 2560;
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "normq.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "normq"));
	CUdeviceptr dX, dW, dY, dQ, dS32, dS16, dSC;
	CK(cuMemAlloc(&dX, 4 * n)); CK(cuMemAlloc(&dW, 4 * n)); CK(cuMemAlloc(&dY, 4 * n)); CK(cuMemAlloc(&dQ, 2 * n));
	CK(cuMemAlloc(&dS32, 4 * n / 32 + 64)); CK(cuMemAlloc(&dS16, 4 * n / 16 + 64)); CK(cuMemAlloc(&dSC, 256));
	float *x = malloc(4 * n), *w = malloc(4 * n), *y = malloc(4 * n), *yg = malloc(4 * n);
	int16_t *q = malloc(2 * n), *qg = malloc(2 * n);
	int32_t *s32 = malloc(4 * n / 32), *s16 = malloc(4 * n / 16), *s32g = malloc(4 * n / 32), *s16g = malloc(4 * n / 16);
	const float mags[] = { 1.f, 1e-3f, 37.5f, 1e4f, 0.f, 2.5f };
	int fails = 0;
	for (unsigned m = 0; m < sizeof mags / sizeof mags[0]; m++) {
		for (int i = 0; i < n; i++) { x[i] = mags[m] * frand() * (1.f + (i % 7) * 0.25f); w[i] = 1.f + 0.5f * frand(); }
		if (m == 5) x[n - 1] = 1e4f;
		float xs = rmsnorm_quant_gpu(x, w, n, eps, y, q, s32, s16);
		CK(cuMemcpyHtoD(dX, x, 4 * n)); CK(cuMemcpyHtoD(dW, w, 4 * n));
		CK(cuMemsetD32(dY, 0x7fc00000u, n)); CK(cuMemsetD32(dQ, 0x7fff7fffu, n / 2));
		CK(cuMemsetD32(dS32, 0x7fffffffu, n / 32)); CK(cuMemsetD32(dS16, 0x7fffffffu, n / 16)); CK(cuMemsetD32(dSC, 0x7fc00000u, 64));
		void *params[] = { &dX, &dW, &dY, &dQ, &dS32, &dS16, &dSC, &eps };
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		float xsg;
		CK(cuMemcpyDtoH(yg, dY, 4 * n)); CK(cuMemcpyDtoH(qg, dQ, 2 * n)); CK(cuMemcpyDtoH(s32g, dS32, 4 * n / 32));
		CK(cuMemcpyDtoH(s16g, dS16, 4 * n / 16)); CK(cuMemcpyDtoH(&xsg, dSC, 4));
		int by = 0, bq = 0, b32 = 0, b16 = 0;
		for (int i = 0; i < n; i++) if (memcmp(&y[i], &yg[i], 4)) { if (by < 3) printf("  y[%d]: gpu %.9g ref %.9g\n", i, yg[i], y[i]); by++; }
		for (int i = 0; i < n; i++) if (q[i] != qg[i]) { if (bq < 3) printf("  q[%d]: gpu %d ref %d\n", i, qg[i], q[i]); bq++; }
		for (int i = 0; i < n / 32; i++) if (s32[i] != s32g[i]) b32++;
		for (int i = 0; i < n / 16; i++) if (s16[i] != s16g[i]) b16++;
		int bs = memcmp(&xs, &xsg, 4) != 0;
		int bad = by || bq || b32 || b16 || bs;
		printf("mag %-8g: %s (y %d, codes %d, s32 %d, s16 %d differ; scale gpu %.9g ref %.9g)\n", mags[m], bad ? "MISMATCH" : "BIT-EXACT", by, bq, b32, b16, xsg, xs);
		if (bad) fails++;
	}
	return fails ? 3 : 0;
}
