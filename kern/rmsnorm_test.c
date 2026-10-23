/* rmsnorm_test N: the fp32 RMSNorm kernel against rmsnorm_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 13;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	int n = argc > 1 ? atoi(argv[1]) : 2560;
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	char cubin[64];
	snprintf(cubin, sizeof cubin, "rmsnorm_k%d.cubin", n);
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "rmsnorm"));
	CUdeviceptr dX, dW, dY;
	CK(cuMemAlloc(&dX, 4 * n)); CK(cuMemAlloc(&dW, 4 * n)); CK(cuMemAlloc(&dY, 4 * n));
	float *x = malloc(4 * n), *w = malloc(4 * n), *y = malloc(4 * n), *yg = malloc(4 * n);
	const float mags[] = { 1.f, 1e-3f, 37.5f, 1e4f, 0.f, 2.5f };
	int fails = 0;
	for (unsigned m = 0; m < sizeof mags / sizeof mags[0]; m++) {
		for (int i = 0; i < n; i++) { x[i] = mags[m] * frand() * (1.f + (i % 7) * 0.25f); w[i] = 1.f + 0.5f * frand(); }
		rmsnorm_gpu(x, w, n, eps, y);
		CK(cuMemcpyHtoD(dX, x, 4 * n)); CK(cuMemcpyHtoD(dW, w, 4 * n)); CK(cuMemsetD32(dY, 0x7fc00000u, n));
		void *params[] = { &dX, &dW, &dY, &eps };
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(yg, dY, 4 * n));
		int by = 0;
		for (int i = 0; i < n; i++) if (memcmp(&y[i], &yg[i], 4)) { if (by < 3) printf("  y[%d]: gpu %.9g ref %.9g\n", i, yg[i], y[i]); by++; }
		printf("n=%d mag %-8g: %s (%d of %d differ)\n", n, mags[m], by ? "MISMATCH" : "BIT-EXACT", by, n);
		if (by) fails++;
	}
	return fails ? 3 : 0;
}
