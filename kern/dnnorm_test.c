/* dnnorm_test: the DeltaNet readout norm + gate against dnnorm_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "deltanet.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 61;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "dnnorm.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "dnnorm"));
	float *o = malloc(6144 * 4), *g = malloc(6144 * 4), *y = malloc(6144 * 4), *yg = malloc(6144 * 4), w[128];
	CUdeviceptr dO, dW, dG, dY;
	CK(cuMemAlloc(&dO, 6144 * 4)); CK(cuMemAlloc(&dW, 512)); CK(cuMemAlloc(&dG, 6144 * 4)); CK(cuMemAlloc(&dY, 6144 * 4));
	int fails = 0;
	for (int trial = 0; trial < 3; trial++) {
		float sc = trial == 1 ? 1e-3f : trial == 2 ? 30.f : 1.f;
		for (int i = 0; i < 6144; i++) { o[i] = sc * frand(); g[i] = 6.f * frand(); }
		for (int i = 0; i < 128; i++) w[i] = 1.f + 0.5f * frand();
		dnnorm_gpu(o, w, g, eps, y);
		CK(cuMemcpyHtoD(dO, o, 6144 * 4)); CK(cuMemcpyHtoD(dW, w, 512)); CK(cuMemcpyHtoD(dG, g, 6144 * 4)); CK(cuMemsetD32(dY, 0x7fc00000u, 6144));
		void *params[] = { &dO, &dW, &dG, &dY, &eps };
		CK(cuLaunchKernel(fn, 48, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(yg, dY, 6144 * 4));
		int bad = 0;
		for (int i = 0; i < 6144; i++) if (memcmp(&y[i], &yg[i], 4)) { if (bad < 3) printf("  y[%d]: gpu %.9g ref %.9g\n", i, yg[i], y[i]); bad++; }
		printf("trial %d: %s (%d of 6144 differ)\n", trial, bad ? "MISMATCH" : "BIT-EXACT", bad);
		if (bad) fails++;
	}
	return fails ? 3 : 0;
}
