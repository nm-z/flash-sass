/* conv1d_test: conv1d + SiLU state step against conv1d_silu_gpu, bit for bit, over 4 consecutive positions. */
#define _POSIX_C_SOURCE 200809L
#include "deltanet.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 29;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	const int n = 10240;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "conv1d.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "conv1d"));
	float *x = malloc(4 * n), *w = malloc(16 * n), *state = malloc(16 * n), *stateg = malloc(16 * n), *u = malloc(4 * n), *ug = malloc(4 * n);
	for (int i = 0; i < 4 * n; i++) { w[i] = frand(); state[i] = 3.f * frand(); }
	CUdeviceptr dX, dS, dW, dU;
	CK(cuMemAlloc(&dX, 4 * n)); CK(cuMemAlloc(&dS, 16 * n)); CK(cuMemAlloc(&dW, 16 * n)); CK(cuMemAlloc(&dU, 4 * n));
	CK(cuMemcpyHtoD(dS, state, 16 * n)); CK(cuMemcpyHtoD(dW, w, 16 * n));
	int fails = 0;
	unsigned un = (unsigned)n;
	for (unsigned t = 5; t < 9; t++) {
		for (int i = 0; i < n; i++) x[i] = 4.f * frand();
		conv1d_silu_gpu(x, state, w, t, u, n);
		CK(cuMemcpyHtoD(dX, x, 4 * n)); CK(cuMemsetD32(dU, 0x7fc00000u, n));
		void *params[] = { &dX, &dS, &dW, &dU, &t, &un };
		CK(cuLaunchKernel(fn, n / 1024, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ug, dU, 4 * n)); CK(cuMemcpyDtoH(stateg, dS, 16 * n));
		int bu = 0, bs = 0;
		for (int i = 0; i < n; i++) if (memcmp(&u[i], &ug[i], 4)) { if (bu < 3) printf("  u[%d]: gpu %.9g ref %.9g\n", i, ug[i], u[i]); bu++; }
		for (int i = 0; i < 4 * n; i++) if (memcmp(&state[i], &stateg[i], 4)) bs++;
		printf("t=%u: %s (u %d of %d differ, state %d of %d differ)\n", t, (bu || bs) ? "MISMATCH" : "BIT-EXACT", bu, n, bs, 4 * n);
		if (bu || bs) fails++;
	}
	return fails ? 3 : 0;
}
