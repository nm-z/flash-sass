/* ffnact_test: SiLU(g) * u + per-32 quantiser (one warp per expert) against
 * silu_gpu / quant_act_gpu32, bit for bit. usage: ffnact_test [n] [experts] */
#define _POSIX_C_SOURCE 200809L
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

static uint32_t rs = 19;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	int n = argc > 1 ? atoi(argv[1]) : 640, E = argc > 2 ? atoi(argv[2]) : 11;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	char cubin[64];
	snprintf(cubin, sizeof cubin, "ffnact_k%d.cubin", n);
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "ffnact"));
	int nregs; CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	size_t N = (size_t)n * E;
	CUdeviceptr dG, dU, dQ, dS32, dS16, dSC;
	CK(cuMemAlloc(&dG, 4 * N)); CK(cuMemAlloc(&dU, 4 * N)); CK(cuMemAlloc(&dQ, 2 * N));
	CK(cuMemAlloc(&dS32, 4 * N / 32 + 64)); CK(cuMemAlloc(&dS16, 4 * N / 16 + 64)); CK(cuMemAlloc(&dSC, 4 * N / 32 + 64));
	float *g = malloc(4 * N), *u = malloc(4 * N), *x = malloc(4 * N);
	int16_t *q = malloc(2 * N), *qg = malloc(2 * N);
	int32_t *s32 = malloc(4 * N / 32), *s16 = malloc(4 * N / 16), *s32g = malloc(4 * N / 32), *s16g = malloc(4 * N / 16);
	float *as = malloc(4 * N / 32), *asg = malloc(4 * N / 32);
	const float mags[] = { 1.f, 6.f, 0.05f, 40.f, 0.f };
	int fails = 0;
	unsigned un = (unsigned)n;
	for (unsigned m = 0; m < sizeof mags / sizeof mags[0]; m++) {
		for (size_t i = 0; i < N; i++) { g[i] = mags[m] * frand() * (1.f + (i % 5) * 0.5f); u[i] = 2.f * frand(); }
		if (m == 3) g[7] = 100.f, g[9] = -100.f;    /* saturated gates */
		for (size_t i = 0; i < N; i++) x[i] = silu_gpu(g[i]) * u[i];
		for (int e = 0; e < E; e++) quant_act_gpu32(x + (size_t)e * n, n, q + (size_t)e * n, s32 + (size_t)e * n / 32, s16 + (size_t)e * n / 16, as + (size_t)e * n / 32);
		CK(cuMemcpyHtoD(dG, g, 4 * N)); CK(cuMemcpyHtoD(dU, u, 4 * N));
		CK(cuMemsetD32(dQ, 0x7fff7fffu, N / 2));
		CK(cuMemsetD32(dS32, 0x7fffffffu, N / 32)); CK(cuMemsetD32(dS16, 0x7fffffffu, N / 16)); CK(cuMemsetD32(dSC, 0x7fc00000u, N / 32));
		void *params[] = { &dG, &dU, &dQ, &dS32, &dS16, &dSC, &un };
		CK(cuLaunchKernel(fn, E, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(qg, dQ, 2 * N)); CK(cuMemcpyDtoH(s32g, dS32, 4 * N / 32)); CK(cuMemcpyDtoH(s16g, dS16, 4 * N / 16)); CK(cuMemcpyDtoH(asg, dSC, 4 * N / 32));
		int bq = 0, b32 = 0, b16 = 0, bs = 0;
		for (size_t i = 0; i < N; i++) if (q[i] != qg[i]) { if (bq < 3) printf("  q[%zu]: gpu %d ref %d (g %g u %g x %g)\n", i, qg[i], q[i], g[i], u[i], x[i]); bq++; }
		for (size_t i = 0; i < N / 32; i++) if (s32[i] != s32g[i]) b32++;
		for (size_t i = 0; i < N / 16; i++) if (s16[i] != s16g[i]) b16++;
		for (size_t i = 0; i < N / 32; i++) if (memcmp(&as[i], &asg[i], 4)) { if (bs < 3) printf("  as[%zu]: gpu %.9g ref %.9g\n", i, asg[i], as[i]); bs++; }
		printf("n=%d x%d mag %-5g: %s (codes %d/%zu differ, s32 %d, s16 %d, scales %d/%zu) [%d regs]\n", n, E, mags[m],
		       (bq || b32 || b16 || bs) ? "MISMATCH" : "BIT-EXACT", bq, N, b32, b16, bs, N / 32, nregs);
		if (bq || b32 || b16 || bs) fails++;
	}
	return fails ? 3 : 0;
}
