/*
 * gemv_bf16_test: run the MaxAs gemv_bf16 kernel on a real BF16 tensor and compare
 * bit-for-bit with gemv_bf16_ref.
 *
 *   gemv_bf16_test SHARD1 [tensor] [iters]
 *
 * Default tensor blk.4.ffn_gate_inp.weight (router, 512 rows x 2560). Any BF16
 * tensor with K = 2560 or K = 320 works (e.g. blk.4.hc_attn_up.weight, 10240 x 320).
 * GEMV_GRID caps the grid (persistent-style looping).
 */
#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include "quant.h"
#include <cuda.h>
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint32_t rs = 4242;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: gemv_bf16_test SHARD1 [tensor] [iters]\n"); return 2; }
	const char *tname = argc > 2 ? argv[2] : "blk.4.ffn_gate_inp.weight";
	int iters = argc > 3 ? atoi(argv[3]) : 20;
	gguf_file g;
	if (gguf_open(&g, argv[1])) return 1;
	const gguf_tensor *t = gguf_find(&g, tname);
	if (!t || !t->data) { fprintf(stderr, "%s: not found\n", tname); return 1; }
	if (t->type != GG_BF16) { fprintf(stderr, "%s is %s, not BF16\n", tname, gguf_type_name(t->type)); return 1; }
	int K = (int)t->dims[0], rows = (int)(t->elems / K);
	if (K != 2560 && K != 320 && K != 10240) { fprintf(stderr, "kernels exist for K=2560, K=320 and K=10240 (as 4 slices), tensor has K=%d\n", K); return 1; }
	/* K=10240 runs as 4 ordered K=2560 column slices accumulating into y */
	int slices = K == 10240 ? 4 : 1, Ks = K / slices;
	int L = bf16_lanes_per_row(Ks), RPC = 256 / L;
	printf("%s: BF16 K=%d rows=%d (%.1f MB), %d lanes per row, %d slice(s)\n", tname, K, rows, (double)rows * K * 2 / 1e6, L, slices);

	const uint16_t *W = (const uint16_t *)t->data;
	float *x = malloc(sizeof(float) * K);
	for (int k = 0; k < K; k++) x[k] = frand() * (1.f + (k % 3));
	const char *pat = getenv("GEMV_PATTERN");
	if (pat && !strcmp(pat, "ones")) for (int k = 0; k < K; k++) x[k] = 1.f;
	float *yref = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
	double t0 = now();
	if (slices == 1) gemv_bf16_ref(W, rows, K, x, yref);
	else {
		uint16_t *Ws = malloc(sizeof(uint16_t) * (size_t)rows * Ks);
		float *p = malloc(sizeof(float) * rows);
		for (int r = 0; r < rows; r++) yref[r] = 0.f;
		for (int s = 0; s < slices; s++) {
			for (int r = 0; r < rows; r++) memcpy(Ws + (size_t)r * Ks, W + (size_t)r * K + s * Ks, sizeof(uint16_t) * Ks);
			gemv_bf16_ref(Ws, rows, Ks, x + s * Ks, p);
			for (int r = 0; r < rows; r++) yref[r] = yref[r] + p[r];
		}
		free(Ws); free(p);
	}
	double t1 = now();
	printf("reference %.1f ms\n", (t1 - t0) * 1e3);

	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	char cubin[64];
	if (slices == 1) snprintf(cubin, sizeof cubin, "gemv_bf16_k%d.cubin", K);
	else snprintf(cubin, sizeof cubin, "gemv_bf16_hc.cubin");
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "gemv_bf16"));
	int nregs, maxb;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	CUdeviceptr dW, dX, dY, dD;
	size_t wbytes = (size_t)rows * K * 2;
	CK(cuMemAlloc(&dW, wbytes));
	CK(cuMemAlloc(&dX, sizeof(float) * K));
	CK(cuMemAlloc(&dY, sizeof(float) * rows));
	CK(cuMemAlloc(&dD, 4096));
	CK(cuMemcpyHtoD(dW, W, wbytes));
	CK(cuMemcpyHtoD(dX, x, sizeof(float) * K));
	unsigned urows = (unsigned)rows;
	unsigned grid = (urows + RPC - 1) / RPC;
	if (getenv("GEMV_GRID") && (unsigned)atoi(getenv("GEMV_GRID")) < grid) grid = (unsigned)atoi(getenv("GEMV_GRID"));
	printf("kernel: %d registers, %d CTAs/SM; grid %u x 256, %u rows per CTA iteration\n", nregs, maxb, grid, RPC);
	CUdeviceptr sW[4], sX[4];
	void *params[4][5];
	for (int s = 0; s < slices; s++) {
		sW[s] = dW + (size_t)s * Ks * 2;
		sX[s] = dX + (size_t)s * Ks * 4;
		params[s][0] = &sW[s]; params[s][1] = &sX[s]; params[s][2] = &dY; params[s][3] = &urows; params[s][4] = &dD;
	}
	/* the accumulating slices start from y = 0; a single slice overwrites */
	if (slices == 1) CK(cuMemsetD32(dY, 0x7fc00000u, rows)); else CK(cuMemsetD32(dY, 0, rows));
	for (int s = 0; s < slices; s++) CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params[s], NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(y, dY, sizeof(float) * rows));
	int bad = 0;
	double maxabs = 0;
	for (int r = 0; r < rows; r++) {
		if (memcmp(&y[r], &yref[r], 4)) { if (bad < 6) printf("  row %d: gpu % .9g ref % .9g\n", r, y[r], yref[r]); bad++; }
		double a = fabs((double)y[r] - yref[r]);
		if (a > maxabs || isnan(a)) maxabs = a;
	}
	printf("%s: %d of %d rows differ (max abs %.3e)\n", bad ? "MISMATCH" : "BIT-EXACT", bad, rows, maxabs);
	if (iters > 0) {
		CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params[0], NULL));
		CK(cuCtxSynchronize());
		double ta = now();
		for (int i = 0; i < iters; i++)
			for (int s = 0; s < slices; s++) CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params[s], NULL));
		CK(cuCtxSynchronize());
		double tb = now();
		double per = (tb - ta) / iters;
		printf("%.1f us per matvec, %.1f GB/s of weights\n", per * 1e6, (double)wbytes / per / 1e9);
	}
	return bad ? 3 : 0;
}
