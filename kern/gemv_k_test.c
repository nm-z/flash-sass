/*
 * gemv_k_test: K-quant GEMV kernels on real tensors, bit-exact against the C
 * reference in the kernel's accumulation order. Currently Q4_K.
 *
 *   gemv_k_test SHARD1 [tensor] [iters]
 *
 * Default tensor blk.3.attn_output.weight (Q4_K, K=6144, 2560 rows). K=2560
 * examples: blk.7.ssm_out... use any Q4_K tensor with K = 2560 or 6144, e.g.
 * blk.12.attn_qkv.weight. GEMV_GRID caps the grid (persistent-style looping).
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
static uint32_t rs = 99;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: gemv_k_test SHARD1 [tensor] [iters]\n"); return 2; }
	const char *tname = argc > 2 ? argv[2] : "blk.3.attn_output.weight";
	int iters = argc > 3 ? atoi(argv[3]) : 20;
	gguf_file g;
	if (gguf_open(&g, argv[1])) return 1;
	const gguf_tensor *t = gguf_find(&g, tname);
	if (!t || !t->data) { fprintf(stderr, "%s: not found\n", tname); return 1; }
	if (t->type != GG_Q4_K) { fprintf(stderr, "%s is %s; only Q4_K so far\n", tname, gguf_type_name(t->type)); return 1; }
	int K = (int)t->dims[0], rows = (int)(t->elems / K);
	if (K != 2560 && K != 6144) { fprintf(stderr, "kernels exist for K=2560 and K=6144, tensor has K=%d\n", K); return 1; }
	int L = q4_k_lanes_per_row(K), RPC = 256 / L, stride = q4_k_tile_stride(K);
	printf("%s: Q4_K K=%d rows=%d (%.1f MB), %d lanes per row, tile stride %d\n", tname, K, rows, (double)rows * stride / 1e6, L, stride);

	/* repack and check the tile against the standard dequantiser on a few rows */
	uint8_t *tile = malloc((size_t)stride * rows);
	double t0 = now();
	repack_q4_k(t->data, rows, K, tile);
	double t1 = now();
	{
		float *a = malloc(sizeof(float) * K), *b = malloc(sizeof(float) * K);
		int bad = 0, B = K / 256;
		for (int r = 0; r < rows && r < 7; r += 3) {
			dequant(GG_Q4_K, t->data + (size_t)r * B * 144, a, K);
			const uint8_t *hdr = tile + (size_t)r * stride, *codes = hdr + 16 * B;
			for (int e = 0; e < K; e++) {
				int s = e / 32, blk = s >> 3, j = s & 7, sc, mn;
				q4_k_scale_min(hdr + 16 * blk + 4, j, &sc, &mn);
				int q = (codes[128 * blk + 16 * j + ((e & 31) >> 1)] >> (4 * (e & 1))) & 0xF;
				uint16_t dh, mh;
				memcpy(&dh, hdr + 16 * blk, 2); memcpy(&mh, hdr + 16 * blk + 2, 2);
				b[e] = fp16_to_fp32(dh) * sc * q - fp16_to_fp32(mh) * mn;
				if (b[e] != a[e]) { if (bad < 3) printf("  repack check row %d elem %d: tile %g gguf %g\n", r, e, b[e], a[e]); bad++; }
			}
		}
		printf("repack %.1f ms, tile-vs-gguf dequant check: %s\n", (t1 - t0) * 1e3, bad ? "DIFFERS" : "identical");
		free(a); free(b);
		if (bad) return 4;
	}
	float *x = malloc(sizeof(float) * K);
	for (int k = 0; k < K; k++) x[k] = frand() * (1.f + (k % 5));
	const char *pat = getenv("GEMV_PATTERN");
	if (pat && !strcmp(pat, "ones")) for (int k = 0; k < K; k++) x[k] = 1.f;
	else if (pat && !strncmp(pat, "onehot:", 7)) { for (int k = 0; k < K; k++) x[k] = 0.f; x[atoi(pat + 7) % K] = 1.f; }
	else if (pat && !strncmp(pat, "twohot:", 7)) { for (int k = 0; k < K; k++) x[k] = 0.f; int a1 = atoi(pat + 7), b1 = atoi(strchr(pat, ',') + 1); x[a1 % K] = 1.f; x[b1 % K] = 0.5f; }
	else if (pat && !strncmp(pat, "range:", 6)) { for (int k = 0; k < K; k++) x[k] = 0.f; int a1 = atoi(pat + 6), b1 = atoi(strchr(pat, ',') + 1); for (int k = a1; k < b1 && k < K; k++) x[k] = 1.f; }
	else if (pat && !strncmp(pat, "lane:", 5)) { for (int k = 0; k < K; k++) x[k] = 0.f; int l0 = atoi(pat + 5); for (int k = 0; k < 32; k++) x[(32 * l0 + k) % K] = 1.f; }
	int8_t *xq = malloc(K);
	int32_t *sum32 = malloc(sizeof(int32_t) * K / 32);
	float xs = quant_act_int8(x, xq, sum32, K);
	int16_t *x16 = malloc(sizeof(int16_t) * K);
	for (int k = 0; k < K; k++) x16[k] = xq[k];
	float *yref = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
	t0 = now();
	gemv_q4_k_ref(tile, rows, K, xq, sum32, xs, yref);
	t1 = now();
	printf("reference %.1f ms\n", (t1 - t0) * 1e3);

	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	char cubin[64];
	snprintf(cubin, sizeof cubin, "gemv_q4_k_k%d.cubin", K);
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "gemv_q4_k"));
	int nregs, maxb;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	CUdeviceptr dW, dX, dS, dY, dD;
	CK(cuMemAlloc(&dW, (size_t)stride * rows));
	CK(cuMemAlloc(&dX, sizeof(int16_t) * K));
	CK(cuMemAlloc(&dS, sizeof(int32_t) * K / 32 + 64));
	CK(cuMemAlloc(&dY, sizeof(float) * rows));
	CK(cuMemAlloc(&dD, 4096));
	CK(cuMemcpyHtoD(dW, tile, (size_t)stride * rows));
	CK(cuMemcpyHtoD(dX, x16, sizeof(int16_t) * K));
	CK(cuMemcpyHtoD(dS, sum32, sizeof(int32_t) * K / 32));
	CK(cuMemsetD32(dY, 0x7fc00000u, rows));
	unsigned urows = (unsigned)rows;
	void *params[] = { &dW, &dX, &dS, &dY, &urows, &xs, &dD };
	unsigned grid = (urows + RPC - 1) / RPC;
	if (getenv("GEMV_GRID") && (unsigned)atoi(getenv("GEMV_GRID")) < grid) grid = (unsigned)atoi(getenv("GEMV_GRID"));
	printf("kernel: %d registers, %d CTAs/SM; grid %u x 256, %u rows per CTA iteration\n", nregs, maxb, grid, RPC);
	CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
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
		CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double ta = now();
		for (int i = 0; i < iters; i++) CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double tb = now();
		double per = (tb - ta) / iters;
		printf("%.1f us per matvec, %.1f GB/s of tile\n", per * 1e6, (double)stride * rows / per / 1e9);
	}
	return bad ? 3 : 0;
}
