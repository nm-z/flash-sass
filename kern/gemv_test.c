/*
 * gemv_test: run the MaxAs gemv_q2_0 kernel on a real Q2_0 tensor and compare
 * bit-for-bit with the C reference in exactly the kernel's accumulation order.
 *
 *   gemv_test SHARD1 [tensor] [expert|all] [iters]
 *
 * Default tensor blk.4.ffn_gate_exps.weight (512 experts x 640 rows x 2560).
 * "all" runs every expert's rows as one 327680-row matvec (236 MB of weights)
 * for a bandwidth number.
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
static uint32_t rs = 777;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: gemv_test SHARD1 [tensor] [expert|all] [iters]\n"); return 2; }
	const char *tname = argc > 2 ? argv[2] : "blk.4.ffn_gate_exps.weight";
	const char *which = argc > 3 ? argv[3] : "0";
	int iters = argc > 4 ? atoi(argv[4]) : 20;
	gguf_file g;
	if (gguf_open(&g, argv[1])) return 1;
	const gguf_tensor *t = gguf_find(&g, tname);
	if (!t || !t->data) { fprintf(stderr, "%s: not found\n", tname); return 1; }
	if (t->type != GG_Q2_0) { fprintf(stderr, "%s is %s, not Q2_0\n", tname, gguf_type_name(t->type)); return 1; }
	int K = (int)t->dims[0];
	if (K != 2560) { fprintf(stderr, "kernel is built for K=2560, tensor has K=%d\n", K); return 1; }
	int rows_per_expert = t->nd >= 2 ? (int)t->dims[1] : (int)(t->elems / K);
	int rows, row0;
	if (!strcmp(which, "all")) { rows = (int)(t->elems / K); row0 = 0; }
	else { rows = rows_per_expert; row0 = atoi(which) * rows_per_expert; }
	printf("%s: %s K=%d rows %d..%d (%d rows, %.1f MB of tile)\n", tname, gguf_type_name(t->type), K, row0, row0 + rows - 1, rows,
	       (double)rows * q2_0_tile_stride(K) / 1e6);

	/* host side: repack, activations, reference */
	int stride = q2_0_tile_stride(K);
	uint8_t *tile = malloc((size_t)stride * rows);
	double t0 = now();
	repack_q2_0(t->data + (size_t)row0 * (K / 64) * 18, rows, K, tile);
	double t1 = now();
	float *x = malloc(sizeof(float) * K);
	for (int k = 0; k < K; k++) x[k] = frand() * (1.f + (k % 5));
	/* diagnostic activation patterns: "zero", "onehot:K" (x[K] = 1, rest 0), "ones" */
	const char *pat = argc > 5 ? argv[5] : "random";
	if (!strcmp(pat, "zero")) for (int k = 0; k < K; k++) x[k] = 0.f;
	else if (!strncmp(pat, "onehot:", 7)) { for (int k = 0; k < K; k++) x[k] = 0.f; x[atoi(pat + 7) % K] = 1.f; }
	else if (!strcmp(pat, "ones")) for (int k = 0; k < K; k++) x[k] = 1.f;
	else if (!strcmp(pat, "alt")) for (int k = 0; k < K; k++) x[k] = (k & 1) ? -1.f : 1.f;
	int8_t *xq = malloc(K);
	int32_t *sum32 = malloc(sizeof(int32_t) * K / 32);
	float xs = quant_act_int8(x, xq, sum32, K);
	int16_t *x16 = malloc(sizeof(int16_t) * K);
	for (int k = 0; k < K; k++) x16[k] = xq[k];
	float *yref = malloc(sizeof(float) * rows), *y = malloc(sizeof(float) * rows);
	double t2 = now();
	gemv_q2_0_ref(tile, rows, K, xq, sum32, xs, yref);
	double t3 = now();
	printf("repack %.1f ms (%.2f GB/s read), reference %.1f ms\n", (t1 - t0) * 1e3, (double)rows * (K / 64) * 18 / (t1 - t0) / 1e9, (t3 - t2) * 1e3);

	/* device */
	CK(cuInit(0));
	CUdevice dev;
	CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CUmodule mod;
	CK(cuModuleLoad(&mod, "gemv_q2_0.cubin"));
	CUfunction fn;
	CK(cuModuleGetFunction(&fn, mod, "gemv_q2_0"));
	int nregs, smem;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
	int maxb;
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	printf("kernel: %d registers, %d B smem, %d CTAs/SM at 256 threads\n", nregs, smem, maxb);
	CUdeviceptr dW, dX, dS, dY;
	CK(cuMemAlloc(&dW, (size_t)stride * rows));
	CK(cuMemAlloc(&dX, sizeof(int16_t) * K));
	CK(cuMemAlloc(&dS, sizeof(int32_t) * K / 32));
	CK(cuMemAlloc(&dY, sizeof(float) * rows));
	CK(cuMemcpyHtoD(dW, tile, (size_t)stride * rows));
	CK(cuMemcpyHtoD(dX, x16, sizeof(int16_t) * K));
	CK(cuMemcpyHtoD(dS, sum32, sizeof(int32_t) * K / 32));
	CK(cuMemsetD32(dY, 0x7fc00000u, rows));   /* NaN: any row the kernel skips shows up */
	/* debug trace: 12 words per (row, lane16) */
	size_t dbg_bytes = (size_t)rows * 16 * 48;
	CUdeviceptr dD;
	CK(cuMemAlloc(&dD, dbg_bytes));
	CK(cuMemsetD8(dD, 0, dbg_bytes));
	unsigned urows = (unsigned)rows;
	void *params[] = { &dW, &dX, &dS, &dY, &urows, &xs, &dD };
	unsigned grid = (urows + 15) / 16;
	CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(y, dY, sizeof(float) * rows));

	/* compare */
	int bad = 0;
	double maxabs = 0;
	for (int r = 0; r < rows; r++) {
		if (memcmp(&y[r], &yref[r], 4)) {
			if (bad < 8) printf("  row %d: gpu % .9g ref % .9g (diff % .3e)\n", r, y[r], yref[r], (double)y[r] - yref[r]);
			bad++;
		}
		double a = fabs((double)y[r] - yref[r]);
		if (a > maxabs || isnan(a)) maxabs = a;
	}
	printf("%s: %d of %d rows differ (max abs %.3e)\n", bad ? "MISMATCH" : "BIT-EXACT", bad, rows, maxabs);

	/* per-lane trace of the first rows against the same quantities computed here */
	if (bad) {
		uint32_t *tr = malloc(dbg_bytes);
		CK(cuMemcpyDtoH(tr, dD, dbg_bytes));
		int shown = 0;
		int B = K / 64, scales = stride - 16 * B;
		for (int r = 0; r < rows && shown < 12; r++) {
			const uint8_t *row = tile + (size_t)r * stride;
			float v[16];
			for (int l = 0; l < 16; l++) {
				float acc = 0.f;
				uint32_t e[12] = { 0 };
				for (int i = 0; i < 5; i++) {
					int h = l + 16 * i, b = h >> 1;
					const uint8_t *qs = row + scales + 16 * b + 8 * (h & 1);
					int32_t dot = 0;
					for (int j = 0; j < 32; j++) dot += ((qs[j >> 2] >> (2 * (j & 3))) & 3) * xq[32 * h + j];
					uint16_t dh;
					memcpy(&dh, row + 2 * b, 2);
					dot -= sum32[h];
					acc = fmaf(fp16_to_fp32(dh), (float)dot, acc);
					if (i == 0) {
						memcpy(&e[0], qs, 4); memcpy(&e[1], qs + 4, 4);
						memcpy(&e[2], x16 + 32 * h, 4); memcpy(&e[3], x16 + 32 * h + 2, 4);
						e[4] = (uint32_t)(-sum32[h]); e[5] = dh; e[6] = (uint32_t)dot; memcpy(&e[7], &acc, 4);
					}
				}
				memcpy(&e[8], &acc, 4);
				v[l] = acc;
				e[10] = (uint32_t)r;
				const uint32_t *g = tr + ((size_t)r * 16 + l) * 12;
				static const char *nm[] = { "q0", "q1", "act00", "act01", "nsum0", "d0", "dot0", "accf0", "accf_pre", "accf_post", "row" };
				for (int k = 0; k < 11; k++) {
					if (k == 9) continue;
					if (g[k] != e[k]) {
						printf("  trace row %d lane %2d: %s gpu 0x%08x cpu 0x%08x", r, l, nm[k], g[k], e[k]);
						if (k >= 7) { float a, c; memcpy(&a, &g[k], 4); memcpy(&c, &e[k], 4); printf(" (%g vs %g)", a, c); }
						printf("\n");
						shown++;
						break;
					}
				}
			}
			float w[16];
			for (int o = 8; o >= 1; o >>= 1) { for (int l = 0; l < 16; l++) w[l] = v[l] + v[l ^ o]; memcpy(v, w, sizeof v); }
			float gp;
			memcpy(&gp, tr + ((size_t)r * 16) * 12 + 9, 4);
			if (gp != v[0] && shown < 12) { printf("  trace row %d: accf_post gpu %g cpu %g\n", r, gp, v[0]); shown++; }
		}
		/* row 0, every lane: value before the butterfly and what the GPU had after it */
		printf("  row 0 lanes (pre -> gpu post):\n");
		double sum16 = 0;
		for (int l = 0; l < 16; l++) {
			float pre, post;
			memcpy(&pre, tr + (size_t)l * 12 + 8, 4);
			memcpy(&post, tr + (size_t)l * 12 + 9, 4);
			sum16 += pre;
			printf("    %2d: % .6g -> % .6g\n", l, pre, post);
		}
		printf("  sum of the 16 pre values: %.6g\n", sum16);
		free(tr);
	}

	/* timing */
	if (iters > 0) {
		CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double ta = now();
		for (int i = 0; i < iters; i++) CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double tb = now();
		double per = (tb - ta) / iters;
		printf("%.1f us per matvec, %.1f GB/s of tile, %.1f GMAC/s\n", per * 1e6, (double)stride * rows / per / 1e9, (double)rows * K / per / 1e9);
	}
	return bad ? 3 : 0;
}
