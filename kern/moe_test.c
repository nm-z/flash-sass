/* moe_test: the MoE GEMV modes against the references, bit for bit:
 *   gather: 10 routed experts' gate matrices (640 x 2560 Q2_0) in one launch
 *   down:   10 routed experts' down matrices (2560 x 640 Q2_0), y = fma chain over experts
 *   acc640: y = fma(s, dot, y) with the scale read from device memory (shared expert down)
 *   bias:   BF16 48 x 2560 with a per-row bias (ssm_alpha + ssm_dt.bias)
 * Random Q2_0 blocks for a table of NEXP experts, random routing. */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include "gguf.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 123;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

/* random raw Q2_0 rows (18-byte blocks of 64: fp16 d + 16 code bytes), repacked to tiles */
static uint8_t *random_q2_tiles(int rows, int K, int *stride) {
	size_t rawbytes = (size_t)rows * (K / 64) * 18;
	uint8_t *raw = malloc(rawbytes);
	for (size_t i = 0; i < rawbytes; i++) raw[i] = (uint8_t)urand();
	for (size_t b = 0; b < (size_t)rows * (K / 64); b++) { uint16_t h = (uint16_t)(0x2c00 + (urand() % 0x1000)); memcpy(raw + 18 * b, &h, 2); }
	*stride = q2_0_tile_stride(K);
	uint8_t *tile = malloc((size_t)*stride * rows);
	repack_q2_0(raw, rows, K, tile);
	free(raw);
	return tile;
}
static void quant(const float *x, int K, int16_t *x16, int32_t *s32, float *as, int8_t *xq) {
	int32_t *s16 = malloc(4 * K / 16);
	quant_act_gpu32(x, K, x16, s32, s16, as);
	for (int k = 0; k < K; k++) xq[k] = (int8_t)x16[k];
	free(s16);
}
static int cmp(const char *what, const float *ref, const float *got, int n) {
	int bad = 0;
	for (int i = 0; i < n; i++) if (memcmp(&ref[i], &got[i], 4)) { if (bad < 3) printf("  %s[%d]: gpu %.9g ref %.9g\n", what, i, got[i], ref[i]); bad++; }
	printf("%-7s %s (%d of %d differ)\n", what, bad ? "MISMATCH" : "BIT-EXACT", bad, n);
	return bad;
}

int main(void) {
	const int NEXP = 16, NR = 10, G = 16;      /* experts in the table, routed, grid */
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CUmodule mg, md, ma, mb; CUfunction fg, fd, fa, fb;
	CK(cuModuleLoad(&mg, "gemv_q2_0_moeg.cubin")); CK(cuModuleGetFunction(&fg, mg, "gemv_q2_0"));
	CK(cuModuleLoad(&md, "gemv_q2_0_moed.cubin")); CK(cuModuleGetFunction(&fd, md, "gemv_q2_0"));
	CK(cuModuleLoad(&ma, "gemv_q2_0_acc640.cubin")); CK(cuModuleGetFunction(&fa, ma, "gemv_q2_0"));
	CK(cuModuleLoad(&mb, "gemv_bf16_bias.cubin")); CK(cuModuleGetFunction(&fb, mb, "gemv_bf16"));
	int fails = 0;
	float xs = 1.f;
	unsigned idx[10];
	for (int e = 0; e < NR; e++) { int again; do { idx[e] = urand() % NEXP; again = 0; for (int j = 0; j < e; j++) if (idx[j] == idx[e]) again = 1; } while (again); }
	float we[10]; for (int e = 0; e < NR; e++) we[e] = 0.05f + 0.1f * (float)(urand() % 100) / 100.f;
	CUdeviceptr dIDX, dWE; CK(cuMemAlloc(&dIDX, 64)); CK(cuMemAlloc(&dWE, 64));
	CK(cuMemcpyHtoD(dIDX, idx, 40)); CK(cuMemcpyHtoD(dWE, we, 40));

	/* ---- gather: K = 2560, 640 rows per expert ---- */
	{
		int K = 2560, R = 640, st;
		uint8_t *tiles = malloc((size_t)NEXP * R * q2_0_tile_stride(K));
		for (int e = 0; e < NEXP; e++) { uint8_t *t = random_q2_tiles(R, K, &st); memcpy(tiles + (size_t)e * st * R, t, (size_t)st * R); free(t); }
		float *x = malloc(4 * K); for (int k = 0; k < K; k++) x[k] = 3.f * frand();
		int16_t *x16 = malloc(2 * K); int32_t *s32 = malloc(4 * K / 32); float *as = malloc(4 * K / 32); int8_t *xq = malloc(K);
		quant(x, K, x16, s32, as, xq);
		float *yref = malloc(4 * NR * R), *ygot = malloc(4 * NR * R);
		for (int e = 0; e < NR; e++) gemv_q2_0_ref(tiles + (size_t)idx[e] * st * R, R, K, xq, s32, as, yref + e * R);
		CUdeviceptr dW, dX, dS, dY, dAS;
		CK(cuMemAlloc(&dW, (size_t)NEXP * st * R)); CK(cuMemAlloc(&dX, 2 * K)); CK(cuMemAlloc(&dS, 4 * K / 32)); CK(cuMemAlloc(&dY, 4 * NR * R)); CK(cuMemAlloc(&dAS, 4 * K / 32));
		CK(cuMemcpyHtoD(dW, tiles, (size_t)NEXP * st * R)); CK(cuMemcpyHtoD(dX, x16, 2 * K)); CK(cuMemcpyHtoD(dS, s32, 4 * K / 32)); CK(cuMemcpyHtoD(dAS, as, 4 * K / 32));
		CK(cuMemsetD32(dY, 0x7fc00000u, NR * R));
		unsigned rows = R;
		void *params[] = { &dW, &dX, &dS, &dY, &rows, &xs, &dIDX, &dAS, &dWE };
		CK(cuLaunchKernel(fg, G, 1, 1, 256, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ygot, dY, 4 * NR * R));
		fails += cmp("gather", yref, ygot, NR * R);
	}
	/* ---- down: K = 640, 2560 rows per expert, 10 activation vectors ---- */
	{
		int K = 640, R = 2560, st;
		uint8_t *tiles = malloc((size_t)NEXP * R * q2_0_tile_stride(K));
		for (int e = 0; e < NEXP; e++) { uint8_t *t = random_q2_tiles(R, K, &st); memcpy(tiles + (size_t)e * st * R, t, (size_t)st * R); free(t); }
		int16_t *x16 = malloc(2 * K * 11); int32_t *s32 = malloc(4 * K / 32 * 11); float *as = malloc(4 * K / 32 * 11); int8_t *xq = malloc(K * 11);
		for (int e = 0; e < 11; e++) { float x[640]; for (int k = 0; k < K; k++) x[k] = 2.f * frand(); quant(x, K, x16 + e * K, s32 + e * K / 32, as + e * K / 32, xq + e * K); }
		float *yref = malloc(4 * R), *ygot = malloc(4 * R), *p = malloc(4 * R);
		for (int e = 0; e < NR; e++) {
			gemv_q2_0_ref(tiles + (size_t)idx[e] * st * R, R, K, xq + e * K, s32 + e * K / 32, as + e * K / 32, p);
			for (int r = 0; r < R; r++) yref[r] = e ? fmaf(we[e], p[r], yref[r]) : we[0] * p[r];
		}
		CUdeviceptr dW, dX, dS, dY, dAS;
		CK(cuMemAlloc(&dW, (size_t)NEXP * st * R)); CK(cuMemAlloc(&dX, 2 * K * 11)); CK(cuMemAlloc(&dS, 4 * K / 32 * 11)); CK(cuMemAlloc(&dY, 4 * R)); CK(cuMemAlloc(&dAS, 4 * K / 32 * 11));
		CK(cuMemcpyHtoD(dW, tiles, (size_t)NEXP * st * R)); CK(cuMemcpyHtoD(dX, x16, 2 * K * 11)); CK(cuMemcpyHtoD(dS, s32, 4 * K / 32 * 11)); CK(cuMemcpyHtoD(dAS, as, 4 * K / 32 * 11));
		CK(cuMemsetD32(dY, 0x7fc00000u, R));
		unsigned rows = R;
		void *params[] = { &dW, &dX, &dS, &dY, &rows, &xs, &dIDX, &dAS, &dWE };
		CK(cuLaunchKernel(fd, G, 1, 1, 256, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ygot, dY, 4 * R));
		fails += cmp("down", yref, ygot, R);
		/* ---- acc640: shared expert down on top: y = fma(s, dot_s, y) ---- */
		float s = 0.731f; CUdeviceptr dSC; CK(cuMemAlloc(&dSC, 4)); CK(cuMemcpyHtoD(dSC, &s, 4));
		gemv_q2_0_ref(tiles + (size_t)idx[0] * st * R, R, K, xq + 10 * K, s32 + 10 * K / 32, as + 10 * K / 32, p);
		for (int r = 0; r < R; r++) yref[r] = fmaf(s, p[r], yref[r]);
		CUdeviceptr dW0 = dW + (size_t)idx[0] * st * R, dX10 = dX + 2 * K * 10, dS10 = dS + 4 * K / 32 * 10, dAS10 = dAS + 4 * K / 32 * 10;
		void *params2[] = { &dW0, &dX10, &dS10, &dY, &rows, &xs, &dIDX, &dAS10, &dSC };
		CK(cuLaunchKernel(fa, G, 1, 1, 256, 1, 1, 0, 0, params2, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ygot, dY, 4 * R));
		fails += cmp("acc640", yref, ygot, R);
	}
	/* ---- bias: BF16 48 x 2560 + bias ---- */
	{
		int K = 2560, R = 48;
		uint16_t *w = malloc(2 * R * K); float *x = malloc(4 * K), bias[48], *yref = malloc(4 * R), *ygot = malloc(4 * R);
		for (int i = 0; i < R * K; i++) { float f = frand(); uint32_t u; memcpy(&u, &f, 4); w[i] = (uint16_t)(u >> 16); }
		for (int k = 0; k < K; k++) x[k] = frand();
		for (int r = 0; r < R; r++) bias[r] = 2.f * frand();
		gemv_bf16_ref(w, R, K, x, yref);
		for (int r = 0; r < R; r++) yref[r] = yref[r] + bias[r];
		CUdeviceptr dW, dX, dY, dB;
		CK(cuMemAlloc(&dW, 2 * R * K)); CK(cuMemAlloc(&dX, 4 * K)); CK(cuMemAlloc(&dY, 4 * R)); CK(cuMemAlloc(&dB, 4 * R));
		CK(cuMemcpyHtoD(dW, w, 2 * R * K)); CK(cuMemcpyHtoD(dX, x, 4 * K)); CK(cuMemcpyHtoD(dB, bias, 4 * R)); CK(cuMemsetD32(dY, 0x7fc00000u, R));
		unsigned rows = R;
		void *params[] = { &dW, &dX, &dY, &rows, &dB };
		CK(cuLaunchKernel(fb, G, 1, 1, 256, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ygot, dY, 4 * R));
		fails += cmp("bias", yref, ygot, R);
	}
	return fails ? 3 : 0;
}
