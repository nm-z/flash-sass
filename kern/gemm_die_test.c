/*
 * gemm_die_test: the prefill GEMM ops inside the persistent die kernel (die.pl: gemm_q4k, gemm_bf16, gemm_q2, gemm_q3, gemm_iq4, gemm_q5k, gemm_q6k,
 * gemm_q40, gemm_q50, gemm_iq4nl, gemm_q80)
 * on a real tensor against a double-precision CPU reference, one die, a one-phase program.
 *
 *   gemm_die_test SHARD1 [tensor] [npos] [rows_used]        (DIE_CUBIN=die.cubin by default)
 *
 * The same contract as gemm_q4k_test: Xt k-major fp32 [K][ldx] (ldx = npos rounded up to 128, zero past npos),
 * C [rows][ldx]; rows past rows_used are clamped by the op and must stay untouched (NaN). Every 7th row and the
 * last 16 are checked. Run on one GPU through gpuq: CUDA_VISIBLE_DEVICES=7 ./gemm_die_test SHARD1
 */
#define _POSIX_C_SOURCE 200809L
#include "dieops.h"
#include "gguf.h"
#include "quant.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

typedef struct { uint32_t w[32]; } desc_t;
static void d_init(desc_t *d, int op, unsigned units) { memset(d, 0, sizeof *d); d->w[0] = (uint32_t)op; d->w[1] = units; }
static void d_ptr(desc_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w + 1] = (uint32_t)(p >> 32); }
static void d_u32(desc_t *d, int addr, uint32_t v) { d->w[2 + (addr - 0x140) / 4] = v; }

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static uint32_t rs = 99;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: gemm_die_test SHARD1 [tensor] [npos] [rows_used]\n"); return 2; }
	const char *tname = argc > 2 ? argv[2] : "blk.3.attn_output.weight";
	int npos = argc > 3 ? atoi(argv[3]) : 256;
	gguf_file g;
	if (gguf_open(&g, argv[1])) return 1;
	const gguf_tensor *t = gguf_find(&g, tname);
	if (!t || !t->data) { fprintf(stderr, "%s: not found\n", tname); return 1; }
	int bf16 = t->type == GG_BF16, q2 = t->type == GG_Q2_0, q3 = t->type == GG_Q3_K, q4 = t->type == GG_Q4_K, iq4 = t->type == GG_IQ4_XS, q5 = t->type == GG_Q5_K, q6 = t->type == GG_Q6_K;
	int b32 = t->type == GG_Q4_0 || t->type == GG_Q5_0 || t->type == GG_IQ4_NL || t->type == GG_Q8_0;
	if (!bf16 && !q2 && !q3 && !q4 && !iq4 && !q5 && !q6 && !b32) { fprintf(stderr, "%s is %s; Q4_K, Q5_K, Q6_K, BF16, Q2_0, Q3_K, IQ4_XS, Q4_0, Q5_0, IQ4_NL and Q8_0\n", tname, gguf_type_name(t->type)); return 1; }
	int op = bf16 ? OP_GEMM_BF16 : q2 ? OP_GEMM_Q2 : q3 ? OP_GEMM_Q3 : iq4 ? OP_GEMM_IQ4 : q5 ? OP_GEMM_Q5K : q6 ? OP_GEMM_Q6K :
	         t->type == GG_Q4_0 ? OP_GEMM_Q40 : t->type == GG_Q5_0 ? OP_GEMM_Q50 : t->type == GG_Q8_0 ? OP_GEMM_Q80 : b32 ? OP_GEMM_IQ4NL : OP_GEMM_Q4K;
	int K = (int)t->dims[0], rows = (int)(t->elems / K);
	if (t->nd > 2) rows = (int)t->dims[1];
	int rows_used = argc > 4 ? atoi(argv[4]) : rows;
	if (K % (bf16 ? 16 : b32 ? 32 : q2 ? 64 : 256) || rows_used < 1 || rows_used > rows) { fprintf(stderr, "bad K=%d or rows_used=%d\n", K, rows_used); return 1; }
	int ldx = (npos + 127) & ~127;
	uint32_t gelems, gbytes;
	gguf_block(t->type, &gelems, &gbytes);
	size_t rowbytes = (size_t)K / gelems * gbytes;
	int stride = bf16 ? 2 * K : q2 ? q2_0_tile_stride(K) : q3 ? q3_k_tile_stride(K) : iq4 ? iq4_xs_tile_stride(K) : q5 ? q5_k_tile_stride(K) : q6 ? q6_k_tile_stride(K) :
	             b32 ? b32_tile_stride(t->type, K) : q4_k_tile_stride(K);
	uint8_t *tile = malloc((size_t)stride * rows);
	if (bf16) memcpy(tile, t->data, (size_t)stride * rows);
	else if (q2) repack_q2_0(t->data, rows, K, tile);
	else if (q3) repack_q3_k(t->data, rows, K, tile);
	else if (iq4) repack_iq4_xs(t->data, rows, K, tile);
	else if (q5) repack_q5_k(t->data, rows, K, tile);
	else if (q6) repack_q6_k(t->data, rows, K, tile);
	else if (b32) repack_b32(t->type, t->data, rows, K, tile);
	else repack_q4_k(t->data, rows, K, tile);
	int q3_Q = q4_k_lanes_per_row(K) / 8;
	unsigned coff = q2 ? (unsigned)(stride - 16 * (K / 64)) : q3 || iq4 || q6 ? 16u * (unsigned)q3_Q : 0;
	unsigned crow = (unsigned)stride, cstride = q3 ? 112 : 16, srow = (unsigned)stride, sstride = q3 || iq4 || q6 ? (q3_Q == 4 ? 2 : 1) : 2;
	printf("%s: %s K=%d rows=%d (%d used), npos=%d ldx=%d, op %d\n", tname, gguf_type_name(t->type), K, rows, rows_used, npos, ldx, op);

	float *x = calloc((size_t)K * ldx, sizeof(float));
	for (int k = 0; k < K; k++) for (int p = 0; p < npos; p++) x[(size_t)k * ldx + p] = frand() * (1.f + (k % 5));
	int *check = malloc(sizeof(int) * rows_used), nchk = 0;
	for (int r = 0; r < rows_used; r++) if (r % 7 == 0 || r >= rows_used - 16) check[nchk++] = r;
	double *ref = malloc(sizeof(double) * (size_t)nchk * npos);
	float *w = malloc(sizeof(float) * K);
	for (int i = 0; i < nchk; i++) {
		dequant(t->type, t->data + (size_t)check[i] * rowbytes, w, K);
		for (int p = 0; p < npos; p++) { double acc = 0; for (int k = 0; k < K; k++) acc += (double)w[k] * x[(size_t)k * ldx + p]; ref[(size_t)i * npos + p] = acc; }
	}

	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	CUdeviceptr dPROG, dCTL, dW, dX, dC;
	CK(cuMemAlloc(&dPROG, 128 * 16 * 4)); CK(cuMemAlloc(&dCTL, 8192)); CK(cuMemsetD8(dCTL, 0, 8192));
	CK(cuMemAlloc(&dW, (size_t)stride * rows)); CK(cuMemAlloc(&dX, sizeof(float) * (size_t)K * ldx)); CK(cuMemAlloc(&dC, sizeof(float) * (size_t)rows * ldx));
	CK(cuMemcpyHtoD(dW, tile, (size_t)stride * rows)); CK(cuMemcpyHtoD(dX, x, sizeof(float) * (size_t)K * ldx));
	CK(cuMemsetD32(dC, 0x7fc00000u, (size_t)rows * ldx));
	desc_t prog[1];
	d_init(&prog[0], op, 16);
	d_ptr(&prog[0], 0x140, dW + coff); d_ptr(&prog[0], 0x148, dX); d_ptr(&prog[0], 0x150, dC);
	d_u32(&prog[0], 0x158, (uint32_t)rows_used); d_u32(&prog[0], 0x15c, (uint32_t)npos); d_u32(&prog[0], 0x160, (uint32_t)K); d_u32(&prog[0], 0x164, (uint32_t)ldx);
	d_ptr(&prog[0], 0x170, dW); d_u32(&prog[0], 0x178, crow); d_u32(&prog[0], 0x17c, cstride); d_u32(&prog[0], 0x180, srow); d_u32(&prog[0], 0x184, sstride);
/* GEMM_SLICE=1 with a Q6_K K=6144 tensor: the holder's four K=1536 column slices (kern/hold.c), four GEMMs into a scratch, slab_sum */
desc_t sprog[5]; unsigned nphs = 1;
if (getenv("GEMM_SLICE") && q6 && K == 6144) {
	int st1 = q6_k_tile_stride(1536), Q1 = q4_k_lanes_per_row(1536) / 8;
	size_t sl = rowbytes / 4;
	uint8_t *stile = malloc((size_t)st1 * rows * 4), *tmp = malloc(sl * (size_t)rows);
	for (int sidx = 0; sidx < 4; sidx++) {
		for (int r = 0; r < rows; r++) memcpy(tmp + (size_t)r * sl, t->data + (size_t)r * rowbytes + sl * sidx, sl);
		repack_q6_k(tmp, rows, 1536, stile + (size_t)st1 * rows * sidx);
	}
	CUdeviceptr dWs, dSC; CK(cuMemAlloc(&dWs, (size_t)st1 * rows * 4)); CK(cuMemcpyHtoD(dWs, stile, (size_t)st1 * rows * 4));
	CK(cuMemAlloc(&dSC, sizeof(float) * 4 * (size_t)rows_used * ldx));
	for (int sidx = 0; sidx < 4; sidx++) {
		CUdeviceptr Ws = dWs + (size_t)st1 * rows * sidx;
		d_init(&sprog[sidx], OP_GEMM_Q6K, 16);
		d_ptr(&sprog[sidx], 0x140, Ws + 16u * (unsigned)Q1); d_ptr(&sprog[sidx], 0x148, dX + sizeof(float) * 1536 * (size_t)ldx * sidx); d_ptr(&sprog[sidx], 0x150, dSC + sizeof(float) * (size_t)rows_used * ldx * sidx);
		d_u32(&sprog[sidx], 0x158, (uint32_t)rows_used); d_u32(&sprog[sidx], 0x15c, (uint32_t)npos); d_u32(&sprog[sidx], 0x160, 1536); d_u32(&sprog[sidx], 0x164, (uint32_t)ldx);
		d_ptr(&sprog[sidx], 0x170, Ws); d_u32(&sprog[sidx], 0x178, (uint32_t)st1); d_u32(&sprog[sidx], 0x17c, 16); d_u32(&sprog[sidx], 0x180, (uint32_t)st1); d_u32(&sprog[sidx], 0x184, Q1 == 4 ? 2 : 1);
	}
	unsigned n4 = (unsigned)rows_used * (unsigned)ldx / 4u, lgx = 0; while ((1 << lgx) < ldx) lgx++;
	d_init(&sprog[4], OP_SLAB_SUM, (n4 + 255) / 256); d_ptr(&sprog[4], 0x140, dSC); d_ptr(&sprog[4], 0x148, dC);
	d_u32(&sprog[4], 0x150, n4); d_u32(&sprog[4], 0x154, 4); d_u32(&sprog[4], 0x158, (uint32_t)rows_used); d_u32(&sprog[4], 0x15c, lgx);
	CK(cuMemcpyHtoD(dPROG, sprog, sizeof sprog)); nphs = 5;
	printf("sliced: four K=1536 GEMMs and their sum\n");
} else CK(cuMemcpyHtoD(dPROG, prog, sizeof prog));
unsigned nph = nphs, ntok = 1, zero = 0;
	CK(cuMemcpyHtoD(dCTL + 8, &zero, 4));
	void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	double t0 = now();
	CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	double t1 = now();
	float *c = malloc(sizeof(float) * (size_t)rows * ldx);
	CK(cuMemcpyDtoH(c, dC, sizeof(float) * (size_t)rows * ldx));
	int bad = 0, shown = 0; double maxrel = 0;
	for (int i = 0; i < nchk; i++) {
		int r = check[i]; double scale = 0;
		for (int p = 0; p < npos; p++) { double a = fabs(ref[(size_t)i * npos + p]); if (a > scale) scale = a; }
		for (int p = 0; p < npos; p++) {
			double got = c[(size_t)r * ldx + p], want = ref[(size_t)i * npos + p], rel = fabs(got - want) / (scale + 1e-30);
			if (rel > maxrel || isnan(got)) maxrel = isnan(got) ? INFINITY : rel;
			if (rel > 2e-4 || isnan(got)) { if (shown++ < 6) printf("  row %d pos %d: gpu % .7g ref % .7g\n", r, p, got, want); bad++; }
		}
	}
	int touched = 0;
	for (int r = rows_used; r < rows && !touched; r++) for (int p = 0; p < ldx; p++) if (!isnan(c[(size_t)r * ldx + p])) { touched = 1; break; }
	printf("%s: %d of %d checked values beyond 2e-4 of the row scale (max %.3g); rows beyond rows_used %s; %.1f us for the phase (%.0f GMAC/s)\n",
	       bad || touched ? "MISMATCH" : "OK", bad, nchk * npos, maxrel, touched ? "WRITTEN" : "untouched", (t1 - t0) * 1e6, (double)rows_used * ldx * K / (t1 - t0) / 1e9);
	return bad || touched ? 3 : 0;
}
