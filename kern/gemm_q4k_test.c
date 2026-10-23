/*
 * gemm_q4k_test: the prefill GEMMs (gemm_<fmt>.cubin, gen_gemm.rb; Q4_K tiles and BF16 rows so far) on a real tensor
 * against a double-precision CPU reference.
 *
 *   gemm_q4k_test SHARD1 [tensor] [npos] [iters] [rows_used]
 *
 * Default tensor blk.3.attn_output.weight (Q4_K, K=6144, 2560 rows), 256 positions. Xt is k-major fp32 [K][ldx]
 * with ldx = npos rounded up to 128 and zero padding; C is [rows][ldx]. rows_used < rows launches the kernel on a
 * row prefix (its clamp and bounds); C beyond it must stay untouched (NaN). Every 7th row and the last 16 rows are
 * checked (the full reference is slow); GEMM_ALLROWS=1 checks every row.
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
	if (argc < 2) { fprintf(stderr, "usage: gemm_q4k_test SHARD1 [tensor] [npos] [iters] [rows_used]\n"); return 2; }
	const char *tname = argc > 2 ? argv[2] : "blk.3.attn_output.weight";
	int npos = argc > 3 ? atoi(argv[3]) : 256;
	int iters = argc > 4 ? atoi(argv[4]) : 10;
	gguf_file g;
	if (gguf_open(&g, argv[1])) return 1;
	const gguf_tensor *t = gguf_find(&g, tname);
	if (!t || !t->data) { fprintf(stderr, "%s: not found\n", tname); return 1; }
	int bf16 = t->type == GG_BF16, q2 = t->type == GG_Q2_0, q3 = t->type == GG_Q3_K, iq4 = t->type == GG_IQ4_XS, q5 = t->type == GG_Q5_K, q6 = t->type == GG_Q6_K;
	int b32 = t->type == GG_Q4_0 || t->type == GG_Q5_0 || t->type == GG_IQ4_NL || t->type == GG_Q8_0;   /* 32-k blocks, ref/quant.c repack_b32 */
	if (t->type != GG_Q4_K && !bf16 && !q2 && !q3 && !iq4 && !q5 && !q6 && !b32) { fprintf(stderr, "%s is %s; Q4_K, Q5_K, Q6_K, BF16, Q2_0, Q3_K, IQ4_XS, Q4_0, Q5_0, IQ4_NL and Q8_0 so far\n", tname, gguf_type_name(t->type)); return 1; }
	const char *fmt = bf16 ? "bf16" : q2 ? "q2" : q3 ? "q3" : iq4 ? "iq4" : q5 ? "q5k" : q6 ? "q6k" : t->type == GG_Q4_0 ? "q40" : t->type == GG_Q5_0 ? "q50" :
	                  t->type == GG_Q8_0 ? "q80" : b32 ? "iq4nl" : "q4k";
	int K = (int)t->dims[0], rows = (int)(t->elems / K);
	if (t->nd > 2) rows = (int)t->dims[1];   /* an expert tensor: expert 0 */
	int rows_used = argc > 5 ? atoi(argv[5]) : rows;
	if (K % (bf16 ? 16 : b32 ? 32 : q2 ? 64 : 256) || rows_used < 1 || rows_used > rows) { fprintf(stderr, "bad K=%d or rows_used=%d\n", K, rows_used); return 1; }
	int ldx = (npos + 127) & ~127;
	uint32_t gelems, gbytes;
	gguf_block(t->type, &gelems, &gbytes);
	size_t rowbytes = (size_t)K / gelems * gbytes;   /* the tensor's own row (the reference dequantises it) */
	int stride = bf16 ? 2 * K : q2 ? q2_0_tile_stride(K) : q3 ? q3_k_tile_stride(K) : iq4 ? iq4_xs_tile_stride(K) : q5 ? q5_k_tile_stride(K) : q6 ? q6_k_tile_stride(K) : b32 ? b32_tile_stride(t->type, K) : q4_k_tile_stride(K);
	printf("%s: %s K=%d rows=%d (%d used), npos=%d ldx=%d, tile stride %d\n", tname, gguf_type_name(t->type), K, rows, rows_used, npos, ldx, stride);

	uint8_t *tile = malloc((size_t)stride * rows);
	if (bf16) memcpy(tile, t->data, (size_t)stride * rows);   /* bf16 rows go in as they are */
	else if (q2) repack_q2_0(t->data, rows, K, tile);
	else if (q3) repack_q3_k(t->data, rows, K, tile);
	else if (iq4) repack_iq4_xs(t->data, rows, K, tile);
	else if (q5) repack_q5_k(t->data, rows, K, tile);
	else if (q6) repack_q6_k(t->data, rows, K, tile);
	else if (b32) repack_b32(t->type, t->data, rows, K, tile);
	else repack_q4_k(t->data, rows, K, tile);
	/* the Q2_0 kernel's code and scale placement: the dense tile keeps the fp16 scales (2 B per block, padded to 16 B) before the row's 16 B code blocks;
	   Q3_K / IQ4_XS: the d area (16 B per lane group Q) before the block records / nibbles, sstride carries log2 Q */
	int q3_Q = q4_k_lanes_per_row(K) / 8;
	unsigned q2_scales = q2 ? (unsigned)(stride - 16 * (K / 64)) : q3 || iq4 || q6 ? 16u * (unsigned)q3_Q : 0;
	unsigned crow = (unsigned)stride, cstride = q3 ? 112 : 16, srow = (unsigned)stride, sstride = q3 || iq4 || q6 ? (q3_Q == 4 ? 2 : 1) : 2;
	float *x = calloc((size_t)K * ldx, sizeof(float));
	for (int k = 0; k < K; k++) for (int p = 0; p < npos; p++) x[(size_t)k * ldx + p] = frand() * (1.f + (k % 5));
	const char *pat = getenv("GEMM_PATTERN");
	int onehot = -1;
	if (pat && !strncmp(pat, "onehot:", 7)) {   /* X[k][p] = (k == k0 + p): C[r][p] is the weight W[r][k0 + p] itself */
		onehot = atoi(pat + 7);
		memset(x, 0, sizeof(float) * (size_t)K * ldx);
		for (int p = 0; p < npos && onehot + p < K; p++) x[(size_t)(onehot + p) * ldx + p] = 1.f;
	}

	/* reference rows: the dequantised row (the standard dequantiser; the tile holds the same values) against x in double */
	int all = getenv("GEMM_ALLROWS") != NULL;
	int *check = malloc(sizeof(int) * rows_used), nchk = 0;
	for (int r = 0; r < rows_used; r++) if (all || r % 7 == 0 || r >= rows_used - 16) check[nchk++] = r;
	double *ref = malloc(sizeof(double) * (size_t)nchk * npos);
	float *w = malloc(sizeof(float) * K);
	double t0 = now();
	for (int i = 0; i < nchk; i++) {
		dequant(t->type, t->data + (size_t)check[i] * rowbytes, w, K);
		for (int p = 0; p < npos; p++) {
			double acc = 0;
			for (int k = 0; k < K; k++) acc += (double)w[k] * x[(size_t)k * ldx + p];
			ref[(size_t)i * npos + p] = acc;
		}
	}
	printf("reference: %d rows in %.1f ms\n", nchk, (now() - t0) * 1e3);

	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	char cubin[64], entry[32];
	snprintf(cubin, sizeof cubin, "gemm_%s.cubin", fmt); snprintf(entry, sizeof entry, "gemm_%s", fmt);
	CK(cuModuleLoad(&mod, getenv("GEMM_CUBIN") ? getenv("GEMM_CUBIN") : cubin));
	CK(cuModuleGetFunction(&fn, mod, entry));
	int nregs, maxb, smem;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuFuncGetAttribute(&smem, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	CUdeviceptr dW, dX, dC, dD;
	CK(cuMemAlloc(&dW, (size_t)stride * rows));
	CK(cuMemAlloc(&dX, sizeof(float) * (size_t)K * ldx));
	CK(cuMemAlloc(&dC, sizeof(float) * (size_t)rows * ldx));
	CK(cuMemAlloc(&dD, 16384));
	CK(cuMemsetD32(dD, 0xffffffffu, 4096));
	CK(cuMemcpyHtoD(dW, tile, (size_t)stride * rows));
	CK(cuMemcpyHtoD(dX, x, sizeof(float) * (size_t)K * ldx));
	CK(cuMemsetD32(dC, 0x7fc00000u, (size_t)rows * ldx));
	unsigned urows = (unsigned)rows_used, unpos = (unsigned)npos, uk = (unsigned)K, uldx = (unsigned)ldx;
	CUdeviceptr dWc = dW + q2_scales, dS = dW;   /* q2: codes after the scale area; the other kernels ignore the placement params */
	void *params[] = { &dWc, &dX, &dC, &urows, &unpos, &uk, &uldx, &dD, &dS, &crow, &cstride, &srow, &sstride };
	unsigned gx = (urows + 127) / 128, gy = (unsigned)ldx / 128;
	printf("kernel: %d registers, %d B shared, %d CTAs/SM; grid %u x %u x 256\n", nregs, smem, maxb, gx, gy);
	CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	float *c = malloc(sizeof(float) * (size_t)rows * ldx);
	CK(cuMemcpyDtoH(c, dC, sizeof(float) * (size_t)rows * ldx));
	if (getenv("GEMM_DEBUG")) {   /* the kernel built with GEMM_DEBUG=1: CTA (0,0) per thread: c00, ws, nm, wv0, Ax00, By00, hdr0, readBs; GEMM_DEBUG_PRE=1: the first pass's registers */
		uint32_t dbg[4096];
		CK(cuMemcpyDtoH(dbg, dD, sizeof dbg));
		const char *nm[] = { "cx00y00", "ws", "nm", "wv0", "j0Ax00", "j0By00", "hdr0", "readBs" };
		const char *nmpre[] = { "cb/hw", "hb/m0", "nm", "wv0", "wv1", "wv2", "wv3", "ws", "dd", "dm", "wq0", "wh0/wq1" };
		int nsec = getenv("GEMM_DEBUG_PRE") ? 12 : 8;
		for (int s = 0; s < nsec; s++) {
			const char *label = getenv("GEMM_DEBUG_PRE") ? nmpre[s] : nm[s];
			printf("  %-8s", label);
			for (int i = 0; i < 6; i++) { float f; memcpy(&f, &dbg[256 * s + i * 43], 4); printf(" t%-3d %08x (% .4g)", i * 43, dbg[256 * s + i * 43], f); }
			printf("\n");
		}
		for (int s = 0; s < 0; s++) {
			printf("  %-8s", nm[s]);
			for (int i = 0; i < 6; i++) { float f; memcpy(&f, &dbg[256 * s + i * 43], 4); printf(" t%-3d %08x (% .4g)", i * 43, dbg[256 * s + i * 43], f); }
			printf("\n");
		}
	}

	int bad = 0, shown = 0;
	double maxrel = 0;
	for (int i = 0; i < nchk; i++) {
		int r = check[i];
		double scale = 0;
		for (int p = 0; p < npos; p++) { double a = fabs(ref[(size_t)i * npos + p]); if (a > scale) scale = a; }
		for (int p = 0; p < npos; p++) {
			double got = c[(size_t)r * ldx + p], want = ref[(size_t)i * npos + p];
			double rel = fabs(got - want) / (scale + 1e-30);
			if (rel > maxrel || isnan(got)) maxrel = isnan(got) ? INFINITY : rel;
			if (rel > 2e-4 || isnan(got)) { if (shown++ < 8) printf("  row %d pos %d: gpu % .7g ref % .7g (row scale %.3g)\n", r, p, got, want, scale); bad++; }
			if (onehot >= 0 && r == check[0] && p < 64) {   /* the weights of the first checked row, one per position */
				int kk = onehot + p;
				printf("    k=%d (blk %d sub %d el %d): gpu % .6g ref % .6g%s", kk, kk / 256, (kk & 255) >> 5, kk & 31, got, want, rel > 2e-4 ? "  <-- differs" : "");
				if (q3) {   /* the tile's code, inverted high bit and scale of that weight, and what the GPU's value implies */
					int b = kk / 256, g = (kk & 255) >> 4, l = kk & 15, Q = q3_Q;
					const uint8_t *row = tile + (size_t)r * stride, *rec = row + 16 * Q + 112 * b;
					uint32_t cw; memcpy(&cw, rec + 4 * g, 4);
					int code = (cw >> (16 * (l >> 3) + 2 * (l & 7))) & 3, hinv = (rec[64 + 2 * g + (l >> 3)] >> (l & 7)) & 1, sc = (int8_t)rec[96 + g];
					uint16_t dh; memcpy(&dh, row + 16 * (b % Q) + 2 * (b / Q), 2);
					double ds = (double)fp16_to_fp32(dh) * sc;
					printf("  [code %d hinv %d sc %d -> %d; gpu/ds = %.3f]", code, hinv, sc, code - 4 * hinv, ds != 0 ? got / ds : 0.0);
				}
				if (iq4) {   /* the tile's nibble and scale of that weight, the codebook value, and what the GPU's value implies */
					int b = kk / 256, j = (kk & 255) >> 5, e = kk & 31, Q = q3_Q, B = K / 256;
					const uint8_t *row = tile + (size_t)r * stride, *nib = row + 16 * Q;
					int code = (nib[128 * b + 16 * j + (e >> 1)] >> (4 * (e & 1))) & 15, sc = (int8_t)nib[128 * B + 8 * b + j];
					uint16_t dh; memcpy(&dh, row + 16 * (b % Q) + 2 * (b / Q), 2);
					double ds = (double)fp16_to_fp32(dh) * sc;
					printf("  [code %d -> %d sc %d; gpu/ds = %.3f]", code, iq4_kvalue(code), sc, ds != 0 ? got / ds : 0.0);
				}
				printf("\n");
			}
		}
	}
	if (onehot >= 0) {   /* which 8-k groups carry the differences */
		int grp[32] = {0};
		for (int i = 0; i < nchk; i++) for (int p = 0; p < npos; p++) {
			double got = c[(size_t)check[i] * ldx + p], want = ref[(size_t)i * npos + p];
			if (fabs(got - want) > 1e-6 + 1e-4 * fabs(want)) grp[(p >> 3) & 31]++;
		}
		printf("  differing weights per 8-k group (k = %d + 8 g ..): ", onehot);
		for (int g = 0; g < 32 && 8 * g < npos; g++) if (grp[g]) printf(" g%d:%d", g, grp[g]);
		printf("\n");
	}
	int touched = 0;
	for (int r = rows_used; r < rows && touched < 1; r++) for (int p = 0; p < ldx; p++) if (!isnan(c[(size_t)r * ldx + p])) { touched++; printf("  row %d (beyond rows_used) was written\n", r); break; }
	printf("%s: %d of %d checked values beyond 2e-4 of the row scale (max %.3g); rows beyond rows_used %s\n",
	       bad || touched ? "MISMATCH" : "OK", bad, nchk * npos, maxrel, touched ? "WRITTEN" : "untouched");
	if (iters > 0) {
		CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double ta = now();
		for (int i = 0; i < iters; i++) CK(cuLaunchKernel(fn, gx, gy, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double per = (now() - ta) / iters;
		double macs = (double)gx * 128 * ldx * K;
		printf("%.1f us per GEMM: %.1f GMAC/s over the padded tiles (%.1f useful), %.1f GB/s of tile\n", per * 1e6, macs / per / 1e9,
		       (double)rows_used * npos * K / per / 1e9, (double)stride * rows_used / per / 1e9);
	}
	return bad || touched ? 3 : 0;
}
