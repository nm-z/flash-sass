/* Sanity checks for the reference decoders against the real GGUF:
 *   ref_test SHARD1 [tensor ...]
 * Decodes the first rows of each listed tensor (default: one of every type),
 * prints statistics, and checks the Q2_0 integer GEMV against a double dot of
 * the dequantised values. */
#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include "quant.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static void stats(const char *name, uint32_t type, const float *w, uint64_t n) {
	double s = 0, s2 = 0, amax = 0;
	uint64_t nan = 0, zero = 0;
	for (uint64_t i = 0; i < n; i++) {
		if (isnan(w[i]) || isinf(w[i])) { nan++; continue; }
		s += w[i]; s2 += (double)w[i] * w[i];
		if (fabs(w[i]) > amax) amax = fabs(w[i]);
		if (w[i] == 0.f) zero++;
	}
	printf("%-36s %-7s n=%-8" PRIu64 " mean=% .3e rms=%.3e amax=%.3e zeros=%.1f%% nan=%" PRIu64 "\n", name, gguf_type_name(type), n,
	       s / n, sqrt(s2 / n), amax, 100.0 * zero / n, nan);
}

static uint32_t rs = 12345;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: ref_test SHARD1 [tensor ...]\n"); return 2; }
	gguf_file g;
	double t0 = now();
	if (gguf_open(&g, argv[1])) return 1;
	printf("opened %s: %" PRIu64 " tensors in %.2f s\n", argv[1], g.ntensors, now() - t0);
	const char *defaults[] = {
		"blk.4.ffn_gate_exps.weight", "blk.4.attn_qkv.weight", "blk.4.attn_gate.weight", "blk.4.ssm_out.weight",
		"blk.3.attn_output.weight", "blk.3.attn_v.weight", "blk.7.attn_output.weight", "blk.4.hc_attn_down.weight",
		"blk.4.ssm_conv1d.weight", "blk.1.ple_conv1d.weight", "blk.3.ffn_down_shexp.weight", "blk.0.ffn_down_shexp.weight",
		"blk.8.ffn_down_shexp.weight", "token_embd.weight", "output.weight", NULL
	};
	int n = argc > 2 ? argc - 2 : 0;
	for (int i = 0; n ? i < n : defaults[i] != NULL; i++) {
		const char *name = n ? argv[2 + i] : defaults[i];
		const gguf_tensor *t = gguf_find(&g, name);
		if (!t) { printf("%s: not found\n", name); continue; }
		if (!t->data) { printf("%s: outside the mapping\n", name); continue; }
		uint32_t be, bb;
		gguf_block(t->type, &be, &bb);
		uint64_t K = t->dims[0], rows = t->elems / K;
		uint64_t take = rows < 4 ? rows : 4;
		if (!be) { printf("%s: type %u unsupported\n", name, t->type); continue; }
		if (K % be) { printf("%s: K=%" PRIu64 " not a multiple of block %u\n", name, K, be); continue; }
		uint64_t rowbytes = K / be * bb;
		if (rowbytes * rows != t->nbytes) printf("%s: WARNING size %" PRIu64 " != rows*rowbytes %" PRIu64 "\n", name, t->nbytes, rowbytes * rows);
		float *w = malloc(sizeof(float) * K * take);
		double t1 = now();
		for (uint64_t r = 0; r < take; r++) dequant(t->type, t->data + r * rowbytes, w + r * K, K);
		double t2 = now();
		char label[80];
		snprintf(label, sizeof label, "%s[%" PRIu64 "x%" PRIu64 "]", name, rows, K);
		stats(label, t->type, w, K * take);
		(void)t1; (void)t2;
		free(w);
	}

	/* Q2_0 GEMV: integer path in GPU order vs double dot of the dequantised row */
	const gguf_tensor *t = gguf_find(&g, "blk.4.ffn_gate_exps.weight");
	if (t && t->data) {
		int K = (int)t->dims[0], rows = (int)t->dims[1];   /* one expert: 640 rows of 2560 */
		float *x = malloc(sizeof(float) * K);
		for (int k = 0; k < K; k++) x[k] = frand() * (1.f + (k % 7));
		int8_t *xq = malloc(K);
		int16_t *x16 = malloc(sizeof(int16_t) * K);
		int32_t *sum32 = malloc(sizeof(int32_t) * K / 32), *sum16 = malloc(sizeof(int32_t) * K / 16);
		float *as = malloc(sizeof(float) * K / 32);
		quant_act_gpu32(x, K, x16, sum32, sum16, as);
		for (int k = 0; k < K; k++) xq[k] = (int8_t)x16[k];
		float *xd = malloc(sizeof(float) * K);
		for (int k = 0; k < K; k++) xd[k] = as[k / 32] * xq[k];
		int stride = q2_0_tile_stride(K);
		uint8_t *tile = malloc((size_t)stride * rows);
		double t1 = now();
		repack_q2_0(t->data, rows, K, tile);
		double t2 = now();
		float *y = malloc(sizeof(float) * rows), *yf = malloc(sizeof(float) * rows);
		if (gemv_q2_0_ref(tile, rows, K, xq, sum32, as, y)) {
			fprintf(stderr, "unsupported Q2_0 reference shape K=%d\n", K);
			return 1;
		}
		double t3 = now();
		gemv_q2_0_float(t->data, rows, K, xd, yf);
		double maxrel = 0, maxabs = 0;
		for (int r = 0; r < rows; r++) {
			double a = fabs(y[r] - yf[r]);
			double rel = a / (fabs(yf[r]) + 1e-6);
			if (a > maxabs) maxabs = a;
			if (rel > maxrel) maxrel = rel;
		}
		printf("q2_0 gemv expert0 %dx%d: stride %d, repack %.1f ms, ref %.1f ms, int-vs-float max abs %.3e rel %.3e, y[0..3] = %g %g %g %g\n",
		       rows, K, stride, (t2 - t1) * 1e3, (t3 - t2) * 1e3, maxabs, maxrel, y[0], y[1], y[2], y[3]);
		free(x); free(xq); free(sum32); free(xd); free(tile); free(y); free(yf);
	}
	gguf_close(&g);
	return 0;
}
