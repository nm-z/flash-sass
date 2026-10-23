/* deqcheck: our dequant() (ref/quant.c) against ggml's dequantize_row_* on real tensors
 * of the GGUF — the twins and the kernels share ref/quant.c's reading of every block
 * format, so this is the one check that reaches outside our own code.
 *
 *   deqcheck <gguf> <libggml dir> tensor...     (build: see kern/Makefile deqcheck) */
#include "gguf.h"
#include "quant.h"
#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-quants.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int ggml_deq(uint32_t type, const void *src, float *dst, int64_t k) {
	switch (type) {
	case 2:  dequantize_row_q4_0((const block_q4_0 *)src, dst, k); return 0;
	case 6:  dequantize_row_q5_0((const block_q5_0 *)src, dst, k); return 0;
	case 8:  dequantize_row_q8_0((const block_q8_0 *)src, dst, k); return 0;
	case 11: dequantize_row_q3_K((const block_q3_K *)src, dst, k); return 0;
	case 12: dequantize_row_q4_K((const block_q4_K *)src, dst, k); return 0;
	case 13: dequantize_row_q5_K((const block_q5_K *)src, dst, k); return 0;
	case 14: dequantize_row_q6_K((const block_q6_K *)src, dst, k); return 0;
	case 20: dequantize_row_iq4_nl((const block_iq4_nl *)src, dst, k); return 0;
	case 23: dequantize_row_iq4_xs((const block_iq4_xs *)src, dst, k); return 0;
	case 42: dequantize_row_q2_0((const block_q2_0 *)src, dst, k); return 0;
	default: return -1;
	}
}

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: deqcheck <gguf> tensor...\n"); return 2; }
	gguf_file g; if (gguf_open(&g, argv[1])) return 1;
	int bad = 0;
	for (int a = 2; a < argc; a++) {
		const gguf_tensor *t = gguf_find(&g, argv[a]);
		if (!t) { printf("%-36s absent\n", argv[a]); continue; }
		if (!t->data) { printf("%-36s not mapped\n", argv[a]); continue; }
		uint32_t be, bb; gguf_block(t->type, &be, &bb);
		uint64_t K = t->dims[0], rows = 1; for (int i = 1; i < 4; i++) if (t->dims[i]) rows *= t->dims[i];
		size_t rowbytes = (size_t)(K / be) * bb;
		float *ours = malloc(4 * K), *theirs = malloc(4 * K);
		double worst = 0; uint64_t worst_row = 0, worst_i = 0; int unsupported = 0;
		uint64_t pick[5] = { 0, 1, rows / 2, rows - 2, rows - 1 };
		for (int p = 0; p < 5; p++) {
			uint64_t r = pick[p] < rows ? pick[p] : rows - 1;
			const uint8_t *src = t->data + r * rowbytes;
			if (dequant(t->type, src, ours, K)) { unsupported = 1; break; }
			if (ggml_deq(t->type, src, theirs, (int64_t)K)) { unsupported = 2; break; }
			for (uint64_t i = 0; i < K; i++) { double d = fabs((double)ours[i] - theirs[i]); if (d > worst) { worst = d; worst_row = r; worst_i = i; } }
		}
		if (unsupported) { printf("%-36s %-7s %s dequant unsupported\n", argv[a], gguf_type_name(t->type), unsupported == 1 ? "our" : "ggml"); continue; }
		printf("%-36s %-7s K %6llu rows %8llu  max |ours - ggml| %.3g%s", argv[a], gguf_type_name(t->type), (unsigned long long)K, (unsigned long long)rows, worst, worst > 1e-6 ? "  MISMATCH" : "  ok");
		if (worst > 1e-6) { bad++; printf(" (row %llu elem %llu: ours %.6g ggml %.6g)", (unsigned long long)worst_row, (unsigned long long)worst_i, ours[worst_i], theirs[worst_i]); }
		printf("\n");
		free(ours); free(theirs);
	}
	gguf_close(&g);
	return bad ? 3 : 0;
}
