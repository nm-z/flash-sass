/* Boundary regression for the fixed-width Q2_0 GPU-order reference. */
#include "quant.h"

#include <stdio.h>
#include <stdlib.h>

static unsigned checks;
static unsigned failures;

static void check(int ok, const char *name) {
	checks++;
	if (!ok) {
		failures++;
		fprintf(stderr, "FAIL %s\n", name);
	}
}

int main(void) {
	static const struct { int width; int lanes; } supported[] = {
		{ 640, 4 }, { 2560, 16 }
	};
	static const int unsupported[] = { 0, -1, 64, 160, 320, 960, 1280, 2048, 3840, 5120, 6144, 10240 };
	for (size_t i = 0; i < sizeof supported / sizeof supported[0]; i++) {
		int width = supported[i].width;
		check(q2_0_lanes_per_row(width) == supported[i].lanes,
		      "supported power-of-two lane geometry");
		int stride = q2_0_tile_stride(width);
		int8_t *xq = calloc((size_t)width, sizeof *xq);
		int32_t *sum32 = calloc((size_t)width / 32, sizeof *sum32);
		float *as = calloc((size_t)width / 32, sizeof *as);
		uint8_t *tile = stride > 0 ? calloc((size_t)stride, 1) : NULL;
		float y = 17.25f;
		int ok = stride > 0 && xq && sum32 && as && tile;
		if (ok) ok = gemv_q2_0_ref(tile, 1, width, xq, sum32, as, &y) == 0 && y == 0.0f;
		check(ok, "supported zero-tile GEMV stays in bounds");
		free(tile);
		free(as);
		free(sum32);
		free(xq);
	}
	for (size_t i = 0; i < sizeof unsupported / sizeof unsupported[0]; i++) {
		int width = unsupported[i];
		float y = 17.25f;
		check(q2_0_lanes_per_row(width) == 0, "unsupported width has no lane geometry");
		check(gemv_q2_0_ref(NULL, 1, width, NULL, NULL, NULL, &y) == -1 && y == 17.25f,
		      "unsupported width rejected before reading inputs or writing output");
	}
	float y = 17.25f;
	check(gemv_q2_0_ref(NULL, 0, 640, NULL, NULL, NULL, &y) == -1 && y == 17.25f,
	      "empty row set rejected without writing output");
	check(gemv_q2_0_ref(NULL, 1, 640, NULL, NULL, NULL, &y) == -1 && y == 17.25f,
	      "missing supported-width buffers rejected");
	printf("q2_0_guard_test: checks=%u failures=%u\n", checks, failures);
	return failures ? 1 : 0;
}
