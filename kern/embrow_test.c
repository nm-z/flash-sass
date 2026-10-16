/* embrow_test: Q3_K embedding-row decode (broadcast to 4 lanes) against dequant(),
 * on random Q3_K rows. usage: embrow_test [rows] */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include "gguf.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 5;
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

int main(int argc, char **argv) {
	int R = argc > 1 ? atoi(argv[1]) : 300;      /* rows in the fake table */
	size_t bytes = (size_t)R * 1100;
	uint8_t *tab = malloc(bytes);
	for (size_t i = 0; i < bytes; i++) tab[i] = (uint8_t)urand();
	for (int r = 0; r < R; r++)            /* sane fp16 d per block: exponent in a normal range */
		for (int b = 0; b < 10; b++) { uint16_t h = (uint16_t)(0x2c00 + (urand() % 0x1000)); memcpy(tab + (size_t)r * 1100 + 110 * b + 108, &h, 2); }
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "embrow.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "embrow"));
	CUdeviceptr dE, dS;
	CK(cuMemAlloc(&dE, bytes)); CK(cuMemAlloc(&dS, 4 * 10240)); CUdeviceptr dIDS; CK(cuMemAlloc(&dIDS, 16));
	CK(cuMemcpyHtoD(dE, tab, bytes));
	float *ref = malloc(4 * 2560), *got = malloc(4 * 10240);
	int fails = 0;
	const unsigned ids[] = { 0, 1, 2, 17, (unsigned)R - 1, (unsigned)R / 2 };
	for (unsigned k = 0; k < sizeof ids / sizeof ids[0]; k++) {
		unsigned id = ids[k], pos = 0;
		dequant(GG_Q3_K, tab + (size_t)id * 1100, ref, 2560);
		CK(cuMemsetD32(dS, 0x7fc00000u, 10240));
		CK(cuMemcpyHtoD(dIDS, &id, 4));
		void *params[] = { &dE, &dS, &pos, &dIDS };
		CK(cuLaunchKernel(fn, 10, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(got, dS, 4 * 10240));
		int bad = 0;
		for (int l = 0; l < 4; l++)
			for (int v = 0; v < 2560; v++)
				if (memcmp(&ref[v], &got[2560 * l + v], 4)) { if (bad < 4) printf("  id %u lane %d v %d: gpu %.9g ref %.9g\n", id, l, v, got[2560 * l + v], ref[v]); bad++; }
		printf("id %u: %s (%d of 10240 differ)\n", id, bad ? "MISMATCH" : "BIT-EXACT", bad);
		if (bad) fails++;
	}
	return fails ? 3 : 0;
}
