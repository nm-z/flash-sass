/* plerows_test: n-gram row hash + host-mapped IQ4_NL row gather against
 * ple_rows_c + dequant, bit for bit, on a small random table (the real head
 * offsets/vocabularies are scaled down; the multipliers are the model's). */
#define _POSIX_C_SOURCE 200809L
#include "ple.h"
#include "quant.h"
#include "gguf.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 11;
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }

int main(void) {
	const size_t R = 1u << 20;                   /* rows in the fake table */
	const uint32_t absent = PLE_EOS_TOKEN, endid = getenv("PLE_DISTINCT_IDS") ? 248046u : PLE_EOS_TOKEN;
	uint64_t off[16], voc[16];
	for (int h = 0; h < 16; h++) { off[h] = (uint64_t)h * 65000; voc[h] = 60000 + 13 * h; }
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN | CU_CTX_MAP_HOST, dev));
	int ng = getenv("PLE_NOGATHER") != NULL;
	CK(cuModuleLoad(&mod, ng ? "plerows_ng.cubin" : "plerows.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "plerows"));
	/* the table in host-mapped memory, read by the GPU over PCIe */
	uint8_t *tab; CUdeviceptr dTAB;
	CK(cuMemHostAlloc((void **)&tab, R * 90, CU_MEMHOSTALLOC_DEVICEMAP));
	CK(cuMemHostGetDevicePointer(&dTAB, tab, 0));
	for (size_t i = 0; i < R * 90; i++) tab[i] = (uint8_t)urand();
	for (size_t r = 0; r < R; r++) for (int b = 0; b < 5; b++) { uint16_t hd = (uint16_t)(0x2c00 + (urand() % 0x1000)); memcpy(tab + r * 90 + 18 * b, &hd, 2); }
	/* constants block */
	uint8_t pc[160]; memset(pc, 0, sizeof pc);
	memcpy(pc, ple_multipliers, 24);
	for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)off[h], v32 = (uint32_t)voc[h]; memcpy(pc + 24 + 4 * h, &o32, 4); memcpy(pc + 88 + 4 * h, &v32, 4); }
	memcpy(pc + 152, &absent, 4); memcpy(pc + 156, &endid, 4);
	/* a token sequence with ends planted */
	const int T = 40;
	uint32_t ids[40];
	for (int i = 0; i < T; i++) ids[i] = urand() % 248320;
	ids[0] = 42; ids[1] = 248046; ids[2] = 198;
	ids[10] = endid; ids[20] = endid; ids[21] = endid; ids[30] = absent;
	CUdeviceptr dIDS, dPC, dG, dROWS;
	CK(cuMemAlloc(&dIDS, 4 * T)); CK(cuMemAlloc(&dPC, 160)); CK(cuMemAlloc(&dG, 2560 * 4)); CK(cuMemAlloc(&dROWS, 64));
	CK(cuMemcpyHtoD(dIDS, ids, 4 * T)); CK(cuMemcpyHtoD(dPC, pc, 160));
	float *gref = malloc(2560 * 4), *ggot = malloc(2560 * 4);
	uint32_t rowsg[16];
	uint64_t rows[16];
	int fails = 0;
	const unsigned poss[] = { 0, 1, 2, 3, 9, 10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 39 };
	for (unsigned k = 0; k < sizeof poss / sizeof poss[0]; k++) {
		unsigned pos = poss[k];
		ple_rows_c(ids, pos, absent, &endid, 1, ple_multipliers, off, voc, rows);
		for (int h = 0; h < 16; h++) dequant(GG_IQ4_NL, tab + rows[h] * 90, gref + 160 * h, 160);
		CK(cuMemsetD32(dG, 0x7fc00000u, 2560)); CK(cuMemsetD32(dROWS, 0xffffffffu, 16));
		void *params[] = { &dIDS, &dPC, &dTAB, &dG, &dROWS, &pos };
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(ggot, dG, 2560 * 4)); CK(cuMemcpyDtoH(rowsg, dROWS, 64));
		int br = 0, bg = 0;
		for (int h = 0; h < 16; h++) if (rowsg[h] != rows[h]) { if (br < 3) printf("  pos %u head %d: row gpu %u ref %llu\n", pos, h, rowsg[h], (unsigned long long)rows[h]); br++; }
		for (int i = 0; i < (ng ? 0 : 2560); i++) if (memcmp(&gref[i], &ggot[i], 4)) { if (bg < 3) printf("  pos %u g[%d]: gpu %.9g ref %.9g\n", pos, i, ggot[i], gref[i]); bg++; }
		printf("pos %2u: rows %s (%d differ), g %s (%d of 2560 differ)\n", pos, br ? "MISMATCH" : "OK", br, bg ? "MISMATCH" : "BIT-EXACT", bg);
		if (br || bg) fails++;
	}
	return fails ? 3 : 0;
}
