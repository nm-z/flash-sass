/* plegate_test: the PLE lane gate + value broadcast against ple_gate_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "ple.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 29;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "plegate.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "plegate"));
	float *kn = malloc(10240 * 4), *qn = malloc(10240 * 4), *v = malloc(2560 * 4), *g = malloc(10240 * 4), *gg = malloc(10240 * 4), gates[4], gatesg[4];
	CUdeviceptr dKN, dQN, dV, dG, dGT;
	CK(cuMemAlloc(&dKN, 10240 * 4)); CK(cuMemAlloc(&dQN, 10240 * 4)); CK(cuMemAlloc(&dV, 2560 * 4)); CK(cuMemAlloc(&dG, 10240 * 4)); CK(cuMemAlloc(&dGT, 16));
	int fails = 0;
	for (int trial = 0; trial < 4; trial++) {
		float sc = trial == 1 ? 1e-4f : trial == 2 ? 40.f : 1.f;
		for (int i = 0; i < 10240; i++) { kn[i] = sc * frand(); qn[i] = frand(); }
		for (int i = 0; i < 2560; i++) v[i] = 3.f * frand();
		if (trial == 3) for (int i = 0; i < 2560; i++) qn[2 * 2560 + i] = 0.f;      /* lane 2: s = 0 -> gate 0.5 */
		ple_gate_gpu(kn, qn, v, g, gates);
		CK(cuMemcpyHtoD(dKN, kn, 10240 * 4)); CK(cuMemcpyHtoD(dQN, qn, 10240 * 4)); CK(cuMemcpyHtoD(dV, v, 2560 * 4));
		CK(cuMemsetD32(dG, 0x7fc00000u, 10240)); CK(cuMemsetD32(dGT, 0x7fc00000u, 4));
		void *params[] = { &dKN, &dQN, &dV, &dG, &dGT };
		CK(cuLaunchKernel(fn, 4, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(gg, dG, 10240 * 4)); CK(cuMemcpyDtoH(gatesg, dGT, 16));
		int bg = 0, bt = 0;
		for (int l = 0; l < 4; l++) if (memcmp(&gates[l], &gatesg[l], 4)) { printf("  gate[%d]: gpu %.9g ref %.9g\n", l, gatesg[l], gates[l]); bt++; }
		for (int i = 0; i < 10240; i++) if (memcmp(&g[i], &gg[i], 4)) { if (bg < 3) printf("  g[%d]: gpu %.9g ref %.9g\n", i, gg[i], g[i]); bg++; }
		printf("trial %d: gates %s, G %s (%d of 10240 differ); gates = %g %g %g %g\n", trial, bt ? "MISMATCH" : "BIT-EXACT", bg ? "MISMATCH" : "BIT-EXACT", bg, gates[0], gates[1], gates[2], gates[3]);
		if (bg || bt) fails++;
	}
	return fails ? 3 : 0;
}
