/* dnstep_test: the DeltaNet head-step kernel (48 heads) against deltanet_step_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "deltanet.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 23;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
	int tiled = argc > 1 && !strcmp(argv[1], "tiled");
	const int H = 48, KH = 16, D = 128;
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, tiled ? "dnstep_tiled.cubin" : "dnstep.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "dnstep"));
	int nregs, maxb;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	size_t sbytes = (size_t)H * D * D * 4;
	float *S = malloc(sbytes), *S2 = malloc(sbytes), *Sg = malloc(sbytes);
	float *q = malloc(KH * D * 4), *k = malloc(KH * D * 4), *v = malloc(H * D * 4);
	float *a = malloc(H * 4), *b = malloc(H * 4), *sa = malloc(H * 4), *o = malloc(H * D * 4), *og = malloc(H * D * 4);
	for (size_t i = 0; i < (size_t)H * D * D; i++) S[i] = 0.5f * frand();
	for (int i = 0; i < KH * D; i++) { q[i] = frand() * 2.f; k[i] = frand() * 2.f; }
	for (int i = 0; i < H * D; i++) v[i] = frand();
	for (int h = 0; h < H; h++) { a[h] = 3.f * frand(); b[h] = 3.f * frand(); sa[h] = -expf(frand()); }
	memcpy(S2, S, sbytes);
	for (int h = 0; h < H; h++) {
		int kh = tiled ? (h & 15) : h / 3;
		deltanet_step_gpu(S2 + (size_t)h * D * D, q + kh * D, k + kh * D, v + h * D, a[h], b[h], sa[h], eps, o + h * D);
	}
	CUdeviceptr dS, dQ, dK, dV, dA, dB, dSA, dO;
	CK(cuMemAlloc(&dS, sbytes)); CK(cuMemAlloc(&dQ, KH * D * 4)); CK(cuMemAlloc(&dK, KH * D * 4)); CK(cuMemAlloc(&dV, H * D * 4));
	CK(cuMemAlloc(&dA, H * 4)); CK(cuMemAlloc(&dB, H * 4)); CK(cuMemAlloc(&dSA, H * 4)); CK(cuMemAlloc(&dO, H * D * 4));
	CK(cuMemcpyHtoD(dS, S, sbytes)); CK(cuMemcpyHtoD(dQ, q, KH * D * 4)); CK(cuMemcpyHtoD(dK, k, KH * D * 4)); CK(cuMemcpyHtoD(dV, v, H * D * 4));
	CK(cuMemcpyHtoD(dA, a, H * 4)); CK(cuMemcpyHtoD(dB, b, H * 4)); CK(cuMemcpyHtoD(dSA, sa, H * 4)); CK(cuMemsetD32(dO, 0x7fc00000u, H * D));
	void *params[] = { &dS, &dQ, &dK, &dV, &dA, &dB, &dSA, &dO, &eps };
	CK(cuLaunchKernel(fn, H, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(Sg, dS, sbytes)); CK(cuMemcpyDtoH(og, dO, H * D * 4));
	int bs = 0, bo = 0;
	for (size_t i = 0; i < (size_t)H * D * D; i++) if (memcmp(&Sg[i], &S2[i], 4)) { if (bs < 3) printf("  S[%zu] (head %zu c %zu i %zu): gpu %.9g ref %.9g\n", i, i / (D * D), (i / D) % D, i % D, Sg[i], S2[i]); bs++; }
	for (int i = 0; i < H * D; i++) if (memcmp(&og[i], &o[i], 4)) { if (bo < 3) printf("  o[%d] (head %d c %d): gpu %.9g ref %.9g\n", i, i / D, i % D, og[i], o[i]); bo++; }
	printf("dnstep %s: %d registers, %d CTAs/SM; %s (state %d of %d differ, o %d of %d differ)\n", tiled ? "tiled" : "grouped", nregs, maxb,
	       (bs || bo) ? "MISMATCH" : "BIT-EXACT", bs, H * D * D, bo, H * D);
	/* timing: 48 heads, state 6 MB in + 6 MB out */
	CK(cuMemcpyHtoD(dS, S, sbytes));
	double t0 = now();
	for (int it = 0; it < 20; it++) CK(cuLaunchKernel(fn, H, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	double t1 = now();
	printf("%.1f us per step of 48 heads (%.1f GB/s of state)\n", (t1 - t0) / 20 * 1e6, 2.0 * sbytes / ((t1 - t0) / 20) / 1e9);
	return (bs || bo) ? 3 : 0;
}
