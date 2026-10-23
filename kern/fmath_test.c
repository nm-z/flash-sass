/* fmath_test: exp/sigmoid/SiLU/softplus from kern/fmath.pl against ref/fmath.c, bit for bit, on 2560 inputs. */
#define _POSIX_C_SOURCE 200809L
#include "fmath.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 17;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	int n = 2560;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "fmath.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "fmath"));
	CUdeviceptr d[5];
	for (int i = 0; i < 5; i++) CK(cuMemAlloc(&d[i], 4 * n));
	float *x = malloc(4 * n), *out[4], *ref[4];
	for (int i = 0; i < 4; i++) { out[i] = malloc(4 * n); ref[i] = malloc(4 * n); }
	const char *names[4] = { "exp", "sigmoid", "silu", "softplus" };
	const float mags[] = { 1.f, 8.f, 30.f, 100.f, 0.01f };
	int fails = 0;
	for (unsigned m = 0; m < sizeof mags / sizeof mags[0]; m++) {
		for (int i = 0; i < n; i++) x[i] = mags[m] * frand();
		x[0] = 0.f; x[1] = -0.f; x[2] = 87.f; x[3] = -87.f; x[4] = 1e-7f; x[5] = -1e-7f;
		for (int i = 0; i < n; i++) { ref[0][i] = exp_gpu(x[i]); ref[1][i] = sigmoid_gpu(x[i]); ref[2][i] = silu_gpu(x[i]); ref[3][i] = softplus_gpu(x[i]); }
		CK(cuMemcpyHtoD(d[0], x, 4 * n));
		for (int i = 1; i < 5; i++) CK(cuMemsetD32(d[i], 0x7fc00000u, n));
		void *params[] = { &d[0], &d[1], &d[2], &d[3], &d[4] };
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		for (int i = 0; i < 4; i++) CK(cuMemcpyDtoH(out[i], d[i + 1], 4 * n));
		int bad[4] = { 0, 0, 0, 0 };
		double maxrel[4] = { 0, 0, 0, 0 };
		for (int f = 0; f < 4; f++) for (int i = 0; i < n; i++) {
			if (memcmp(&out[f][i], &ref[f][i], 4)) { if (bad[f] < 2) printf("  %s(%.9g): gpu %.9g ref %.9g\n", names[f], x[i], out[f][i], ref[f][i]); bad[f]++; }
			double lib = f == 0 ? exp(x[i]) : f == 1 ? 1 / (1 + exp(-x[i])) : f == 2 ? x[i] / (1 + exp(-x[i])) : log1p(exp(-fabs(x[i]))) + (x[i] > 0 ? x[i] : 0);
			double rel = lib != 0 ? fabs(ref[f][i] - lib) / fabs(lib) : fabs(ref[f][i]);
			if (rel > maxrel[f]) maxrel[f] = rel;
		}
		printf("mag %-5g:", mags[m]);
		for (int f = 0; f < 4; f++) printf(" %s %s (vs libm %.1e)", names[f], bad[f] ? "MISMATCH" : "exact", maxrel[f]);
		printf("\n");
		for (int f = 0; f < 4; f++) if (bad[f]) fails++;
	}
	return fails ? 3 : 0;
}
