/* argmax_test: argmax over n logits (ties -> lowest index) on random vectors with
 * planted maxima and ties; repeated calls check the counter reset. usage: argmax_test [n] [G] */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 3;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

static unsigned argmax_gpu(const float *x, unsigned n, float *mx) {
	unsigned im = 0;
	float m = x[0];
	for (unsigned i = 1; i < n; i++) if (x[i] > m) { m = x[i]; im = i; }
	*mx = m;
	return im;
}

int main(int argc, char **argv) {
	unsigned n = argc > 1 ? (unsigned)strtoul(argv[1], 0, 0) : 248320;
	int G = argc > 2 ? atoi(argv[2]) : 32;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "argmax.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "argmax"));
	CUdeviceptr dX, dOUT, dPART, dCNT;
	CK(cuMemAlloc(&dX, 4 * (size_t)n)); CK(cuMemAlloc(&dOUT, 8)); CK(cuMemAlloc(&dPART, 8 * 32)); CK(cuMemAlloc(&dCNT, 4));
	CK(cuMemsetD32(dCNT, 0, 1));
	float *x = malloc(4 * (size_t)n);
	int fails = 0;
	for (int trial = 0; trial < 12; trial++) {
		for (unsigned i = 0; i < n; i++) x[i] = 10.f * frand();
		unsigned where = (unsigned)(((uint64_t)(rs = rs * 1664525u + 1013904223u) >> 8) % n);
		switch (trial % 4) {
		case 1: x[where] = 20.f; break;                                   /* a planted maximum */
		case 2: x[where] = 20.f; x[(where + 7) % n] = 20.f; break;        /* a tie: lowest index wins */
		case 3: x[n - 1] = 20.f; x[0] = 20.f; break;                      /* tie at both ends -> 0 */
		default: break;
		}
		float mref, mgot; unsigned iref = argmax_gpu(x, n, &mref), out[2];
		CK(cuMemcpyHtoD(dX, x, 4 * (size_t)n));
		CK(cuMemsetD32(dOUT, 0xffffffffu, 2));
		unsigned ug = (unsigned)G; void *params[] = { &dX, &dOUT, &dPART, &dCNT, &n, &ug };
		CK(cuLaunchKernel(fn, G, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(out, dOUT, 8));
		memcpy(&mgot, &out[1], 4);
		int ok = out[0] == iref && mgot == mref;
		printf("trial %2d (n=%u, G=%d): gpu idx %u max %g, ref idx %u max %g -> %s\n", trial, n, G, out[0], mgot, iref, mref, ok ? "OK" : "MISMATCH");
		if (!ok) fails++;
	}
	unsigned cnt; CK(cuMemcpyDtoH(&cnt, dCNT, 4));
	printf("counter after %d calls: %u (want 0)\n", 12, cnt);
	return (fails || cnt) ? 3 : 0;
}
