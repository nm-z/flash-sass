/* hcops_test: hc_silu / hc_mix / hc_write kernels against ref/hc.c, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "hc.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 31;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

static int cmp(const char *what, const float *a, const float *b, int n) {
	int bad = 0;
	for (int i = 0; i < n; i++) if (memcmp(&a[i], &b[i], 4)) { if (bad < 3) printf("  %s[%d]: gpu %.9g ref %.9g\n", what, i, a[i], b[i]); bad++; }
	printf("%s: %s (%d of %d differ)\n", what, bad ? "MISMATCH" : "BIT-EXACT", bad, n);
	return bad != 0;
}

int main(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	int fails = 0;
	CUmodule mod; CUfunction fn;
	CUdeviceptr d0, d1, d2;
	CK(cuMemAlloc(&d0, 10240 * 4)); CK(cuMemAlloc(&d1, 10240 * 4)); CK(cuMemAlloc(&d2, 10240 * 4));
	float *a = malloc(10240 * 4), *b = malloc(10240 * 4), *c = malloc(10240 * 4), *r = malloc(10240 * 4), *g = malloc(10240 * 4);
	/* silu */
	for (int i = 0; i < 320; i++) a[i] = 40.f * frand();
	hc_silu_gpu(a, r);
	CK(cuModuleLoad(&mod, "hc_silu.cubin")); CK(cuModuleGetFunction(&fn, mod, "hcops"));
	CK(cuMemcpyHtoD(d0, a, 320 * 4)); CK(cuMemsetD32(d1, 0x7fc00000u, 320));
	unsigned n = 320;
	void *p1[] = { &d0, &d1, &d2, &n };
	CK(cuLaunchKernel(fn, 2, 1, 1, 256, 1, 1, 0, 0, p1, NULL)); CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(g, d1, 320 * 4));
	fails += cmp("hc_silu", g, r, 320);
	/* mix */
	for (int i = 0; i < 10240; i++) { a[i] = 6.f * frand(); b[i] = 2.f * frand(); }
	hc_mix_gpu(a, b, r);
	CK(cuModuleLoad(&mod, "hc_mix.cubin")); CK(cuModuleGetFunction(&fn, mod, "hcops"));
	CK(cuMemcpyHtoD(d0, a, 10240 * 4)); CK(cuMemcpyHtoD(d1, b, 10240 * 4)); CK(cuMemsetD32(d2, 0x7fc00000u, 2560));
	n = 2560;
	CK(cuLaunchKernel(fn, 10, 1, 1, 256, 1, 1, 0, 0, p1, NULL)); CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(g, d2, 2560 * 4));
	fails += cmp("hc_mix", g, r, 2560);
	/* write */
	for (int i = 0; i < 4; i++) a[i] = 6.f * frand();
	for (int i = 0; i < 2560; i++) b[i] = 2.f * frand();
	for (int i = 0; i < 10240; i++) c[i] = frand();
	memcpy(r, c, 10240 * 4);
	hc_write_gpu(a, b, r);
	CK(cuModuleLoad(&mod, "hc_write.cubin")); CK(cuModuleGetFunction(&fn, mod, "hcops"));
	CK(cuMemcpyHtoD(d0, a, 4 * 4)); CK(cuMemcpyHtoD(d1, b, 2560 * 4)); CK(cuMemcpyHtoD(d2, c, 10240 * 4));
	CK(cuLaunchKernel(fn, 10, 1, 1, 256, 1, 1, 0, 0, p1, NULL)); CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(g, d2, 10240 * 4));
	fails += cmp("hc_write", g, r, 10240);
	return fails ? 3 : 0;
}
