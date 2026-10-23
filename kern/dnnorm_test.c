/* Local DeltaNet norm plus six-destination norm/mix XOUT against CPU twins. */
#define _POSIX_C_SOURCE 200809L
#include "deltanet.h"
#include "hc.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 61;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

static int check_xout(const char *name, CUfunction fn, CUdeviceptr d0,
                      CUdeviceptr d1, CUdeviceptr d2, const float *reference,
                      unsigned n, int norm, float eps) {
	const uint32_t sentinel = 0x7fc01234u;
	const unsigned offsets[] = {0, norm ? 16u : 4u, norm ? 80u : 68u};
	const size_t words = (size_t)n + 2 * 64 + 20 + 4 * 5;
	uint32_t *got = malloc(words * sizeof *got);
	if (!got) exit(2);
	CUdeviceptr destination[6], table, entries[14], saved[14];
	for (unsigned d = 0; d < 6; d++) CK(cuMemAlloc(&destination[d], words * sizeof *got));
	CK(cuMemAlloc(&table, sizeof entries));
	int failures = 0;
	for (unsigned trial = 0; trial < sizeof offsets / sizeof *offsets; trial++) {
		unsigned xoff = offsets[trial], pad = 0, us = 10240;
		/* Different aligned bases catch table-slot and offset mistakes. All six
		 * destinations are separate allocations on the selected fixture GPU. */
		for (unsigned i = 0; i < 14; i++) entries[i] = 0xdeadbeefabc01234ull;
		for (unsigned d = 0; d < 6; d++) {
			CK(cuMemsetD32(destination[d], sentinel, words));
			unsigned slot = (d + trial) % 6;
			entries[4 + slot] = destination[d] + 4u * (64u + 4u * d);
		}
		memcpy(saved, entries, sizeof saved);
		CK(cuMemcpyHtoD(table, entries, sizeof entries));
		CUdeviceptr bases = table + 4 * sizeof(CUdeviceptr);
		/* DNNORM's xoff is at 0x168, after eps at 0x160 and a 0x164 pad.
		 * HCMIX's xoff immediately follows n/us at 0x158/0x15c. */
		void *norm_args[] = {&d0, &d1, &d2, &bases, &eps, &pad, &xoff};
		void *mix_args[] = {&d0, &d1, &bases, &n, &us, &xoff};
		unsigned grid = norm ? n / 128 : (n + 255) / 256;
		CK(cuLaunchKernel(fn, grid, 1, 1, norm ? 32 : 256, 1, 1, 0, 0,
		                 norm ? norm_args : mix_args, NULL));
		CK(cuCtxSynchronize());
		unsigned bad = 0, guard_bad = 0, missing = 0;
		for (unsigned d = 0; d < 6; d++) {
			CK(cuMemcpyDtoH(got, destination[d], words * sizeof *got));
			size_t begin = 64u + 4u * d + xoff / 4u;
			for (size_t i = 0; i < words; i++) {
				int active = i >= begin && i - begin < n;
				uint32_t expected = sentinel;
				if (active) memcpy(&expected, reference + (i - begin), sizeof expected);
				if (got[i] != expected) {
					if (bad < 3) printf("  %s dest%u word%zu got=%08x expected=%08x\n", name, d, i, got[i], expected);
					bad++; guard_bad += !active;
				}
				if (active && got[i] == sentinel) missing++;
			}
		}
		CK(cuMemcpyDtoH(entries, table, sizeof entries));
		int table_bad = memcmp(entries, saved, sizeof entries) != 0;
		printf("%s n=%u xoff=%u destinations=6: mismatches=%u guard_errors=%u missing=%u table_changed=%d %s\n",
		       name, n, xoff, bad, guard_bad, missing, table_bad,
		       bad || missing || table_bad ? "FAIL" : "BIT-EXACT");
		failures += bad || missing || table_bad;
	}
	CK(cuMemFree(table));
	for (unsigned d = 0; d < 6; d++) CK(cuMemFree(destination[d]));
	free(got);
	return failures;
}

int main(void) {
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "dnnorm.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "dnnorm"));
	CUmodule norm_mod, mix_mod; CUfunction norm_fn, mix_fn;
	CK(cuModuleLoad(&norm_mod, "dnnorm_xout.cubin"));
	CK(cuModuleGetFunction(&norm_fn, norm_mod, "dnnorm"));
	CK(cuModuleLoad(&mix_mod, "hc_mix_xout.cubin"));
	CK(cuModuleGetFunction(&mix_fn, mix_mod, "hcops"));
	float *o = malloc(6144 * 4), *g = malloc(6144 * 4), *y = malloc(6144 * 4), *yg = malloc(6144 * 4), w[128];
	CUdeviceptr dO, dW, dG, dY;
	CK(cuMemAlloc(&dO, 6144 * 4)); CK(cuMemAlloc(&dW, 512)); CK(cuMemAlloc(&dG, 6144 * 4)); CK(cuMemAlloc(&dY, 6144 * 4));
	int fails = 0;
	for (int trial = 0; trial < 3; trial++) {
		float sc = trial == 1 ? 1e-3f : trial == 2 ? 30.f : 1.f;
		for (int i = 0; i < 6144; i++) { o[i] = sc * frand(); g[i] = 6.f * frand(); }
		for (int i = 0; i < 128; i++) w[i] = 1.f + 0.5f * frand();
		dnnorm_gpu(o, w, g, eps, y);
		CK(cuMemcpyHtoD(dO, o, 6144 * 4)); CK(cuMemcpyHtoD(dW, w, 512)); CK(cuMemcpyHtoD(dG, g, 6144 * 4)); CK(cuMemsetD32(dY, 0x7fc00000u, 6144));
		void *params[] = { &dO, &dW, &dG, &dY, &eps };
		CK(cuLaunchKernel(fn, 48, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(yg, dY, 6144 * 4));
		int bad = 0;
		for (int i = 0; i < 6144; i++) if (memcmp(&y[i], &yg[i], 4)) { if (bad < 3) printf("  y[%d]: gpu %.9g ref %.9g\n", i, yg[i], y[i]); bad++; }
		printf("trial %d: %s (%d of 6144 differ)\n", trial, bad ? "MISMATCH" : "BIT-EXACT", bad);
		if (bad) fails++;
		fails += check_xout("dnnorm_xout", norm_fn, dO, dW, dG, y, 6144, 1, eps);
	}
	float *up = malloc(10240 * sizeof *up), *xh = malloc(10240 * sizeof *xh);
	if (!up || !xh) exit(2);
	CUdeviceptr dUp, dXh;
	CK(cuMemAlloc(&dUp, 10240 * sizeof *up)); CK(cuMemAlloc(&dXh, 10240 * sizeof *xh));
	for (int trial = 0; trial < 3; trial++) {
		for (unsigned i = 0; i < 10240; i++) { up[i] = 6.f * frand(); xh[i] = 2.f * frand(); }
		hc_mix_gpu(up, xh, y);
		CK(cuMemcpyHtoD(dUp, up, 10240 * sizeof *up)); CK(cuMemcpyHtoD(dXh, xh, 10240 * sizeof *xh));
		fails += check_xout("hc_mix_xout", mix_fn, dUp, dXh, 0, y, 137, 0, eps);
		fails += check_xout("hc_mix_xout", mix_fn, dUp, dXh, 0, y, 2560, 0, eps);
	}
	free(up); free(xh); free(o); free(g); free(y); free(yg);
	CK(cuModuleUnload(mix_mod)); CK(cuModuleUnload(norm_mod)); CK(cuModuleUnload(mod));
	CK(cuCtxDestroy(ctx));
	return fails ? 3 : 0;
}
