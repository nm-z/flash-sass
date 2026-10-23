/* attn_test: attnpart + attnmerge against attn_partials_gpu / attn_merge_gpu, bit for bit.
 * usage: attn_test [n keys] [timing iterations] */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "fmath.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 77;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

static int cmp(const char *what, const float *ref, const float *got, size_t cnt, int show) {
	int bad = 0;
	for (size_t i = 0; i < cnt; i++) if (memcmp(&ref[i], &got[i], 4)) {
		if (bad < show) printf("  %s[%zu]: gpu %.9g (%08x) ref %.9g (%08x)\n", what, i, got[i], *(uint32_t *)&got[i], ref[i], *(uint32_t *)&ref[i]);
		bad++;
	}
	printf("%s: %s (%d of %zu differ)\n", what, bad ? "MISMATCH" : "BIT-EXACT", bad, cnt);
	return bad;
}

int main(int argc, char **argv) {
	unsigned n = argc > 1 ? (unsigned)strtoul(argv[1], 0, 0) : 3000;
	int iters = argc > 2 ? atoi(argv[2]) : 0;
	unsigned npos = (n + 8192 + 64 + 31) / 32 * 32;      /* chunk round-up plus the loops' stray prefetches */
	size_t kcf = ATTN_KC_FLOATS(npos), vcf = (size_t)npos * 512;
	float *qo = malloc(6144 * 4), *gate = malloc(12288 * 4), *kc = calloc(kcf, 4), *vc = calloc(vcf, 4);
	float *out = malloc(6144 * 4), *outg = malloc(6144 * 4);
	float *pm = malloc(24 * 8 * 4), *pd = malloc(24 * 8 * 4), *po = malloc(24 * 8 * 256 * 4);
	float *pmg = malloc(24 * 8 * 4), *pdg = malloc(24 * 8 * 4), *pog = malloc(24 * 8 * 256 * 4);
	for (int i = 0; i < 6144; i++) qo[i] = 2.f * frand();
	for (int i = 0; i < 12288; i++) gate[i] = 4.f * frand();
	for (unsigned t = 0; t < n; t++)
		for (int kv = 0; kv < 2; kv++)
			for (int d = 0; d < 256; d++) { kc[attn_kidx(t, kv, d)] = 2.f * frand(); vc[(size_t)t * 512 + 256 * kv + d] = frand(); }
	double t0 = now();
	attn_partials_gpu(qo, kc, vc, n, pm, pd, po);
	attn_merge_gpu(pm, pd, po, gate, out);
	printf("n=%u: reference %.2f s\n", n, now() - t0);

	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mp, mm; CUfunction fp, fm;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mp, "attnpart.cubin")); CK(cuModuleGetFunction(&fp, mp, "attnpart"));
	CK(cuModuleLoad(&mm, "attnmerge.cubin")); CK(cuModuleGetFunction(&fm, mm, "attnmerge"));
	int nregs, maxb, nregs2;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fp));
	CK(cuFuncGetAttribute(&nregs2, CU_FUNC_ATTRIBUTE_NUM_REGS, fm));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fp, 256, 0));
	printf("attnpart: %d regs, %d CTAs/SM; attnmerge: %d regs\n", nregs, maxb, nregs2);
	CUdeviceptr dQO, dKC, dVC, dPM, dPD, dPO, dGATE, dOUT;
	CK(cuMemAlloc(&dQO, 6144 * 4)); CK(cuMemAlloc(&dKC, kcf * 4)); CK(cuMemAlloc(&dVC, vcf * 4));
	CK(cuMemAlloc(&dPM, 24 * 8 * 4)); CK(cuMemAlloc(&dPD, 24 * 8 * 4)); CK(cuMemAlloc(&dPO, 24 * 8 * 256 * 4));
	CK(cuMemAlloc(&dGATE, 12288 * 4)); CK(cuMemAlloc(&dOUT, 6144 * 4));
	CK(cuMemcpyHtoD(dQO, qo, 6144 * 4)); CK(cuMemcpyHtoD(dKC, kc, kcf * 4)); CK(cuMemcpyHtoD(dVC, vc, vcf * 4));
	CK(cuMemcpyHtoD(dGATE, gate, 12288 * 4));
	CK(cuMemsetD32(dPM, 0x7fc00000u, 24 * 8)); CK(cuMemsetD32(dPD, 0x7fc00000u, 24 * 8));
	CK(cuMemsetD32(dPO, 0x7fc00000u, 24 * 8 * 256)); CK(cuMemsetD32(dOUT, 0x7fc00000u, 6144));
	void *pp[] = { &dQO, &dKC, &dVC, &dPM, &dPD, &dPO, &n };
	void *pmg_[] = { &dPM, &dPD, &dPO, &dGATE, &dOUT };
	CK(cuLaunchKernel(fp, 32, 1, 1, 256, 1, 1, 0, 0, pp, NULL));
	CK(cuLaunchKernel(fm, 24, 1, 1, 256, 1, 1, 0, 0, pmg_, NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(pmg, dPM, 24 * 8 * 4)); CK(cuMemcpyDtoH(pdg, dPD, 24 * 8 * 4));
	CK(cuMemcpyDtoH(pog, dPO, 24 * 8 * 256 * 4)); CK(cuMemcpyDtoH(outg, dOUT, 6144 * 4));
	if (getenv("ATTN_PRINT")) for (int c = 0; c < 2; c++) printf("hq0 c%d: m gpu %.9g ref %.9g | den gpu %.9g ref %.9g | po0 gpu %.9g ref %.9g\n", c, pmg[c], pm[c], pdg[c], pd[c], pog[c * 256], po[c * 256]);
	int bad = 0;
	bad += cmp("pm", pm, pmg, 24 * 8, 4);
	bad += cmp("pd", pd, pdg, 24 * 8, 4);
	bad += cmp("po", po, pog, 24 * 8 * 256, 4);
	bad += cmp("out", out, outg, 6144, 4);
	if (iters > 0) {
		CK(cuLaunchKernel(fp, 32, 1, 1, 256, 1, 1, 0, 0, pp, NULL)); CK(cuCtxSynchronize());
		double t1 = now();
		for (int it = 0; it < iters; it++) {
			CK(cuLaunchKernel(fp, 32, 1, 1, 256, 1, 1, 0, 0, pp, NULL));
			CK(cuLaunchKernel(fm, 24, 1, 1, 256, 1, 1, 0, 0, pmg_, NULL));
		}
		CK(cuCtxSynchronize());
		double dt = (now() - t1) / iters;
		printf("timing: %.1f us per step, %.1f GB/s of K+V (%.1f MB)\n", dt * 1e6, (double)n * 4096 / dt / 1e9, (double)n * 4096 / 1e6);
	}
	return bad ? 3 : 0;
}
