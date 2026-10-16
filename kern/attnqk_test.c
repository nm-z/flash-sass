/* attnqk_test: q/k RMSNorm + RoPE + KV append against qk_norm_rope_gpu, bit for bit. */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static uint32_t rs = 41;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }

int main(void) {
	const int T = 5;   /* positions exercised */
	float eps = 1e-6f;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "attnqk.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "attnqk"));
	float *q = malloc(12288 * 4), *k = malloc(512 * 4), *v = malloc(512 * 4), qn[256], kn[256], cs[32], sn[32];
	float *qo = malloc(6144 * 4), *qog = malloc(6144 * 4), *vc = malloc(T * 2048), *vcg = malloc(T * 2048), kt[256];
	size_t kcb = ATTN_KC_FLOATS(T) * 4;
	float *kc = malloc(kcb), *kcg = malloc(kcb);
	for (size_t i = 0; i < kcb / 4; i++) ((uint32_t *)kc)[i] = 0x7fc00000u;
	for (int i = 0; i < 256; i++) { qn[i] = 1.f + 0.3f * frand(); kn[i] = 1.f + 0.3f * frand(); }
	CUdeviceptr dQ, dK, dV, dQN, dKN, dCS, dSN, dQO, dKC, dVC;
	CK(cuMemAlloc(&dQ, 12288 * 4)); CK(cuMemAlloc(&dK, 512 * 4)); CK(cuMemAlloc(&dV, 512 * 4)); CK(cuMemAlloc(&dQN, 1024)); CK(cuMemAlloc(&dKN, 1024));
	CK(cuMemAlloc(&dCS, 128)); CK(cuMemAlloc(&dSN, 128)); CK(cuMemAlloc(&dQO, 6144 * 4)); CK(cuMemAlloc(&dKC, kcb)); CK(cuMemAlloc(&dVC, T * 2048));
	CK(cuMemcpyHtoD(dQN, qn, 1024)); CK(cuMemcpyHtoD(dKN, kn, 1024));
	CK(cuMemsetD32(dKC, 0x7fc00000u, kcb / 4)); CK(cuMemsetD32(dVC, 0x7fc00000u, T * 512));
	int fails = 0;
	for (unsigned t = 0; t < (unsigned)T; t++) {
		for (int i = 0; i < 12288; i++) q[i] = 3.f * frand();
		for (int i = 0; i < 512; i++) { k[i] = 3.f * frand(); v[i] = frand(); }
		rope_table_gpu(t * 977 + 3, 1e7f, cs, sn);
		for (int h = 0; h < 24; h++) qk_norm_rope_gpu(q + 512 * h, qn, eps, cs, sn, qo + 256 * h);
		for (int h = 0; h < 2; h++) { qk_norm_rope_gpu(k + 256 * h, kn, eps, cs, sn, kt); for (int d = 0; d < 256; d++) kc[attn_kidx(t, h, d)] = kt[d]; memcpy(vc + 512 * t + 256 * h, v + 256 * h, 1024); }
		CK(cuMemcpyHtoD(dQ, q, 12288 * 4)); CK(cuMemcpyHtoD(dK, k, 2048)); CK(cuMemcpyHtoD(dV, v, 2048));
		CK(cuMemcpyHtoD(dCS, cs, 128)); CK(cuMemcpyHtoD(dSN, sn, 128)); CK(cuMemsetD32(dQO, 0x7fc00000u, 6144));
		void *params[] = { &dQ, &dK, &dV, &dQN, &dKN, &dCS, &dSN, &dQO, &dKC, &dVC, &t, &eps };
		CK(cuLaunchKernel(fn, 26, 1, 1, 32, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		CK(cuMemcpyDtoH(qog, dQO, 6144 * 4));
		int bq = 0;
		for (int i = 0; i < 6144; i++) if (memcmp(&qo[i], &qog[i], 4)) { if (bq < 3) printf("  qo[%d] (head %d dim %d): gpu %.9g ref %.9g\n", i, i / 256, i % 256, qog[i], qo[i]); bq++; }
		printf("t=%u: q %s (%d of 6144 differ)\n", t, bq ? "MISMATCH" : "BIT-EXACT", bq);
		if (bq && t == 0) { printf("  head 0 bad dims:"); for (int i = 0; i < 256; i++) if (memcmp(&qo[i], &qog[i], 4)) printf(" %d", i); printf("\n  head 1 bad dims:"); for (int i = 256; i < 512; i++) if (memcmp(&qo[i], &qog[i], 4)) printf(" %d", i - 256); printf("\n"); }
		if (bq) fails++;
	}
	CK(cuMemcpyDtoH(kcg, dKC, kcb)); CK(cuMemcpyDtoH(vcg, dVC, T * 2048));
	int bk = 0, bv = 0;
	for (size_t i = 0; i < kcb / 4; i++) if (memcmp(&kc[i], &kcg[i], 4)) bk++;
	for (int i = 0; i < T * 512; i++) if (memcmp(&vc[i], &vcg[i], 4)) bv++;
	printf("  V bad (pos 0):"); for (int i = 0; i < 512; i++) if (memcmp(&vc[i], &vcg[i], 4)) printf(" %d", i); printf("\n");
	printf("KV cache: K %s (%d differ), V %s (%d differ)\n", bk ? "MISMATCH" : "BIT-EXACT", bk, bv ? "MISMATCH" : "BIT-EXACT", bv);
	return (fails || bk || bv) ? 3 : 0;
}
