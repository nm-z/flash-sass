/* qsa_test: the QSA ops through the persistent kernel (die.cubin) against ref/qsa.c, token by token.
 *   qsa_test kv <T> <NBLK>     attnqkq (int8 KV append) + kvgather + attnpartn + attnmerge: caches, compact caches and the
 *                              attention output bit for bit; the attended blocks are a pseudo-random ascending selection
 * Run with the die index in CUDA_VISIBLE_DEVICES. */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "qsa.h"
#include "fmath.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

#include "dieops.h"
static uint32_t rs = 4242;
static uint32_t urand(void) { rs = rs * 1664525u + 1013904223u; return rs >> 8; }
static float frand(void) { return (urand() & 0xffff) / 32768.f - 1.f; }

typedef struct { uint32_t w[32]; } desc_t;
static void d_init(desc_t *d, int op, unsigned units) { memset(d, 0, sizeof *d); d->w[0] = (uint32_t)op; d->w[1] = units; }
static void d_ptr(desc_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w + 1] = (uint32_t)(p >> 32); }
static void d_u32(desc_t *d, int addr, uint32_t v) { d->w[2 + (addr - 0x140) / 4] = v; }
static void d_f32(desc_t *d, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); d_u32(d, addr, u); }
/* token-relative: the kernel adds its token counter to this parameter */
static void d_tok(desc_t *d, int addr, uint32_t base) { d_u32(d, addr, base); d->w[31] |= 1u << (2 + (addr - 0x140) / 4); }

static CUmodule mod; static CUfunction fn; static CUdeviceptr dPROG, dCTL;
static void dev_init(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	CK(cuMemAlloc(&dPROG, 128 * 256)); CK(cuMemAlloc(&dCTL, 8192)); CK(cuMemsetD8(dCTL, 0, 8192));
}
static CUdeviceptr dalloc(size_t n) { CUdeviceptr p; CK(cuMemAlloc(&p, n ? n : 4)); CK(cuMemsetD8(p, 0, n ? n : 4)); return p; }
/* run the program once at token t (the kernel's token counter starts at t) */
static void run_token(const desc_t *prog, unsigned nph, unsigned t) {
	CK(cuMemcpyHtoD(dPROG, prog, sizeof(desc_t) * nph));
	CK(cuMemcpyHtoD(dCTL + 8, &t, 4));
	unsigned ntok = t + 1; void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
}
static int cmpb(const char *what, const void *ref, const void *got, size_t n, int *shown) {
	if (!memcmp(ref, got, n)) return 0;
	size_t i = 0; while (((const uint8_t *)ref)[i] == ((const uint8_t *)got)[i]) i++;
	if ((*shown)++ < 6) printf("  %s differs at byte %zu: gpu %02x ref %02x\n", what, i, ((const uint8_t *)got)[i], ((const uint8_t *)ref)[i]);
	return 1;
}

static int test_kv(unsigned T, unsigned NBLK) {
	const float eps = 1e-6f;
	unsigned npos = T + 64;
	float *rcs = malloc(4 * 32 * (size_t)npos), *rsn = malloc(4 * 32 * (size_t)npos);
	for (unsigned t = 0; t < npos; t++) rope_table_gpu(t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	float qn[256], kn[256]; for (int i = 0; i < 256; i++) { qn[i] = 0.5f + 0.5f * frand(); kn[i] = 0.5f + 0.5f * frand(); }
	/* host mirrors of the caches */
	int8_t *kq = calloc((size_t)npos * 512, 1), *vq = calloc((size_t)npos * 512, 1);
	uint16_t *ks = calloc((size_t)npos * 16, 2), *vs = calloc((size_t)npos * 16, 2);
	unsigned cpos = 4 * (NBLK + 1) + 2048; cpos = (cpos + 31) / 32 * 32;
	size_t kcf = ATTN_KC_FLOATS(cpos), vcf = (size_t)cpos * 512;
	CUdeviceptr dQG = dalloc(4 * 12288), dK = dalloc(4 * 512), dV = dalloc(4 * 512), dQN = dalloc(4 * 256), dKN = dalloc(4 * 256), dCS = dalloc(4 * 32 * (size_t)npos), dSN = dalloc(4 * 32 * (size_t)npos);
	CUdeviceptr dQO = dalloc(4 * 6144), dKQ = dalloc((size_t)npos * 512), dVQ = dalloc((size_t)npos * 512), dKS = dalloc((size_t)npos * 32), dVS = dalloc((size_t)npos * 32);
	CUdeviceptr dKC = dalloc(kcf * 4), dVC = dalloc(vcf * 4), dSEL = dalloc(4 * (NBLK + 8)), dPM = dalloc(4 * 24 * 8), dPD = dalloc(4 * 24 * 8), dPO = dalloc(4 * 24 * 8 * 256), dAOUT = dalloc(4 * 6144);
	CK(cuMemcpyHtoD(dQN, qn, 4 * 256)); CK(cuMemcpyHtoD(dKN, kn, 4 * 256)); CK(cuMemcpyHtoD(dCS, rcs, 4 * 32 * (size_t)npos)); CK(cuMemcpyHtoD(dSN, rsn, 4 * 32 * (size_t)npos));
	desc_t prog[4];
	d_init(&prog[0], OP_ATTNQKQ, 26);
	d_ptr(&prog[0], 0x140, dQG); d_ptr(&prog[0], 0x148, dK); d_ptr(&prog[0], 0x150, dV); d_ptr(&prog[0], 0x158, dQN); d_ptr(&prog[0], 0x160, dKN);
	d_ptr(&prog[0], 0x168, dCS); d_ptr(&prog[0], 0x170, dSN); d_ptr(&prog[0], 0x178, dQO); d_ptr(&prog[0], 0x180, dKQ); d_ptr(&prog[0], 0x188, dVQ);
	d_tok(&prog[0], 0x190, 0); d_f32(&prog[0], 0x194, eps); d_ptr(&prog[0], 0x198, dKS); d_ptr(&prog[0], 0x1a0, dVS);
	d_init(&prog[1], OP_KVGATHER, NBLK + 1);
	d_ptr(&prog[1], 0x140, dKQ); d_ptr(&prog[1], 0x148, dKS); d_ptr(&prog[1], 0x150, dVQ); d_ptr(&prog[1], 0x158, dVS); d_ptr(&prog[1], 0x160, dKC); d_ptr(&prog[1], 0x168, dVC);
	d_ptr(&prog[1], 0x170, dSEL); d_tok(&prog[1], 0x178, 0); d_u32(&prog[1], 0x17c, NBLK);
	d_init(&prog[2], OP_ATTNPARTN, 32);
	d_ptr(&prog[2], 0x140, dQO); d_ptr(&prog[2], 0x148, dKC); d_ptr(&prog[2], 0x150, dVC); d_ptr(&prog[2], 0x158, dPM); d_ptr(&prog[2], 0x160, dPD); d_ptr(&prog[2], 0x168, dPO);
	d_tok(&prog[2], 0x170, 1); d_u32(&prog[2], 0x174, NBLK);
	d_init(&prog[3], OP_ATTNMERGE, 24);
	d_ptr(&prog[3], 0x140, dPM); d_ptr(&prog[3], 0x148, dPD); d_ptr(&prog[3], 0x150, dPO); d_ptr(&prog[3], 0x158, dQG); d_ptr(&prog[3], 0x160, dAOUT);

	float *qg = malloc(4 * 12288), *k = malloc(4 * 512), *v = malloc(4 * 512), *qo = malloc(4 * 6144), *out = malloc(4 * 6144), *outg = malloc(4 * 6144);
	float *kc = calloc(kcf, 4), *vc = calloc(vcf, 4), *kcg = malloc(kcf * 4), *vcg = malloc(vcf * 4);
	int8_t *kqg = malloc((size_t)npos * 512), *vqg = malloc((size_t)npos * 512); uint16_t *ksg = malloc((size_t)npos * 32), *vsg = malloc((size_t)npos * 32);
	uint32_t *sel = malloc(4 * (NBLK + 8));
	int bad = 0, shown = 0;
	for (unsigned t = 0; t < T; t++) {
		for (int i = 0; i < 12288; i++) qg[i] = 2.f * frand();
		for (int i = 0; i < 512; i++) { k[i] = 2.f * frand(); v[i] = frand(); }
		if (t % 7 == 3) for (int i = 0; i < 512; i++) v[i] *= 1e-4f;          /* tiny scales (fp16 subnormals) */
		if (t % 11 == 5) for (int i = 256; i < 512; i++) v[i] = 0.f;          /* an all-zero kv head */
		if (t % 13 == 6) for (int i = 0; i < 512; i++) v[i] *= 3e3f;
		unsigned nv = (t + 1) >> 2, ns = nv < NBLK ? nv : NBLK;
		if (nv > NBLK) {   /* a pseudo-random ascending selection of NBLK of the nv blocks */
			unsigned *perm = malloc(4 * nv); for (unsigned i = 0; i < nv; i++) perm[i] = i;
			for (unsigned i = 0; i < NBLK; i++) { unsigned j = i + urand() % (nv - i), tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; sel[i] = perm[i]; }
			for (unsigned i = 1; i < NBLK; i++) { uint32_t x = sel[i]; int j = (int)i - 1; while (j >= 0 && sel[j] > x) { sel[j + 1] = sel[j]; j--; } sel[j + 1] = x; }
			free(perm);
			CK(cuMemcpyHtoD(dSEL, sel, 4 * NBLK));
		}
		CK(cuMemcpyHtoD(dQG, qg, 4 * 12288)); CK(cuMemcpyHtoD(dK, k, 4 * 512)); CK(cuMemcpyHtoD(dV, v, 4 * 512));
		run_token(prog, 4, t);
		/* the twin: q/k norm + RoPE, int8 append, gather, attention */
		for (int h = 0; h < 24; h++) qk_norm_rope_gpu(qg + 512 * h, qn, eps, rcs + 32 * t, rsn + 32 * t, qo + 256 * h);
		for (int h = 0; h < 2; h++) {
			float y[256];
			qk_norm_rope_gpu(k + 256 * h, kn, eps, rcs + 32 * t, rsn + 32 * t, y);
			for (int b = 0; b < 8; b++) {
				size_t row = (size_t)t * 2 + h;
				kv_quant32_gpu(y + 32 * b, kq + row * 256 + 32 * b, &ks[row * 8 + b]);
				kv_quant32_gpu(v + 256 * h + 32 * b, vq + row * 256 + 32 * b, &vs[row * 8 + b]);
			}
		}
		unsigned n = qsa_gather_gpu(kq, ks, vq, vs, sel, t, NBLK, kc, vc);
		attn_decode_gpu(qo, kc, vc, n, qg, out);
		CK(cuMemcpyDtoH(kqg, dKQ, (size_t)npos * 512)); CK(cuMemcpyDtoH(vqg, dVQ, (size_t)npos * 512)); CK(cuMemcpyDtoH(ksg, dKS, (size_t)npos * 32)); CK(cuMemcpyDtoH(vsg, dVS, (size_t)npos * 32));
		{ float *qog = malloc(4 * 6144); CK(cuMemcpyDtoH(qog, dQO, 4 * 6144)); int bq = cmpb("qo", qo, qog, 4 * 6144, &shown); if (bq && !bad) printf("  (qo differs at t = %u)\n", t); free(qog); }
		if (getenv("QSA_DUMP") && t == 0) { printf("kq gpu:"); for (int i = 0; i < 40; i++) printf(" %d", kqg[i]); printf("\n    ref:"); for (int i = 0; i < 40; i++) printf(" %d", kq[i]); printf("\nks gpu:"); for (int i = 0; i < 16; i++) printf(" %04x", ksg[i]); printf("\n    ref:"); for (int i = 0; i < 16; i++) printf(" %04x", ks[i]); printf("\nvq gpu:"); for (int i = 0; i < 40; i++) printf(" %d", vqg[i]); printf("\n    ref:"); for (int i = 0; i < 40; i++) printf(" %d", vq[i]); printf("\n"); }
		int b = 0;
		b |= cmpb("kq", kq + (size_t)t * 512, kqg + (size_t)t * 512, 512, &shown); b |= cmpb("vq", vq + (size_t)t * 512, vqg + (size_t)t * 512, 512, &shown);
		b |= cmpb("ks", ks + (size_t)t * 16, ksg + (size_t)t * 16, 32, &shown); b |= cmpb("vs", vs + (size_t)t * 16, vsg + (size_t)t * 16, 32, &shown);
		CK(cuMemcpyDtoH(kcg, dKC, kcf * 4)); CK(cuMemcpyDtoH(vcg, dVC, vcf * 4));
		unsigned nb = ns + ((t + 1) & 3 ? 1 : 0);
		for (unsigned j = 0; j < nb; j++) for (unsigned i = 0; i < 4; i++) {
			unsigned slot = 4 * j + i;
			for (int kv = 0; kv < 2; kv++) for (int d = 0; d < 256; d += 4) {
				b |= cmpb("kc", &kc[attn_kidx(slot, kv, d)], &kcg[attn_kidx(slot, kv, d)], 16, &shown);
				b |= cmpb("vc", &vc[(size_t)slot * 512 + 256 * kv + d], &vcg[(size_t)slot * 512 + 256 * kv + d], 16, &shown);
			}
		}
		CK(cuMemcpyDtoH(outg, dAOUT, 4 * 6144));
		b |= cmpb("out", out, outg, 4 * 6144, &shown);
		if (b && !bad) printf("first mismatch at t = %u (n = %u keys, %u blocks selected)\n", t, n, ns);
		bad += b;
	}
	printf("kv: %u tokens, NBLK %u: %s (%d tokens differ)\n", T, NBLK, bad ? "MISMATCH" : "BIT-EXACT", bad);
	return bad;
}

static size_t pkt_idx(unsigned b, int d) { return (size_t)(b >> 5) * 4096 + (size_t)(d >> 2) * 128 + (b & 31) * 4 + (d & 3); }

static int test_idxqk(unsigned T) {
	const float eps = 1e-6f;
	unsigned npos = T + 64, nblk = (T + 3) / 4, ngrp = (nblk + 31) / 32 + 1;
	float *rcs = malloc(4 * 32 * (size_t)npos), *rsn = malloc(4 * 32 * (size_t)npos);
	for (unsigned t = 0; t < npos; t++) rope_table_gpu(t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	float qn[128], kn[128]; for (int i = 0; i < 128; i++) { qn[i] = 0.5f + 0.5f * frand(); kn[i] = 0.5f + 0.5f * frand(); }
	CUdeviceptr dIQK = dalloc(4 * 640), dQN = dalloc(4 * 128), dKN = dalloc(4 * 128), dCS = dalloc(4 * 32 * (size_t)npos), dSN = dalloc(4 * 32 * (size_t)npos), dIQ = dalloc(4 * 512), dRING = dalloc(4 * 512), dPKT = dalloc((size_t)ngrp * 8192), dHST = dalloc(4 * 1280);
	CK(cuMemcpyHtoD(dQN, qn, 4 * 128)); CK(cuMemcpyHtoD(dKN, kn, 4 * 128)); CK(cuMemcpyHtoD(dCS, rcs, 4 * 32 * (size_t)npos)); CK(cuMemcpyHtoD(dSN, rsn, 4 * 32 * (size_t)npos));
	CK(cuMemsetD32(dHST, 0xdeadbeefu, 1280));
	desc_t prog[1];
	d_init(&prog[0], OP_IDXQK, 6);
	d_ptr(&prog[0], 0x140, dIQK); d_ptr(&prog[0], 0x148, dQN); d_ptr(&prog[0], 0x150, dKN); d_ptr(&prog[0], 0x158, dCS); d_ptr(&prog[0], 0x160, dSN); d_ptr(&prog[0], 0x168, dIQ);
	d_ptr(&prog[0], 0x170, dRING); d_ptr(&prog[0], 0x178, dPKT); d_ptr(&prog[0], 0x180, dHST); d_tok(&prog[0], 0x188, 0); d_f32(&prog[0], 0x18c, eps);
	float iqk[640], iq[512], iqg[512], ring[4 * 128];
	uint16_t *pool = calloc((size_t)(nblk + 1) * 128, 2), *pktg = malloc((size_t)ngrp * 8192), *pkt = calloc((size_t)ngrp * 4096, 2);
	uint32_t hst[1280];
	int bad = 0, shown = 0;
	for (unsigned t = 0; t < T; t++) {
		for (int i = 0; i < 640; i++) iqk[i] = 1.5f * frand();
		CK(cuMemcpyHtoD(dIQK, iqk, 4 * 640));
		run_token(prog, 1, t);
		for (int h = 0; h < 4; h++) idx_norm_rope_gpu(iqk + 128 * h, qn, eps, rcs + 32 * t, rsn + 32 * t, iq + 128 * h);
		memcpy(ring + 128 * (t & 3), iqk + 512, 4 * 128);
		if ((t & 3) == 3) {
			unsigned b = t >> 2; uint16_t p[128];
			idx_pool_gpu(ring, kn, eps, rcs + 32 * (t - 3), rsn + 32 * (t - 3), p);
			for (int d = 0; d < 128; d++) { pool[(size_t)b * 128 + d] = p[d]; pkt[pkt_idx(b, d)] = p[d]; }
		}
		int b = 0;
		CK(cuMemcpyDtoH(iqg, dIQ, 4 * 512)); b |= cmpb("iq", iq, iqg, 4 * 512, &shown);
		float ringg[512]; CK(cuMemcpyDtoH(ringg, dRING, 4 * 512)); b |= cmpb("ring", ring, ringg, 4 * 128 * ((t & 3) + 1), &shown);
		CK(cuMemcpyDtoH(pktg, dPKT, (size_t)ngrp * 8192)); b |= cmpb("pool", pkt, pktg, (size_t)ngrp * 8192, &shown);
		CK(cuMemcpyDtoH(hst, dHST, 4 * 1280)); for (int i = 0; i < 1280; i++) if (hst[i]) { if (shown++ < 4) printf("  HST[%d] = %08x not zeroed\n", i, hst[i]); b = 1; break; }
		CK(cuMemsetD32(dHST, 0xdeadbeefu, 1280));
		if (b && !bad) printf("first mismatch at t = %u\n", t);
		bad += b;
	}
	printf("idxqk: %u tokens: %s (%d tokens differ)\n", T, bad ? "MISMATCH" : "BIT-EXACT", bad);
	return bad;
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: qsa_test kv <T> <NBLK>\n"); return 2; }
	dev_init();
	if (!strcmp(argv[1], "idxqk")) return test_idxqk(argc > 2 ? (unsigned)atoi(argv[2]) : 100) ? 3 : 0;
	if (!strcmp(argv[1], "kv")) return test_kv(argc > 2 ? (unsigned)atoi(argv[2]) : 200, argc > 3 ? (unsigned)atoi(argv[3]) : 8) ? 3 : 0;
	fprintf(stderr, "unknown test %s\n", argv[1]); return 2;
}
