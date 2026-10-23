/* qsa_test: the QSA ops through the persistent kernel (die.cubin) against ref/qsa.c, token by token.
 *   qsa_test kv <T> <NBLK>     attnqkq (int8 KV append) + kvgather + attnpartn + attnmerge: caches, compact caches and the
 *                              attention output bit for bit; the attended blocks are a pseudo-random ascending selection
 * Run with the die index in CUDA_VISIBLE_DEVICES. */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "qsa.h"
#include "quant_range_cases.h"
#include "fmath.h"
#include "quant.h"
#include "gguf.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

/* Current CUDA headers rename this to a symbol absent from M60's driver.
 * Keep the original elapsed-time ABI, just as the context ABI below. */
#ifdef cuEventElapsedTime
#undef cuEventElapsedTime
#endif
CUresult cuEventElapsedTime(float *milliseconds, CUevent start, CUevent end);
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

#include "test_guards.h"

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

static int qsa_hostkv_mode_(void) {
    const char *s=getenv("QSA_HOSTKV");
    if(!s||!strcmp(s,"0"))return 0;
    if(!strcmp(s,"1"))return 1;
    fprintf(stderr,"QSA_HOSTKV must be exactly 0 or 1\n");exit(2);
}
static int qsa_legacy_uint_(const char *s,unsigned lo,unsigned hi,unsigned *out) {
    if(!s||!*s)return 1;unsigned v=0;
    for(;*s;s++){if(*s<'0'||*s>'9')return 1;unsigned d=(unsigned)(*s-'0');if(v>hi/10||(v==hi/10&&d>hi%10))return 1;v=10*v+d;}
    if(v<lo)return 1;*out=v;return 0;
}
static void qsa_legacy_gather_args_(int argc,char **argv) {
    if(argc<2||strcmp(argv[1],"gathertime"))return;
    unsigned ignored;
    if(argc>4||(argc>2&&qsa_legacy_uint_(argv[2],0,1,&ignored))||(argc>3&&qsa_legacy_uint_(argv[3],512,65523,&ignored))) {
        fprintf(stderr,"legacy gathertime needs host0|1 and nv512..65523; use qsa_gather_bench for small-nv/changing selections\n");exit(2);
    }
}

static void dev_init(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	dPROG = test_guard_alloc("program", 128 * 256, 0);
	dCTL = test_guard_alloc("control", 8192, 0);
}
static CUdeviceptr dalloc(size_t n) { return test_guard_alloc("qsa", n, 0); }
/* the same, in pinned host memory the die reads and writes over PCIe (QSA_HOSTKV=1 runs the KV cache of the kv test there) */
static CUdeviceptr halloc(size_t n) { return test_guard_alloc("qsa", n, 1); }
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


static unsigned quant_extreme_fixture(unsigned token, unsigned group) {
 return (2u * token + group) % QUANT_RANGE_SIGNED_COUNT;
}
static uint32_t quant_raw_max_bits(const float *x) {
 uint32_t maximum = 0;
 for (unsigned i = 0; i < 32; ++i) {
  uint32_t magnitude = quant_float_bits(x[i]) & 0x7fffffffu;
  if (magnitude > maximum) maximum = magnitude;
 }
 return maximum;
}
static void quant_extremes_(float *v, unsigned token) {
 for (unsigned group = 0; group < 16; ++group) {
  unsigned fixture = quant_extreme_fixture(token, group);
  quant_range_group(v + 32 * group, quant_range_amplitude(fixture));
 }
}
static int quant_extremes_check(unsigned token, const float *v,
                               const int8_t *cpu_q, const uint16_t *cpu_s,
                               const int8_t *gpu_q, const uint16_t *gpu_s,
                               unsigned char *seen) {
 int bad = 0;
 for (unsigned group = 0; group < 16; ++group) {
  unsigned fixture = quant_extreme_fixture(token, group);
  const quant_range_case *test = &quant_range_cases[fixture / 2];
  size_t qi = (size_t)token * 512 + 32 * group;
  size_t si = (size_t)token * 16 + group;
  if (gpu_s[si] != test->scale_bits || cpu_s[si] != test->scale_bits) {
   fprintf(stderr, "quant-range scale failure t=%u group=%u case=%s expected=%04x cpu=%04x gpu=%04x\n",
           token, group, test->name, test->scale_bits, cpu_s[si], gpu_s[si]);
   bad = 1;
  }
  for (unsigned i = 0; i < 32; ++i) {
   float x = v[32 * group + i];
   int expected = x > 0.f ? 127 : x < 0.f ? -127 : 0;
   if (gpu_q[qi + i] != expected || cpu_q[qi + i] != expected) {
    fprintf(stderr, "quant-range code failure t=%u group=%u channel=%u expected=%d cpu=%d gpu=%d\n",
            token, group, i, expected, cpu_q[qi + i], gpu_q[qi + i]);
    bad = 1; break;
   }
  }
  if (!seen[fixture]) {
   printf("quant-range gpu fixture=%u case=%s input=%08x expected_scale=%04x cpu_scale=%04x gpu_scale=%04x cpu_q1=%d gpu_q1=%d cpu_q2=%d gpu_q2=%d cpu_deq1=%08x gpu_bits_deq1=%08x\n",
          fixture, test->name, quant_float_bits(quant_range_amplitude(fixture)),
          test->scale_bits, cpu_s[si], gpu_s[si], cpu_q[qi + 1], gpu_q[qi + 1],
          cpu_q[qi + 2], gpu_q[qi + 2],
          quant_float_bits(kv_deq_gpu(cpu_q[qi + 1], cpu_s[si])),
          quant_float_bits(kv_deq_gpu(gpu_q[qi + 1], gpu_s[si])));
   seen[fixture] = 1;
  }
 }
 return bad;
}

/* FLASH_PADDING_POISON_TEST: test-only padding, never visible data. */
static int padding_poison_value_(const char *mode, float *value) {
    if (!mode || !strcmp(mode, "off")) return 0;
    uint32_t bits;
    if (!strcmp(mode, "nan")) bits = 0x7fc12345u;
    else if (!strcmp(mode, "inf")) bits = 0x7f800000u;
    else if (!strcmp(mode, "-inf")) bits = 0xff800000u;
    else if (!strcmp(mode, "max")) bits = 0x7f7fffffu;
    else if (!strcmp(mode, "finite")) bits = 0x40500000u;
    else { fprintf(stderr, "invalid padding poison: %s\n", mode); exit(2); }
    memcpy(value, &bits, sizeof bits);
    return 1;
}
static int test_kv(unsigned T, unsigned NBLK) {
    const char *poison_k_mode = getenv("QSA_POISON_K");
    const char *poison_v_mode = getenv("QSA_POISON_V");
    if (!poison_k_mode) poison_k_mode = getenv("QSA_POISON_PADDING");
    if (!poison_v_mode) poison_v_mode = getenv("QSA_POISON_PADDING");
    float poison_k_value = 0.f, poison_v_value = 0.f;
    int poison_k = padding_poison_value_(poison_k_mode, &poison_k_value);
    int poison_v = padding_poison_value_(poison_v_mode, &poison_v_value);

	const float eps = 1e-6f;
	unsigned npos = T + 64;
	const int trace_check = getenv("QSA_KV_TRACE_CHECK") != NULL;
	CUdeviceptr dTRACE = 0;
	uint32_t trace_expected[32], trace_got[32];
	if (trace_check) {
		dTRACE = dalloc((size_t)npos * 32 * sizeof(uint32_t));
		CK(cuMemsetD32(dTRACE, 0xffffffffu, (size_t)npos * 32));
	}
	float *rcs = malloc(4 * 32 * (size_t)npos), *rsn = malloc(4 * 32 * (size_t)npos);
	for (unsigned t = 0; t < npos; t++) rope_table_gpu(t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	float qn[256], kn[256]; for (int i = 0; i < 256; i++) { qn[i] = 0.5f + 0.5f * frand(); kn[i] = 0.5f + 0.5f * frand(); }
	/* host mirrors of the caches */
	int8_t *kq = calloc((size_t)npos * 512, 1), *vq = calloc((size_t)npos * 512, 1);
	uint16_t *ks = calloc((size_t)npos * 16, 2), *vs = calloc((size_t)npos * 16, 2);
	unsigned cpos = 4 * (NBLK + 1) + 2048; cpos = (cpos + 31) / 32 * 32;
	size_t kcf = ATTN_KC_FLOATS(cpos), vcf = (size_t)cpos * 512;
	CUdeviceptr dQG = dalloc(4 * 12288), dK = dalloc(4 * 512), dV = dalloc(4 * 512), dQN = dalloc(4 * 256), dKN = dalloc(4 * 256), dCS = dalloc(4 * 32 * (size_t)npos), dSN = dalloc(4 * 32 * (size_t)npos);
	int hk = qsa_hostkv_mode_();
	CUdeviceptr dQO = dalloc(4 * 6144), dKQ = hk ? halloc((size_t)npos * 512) : dalloc((size_t)npos * 512), dVQ = hk ? halloc((size_t)npos * 512) : dalloc((size_t)npos * 512), dKS = hk ? halloc((size_t)npos * 32) : dalloc((size_t)npos * 32), dVS = hk ? halloc((size_t)npos * 32) : dalloc((size_t)npos * 32);
	CUdeviceptr dKC = dalloc(kcf * 4), dVC = dalloc(vcf * 4), dSEL = dalloc(4 * (NBLK + 8)), dPM = dalloc(4 * 24 * 8), dPD = dalloc(4 * 24 * 8), dPO = dalloc(4 * 24 * 8 * 256), dAOUT = dalloc(4 * 6144);
	CK(cuMemcpyHtoD(dQN, qn, 4 * 256)); CK(cuMemcpyHtoD(dKN, kn, 4 * 256)); CK(cuMemcpyHtoD(dCS, rcs, 4 * 32 * (size_t)npos)); CK(cuMemcpyHtoD(dSN, rsn, 4 * 32 * (size_t)npos));
	desc_t prog[4];
	d_init(&prog[0], OP_ATTNQKQ, 26);
	d_ptr(&prog[0], 0x140, dQG); d_ptr(&prog[0], 0x148, dK); d_ptr(&prog[0], 0x150, dV); d_ptr(&prog[0], 0x158, dQN); d_ptr(&prog[0], 0x160, dKN);
	d_ptr(&prog[0], 0x168, dCS); d_ptr(&prog[0], 0x170, dSN); d_ptr(&prog[0], 0x178, dQO); d_ptr(&prog[0], 0x180, dKQ); d_ptr(&prog[0], 0x188, dVQ);
	d_tok(&prog[0], 0x190, 0); d_f32(&prog[0], 0x194, eps); d_ptr(&prog[0], 0x198, dKS); d_ptr(&prog[0], 0x1a0, dVS);
	d_ptr(&prog[0], 0x1a8, dTRACE);
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
	const int last_only = getenv("QSA_KV_LAST_ONLY") != NULL;
	const int extremes = getenv("QSA_KV_EXTREMES") != NULL;
	unsigned char quant_seen[QUANT_RANGE_SIGNED_COUNT] = {0};
	for (unsigned t = 0; t < T; t++) {
		int b = 0, check_attn = !last_only || t + 1 == T;
		for (int i = 0; i < 12288; i++) qg[i] = 2.f * frand();
		for (int i = 0; i < 512; i++) { k[i] = 2.f * frand(); v[i] = frand(); }
		if (t % 7 == 3) for (int i = 0; i < 512; i++) v[i] *= 1e-4f;          /* tiny scales (fp16 subnormals) */
		if (t % 11 == 5) for (int i = 256; i < 512; i++) v[i] = 0.f;          /* an all-zero kv head */
		if (t % 13 == 6) for (int i = 0; i < 512; i++) v[i] *= 3e3f;
		if (extremes) quant_extremes_(v, t);
		unsigned nv = (t + 1) >> 2, ns = nv < NBLK ? nv : NBLK;
		if (nv > NBLK) {   /* a pseudo-random ascending selection of NBLK of the nv blocks */
			unsigned *perm = malloc(4 * nv); for (unsigned i = 0; i < nv; i++) perm[i] = i;
			for (unsigned i = 0; i < NBLK; i++) { unsigned j = i + urand() % (nv - i), tmp = perm[i]; perm[i] = perm[j]; perm[j] = tmp; sel[i] = perm[i]; }
			for (unsigned i = 1; i < NBLK; i++) { uint32_t x = sel[i]; int j = (int)i - 1; while (j >= 0 && sel[j] > x) { sel[j + 1] = sel[j]; j--; } sel[j + 1] = x; }
			free(perm);
			CK(cuMemcpyHtoD(dSEL, sel, 4 * NBLK));
		}
		CK(cuMemcpyHtoD(dQG, qg, 4 * 12288)); CK(cuMemcpyHtoD(dK, k, 4 * 512)); CK(cuMemcpyHtoD(dV, v, 4 * 512));

        int poison_terminal = (poison_k || poison_v) && t + 1 == T;
        if (!poison_terminal) run_token(prog, 4, t);
        else {
            run_token(prog, 2, t);
            CK(cuMemcpyDtoH(kcg, dKC, kcf * sizeof(float)));
            CK(cuMemcpyDtoH(vcg, dVC, vcf * sizeof(float)));
            unsigned poison_n = qsa_nkeys(t, NBLK);
            for (unsigned j = poison_n; j < cpos; j++)
                for (int kv = 0; kv < 2; kv++)
                    for (int d = 0; d < 256; d++) {
                        float sign = (j + (unsigned)d + (unsigned)kv) & 1 ? -1.f : 1.f;
                        if (poison_k) kcg[attn_kidx(j, kv, d)] = !strcmp(poison_k_mode, "finite") ? sign * poison_k_value : poison_k_value;
                        if (poison_v) vcg[(size_t)j * 512 + kv * 256 + d] = !strcmp(poison_v_mode, "finite") ? sign * poison_v_value : poison_v_value;
                    }
            CK(cuMemcpyHtoD(dKC, kcg, kcf * sizeof(float)));
            CK(cuMemcpyHtoD(dVC, vcg, vcf * sizeof(float)));
            run_token(prog + 2, 2, t);
        }

		/* the twin: q/k norm + RoPE, int8 append, gather, attention */
		for (int h = 0; h < 24; h++) qk_norm_rope_gpu(qg + 512 * h, qn, eps, rcs + 32 * t, rsn + 32 * t, qo + 256 * h);
		for (int h = 0; h < 2; h++) {
			float y[256];
			qk_norm_rope_gpu(k + 256 * h, kn, eps, rcs + 32 * t, rsn + 32 * t, y);
			for (int b = 0; b < 8; b++) {
				size_t row = (size_t)t * 2 + h;
				if (trace_check) {
					trace_expected[h * 8 + b] = quant_raw_max_bits(y + 32 * b);
					trace_expected[16 + h * 8 + b] = quant_raw_max_bits(v + 256 * h + 32 * b);
				}
				kv_quant32_gpu(y + 32 * b, kq + row * 256 + 32 * b, &ks[row * 8 + b]);
				kv_quant32_gpu(v + 256 * h + 32 * b, vq + row * 256 + 32 * b, &vs[row * 8 + b]);
			}
		}
		unsigned n = qsa_gather_gpu(kq, ks, vq, vs, sel, t, NBLK, kc, vc);
		if (trace_check) {
			CK(cuMemcpyDtoH(trace_got, dTRACE + (size_t)t * sizeof trace_got, sizeof trace_got));
			b |= cmpb("raw KV maxima", trace_expected, trace_got, sizeof trace_got, &shown);
		}
		if (check_attn) attn_decode_gpu(qo, kc, vc, n, qg, out);
		CK(cuMemcpyDtoH(kqg, dKQ, (size_t)npos * 512)); CK(cuMemcpyDtoH(vqg, dVQ, (size_t)npos * 512)); CK(cuMemcpyDtoH(ksg, dKS, (size_t)npos * 32)); CK(cuMemcpyDtoH(vsg, dVS, (size_t)npos * 32));
		if (extremes) b |= quant_extremes_check(t, v, vq, vs, vqg, vsg, quant_seen);
        for(unsigned j=0;j<16;j++){size_t si=(size_t)t*16+j;if(!isfinite(qsa_h2f(ksg[si]))||!isfinite(qsa_h2f(vsg[si]))){fprintf(stderr,"nonfinite KV scale at t=%u group=%u\n",t,j);b=1;}}
        for(unsigned j=0;j<512;j++){size_t qi=(size_t)t*512+j,si=(size_t)t*16+j/32;if(!isfinite(kv_deq_gpu(kqg[qi],ksg[si]))||!isfinite(kv_deq_gpu(vqg[qi],vsg[si])))b=1;}

		{ float *qog = malloc(4 * 6144); CK(cuMemcpyDtoH(qog, dQO, 4 * 6144));
			/* Corrupt only the readback: prove that an isolated query failure reaches main. */
			if (getenv("QSA_INJECT_QO_MISMATCH") && t + 1 == T) ((unsigned char *)qog)[0] ^= 1;
			int bq = cmpb("qo", qo, qog, 4 * 6144, &shown); b |= bq;
			if (bq && !bad) printf("  (qo differs at t = %u)\n", t);
			free(qog); }
		if (getenv("QSA_DUMP") && t == 0) { printf("kq gpu:"); for (int i = 0; i < 40; i++) printf(" %d", kqg[i]); printf("\n    ref:"); for (int i = 0; i < 40; i++) printf(" %d", kq[i]); printf("\nks gpu:"); for (int i = 0; i < 16; i++) printf(" %04x", ksg[i]); printf("\n    ref:"); for (int i = 0; i < 16; i++) printf(" %04x", ks[i]); printf("\nvq gpu:"); for (int i = 0; i < 40; i++) printf(" %d", vqg[i]); printf("\n    ref:"); for (int i = 0; i < 40; i++) printf(" %d", vq[i]); printf("\n"); }
		b |= cmpb("kq", kq + (size_t)t * 512, kqg + (size_t)t * 512, 512, &shown); b |= cmpb("vq", vq + (size_t)t * 512, vqg + (size_t)t * 512, 512, &shown);
		b |= cmpb("ks", ks + (size_t)t * 16, ksg + (size_t)t * 16, 32, &shown); b |= cmpb("vs", vs + (size_t)t * 16, vsg + (size_t)t * 16, 32, &shown);
		/* All GPU stages and append/query checks run at every position. The opt-in
		 * long regression computes expensive scalar attention only at the last one. */
		if (check_attn) {
		CK(cuMemcpyDtoH(kcg, dKC, kcf * 4)); CK(cuMemcpyDtoH(vcg, dVC, vcf * 4));
		unsigned nb = ns + ((t + 1) & 3 ? 1 : 0);
		for (unsigned j = 0; j < nb; j++) for (unsigned i = 0; i < 4; i++) {
			unsigned slot = 4 * j + i;
			if (poison_terminal && slot >= n) continue;
			for (int kv = 0; kv < 2; kv++) for (int d = 0; d < 256; d += 4) {
				b |= cmpb("kc", &kc[attn_kidx(slot, kv, d)], &kcg[attn_kidx(slot, kv, d)], 16, &shown);
				b |= cmpb("vc", &vc[(size_t)slot * 512 + 256 * kv + d], &vcg[(size_t)slot * 512 + 256 * kv + d], 16, &shown);
			}
		}
		CK(cuMemcpyDtoH(outg, dAOUT, 4 * 6144));
        for(int j=0;j<6144;j++)if(!isfinite(outg[j]))b=1;
		b |= cmpb("out", out, outg, 4 * 6144, &shown);
		}
		if (b && !bad) printf("first mismatch at t = %u (n = %u keys, %u blocks selected)\n", t, n, ns);
		bad += b;
	}
	if (trace_check)
		printf("quant-range raw-maxima trace: %zu groups; %s\n", (size_t)T * 32, bad ? "FAIL" : "PASS");
	unsigned coverage = 0;
	int coverage_bad = 0;
	if (extremes) {
		for (unsigned i = 0; i < QUANT_RANGE_SIGNED_COUNT; ++i) coverage += quant_seen[i] != 0;
		coverage_bad = coverage != QUANT_RANGE_SIGNED_COUNT;
		printf("quant-range coverage: %u/%u signed fixtures; %s\n",
		       coverage, (unsigned)QUANT_RANGE_SIGNED_COUNT, bad || coverage_bad ? "FAIL" : "PASS");
		if (coverage_bad) fputs("incomplete extreme fixture coverage; use at least 16 tokens\n", stderr);
	}
	printf("kv: %u tokens, NBLK %u: %s (%d tokens differ; query/append checks %u, gather/attention checks %u)\n", T, NBLK, bad || coverage_bad ? "MISMATCH" : "BIT-EXACT", bad, T, last_only && T ? 1 : T);
	printf("padding: K=%s V=%s historical_cache=%s valid-data-seed=4242\n", poison_k_mode ? poison_k_mode : "off", poison_v_mode ? poison_v_mode : "off", hk ? "host" : "device");
	bad += test_guard_check();
	test_guard_free();
	return bad || coverage_bad;
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

static int test_idxscore(unsigned nv, unsigned NBLK) {
	unsigned ngrp = (nv + 31) / 32 + 1, t = 4 * nv + (urand() & 3) - ((nv > 0) ? 1 : 0);
	if (nv == 0) t = 0; else if (t > 262143) t = 262143;   /* last supported zero-based position; the RNG draw above is kept */
	float iq[512]; for (int i = 0; i < 512; i++) iq[i] = 1.5f * frand();
	uint16_t *pool = calloc((size_t)(nv + 1) * 128, 2), *pkt = calloc((size_t)ngrp * 4096, 2);
	for (unsigned b = 0; b < nv; b++) for (int d = 0; d < 128; d++) { uint16_t h = qsa_f2h(2.f * frand() + (b % 5 == 0 ? 0.7f : 0.f)); pool[(size_t)b * 128 + d] = h; pkt[pkt_idx(b, d)] = h; }
	CUdeviceptr dPKT = dalloc((size_t)ngrp * 8192), dIQ = dalloc(4 * 512), dSC = dalloc(4 * ((size_t)ngrp * 32 + 32)), dHST = dalloc(4 * 1280);
	CK(cuMemcpyHtoD(dPKT, pkt, (size_t)ngrp * 8192)); CK(cuMemcpyHtoD(dIQ, iq, 4 * 512));
	CK(cuMemsetD32(dSC, 0x7fc00000u, ngrp * 32 + 32));
	desc_t prog[1];
	d_init(&prog[0], OP_IDXSCORE, 16);
	d_ptr(&prog[0], 0x140, dPKT); d_ptr(&prog[0], 0x148, dIQ); d_ptr(&prog[0], 0x150, dSC); d_ptr(&prog[0], 0x158, dHST); d_tok(&prog[0], 0x160, 0); d_u32(&prog[0], 0x164, NBLK);
	run_token(prog, 1, t);
	float *sc = malloc(4 * (nv + 1)), *scg = malloc(4 * ((size_t)ngrp * 32 + 32));
	idx_scores_gpu(iq, pool, nv, sc);
	CK(cuMemcpyDtoH(scg, dSC, 4 * ((size_t)ngrp * 32 + 32)));
	uint32_t hist[256] = { 0 }, histg[1280];
	for (unsigned b = 0; b < nv; b++) { uint32_t u; memcpy(&u, &sc[b], 4); hist[u >> 23]++; }
	CK(cuMemcpyDtoH(histg, dHST, 4 * 1280));
	int bad = 0, shown = 0;
	if (nv > NBLK) { bad |= cmpb("scores", sc, scg, 4 * (size_t)nv, &shown); bad |= cmpb("hist", hist, histg, 4 * 256, &shown); }
	else for (unsigned i = 0; i < nv; i++) if (scg[i] == scg[i] && scg[i] != 0.f) { printf("  wrote scores although nv <= NBLK\n"); bad = 1; break; }
	printf("idxscore: nv %u NBLK %u: %s\n", nv, NBLK, bad ? "MISMATCH" : "BIT-EXACT");
	return bad;
}

static int test_idxsel(unsigned nv, unsigned NBLK, int mode, int npass) {
	unsigned t = 4 * nv - 1;
	float *sc = malloc(4 * (nv + 8)); uint32_t *sel = calloc(NBLK + 8, 4), *selg = malloc(4 * (NBLK + 8)), *selw = calloc(NBLK + 8, 4);
	for (unsigned b = 0; b < nv; b++) {
		float v = fabsf(frand()) * 3.f;
		if (mode == 1 && (urand() % 3)) v = 0.f;                   /* mostly exact zeros: a tie at the threshold */
		if (mode == 2) v = (float)(urand() % 7) * 0.5f;             /* a few distinct values */
		if (mode == 3) v = 1.25f;                                   /* all equal */
		sc[b] = v;
	}
	CUdeviceptr dSC = dalloc(4 * (nv + 8)), dHST = dalloc(4 * 1280), dSEL = dalloc(4 * (NBLK + 8));
	CK(cuMemcpyHtoD(dSC, sc, 4 * nv));
	uint32_t hst[1280] = { 0 };
	for (unsigned b = 0; b < nv; b++) { uint32_t u; memcpy(&u, &sc[b], 4); hst[u >> 23]++; }
	CK(cuMemcpyHtoD(dHST, hst, 4 * 1280));
	CK(cuMemsetD32(dSEL, 0xffffffffu, NBLK + 8));
	desc_t prog[5];
	for (int p = 0; p < 5; p++) { d_init(&prog[p], OP_IDXSEL, 16); d_ptr(&prog[p], 0x140, dSC); d_ptr(&prog[p], 0x148, dHST); d_ptr(&prog[p], 0x150, dSEL); d_tok(&prog[p], 0x158, 0); d_u32(&prog[p], 0x15c, NBLK); d_u32(&prog[p], 0x160, 2 + p); }
	run_token(prog, (unsigned)npass, t);
	unsigned k = idx_select_gpu(sc, nv, NBLK, sel);
	CK(cuMemcpyDtoH(selg, dSEL, 4 * (NBLK + 8)));
	int bad = 0, shown = 0;
	if (getenv("QSA_DEBUG") && nv <= 8000)
	{ uint32_t hg[1280]; CK(cuMemcpyDtoH(hg, dHST, 4 * 1280)); uint32_t *bits = malloc(4 * nv), Th = 0; unsigned gt = 0, eq = 0; for (unsigned b = 0; b < nv; b++) memcpy(&bits[b], &sc[b], 4);
		for (unsigned i = 0; i < nv; i++) { unsigned c = 0; for (unsigned j = 0; j < nv; j++) if (bits[j] > bits[i] || (bits[j] == bits[i] && j < i)) c++; if (c == (nv < NBLK ? nv : NBLK) - 1) Th = bits[i]; }
		for (unsigned i = 0; i < nv; i++) { if (bits[i] > Th) gt++; else if (bits[i] == Th) eq++; }
		for (int p = 2; p <= 6; p++) printf("  pass %d scan: pre %08x kk %u\n", p, hg[1056 + 2 * p], hg[1057 + 2 * p]);
		{ uint32_t eh[4][256] = { { 0 } }; unsigned sh[4] = { 23, 15, 7, 0 }, mk[4] = { 0xff, 0xff, 0xff, 0x7f }; uint32_t pf[4] = { 0 };
			for (int L = 0; L < 4; L++) { pf[L] = L ? (L == 1 ? Th >> 23 : L == 2 ? Th >> 15 : Th >> 7) : 0; for (unsigned i = 0; i < nv; i++) { if (L == 0 || (bits[i] >> (sh[L - 1])) == pf[L]) eh[L][(bits[i] >> sh[L]) & mk[L]]++; }
				unsigned bad2 = 0, s2 = 0, es = 0; for (int i = 0; i < 256; i++) { s2 += hg[256 * L + i]; es += eh[L][i]; if (hg[256 * L + i] != eh[L][i]) { if (bad2 < 4 && L == 1) printf("    G2 bin %02x: gpu %u host %u\n", i, hg[256 * L + i], eh[L][i]); bad2++; } } printf("  G%d: %u bins differ, gpu sum %u host sum %u\n", L + 1, bad2, s2, es); } }
		printf("  G1[0x7e] = %u, above: ", hg[0x7e]); { unsigned a = 0; for (int i = 0x7f; i < 256; i++) a += hg[i]; printf("%u\n", a); }
		printf("  T: gpu %08x host %08x ; need: gpu %u host %u (gt %u eq %u); hist2 sum %u hist3 sum %u hist4 sum %u\n", hg[1088], Th, hg[1089], NBLK - gt, gt, eq, 0u, 0u, 0u); free(bits); }
	bad |= cmpb("sel", sel, selg, 4 * (size_t)k, &shown);
	printf("idxsel: nv %u NBLK %u mode %d: %s\n", nv, NBLK, mode, bad ? "MISMATCH" : "BIT-EXACT");
	if (bad) { printf("  twin:"); for (unsigned i = 0; i < k && i < 24; i++) printf(" %u", sel[i]); printf("\n  gpu :"); for (unsigned i = 0; i < k && i < 24; i++) printf(" %u", selg[i]); printf("\n"); }
	(void)selw;
	return bad;
}

/* the whole indexer: idxqk, idxscore, the five idxsel passes, against the twins; then the time of the 8 phases per token */
static int test_idxfull(unsigned nv, unsigned NBLK, int iters) {
	const float eps = 1e-6f;
	unsigned t = 4 * nv - 1, npos = t + 8 + (unsigned)iters, ngrp = (npos / 4 + 31) / 32 + 1, nsc = ngrp * 32 + 32;
	float *rcs = malloc(4 * 32 * (size_t)npos), *rsn = malloc(4 * 32 * (size_t)npos);
	for (unsigned i = 0; i < npos; i++) rope_table_gpu(i, 1e7f, rcs + 32 * i, rsn + 32 * i);
	float qn[128], kn[128], iqk[640]; for (int i = 0; i < 128; i++) { qn[i] = 0.5f + 0.5f * frand(); kn[i] = 0.5f + 0.5f * frand(); }
	for (int i = 0; i < 640; i++) iqk[i] = 1.5f * frand();
	uint16_t *pkt = calloc((size_t)ngrp * 4096, 2);
	for (unsigned b = 0; b < nv; b++) for (int d = 0; d < 128; d++) pkt[pkt_idx(b, d)] = qsa_f2h(2.f * frand() + (b % 5 == 0 ? 0.7f : 0.f));
	CUdeviceptr dIQK = dalloc(4 * 640), dQN = dalloc(512), dKN = dalloc(512), dCS = dalloc(4 * 32 * (size_t)npos), dSN = dalloc(4 * 32 * (size_t)npos), dIQ = dalloc(2048), dRING = dalloc(2048), dPKT = dalloc((size_t)ngrp * 8192), dHST = dalloc(4 * 1280), dSC = dalloc(4 * (size_t)nsc), dSEL = dalloc(4 * (NBLK + 8));
	CK(cuMemcpyHtoD(dIQK, iqk, 4 * 640)); CK(cuMemcpyHtoD(dQN, qn, 512)); CK(cuMemcpyHtoD(dKN, kn, 512)); CK(cuMemcpyHtoD(dCS, rcs, 4 * 32 * (size_t)npos)); CK(cuMemcpyHtoD(dSN, rsn, 4 * 32 * (size_t)npos));
	CK(cuMemcpyHtoD(dPKT, pkt, (size_t)ngrp * 8192));
	desc_t prog[8];
	d_init(&prog[0], OP_IDXQK, 6);
	d_ptr(&prog[0], 0x140, dIQK); d_ptr(&prog[0], 0x148, dQN); d_ptr(&prog[0], 0x150, dKN); d_ptr(&prog[0], 0x158, dCS); d_ptr(&prog[0], 0x160, dSN); d_ptr(&prog[0], 0x168, dIQ);
	d_ptr(&prog[0], 0x170, dRING); d_ptr(&prog[0], 0x178, dPKT); d_ptr(&prog[0], 0x180, dHST); d_tok(&prog[0], 0x188, 0); d_f32(&prog[0], 0x18c, eps);
	d_init(&prog[1], OP_IDXSCORE, 16);
	d_ptr(&prog[1], 0x140, dPKT); d_ptr(&prog[1], 0x148, dIQ); d_ptr(&prog[1], 0x150, dSC); d_ptr(&prog[1], 0x158, dHST); d_tok(&prog[1], 0x160, 0); d_u32(&prog[1], 0x164, NBLK);
	for (int p = 0; p < 5; p++) { d_init(&prog[2 + p], OP_IDXSEL, 16); d_ptr(&prog[2 + p], 0x140, dSC); d_ptr(&prog[2 + p], 0x148, dHST); d_ptr(&prog[2 + p], 0x150, dSEL); d_tok(&prog[2 + p], 0x158, 0); d_u32(&prog[2 + p], 0x15c, NBLK); d_u32(&prog[2 + p], 0x160, 2 + p); }
	run_token(prog, 7, t);
	/* the twin from what the device holds: its pool, its IQ */
	uint16_t *pktg = malloc((size_t)ngrp * 8192); float iqg[512]; CK(cuMemcpyDtoH(pktg, dPKT, (size_t)ngrp * 8192)); CK(cuMemcpyDtoH(iqg, dIQ, 2048));
	float iq[512]; for (int h = 0; h < 4; h++) idx_norm_rope_gpu(iqk + 128 * h, qn, eps, rcs + 32 * t, rsn + 32 * t, iq + 128 * h);
	uint16_t *pool = malloc((size_t)nv * 256); for (unsigned b = 0; b < nv; b++) for (int d = 0; d < 128; d++) pool[(size_t)b * 128 + d] = pktg[pkt_idx(b, d)];
	float *sc = malloc(4 * (size_t)nv); idx_scores_gpu(iq, pool, nv, sc);
	uint32_t *sel = malloc(4 * (NBLK + 8)), *selg = malloc(4 * (NBLK + 8)); unsigned k = idx_select_gpu(sc, nv, NBLK, sel);
	CK(cuMemcpyDtoH(selg, dSEL, 4 * (NBLK + 8)));
	int bad = 0, shown = 0; bad |= cmpb("iq", iq, iqg, 2048, &shown);
	if (nv > NBLK) bad |= cmpb("sel", sel, selg, 4 * (size_t)k, &shown);
	printf("idxfull: nv %u NBLK %u: %s\n", nv, NBLK, bad ? "MISMATCH" : "BIT-EXACT");
	if (iters > 0) {   /* the time of the 7 phases: iters tokens in one launch (the context grows by iters/4 blocks) */
		CK(cuMemcpyHtoD(dPROG, prog, sizeof(desc_t) * 7)); unsigned nph = 7, tk = t;
		CK(cuMemcpyHtoD(dCTL + 8, &tk, 4));
		unsigned ntok = t + (unsigned)iters; void *params[] = { &dPROG, &nph, &dCTL, &ntok };
		CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
		CK(cuEventRecord(e0, 0)); CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL)); CK(cuEventRecord(e1, 0)); CK(cuCtxSynchronize());
		float ms; CK(cuEventElapsedTime(&ms, e0, e1));
		printf("  %d tokens (7 phases each) in %.2f ms: %.1f us per token\n", iters, ms, 1e3 * ms / iters);
		{ uint32_t ts[16]; CK(cuMemcpyDtoH(ts, dCTL + 0x100, 4 * 7)); printf("  last token, us per phase (idxscore, idxsel 2..6; idxqk, the first phase, has no start stamp):"); for (int i = 1; i < 7; i++) printf(" %.1f", (uint32_t)(ts[i] - ts[i - 1]) / 1e3); printf("\n"); }
	}
	return bad;
}

/* embrow: image rows from the fp16 table, token rows from a Q3_K table, through the same op */
static int test_embimg(void) {
	const int R = 32; unsigned npos = 6;
	uint8_t *tab = malloc((size_t)R * 1100); for (size_t i = 0; i < (size_t)R * 1100; i++) tab[i] = (uint8_t)urand();
	for (int r = 0; r < R; r++) for (int b = 0; b < 10; b++) { uint16_t h = (uint16_t)(0x2c00 + (urand() % 0x1000)); memcpy(tab + (size_t)r * 1100 + 110 * b + 108, &h, 2); }
	unsigned nimg = 4; uint16_t *img = malloc(2 * 2560 * (size_t)nimg); for (size_t i = 0; i < 2560 * (size_t)nimg; i++) img[i] = qsa_f2h(3.f * frand());
	uint32_t ids[6] = { 5, 7, 9, 11, 13, 15 }, imap[6] = { 0xffffffffu, 3, 0xffffffffu, 0xffffffffu, 0, 2 };
	CUdeviceptr dE = dalloc((size_t)R * 1100), dS = dalloc(4 * 10240), dIDS = dalloc(64), dIMAP = dalloc(64), dIMG = dalloc(2 * 2560 * (size_t)nimg);
	CK(cuMemcpyHtoD(dE, tab, (size_t)R * 1100)); CK(cuMemcpyHtoD(dIDS, ids, 24)); CK(cuMemcpyHtoD(dIMAP, imap, 24)); CK(cuMemcpyHtoD(dIMG, img, 2 * 2560 * (size_t)nimg));
	desc_t prog[1]; d_init(&prog[0], OP_EMBROW, 10);
	d_ptr(&prog[0], 0x140, dE); d_ptr(&prog[0], 0x148, dS); d_tok(&prog[0], 0x150, 0); d_ptr(&prog[0], 0x158, dIDS); d_ptr(&prog[0], 0x160, dIMAP); d_ptr(&prog[0], 0x168, dIMG);
	int bad = 0, shown = 0;
	for (unsigned t = 0; t < npos; t++) {
		CK(cuMemsetD32(dS, 0x7fc00000u, 10240));
		run_token(prog, 1, t);
		float ref[2560], got[10240];
		if (imap[t] == 0xffffffffu) dequant(GG_Q3_K, tab + (size_t)ids[t] * 1100, ref, 2560);
		else for (int j = 0; j < 2560; j++) ref[j] = qsa_h2f(img[(size_t)imap[t] * 2560 + j]);
		CK(cuMemcpyDtoH(got, dS, 4 * 10240));
		for (int l = 0; l < 4; l++) bad |= cmpb(imap[t] == 0xffffffffu ? "token row" : "image row", ref, got + 2560 * l, 4 * 2560, &shown);
	}
	printf("embrow with images: %s\n", bad ? "MISMATCH" : "BIT-EXACT");
	return bad;
}

/* Legacy zero-filled, repeated-layout timing over 51 positions; not a bandwidth measurement. */
static int test_gathertime(int host, unsigned nv) {
    if(nv<512||nv>65523){fprintf(stderr,"legacy gathertime requires nv512..65523; use qsa_gather_bench\n");return 2;}
    unsigned legacy_lo=0xffffffffu,legacy_hi=0,legacy_sum=0;
    for(unsigned j=0;j<51;j++){unsigned n=qsa_nkeys(4u*nv+j-1u,512);if(n<legacy_lo)legacy_lo=n;if(n>legacy_hi)legacy_hi=n;legacy_sum+=n;}
    printf("legacy gathertime: UNVERIFIED ZERO-FILLED FIXTURE, fixed reused selection; visible_start=%u visible_end=%u positions=51 keys_first=%u keys_last=%u keys_min=%u keys_max=%u keys_mean=%.9f; counter gate not run; use qsa_gather_bench for changing IDs\n",4u*nv,4u*nv+50u,qsa_nkeys(4u*nv-1u,512),qsa_nkeys(4u*nv+49u,512),legacy_lo,legacy_hi,legacy_sum/51.0);

	unsigned NBLK = 512, t = 4 * nv - 1, npos = 4 * nv + 400;
	size_t cpos = (4 * (NBLK + 1) + 2048 + 31) / 32 * 32;
	CUdeviceptr dKQ = host ? halloc((size_t)npos * 512) : dalloc((size_t)npos * 512), dVQ = host ? halloc((size_t)npos * 512) : dalloc((size_t)npos * 512), dKS = host ? halloc((size_t)npos * 32) : dalloc((size_t)npos * 32), dVS = host ? halloc((size_t)npos * 32) : dalloc((size_t)npos * 32);
	CUdeviceptr dKC = dalloc(ATTN_KC_FLOATS(cpos) * 4), dVC = dalloc(cpos * 512 * 4), dSEL = dalloc(4 * (NBLK + 8));
	uint32_t sel[512]; for (unsigned i = 0; i < NBLK; i++) sel[i] = i * (nv / NBLK);
	CK(cuMemcpyHtoD(dSEL, sel, 4 * NBLK));
	desc_t prog[1]; d_init(&prog[0], OP_KVGATHER, NBLK + 1);
	d_ptr(&prog[0], 0x140, dKQ); d_ptr(&prog[0], 0x148, dKS); d_ptr(&prog[0], 0x150, dVQ); d_ptr(&prog[0], 0x158, dVS); d_ptr(&prog[0], 0x160, dKC); d_ptr(&prog[0], 0x168, dVC); d_ptr(&prog[0], 0x170, dSEL); d_tok(&prog[0], 0x178, 0); d_u32(&prog[0], 0x17c, NBLK);
	CK(cuMemcpyHtoD(dPROG, prog, sizeof(desc_t))); unsigned nph = 1, ntok = t + 1 + 50; CK(cuMemcpyHtoD(dCTL + 8, &t, 4));
	void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	CUevent e0, e1; CK(cuEventCreate(&e0, 0)); CK(cuEventCreate(&e1, 0));
	CK(cuEventRecord(e0, 0)); CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL)); CK(cuEventRecord(e1, 0)); CK(cuCtxSynchronize());
	float ms; CK(cuEventElapsedTime(&ms, e0, e1));
	printf("kvgather, cache in %s memory: %.1f us per token (variable key counts; exact range printed above)\n", host ? "host" : "device", 1e3 * ms / 51);
	return 0;
}

#include "qsa_padding_test.h"

#include "checked.h"
int main(int argc,char**argv){
 typedef struct{const char*name;unsigned n,value[4],lo[4],hi[4];}mode;
 const mode modes[]={
  {"kv",2,{200,8},{1,1},{262144,4096}},
  {"idxqk",1,{100},{1},{262144}},
  {"idxscore",2,{700,512},{0,1},{65536,4096}},
  {"idxsel",4,{700,512,0,5},{1,1,0,1},{65536,4096,3,5}},
  {"idxfull",3,{2000,512,0},{1,1,0},{65536,4096,262144}},
  {"embimg",0,{0},{0},{0}},
  {"gathertime",2,{0,5000},{0,512},{1,65523}},
  {"padding",3,{2051,512,0},{1,1,0},{262144,4096,1000000}}
 };
 if(argc<2){fprintf(stderr,"usage: qsa_test kv|idxqk|idxscore|idxsel|idxfull|embimg|gathertime|padding [arguments]\n");return 2;}
 unsigned which;for(which=0;which<sizeof modes/sizeof*modes;which++)if(!strcmp(argv[1],modes[which].name))break;
 if(which==sizeof modes/sizeof*modes){fprintf(stderr,"unknown test: %s\n",argv[1]);return 2;}
 const mode*m=modes+which;unsigned arg[4];memcpy(arg,m->value,sizeof arg);
 if(argc>2+(int)m->n){fprintf(stderr,"too many arguments for %s\n",m->name);return 2;}
 for(int i=2;i<argc;i++)if(checked_parse_unsigned(argv[i],m->lo[i-2],m->hi[i-2],arg+i-2)){fprintf(stderr,"invalid argument for %s: %s\n",m->name,argv[i]);return 2;}
 if(which==4 && arg[2]>0 && arg[2]>262145u-4u*arg[0]){fprintf(stderr,"idxfull run exceeds supported token positions\n");return 2;}
 (void)qsa_hostkv_mode_(); qsa_legacy_gather_args_(argc,argv);
 /* No allocation or CUDA entry point before validation. */
 dev_init();
 switch(which){
 case 0:return test_kv(arg[0],arg[1])?1:0;
 case 1:return test_idxqk(arg[0])?3:0;
 case 2:return test_idxscore(arg[0],arg[1])?3:0;
 case 3:return test_idxsel(arg[0],arg[1],arg[2],arg[3])?3:0;
 case 4:return test_idxfull(arg[0],arg[1],arg[2])?3:0;
 case 5:return test_embimg()?3:0;
 case 6:return test_gathertime(arg[0],arg[1]);
 case 7:return test_padding(arg[0],arg[1],arg[2])?1:0;
 }
 return 2;
}
