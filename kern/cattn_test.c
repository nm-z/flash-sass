/*
 * cattn_test: the prefill chunk program's attention ops (cattn.rb: cattn_kv, cattn_idx, cattn_sel, cattn_app, cattn_attn) inside
 * the persistent die kernel against the scalar twins of ref/attn.c and ref/qsa.c, one die, one-phase programs, random inputs,
 * several chunks in a row (the caches, the pool and the ring carry over; chunks of npos positions start where the last ended,
 * so a chunk may start inside a block of 4).
 *
 *   cattn_test [ldx] [npos] [nchunk] [nblk]      (DIE_CUBIN=die.cubin; defaults 128 102 3 8)
 *
 * nblk is small so that positions past 4 nblk + 3 select their blocks. The query heads: 6 of KV head 0 (hq0 0). Checked: the
 * decode cache (dequantised K / V), the indexer queries, the pooled keys, the selections (the reference's set from the GPU's
 * queries and pool) and the attention output (over the GPU's selection and cache, double precision).
 */
#define _POSIX_C_SOURCE 200809L
#include "dieops.h"
#include "attn.h"
#include "qsa.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

typedef struct { uint32_t w[32]; } desc_t;
static void d_init(desc_t *d, int op, unsigned units) { memset(d, 0, sizeof *d); d->w[0] = (uint32_t)op; d->w[1] = units; }
static void d_ptr(desc_t *d, int addr, CUdeviceptr p) { int w = 2 + (addr - 0x140) / 4; d->w[w] = (uint32_t)p; d->w[w + 1] = (uint32_t)(p >> 32); }
static void d_u32(desc_t *d, int addr, uint32_t v) { d->w[2 + (addr - 0x140) / 4] = v; }
static void d_f32(desc_t *d, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); d_u32(d, addr, u); }

static uint32_t rs = 11;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static CUfunction fn;
static CUdeviceptr dPROG, dCTL;
static void run(desc_t *d) {
	CK(cuMemcpyHtoD(dPROG, d, sizeof *d));
	unsigned nph = 1, ntok = 1, zero = 0;
	CK(cuMemsetD8(dCTL, 0, 8192)); CK(cuMemcpyHtoD(dCTL + 8, &zero, 4));
	void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
}
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr d; CK(cuMemAlloc(&d, bytes + 64)); CK(cuMemsetD8(d, 0, bytes + 64)); return d; }

/* worst |gpu - ref| relative to the reference's largest magnitude */
typedef struct { double scale, err, wg, wr; long bad, n, at; } cmp_t;
static long cmp_at;   /* the index the next cmp_add reports for its value */
static void cmp_add(cmp_t *c, double got, double ref) { c->n++; if (fabs(ref) > c->scale) c->scale = fabs(ref); double e = isnan(got) ? INFINITY : fabs(got - ref); if (e > c->err) { c->err = e; c->wg = got; c->wr = ref; c->at = cmp_at; } }
static int cmp_print(const char *name, const cmp_t *c, double tol) {
	double rel = c->err / (c->scale + 1e-30);
	printf("%-14s %s: %ld values, worst error %.3g of the scale %.3g (at %ld: gpu %.7g ref %.7g)\n", name, rel <= tol ? "OK" : "MISMATCH", c->n, rel, c->scale, c->at, c->wg, c->wr);
	return rel <= tol ? 0 : 1;
}

int main(int argc, char **argv) {
	setvbuf(stdout, NULL, _IONBF, 0);
	int ldx = argc > 1 ? atoi(argv[1]) : 128, npos = argc > 2 ? atoi(argv[2]) : 102, nchunk = argc > 3 ? atoi(argv[3]) : 3;
	unsigned nblk = argc > 4 ? (unsigned)atoi(argv[4]) : 8;
	int lgx = 0; while ((1 << lgx) < ldx) lgx++;
	if (ldx < 128 || (1 << lgx) != ldx || npos < 1 || npos > ldx || nchunk < 1 || nblk < 1 || nblk > 512) { fprintf(stderr, "bad arguments\n"); return 2; }
	const int NHH = getenv("CATTN_NH") ? atoi(getenv("CATTN_NH")) : 6, T = npos * nchunk;   /* CATTN_NH: the query heads (1..6) */
	const float eps = 1e-6f, base = 1e6f;
	if (T > 32768) { fprintf(stderr, "at most 32768 positions\n"); return 2; }
	printf("ldx %d, npos %d, %d chunks (%d positions), nblk %u\n", ldx, npos, nchunk, T, nblk);

	CK(cuInit(0)); CUdevice dev; CK(cuDeviceGet(&dev, 0)); CUcontext ctx; CK(cuCtxCreate_v2(&ctx, 0, dev));
	CUmodule mod; const char *cubin = getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin";
	CK(cuModuleLoad(&mod, cubin)); CK(cuModuleGetFunction(&fn, mod, "die"));
	dPROG = dalloc(4096); dCTL = dalloc(8192);

	/* weights */
	float kn[256], qn[256], iqn[128], ikn[128];
	for (int i = 0; i < 256; i++) { kn[i] = 1.f + 0.2f * frand(); qn[i] = 1.f + 0.2f * frand(); }
	for (int i = 0; i < 128; i++) { iqn[i] = 1.f + 0.2f * frand(); ikn[i] = 1.f + 0.2f * frand(); }
	CUdeviceptr dKN = dalloc(1024), dQN = dalloc(1024), dIQN = dalloc(512), dIKN = dalloc(512);
	CK(cuMemcpyHtoD(dKN, kn, 1024)); CK(cuMemcpyHtoD(dQN, qn, 1024)); CK(cuMemcpyHtoD(dIQN, iqn, 512)); CK(cuMemcpyHtoD(dIKN, ikn, 512));
	/* the RoPE table of every position, and the chunk's rows of it */
	float *cs = malloc(4 * 32 * (size_t)T), *sn = malloc(4 * 32 * (size_t)T);
	for (int t = 0; t < T; t++) rope_table_gpu((unsigned)t, base, cs + 32 * t, sn + 32 * t);
	/* device state across chunks */
	CUdeviceptr dKQ = dalloc((size_t)T * 512), dKS = dalloc((size_t)T * 32), dVQ = dalloc((size_t)T * 512), dVS = dalloc((size_t)T * 32);
	CUdeviceptr dPKT = dalloc(((size_t)T / 128 + 1) * 8192), dPOOL = dalloc((size_t)128 * 8192 * 4), dRING = dalloc(2048);
	CUdeviceptr dR = dalloc((size_t)T * 512), dRS = dalloc((size_t)T * 64);
	CUdeviceptr dCKV = dalloc((size_t)npos * 1024), dCKS = dalloc((size_t)npos * 128), dSEL = dalloc((size_t)npos * 1024), dIQ = dalloc((size_t)npos * 2048);
	CUdeviceptr dLK = dalloc((size_t)512 * ldx * 4), dLV = dalloc((size_t)512 * ldx * 4), dLI = dalloc((size_t)640 * ldx * 4);
	CUdeviceptr dLQ = dalloc((size_t)NHH * 512 * ldx * 4), dLAO = dalloc((size_t)NHH * 256 * ldx * 4);
	CUdeviceptr dCS = dalloc((size_t)(ldx + 3) * 128), dSN = dalloc((size_t)(ldx + 3) * 128);
	/* host reference state */
	float *kd = calloc((size_t)T * 2 * 256, 4), *vd = calloc((size_t)T * 2 * 256, 4);   /* the GPU's decode cache, dequantised */
	float *rawk = calloc((size_t)T * 128, 4);
	uint16_t *pool_ref = calloc((size_t)(T / 4 + 1) * 128, 2);
	float *pool32 = calloc((size_t)(T / 4 + 1) * 128, 4);
	uint8_t *rep8 = malloc((size_t)T * 512); float *reps = malloc((size_t)T * 64);
	float *lk = malloc(4 * 512 * (size_t)ldx), *lv = malloc(4 * 512 * (size_t)ldx), *li = malloc(4 * 640 * (size_t)ldx), *lq = malloc(4 * (size_t)NHH * 512 * ldx);
	float *crow = calloc((size_t)(ldx + 3) * 32, 4), *srow = calloc((size_t)(ldx + 3) * 32, 4);
	int8_t *kq = malloc((size_t)T * 512), *vq = malloc((size_t)T * 512); uint16_t *ks = malloc((size_t)T * 32), *vs = malloc((size_t)T * 32);
	float *iqdbg = malloc((size_t)npos * 2048), *iq = malloc((size_t)npos * 2048), *lao = malloc(4 * (size_t)NHH * 256 * ldx); uint16_t *sel = malloc((size_t)npos * 1024);
	uint16_t *pkt = malloc(((size_t)T / 128 + 1) * 8192);
	cmp_t ck = { 0 }, cv = { 0 }, ci = { 0 }, cp = { 0 }, ca = { 0 }, cq = { 0 };
	float *gpool = malloc((size_t)128 * 8192 * 4);
	long sel_n = 0, sel_miss = 0, sel_order = 0;
	int fails = 0;

	for (int c = 0; c < nchunk; c++) {
		unsigned t0 = (unsigned)(c * npos);
		memset(lk, 0, 4 * 512 * (size_t)ldx); memset(lv, 0, 4 * 512 * (size_t)ldx); memset(li, 0, 4 * 640 * (size_t)ldx); memset(lq, 0, 4 * (size_t)NHH * 512 * ldx);
		for (int p = 0; p < npos; p++) {
			for (int r = 0; r < 512; r++) { lk[(size_t)r * ldx + p] = frand(); lv[(size_t)r * ldx + p] = frand(); }
			for (int r = 0; r < 640; r++) li[(size_t)r * ldx + p] = frand();
			for (int r = 0; r < NHH * 512; r++) lq[(size_t)r * ldx + p] = frand();
		}
		for (int i = 0; i < ldx + 3; i++) {
			int t = (int)t0 - 3 + i;
			if (t >= 0 && t < T) { memcpy(crow + 32 * i, cs + 32 * t, 128); memcpy(srow + 32 * i, sn + 32 * t, 128); }
		}
		CK(cuMemcpyHtoD(dLK, lk, 4 * 512 * (size_t)ldx)); CK(cuMemcpyHtoD(dLV, lv, 4 * 512 * (size_t)ldx)); CK(cuMemcpyHtoD(dLI, li, 4 * 640 * (size_t)ldx));
		CK(cuMemcpyHtoD(dLQ, lq, 4 * (size_t)NHH * 512 * ldx)); CK(cuMemcpyHtoD(dCS, crow, (size_t)(ldx + 3) * 128)); CK(cuMemcpyHtoD(dSN, srow, (size_t)(ldx + 3) * 128));
		CK(cuMemsetD8(dSEL, 0xff, (size_t)npos * 1024));
		desc_t d;
		d_init(&d, OP_CATTN_KV, (unsigned)npos);
		d_ptr(&d, 0x140, dLK); d_ptr(&d, 0x148, dLV); d_ptr(&d, 0x150, dKN); d_ptr(&d, 0x158, dCS); d_ptr(&d, 0x160, dSN); d_ptr(&d, 0x168, dKQ); d_ptr(&d, 0x170, dKS);
		d_ptr(&d, 0x178, dVQ); d_ptr(&d, 0x180, dVS); d_ptr(&d, 0x188, dCKV); d_ptr(&d, 0x190, dCKS); d_u32(&d, 0x198, t0); d_u32(&d, 0x19c, (unsigned)lgx); d_f32(&d, 0x1a0, eps);
		if (!c) printf("  %s\n", "kv"); run(&d);
		d_init(&d, OP_CATTN_IDX, (unsigned)npos);
		d_ptr(&d, 0x140, dLI); d_ptr(&d, 0x148, dIQN); d_ptr(&d, 0x150, dIKN); d_ptr(&d, 0x158, dCS); d_ptr(&d, 0x160, dSN); d_ptr(&d, 0x168, dIQ); d_ptr(&d, 0x170, dRING);
		d_ptr(&d, 0x178, dPKT); d_ptr(&d, 0x180, dPOOL); d_u32(&d, 0x188, t0); d_u32(&d, 0x18c, (unsigned)npos); d_u32(&d, 0x190, (unsigned)lgx); d_f32(&d, 0x194, eps);
		if (!c) printf("  %s\n", "idx"); run(&d);
		CK(cuMemcpyDtoH(iq, dIQ, (size_t)npos * 2048));
		d_init(&d, OP_CATTN_SEL, (unsigned)npos);
		d_ptr(&d, 0x140, dIQ); d_ptr(&d, 0x148, dPOOL); d_ptr(&d, 0x150, dSEL); d_u32(&d, 0x158, t0); d_u32(&d, 0x15c, nblk);
		if (!c) printf("  %s\n", "sel"); run(&d);
		d_init(&d, OP_CATTN_APP, (unsigned)(npos + 3) / 4);
		d_ptr(&d, 0x140, dCKV); d_ptr(&d, 0x148, dCKS); d_ptr(&d, 0x150, dR); d_ptr(&d, 0x158, dRS); d_u32(&d, 0x160, 0); d_u32(&d, 0x164, t0); d_u32(&d, 0x168, (unsigned)npos);
		if (!c) printf("  %s\n", "app"); run(&d);
		d_init(&d, OP_CATTN_ATTN, (unsigned)npos);
		d_ptr(&d, 0x140, dLQ); d_ptr(&d, 0x148, dQN); d_ptr(&d, 0x150, dCS); d_ptr(&d, 0x158, dSN); d_ptr(&d, 0x160, dR); d_ptr(&d, 0x168, dRS); d_ptr(&d, 0x170, dSEL);
		d_ptr(&d, 0x178, dLAO); d_u32(&d, 0x180, 0); d_u32(&d, 0x184, (unsigned)NHH); d_u32(&d, 0x188, t0); d_u32(&d, 0x18c, nblk); d_u32(&d, 0x190, (unsigned)lgx); d_f32(&d, 0x194, eps);
		if (!c) printf("  %s\n", "attn"); run(&d);
		CK(cuMemcpyDtoH(kq, dKQ, (size_t)T * 512)); CK(cuMemcpyDtoH(vq, dVQ, (size_t)T * 512)); CK(cuMemcpyDtoH(ks, dKS, (size_t)T * 32)); CK(cuMemcpyDtoH(vs, dVS, (size_t)T * 32));
		CK(cuMemcpyDtoH(iqdbg, dIQ, (size_t)npos * 2048)); CK(cuMemcpyDtoH(sel, dSEL, (size_t)npos * 1024)); CK(cuMemcpyDtoH(lao, dLAO, 4 * (size_t)NHH * 256 * ldx));
		CK(cuMemcpyDtoH(pkt, dPKT, ((size_t)T / 128 + 1) * 8192));
		CK(cuMemcpyDtoH(gpool, dPOOL, (size_t)128 * 8192 * 4));
		CK(cuMemcpyDtoH(rep8, dR, (size_t)T * 512)); CK(cuMemcpyDtoH(reps, dRS, (size_t)T * 64));
		/* K / V: the twin's norm, RoPE and quantisation against the GPU's cache; the GPU's dequantised cache is then the reference's */
		for (int p = 0; p < npos; p++) {
			unsigned t = t0 + (unsigned)p;
			for (int h = 0; h < 2; h++) {
				float x[256], y[256], v[256]; int8_t q[32]; uint16_t s16;
				for (int d2 = 0; d2 < 256; d2++) { x[d2] = lk[(size_t)(256 * h + d2) * ldx + p]; v[d2] = lv[(size_t)(256 * h + d2) * ldx + p]; }
				qk_norm_rope_gpu(x, kn, eps, cs + 32 * t, sn + 32 * t, y);
				for (int b = 0; b < 8; b++) {
					kv_quant32_gpu(y + 32 * b, q, &s16);
					for (int e = 0; e < 32; e++) {
						size_t at = ((size_t)t * 2 + h) * 256 + 32 * b + e;
						kd[at] = kv_deq_gpu(kq[at], ks[((size_t)t * 2 + h) * 8 + b]); cmp_at = (long)at; cmp_add(&ck, kd[at], kv_deq_gpu(q[e], s16));
					}
					kv_quant32_gpu(v + 32 * b, q, &s16);
					for (int e = 0; e < 32; e++) {
						size_t at = ((size_t)t * 2 + h) * 256 + 32 * b + e;
						vd[at] = kv_deq_gpu(vq[at], vs[((size_t)t * 2 + h) * 8 + b]); cmp_add(&cv, vd[at], kv_deq_gpu(q[e], s16));
					}
				}
			}
			/* the indexer: queries, raw key, pooled keys */
			for (int h = 0; h < 4; h++) {
				float x[128], y[128];
				for (int d2 = 0; d2 < 128; d2++) x[d2] = li[(size_t)(128 * h + d2) * ldx + p];
				idx_norm_rope_gpu(x, iqn, eps, cs + 32 * t, sn + 32 * t, y);
				for (int d2 = 0; d2 < 128; d2++) cmp_add(&ci, iq[(size_t)p * 512 + 128 * h + d2], y[d2]);
			}
			for (int d2 = 0; d2 < 128; d2++) rawk[(size_t)t * 128 + d2] = li[(size_t)(512 + d2) * ldx + p];
			if ((t & 3) == 3) {
				unsigned b = t >> 2;
				idx_pool_gpu(rawk + (size_t)(t - 3) * 128, ikn, eps, cs + 32 * (t - 3), sn + 32 * (t - 3), pool_ref + (size_t)b * 128);
				{ float m4[128]; const float *rk = rawk + (size_t)(t - 3) * 128; for (int d2 = 0; d2 < 128; d2++) m4[d2] = (((rk[d2] + rk[128 + d2]) + rk[256 + d2]) + rk[384 + d2]) * 0.25f;
				  idx_norm_rope_gpu(m4, ikn, eps, cs + 32 * (t - 3), sn + 32 * (t - 3), pool32 + (size_t)b * 128);
				  for (int d2 = 0; d2 < 128; d2++) cmp_add(&cq, gpool[(size_t)d2 * 8192 + b], pool32[(size_t)b * 128 + d2]); }
				for (int d2 = 0; d2 < 128; d2++) {
					uint16_t g = pkt[((size_t)8192 * (b >> 5) + 256 * (d2 / 4) + 8 * (b & 31)) / 2 + (d2 & 3)];
					cmp_add(&cp, qsa_h2f(g), qsa_h2f(pool_ref[(size_t)b * 128 + d2]));
				}
			}
		}
		/* the selections (the twin's scores of the GPU's queries over the reference pool) and attention over the GPU's selection */
		int nbadp = 0;
		for (int p = 0; p < npos; p++) {
			unsigned t = t0 + (unsigned)p, nv = (t + 1) >> 2, tail = (t + 1) & 3;
			uint32_t gsel[512];
			unsigned ns = nv < nblk ? nv : nblk;
			if (nv > nblk) {
				float *sc = malloc(4 * nv); uint32_t rsel[512];
				for (unsigned b = 0; b < nv; b++) {   /* the scores over the fp32 pool (the GPU's) */
					float tot = 0.f;
					for (int h = 0; h < 4; h++) { float a = 0.f; for (int d2 = 0; d2 < 128; d2++) a = fmaf(iq[(size_t)p * 512 + 128 * h + d2], pool32[(size_t)b * 128 + d2], a); tot += a > 0.f ? a : 0.f; }
					sc[b] = tot;
				}
				unsigned k = idx_select_gpu(sc, nv, nblk, rsel);
				for (unsigned j = 0; j < k; j++) {
					gsel[j] = sel[(size_t)p * 512 + j];
					int found = 0; for (unsigned i = 0; i < k; i++) if (rsel[i] == gsel[j]) found = 1;
					sel_n++; if (!found) sel_miss++;
					if (!found && sel_miss <= 3) {   /* the first few: the GPU's block, its score, the reference's k-th score */
						float kth = 3e38f; for (unsigned i = 0; i < k; i++) if (sc[rsel[i]] < kth) kth = sc[rsel[i]];
						int above = 0, eq = 0; for (unsigned b = 0; b < nv; b++) { if (sc[b] > kth) above++; if (sc[b] == kth) eq++; }
						printf("    sel t %u nv %u: gpu block %u (rank %u) score %.9g; the k-th score %.9g (%d above, %d equal)\n", t, nv, gsel[j], j, sc[gsel[j]], kth, above, eq);
						if (getenv("CATTN_DBG")) {
							uint32_t dd[3]; memcpy(dd, sel + (size_t)p * 512 + 500, 12); float tf; memcpy(&tf, dd, 4);
							printf("      gpu: T %.9g need %u nv %u; scores:", tf, dd[1], dd[2]);
							for (unsigned b = 0; b < nv && b < 32; b++) printf(" %u:%.6g/%.6g", b, iqdbg[(size_t)p * 512 + b], sc[b]);
							printf("\n");
						}
					}
					if (j && gsel[j] <= gsel[j - 1]) sel_order++;
				}
				free(sc);
			} else for (unsigned j = 0; j < ns; j++) gsel[j] = j;
			unsigned nb = ns + (tail ? 1 : 0), keys[2052], nk = 0;
			for (unsigned j = 0; j < nb; j++) {
				unsigned src = j < ns ? gsel[j] : nv;
				for (unsigned i = 0; i < 4; i++) if (4 * src + i <= t) keys[nk++] = 4 * src + i;
			}
			double pe = 0, pm = 1e-30;   /* this position's worst error and largest value over its heads */
			for (int h = 0; h < NHH; h++) {
				float x[256], q[256];
				for (int d2 = 0; d2 < 256; d2++) x[d2] = lq[(size_t)(512 * h + d2) * ldx + p];
				qk_norm_rope_gpu(x, qn, eps, cs + 32 * t, sn + 32 * t, q);
				double *sc = malloc(8 * nk), mx = -1e300, den = 0, o[256] = { 0 };
				for (unsigned k = 0; k < nk; k++) {
					double a = 0; const uint8_t *kk = rep8 + (size_t)keys[k] * 512; const float *ksc = reps + (size_t)keys[k] * 16;
					for (int d2 = 0; d2 < 256; d2++) a += (double)q[d2] * ((double)kk[d2] - 128.0) * ksc[d2 / 32];
					sc[k] = a * 0.0625; if (sc[k] > mx) mx = sc[k];
				}
				for (unsigned k = 0; k < nk; k++) {
					double e = exp(sc[k] - mx); den += e; const uint8_t *vv = rep8 + (size_t)keys[k] * 512 + 256; const float *vsc = reps + (size_t)keys[k] * 16 + 8;
					for (int d2 = 0; d2 < 256; d2++) o[d2] += e * ((double)vv[d2] - 128.0) * vsc[d2 / 32];
				}
				for (int d2 = 0; d2 < 256; d2++) {
					double g = lq[(size_t)(512 * h + 256 + d2) * ldx + p];
					double rv = o[d2] / den / (1.0 + exp(-g)), gv = lao[(size_t)(256 * h + d2) * ldx + p];
					cmp_add(&ca, gv, rv); pm = fmax(pm, fabs(rv)); pe = isnan(gv) ? INFINITY : fmax(pe, fabs(gv - rv));
				}
				free(sc);
			}
			if (pe / pm > 3e-2) { if (!nbadp++) printf("    chunk %d bad positions:", c); printf(" %d", p); }
		}
		if (nbadp) printf(" (%d)\n", nbadp);
		printf("chunk %d (t0 %u) done\n", c, t0);
	}
	fails += cmp_print("cattn_kv K", &ck, 1e-2);   /* one code of a block (MUFU.RCP before the rounding) */
	fails += cmp_print("cattn_kv V", &cv, 1e-2);
	fails += cmp_print("cattn_idx q", &ci, 1e-5);
	fails += cmp_print("cattn_idx pool", &cp, 2e-3);
	fails += cmp_print("cattn_idx POOL", &cq, 1e-5);
	int sel_bad = sel_order || sel_miss * 200 > sel_n;   /* summation orders differ: near-ties may */
	printf("%-14s %s: %ld selected blocks, %ld not in the reference\x27s set, %ld out of order\n", "cattn_sel", sel_bad ? "MISMATCH" : "OK", sel_n, sel_miss, sel_order);
	fails += sel_bad;
	fails += cmp_print("cattn_attn", &ca, 1e-4);
	printf("%s\n", fails ? "MISMATCH" : "OK: all attention ops");
	return fails ? 3 : 0;
}
