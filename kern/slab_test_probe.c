/*
 * slab_test: the prefill chunk program's slab ops (slab.rb: slab_silu, slab_mix, slab_write, slab_rms, slab_conv) inside the
 * persistent die kernel against a double-precision CPU reference, one die, one-phase programs, random inputs.
 *
 *   slab_test [ldx] [npos]        (DIE_CUBIN=die.cubin by default; ldx a power of two >= 128, npos <= ldx)
 *
 * Slabs are k-major [rows][ldx] fp32, positions past npos zero. Every op's maximum error relative to its output scale is
 * printed; beyond 1e-5 (MUFU's ex2 / rcp / rsq are ~1e-7 relative) the op counts as a mismatch. Run on one GPU through gpuq.
 */
#define _POSIX_C_SOURCE 200809L
#include "dieops.h"
#include "deltanet.h"
#include "router.h"
#include "ple.h"
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

static uint32_t rs = 7;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double sig(double x) { return 1.0 / (1.0 + exp(-x)); }
/* IEEE half -> float (exact) */
static float h2f(uint16_t h) {
	uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
	if (e == 0) { if (!m) u = s; else { int sh = 0; while (!(m & 0x400)) { m <<= 1; sh++; } m &= 0x3ff; u = s | ((uint32_t)(113 - sh) << 23) | (m << 13); } }
	else if (e == 31) u = s | 0x7f800000u | (m << 13);
	else u = s | ((e + 112) << 23) | (m << 13);
	float f; memcpy(&f, &u, 4); return f;
}

static CUfunction fn;
/* SLAB_ONLY=name[,name...]: run only those tests */
static int want(const char *name) { const char *o = getenv("SLAB_ONLY"); if (!o) return 1; size_t n = strlen(name); for (const char *p = o; (p = strstr(p, name)); p += n) if ((p == o || p[-1] == ',') && (p[n] == 0 || p[n] == ',')) return 1; return 0; }
static CUdeviceptr dPROG, dCTL;
static int ldx, npos, lgx, bad_ops;

static CUdeviceptr up(const float *h, size_t n) { CUdeviceptr d; CK(cuMemAlloc(&d, 4 * n + 16)); CK(cuMemcpyHtoD(d, h, 4 * n)); return d; }
static CUdeviceptr nanbuf(size_t n) { CUdeviceptr d; CK(cuMemAlloc(&d, 4 * n + 16)); CK(cuMemsetD32(d, 0x7fc00000u, n)); return d; }
static float *down(CUdeviceptr d, size_t n) { float *h = malloc(4 * n); CK(cuMemcpyDtoH(h, d, 4 * n)); return h; }
/* a slab of rows x ldx with random values in the first npos columns (scaled by 1 + row % 3), zero beyond */
static float *slab(int rows, float scale) {
	float *x = calloc((size_t)rows * ldx, sizeof(float));
	for (int r = 0; r < rows; r++) for (int p = 0; p < npos; p++) x[(size_t)r * ldx + p] = scale * frand() * (1.f + r % 3);
	return x;
}
static void run(desc_t *d) {
	CK(cuMemcpyHtoD(dPROG, d, sizeof *d));
	unsigned nph = 1, ntok = 1, zero = 0;
	CK(cuMemsetD8(dCTL, 0, 8192)); CK(cuMemcpyHtoD(dCTL + 8, &zero, 4));
	void *params[] = { &dPROG, &nph, &dCTL, &ntok };
	CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
}
/* compare got against want over rows x ldx (all columns: pads included) */
static void check(const char *name, const float *got, const double *want, size_t n) {
	double scale = 0, maxrel = 0; size_t bad = 0, worst = 0;
	for (size_t i = 0; i < n; i++) if (fabs(want[i]) > scale) scale = fabs(want[i]);
	for (size_t i = 0; i < n; i++) {
		double e = isnan(got[i]) ? INFINITY : fabs(got[i] - want[i]) / (scale + 1e-30);
		if (e > maxrel) { maxrel = e; worst = i; }
		if (e > 1e-5) { if (getenv("SLAB_SHOW") && bad < (size_t)atoi(getenv("SLAB_SHOW"))) printf("    %s bad: row %zu col %zu gpu %.7g ref %.7g\n", name, i / (size_t)ldx, i % (size_t)ldx, got[i], want[i]); bad++; }
	}
	printf("%-12s %s: %zu of %zu beyond 1e-5 of the scale %.3g (max %.3g at row %zu col %zu: gpu %.7g ref %.7g)\n", name, bad ? "MISMATCH" : "OK", bad, n, scale,
	       maxrel, worst / (size_t)ldx, worst % (size_t)ldx, got[worst], want[worst]);
	if (bad) bad_ops++;
}

int main(int argc, char **argv) {
	setvbuf(stdout, NULL, _IONBF, 0);   /* a fault exits at once: the last line printed names the op before it */
	ldx = argc > 1 ? atoi(argv[1]) : 128;
	npos = argc > 2 ? atoi(argv[2]) : ldx - 21;
	for (lgx = 0; (1 << lgx) < ldx; lgx++);
	if (ldx < 128 || (1 << lgx) != ldx || npos < 1 || npos > ldx) { fprintf(stderr, "bad ldx %d / npos %d\n", ldx, npos); return 2; }
	printf("ldx %d, npos %d\n", ldx, npos);
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "die"));
	CK(cuMemAlloc(&dPROG, 128 * 16)); CK(cuMemAlloc(&dCTL, 8192));
	desc_t d;

	if (want("silu")) {   /* silu: the hc down slab [324][ldx], SiLU(0.25 x) for rows < 320 */
		size_t n = (size_t)324 * ldx;
		float *x = slab(324, 4.f); CUdeviceptr dx = up(x, n), dy = nanbuf(n);
		unsigned n4 = (unsigned)(n / 4), us4 = 320u * (unsigned)ldx / 4u;
		d_init(&d, OP_SLAB_SILU, (n4 + 255) / 256); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dy); d_u32(&d, 0x150, n4); d_u32(&d, 0x154, us4);
		run(&d);
		float *y = down(dy, n); double *ref = malloc(8 * n);
		for (size_t i = 0; i < n; i++) { double u = 0.25 * x[i]; ref[i] = i < 4 * (size_t)us4 ? u * sig(u) : x[i]; }
		check("slab_silu", y, ref, n);
	}
	if (want("mix")) {   /* mix: cols = 432 z columns from c0 = 864, U [4 cols][ldx], XH [10240][ldx], Z [2560][ldx] */
		int cols = 432, c0 = 864;
		size_t nu = (size_t)4 * cols * ldx, nx = (size_t)10240 * ldx, nz = (size_t)2560 * ldx;
		float *u = slab(4 * cols, 3.f), *xh = slab(10240, 2.f);
		CUdeviceptr du = up(u, nu), dxh = up(xh, nx), dz = nanbuf(nz);
		d_init(&d, OP_SLAB_MIX, (unsigned)((size_t)cols * ldx / 1024)); d_ptr(&d, 0x140, du); d_ptr(&d, 0x148, dxh); d_ptr(&d, 0x150, dz);
		d_u32(&d, 0x158, (unsigned)lgx); d_u32(&d, 0x15c, (unsigned)cols); d_u32(&d, 0x160, (unsigned)c0);
		run(&d);
		float *z = down(dz, nz);
		size_t n = (size_t)cols * ldx; double *ref = malloc(8 * n); float *got = malloc(4 * n);
		for (int c = 0; c < cols; c++) for (int p = 0; p < ldx; p++) {
			double a = 0;
			for (int l = 0; l < 4; l++) a += sig(u[((size_t)l * cols + c) * ldx + p]) * xh[((size_t)2560 * l + c0 + c) * ldx + p];
			ref[(size_t)c * ldx + p] = 0.25 * a; got[(size_t)c * ldx + p] = z[((size_t)c0 + c) * ldx + p];
		}
		check("slab_mix", got, ref, n);
		int touched = 0;
		for (size_t i = 0; i < nz; i++) { size_t r = i / (size_t)ldx; if ((r < (size_t)c0 || r >= (size_t)(c0 + cols)) && !isnan(z[i])) touched = 1; }
		if (touched) { printf("slab_mix     MISMATCH: rows outside [c0, c0 + cols) written\n"); bad_ops++; }
	}
	if (want("write")) {   /* write: S [10240][ldx] += 2 sigmoid(0.25 D[320 + l]) Y[c] */
		size_t ns = (size_t)10240 * ldx, ny = (size_t)2560 * ldx, nd = (size_t)324 * ldx;
		float *s = slab(10240, 5.f), *y = slab(2560, 1.f), *dd = slab(324, 6.f);
		CUdeviceptr ds = up(s, ns), dy = up(y, ny), ddd = up(dd, nd);
		d_init(&d, OP_SLAB_WRITE, (unsigned)(ns / 1024)); d_ptr(&d, 0x140, ds); d_ptr(&d, 0x148, dy); d_ptr(&d, 0x150, ddd); d_u32(&d, 0x158, (unsigned)lgx);
		run(&d);
		float *got = down(ds, ns); double *ref = malloc(8 * ns);
		for (size_t i = 0; i < ns; i++) {
			size_t r = i / (size_t)ldx, p = i % (size_t)ldx, l = r / 2560, c = r % 2560;
			ref[i] = s[i] + 2.0 * sig(0.25 * dd[(320 + l) * (size_t)ldx + p]) * y[c * (size_t)ldx + p];
		}
		check("slab_write", got, ref, ns);
	}
	if (want("rms")) {   /* rms, the hc norm: 4 vectors of 2560 rows, W [10240] */
		size_t n = (size_t)10240 * ldx;
		float *x = slab(10240, 3.f), *w = malloc(4 * 10240);
		for (int i = 0; i < 10240; i++) w[i] = 1.f + 0.5f * frand();
		CUdeviceptr dx = up(x, n), dw = up(w, 10240), dy = nanbuf(n);
		int lgg = lgx - 5;
		d_init(&d, OP_SLAB_RMS, 4u << lgg); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dw); d_ptr(&d, 0x150, dy); d_ptr(&d, 0x158, 0);
		d_u32(&d, 0x160, (unsigned)ldx); d_u32(&d, 0x164, 320); d_f32(&d, 0x168, 1.f / 2560.f); d_f32(&d, 0x16c, 1e-6f); d_u32(&d, 0x170, (unsigned)lgg); d_u32(&d, 0x174, 2560);
		run(&d);
		float *got = down(dy, n); double *ref = malloc(8 * n);
		for (int v = 0; v < 4; v++) for (int p = 0; p < ldx; p++) {
			double ss = 0;
			for (int k = 0; k < 2560; k++) { double a = x[((size_t)2560 * v + k) * ldx + p]; ss += a * a; }
			double r = 1.0 / sqrt(ss / 2560 + 1e-6);
			for (int k = 0; k < 2560; k++) { size_t i = ((size_t)2560 * v + k) * ldx + p; ref[i] = x[i] * w[2560 * v + k] * r; }
		}
		check("slab_rms hc", got, ref, n);
	}
	if (want("rmsg")) {   /* rms with the gate, the DeltaNet output norm: 8 heads of 128 rows, W [128], G [1024][ldx] */
		size_t n = (size_t)1024 * ldx;
		float *x = slab(1024, 2.f), *g = slab(1024, 4.f), *w = malloc(4 * 128);
		for (int i = 0; i < 128; i++) w[i] = 1.f + 0.5f * frand();
		CUdeviceptr dx = up(x, n), dg = up(g, n), dw = up(w, 128), dy = nanbuf(n);
		int lgg = lgx - 5;
		d_init(&d, OP_SLAB_RMS, 8u << lgg); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dw); d_ptr(&d, 0x150, dy); d_ptr(&d, 0x158, dg);
		d_u32(&d, 0x160, (unsigned)ldx); d_u32(&d, 0x164, 16); d_f32(&d, 0x168, 1.f / 128.f); d_f32(&d, 0x16c, 1e-6f); d_u32(&d, 0x170, (unsigned)lgg); d_u32(&d, 0x174, 0);
		run(&d);
		float *got = down(dy, n); double *ref = malloc(8 * n);
		for (int h = 0; h < 8; h++) for (int p = 0; p < ldx; p++) {
			double ss = 0;
			for (int k = 0; k < 128; k++) { double a = x[((size_t)128 * h + k) * ldx + p]; ss += a * a; }
			double r = 1.0 / sqrt(ss / 128 + 1e-6);
			for (int k = 0; k < 128; k++) { size_t i = ((size_t)128 * h + k) * ldx + p; ref[i] = x[i] * w[k] * r * sig(g[i]); }
		}
		check("slab_rms gate", got, ref, n);
	}
	if (want("conv")) {   /* conv: n = 1728 channels, the ring [4][n] holds x(t0 - 4 .. t0 - 1) at slots t & 3, taps W [n][4] */
		int n = 1728; unsigned t0 = 37;
		size_t nx = (size_t)n * ldx;
		float *x = slab(n, 2.f), *st = malloc(4 * 4 * (size_t)n), *w = malloc(4 * 4 * (size_t)n);
		for (int i = 0; i < 4 * n; i++) { st[i] = 2.f * frand(); w[i] = 0.7f * frand(); }
		CUdeviceptr dx = up(x, nx), dst = up(st, 4 * (size_t)n), dw = up(w, 4 * (size_t)n), du = nanbuf(nx);
		d_init(&d, OP_SLAB_CONV, (unsigned)(nx / 1024)); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dst); d_ptr(&d, 0x150, dw); d_ptr(&d, 0x158, du);
		d_u32(&d, 0x160, t0); d_u32(&d, 0x164, (unsigned)n); d_u32(&d, 0x168, (unsigned)lgx); d_u32(&d, 0x16c, (unsigned)npos);
		run(&d);
		float *got = down(du, nx), *gst = down(dst, 4 * (size_t)n); double *ref = malloc(8 * nx), *rst = malloc(8 * 4 * (size_t)n);
		for (int c = 0; c < n; c++) {
			for (int p = 0; p < ldx; p++) {
				double a = 0;
				for (int j = 0; j < 4; j++) {
					int q = p - 3 + j; long t = (long)t0 + q;
					double xv = q >= 0 ? x[(size_t)c * ldx + q] : st[(size_t)(t & 3) * n + c];
					a += w[4 * c + j] * xv;
				}
				ref[(size_t)c * ldx + p] = p < npos ? a * sig(a) : 0.0;
			}
			for (int s = 0; s < 4; s++) rst[(size_t)s * n + c] = st[(size_t)s * n + c];
			for (int q = npos - 4; q < npos; q++) if (q >= 0) rst[(size_t)((t0 + q) & 3) * n + c] = x[(size_t)c * ldx + q];
		}
		check("slab_conv", got, ref, nx);
		check("slab_conv st", gst, rst, 4 * (size_t)n);
	}
	if (want("dnstepc")) {   /* dnstepc: 4 value heads on key heads 0..3 (DN_TILED: kh = h & 15), q / k / v slabs [512][ldx], a / b [4][ldx], o [512][ldx],
	       the states [4][128][128]; against deltanet_step_gpu position by position */
		int H = 4; float eps = 1e-6f;
		size_t nv = (size_t)128 * H * ldx, ng = (size_t)H * ldx, nst = (size_t)H * 16384;
		float *q = slab(128 * H, 1.f), *k = slab(128 * H, 1.f), *v = slab(128 * H, 1.f), *a = slab(H, 2.f), *b = slab(H, 2.f);
		float *st = malloc(4 * nst), *sa = malloc(4 * H);
		for (size_t i = 0; i < nst; i++) st[i] = 0.1f * frand();
		for (int h = 0; h < H; h++) sa[h] = -0.5f - 0.25f * h;
		CUdeviceptr dq = up(q, nv), dk = up(k, nv), dv = up(v, nv), da = up(a, ng), db = up(b, ng), dst = up(st, nst), dsa = up(sa, H), dout = nanbuf(nv);
		d_init(&d, OP_DNSTEPC, (unsigned)H);
		d_ptr(&d, 0x140, dst); d_ptr(&d, 0x148, dq); d_ptr(&d, 0x150, dk); d_ptr(&d, 0x158, dv); d_ptr(&d, 0x160, da); d_ptr(&d, 0x168, db);
		d_ptr(&d, 0x170, dsa); d_ptr(&d, 0x178, dout); d_f32(&d, 0x180, eps); d_u32(&d, 0x184, 0); d_u32(&d, 0x188, 0);
		d_u32(&d, 0x18c, (unsigned)ldx); d_u32(&d, 0x190, (unsigned)npos);
		run(&d);
		float *got = down(dout, nv), *gst = down(dst, nst);
		double *ref = calloc(nv, 8), *rst = malloc(8 * nst);
		float qc[128], kc[128], vc[128], oc[128];
		for (int h = 0; h < H; h++) {
			float *S = st + (size_t)h * 16384;
			for (int p = 0; p < npos; p++) {
				for (int i = 0; i < 128; i++) { size_t r = (size_t)128 * h + i; qc[i] = q[r * ldx + p]; kc[i] = k[r * ldx + p]; vc[i] = v[r * ldx + p]; }
				deltanet_step_gpu(S, qc, kc, vc, a[(size_t)h * ldx + p], b[(size_t)h * ldx + p], sa[h], eps, oc);
				for (int i = 0; i < 128; i++) ref[((size_t)128 * h + i) * ldx + p] = oc[i];
			}
		}
		for (size_t i = 0; i < nst; i++) rst[i] = st[i];
		/* columns at or past npos are not written: compare the written ones */
		for (size_t i = 0; i < nv; i++) if (i % (size_t)ldx >= (size_t)npos) { got[i] = 0; ref[i] = 0; }
		check("dnstepc o", got, ref, nv);
		check("dnstepc st", gst, rst, nst);
	}
	if (want("router")) {   /* router: R [513][ldx]; every column against router_gpu (indices exactly, weights and s within the tolerance) */
		size_t nr = (size_t)513 * ldx;
		float *r = slab(513, 4.f);
		for (int p = 0; p < npos; p++) r[(size_t)17 * ldx + p] = r[(size_t)300 * ldx + p] = 9.5f;   /* a tie among the top ten */
		CUdeviceptr dr = up(r, nr), didx = nanbuf((size_t)10 * ldx), dwt = nanbuf((size_t)10 * ldx), dsw = nanbuf((size_t)ldx);
		d_init(&d, OP_SLAB_ROUTER, (unsigned)ldx / 32u); d_ptr(&d, 0x140, dr); d_ptr(&d, 0x148, didx); d_ptr(&d, 0x150, dwt); d_ptr(&d, 0x158, dsw); d_u32(&d, 0x160, (unsigned)ldx);
		run(&d);
		uint32_t *gi = (uint32_t *)down(didx, (size_t)10 * ldx); float *gw = down(dwt, (size_t)10 * ldx), *gs = down(dsw, (size_t)ldx);
		size_t nw = (size_t)11 * npos; float *got = malloc(4 * nw); double *ref = malloc(8 * nw); int idx_bad = 0;
		float col[512];
		for (int p = 0; p < npos; p++) {
			for (int e = 0; e < 512; e++) col[e] = r[(size_t)e * ldx + p];
			unsigned ri[10]; float rw[10], rsh;
			router_gpu(col, r[(size_t)512 * ldx + p], ri, rw, &rsh);
			for (int j = 0; j < 10; j++) { if (gi[10 * p + j] != ri[j] && (gi[10 * p + j] >= 512 || col[gi[10 * p + j]] != col[ri[j]])) { if (idx_bad++ < 4) printf("    router idx bad: position %d rank %d gpu %u ref %u\n", p, j, gi[10 * p + j], ri[j]); } got[11 * p + j] = gw[10 * p + j]; ref[11 * p + j] = rw[j]; }
			got[11 * p + 10] = gs[p]; ref[11 * p + 10] = rsh;
		}
		printf("slab_router  %s: %d of %d indices differ\n", idx_bad ? "MISMATCH" : "OK", idx_bad, 10 * npos);
		if (idx_bad) bad_ops++;
		check("slab_router w", got, ref, nw);
	}
	if (want("tr")) {   /* tr: X [2560][ldx] -> Y [ldx][2560] */
		size_t n = (size_t)2560 * ldx;
		float *x = slab(2560, 1.f);
		CUdeviceptr dx = up(x, n), dy = nanbuf(n);
		d_init(&d, OP_SLAB_TR, (unsigned)(n / 1024)); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dy); d_u32(&d, 0x150, 2560); d_u32(&d, 0x154, (unsigned)lgx);
		run(&d);
		float *y = down(dy, n); double *ref = malloc(8 * n);
		for (int r = 0; r < 2560; r++) for (int p = 0; p < ldx; p++) ref[(size_t)p * 2560 + r] = x[(size_t)r * ldx + p];
		check("slab_tr", y, ref, n);
	}
	if (want("moe")) {   /* the MoE pieces on one die with synthetic unit-layout weights: group, ebatch gu / dn, swiglu, saxpy, zero, sum, rs */
		int npairs = 10 * npos;
		/* units: expert e has cnt[e] in 0..3 consecutive slots from first[e] */
		uint32_t emap[513]; int first[512], cnt[512], nunits = 0;
		for (int e = 0; e < 512; e++) { cnt[e] = (int)((rs >> 13) % 4); frand(); first[e] = nunits; nunits += cnt[e]; emap[e] = ((uint32_t)first[e] << 8) | (uint32_t)cnt[e]; }
		emap[512] = 0;
		/* routing: ten distinct experts per position, weights in (0, 1) */
		uint32_t *idx = malloc(4 * (size_t)npairs); float *rw = malloc(4 * (size_t)npairs);
		for (int p = 0; p < npos; p++) for (int j = 0; j < 10; j++) {
			uint32_t e; int dup;
			do { e = (rs >> 9) % 512; frand(); dup = 0; for (int i = 0; i < j; i++) if (idx[10 * p + i] == e) dup = 1; } while (dup);
			idx[10 * p + j] = e; rw[10 * p + j] = 0.5f + 0.5f * frand();
		}
		/* gate / up units: 64 rows of K 2560 Q2_0 rows of 720 B (80 B of fp16 scales, 40 x 16 B of codes); down units: 2560 x 16 B codes, 2560 scales */
		size_t gub = (size_t)nunits * 64 * 720, dnb = (size_t)nunits * 46080;
		uint8_t *wg = malloc(gub), *wd = malloc(dnb);
		for (size_t i = 0; i < gub; i++) wg[i] = (uint8_t)(rs >> 17), frand();
		for (size_t i = 0; i < dnb; i++) wd[i] = (uint8_t)(rs >> 17), frand();
		for (int u = 0; u < nunits * 64; u++) for (int b = 0; b < 40; b++) { uint16_t h = (uint16_t)(0x2000 + (rs >> 20) % 0x400); frand(); memcpy(wg + (size_t)u * 720 + 2 * b, &h, 2); }
		for (int u = 0; u < nunits; u++) for (int r = 0; r < 2560; r++) { uint16_t h = (uint16_t)(0x2000 + (rs >> 20) % 0x400); frand(); memcpy(wd + (size_t)u * 46080 + 40960 + 2 * r, &h, 2); }
		float *zt = malloc(4 * (size_t)ldx * 2560); for (size_t i = 0; i < (size_t)ldx * 2560; i++) zt[i] = frand();
		CUdeviceptr didx, drw, dem, dlp, dlw, dit, dni, dwg, dwd, dzt, dg, dh, dpart;
		didx = up((float *)idx, npairs); drw = up(rw, npairs); dem = up((float *)emap, 513);
		dlp = nanbuf(npairs); dlw = nanbuf(npairs); dit = nanbuf(512 * 4); dni = nanbuf(1);
		CK(cuMemAlloc(&dwg, gub)); CK(cuMemcpyHtoD(dwg, wg, gub)); CK(cuMemAlloc(&dwd, dnb)); CK(cuMemcpyHtoD(dwd, wd, dnb));
		dzt = up(zt, (size_t)ldx * 2560); dg = nanbuf((size_t)npairs * 192);
		d_init(&d, OP_SLAB_GROUP, 1); d_ptr(&d, 0x140, didx); d_ptr(&d, 0x148, drw); d_ptr(&d, 0x150, dem); d_ptr(&d, 0x158, dlp); d_ptr(&d, 0x160, dlw);
		d_ptr(&d, 0x168, dit); d_ptr(&d, 0x170, dni); d_u32(&d, 0x178, (unsigned)npairs);
		run(&d);
		uint32_t *lp = (uint32_t *)down(dlp, npairs), *items = (uint32_t *)down(dit, 512 * 4), ni; float *lw = down(dlw, npairs);
		CK(cuMemcpyDtoH(&ni, dni, 4));
		/* the reference grouping: per expert, its pairs' (position, weight) multiset; items in expert order */
		int gbad = 0, rn = 0, off = 0;
		for (int e = 0; e < 512; e++) {
			int n = 0; for (int q = 0; q < npairs; q++) if (idx[q] == (uint32_t)e) n++;
			if (n && cnt[e]) {
				if (rn >= (int)ni || items[4 * rn] != emap[e] || items[4 * rn + 1] != (uint32_t)off || items[4 * rn + 2] != (uint32_t)n || items[4 * rn + 3] != (uint32_t)e) { if (gbad++ < 4) printf("    group item %d bad (expert %d)\n", rn, e); }
				rn++;
			}
			for (int q = 0; q < npairs; q++) if (idx[q] == (uint32_t)e) {
				int found = 0; for (int i = off; i < off + n; i++) if (lp[i] == (uint32_t)(q / 10) && lw[i] == rw[q]) found = 1;
				if (!found && gbad++ < 4) printf("    group pair (expert %d position %d) missing\n", e, q / 10);
			}
			off += n;
		}
		if (rn != (int)ni) { gbad++; printf("    group: %u items, expected %d\n", ni, rn); }
		printf("slab_group   %s: %u items, %d problems\n", gbad ? "MISMATCH" : "OK", ni, gbad);
		if (gbad) bad_ops++;
		/* ebatch gu: G[q][r] for every item's pairs and rows */
		d_init(&d, OP_EBATCH_GU, 16); d_ptr(&d, 0x140, dwg); d_ptr(&d, 0x148, dzt); d_ptr(&d, 0x150, dg); d_ptr(&d, 0x158, dlp); d_ptr(&d, 0x160, dit); d_ptr(&d, 0x168, dni);
		run(&d);
		float *g = down(dg, (size_t)npairs * 192);
		size_t ng = 0; for (unsigned i = 0; i < ni; i++) ng += (size_t)items[4 * i + 2] * 64 * (items[4 * i] & 0xff);
		float *got = malloc(4 * ng); double *ref = malloc(8 * ng); size_t m = 0;
		float wrow[2560];
		for (unsigned i = 0; i < ni; i++) {
			int fs = (int)(items[4 * i] >> 8), c = (int)(items[4 * i] & 0xff), o = (int)items[4 * i + 1], n = (int)items[4 * i + 2];
			for (int r = 0; r < 64 * c; r++) {
				const uint8_t *row = wg + ((size_t)fs * 64 + r) * 720;
				for (int k = 0; k < 2560; k++) { uint16_t h; memcpy(&h, row + 2 * (k / 64), 2); int qv = (row[80 + 16 * (k / 64) + (k % 64) / 4] >> (2 * (k % 4))) & 3; wrow[k] = h2f(h) * (float)(qv - 1); }
				for (int j = 0; j < n; j++) {
					double a = 0; const float *z = zt + (size_t)lp[o + j] * 2560;
					for (int k = 0; k < 2560; k++) a += (double)wrow[k] * z[k];
					ref[m] = a; got[m] = g[(size_t)(o + j) * 192 + r]; m++;
				}
			}
		}
		check("ebatch_gu", got, ref, m);
		/* ebatch dn: PART[r][p] += LW sum_k WD[r][k] H[q][k] with H = the gate rows just computed (any values serve) */
		size_t np = (size_t)2560 * ldx;
		dpart = nanbuf(np);
		d_init(&d, OP_SLAB_ZERO, (unsigned)((np / 4 + 255) / 256)); d_ptr(&d, 0x140, dpart); d_u32(&d, 0x148, (unsigned)(np / 4));
		run(&d);
		dh = dg;
		d_init(&d, OP_EBATCH_DN, getenv("SLAB_DN_UNITS") ? (unsigned)atoi(getenv("SLAB_DN_UNITS")) : 16); d_ptr(&d, 0x140, dwd); d_ptr(&d, 0x148, dh); d_ptr(&d, 0x150, dpart); d_ptr(&d, 0x158, dlp); d_ptr(&d, 0x160, dlw);
		d_ptr(&d, 0x168, dit); d_ptr(&d, 0x170, dni); d_u32(&d, 0x178, (unsigned)ldx);
		run(&d);
		float *part = down(dpart, np); double *pref = calloc(np, 8);
		for (unsigned i = 0; i < ni; i++) {
			int fs = (int)(items[4 * i] >> 8), c = (int)(items[4 * i] & 0xff), o = (int)items[4 * i + 1], n = (int)items[4 * i + 2];
			for (int u = 0; u < c; u++) {
				const uint8_t *un = wd + (size_t)(fs + u) * 46080;
				for (int r = 0; r < 2560; r++) {
					uint16_t hh; memcpy(&hh, un + 40960 + 2 * r, 2); float dsc = h2f(hh);
					for (int j = 0; j < n; j++) {
						double a = 0; const float *hv = g + (size_t)(o + j) * 192 + 64 * u;
						for (int k = 0; k < 64; k++) { int qv = (un[16 * r + k / 4] >> (2 * (k % 4))) & 3; a += (double)(dsc * (float)(qv - 1)) * hv[k]; }
						pref[(size_t)r * ldx + lp[o + j]] += (double)lw[o + j] * a;
					}
				}
			}
		}
		check("ebatch_dn", part, pref, np);
		if (getenv("SLAB_DIAG")) {   /* where the down partial differs: by row block of 256, by position */
			int bym[10] = { 0 }, byp[512] = { 0 };
			for (size_t i = 0; i < np; i++) if (fabs(part[i] - pref[i]) > 1e-4) { bym[(i / (size_t)ldx) / 256]++; byp[i % (size_t)ldx]++; }
			printf("    dn bad by row block:"); for (int m = 0; m < 10; m++) printf(" %d", bym[m]); printf("\n    dn bad by position:");
			for (int p = 0; p < npos; p++) if (byp[p]) printf(" %d:%d", p, byp[p]); printf("\n");
			for (unsigned i = 0; i < ni; i++) { int o = (int)items[4 * i + 1], n = (int)items[4 * i + 2]; for (int j = 0; j < n; j++) if (lp[o + j] == 0) printf("    position 0: item %u (expert %u, count %u, n %d, column %d)\n", i, items[4 * i + 3], items[4 * i] & 0xff, n, j); }
			/* a misplaced contribution keeps the row's total over positions; a wrong value does not */
			double worst_row = 0; int shown = 0;
			for (int r = 0; r < 2560; r++) { double dsum = 0; for (int p = 0; p < ldx; p++) dsum += part[(size_t)r * ldx + p] - pref[(size_t)r * ldx + p]; if (fabs(dsum) > worst_row) worst_row = fabs(dsum); }
			printf("    dn: largest row total difference %.3g\n", worst_row);
			for (int p = 1; p < npos && shown < 6; p++) if (byp[p]) {
				printf("    position %d bad rows:", p); int run = -1;
				for (int r = 0; r < 2560; r++) { int bad = fabs(part[(size_t)r * ldx + p] - pref[(size_t)r * ldx + p]) > 1e-4; if (bad && run < 0) run = r; if ((!bad || r == 2559) && run >= 0) { printf(" %d..%d", run, bad ? r : r - 1); run = -1; } }
				printf("\n"); shown++;
			}
		}
		/* rs to this die's own inbox (owner 0 of every row), then sum over one part: the partial back */
		CUdeviceptr dinbox = nanbuf(np), dtab, dsum = nanbuf(np);
		uint64_t tab[12] = { 0 }; tab[0] = dinbox; dtab = up((float *)tab, 24);
		d_init(&d, OP_SLAB_RS, (unsigned)(np / 4 / 256)); d_ptr(&d, 0x140, dpart); d_ptr(&d, 0x148, dtab); d_u32(&d, 0x150, 0); d_u32(&d, 0x154, 0); d_u32(&d, 0x158, 2560); d_u32(&d, 0x15c, (unsigned)lgx);
		for (int e = 0; e < 5; e++) d_u32(&d, 0x160 + 4 * e, 2560);
		run(&d);
		d_init(&d, OP_SLAB_SUM, (unsigned)((np / 4 + 255) / 256)); d_ptr(&d, 0x140, dinbox); d_ptr(&d, 0x148, dsum); d_u32(&d, 0x150, (unsigned)(np / 4)); d_u32(&d, 0x154, 1); d_u32(&d, 0x158, 2560); d_u32(&d, 0x15c, (unsigned)lgx);
		run(&d);
		float *sum = down(dsum, np);
		check("slab_rs+sum", sum, pref, np);
		/* swiglu and saxpy on the partial */
		float *sw = malloc(4 * (size_t)ldx); for (int p = 0; p < ldx; p++) sw[p] = frand();
		CUdeviceptr dsw = up(sw, ldx), dx = up(part, np), dy = nanbuf(np);
		d_init(&d, OP_SLAB_SWIGLU, (unsigned)((np / 4 + 255) / 256)); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dsum); d_ptr(&d, 0x150, dy); d_u32(&d, 0x158, (unsigned)(np / 4));
		run(&d);
		float *y = down(dy, np); double *yref = malloc(8 * np);
		for (size_t i = 0; i < np; i++) yref[i] = (double)part[i] * sig(part[i]) * sum[i];
		check("slab_swiglu", y, yref, np);
		d_init(&d, OP_SLAB_SAXPY, (unsigned)((np / 4 + 255) / 256)); d_ptr(&d, 0x140, dy); d_ptr(&d, 0x148, dx); d_ptr(&d, 0x150, dsw); d_u32(&d, 0x158, (unsigned)(np / 4)); d_u32(&d, 0x15c, (unsigned)lgx);
		run(&d);
		float *y2 = down(dy, np);
		for (size_t i = 0; i < np; i++) yref[i] = (double)y[i] + (double)sw[i % (size_t)ldx] * part[i];
		check("slab_saxpy", y2, yref, np);
	}
if (want("ple")) {   /* pgate / pconv: block 1's PLE gate and conv over the chunk, against ple_gate_gpu / ple_conv_gpu per position */
	size_t n = (size_t)10240 * ldx;
	float *kn = slab(10240, 1.f), *qn = slab(10240, 1.f), *v = slab(2560, 1.f);
	CUdeviceptr dkn = up(kn, n), dqn = up(qn, n), dv = up(v, (size_t)2560 * ldx), dg = nanbuf(n);
	unsigned lgg = (unsigned)lgx - 5;
	d_init(&d, OP_SLAB_PGATE, 4u * (unsigned)ldx / 32u); d_ptr(&d, 0x140, dkn); d_ptr(&d, 0x148, dqn); d_ptr(&d, 0x150, dv); d_ptr(&d, 0x158, dg); d_u32(&d, 0x160, (unsigned)lgx); d_u32(&d, 0x164, lgg);
	run(&d);
	float *g = down(dg, n); double *gref = calloc(n, 8);
	float ck[10240], cq[10240], cv[2560], cg[10240], gates[4];
	for (int p = 0; p < npos; p++) {
		for (int r = 0; r < 10240; r++) { ck[r] = kn[(size_t)r * ldx + p]; cq[r] = qn[(size_t)r * ldx + p]; }
		for (int r = 0; r < 2560; r++) cv[r] = v[(size_t)r * ldx + p];
		ple_gate_gpu(ck, cq, cv, cg, gates);
		for (int r = 0; r < 10240; r++) gref[(size_t)r * ldx + p] = cg[r];
		if (getenv("SLAB_DIAG") && p < 4) { printf("    pgate p %d: ref gates %.5f %.5f %.5f %.5f, gpu G/V", p, gates[0], gates[1], gates[2], gates[3]);
			for (int l = 0; l < 4; l++) printf(" %.5f", g[(size_t)(2560 * l + 7) * ldx + p] / v[(size_t)7 * ldx + p]); printf("\n"); }
	}
	if (getenv("SLAB_DIAG")) for (int p = 0; p < 3; p++) {   /* the twin's s' per lane, the GPU's G rows of each lane (PGATE_DBG: G = s') */
		double ss[4] = { 0 };
		for (int l = 0; l < 4; l++) for (int c = 0; c < 2560; c++) ss[l] += (double)kn[(size_t)(2560 * l + c) * ldx + p] * qn[(size_t)(2560 * l + c) * ldx + p];
		printf("    p %d: s' %.5g %.5g %.5g %.5g; G %.5g %.5g %.5g %.5g\n", p, ss[0] / sqrt(2560.), ss[1] / sqrt(2560.), ss[2] / sqrt(2560.), ss[3] / sqrt(2560.),
		       g[p], g[(size_t)2560 * ldx + p], g[(size_t)5120 * ldx + p], g[(size_t)7680 * ldx + p]);
	}
	check("slab_pgate", g, gref, n);
	/* pconv from t0 = 13 (the ring holds positions 3 .. 12), then a chunk of 7 positions at 0 (the start: nothing before) */
	for (int pass = 0; pass < 2; pass++) {
		unsigned t0 = pass ? 0u : 13u; int np = pass ? (npos < 7 ? npos : 7) : npos;
		float *x = slab(10240, 1.f), *gg = slab(10240, 1.f), *s0 = slab(10240, 1.f);
		for (int r = 0; r < 10240; r++) for (int p = np; p < ldx; p++) x[(size_t)r * ldx + p] = gg[(size_t)r * ldx + p] = s0[(size_t)r * ldx + p] = 0.f;
		static float st[10][10240]; float *ri = malloc(4 * 10 * 10240);
		for (int k = 0; k < 10; k++) for (int c = 0; c < 10240; c++) st[k][c] = ri[(size_t)k * 10240 + c] = frand();
		uint16_t *w16 = malloc(2 * 4 * 10240); float *w32 = malloc(4 * 4 * 10240);
		for (int i = 0; i < 4 * 10240; i++) { w16[i] = (uint16_t)(0x3000 + (rs >> 20) % 0x800); frand(); w32[i] = h2f(w16[i]); }
		CUdeviceptr dx = up(x, n), dri = up(ri, 10 * 10240), dro = nanbuf(10 * 10240), dw, dgg = up(gg, n), ds = up(s0, n);
		CK(cuMemAlloc(&dw, 2 * 4 * 10240)); CK(cuMemcpyHtoD(dw, w16, 2 * 4 * 10240));
		d_init(&d, OP_SLAB_PCONV, 40u * (unsigned)ldx); d_ptr(&d, 0x140, dx); d_ptr(&d, 0x148, dri); d_ptr(&d, 0x150, dro); d_ptr(&d, 0x158, dw); d_ptr(&d, 0x160, dgg); d_ptr(&d, 0x168, ds);
		d_u32(&d, 0x170, t0); d_u32(&d, 0x174, t0 % 10); d_u32(&d, 0x178, (unsigned)np); d_u32(&d, 0x17c, (unsigned)lgx);
		run(&d);
		float *sg = down(ds, n), *ro = down(dro, 10 * 10240); double *sref = malloc(8 * n), *rref = malloc(8 * 10 * 10240);
		if (!t0) for (int k = 0; k < 10; k++) for (int c = 0; c < 10240; c++) st[k][c] = 0.f;   /* nothing before position 0 (the GPU reads 0 there) */
		float cx[10240], cgg[10240], cs[10240];
		for (int p = 0; p < np; p++) {
			for (int r = 0; r < 10240; r++) { cx[r] = x[(size_t)r * ldx + p]; cgg[r] = gg[(size_t)r * ldx + p]; cs[r] = s0[(size_t)r * ldx + p]; }
			ple_conv_gpu(cx, st, w32, t0 + (unsigned)p, cgg, cs);
			for (int r = 0; r < 10240; r++) sref[(size_t)r * ldx + p] = cs[r];
		}
		for (int r = 0; r < 10240; r++) for (int p = np; p < ldx; p++) sref[(size_t)r * ldx + p] = 0.;
		/* RO: the slots of the last ten positions (those at or after 0); the others are not written (NaN) */
		for (int k = 0; k < 10; k++) for (int c = 0; c < 10240; c++) rref[(size_t)k * 10240 + c] = st[k][c];
		int lo = (int)t0 + np - 10;
		for (int k = 0; k < 10; k++) { int pos = lo + k; if (pos < 0) for (int c = 0; c < 10240; c++) { rref[(size_t)(((pos % 10) + 10) % 10) * 10240 + c] = 0.; } }
		for (int k = 0; k < 10; k++) { int pos = lo + k; if (pos < 0) for (int c = 0; c < 10240; c++) ro[(size_t)(((pos % 10) + 10) % 10) * 10240 + c] = 0.f; }
		if (getenv("SLAB_DIAG")) for (int c = 0; c < 2; c++) for (int p = 20; p < 22 && p < np; p++) {
			size_t at = (size_t)c * ldx + p;
			double y = w32[4 * c] * x[at - 9] + w32[4 * c + 1] * x[at - 6] + w32[4 * c + 2] * x[at - 3] + w32[4 * c + 3] * x[at];
			printf("    pconv c %d p %d: x %.4g %.4g %.4g %.4g w %.4g %.4g %.4g %.4g g %.4g s0 %.4g y %.5g; gpu S %.6g ref %.6g\n", c, p, x[at - 9], x[at - 6], x[at - 3], x[at],
			       w32[4 * c], w32[4 * c + 1], w32[4 * c + 2], w32[4 * c + 3], gg[at], s0[at], y, sg[at], sref[at]);
		}
		check(pass ? "slab_pconv 0" : "slab_pconv 13", sg, sref, n);
		int saved = ldx; ldx = 10240;
		check(pass ? "pconv ring 0" : "pconv ring 13", ro, rref, (size_t)10 * 10240);
		ldx = saved;
	}
}
	printf("%s\n", bad_ops ? "MISMATCH" : "OK: all slab ops");
	return bad_ops ? 3 : 0;
}
