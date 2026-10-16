#include "attn.h"
#include "fmath.h"
#include <math.h>
#include <string.h>

void qk_norm_rope_gpu(const float *x, const float *w, float eps, const float *cs, const float *sn, float *y) {
	float v[32], t[32];
	for (int l = 0; l < 32; l++) {
		float ss = 0.f;
		for (int k = 0; k < 8; k++) ss = fmaf(x[8 * l + k], x[8 * l + k], ss);
		v[l] = ss;
	}
	for (int o = 16; o >= 1; o >>= 1) {
		for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ o];
		memcpy(v, t, sizeof v);
	}
	float m = v[0] * (1.0f / 256.0f);
	float vv = m + eps;
	float s = sqrtf(vv);
	float r = rcp_gpu(s);
	for (int d = 0; d < 256; d++) y[d] = (x[d] * r) * w[d];
	for (int p = 0; p < 32; p++) {
		float yp = y[p], yq = y[p + 32];
		float a = yp * cs[p], b = yq * cs[p];
		y[p] = fmaf(yq, -sn[p], a);
		y[p + 32] = fmaf(yp, sn[p], b);
	}
}

void rope_table_gpu(unsigned t, float base, float *cs, float *sn) {
	for (int p = 0; p < 32; p++) {
		double theta = pow((double)base, -2.0 * p / 64.0);
		double ang = (double)t * theta;
		cs[p] = (float)cos(ang);
		sn[p] = (float)sin(ang);
	}
}

#include <stdlib.h>

/* xor 16..1 butterfly sum over 32 lanes, as the warp does it */
static float warp_sum(const float *v32) {
	float v[32], t[32];
	memcpy(v, v32, sizeof v);
	for (int o = 16; o >= 1; o >>= 1) {
		for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ o];
		memcpy(v, t, sizeof v);
	}
	return v[0];
}

void attn_partials_gpu(const float *qo, const float *kc, const float *vc, unsigned n, float *pm, float *pd, float *po) {
	unsigned ck = (n + 8191) / 8192 * 1024;    /* keys per chunk */
	float *s = malloc(sizeof(float) * 1024 * 12), *p = malloc(sizeof(float) * 1024 * 12);
	float (*acc)[12][256] = malloc(sizeof(float) * 4 * 12 * 256);
	for (int kv = 0; kv < 2; kv++) {
		for (int c = 0; c < ATTN_NCH; c++) {
			float m[12], den[12];
			for (int i = 0; i < 12; i++) { m[i] = ATTN_MINIT; den[i] = 0.f; }
			memset(acc, 0, sizeof(float) * 4 * 12 * 256);
			for (unsigned tb = c * ck; tb < (c + 1) * ck; tb += 1024) {
				for (int jj = 0; jj < 1024; jj++) {
					unsigned j = tb + jj;
					for (int i = 0; i < 12; i++) {
						float a = -INFINITY;
						if (j < n) {
							const float *q = qo + 256 * (12 * kv + i);
							a = 0.f;
							for (int d = 0; d < 256; d++) a = fmaf(q[d], kc[attn_kidx(j, kv, d)], a);
							a = a * 0.0625f;
						}
						s[12 * jj + i] = a;
					}
				}
				for (int i = 0; i < 12; i++) {
					float tm = -INFINITY;
					for (int jj = 0; jj < 1024; jj++) if (s[12 * jj + i] > tm) tm = s[12 * jj + i];
					float mnew = m[i] > tm ? m[i] : tm;
					float scale = exp_gpu(m[i] - mnew);
					for (int jj = 0; jj < 1024; jj++) p[12 * jj + i] = tb + jj < n ? exp_gpu(s[12 * jj + i] - mnew) : 0.f;
					/* den: per-thread slot sum, warp butterfly, warps in order */
					float ws[8];
					for (int w = 0; w < 8; w++) {
						float v[32];
						for (int l = 0; l < 32; l++) {
							int t = 32 * w + l;
							float a = p[12 * t + i] + p[12 * (256 + t) + i];
							a = a + p[12 * (512 + t) + i];
							v[l] = a + p[12 * (768 + t) + i];
						}
						ws[w] = warp_sum(v);
					}
					float td = ws[0];
					for (int w = 1; w < 8; w++) td = td + ws[w];
					den[i] = fmaf(den[i], scale, td);
					for (int r = 0; r < 4; r++)
						for (int d = 0; d < 256; d++) {
							float o = acc[r][i][d] * scale;
							for (int k = 0; k < 256; k++) {
								unsigned j = tb + 4 * k + r;
								if (j < n) o = fmaf(p[12 * (4 * k + r) + i], vc[(size_t)j * 512 + 256 * kv + d], o);
							}
							acc[r][i][d] = o;
						}
					m[i] = mnew;
				}
			}
			for (int i = 0; i < 12; i++) {
				int hq = 12 * kv + i;
				pm[ATTN_NCH * hq + c] = m[i]; pd[ATTN_NCH * hq + c] = den[i];
				for (int d = 0; d < 256; d++)
					po[((size_t)ATTN_NCH * hq + c) * 256 + d] = (acc[0][i][d] + acc[1][i][d]) + (acc[2][i][d] + acc[3][i][d]);
			}
		}
	}
	free(s); free(p); free(acc);
}

void attn_merge_gpu(const float *pm, const float *pd, const float *po, const float *gate, float *out) {
	for (int hq = 0; hq < 24; hq++) {
		float M = pm[ATTN_NCH * hq];
		for (int c = 1; c < ATTN_NCH; c++) if (pm[ATTN_NCH * hq + c] > M) M = pm[ATTN_NCH * hq + c];
		float e[ATTN_NCH], den = 0.f;
		for (int c = 0; c < ATTN_NCH; c++) {
			e[c] = exp_gpu(pm[ATTN_NCH * hq + c] - M);
			den = c ? fmaf(pd[ATTN_NCH * hq + c], e[c], den) : pd[ATTN_NCH * hq + c] * e[c];
		}
		float rd = rcp_gpu(den);
		for (int d = 0; d < 256; d++) {
			float o = 0.f;
			for (int c = 0; c < ATTN_NCH; c++) { float pv = po[((size_t)ATTN_NCH * hq + c) * 256 + d]; o = c ? fmaf(pv, e[c], o) : pv * e[c]; }
			o = o * rd;
			out[256 * hq + d] = o * sigmoid_gpu(gate[512 * hq + 256 + d]);
		}
	}
}

void attn_decode_gpu(const float *qo, const float *kc, const float *vc, unsigned n, const float *gate, float *out) {
	float *pm = malloc(sizeof(float) * 24 * ATTN_NCH), *pd = malloc(sizeof(float) * 24 * ATTN_NCH);
	float *po = malloc(sizeof(float) * 24 * ATTN_NCH * 256);
	attn_partials_gpu(qo, kc, vc, n, pm, pd, po);
	attn_merge_gpu(pm, pd, po, gate, out);
	free(pm); free(pd); free(po);
}
