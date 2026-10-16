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
