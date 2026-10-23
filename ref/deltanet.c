#include "deltanet.h"
#include "fmath.h"
#include <math.h>
#include <string.h>

void l2norm128_gpu(const float *q, float eps, float *qn) {
	float v[32], t[32];
	for (int l = 0; l < 32; l++) {
		float ss = 0.f;
		for (int k = 0; k < 4; k++) ss = fmaf(q[4 * l + k], q[4 * l + k], ss);
		v[l] = ss;
	}
	for (int o = 16; o >= 1; o >>= 1) {
		for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ o];
		memcpy(v, t, sizeof v);
	}
	float nrm = sqrtf(v[0]);
	if (nrm < eps) nrm = eps;        /* max(nrm, eps) */
	float r = rcp_gpu(nrm);
	for (int i = 0; i < 128; i++) qn[i] = q[i] * r;
}

/* xor-butterfly over 8 parts in the GPU's order */
static float combine8(const float *a) {
	float v[8], t[8];
	memcpy(v, a, sizeof v);
	for (int o = 1; o <= 4; o <<= 1) {
		for (int p = 0; p < 8; p++) t[p] = v[p] + v[p ^ o];
		memcpy(v, t, sizeof v);
	}
	return v[0];
}

void deltanet_step_gpu(float *S, const float *q, const float *k, const float *v, float a, float b, float ssm_a, float eps, float *o) {
	float qn[128], kn[128];
	l2norm128_gpu(q, eps, qn);
	l2norm128_gpu(k, eps, kn);
	float sp = softplus_gpu(a);
	float g = ssm_a * sp;
	float alpha = exp_gpu(g);
	float beta = sigmoid_gpu(b);
	const float inv_sqrt128 = 0.088388347648318440550f;
	for (int c = 0; c < 128; c++) {
		float *row = S + 128 * c;
		for (int i = 0; i < 128; i++) row[i] = alpha * row[i];
		float part[8];
		for (int p = 0; p < 8; p++) {
			float acc = 0.f;
			for (int ii = 0; ii < 16; ii++) acc = fmaf(kn[16 * p + ii], row[16 * p + ii], acc);
			part[p] = acc;
		}
		float r = combine8(part);
		float e = v[c] - r;
		float be = beta * e;
		for (int i = 0; i < 128; i++) row[i] = fmaf(kn[i], be, row[i]);
		for (int p = 0; p < 8; p++) {
			float acc = 0.f;
			for (int ii = 0; ii < 16; ii++) acc = fmaf(qn[16 * p + ii], row[16 * p + ii], acc);
			part[p] = acc;
		}
		o[c] = combine8(part) * inv_sqrt128;
	}
}

void conv1d_silu_gpu(const float *x, float *state, const float *w, unsigned t, float *u, int n) {
	const float *s0 = state + (size_t)((t + 1) & 3) * n, *s1 = state + (size_t)((t + 2) & 3) * n, *s2 = state + (size_t)((t + 3) & 3) * n;
	float *sw = state + (size_t)(t & 3) * n;
	for (int c = 0; c < n; c++) {
		float y = w[4 * c] * s0[c];
		y = fmaf(w[4 * c + 1], s1[c], y);
		y = fmaf(w[4 * c + 2], s2[c], y);
		y = fmaf(w[4 * c + 3], x[c], y);
		u[c] = silu_gpu(y);
		sw[c] = x[c];
	}
}

void dnnorm_gpu(const float *o, const float *w, const float *gate, float eps, float *y) {
	for (int h = 0; h < 48; h++) {
		const float *x = o + 128 * h;
		float v[32], t[32];
		for (int l = 0; l < 32; l++) {
			float ss = x[4 * l] * x[4 * l];
			for (int k = 1; k < 4; k++) ss = fmaf(x[4 * l + k], x[4 * l + k], ss);
			v[l] = ss;
		}
		for (int ofs = 16; ofs >= 1; ofs >>= 1) {
			for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ ofs];
			memcpy(v, t, sizeof v);
		}
		float m = v[0] * (1.0f / 128.0f);
		float s = sqrtf(m + eps);
		float r = rcp_gpu(s);
		for (int c = 0; c < 128; c++) y[128 * h + c] = ((x[c] * r) * w[c]) * sigmoid_gpu(gate[128 * h + c]);
	}
}
