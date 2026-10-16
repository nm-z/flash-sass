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
