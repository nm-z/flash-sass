#include "hc.h"
#include "fmath.h"
#include <math.h>

void hc_silu_gpu(const float *d, float *t) {
	for (int i = 0; i < 320; i++) t[i] = silu_gpu(d[i] * 0.25f);
}

void hc_mix_gpu(const float *up, const float *xh, float *z) {
	for (int c = 0; c < 2560; c++) {
		float g0 = sigmoid_gpu(up[c]), g1 = sigmoid_gpu(up[2560 + c]), g2 = sigmoid_gpu(up[5120 + c]), g3 = sigmoid_gpu(up[7680 + c]);
		float acc = g0 * xh[c];
		acc = fmaf(g1, xh[2560 + c], acc);
		acc = fmaf(g2, xh[5120 + c], acc);
		acc = fmaf(g3, xh[7680 + c], acc);
		z[c] = acc * 0.25f;
	}
}

void hc_write_gpu(const float *inj, const float *y, float *S) {
	float w[4];
	for (int l = 0; l < 4; l++) w[l] = 2.0f * sigmoid_gpu(0.25f * inj[l]);
	for (int c = 0; c < 2560; c++)
		for (int l = 0; l < 4; l++) S[2560 * l + c] = fmaf(w[l], y[c], S[2560 * l + c]);
}
