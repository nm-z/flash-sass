#include "fmath.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static float f32(uint32_t b) { float f; memcpy(&f, &b, 4); return f; }
static uint32_t u32(float f) { uint32_t b; memcpy(&b, &f, 4); return b; }

float rcp_gpu(float x) { return 1.0f / x; }

float exp_gpu(float x) {
	const float log2e = 1.44269502162933349609375f;        /* 0x3fb8aa3b */
	const float ln2hi = 0.693145751953125f;                  /* 0x3f317200 */
	const float ln2lo = 1.428606765330187045e-06f;           /* 0x35bfbe8e */
	const float magic = 12582912.0f;                         /* 1.5 * 2^23 */
	if (x > 88.0f) x = 88.0f;
	if (x < -87.0f) x = -87.0f;
	float t = x * log2e;
	float n = (t + magic) - magic;                           /* rint(t) */
	float r = fmaf(-n, ln2hi, x);
	r = fmaf(-n, ln2lo, r);
	float p = 1.0f / 720.0f;
	p = fmaf(p, r, 1.0f / 120.0f);
	p = fmaf(p, r, 1.0f / 24.0f);
	p = fmaf(p, r, 1.0f / 6.0f);
	p = fmaf(p, r, 0.5f);
	p = fmaf(p, r, 1.0f);
	p = fmaf(p, r, 1.0f);
	int32_t ni = (int32_t)lrintf(n);
	float scale = f32((uint32_t)((ni + 127) << 23));
	return p * scale;
}

float sigmoid_gpu(float x) {
	float e = exp_gpu(-x);
	float d = 1.0f + e;
	return rcp_gpu(d);
}

float silu_gpu(float x) { return x * sigmoid_gpu(x); }

float log1p_gpu(float e) {
	float d = 2.0f + e;
	float r = rcp_gpu(d);
	float u = e * r;
	float u2 = u * u;
	float p = 1.0f / 11.0f;
	p = fmaf(p, u2, 1.0f / 9.0f);
	p = fmaf(p, u2, 1.0f / 7.0f);
	p = fmaf(p, u2, 1.0f / 5.0f);
	p = fmaf(p, u2, 1.0f / 3.0f);
	p = fmaf(p, u2, 1.0f);
	float t = 2.0f * u;
	return t * p;
}

float softplus_gpu(float x) {
	float ax = fabsf(x);
	float e = exp_gpu(-ax);
	float l = log1p_gpu(e);
	float m = x > 0.0f ? x : 0.0f;
	return m + l;
}

/* silence unused warnings */
static void unused(void) { (void)u32; }
