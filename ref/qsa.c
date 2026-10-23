#include "qsa.h"
#include "attn.h"
#include "fmath.h"
#include <math.h>
#include <fenv.h>
#include <stdlib.h>
#include <string.h>

uint16_t qsa_f2h(float f) {
	uint32_t x; memcpy(&x, &f, 4);
	uint32_t sign = (x >> 16) & 0x8000u, m = x & 0x7fffffu;
	int32_t e = (int32_t)((x >> 23) & 0xff) - 127 + 15;
	if (((x >> 23) & 0xff) == 0xff) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0));
	if (e >= 31) return (uint16_t)(sign | 0x7c00u);
	if (e <= 0) {
		if (e < -10) return (uint16_t)sign;
		m |= 0x800000u;
		int shift = 14 - e;
		uint32_t half = m >> shift, rem = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
		if (rem > halfway || (rem == halfway && (half & 1u))) half++;
		return (uint16_t)(sign | half);
	}
	uint32_t h = ((uint32_t)e << 10) | (m >> 13), rem = m & 0x1fffu;
	if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) h++;   /* a carry runs into the exponent, which is right */
	return (uint16_t)(sign | h);
}

float qsa_h2f(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ffu, x;
	if (e == 0) {
		if (m == 0) x = sign;
		else { int s = 0; while (!(m & 0x400u)) { m <<= 1; s++; } m &= 0x3ffu; x = sign | ((uint32_t)(127 - 15 - s + 1) << 23) | (m << 13); }
	} else if (e == 31) x = sign | 0x7f800000u | (m << 13);
	else x = sign | ((e - 15 + 127) << 23) | (m << 13);
	float f; memcpy(&f, &x, 4); return f;
}

void kv_quant32_gpu(const float *x, int8_t *q, uint16_t *s16) {
    float bounded[32];
    for (int e=0;e<32;e++) {
        float v=x[e];
        if(v>QSA_KV_MAX_ABS)v=QSA_KV_MAX_ABS;
        if(v<-QSA_KV_MAX_ABS)v=-QSA_KV_MAX_ABS;
        bounded[e]=v;
    }
    x=bounded;

	const float tiny = 7.8886090522101180541e-31f;   /* 2^-100 */
	float gmax = 0.f;
	for (int e = 0; e < 32; e++) { float a = fabsf(x[e]); if (a > gmax) gmax = a; }
	if (gmax < tiny) gmax = tiny;
	float r = rcp_gpu(gmax);
	float inv = r * 127.0f;
	for (int e = 0; e < 32; e++) q[e] = (int8_t)lrintf(x[e] * inv);
	*s16 = qsa_f2h(gmax * (1.0f / 127.0f));
}

float kv_deq_gpu(int8_t q, uint16_t s16) { return (float)q * qsa_h2f(s16); }

unsigned qsa_nkeys(unsigned t, unsigned nblk) {
	unsigned nv = (t + 1) >> 2, tail = (t + 1) & 3;
	return nv <= nblk ? t + 1 : 4 * nblk + tail;
}

unsigned qsa_gather_gpu(const int8_t *kq, const uint16_t *ks, const int8_t *vq, const uint16_t *vs, const uint32_t *sel, unsigned t, unsigned nblk, float *kc, float *vc) {
	unsigned nv = (t + 1) >> 2, tail = (t + 1) & 3, ns = nv < nblk ? nv : nblk;
	unsigned nb = ns + (tail ? 1 : 0);
	for (unsigned j = 0; j < nb; j++) {
		unsigned src = j < ns ? (nv <= nblk ? j : sel[j]) : nv;
		for (unsigned i = 0; i < 4; i++) {
			unsigned slot = 4 * j + i, tok = 4 * src + i;
			for (int kv = 0; kv < 2; kv++)
				for (int d = 0; d < 256; d++) {
					size_t row = (size_t)tok * 2 + kv;
					kc[attn_kidx(slot, kv, d)] = kv_deq_gpu(kq[row * 256 + d], ks[row * 8 + (d >> 5)]);
					vc[(size_t)slot * 512 + 256 * kv + d] = kv_deq_gpu(vq[row * 256 + d], vs[row * 8 + (d >> 5)]);
				}
		}
	}
	return qsa_nkeys(t, nblk);
}

void idx_norm_rope_gpu(const float *x, const float *w, float eps, const float *cs, const float *sn, float *y) {
	float v[32], t[32];
	for (int l = 0; l < 32; l++) {
		float ss = 0.f;
		for (int k = 0; k < 4; k++) ss = fmaf(x[4 * l + k], x[4 * l + k], ss);
		v[l] = ss;
	}
	for (int o = 16; o >= 1; o >>= 1) {
		for (int l = 0; l < 32; l++) t[l] = v[l] + v[l ^ o];
		memcpy(v, t, sizeof v);
	}
	float m = v[0] * (1.0f / 128.0f);
	float vv = m + eps;
	float s = sqrtf(vv);
	float r = rcp_gpu(s);
	for (int d = 0; d < 128; d++) y[d] = (x[d] * r) * w[d];
	for (int p = 0; p < 32; p++) {
		float yp = y[p], yq = y[p + 32];
		float a = yp * cs[p], b = yq * cs[p];
		y[p] = fmaf(yq, -sn[p], a);
		y[p + 32] = fmaf(yp, sn[p], b);
	}
}

void idx_pool_gpu(const float *raw, const float *kn, float eps, const float *cs, const float *sn, uint16_t *out) {
	float m[128], y[128];
	for (int d = 0; d < 128; d++) {
		float s = ((raw[d] + raw[128 + d]) + raw[256 + d]) + raw[384 + d];
		m[d] = s * 0.25f;
	}
	idx_norm_rope_gpu(m, kn, eps, cs, sn, y);
	for (int d = 0; d < 128; d++) out[d] = qsa_f2h(y[d]);
}

void idx_scores_gpu(const float *iq, const uint16_t *pool, unsigned nv, float *score) {
	for (unsigned b = 0; b < nv; b++) {
		float s = 0.f;
		for (int h = 0; h < QSA_NIDXH; h++) {
			float a = 0.f;
			for (int d = 0; d < QSA_IDIM; d++) a = fmaf(iq[h * QSA_IDIM + d], qsa_h2f(pool[(size_t)b * QSA_IDIM + d]), a);
			a = (a > 0.f ? a : 0.f) + 0.f;
			s = h ? s + a : a;
		}
		score[b] = s;
	}
}

typedef struct { float s; uint32_t i; } sc_t;
static int cmp_sc(const void *a, const void *b) {
	const sc_t *x = a, *y = b;
	if (x->s != y->s) return x->s > y->s ? -1 : 1;
	return x->i < y->i ? -1 : x->i > y->i;
}
static int cmp_u32(const void *a, const void *b) { uint32_t x = *(const uint32_t *)a, y = *(const uint32_t *)b; return x < y ? -1 : x > y; }

unsigned idx_select_gpu(const float *score, unsigned nv, unsigned nblk, uint32_t *sel) {
	unsigned k = nv < nblk ? nv : nblk;
	sc_t *a = malloc(sizeof *a * (nv ? nv : 1));
	for (unsigned i = 0; i < nv; i++) { a[i].s = score[i]; a[i].i = i; }
	qsort(a, nv, sizeof *a, cmp_sc);
	for (unsigned i = 0; i < k; i++) sel[i] = a[i].i;
	qsort(sel, k, 4, cmp_u32);
	free(a);
	return k;
}

int kv_quant32_checked(const float *x,int8_t *q,uint16_t *s16) {
    if(!x||!q||!s16||fegetround()!=FE_TONEAREST)return 1;
    for(int i=0;i<32;i++)if(!isfinite(x[i]))return 1;
    int8_t tq[32];uint16_t ts;kv_quant32_gpu(x,tq,&ts);
    memcpy(q,tq,sizeof tq);*s16=ts;return 0;
}
