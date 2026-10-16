#include "quant.h"
#include "gguf.h"
#include <math.h>
#include <string.h>

float fp16_to_fp32(uint16_t h) {
	uint32_t sign = (uint32_t)(h & 0x8000) << 16;
	uint32_t exp = (h >> 10) & 0x1f;
	uint32_t man = h & 0x3ff;
	uint32_t bits;
	if (exp == 0) {
		if (man == 0) bits = sign;
		else {
			/* subnormal: normalise */
			int e = -1;
			do { e++; man <<= 1; } while (!(man & 0x400));
			man &= 0x3ff;
			bits = sign | ((uint32_t)(127 - 15 - e) << 23) | (man << 13);
		}
	} else if (exp == 31) bits = sign | 0x7f800000 | (man << 13);
	else bits = sign | ((exp + 112) << 23) | (man << 13);
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

float bf16_to_fp32(uint16_t b) {
	uint32_t bits = (uint32_t)b << 16;
	float f;
	memcpy(&f, &bits, 4);
	return f;
}

static const int8_t kvalues_iq4nl[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };

static inline uint16_t ld16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }
static inline uint32_t ld32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

static void scale_min_k4(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
	if (j < 4) { *sc = q[j] & 63; *m = q[j + 4] & 63; }
	else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

int dequant(uint32_t type, const void *vsrc, float *y, uint64_t n) {
	const uint8_t *x = vsrc;
	uint32_t be, bb;
	gguf_block(type, &be, &bb);
	if (!be) return -1;
	uint64_t nb = n / be;
	for (uint64_t i = 0; i < nb; i++, x += bb, y += be) {
		switch (type) {
		case GG_F32: memcpy(y, x, 4); break;
		case GG_F16: y[0] = fp16_to_fp32(ld16(x)); break;
		case GG_BF16: y[0] = bf16_to_fp32(ld16(x)); break;
		case GG_Q2_0: {
			float d = fp16_to_fp32(ld16(x));
			const uint8_t *qs = x + 2;
			for (int j = 0; j < 64; j++) y[j] = d * (float)(((qs[j >> 2] >> (2 * (j & 3))) & 3) - 1);
			break;
		}
		case GG_Q4_0: {
			float d = fp16_to_fp32(ld16(x));
			const uint8_t *qs = x + 2;
			for (int j = 0; j < 16; j++) { y[j] = d * (float)((qs[j] & 0xf) - 8); y[j + 16] = d * (float)((qs[j] >> 4) - 8); }
			break;
		}
		case GG_Q5_0: {
			float d = fp16_to_fp32(ld16(x));
			uint32_t qh = ld32(x + 2);
			const uint8_t *qs = x + 6;
			for (int j = 0; j < 16; j++) {
				int xh0 = ((qh >> j) & 1) << 4, xh1 = (qh >> (j + 12)) & 0x10;
				y[j] = d * (float)(((qs[j] & 0xf) | xh0) - 16);
				y[j + 16] = d * (float)(((qs[j] >> 4) | xh1) - 16);
			}
			break;
		}
		case GG_Q8_0: {
			float d = fp16_to_fp32(ld16(x));
			const int8_t *qs = (const int8_t *)x + 2;
			for (int j = 0; j < 32; j++) y[j] = d * (float)qs[j];
			break;
		}
		case GG_IQ4_NL: {
			float d = fp16_to_fp32(ld16(x));
			const uint8_t *qs = x + 2;
			for (int j = 0; j < 16; j++) { y[j] = d * kvalues_iq4nl[qs[j] & 0xf]; y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4]; }
			break;
		}
		case GG_IQ4_XS: {
			float d = fp16_to_fp32(ld16(x));
			uint16_t sh = ld16(x + 2);
			const uint8_t *sl = x + 4, *qs = x + 8;
			float *yy = y;
			for (int ib = 0; ib < 8; ib++, qs += 16, yy += 32) {
				int ls = ((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
				float dl = d * (float)(ls - 32);
				for (int j = 0; j < 16; j++) { yy[j] = dl * kvalues_iq4nl[qs[j] & 0xf]; yy[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4]; }
			}
			break;
		}
		case GG_Q3_K: {
			const uint8_t *hm = x, *q = x + 32;
			uint32_t aux[4];
			memcpy(aux, x + 96, 12);
			float d_all = fp16_to_fp32(ld16(x + 108));
			const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
			uint32_t tmp = aux[2];
			aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
			aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
			aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
			aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
			const int8_t *scales = (const int8_t *)aux;
			int is = 0;
			uint8_t m = 1;
			float *yy = y;
			for (int nn = 0; nn < 256; nn += 128) {
				int shift = 0;
				for (int j = 0; j < 4; j++) {
					float dl = d_all * (float)(scales[is++] - 32);
					for (int l = 0; l < 16; l++) *yy++ = dl * (float)((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
					dl = d_all * (float)(scales[is++] - 32);
					for (int l = 0; l < 16; l++) *yy++ = dl * (float)((int8_t)((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) ? 0 : 4));
					shift += 2;
					m <<= 1;
				}
				q += 32;
			}
			break;
		}
		case GG_Q4_K: {
			float d = fp16_to_fp32(ld16(x)), min = fp16_to_fp32(ld16(x + 2));
			const uint8_t *sc8 = x + 4, *q = x + 16;
			float *yy = y;
			for (int j = 0, is = 0; j < 256; j += 64, q += 32, is += 2) {
				uint8_t sc, m;
				scale_min_k4(is, sc8, &sc, &m);
				float d1 = d * sc, m1 = min * m;
				scale_min_k4(is + 1, sc8, &sc, &m);
				float d2 = d * sc, m2 = min * m;
				for (int l = 0; l < 32; l++) *yy++ = d1 * (float)(q[l] & 0xf) - m1;
				for (int l = 0; l < 32; l++) *yy++ = d2 * (float)(q[l] >> 4) - m2;
			}
			break;
		}
		case GG_Q5_K: {
			float d = fp16_to_fp32(ld16(x)), min = fp16_to_fp32(ld16(x + 2));
			const uint8_t *sc8 = x + 4, *qh = x + 16, *ql = x + 48;
			float *yy = y;
			uint8_t u1 = 1, u2 = 2;
			for (int j = 0, is = 0; j < 256; j += 64, ql += 32, is += 2, u1 <<= 2, u2 <<= 2) {
				uint8_t sc, m;
				scale_min_k4(is, sc8, &sc, &m);
				float d1 = d * sc, m1 = min * m;
				scale_min_k4(is + 1, sc8, &sc, &m);
				float d2 = d * sc, m2 = min * m;
				for (int l = 0; l < 32; l++) *yy++ = d1 * (float)((ql[l] & 0xf) + ((qh[l] & u1) ? 16 : 0)) - m1;
				for (int l = 0; l < 32; l++) *yy++ = d2 * (float)((ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
			}
			break;
		}
		case GG_Q6_K: {
			const uint8_t *ql = x, *qh = x + 128;
			const int8_t *sc = (const int8_t *)x + 192;
			float d = fp16_to_fp32(ld16(x + 208));
			float *yy = y;
			for (int nn = 0; nn < 256; nn += 128, yy += 128, ql += 64, qh += 32, sc += 8) {
				for (int l = 0; l < 32; l++) {
					int is = l / 16;
					int q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
					int q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
					int q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
					int q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
					yy[l] = d * sc[is] * q1;
					yy[l + 32] = d * sc[is + 2] * q2;
					yy[l + 64] = d * sc[is + 4] * q3;
					yy[l + 96] = d * sc[is + 6] * q4;
				}
			}
			break;
		}
		default: return -1;
		}
	}
	return 0;
}

float quant_act_int8(const float *x, int8_t *q, int32_t *sum32, int n) {
	float amax = 0.f;
	for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > amax) amax = a; }
	float scale = amax / 127.f, inv = amax > 0.f ? 127.f / amax : 0.f;
	for (int i = 0; i < n; i++) {
		long v = lrintf(x[i] * inv);
		if (v > 127) v = 127;
		if (v < -127) v = -127;
		q[i] = (int8_t)v;
	}
	for (int h = 0; h < n / 32; h++) {
		int32_t s = 0;
		for (int j = 0; j < 32; j++) s += q[32 * h + j];
		sum32[h] = s;
	}
	return scale;
}

int q2_0_tile_stride(int K) {
	int B = K / 64;
	int scales = (2 * B + 15) / 16 * 16;
	return scales + 16 * B;
}

int repack_q2_0(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 64, stride = q2_0_tile_stride(K), scales = stride - 16 * B;
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 18;
		uint8_t *d = dst + (size_t)r * stride;
		memset(d, 0, scales);
		for (int b = 0; b < B; b++) {
			memcpy(d + 2 * b, s + 18 * b, 2);
			memcpy(d + scales + 16 * b, s + 18 * b + 2, 16);
		}
	}
	return stride;
}

int q2_0_lanes_per_row(int K) {
	int halves = K / 32;
	return halves / 5;   /* 2560 -> 16 lanes, 640 -> 4 lanes */
}

void gemv_q2_0_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y) {
	int B = K / 64, stride = q2_0_tile_stride(K), scales = stride - 16 * B;
	int L = q2_0_lanes_per_row(K);
	float v[32] = { 0 }, w[32] = { 0 };
	for (int r = 0; r < rows; r++) {
		const uint8_t *row = tile + (size_t)r * stride;
		for (int l = 0; l < L; l++) {
			float acc = 0.f;
			for (int i = 0; i < 5; i++) {
				int h = l + L * i, b = h >> 1;
				const uint8_t *qs = row + scales + 16 * b + 8 * (h & 1);
				int32_t dot = 0;
				for (int j = 0; j < 32; j++) dot += ((qs[j >> 2] >> (2 * (j & 3))) & 3) * xq[32 * h + j];
				float d = fp16_to_fp32(ld16(row + 2 * b));
				acc = fmaf(d, (float)(dot - sum32[h]), acc);
			}
			v[l] = acc;
		}
		for (int o = L / 2; o >= 1; o >>= 1) {
			for (int l = 0; l < L; l++) w[l] = v[l] + v[l ^ o];
			memcpy(v, w, sizeof(float) * L);
		}
		y[r] = v[0] * xscale;
	}
}

void gemv_q2_0_float(const uint8_t *src, int rows, int K, const float *x, float *y) {
	float w[4096];
	if (K > 4096) K = 4096;
	for (int r = 0; r < rows; r++) {
		dequant(GG_Q2_0, src + (size_t)r * (K / 64) * 18, w, (uint64_t)K);
		double s = 0;
		for (int k = 0; k < K; k++) s += (double)w[k] * x[k];
		y[r] = (float)s;
	}
}

void act_nsum64(const int8_t *q, int32_t *nsum64, int n) {
	for (int b = 0; b < n / 64; b++) {
		int32_t s = 0;
		for (int j = 0; j < 64; j++) s += q[64 * b + j];
		nsum64[b] = -s;
	}
}

void gemv_q2_0_ref_v2(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *nsum64, float xscale, float *y) {
	int B = K / 64, stride = q2_0_tile_stride(K), scales = stride - 16 * B;
	for (int r = 0; r < rows; r++) {
		const uint8_t *row = tile + (size_t)r * stride;
		float acc = 0.f;
		for (int b = 0; b < B; b++) {
			const uint8_t *qs = row + scales + 16 * b;
			int32_t dot = 0;
			for (int j = 0; j < 64; j++) dot += ((qs[j >> 2] >> (2 * (j & 3))) & 3) * xq[64 * b + j];
			float d = fp16_to_fp32(ld16(row + 2 * b));
			acc = fmaf(d, (float)(dot + nsum64[b]), acc);
		}
		y[r] = acc * xscale;
	}
}
