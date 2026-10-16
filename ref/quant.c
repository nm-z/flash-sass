#include "quant.h"
#include "gguf.h"
#include <math.h>
#include <stdlib.h>
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

int bf16_lanes_per_row(int K) {
	return K >= 2560 ? 32 : 8;   /* 2560 -> 80 elements per lane, 320 -> 40 */
}

void gemv_bf16_ref(const uint16_t *W, int rows, int K, const float *x, float *y) {
	int L = bf16_lanes_per_row(K), C = K / (8 * L);
	float v[32], w[32];
	for (int r = 0; r < rows; r++) {
		const uint16_t *row = W + (size_t)r * K;
		for (int l = 0; l < L; l++) {
			float acc = 0.f;
			for (int j = 0; j < C; j++) {
				int base = 8 * (l + L * j);
				for (int i = 0; i < 8; i++) acc = fmaf(bf16_to_fp32(row[base + i]), x[base + i], acc);
			}
			v[l] = acc;
		}
		for (int o = L / 2; o >= 1; o >>= 1) {
			for (int l = 0; l < L; l++) w[l] = v[l] + v[l ^ o];
			memcpy(v, w, sizeof(float) * L);
		}
		y[r] = v[0];
	}
}

int q4_k_tile_stride(int K) { return 144 * (K / 256); }

int q4_k_lanes_per_row(int K) { return K >= 6144 ? 32 : 16; }

void q4_k_scale_min(const uint8_t *q, int j, int *scale, int *min) {
	if (j < 4) { *scale = q[j] & 63; *min = q[j + 4] & 63; }
	else { *scale = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *min = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}

int repack_q4_k(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 256, stride = q4_k_tile_stride(K);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 144;
		uint8_t *hdr = dst + (size_t)r * stride, *codes = hdr + 16 * B;
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + 144 * b;
			memcpy(hdr + 16 * b, blk, 16);               /* d, dmin, scales[12] */
			const uint8_t *qs = blk + 16;
			for (int g = 0; g < 4; g++) {
				uint8_t *lo = codes + 128 * b + 16 * (2 * g), *hi = lo + 16;
				for (int i = 0; i < 16; i++) {
					const uint8_t *p = qs + 32 * g + 2 * i;
					lo[i] = (uint8_t)((p[0] & 0xF) | ((p[1] & 0xF) << 4));
					hi[i] = (uint8_t)((p[0] >> 4) | ((p[1] >> 4) << 4));
				}
			}
		}
	}
	return stride;
}

void gemv_q4_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y) {
	int B = K / 256, stride = q4_k_tile_stride(K);
	int L = q4_k_lanes_per_row(K), n = (K / 32) / L;
	float v[32], w[32];
	for (int r = 0; r < rows; r++) {
		const uint8_t *hdr = tile + (size_t)r * stride, *codes = hdr + 16 * B;
		for (int l = 0; l < L; l++) {
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int s = l + L * i, b = s >> 3, j = s & 7;
				const uint8_t *h = hdr + 16 * b, *q = codes + 128 * b + 16 * j;
				int sc, mn;
				q4_k_scale_min(h + 4, j, &sc, &mn);
				int32_t dot = 0;
				for (int e = 0; e < 32; e++) dot += ((q[e >> 1] >> (4 * (e & 1))) & 0xF) * xq[32 * s + e];
				float d = fp16_to_fp32(ld16(h)), dmin = fp16_to_fp32(ld16(h + 2));
				float t1 = (float)dot * (float)sc;
				acc = fmaf(d, t1, acc);
				float t2 = (float)(mn * sum32[s]);
				if (!getenv("GEMV_NOMIN")) acc = fmaf(-dmin, t2, acc);
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

int q5_k_tile_stride(int K) { return 176 * (K / 256); }

int repack_q5_k(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 256, stride = q5_k_tile_stride(K);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 176;
		uint8_t *hdr = dst + (size_t)r * stride, *codes = hdr + 16 * B, *hb = hdr + 144 * B;
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + 176 * b;
			memcpy(hdr + 16 * b, blk, 16);
			const uint8_t *qh = blk + 16, *qs = blk + 48;
			for (int g = 0; g < 4; g++) {
				uint8_t *lo = codes + 128 * b + 16 * (2 * g), *hi = lo + 16;
				for (int i = 0; i < 16; i++) {
					const uint8_t *p = qs + 32 * g + 2 * i;
					lo[i] = (uint8_t)((p[0] & 0xF) | ((p[1] & 0xF) << 4));
					hi[i] = (uint8_t)((p[0] >> 4) | ((p[1] >> 4) << 4));
				}
			}
			for (int j = 0; j < 8; j++) {
				uint32_t w = 0;
				for (int l = 0; l < 32; l++) w |= (uint32_t)((qh[l] >> j) & 1) << l;
				memcpy(hb + 32 * b + 4 * j, &w, 4);
			}
		}
	}
	return stride;
}

void gemv_q5_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y) {
	int B = K / 256, stride = q5_k_tile_stride(K);
	int L = q4_k_lanes_per_row(K), n = (K / 32) / L;
	float v[32], w[32];
	for (int r = 0; r < rows; r++) {
		const uint8_t *hdr = tile + (size_t)r * stride, *codes = hdr + 16 * B, *hb = hdr + 144 * B;
		for (int l = 0; l < L; l++) {
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int s = l + L * i, b = s >> 3, j = s & 7;
				const uint8_t *h = hdr + 16 * b, *q = codes + 128 * b + 16 * j;
				uint32_t hw;
				memcpy(&hw, hb + 32 * b + 4 * j, 4);
				int sc, mn;
				q4_k_scale_min(h + 4, j, &sc, &mn);
				int32_t dot = 0;
				for (int e = 0; e < 32; e++) dot += (((q[e >> 1] >> (4 * (e & 1))) & 0xF) + 16 * ((hw >> e) & 1)) * xq[32 * s + e];
				float d = fp16_to_fp32(ld16(h)), dmin = fp16_to_fp32(ld16(h + 2));
				float t1 = (float)dot * (float)sc;
				acc = fmaf(d, t1, acc);
				float t2 = (float)(mn * sum32[s]);
				if (!getenv("GEMV_NOMIN")) acc = fmaf(-dmin, t2, acc);
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

/* ---- Q3_K: per-16 groups, 2-bit codes plus inverted high bits ---- */

int q3_k_tile_stride(int K) { return 16 * (q4_k_lanes_per_row(K) / 8) + 112 * (K / 256); }

/* the 16 6-bit scales of a block, as ggml unpacks them */
static void q3_k_scales(const uint8_t *sc12, int8_t *sc16) {
	uint32_t aux[4];
	memcpy(aux, sc12, 12);
	const uint32_t kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
	uint32_t tmp = aux[2];
	aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
	aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
	aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
	aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
	memcpy(sc16, aux, 16);
}

int repack_q3_k(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 256, L = q4_k_lanes_per_row(K), Q = L / 8, stride = q3_k_tile_stride(K);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 110;
		uint8_t *pre = dst + (size_t)r * stride, *blocks = pre + 16 * Q;
		memset(pre, 0, 16 * Q);
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + 110 * b, *hm = blk, *qs = blk + 32;
			uint8_t *rec = blocks + 112 * b;
			memcpy(pre + 16 * (b % Q) + 2 * (b / Q), blk + 108, 2);
			int8_t sc[16];
			q3_k_scales(blk + 96, sc);
			for (int g = 0; g < 16; g++) {
				int nn = g >> 3, j = (g >> 1) & 3, which = g & 1;
				uint32_t cw = 0, hw = 0;
				for (int l = 0; l < 16; l++) {
					int code = (qs[32 * nn + 16 * which + l] >> (2 * j)) & 3;
					int h = (hm[16 * which + l] >> (4 * nn + j)) & 1;
					cw |= (uint32_t)code << (16 * (l >> 3) + 2 * (l & 7));
					hw |= (uint32_t)(h ^ 1) << l;
				}
				memcpy(rec + 4 * g, &cw, 4);
				rec[64 + 2 * g] = (uint8_t)hw;
				rec[65 + 2 * g] = (uint8_t)(hw >> 8);
				rec[96 + g] = (uint8_t)(sc[g] - 32);
			}
		}
	}
	return stride;
}

void gemv_q3_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, float xscale, float *y) {
	int B = K / 256, stride = q3_k_tile_stride(K);
	int L = q4_k_lanes_per_row(K), Q = L / 8, n = B / Q;
	float v[32], w[32];
	for (int r = 0; r < rows; r++) {
		const uint8_t *pre = tile + (size_t)r * stride, *blocks = pre + 16 * Q;
		for (int l = 0; l < L; l++) {
			int q = l >> 3, jj = l & 7;
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int b = Q * i + q;
				const uint8_t *rec = blocks + 112 * b;
				int32_t sumi = 0;
				for (int gg = 0; gg < 2; gg++) {
					int g = 2 * jj + gg;
					uint32_t cw;
					memcpy(&cw, rec + 4 * g, 4);
					uint32_t hw = rec[64 + 2 * g] | (rec[65 + 2 * g] << 8);
					int32_t dot = 0;
					for (int e = 0; e < 16; e++) {
						int code = (cw >> (16 * (e >> 3) + 2 * (e & 7))) & 3, hinv = (hw >> e) & 1;
						dot += (code - 4 * hinv) * xq[256 * b + 16 * g + e];
					}
					sumi += (int8_t)rec[96 + g] * dot;
				}
				acc = fmaf(fp16_to_fp32(ld16(pre + 16 * q + 2 * i)), (float)sumi, acc);
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

/* ---- Q6_K: per-16 int8 scales, low nibbles plus 2-bit high fields ---- */

int q6_k_tile_stride(int K) { return 16 * (q4_k_lanes_per_row(K) / 8) + 208 * (K / 256); }

void act_sum16(const int8_t *q, int32_t *sum16, int n) {
	for (int g = 0; g < n / 16; g++) {
		int32_t s = 0;
		for (int e = 0; e < 16; e++) s += q[16 * g + e];
		sum16[g] = s;
	}
}

/* q6 of element e (0..255) of a raw Q6_K block */
static int q6_k_elem(const uint8_t *blk, int e) {
	const uint8_t *ql = blk + 64 * (e >> 7), *qh = blk + 128 + 32 * (e >> 7);
	int l = e & 31, part = (e & 127) >> 5;      /* part 0..3 = q1..q4 of the decoder */
	int lo;
	if (part == 0) lo = ql[l] & 0xF;
	else if (part == 1) lo = ql[l + 32] & 0xF;
	else if (part == 2) lo = ql[l] >> 4;
	else lo = ql[l + 32] >> 4;
	int hi = (qh[l] >> (2 * part)) & 3;
	return lo | (hi << 4);
}

int repack_q6_k(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 256, L = q4_k_lanes_per_row(K), Q = L / 8, stride = q6_k_tile_stride(K);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 210;
		uint8_t *pre = dst + (size_t)r * stride, *blocks = pre + 16 * Q;
		memset(pre, 0, 16 * Q);
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + 210 * b;
			uint8_t *rec = blocks + 208 * b;
			memcpy(pre + 16 * (b % Q) + 2 * (b / Q), blk + 208, 2);
			memcpy(rec + 192, blk + 192, 16);
			for (int j = 0; j < 8; j++) {
				for (int i = 0; i < 16; i++) {
					int a = q6_k_elem(blk, 32 * j + 2 * i), c = q6_k_elem(blk, 32 * j + 2 * i + 1);
					rec[16 * j + i] = (uint8_t)((a & 0xF) | ((c & 0xF) << 4));
				}
				for (int w = 0; w < 2; w++) {
					uint32_t hw = 0;
					for (int e = 0; e < 16; e++) {
						int hi = q6_k_elem(blk, 32 * j + 16 * w + e) >> 4;
						hw |= (uint32_t)hi << (16 * (e >> 3) + 2 * (e & 7));
					}
					memcpy(rec + 128 + 8 * j + 4 * w, &hw, 4);
				}
			}
		}
	}
	return stride;
}

void gemv_q6_k_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum16, float xscale, float *y) {
	int B = K / 256, stride = q6_k_tile_stride(K);
	int L = q4_k_lanes_per_row(K), Q = L / 8, n = B / Q;
	float v[32], w[32];
	(void)sum16;
	for (int r = 0; r < rows; r++) {
		const uint8_t *pre = tile + (size_t)r * stride, *blocks = pre + 16 * Q;
		for (int l = 0; l < L; l++) {
			int q = l >> 3, jj = l & 7;
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int b = Q * i + q;
				const uint8_t *rec = blocks + 208 * b, *nib = rec + 16 * jj;
				int32_t sumi = 0;
				for (int gg = 0; gg < 2; gg++) {
					uint32_t hw;
					memcpy(&hw, rec + 128 + 8 * jj + 4 * gg, 4);
					int32_t dot = 0;
					for (int e = 0; e < 16; e++) {
						int el = 16 * gg + e;
						int q6 = ((nib[el >> 1] >> (4 * (el & 1))) & 0xF) | (((hw >> (16 * (e >> 3) + 2 * (e & 7))) & 3) << 4);
						dot += (q6 - 32) * xq[256 * b + 32 * jj + el];
					}
					sumi += (int8_t)rec[192 + 2 * jj + gg] * dot;
				}
				acc = fmaf(fp16_to_fp32(ld16(pre + 16 * q + 2 * i)), (float)sumi, acc);
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

/* ---- IQ4_XS: per-32 6-bit scales, non-linear 4-bit codes ---- */

int iq4_xs_tile_stride(int K) { return 16 * (q4_k_lanes_per_row(K) / 8) + 136 * (K / 256); }
int iq4_kvalue(int q) { return kvalues_iq4nl[q & 15]; }

int repack_iq4_xs(const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 256, L = q4_k_lanes_per_row(K), Q = L / 8, stride = iq4_xs_tile_stride(K);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * 136;
		uint8_t *pre = dst + (size_t)r * stride, *nib = pre + 16 * Q, *scs = nib + 128 * B;
		memset(pre, 0, 16 * Q);
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + 136 * b, *sl = blk + 4, *qs = blk + 8;
			uint16_t sh = ld16(blk + 2);
			memcpy(pre + 16 * (b % Q) + 2 * (b / Q), blk, 2);
			for (int j = 0; j < 8; j++) {
				int ls = ((sl[j / 2] >> (4 * (j % 2))) & 0xf) | (((sh >> (2 * j)) & 3) << 4);
				scs[8 * b + j] = (uint8_t)(ls - 32);
				const uint8_t *q = qs + 16 * j;
				for (int i = 0; i < 16; i++) {
					int e0 = 2 * i, e1 = 2 * i + 1;
					int n0 = e0 < 16 ? (q[e0] & 0xF) : (q[e0 - 16] >> 4);
					int n1 = e1 < 16 ? (q[e1] & 0xF) : (q[e1 - 16] >> 4);
					nib[128 * b + 16 * j + i] = (uint8_t)(n0 | (n1 << 4));
				}
			}
		}
	}
	return stride;
}

void gemv_iq4_xs_ref(const uint8_t *tile, int rows, int K, const int8_t *xq, float xscale, float *y) {
	int B = K / 256, stride = iq4_xs_tile_stride(K);
	int L = q4_k_lanes_per_row(K), Q = L / 8, n = B / Q;
	float v[32], w[32];
	for (int r = 0; r < rows; r++) {
		const uint8_t *pre = tile + (size_t)r * stride, *nib = pre + 16 * Q, *scs = nib + 128 * B;
		for (int l = 0; l < L; l++) {
			int q = l >> 3, jj = l & 7;
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int b = Q * i + q;
				const uint8_t *p = nib + 128 * b + 16 * jj;
				int32_t dot = 0;
				for (int e = 0; e < 32; e++) dot += kvalues_iq4nl[(p[e >> 1] >> (4 * (e & 1))) & 0xF] * xq[256 * b + 32 * jj + e];
				int32_t sumi = (int8_t)scs[8 * b + jj] * dot;
				acc = fmaf(fp16_to_fp32(ld16(pre + 16 * q + 2 * i)), (float)sumi, acc);
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

/* ---- block-32 formats at K = 640 ---- */

static int b32_block_bytes(uint32_t type) { return type == GG_Q8_0 ? 32 : 16; }

int b32_tile_stride(uint32_t type, int K) {
	int B = K / 32;
	return 48 + B * b32_block_bytes(type) + (type == GG_Q5_0 ? 4 * B : 0);
}

/* integer code of element e (0..31) of a raw block: the value before the offset */
static int b32_code(uint32_t type, const uint8_t *blk, int e) {
	switch (type) {
	case GG_Q4_0: return e < 16 ? (blk[2 + e] & 0xF) : (blk[2 + e - 16] >> 4);
	case GG_IQ4_NL: return e < 16 ? (blk[2 + e] & 0xF) : (blk[2 + e - 16] >> 4);
	case GG_Q5_0: {
		uint32_t qh = ld32(blk + 2);
		int lo = e < 16 ? (blk[6 + e] & 0xF) : (blk[6 + e - 16] >> 4);
		int hi = e < 16 ? ((qh >> e) & 1) : ((qh >> (e - 16 + 12)) & 0x10) >> 4;
		return lo | (hi << 4);
	}
	case GG_Q8_0: return (int8_t)blk[2 + e];
	}
	return 0;
}

static int b32_offset(uint32_t type) { return type == GG_Q4_0 ? 8 : type == GG_Q5_0 ? 16 : 0; }

int repack_b32(uint32_t type, const uint8_t *src, int rows, int K, uint8_t *dst) {
	int B = K / 32, stride = b32_tile_stride(type, K), bb = b32_block_bytes(type);
	uint32_t gb;
	gguf_block(type, NULL, &gb);
	if (!dst) return stride;
	for (int r = 0; r < rows; r++) {
		const uint8_t *s = src + (size_t)r * B * gb;
		uint8_t *pre = dst + (size_t)r * stride, *data = pre + 48, *qh = data + bb * B;
		memset(pre, 0, 48);
		for (int b = 0; b < B; b++) {
			const uint8_t *blk = s + gb * b;
			memcpy(pre + 2 * b, blk, 2);
			if (type == GG_Q8_0) {
				memcpy(data + 32 * b, blk + 2, 32);
			} else {
				for (int i = 0; i < 16; i++) {
					int c0 = b32_code(type, blk, 2 * i) & 0xF, c1 = b32_code(type, blk, 2 * i + 1) & 0xF;
					data[16 * b + i] = (uint8_t)(c0 | (c1 << 4));
				}
				if (type == GG_Q5_0) {
					uint32_t w = 0;
					for (int e = 0; e < 32; e++) w |= (uint32_t)(b32_code(type, blk, e) >> 4) << e;
					memcpy(qh + 4 * b, &w, 4);
				}
			}
		}
	}
	return stride;
}

static int b32_tile_code(uint32_t type, const uint8_t *pre, int K, int e) {
	int B = K / 32, b = e >> 5, el = e & 31, bb = b32_block_bytes(type);
	const uint8_t *data = pre + 48;
	if (type == GG_Q8_0) return (int8_t)data[32 * b + el];
	int q = (data[16 * b + (el >> 1)] >> (4 * (el & 1))) & 0xF;
	if (type == GG_Q5_0) { uint32_t w; memcpy(&w, data + bb * B + 4 * b, 4); q |= ((w >> el) & 1) << 4; }
	return q;
}

float b32_tile_value(uint32_t type, const uint8_t *tile, int K, int r, int e) {
	const uint8_t *pre = tile + (size_t)r * b32_tile_stride(type, K);
	float d = fp16_to_fp32(ld16(pre + 2 * (e >> 5)));
	int q = b32_tile_code(type, pre, K, e);
	if (type == GG_IQ4_NL) return d * kvalues_iq4nl[q];
	return d * (float)(q - b32_offset(type));
}

void gemv_b32_ref(uint32_t type, const uint8_t *tile, int rows, int K, const int8_t *xq, const int32_t *sum32, float xscale, float *y) {
	int B = K / 32, stride = b32_tile_stride(type, K), L = 4, n = B / L, off = b32_offset(type);
	float v[4], w[4];
	(void)sum32;
	for (int r = 0; r < rows; r++) {
		const uint8_t *pre = tile + (size_t)r * stride;
		for (int l = 0; l < L; l++) {
			float acc = 0.f;
			for (int i = 0; i < n; i++) {
				int b = l + L * i;
				int32_t dot = 0;
				for (int e = 0; e < 32; e++) {
					int q = b32_tile_code(type, pre, K, 32 * b + e);
					int val = type == GG_IQ4_NL ? kvalues_iq4nl[q] : q - off;
					dot += val * xq[32 * b + e];
				}
				acc = fmaf(fp16_to_fp32(ld16(pre + 2 * b)), (float)dot, acc);
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
