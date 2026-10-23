#include "ple.h"

const uint64_t ple_multipliers[PLE_NGRAM] = { 23703573157769ull, 20109073645365ull, 8052911324071ull };
const uint64_t ple_head_offsets[PLE_HEADS] = {
	0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297,
	160000374, 180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275 };
const uint64_t ple_head_vocab[PLE_HEADS] = {
	20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
	20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171 };

static int is_end(uint32_t id, const uint32_t *ends, int nends) {
	for (int i = 0; i < nends; i++) if (ends[i] == id) return 1;
	return 0;
}

void ple_rows(const uint32_t *ids, size_t pos, uint32_t absent, const uint32_t *ends, int nends, uint64_t rows[PLE_HEADS]) {
	ple_rows_c(ids, pos, absent, ends, nends, ple_multipliers, ple_head_offsets, ple_head_vocab, rows);
}

void ple_rows_c(const uint32_t *ids, size_t pos, uint32_t absent, const uint32_t *ends, int nends, const uint64_t *mult, const uint64_t *off, const uint64_t *voc, uint64_t rows[PLE_HEADS]) {
	uint64_t ctx[PLE_NGRAM];
	for (size_t back = 0; back < PLE_NGRAM; back++) {
		int ok = back <= pos;
		for (size_t p = pos - back; ok && p < pos; p++) if (is_end(ids[p], ends, nends)) ok = 0;
		ctx[back] = ok ? ids[pos - back] : absent;
	}
	int h = 0;
	for (int order = 2; order <= PLE_NGRAM; order++) {
		uint64_t mixed = 0;
		for (int i = 0; i < order; i++) mixed ^= ctx[i] * mult[i];
		for (int k = 0; k < PLE_HEADS / (PLE_NGRAM - 1); k++, h++) rows[h] = off[h] + mixed % voc[h];
	}
}

#include "fmath.h"
#include <math.h>
#include <string.h>

void ple_gate_gpu(const float *kn, const float *qn, const float *v, float *g, float *gates) {
	const float rsc = (float)(1.0 / sqrt(2560.0));
	for (int l = 0; l < 4; l++) {
		float ws[8];
		for (int w = 0; w < 8; w++) {
			float lv[32], tv[32];
			for (int ln = 0; ln < 32; ln++) {
				int t = 32 * w + ln;
				const float *a = kn + 2560 * l + t, *b = qn + 2560 * l + t;
				float s = a[0] * b[0];
				for (int k = 1; k < 10; k++) s = fmaf(a[256 * k], b[256 * k], s);
				lv[ln] = s;
			}
			for (int o = 16; o >= 1; o >>= 1) {
				for (int ln = 0; ln < 32; ln++) tv[ln] = lv[ln] + lv[ln ^ o];
				memcpy(lv, tv, sizeof lv);
			}
			ws[w] = lv[0];
		}
		float s = ws[0] + ws[1];
		for (int w = 2; w < 8; w++) s = s + ws[w];
		s = s * rsc;
		float mag = fabsf(s);
		if (!(mag > 1e-6f)) mag = 1e-6f;
		float root = sqrtf(mag);
		float sign = (float)((s > 0.f) - (0.f > s));
		float gate = sigmoid_gpu(sign * root);
		gates[l] = gate;
		for (int c = 0; c < 2560; c++) g[2560 * l + c] = gate * v[c];
	}
}

void ple_conv_gpu(const float *x, float state[10][10240], const float *w, unsigned t, const float *g, float *s) {
	const float *x9 = t >= 9 ? state[(t - 9) % 10] : NULL, *x6 = t >= 6 ? state[(t - 6) % 10] : NULL, *x3 = t >= 3 ? state[(t - 3) % 10] : NULL;
	for (int c = 0; c < 10240; c++) {
		float y = w[4 * c] * (x9 ? x9[c] : 0.f);
		y = fmaf(w[4 * c + 1], x6 ? x6[c] : 0.f, y);
		y = fmaf(w[4 * c + 2], x3 ? x3[c] : 0.f, y);
		y = fmaf(w[4 * c + 3], x[c], y);
		s[c] = s[c] + (g[c] + silu_gpu(y));
	}
	memcpy(state[t % 10], x, 4 * 10240);
}
