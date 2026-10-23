#include "sample.h"
#include "fmath.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uint32_t idx; float val; } cand_t;
static int cand_cmp(const void *a, const void *b) {
	const cand_t *x = a, *y = b;
	if (x->val > y->val) return -1;
	if (x->val < y->val) return 1;
	return x->idx < y->idx ? -1 : x->idx > y->idx;
}

void topk_gpu(const float *x, unsigned n, int K, uint32_t *idx, float *val) {
	cand_t *c = malloc(sizeof *c * (n ? n : 1));
	for (unsigned i = 0; i < n; i++) c[i] = (cand_t){ i, x[i] };
	qsort(c, n, sizeof *c, cand_cmp);
	for (int r = 0; r < 32; r++) {
		if (r < K && (unsigned)r < n) { idx[r] = c[r].idx; val[r] = c[r].val; }
		else { idx[r] = 0xffffffffu; val[r] = -INFINITY; }
	}
	free(c);
}

static void bfly_sum(float *t) {   /* t[l] += t[l ^ s] for s = 16, 8, 4, 2, 1 */
	for (int s = 16; s >= 1; s >>= 1) { float n[32]; for (int l = 0; l < 32; l++) n[l] = t[l] + t[l ^ s]; memcpy(t, n, sizeof n); }
}
static void scan_sum(float *c) {   /* inclusive Kogge-Stone: c[l] += c[l - s] for s = 1, 2, 4, 8, 16 */
	for (int s = 1; s <= 16; s <<= 1) { float n[32]; for (int l = 0; l < 32; l++) n[l] = l >= s ? c[l] + c[l - s] : c[l]; memcpy(c, n, sizeof n); }
}

uint32_t sample_gpu(const uint32_t idx[][32], const float val[][32], const uint32_t *bases, int ndies, int K, float topp, float invT, float u) {
	cand_t c[6 * 32]; int nc = 0;
	for (int d = 0; d < 6; d++)
		for (int l = 0; l < 32; l++) {
			if (d < ndies) c[nc++] = (cand_t){ idx[d][l] + bases[d], val[d][l] };
			else c[nc++] = (cand_t){ 0xffffffffu, -INFINITY };
		}
	qsort(c, (size_t)nc, sizeof *c, cand_cmp);
	float sm[32]; uint32_t si[32];
	for (int l = 0; l < 32; l++) { if (l < K) { sm[l] = c[l].val; si[l] = c[l].idx; } else { sm[l] = -INFINITY; si[l] = 0xffffffffu; } }
	float m0 = sm[0], e[32], tot[32], cum[32], w[32], W[32];
	for (int l = 0; l < 32; l++) { e[l] = l < K ? exp_gpu(sm[l] - m0) : 0.f; tot[l] = e[l]; cum[l] = e[l]; }
	bfly_sum(tot); scan_sum(cum);
	float thr0 = tot[0] * topp, excl[32];
	for (int l = 0; l < 32; l++) excl[l] = l > 0 ? cum[l - 1] : 0.f;   /* SHFL.UP of the inclusive scan */
	for (int l = 0; l < 32; l++) {
		int keep = l < K && excl[l] < thr0;
		w[l] = keep ? exp_gpu((sm[l] - m0) * invT) : 0.f;
		W[l] = w[l]; cum[l] = w[l];
	}
	bfly_sum(W); scan_sum(cum);
	float thr = W[0] * u;
	unsigned cand = 32, lk = 0;
	for (int l = 0; l < 32; l++) { if (cum[l] > thr && (unsigned)l < cand) cand = (unsigned)l; if (w[l] > 0.f && (unsigned)l > lk) lk = (unsigned)l; }
	unsigned r = cand < lk ? cand : lk;
	return si[r];
}
