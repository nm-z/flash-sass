#include "router.h"
#include "fmath.h"
#include <math.h>
#include <string.h>

void router_gpu(const float *r, float sh, unsigned *idx, float *w, float *s) {
	float v[512], m[10], e[10];
	memcpy(v, r, sizeof v);
	float m0 = v[0];
	for (int i = 1; i < 512; i++) if (v[i] > m0) m0 = v[i];
	for (int j = 0; j < 10; j++) {
		float best = v[0];
		for (int i = 1; i < 512; i++) if (v[i] > best) best = v[i];
		unsigned bi = 0xffffffffu;
		for (int i = 0; i < 512; i++) if (memcmp(&v[i], &best, 4) == 0 && (unsigned)i < bi) bi = (unsigned)i;
		idx[j] = bi;
		m[j] = best;
		v[bi] = -INFINITY;
	}
	float den = 0.f;
	for (int j = 0; j < 10; j++) { e[j] = exp_gpu(m[j] - m0); den = j ? den + e[j] : e[j]; }
	float rw = rcp_gpu(den);
	for (int j = 0; j < 10; j++) w[j] = e[j] * rw;
	*s = sigmoid_gpu(sh);
}
