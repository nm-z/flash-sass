#include "attn.h"
#include "checked.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static float q[6144], k[32 * 512], v[512], gate[12288], out[6144];
static float pm[24 * 8], pd[24 * 8], po[24 * 8 * 256];

static void unchanged(const void *p, size_t bytes) {
	const unsigned char *b = p;
	for (size_t i = 0; i < bytes; i++) assert(b[i] == 0x5a);
}
int main(void) {
	const unsigned invalid[] = {0, ATTN_MAX_KEYS + 1u, UINT_MAX};
	for (unsigned j = 0; j < sizeof invalid / sizeof *invalid; j++) {
		memset(out, 0x5a, sizeof out);
		memset(pm, 0x5a, sizeof pm);
		memset(pd, 0x5a, sizeof pd);
		memset(po, 0x5a, sizeof po);
		/* NULL inputs prove count rejection happens before dereferencing them. */
		assert(attn_decode_gpu(NULL, NULL, NULL, invalid[j], NULL, out) == 1);
		unchanged(out, sizeof out);
		assert(attn_partials_gpu(NULL, NULL, NULL, invalid[j], pm, pd, po) == 1);
		unchanged(pm, sizeof pm); unchanged(pd, sizeof pd); unchanged(po, sizeof po);
		assert(attn_decode_checked(q, k, sizeof k / sizeof *k, v, 512, invalid[j], gate, out) == 1);
		unchanged(out, sizeof out);
	}
	for (unsigned i = 0; i < 512; i++) v[i] = 2.f;
	assert(attn_partials_gpu(q, k, v, 1, pm, pd, po) == 0);
	for (unsigned h = 0; h < 24; h++) {
		assert(pm[h * 8] == 0.f); assert(pd[h * 8] == 1.f);
		for (unsigned c = 1; c < 8; c++) {
			assert(pm[h * 8 + c] == ATTN_MINIT); assert(pd[h * 8 + c] == 0.f);
			for (unsigned d = 0; d < 256; d++) assert(po[(h * 8 + c) * 256 + d] == 0.f);
		}
	}
	assert(attn_decode_gpu(q, k, v, 1, gate, out) == 0);
	for (unsigned i = 0; i < 6144; i++) assert(fabsf(out[i] - 1.f) < 1e-6f);
	puts("attention count contract: 3 invalid counts preserve outputs; one-key partials/decode and empty partitions pass");
	return 0;
}
