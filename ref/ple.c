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
	uint64_t ctx[PLE_NGRAM];
	for (size_t back = 0; back < PLE_NGRAM; back++) {
		int ok = back <= pos;
		for (size_t p = pos - back; ok && p < pos; p++) if (is_end(ids[p], ends, nends)) ok = 0;
		ctx[back] = ok ? ids[pos - back] : absent;
	}
	int h = 0;
	for (int order = 2; order <= PLE_NGRAM; order++) {
		uint64_t mixed = 0;
		for (int i = 0; i < order; i++) mixed ^= ctx[i] * ple_multipliers[i];
		for (int k = 0; k < PLE_HEADS / (PLE_NGRAM - 1); k++, h++) rows[h] = ple_head_offsets[h] + mixed % ple_head_vocab[h];
	}
}
