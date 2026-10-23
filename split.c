/* The row/expert/head split over the dies, shared by the holder, the program builder,
 * the twin and the tests so every one of them draws the same boundaries.
 *
 * SPLIT_SHARES=s0,s1,...  (integers, one per die; default equal) weights die d's share:
 * die d owns [split_bound(total, d, n), split_bound(total, d + 1, n)). Die 0 of archy has
 * a bad VRAM band above 5.76 GiB (cx-flash's device-usable-bytes), so it gets a smaller
 * share and the attention die (ATTN_DIE, default 0: the die that holds attn_q/k/v and
 * runs the attention core) moves off it. */
#include "split.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int shares_n = -1; static unsigned long shares[16], shares_sum;

static void load_shares(void) {
	shares_n = 0; shares_sum = 0;
	const char *s = getenv("SPLIT_SHARES");
	if (!s) return;
	char *copy = strdup(s), *save = NULL;
	for (char *tok = strtok_r(copy, ",", &save); tok && shares_n < 16; tok = strtok_r(NULL, ",", &save)) {
		unsigned long v = strtoul(tok, NULL, 10);
		if (!v) { fprintf(stderr, "split: SPLIT_SHARES entries must be positive\n"); exit(1); }
		shares[shares_n++] = v; shares_sum += v;
	}
	free(copy);
}

unsigned split_bound(unsigned total, int d, int n) {
	if (shares_n < 0) load_shares();
	if (d <= 0) return 0;
	if (d >= n) return total;
	if (shares_n == 0) return (unsigned)((unsigned long)total * (unsigned)d / (unsigned)n);
	if (shares_n != n) { fprintf(stderr, "split: SPLIT_SHARES has %d entries for %d dies\n", shares_n, n); exit(1); }
	unsigned long acc = 0; for (int i = 0; i < d; i++) acc += shares[i];
	return (unsigned)((unsigned long)total * acc / shares_sum);
}

int attn_die(void) {
	const char *s = getenv("ATTN_DIE");
	return s ? atoi(s) : 0;
}

/* ATTN_MAP=d0,d1,...,d11: the die that runs attention layer i (block 3 + 4 i) and keeps its projections, KV cache and indexer
 * pool; unset: every attention layer on attn_die() */
int attn_die_of(int blk) {
	static int map[12], mapped = -1;
	if (mapped < 0) {
		mapped = 0;
		const char *s = getenv("ATTN_MAP");
		if (s) {
			char *copy = strdup(s), *save = NULL; int n = 0;
			for (char *tok = strtok_r(copy, ",", &save); tok && n < 12; tok = strtok_r(NULL, ",", &save)) map[n++] = atoi(tok);
			free(copy);
			if (n != 12) { fprintf(stderr, "split: ATTN_MAP needs 12 dies, got %d\n", n); exit(1); }
			mapped = 1;
		}
	}
	int i = (blk - 3) / 4;
	return mapped && i >= 0 && i < 12 ? map[i] : attn_die();
}
