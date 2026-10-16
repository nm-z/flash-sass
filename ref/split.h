#pragma once
/* The split of rows / experts / heads over n dies (ref/split.c): die d owns
 * [split_bound(total, d, n), split_bound(total, d + 1, n)); SPLIT_SHARES weights the
 * dies, ATTN_DIE names the die that holds the attention projections. */
unsigned split_bound(unsigned total, int d, int n);
int attn_die(void);
int attn_die_of(int blk);   /* the die that runs attention block blk (ATTN_MAP), else attn_die() */
