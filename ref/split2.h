#pragma once
/* The K-split layout (v2) over n = 6 dies, shared by the loader, the program builder and the tests.
 *
 * The residual stream S (4 lanes x 2560) is distributed by channel: die d owns channels
 * [c0, c0 + cn), whole 32-channel blocks, 80 blocks split 14,14,13,13,13,13. Projections that sum
 * into the stream are K-split: each die multiplies its slice of the input by the matching columns
 * and the partial outputs are reduce-scattered to the owners (then all-gathered where every die
 * needs the whole vector). Projections that fan out from a gathered vector stay row-split:
 *
 *   hc_*_dj (324 x 10240 BF16)    columns l 2560 + c, c in C_d, all rows       -> [324][4][cn]
 *   hc_*_up (10240 x 320 BF16)    rows l 2560 + c, c in C_d                    -> [4 cn][320]
 *   ffn_gate_inp_x (513 x 2560)   columns C_d, all rows                        -> [513][cn]
 *   attn_qkv / attn_gate / ssm_*  rows of the die's value heads [h0, h0 + 8) and key heads
 *   ssm_out (2560 x 6144)         columns of the die's value heads (128 per head), all rows
 *   attn_q (12288 x 2560)         rows of the die's query heads [q0, q0 + 4) (q ++ gate, 512 rows each)
 *   attn_output (2560 x 6144)     columns of the die's query heads (256 per head), all rows
 *   experts (512 x 640 FFN rows)  64-row units u = 10 e + j; die u mod 6 owns unit u: its gate and
 *                                 up rows [64 j, 64 j + 64) and its down columns [64 j, 64 j + 64)
 *   shared expert                 as expert 512
 *   ple_key (10240 x 2560)        rows l 2560 + c, c in C_d;  ple_value (2560 x 2560) rows C_d
 *   output_hc_down (320 x 10240)  columns as hc_*_dj;  output_hc_up rows as hc_*_up
 *   output (248320 x 2560)        rows split evenly;  token_embd on the die that holds it
 *
 * The hc down rows (324) are owned in RS order by die j / 54; the stream rows of a reduce-scatter
 * of a 2560-row partial by the channel owner. */
#include <stdint.h>

#define SPLIT2_DIES 6
#define SPLIT2_UNITS 10            /* 64-row units of the 640-row expert FFN */
#define SPLIT2_UNIT_ROWS 64
#define SPLIT2_EXPERTS 513         /* 512 routed experts + the shared expert as expert 512 */

typedef struct {
	int d, n;
	int cb0, cbn;      /* channel blocks of 32: [cb0, cb0 + cbn) of 80 */
	int c0, cn;        /* channels [c0, c0 + cn) */
	int h0, hn;        /* value heads [h0, h0 + hn) of 48 */
	int klo, nk;       /* key heads (klo + j) % 16, j < nk (value head h reads key head h % 16) */
	int q0, qn;        /* query heads [q0, q0 + qn) of 24 */
	int j0, jn;        /* hc down rows [j0, j0 + jn) of 324 owned in the reduce-scatter */
	int o0, on;        /* output head rows [o0, o0 + on) of 248320 */
} split2_t;

static inline int split2_bound(int total, int d, int n) { return (int)((long)total * d / n); }

static inline void split2_of(split2_t *s, int d, int n) {
	s->d = d; s->n = n;
	/* 80 blocks: 14,14,13,13,13,13 for n = 6 (the larger shares first) */
	int base = 80 / n, extra = 80 % n;
	s->cb0 = d * base + (d < extra ? d : extra); s->cbn = base + (d < extra);
	s->c0 = 32 * s->cb0; s->cn = 32 * s->cbn;
	s->h0 = split2_bound(48, d, n); s->hn = split2_bound(48, d + 1, n) - s->h0;
	s->klo = s->h0 % 16; s->nk = s->hn < 16 ? s->hn : 16;
	s->q0 = split2_bound(24, d, n); s->qn = split2_bound(24, d + 1, n) - s->q0;
	s->j0 = split2_bound(324, d, n); s->jn = split2_bound(324, d + 1, n) - s->j0;
	s->o0 = split2_bound(248320, d, n); s->on = split2_bound(248320, d + 1, n) - s->o0;
}

/* the die that owns 64-row unit j of expert e (e = 512: the shared expert) */
static inline int split2_unit_die(int e, int j, int n) { return (SPLIT2_UNITS * e + j) % n; }
/* the units of expert e die d owns, ascending, into u[]; returns their count (1 or 2 for n = 6) */
static inline int split2_units(int e, int d, int n, int *u) {
	int k = 0;
	for (int j = 0; j < SPLIT2_UNITS; j++) if (split2_unit_die(e, j, n) == d) u[k++] = j;
	return k;
}
/* the channel owner of hidden row r (a reduce-scatter destination) */
static inline int split2_channel_die(int r, int n) {
	int b = r / 32, base = 80 / n, extra = 80 % n, big = extra * (base + 1);
	return b < big ? b / (base + 1) : extra + (b - big) / base;
}
