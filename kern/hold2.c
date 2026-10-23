/* hold2: the one-process weight holder for all six dies of the split, with the routed experts in
 * 64-row units (the K-split MoE layout, ref/split2.h). It reads every tensor of the GGUF shard once,
 * in file order, slices it for each die and uploads each die's slice into that die's arena; then it
 * exports the six arenas with CUDA IPC handles, writes six manifests in the hold.c format and sleeps
 * until SIGTERM. Everything except the routed experts is laid out exactly as hold.c's split=d/6 does
 * (same SPLIT_SHARES / ATTN_MAP / ATTN_DIE, same merges, same tiles), so the engine maps it the same.
 *
 *   hold2 <gguf> <manifest0> ... <manifest5> [token_embd] [output]
 *   HOLD2_DEVICES=p0,...,p5 (default 0,1,2,3,4,5): the physical die (CUDA ordinal under CUDA_DEVICE_ORDER=PCI_BUS_ID with all
 *   dies visible) that holds logical die d; the manifest records the physical index, as hold.c does.
 *
 * Routed experts: expert e's 640 FFN rows are ten 64-row units u = 10 e + j. Per block, unit u goes to
 * one die by a smooth weighted round robin whose weights are the dies' expert capacities (the v1 expert
 * counts the shares give) with the block's shared-expert die (blk % 6, which also runs the whole shared
 * expert as before) scaled by HOLD2_SHEXP_FRAC (default 0.45) so its MoE work stays level with the rest.
 * A die keeps its units in (expert, j) order:
 *   ffn_gate_exps / ffn_up_exps   Q2_0 tiles of the units' gate / up rows: 64 rows x K 2560 per unit
 *                                 (manifest rows 64, experts = units)
 *   ffn_down_exps                 per unit, block j of every one of the expert's 2560 down rows:
 *                                 2560 x 16 B of codes, then 2560 x fp16 scales (46080 B per unit;
 *                                 manifest rows 2560, K 64, stride 18, experts = units)
 *   emap                          u32[513]: (first unit << 8) | units of expert e on this die
 */
#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include "quant.h"
#include "prog.h"
#include "split.h"
#include <cuda.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

#define ND 6
#define NBLK 48
#define NEXP 512
#define NUNIT (NEXP * 10)
#define UNIT_DOWN_BYTES (2560 * 18)

static volatile sig_atomic_t stop;
static void onsig(int s) { (void)s; stop = 1; }

#define Q6K_SLICED(type, K) ((type) == GG_Q6_K && (K) == 6144)
static int tile_stride(uint32_t type, int K) {
	switch (type) {
	case GG_Q2_0: return q2_0_tile_stride(K);
	case GG_Q4_K: return q4_k_tile_stride(K);
	case GG_Q5_K: return q5_k_tile_stride(K);
	case GG_Q3_K: return q3_k_tile_stride(K);
	case GG_Q6_K: return q6_k_tile_stride(K);
	case GG_IQ4_XS: return iq4_xs_tile_stride(K);
	case GG_Q4_0: case GG_Q5_0: case GG_Q8_0: case GG_IQ4_NL: return b32_tile_stride(type, K);
	default: return 0;
	}
}
static int repack(uint32_t type, const uint8_t *src, int rows, int K, uint8_t *dst) {
	switch (type) {
	case GG_Q2_0: return repack_q2_0(src, rows, K, dst);
	case GG_Q4_K: return repack_q4_k(src, rows, K, dst);
	case GG_Q5_K: return repack_q5_k(src, rows, K, dst);
	case GG_Q3_K: return repack_q3_k(src, rows, K, dst);
	case GG_Q6_K: return repack_q6_k(src, rows, K, dst);
	case GG_IQ4_XS: return repack_iq4_xs(src, rows, K, dst);
	default: return repack_b32(type, src, rows, K, dst);
	}
}

typedef struct { const char *out, *a, *b; int zero_b; } merge_t;
static const merge_t merges[] = {
	{ "hc_attn_dj.weight",    "hc_attn_down.weight", "hc_attn_inject.weight",   0 },
	{ "hc_ffn_dj.weight",     "hc_ffn_down.weight",  "hc_ffn_inject.weight",    0 },
	{ "ffn_gate_inp_x.weight", "ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight", 0 },
	{ "ssm_ab.weight",        "ssm_alpha.weight",    "ssm_beta.weight",         0 },
	{ "ssm_ab.bias",          "ssm_dt.bias",         "ssm_dt.bias",             1 },
	{ "indexer_x.weight",    "indexer.q_proj.weight", "indexer.k_proj.weight",  0 },
};
#define NMERGE ((int)(sizeof merges / sizeof merges[0]))

static const char *suffix(const char *name) {
	if (strncmp(name, "blk.", 4)) return NULL;
	const char *p = strchr(name + 4, '.');
	return p ? p + 1 : NULL;
}
static int merged_source(const char *name) {
	const char *s = suffix(name);
	if (!s) return 0;
	for (int i = 0; i < NMERGE; i++) if (!strcmp(s, merges[i].a) || !strcmp(s, merges[i].b)) return 1;
	return 0;
}
static int wanted(const char *name, int extra_embd, int extra_out) {
	if (!strncmp(name, "blk.", 4)) { int b = atoi(name + 4); return b >= 0 && b < NBLK && !merged_source(name); }
	if (!strncmp(name, "token_embd", 10)) return extra_embd;
	if (!strncmp(name, "output", 6)) return extra_out;
	return 0;
}
static const gguf_tensor *blk_tensor(const gguf_file *g, int b, const char *suf) {
	char name[128]; snprintf(name, sizeof name, "blk.%d.%s", b, suf);
	const gguf_tensor *t = gguf_find(g, name);
	return t && t->data ? t : NULL;
}
static int routed_expert_tensor(const char *name) {
	const char *x = suffix(name);
	return x && (!strcmp(x, "ffn_gate_exps.weight") || !strcmp(x, "ffn_up_exps.weight") || !strcmp(x, "ffn_down_exps.weight"));
}

/* ---- hold.c's split (everything but the routed experts) ---- */
typedef struct { int r0, n; } range_t;
typedef struct { range_t r[6]; int nr; int rows_total; int experts; } sel_t;
static void sel_whole(sel_t *s, int rows, int experts) { s->nr = 1; s->r[0].r0 = 0; s->r[0].n = experts > 1 ? experts : rows; s->rows_total = rows; s->experts = experts; }
static void select_rows(const split_t *p, const char *name, int rows, int experts, sel_t *s) {
	sel_whole(s, rows, experts);
	if (!strcmp(name, "output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->o0, p->on }; return; }
	if (!strcmp(name, "output_hc_down.weight")) { s->nr = 1; s->r[0] = (range_t){ p->hd0, p->hdn }; return; }
	if (!strcmp(name, "output_hc_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; return; }
	const char *x = suffix(name);
	if (!x) return;
	if (strstr(x, "_shexp.weight") && !p->shexp) { s->nr = 1; s->r[0] = (range_t){ 0, 0 }; return; }
	if (!strcmp(x, "hc_attn_dj.weight") || !strcmp(x, "hc_ffn_dj.weight")) { s->nr = 1; s->r[0] = (range_t){ p->dj0, p->djn }; }
	else if (!strcmp(x, "hc_attn_up.weight") || !strcmp(x, "hc_ffn_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; }
	else if (!strcmp(x, "attn_qkv.weight") || !strcmp(x, "ssm_conv1d.weight")) {
		int n1 = p->klo + p->nk <= 16 ? p->nk : 16 - p->klo, n2 = p->nk - n1; s->nr = 0;
		s->r[s->nr++] = (range_t){ 128 * p->klo, 128 * n1 }; if (n2) s->r[s->nr++] = (range_t){ 0, 128 * n2 };
		s->r[s->nr++] = (range_t){ 2048 + 128 * p->klo, 128 * n1 }; if (n2) s->r[s->nr++] = (range_t){ 2048, 128 * n2 };
		s->r[s->nr++] = (range_t){ 4096 + 128 * p->h0, 128 * p->hn };
	}
	else if (!strcmp(x, "attn_gate.weight")) { s->nr = 1; s->r[0] = (range_t){ 128 * p->h0, 128 * p->hn }; }
	else if (!strcmp(x, "ssm_ab.weight") || !strcmp(x, "ssm_ab.bias")) { s->nr = 2; s->r[0] = (range_t){ p->h0, p->hn }; s->r[1] = (range_t){ 48 + p->h0, p->hn }; s->rows_total = 96; }
	else if (!strcmp(x, "ssm_a")) { s->nr = 1; s->r[0] = (range_t){ p->h0, p->hn }; s->rows_total = 48; }
	else if (!strcmp(x, "ssm_out.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "attn_q.weight") || !strcmp(x, "attn_k.weight") || !strcmp(x, "attn_v.weight") || !strcmp(x, "indexer_x.weight") || !strcmp(x, "indexer.q_norm.weight") || !strcmp(x, "indexer.k_norm.weight")) { s->nr = 1; s->r[0] = (range_t){ 0, p->d == attn_die_of(atoi(name + 4)) ? rows : 0 }; }
	else if (!strcmp(x, "attn_output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "ple_key.weight")) { s->nr = 1; s->r[0] = (range_t){ p->pk0, p->pkn }; }
	else if (!strcmp(x, "ple_value.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
}
static size_t sel_bytes(const sel_t *s, size_t nbytes, int rows, int st) {
	size_t n = 0; for (int i = 0; i < s->nr; i++) n += s->r[i].n;
	if (s->experts > 1) return st ? (size_t)st * rows * n : nbytes / s->experts * n;
	size_t rowb = st ? (size_t)st : nbytes / s->rows_total;
	return rowb * n;
}
static int dev_row_stride(uint32_t type, int K) { return Q6K_SLICED(type, K) ? 4 * q6_k_tile_stride(1536) : tile_stride(type, K); }

/* ---- the dies ---- */
typedef struct { CUcontext ctx; CUdeviceptr arena; size_t off, total; FILE *m; const char *mpath; } die_t;
static die_t D[ND];
static int dry;
static size_t cap; static uint8_t *buf;
static void need(size_t bytes) { if (bytes > cap) { free(buf); cap = bytes; buf = malloc(cap); if (!buf) { fprintf(stderr, "hold2: out of host memory\n"); exit(1); } } }
static void emit(int d, const char *name, const uint8_t *img, size_t bytes, uint32_t type, int rows, int K, int st, int experts) {
	die_t *x = &D[d];
	x->off = (x->off + 255) & ~(size_t)255;
	if (dry && getenv("HOLD2_LIST") && x->mpath) {   /* HOLD_DRY=1 HOLD2_LIST=1: the manifest lines of every die on stdout, "die name off bytes type rows K stride experts" */
		printf("%d %s %zu %zu %u %d %d %d %d\n", d, name, x->off, bytes, type, rows, K, st, experts);
	}
	if (!dry) {
		if (x->off + bytes > x->total) { fprintf(stderr, "hold2: die %d arena overflow at %s\n", d, name); exit(1); }
		if (bytes) { CK(cuCtxSetCurrent(x->ctx)); CK(cuMemcpyHtoD(x->arena + x->off, img, bytes)); }
		fprintf(x->m, "%s %zu %zu %u %d %d %d %d\n", name, x->off, bytes, type, rows, K, st, experts);
	}
	x->off += bytes;
}
static void emit_sel(int d, const char *name, uint32_t type, const uint8_t *src, size_t nbytes, int rows, int K, int experts, const sel_t *s) {
	int st = strncmp(name, "token_embd", 10) ? dev_row_stride(type, K) : 0;
	size_t bytes = sel_bytes(s, nbytes, rows, st), o = 0;
	int nsel = 0;
	if (dry) {   /* the sizing pass: the manifest line's row count as the loading pass will write it (HOLD2_LIST shows it) */
		for (int i = 0; i < s->nr; i++) nsel += s->r[i].n;
		if (Q6K_SLICED(type, K) && experts == 1) emit(d, name, NULL, bytes, type, 4 * nsel, 1536, (int)q6_k_tile_stride(1536), 1);
		else if (s->experts > 1) emit(d, name, NULL, bytes, type, rows, K, st, nsel);
		else emit(d, name, NULL, bytes, type, nsel, K, st, 1);
		return;
	}
	need(bytes);
	if (Q6K_SLICED(type, K) && experts == 1) {
		for (int i = 0; i < s->nr; i++) nsel += s->r[i].n;
		size_t srow = nbytes / s->rows_total, sl = srow / 4, st1 = (size_t)q6_k_tile_stride(1536);
		uint8_t *tmp = malloc(sl * (size_t)(nsel ? nsel : 1));
		for (int sidx = 0; sidx < 4; sidx++) {
			int k = 0;
			for (int i = 0; i < s->nr; i++) for (int r = s->r[i].r0; r < s->r[i].r0 + s->r[i].n; r++, k++) memcpy(tmp + sl * k, src + srow * r + sl * sidx, sl);
			repack_q6_k(tmp, nsel, 1536, buf + st1 * (size_t)nsel * sidx);
		}
		free(tmp);
		emit(d, name, buf, bytes, type, 4 * nsel, 1536, (int)st1, 1);
		return;
	}
	if (s->experts > 1) {
		size_t per = nbytes / experts, dper = st ? (size_t)st * rows : per;
		for (int i = 0; i < s->nr; i++) for (int e = s->r[i].r0; e < s->r[i].r0 + s->r[i].n; e++, nsel++) {
			if (st) repack(type, src + per * e, rows, K, buf + o); else memcpy(buf + o, src + per * e, per);
			o += dper;
		}
		emit(d, name, buf, bytes, type, rows, K, st, nsel);
		return;
	}
	size_t srow = nbytes / s->rows_total;
	for (int i = 0; i < s->nr; i++) {
		int r0 = s->r[i].r0, n = s->r[i].n;
		if (st) { repack(type, src + srow * r0, n, K, buf + o); o += (size_t)st * n; }
		else { memcpy(buf + o, src + srow * r0, srow * n); o += srow * n; }
		nsel += n;
	}
	emit(d, name, buf, bytes, type, nsel, K, st, 1);
}

/* ---- routed expert units ---- */
static uint8_t assign[NBLK][NUNIT];   /* die of unit u = 10 e + j in block b */
static int units_of[NBLK][ND];
static void assign_units(void) {
	double frac = getenv("HOLD2_SHEXP_FRAC") ? atof(getenv("HOLD2_SHEXP_FRAC")) : 0.45;
	int capw[ND];
	for (int d = 0; d < ND; d++) capw[d] = (int)(split_bound(NEXP, d + 1, ND) - split_bound(NEXP, d, ND));
	for (int b = 0; b < NBLK; b++) {
		double w[ND], sum = 0; long q[ND]; long qs = 0;
		for (int d = 0; d < ND; d++) { w[d] = capw[d] * (d == b % ND ? frac : 1.0); sum += w[d]; }
		/* integer quotas by largest remainder */
		double rem[ND];
		for (int d = 0; d < ND; d++) { double x = NUNIT * w[d] / sum; q[d] = (long)x; rem[d] = x - (double)q[d]; qs += q[d]; }
		while (qs < NUNIT) { int best = 0; for (int d = 1; d < ND; d++) if (rem[d] > rem[best]) best = d; q[best]++; rem[best] = -1; qs++; }
		/* smooth weighted round robin over the units in order: each expert's ten units spread over the dies */
		long cur[ND] = { 0 };
		for (int u = 0; u < NUNIT; u++) {
			int best = 0;
			for (int d = 0; d < ND; d++) { cur[d] += q[d]; if (cur[d] > cur[best]) best = d; }
			cur[best] -= NUNIT;
			assign[b][u] = (uint8_t)best;
		}
		for (int d = 0; d < ND; d++) units_of[b][d] = 0;
		for (int u = 0; u < NUNIT; u++) units_of[b][assign[b][u]]++;
		for (int d = 0; d < ND; d++) if (units_of[b][d] != q[d]) { fprintf(stderr, "hold2: block %d die %d got %d units for quota %ld\n", b, d, units_of[b][d], q[d]); exit(1); }
	}
}
/* gate / up: unit (e, j) of a [512][640][2560] Q2_0 tensor is rows 64 j .. 64 j + 63 of expert e */
static void emit_units_rows(int b, const char *name, const gguf_tensor *t) {
	const int K = 2560, rows_unit = 64, st = q2_0_tile_stride(K);
	if (t->type != GG_Q2_0 || (int)t->dims[0] != K || t->dims[1] != 640 || t->dims[2] != NEXP) { fprintf(stderr, "hold2: %s: unexpected expert tensor shape/type\n", name); exit(1); }
	size_t ubytes = (size_t)st * rows_unit, srow = (size_t)K / 64 * 18;
	if (dry) { for (int d = 0; d < ND; d++) emit(d, name, NULL, ubytes * units_of[b][d], GG_Q2_0, rows_unit, K, st, units_of[b][d]); return; }
	static uint8_t *dbuf[ND]; static size_t dcap[ND];
	size_t fill[ND] = { 0 };
	for (int d = 0; d < ND; d++) { size_t want = ubytes * units_of[b][d]; if (want > dcap[d]) { free(dbuf[d]); dbuf[d] = malloc(want); dcap[d] = want; } }
	for (int e = 0; e < NEXP; e++) for (int j = 0; j < 10; j++) {
		int d = assign[b][10 * e + j];
		const uint8_t *src = t->data + ((size_t)e * 640 + (size_t)rows_unit * j) * srow;
		repack_q2_0(src, rows_unit, K, dbuf[d] + fill[d]);
		fill[d] += ubytes;
	}
	for (int d = 0; d < ND; d++) emit(d, name, dbuf[d], fill[d], GG_Q2_0, rows_unit, K, st, units_of[b][d]);
}
/* down: unit (e, j) is block j (columns 64 j .. 64 j + 63) of each of expert e's 2560 rows (180 B per row) */
static void emit_units_down(int b, const char *name, const gguf_tensor *t) {
	if (t->type != GG_Q2_0 || t->dims[0] != 640 || t->dims[1] != 2560 || t->dims[2] != NEXP) { fprintf(stderr, "hold2: %s: unexpected expert tensor shape/type\n", name); exit(1); }
	if (dry) { for (int d = 0; d < ND; d++) emit(d, name, NULL, (size_t)UNIT_DOWN_BYTES * units_of[b][d], GG_Q2_0, 2560, 64, 18, units_of[b][d]); return; }
	static uint8_t *dbuf[ND]; static size_t dcap[ND];
	int slot[ND] = { 0 };
	for (int d = 0; d < ND; d++) { size_t want = (size_t)UNIT_DOWN_BYTES * units_of[b][d]; if (want > dcap[d]) { free(dbuf[d]); dbuf[d] = malloc(want); dcap[d] = want; } }
	for (int e = 0; e < NEXP; e++) {
		const uint8_t *ex = t->data + (size_t)e * 2560 * 180;
		for (int j = 0; j < 10; j++) {
			int d = assign[b][10 * e + j];
			uint8_t *u = dbuf[d] + (size_t)UNIT_DOWN_BYTES * slot[d]++;
			uint8_t *codes = u, *scales = u + 2560 * 16;
			for (int r = 0; r < 2560; r++) {
				const uint8_t *blk = ex + (size_t)r * 180 + 18 * j;
				memcpy(scales + 2 * r, blk, 2);
				memcpy(codes + 16 * r, blk + 2, 16);
			}
		}
	}
	for (int d = 0; d < ND; d++) emit(d, name, dbuf[d], (size_t)UNIT_DOWN_BYTES * slot[d], GG_Q2_0, 2560, 64, 18, slot[d]);
}
static void emit_emap(int b) {
	for (int d = 0; d < ND; d++) {
		uint32_t emap[NEXP + 1]; int slot = 0, hist[11] = { 0 };
		for (int e = 0; e < NEXP; e++) {
			int first = slot, n = 0;
			for (int j = 0; j < 10; j++) if (assign[b][10 * e + j] == d) { n++; slot++; }
			emap[e] = ((uint32_t)first << 8) | (uint32_t)n;
			hist[n]++;
		}
		emap[NEXP] = 0;
		if (dry && getenv("HOLD2_LIST")) {   /* the longest unit list ten routed experts can make on this die (moesel2 lists at most 32) */
			int longest = 0, left = 10;
			for (int n = 10; n >= 0 && left > 0; n--) { int take = hist[n] < left ? hist[n] : left; longest += take * n; left -= take; }
			fprintf(stderr, "hold2: block %d die %d: %d units, longest 10-expert list %d%s\n", b, d, slot, longest, longest > 32 ? " OVER 32" : "");
		}
		char name[128]; snprintf(name, sizeof name, "blk.%d.emap", b);
		emit(d, name, (const uint8_t *)emap, sizeof emap, GG_F32, NEXP + 1, 1, 0, 1);
	}
}

int main(int argc, char **argv) {
	if (argc < 2 + ND) { fprintf(stderr, "usage: hold2 <gguf> <manifest0..5> [token_embd] [output]\n"); return 2; }
	const char *path = argv[1];
	int xe = 0, xo = 0;
	for (int d = 0; d < ND; d++) D[d].mpath = argv[2 + d];
	for (int i = 2 + ND; i < argc; i++) { if (!strcmp(argv[i], "token_embd")) xe = 1; if (!strcmp(argv[i], "output")) xo = 1; }
	dry = getenv("HOLD_DRY") != NULL;
	int phys[ND] = { 0, 1, 2, 3, 4, 5 };
	if (getenv("HOLD2_DEVICES")) {
		const char *s = getenv("HOLD2_DEVICES"); char *e;
		for (int d = 0; d < ND; d++) { phys[d] = (int)strtol(s, &e, 10); if (e == s || phys[d] < 0) { fprintf(stderr, "hold2: HOLD2_DEVICES needs %d dies\n", ND); return 2; } s = *e == ',' ? e + 1 : e; }
	}
	for (int d = 0; d < ND && !dry; d++) { char rp[600]; snprintf(rp, sizeof rp, "%s.ready", D[d].mpath); unlink(rp); }
	assign_units();
	gguf_file g;
	if (gguf_open(&g, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
	/* pass 1 (dry) sizes every die's arena, pass 2 loads; both walk the same emits */
	for (int pass = dry ? 1 : 0; pass < 2; pass++) {
		int sizing = pass == 0;
		if (pass == 1 && !dry) {
			CK(cuInit(0));
			for (int d = 0; d < ND; d++) {
				D[d].total = ((D[d].off + 255) & ~(size_t)255) + (4u << 20);   /* slack after the last tensor, as hold.c */
				CUdevice dev; CK(cuDeviceGet(&dev, phys[d]));
				CK(cuCtxCreate_v2(&D[d].ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
				CK(cuMemAlloc(&D[d].arena, D[d].total));
				CUipcMemHandle h; CK(cuIpcGetMemHandle(&h, D[d].arena));
				D[d].m = fopen(D[d].mpath, "w");
				if (!D[d].m) { perror(D[d].mpath); return 1; }
				fprintf(D[d].m, "arena %zu die %d handle ", D[d].total, phys[d]);
				for (size_t i = 0; i < sizeof h; i++) fprintf(D[d].m, "%02x", ((unsigned char *)&h)[i]);
				fprintf(D[d].m, "\n");
				D[d].off = 0;
				fprintf(stderr, "hold2: die %d (physical %d) arena %.3f GB\n", d, phys[d], D[d].total / 1e9);
			}
		}
		int save_dry = dry; if (sizing) dry = 1;
		for (int d = 0; d < ND; d++) if (sizing) D[d].off = 0;
		for (uint64_t i = 0; i < g.ntensors; i++) {
			const gguf_tensor *t = &g.t[i];
			if (!wanted(t->name, xe, xo)) continue;
			int K = (int)t->dims[0], rows = t->nd > 1 ? (int)t->dims[1] : 1, experts = t->nd > 2 ? (int)t->dims[2] : 1;
			if (!t->data) { fprintf(stderr, "%s: not mapped\n", t->name); return 1; }
			int b = strncmp(t->name, "blk.", 4) ? -1 : atoi(t->name + 4);
			if (b >= 0 && routed_expert_tensor(t->name)) {
				if (strstr(t->name, "ffn_down_exps")) emit_units_down(b, t->name, t); else emit_units_rows(b, t->name, t);
				continue;
			}
			for (int d = 0; d < ND; d++) {
				if (!strncmp(t->name, "token_embd", 10) && d != 0) continue;   /* the embedding lives on die 0 only, as the v1 holders load it */
				split_t sp; split_of(&sp, d, ND, b < 0 ? 0 : b);
				sel_t s; select_rows(&sp, t->name, rows, experts, &s);
				emit_sel(d, t->name, t->type, t->data, t->nbytes, rows, K, experts, &s);
			}
		}
		uint8_t *mb = NULL; size_t mcap = 0;
		for (int b = 0; b < NBLK; b++) {
			for (int i = 0; i < NMERGE; i++) {
				const gguf_tensor *a = blk_tensor(&g, b, merges[i].a), *c = blk_tensor(&g, b, merges[i].b);
				if (!a || !c) continue;
				size_t nb = merges[i].zero_b ? a->nbytes : c->nbytes, bytes = a->nbytes + nb;
				if (a->type != c->type || a->dims[0] != c->dims[0]) { fprintf(stderr, "blk.%d.%s: cannot merge\n", b, merges[i].out); return 1; }
				if (!dry) {
					if (bytes > mcap) { free(mb); mcap = bytes; mb = malloc(mcap); }
					memcpy(mb, a->data, a->nbytes);
					if (merges[i].zero_b) memset(mb + a->nbytes, 0, nb); else memcpy(mb + a->nbytes, c->data, nb);
				}
				int K = (int)a->dims[0], rows = a->nd > 1 ? (int)(bytes / (a->nbytes / a->dims[1])) : (int)(bytes / 4);
				char name[128]; snprintf(name, sizeof name, "blk.%d.%s", b, merges[i].out);
				for (int d = 0; d < ND; d++) {
					split_t sp; split_of(&sp, d, ND, b);
					sel_t s; select_rows(&sp, name, rows, 1, &s);
					emit_sel(d, name, a->type, mb, bytes, rows, K, 1, &s);
				}
			}
			/* the prefill's split shared expert: die d's gate / up rows [sh0, sh0+shn) and its down columns [sh0, sh0+shn) (32-column
			   blocks of a 32-block type), as ffn_{gate,up,down}_shexp_s */
			const gguf_tensor *sg = blk_tensor(&g, b, "ffn_gate_shexp.weight"), *su = blk_tensor(&g, b, "ffn_up_shexp.weight"), *sdn = blk_tensor(&g, b, "ffn_down_shexp.weight");
			if (sg && su && sdn && sdn->dims[0] == 640 && sg->dims[1] == 640) {
				uint32_t be = 0, bb = 0; gguf_block(sdn->type, &be, &bb);
				for (int d = 0; d < ND; d++) {
					split_t sp; split_of(&sp, d, ND, b);
					char name[128];
					sel_t s; s.nr = 1; s.r[0] = (range_t){ sp.sh0, sp.shn }; s.rows_total = 640; s.experts = 1;
					snprintf(name, sizeof name, "blk.%d.ffn_gate_shexp_s.weight", b); emit_sel(d, name, sg->type, sg->data, sg->nbytes, 640, 2560, 1, &s);
					snprintf(name, sizeof name, "blk.%d.ffn_up_shexp_s.weight", b); emit_sel(d, name, su->type, su->data, su->nbytes, 640, 2560, 1, &s);
					if (be != 32 && be != 64) { fprintf(stderr, "blk.%d.ffn_down_shexp: blocks of %u (the column split needs 32 or 64)\n", b, be); exit(1); }
					size_t rowb = (size_t)640 / be * bb, nrb = (size_t)sp.shn / be * bb, bytes = 2560 * nrb;
					if (!dry) {
						if (bytes > mcap) { free(mb); mcap = bytes; mb = malloc(mcap); }
						for (int r = 0; r < 2560; r++) memcpy(mb + (size_t)r * nrb, (const uint8_t *)sdn->data + (size_t)r * rowb + (size_t)sp.sh0 / be * bb, nrb);
					}
					sel_t w; sel_whole(&w, 2560, 1);
					snprintf(name, sizeof name, "blk.%d.ffn_down_shexp_s.weight", b); emit_sel(d, name, sdn->type, mb, bytes, 2560, sp.shn, 1, &w);
				}
			}
			emit_emap(b);
		}
		free(mb);
		if (sizing || save_dry) {
			for (int d = 0; d < ND; d++) {
				long u = 0; for (int b2 = 0; b2 < NBLK; b2++) u += units_of[b2][d];
				fprintf(stderr, "hold2: die %d needs %.3f GB (%ld expert units over %d blocks)\n", d, (D[d].off + (4u << 20)) / 1e9, u, NBLK);
			}
		}
		dry = save_dry;
		if (dry) { gguf_close(&g); return 0; }
	}
	for (int d = 0; d < ND; d++) fclose(D[d].m);
	gguf_close(&g);
	for (int d = 0; d < ND; d++) { char rp[600]; snprintf(rp, sizeof rp, "%s.ready", D[d].mpath); FILE *rf = fopen(rp, "w"); if (rf) { fputs("ready\n", rf); fclose(rf); } }
	fprintf(stderr, "hold2: ready; holding until SIGTERM\n");
	signal(SIGTERM, onsig); signal(SIGINT, onsig);
	while (!stop) pause();
	for (int d = 0; d < ND; d++) { CK(cuCtxSetCurrent(D[d].ctx)); CK(cuMemFree(D[d].arena)); }
	return 0;
}
