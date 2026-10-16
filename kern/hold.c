/* hold: the resident weight holder. Loads the tensors of blocks [b0, b1) (plus the
 * non-block tensors when asked) of the GGUF shard onto one die, repacked into the
 * kernels' tile layouts, in a single arena; exports the arena with a CUDA IPC
 * handle and writes a manifest (handle, then one line per tensor: name, offset,
 * bytes, type, rows, K, tile stride); then sleeps until killed, so kernel tests
 * map the arena (hold_open in hold.h) instead of reloading from NFS.
 *
 *   hold <gguf> <die> <b0> <b1> <manifest> [token_embd] [output] [cache=DIR] [split=d/n]
 *     token_embd / output also load those tensors; cache=DIR writes every device
 *     image (tiles or raw) to DIR/<name> so the C references read local disk, not NFS;
 *     split=d/n keeps only what die d of the n-die split owns (kern/prog.h split_of)
 *
 * Layout rules (what the kernels expect):
 *   Q2_0, Q4_K, Q5_K, Q3_K, Q6_K, IQ4_XS, Q4_0/Q5_0/Q8_0/IQ4_NL: repack_* tiles,
 *     rows = ne1 (output rows), K = ne0; 3-d expert tensors are repacked per
 *     expert, contiguously (expert stride = tile stride x rows).
 *   BF16 / F16 / F32: raw, as in the file.
 *   Merged rows (one GEMV phase instead of two or eight): per block
 *     hc_attn_dj / hc_ffn_dj = hc_*_down ++ hc_*_inject (324 x 10240 BF16),
 *     ffn_gate_inp_x = ffn_gate_inp ++ ffn_gate_inp_shexp (513 x 2560),
 *     ssm_ab = ssm_alpha ++ ssm_beta (96 x 2560) with ssm_ab.bias = ssm_dt.bias ++ 48
 *     zeros (so the one bias GEMV adds +0 to the beta rows). The sources are not
 *     loaded separately. emap (u32[512]) maps a global expert to the die's local
 *     expert index (0xffffffff: not here).
 *   Split rows (split=d/n): hc_*_dj rows [dj0, +djn); hc_*_up rows l 2560 + 32 cb for
 *     the die's column blocks; attn_qkv and ssm_conv1d rows q[klo..khi] k[klo..khi]
 *     v[h0..h0+hn); attn_gate rows 128 h; ssm_ab rows h and 48 + h (bias likewise);
 *     ssm_a entries h; ssm_out rows [y0, +yn); experts [e0, +en); everything else
 *     whole (replicated).
 */
#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include "quant.h"
#include "prog.h"
#include <cuda.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static volatile sig_atomic_t stop;
static void onsig(int s) { (void)s; stop = 1; }

/* tile stride for a quantised matrix row of K elements, 0 for raw types */
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
};
#define NMERGE ((int)(sizeof merges / sizeof merges[0]))

/* the part of a block tensor name after "blk.N." */
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
static int wanted(const char *name, int b0, int b1, int extra_embd, int extra_out) {
	if (strstr(name, ".indexer.")) return 0;   /* the sparse-attention indexer is not used below 2048 tokens */
	if (!strncmp(name, "blk.", 4)) { int b = atoi(name + 4); return b >= b0 && b < b1 && !merged_source(name); }
	if (!strncmp(name, "token_embd", 10)) return extra_embd;
	if (!strncmp(name, "output", 6)) return extra_out;
	return 0;
}
/* NULL when the block has no such tensor (attention blocks have no ssm_*) */
static const gguf_tensor *blk_tensor(const gguf_file *g, int b, const char *suf) {
	char name[128]; snprintf(name, sizeof name, "blk.%d.%s", b, suf);
	const gguf_tensor *t = gguf_find(g, name);
	return t && t->data ? t : NULL;
}

/* ---- the split: which rows (or experts) of a tensor this die keeps ---- */
typedef struct { int r0, n; } range_t;
typedef struct { range_t r[4]; int nr; int rows_total; int experts; } sel_t;   /* rows_total: rows the ranges index */
static int split_on;
static split_t SP;
static void sel_whole(sel_t *s, int rows, int experts) { s->nr = 1; s->r[0].r0 = 0; s->r[0].n = experts > 1 ? experts : rows; s->rows_total = rows; s->experts = experts; }
static void select_rows(const char *name, int rows, int experts, sel_t *s) {
	sel_whole(s, rows, experts);
	if (!split_on) return;
	const split_t *p = &SP;
	/* the head: output.weight rows, output_hc_down rows, output_hc_up the 4 lanes of the z columns; token_embd whole */
	if (!strcmp(name, "output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->o0, p->on }; return; }
	if (!strcmp(name, "output_hc_down.weight")) { s->nr = 1; s->r[0] = (range_t){ p->hd0, p->hdn }; return; }
	if (!strcmp(name, "output_hc_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; return; }
	const char *x = suffix(name);
	if (!x) return;
	/* the shared expert runs on one die per block */
	if (strstr(x, "_shexp.weight") && !p->shexp) { s->nr = 1; s->r[0] = (range_t){ 0, 0 }; return; }
	if (!strcmp(x, "hc_attn_dj.weight") || !strcmp(x, "hc_ffn_dj.weight")) { s->nr = 1; s->r[0] = (range_t){ p->dj0, p->djn }; }
	else if (!strcmp(x, "hc_attn_up.weight") || !strcmp(x, "hc_ffn_up.weight")) { s->nr = 4; for (int l = 0; l < 4; l++) s->r[l] = (range_t){ 2560 * l + 32 * p->cb0, 32 * p->cbn }; }
	else if (!strcmp(x, "attn_qkv.weight") || !strcmp(x, "ssm_conv1d.weight")) {
		s->nr = 3; s->r[0] = (range_t){ 128 * p->klo, 128 * p->nk }; s->r[1] = (range_t){ 2048 + 128 * p->klo, 128 * p->nk }; s->r[2] = (range_t){ 4096 + 128 * p->h0, 128 * p->hn };
	}
	else if (!strcmp(x, "attn_gate.weight")) { s->nr = 1; s->r[0] = (range_t){ 128 * p->h0, 128 * p->hn }; }
	else if (!strcmp(x, "ssm_ab.weight") || !strcmp(x, "ssm_ab.bias")) { s->nr = 2; s->r[0] = (range_t){ p->h0, p->hn }; s->r[1] = (range_t){ 48 + p->h0, p->hn }; s->rows_total = 96; }
	else if (!strcmp(x, "ssm_a")) { s->nr = 1; s->r[0] = (range_t){ p->h0, p->hn }; s->rows_total = 48; }
	else if (!strcmp(x, "ssm_out.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	/* v1 of the attention block: die 0 runs the attention proper, the output projection is row-split */
	else if (!strcmp(x, "attn_q.weight") || !strcmp(x, "attn_k.weight") || !strcmp(x, "attn_v.weight")) { s->nr = 1; s->r[0] = (range_t){ 0, p->d == 0 ? rows : 0 }; }
	else if (!strcmp(x, "attn_output.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "ple_key.weight")) { s->nr = 1; s->r[0] = (range_t){ p->pk0, p->pkn }; }
	else if (!strcmp(x, "ple_value.weight")) { s->nr = 1; s->r[0] = (range_t){ p->y0, p->yn }; }
	else if (!strcmp(x, "ffn_gate_exps.weight") || !strcmp(x, "ffn_up_exps.weight") || !strcmp(x, "ffn_down_exps.weight")) { s->nr = 1; s->r[0] = (range_t){ p->e0, p->en }; }
}

static FILE *m; static const char *cache; static CUdeviceptr arena; static size_t off;
static size_t cap; static uint8_t *buf;
static void need(size_t bytes) { if (bytes > cap) { free(buf); cap = bytes; buf = malloc(cap); } }
static void emit(const char *name, const uint8_t *img, size_t bytes, uint32_t type, int rows, int K, int st, int experts) {
	off = (off + 255) & ~(size_t)255;
	if (bytes) CK(cuMemcpyHtoD(arena + off, img, bytes));
	if (cache) {
		char fn[512]; snprintf(fn, sizeof fn, "%s/%s", cache, name);
		FILE *cf = fopen(fn, "wb");
		if (!cf || fwrite(img, 1, bytes, cf) != bytes) { perror(fn); exit(1); }
		fclose(cf);
	}
	fprintf(m, "%s %zu %zu %u %d %d %d %d\n", name, off, bytes, type, rows, K, st, experts);
	fprintf(stderr, "  %-40s %8.1f MB  %s%s\n", name, bytes / 1e6, gguf_type_name(type), st ? " tiles" : "");
	off += bytes;
}
/* bytes the selection occupies on the device (the pass-1 size) */
static size_t sel_bytes(const sel_t *s, size_t nbytes, int rows, int st) {
	size_t n = 0; for (int i = 0; i < s->nr; i++) n += s->r[i].n;
	if (s->experts > 1) return st ? (size_t)st * rows * n : nbytes / s->experts * n;
	size_t rowb = st ? (size_t)st : nbytes / s->rows_total;
	return rowb * n;
}
/* repack / copy the selected rows (or experts) of a raw source image into buf and emit it */
static void emit_sel(const char *name, uint32_t type, const uint8_t *src, size_t nbytes, int rows, int K, int experts, const sel_t *s) {
	int st = strncmp(name, "token_embd", 10) ? tile_stride(type, K) : 0;   /* embrow reads raw Q3_K rows */
	size_t bytes = sel_bytes(s, nbytes, rows, st), o = 0;
	need(bytes);
	int nsel = 0;
	if (s->experts > 1) {
		size_t per = nbytes / experts, dper = st ? (size_t)st * rows : per;
		for (int i = 0; i < s->nr; i++) for (int e = s->r[i].r0; e < s->r[i].r0 + s->r[i].n; e++, nsel++) {
			if (st) repack(type, src + per * e, rows, K, buf + o); else memcpy(buf + o, src + per * e, per);
			o += dper;
		}
		emit(name, buf, bytes, type, rows, K, st, nsel);
		return;
	}
	size_t srow = nbytes / s->rows_total;
	for (int i = 0; i < s->nr; i++) {
		int r0 = s->r[i].r0, n = s->r[i].n;
		if (st) { repack(type, src + srow * r0, n, K, buf + o); o += (size_t)st * n; }
		else { memcpy(buf + o, src + srow * r0, srow * n); o += srow * n; }
		nsel += n;
	}
	emit(name, buf, bytes, type, nsel, K, st, 1);
}

int main(int argc, char **argv) {
	if (argc < 6) { fprintf(stderr, "usage: hold <gguf> <die> <b0> <b1> <manifest> [token_embd] [output] [cache=DIR] [split=d/n]\n"); return 2; }
	const char *path = argv[1], *mpath = argv[5];
	int die = atoi(argv[2]), b0 = atoi(argv[3]), b1 = atoi(argv[4]), xe = 0, xo = 0, sd = 0, sn = 1;
	for (int i = 6; i < argc; i++) {
		if (!strcmp(argv[i], "token_embd")) xe = 1;
		if (!strcmp(argv[i], "output")) xo = 1;
		if (!strncmp(argv[i], "cache=", 6)) cache = argv[i] + 6;
		if (!strncmp(argv[i], "split=", 6)) { if (sscanf(argv[i] + 6, "%d/%d", &sd, &sn) != 2 || sn < 1 || sn > 6 || sd < 0 || sd >= sn) { fprintf(stderr, "bad split %s\n", argv[i]); return 2; } split_on = 1; }
	}
	gguf_file g;
	if (gguf_open(&g, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
	/* pass 1: sizes (per block, the split depends on the block for the shared expert only) */
	size_t total = 0; int count = 0;
	for (int i = 0; i < g.ntensors; i++) {
		const gguf_tensor *t = &g.t[i];
		if (!wanted(t->name, b0, b1, xe, xo)) continue;
		int K = (int)t->dims[0], rows = t->nd > 1 ? (int)t->dims[1] : 1, experts = t->nd > 2 ? (int)t->dims[2] : 1;
		int st = strncmp(t->name, "token_embd", 10) ? tile_stride(t->type, K) : 0;
		sel_t s; if (split_on) split_of(&SP, sd, sn, strncmp(t->name, "blk.", 4) ? 0 : atoi(t->name + 4)); select_rows(t->name, rows, experts, &s);
		total = (total + 255) & ~(size_t)255;
		total += sel_bytes(&s, t->nbytes, rows, st); count++;
	}
	for (int b = b0; b < b1; b++) {
		if (split_on) split_of(&SP, sd, sn, b);
		for (int i = 0; i < NMERGE; i++) {
			const gguf_tensor *a = blk_tensor(&g, b, merges[i].a), *c = blk_tensor(&g, b, merges[i].b);
			if (!a || !c) continue;
			size_t nb = merges[i].zero_b ? a->nbytes : c->nbytes, bytes = a->nbytes + nb;
			int rows = a->nd > 1 ? (int)(bytes / (a->nbytes / a->dims[1])) : (int)(bytes / 4);
			char name[128]; snprintf(name, sizeof name, "blk.%d.%s", b, merges[i].out);
			sel_t s; select_rows(name, rows, 1, &s);
			total = (total + 255) & ~(size_t)255;
			total += sel_bytes(&s, bytes, rows, 0); count++;
		}
		total = (total + 255) & ~(size_t)255; total += 512 * 4; count++;   /* emap */
	}
	/* slack after the last tensor: a GEMV may still issue (predicated-off or
	 * otherwise harmless) loads for the rows of its last CTA iteration beyond the
	 * matrix end; keep the arena mapped there */
	total = ((total + 255) & ~(size_t)255) + (4u << 20);
	fprintf(stderr, "hold: %d tensors, %.2f GB on die %d%s\n", count, total / 1e9, die, split_on ? " (split)" : "");
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx;
	CK(cuDeviceGet(&dev, die));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
	CK(cuMemAlloc(&arena, total));
	CUipcMemHandle h;
	CK(cuIpcGetMemHandle(&h, arena));
	m = fopen(mpath, "w");
	if (!m) { perror(mpath); return 1; }
	fprintf(m, "arena %zu die %d handle ", total, die);
	for (size_t i = 0; i < sizeof h; i++) fprintf(m, "%02x", ((unsigned char *)&h)[i]);
	fprintf(m, "\n");
	/* pass 2: load */
	for (int i = 0; i < g.ntensors; i++) {
		const gguf_tensor *t = &g.t[i];
		if (!wanted(t->name, b0, b1, xe, xo)) continue;
		int K = (int)t->dims[0], rows = t->nd > 1 ? (int)t->dims[1] : 1, experts = t->nd > 2 ? (int)t->dims[2] : 1;
		if (!t->data) { fprintf(stderr, "%s: not mapped\n", t->name); return 1; }
		sel_t s; if (split_on) split_of(&SP, sd, sn, strncmp(t->name, "blk.", 4) ? 0 : atoi(t->name + 4)); select_rows(t->name, rows, experts, &s);
		emit_sel(t->name, t->type, t->data, t->nbytes, rows, K, experts, &s);
	}
	uint8_t *mb = NULL; size_t mcap = 0;
	for (int b = b0; b < b1; b++) {
		if (split_on) split_of(&SP, sd, sn, b);
		for (int i = 0; i < NMERGE; i++) {
			const gguf_tensor *a = blk_tensor(&g, b, merges[i].a), *c = blk_tensor(&g, b, merges[i].b);
			if (!a || !c) continue;
			size_t nb = merges[i].zero_b ? a->nbytes : c->nbytes, bytes = a->nbytes + nb;
			if (a->type != c->type || a->dims[0] != c->dims[0]) { fprintf(stderr, "blk.%d.%s: cannot merge %s and %s\n", b, merges[i].out, merges[i].a, merges[i].b); return 1; }
			if (bytes > mcap) { free(mb); mcap = bytes; mb = malloc(mcap); }
			memcpy(mb, a->data, a->nbytes);
			if (merges[i].zero_b) memset(mb + a->nbytes, 0, nb); else memcpy(mb + a->nbytes, c->data, nb);
			int K = (int)a->dims[0], rows = a->nd > 1 ? (int)(bytes / (a->nbytes / a->dims[1])) : (int)(bytes / 4);
			char name[128]; snprintf(name, sizeof name, "blk.%d.%s", b, merges[i].out);
			sel_t s; select_rows(name, rows, 1, &s);
			emit_sel(name, a->type, mb, bytes, rows, K, 1, &s);
		}
		/* emap: global expert -> local index */
		uint32_t emap[512];
		int e0 = split_on ? SP.e0 : 0, en = split_on ? SP.en : 512;
		for (int e = 0; e < 512; e++) emap[e] = (e >= e0 && e < e0 + en) ? (uint32_t)(e - e0) : 0xffffffffu;
		char name[128]; snprintf(name, sizeof name, "blk.%d.emap", b);
		emit(name, (const uint8_t *)emap, sizeof emap, GG_F32, 512, 1, 0, 1);
	}
	fclose(m);
	gguf_close(&g);
	fprintf(stderr, "hold: ready (%s); holding until SIGTERM\n", mpath);
	signal(SIGTERM, onsig); signal(SIGINT, onsig);
	while (!stop) pause();
	CK(cuMemFree(arena));
	return 0;
}
