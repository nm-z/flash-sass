/* model_test: whole tokens through the split persistent programs — the embedding, a
 * run of blocks [b0, b1) (DeltaNet, block 1's PLE, attention) and the head — on n dies
 * against the C twins, token by token (one launch per die per token), with every
 * block's state carried. Greedy: the head's token feeds the next embedding once the
 * prompt is used up.
 *
 *   model_test <fullcache> <b0> <b1> <T> <flags> <manifest0> [manifest1 ...]
 *     flags: a string of  e (embed first)  h (head last)  or  -  for neither
 *     fullcache: a holder's tile cache with every tensor the twin needs (blocks, head,
 *       embedding); manifest_i: die i's running holder (hold ... <b0> <b1> ... split=i/n)
 * Env: MODEL_NPH caps every die's program; MODEL_PROF prints die 0's per-op times;
 * MODEL_NOTWIN skips the twin (timing); MODEL_IDS=file reads prompt ids (u32 text).
 * MODEL_ATTN=dense (default) or qsa (production INT8 chain). FLASH_NBLK defaults
 * to 512; FLASH_KV_HOST defaults to 1. QSA validation requires all teacher-forced
 * ids. MODEL_INJECT=layer|logits|selection must make comparison fail; selection
 * needs T >= 4*(FLASH_NBLK+1). Per-token mode checks every token/layer, loop mode
 * every prediction and the final token layer stages. */
#define _POSIX_C_SOURCE 200809L
#include "block.h"
#include "attn.h"
#include "ple.h"
#include "hold.h"
#include "prog.h"
#include "split.h"
#include "gguf.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
static uint32_t rs = 909;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }
static float h2f(uint16_t h) {
	uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
	if (e == 0) { if (!m) u = s; else { int sh = 0; while (!(m & 0x400)) { m <<= 1; sh++; } m &= 0x3ff; u = s | ((uint32_t)(113 - sh) << 23) | (m << 13); } }
	else if (e == 31) u = s | 0x7f800000u | (m << 13);
	else u = s | ((e + 112) << 23) | (m << 13);
	float f; memcpy(&f, &u, 4); return f;
}

#define MAXD 6
#define MAXB 48
static int nd;
static const char *fullcache;
static CUcontext ctx[MAXD];
static hold_arena ar[MAXD];
static char tname[128];
static const char *tn(int blk, const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static void *hostimg_raw(const char *name) {
    char fn[512]; snprintf(fn, sizeof fn, "%s/%s", fullcache, name);
    int fd = open(fn, O_RDONLY); struct stat st;
    if (fd < 0) { perror(fn); exit(1); }
    if (fstat(fd, &st) || st.st_size <= 0) { perror(fn); close(fd); exit(1); }
    void *p = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (p == MAP_FAILED) { perror(fn); exit(1); }
    return p;
}
static void *hostimg(int blk, const char *suffix) { return hostimg_raw(tn(blk, suffix)); }
static CUdeviceptr dptr_raw(int d, const char *name) { CUdeviceptr p = hold_ptr(&ar[d], name); if (!p) { fprintf(stderr, "die %d: no %s\n", d, name); exit(1); } return p; }
static CUdeviceptr dptr(int d, int blk, const char *suffix) { return dptr_raw(d, tn(blk, suffix)); }
static uint32_t ttype(int blk, const char *suffix) { const hold_tensor *t = hold_find(&ar[0], tn(blk, suffix)); if (!t) { fprintf(stderr, "no %s\n", tname); exit(1); } return t->type; }
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }

static int cmpbuf(const char *what, const void *ref, CUdeviceptr dev, size_t bytes, int elemsz) {
	void *got = malloc(bytes);
	CK(cuMemcpyDtoH(got, dev, bytes));
	size_t n = bytes / elemsz; int bad = 0; size_t first = 0;
	for (size_t i = 0; i < n; i++) if (memcmp((const char *)ref + i * elemsz, (const char *)got + i * elemsz, elemsz)) { if (!bad) first = i; bad++; }
	if (bad) {
		printf("  %-10s MISMATCH %d of %zu (first at %zu", what, bad, n, first);
		if (elemsz == 4) printf(": gpu %.9g ref %.9g", ((float *)got)[first], ((const float *)ref)[first]);
		printf(")\n");
	}
	free(got);
	return bad;
}

#include "model_qsa_test.h"
#include "model_capture.h"

/* the twin's per-block state (dn_hbufs_t carries one block's) */
typedef struct { float *dn_state, *conv_state, *ple_state, *kc, *vc; } hstate_t;

int main(int argc, char **argv) {
	if (argc < 7) { fprintf(stderr, "usage: model_test <fullcache> <b0> <b1> <T> <flags: e h -> <manifest0> [manifest1 ...]\n"); return 2; }
	fullcache = argv[1]; int b0 = atoi(argv[2]), b1 = atoi(argv[3]), T = atoi(argv[4]); const char *flags = argv[5]; nd = argc - 6; float eps = 1e-6f;
	int do_embed = strchr(flags, 'e') != NULL, do_head = strchr(flags, 'h') != NULL, do_loop = strchr(flags, 'l') != NULL;   /* l: one launch for all T tokens */
	if (nd < 1 || nd > MAXD || b0 < 0 || b1 > MAXB || b0 >= b1 || T < 1) { fprintf(stderr, "bad arguments\n"); return 2; }
	int notwin = getenv("MODEL_NOTWIN") != NULL;
	const char *attn_mode = getenv("MODEL_ATTN") ? getenv("MODEL_ATTN") : "dense";
	if (strcmp(attn_mode, "dense") && strcmp(attn_mode, "qsa")) {
		fprintf(stderr, "MODEL_ATTN must be dense or qsa\n"); return 2;
	}
	model_qsa = !strcmp(attn_mode, "qsa");
	model_host = getenv("FLASH_KV_HOST") ? atoi(getenv("FLASH_KV_HOST")) != 0 : 1;
	model_budget = getenv("FLASH_NBLK") ? (unsigned)atoi(getenv("FLASH_NBLK")) : QSA_NBLK;
	if (model_qsa && (model_budget < 1 || model_budget > 4096)) {
		fprintf(stderr, "FLASH_NBLK must be in [1,4096]\n"); return 2;
	}
	if (model_qsa && getenv("MODEL_NPH")) {
		fprintf(stderr, "MODEL_NPH cannot truncate a production QSA validation program\n"); return 2;
	}
	int model_attention_layers = 0;
	for (int b = b0; b < b1; b++) if (BLK_IS_ATTN(b)) model_attention_layers++;
	if (model_qsa && !model_attention_layers) {
		fprintf(stderr, "MODEL_ATTN=qsa needs at least one attention block\n"); return 2;
	}
	if (getenv("MODEL_INJECT")) {
		const char *inj = getenv("MODEL_INJECT");
		if (!model_qsa || notwin ||
		    (strcmp(inj, "layer") && strcmp(inj, "logits") && strcmp(inj, "selection")) ||
		    (!strcmp(inj, "logits") && !do_head) ||
		    (!strcmp(inj, "selection") && (unsigned)T < 4u * (model_budget + 1u))) {
			fprintf(stderr, "MODEL_INJECT requires QSA validation and a reachable requested comparison\n"); return 2;
		}
	}
	model_capture_check(b0,b1,do_embed,do_head,do_loop);
	model_maxp = qsa_cache_positions((unsigned)T);
	unsigned compact_positions = qsa_compact_positions(model_budget);
	model_kcf = ATTN_KC_FLOATS(compact_positions);
	model_vcf = (size_t)compact_positions * 512;
	printf("attention mode: %s%s\n", attn_mode, model_qsa ? " (production INT8 QSA; instruction-order twin)" : " (FP32 dense baseline)");
	if (model_qsa) printf("FLASH_NBLK=%u FLASH_KV_HOST=%d; cache positions=%u compact positions=%u\n", model_budget, model_host, model_maxp, compact_positions);
	unsigned npos = ((unsigned)T + 8192 + 64 + 31) / 32 * 32;
	size_t kcf = ATTN_KC_FLOATS(npos), vcf = (size_t)npos * 512;
	CK(cuInit(0));
	CUmodule mod[MAXD]; CUfunction fn[MAXD]; int dev_of[MAXD];
	for (int d = 0; d < nd; d++) {
		FILE *f = fopen(argv[6 + d], "r"); size_t bytes; int die;
		if (!f || fscanf(f, "arena %zu die %d", &bytes, &die) != 2) { fprintf(stderr, "bad manifest %s\n", argv[6 + d]); return 1; }
		fclose(f);
		dev_of[d] = die;
		CUdevice dev; if (hold_ordinal(die) < 0) { fprintf(stderr, "die %d is not in CUDA_VISIBLE_DEVICES\n", die); return 1; } CK(cuDeviceGet(&dev, hold_ordinal(die)));
		CK(cuCtxCreate_v2(&ctx[d], CU_CTX_SCHED_SPIN, dev));
		if (hold_open(&ar[d], argv[6 + d])) return 1;
		CK(cuModuleLoad(&mod[d], "die.cubin")); CK(cuModuleGetFunction(&fn[d], mod[d], "die"));
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); for (int e = 0; e < nd; e++) if (e != d) CK(cuCtxEnablePeerAccess(ctx[e], 0)); }
	/* prompt ids: MODEL_IDS or random; the PLE's synthetic table and constants */
	uint32_t *ids = calloc((size_t)T + 2, 4); int nprompt = T;
	if (getenv("MODEL_IDS")) {
		FILE *f = fopen(getenv("MODEL_IDS"), "r"); if (!f) { perror(getenv("MODEL_IDS")); return 1; }
		nprompt = 0; while (nprompt < T && fscanf(f, "%u", &ids[nprompt]) == 1) nprompt++;
		fclose(f);
	} else for (int i = 0; i < T; i++) { ids[i] = (uint32_t)((rs >> 8) % 248000u); frand(); }
	if (model_qsa && !notwin && nprompt != T) {
		fprintf(stderr, "QSA comparison needs identical teacher-forced IDs at all %d positions; MODEL_IDS supplied %d. Use MODEL_NOTWIN for separate free-running smoke/timing.\n", T, nprompt); return 2;
	}
	for (int i = 0; i < nprompt; i++) if (ids[i] >= 248320u) {
		fprintf(stderr, "MODEL_IDS token %d outside vocabulary: %u\n", i, ids[i]); return 2;
	}
	if (model_capture_enabled() && nprompt != T) { fprintf(stderr,"MODEL_CAPTURE needs exactly T teacher-forced IDs\n"); return 2; }
	const size_t PLE_R = 1u << 20; uint8_t *ptab = NULL, *ctab = NULL; unsigned pcompact = 0; uint8_t ppc[160]; uint64_t pmult[3], poff[16], pvoc[16]; uint32_t pabsent = 248044, pend = 248046;
/* the 16 real rows of position t -> the compact table slot the GPU reads (plerows compact mode) */
#define PLE_STAGE(t) do { uint64_t rows_[16]; ple_rows_c(ids, (size_t)(t), pabsent, &pend, 1, pmult, poff, pvoc, rows_); for (int h_ = 0; h_ < 16; h_++) memcpy(ctab + ((size_t)(t) * 16 + h_) * 90, ptab + rows_[h_] * 90, 90); } while (0)
	int has_ple = b0 <= 1 && 1 < b1;
	if (has_ple) {
		memcpy(pmult, ple_multipliers, sizeof pmult);
		if (getenv("MODEL_PLE")) {
			/* the real table: shard 2's tensor data (320,001,536 rows x 90 B) as a file (a tmpfs copy).
			   CUDA 11.4 refuses to pin a file mapping, so the twin reads the mapping with the real hash
			   while the GPU gets a compact table: the 16 rows of every position (teacher-forced: all T
			   ids are known up front), staged in pinned memory and indexed 16 pos + h (plerows' compact
			   mode). Free-running decode with the real table needs a long-lived pinned copy (later). */
			if (do_loop && nprompt < T) { fprintf(stderr, "MODEL_PLE in one-launch mode needs every id up front (MODEL_IDS with %d ids): the rows are staged per position\n", T); return 1; }
			int fd = open(getenv("MODEL_PLE"), O_RDONLY); if (fd < 0) { perror(getenv("MODEL_PLE")); return 1; }
			struct stat st; if (fstat(fd, &st)) { perror("fstat"); return 1; }
			size_t sz = ((size_t)st.st_size + 4095) & ~(size_t)4095;   /* whole pages (the last row ends mid-page) */
			off_t poff_ = getenv("MODEL_PLE_OFF") ? (off_t)atoll(getenv("MODEL_PLE_OFF")) : 0;   /* 192: shard 2 itself (its data section); 0: an extracted copy */
			sz = (sz - (size_t)poff_ + 4095) & ~(size_t)4095;
			ptab = mmap(NULL, sz, PROT_READ, MAP_SHARED, fd, (off_t)(poff_ & ~(off_t)4095)); if (ptab == MAP_FAILED) { perror("mmap ple"); return 1; }
			ptab += poff_ & 4095;
			memcpy(poff, ple_head_offsets, sizeof poff); memcpy(pvoc, ple_head_vocab, sizeof pvoc);
			CK(cuMemHostAlloc((void **)&ctab, (size_t)T * 16 * 90 + 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
			for (int t = 0; t < nprompt; t++) PLE_STAGE(t);   /* generated positions are staged per token (per-token mode) */
			pcompact = 1;
			printf("PLE table %s mapped for the twin; %d prompt positions x 16 rows staged for the GPU\n", getenv("MODEL_PLE"), nprompt);
		} else {
			CK(cuMemHostAlloc((void **)&ptab, PLE_R * 90, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
			for (size_t r = 0; r < PLE_R; r++)
				for (int k = 0; k < 5; k++) {
					uint16_t dd = (uint16_t)(0x3000 + (rs >> 16) % 0x1000); frand();
					memcpy(ptab + r * 90 + 18 * k, &dd, 2);
					for (int j = 2; j < 18; j++) { ptab[r * 90 + 18 * k + j] = (uint8_t)(rs >> 24); frand(); }
				}
			for (int h = 0; h < 16; h++) { poff[h] = (uint64_t)h * 65000; pvoc[h] = 60000 + 13 * h; }
		}
		memset(ppc, 0, sizeof ppc); memcpy(ppc, pmult, 24);
		for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)poff[h], v32 = (uint32_t)pvoc[h]; memcpy(ppc + 24 + 4 * h, &o32, 4); memcpy(ppc + 88 + 4 * h, &v32, 4); }
		memcpy(ppc + 152, &pabsent, 4); memcpy(ppc + 156, &pend, 4);
	}
	float *rcs = malloc(4 * 32 * T), *rsn = malloc(4 * 32 * T);
	for (int t = 0; t < T; t++) rope_table_gpu((unsigned)t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	/* ---- twin weights and states ---- */
	dn_hweights_t *hwb = calloc(MAXB, sizeof *hwb); hstate_t hs[MAXB]; memset(hs, 0, sizeof hs);
	dn_hbufs_t *hb = calloc(1, sizeof *hb);
	hb->logits = malloc(4 * 248320); hb->ids = ids;
	if (!notwin) {
		for (int b = b0; b < b1; b++) {
			dn_hweights_t *w = &hwb[b]; int is_attn = BLK_IS_ATTN(b);
			if (model_qsa) model_href[b] = calloc(10240, 4);
			w->hc_attn_norm = hostimg(b, "hc_attn_norm.weight"); w->hc_attn_dj = hostimg(b, "hc_attn_dj.weight"); w->hc_attn_up = hostimg(b, "hc_attn_up.weight");
			w->hc_ffn_norm = hostimg(b, "hc_ffn_norm.weight"); w->hc_ffn_dj = hostimg(b, "hc_ffn_dj.weight"); w->hc_ffn_up = hostimg(b, "hc_ffn_up.weight");
			if (!is_attn) {
				w->attn_qkv = hostimg(b, "attn_qkv.weight"); w->attn_qkv_type = ttype(b, "attn_qkv.weight");
				w->attn_gate = hostimg(b, "attn_gate.weight"); w->attn_gate_type = ttype(b, "attn_gate.weight");
				w->ssm_out = hostimg(b, "ssm_out.weight"); w->ssm_out_type = ttype(b, "ssm_out.weight");
				w->ssm_ab = hostimg(b, "ssm_ab.weight"); w->ssm_ab_bias = hostimg(b, "ssm_ab.bias"); w->ssm_conv1d = hostimg(b, "ssm_conv1d.weight");
				w->ssm_a = hostimg(b, "ssm_a"); w->ssm_norm = hostimg(b, "ssm_norm.weight");
				hs[b].dn_state = calloc(48 * 16384, 4); hs[b].conv_state = calloc(4 * 10240, 4);
			} else {
				w->attn_q = hostimg(b, "attn_q.weight"); w->attn_q_type = ttype(b, "attn_q.weight");
				w->attn_k = hostimg(b, "attn_k.weight"); w->attn_k_type = ttype(b, "attn_k.weight");
				w->attn_v = hostimg(b, "attn_v.weight"); w->attn_v_type = ttype(b, "attn_v.weight");
				w->attn_output = hostimg(b, "attn_output.weight"); w->attn_output_type = ttype(b, "attn_output.weight");
				w->attn_q_norm = hostimg(b, "attn_q_norm.weight"); w->attn_k_norm = hostimg(b, "attn_k_norm.weight");
				if (model_qsa) {
					w->indexer_x = hostimg(b, "indexer_x.weight");
					w->indexer_qn = hostimg(b, "indexer.q_norm.weight");
					w->indexer_kn = hostimg(b, "indexer.k_norm.weight");
					model_qsa_host_alloc(b);
				} else {
					hs[b].kc = calloc(kcf, 4); hs[b].vc = calloc(vcf, 4);
				}
			}
			w->ffn_gate_inp_x = hostimg(b, "ffn_gate_inp_x.weight");
			w->ffn_gate_exps = hostimg(b, "ffn_gate_exps.weight"); w->ffn_up_exps = hostimg(b, "ffn_up_exps.weight"); w->ffn_down_exps = hostimg(b, "ffn_down_exps.weight");
			w->ffn_gate_shexp = hostimg(b, "ffn_gate_shexp.weight"); w->ffn_gate_shexp_type = ttype(b, "ffn_gate_shexp.weight");
			w->ffn_up_shexp = hostimg(b, "ffn_up_shexp.weight"); w->ffn_up_shexp_type = ttype(b, "ffn_up_shexp.weight");
			w->ffn_down_shexp = hostimg(b, "ffn_down_shexp.weight"); w->ffn_down_shexp_type = ttype(b, "ffn_down_shexp.weight");
			if (b == 1) {
				w->ple_key = hostimg(b, "ple_key.weight"); w->ple_value = hostimg(b, "ple_value.weight");
				w->ple_norm_key = hostimg(b, "ple_norm_key.weight"); w->ple_norm_query = hostimg(b, "ple_norm_query.weight"); w->ple_norm_conv = hostimg(b, "ple_norm_conv.weight");
				const uint16_t *cw = hostimg(b, "ple_conv1d.weight"); float *pc = malloc(4 * 10240 * 4);
				for (int i = 0; i < 10240 * 4; i++) pc[i] = h2f(cw[i]);
				w->ple_conv_w = pc; w->ple_tab = ptab; memcpy(w->ple_mult, pmult, sizeof pmult); memcpy(w->ple_off, poff, sizeof poff); memcpy(w->ple_voc, pvoc, sizeof pvoc);
				w->ple_absent = pabsent; w->ple_end = pend;
				hs[b].ple_state = calloc(10 * 10240, 4);
			}
		}
		if (do_head) { hwb[0].output_hc_norm = hostimg_raw("output_hc_norm.weight"); hwb[0].output_hc_down = hostimg_raw("output_hc_down.weight"); hwb[0].output_hc_up = hostimg_raw("output_hc_up.weight"); hwb[0].output = hostimg_raw("output.weight"); }
		if (do_embed) hwb[0].token_embd = hostimg_raw("token_embd.weight");
	}
	/* ---- the dies ---- */
	dn_weights_t (*dw)[MAXB] = calloc(MAXD, sizeof *dw); dn_bufs_t db[MAXD]; xchg_t xc[MAXD]; split_t sp[MAXD][MAXB];
	CUdeviceptr dPROG[MAXD], dCTL[MAXD], dst_dn[MAXD][MAXB], dst_conv[MAXD][MAXB], dst_kc[MAXD][MAXB], dst_vc[MAXD][MAXB], dst_ple[MAXD];
	memset(dst_dn, 0, sizeof dst_dn); memset(dst_conv, 0, sizeof dst_conv);
	memset(dst_kc, 0, sizeof dst_kc); memset(dst_vc, 0, sizeof dst_vc); memset(dst_ple, 0, sizeof dst_ple);
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		split_t s0; split_of(&s0, d, nd, 0);
		dn_bufs_t *b = &db[d]; memset(b, 0, sizeof *b);
		unsigned nlocmax = 128u * (2u * (unsigned)s0.nk + (unsigned)s0.hn)   /* q, k of the die's key heads, v of its value heads */, cols = 32u * (unsigned)s0.cbn, hrows = 128u * (unsigned)s0.hn;
		b->S = dalloc(4 * 10240); b->S2 = dalloc(4 * 10240); b->xh = dalloc(4 * 10240); b->t = dalloc(4 * 324);
		b->ldown = dalloc(4 * 324); b->lup = dalloc(4 * 4 * cols); b->lz = dalloc(4 * cols);
		b->zq = dalloc(2 * 2560); b->zs32 = dalloc(4 * 80); b->zs16 = dalloc(4 * 160); b->zas = dalloc(4 * 80);
		b->lqkv = dalloc(4 * nlocmax); b->lgate = dalloc(4 * hrows); b->la = dalloc(4 * 2u * (unsigned)s0.hn); b->lu = dalloc(4 * nlocmax); b->lo = dalloc(4 * hrows); b->lon = dalloc(4 * hrows);
		b->oq = dalloc(2 * 6144); b->os32 = dalloc(4 * 192); b->os16 = dalloc(4 * 384); b->oas = dalloc(4 * 192);
		b->ly = dalloc(4u * (unsigned)s0.yn); b->r = dalloc(4 * 513); b->idx = dalloc(64); b->w = dalloc(64); b->s = dalloc(16);
		b->lidx = dalloc(64); b->lw = dalloc(64); b->ne = dalloc(16); b->part = dalloc(4 * 2560);
		b->g = dalloc(4 * 7040); b->gu = dalloc(4 * 7040); b->hq = dalloc(2 * 7040); b->hs32 = dalloc(4 * 220); b->hs16 = dalloc(4 * 440); b->has = dalloc(4 * 220);
		b->out = dalloc(4 * 2560);
		b->qg = dalloc(4 * 12288); b->k = dalloc(4 * 512); b->v = dalloc(4 * 512); b->qo = dalloc(4 * 6144); b->aout = dalloc(4 * 6144);
		b->pm = dalloc(4 * 24 * 8); b->pd = dalloc(4 * 24 * 8); b->po = dalloc(4 * 24 * 8 * 256);
		b->rope_cs = dalloc(4 * 32 * T); b->rope_sn = dalloc(4 * 32 * T);
		CK(cuMemcpyHtoD(b->rope_cs, rcs, 4 * 32 * T)); CK(cuMemcpyHtoD(b->rope_sn, rsn, 4 * 32 * T));
		b->ids = dalloc(4 * ((size_t)T + 2)); CK(cuMemcpyHtoD(b->ids, ids, 4 * ((size_t)T + 2)));
		if (has_ple) {
			b->ple_pc = dalloc(160); CK(cuMemcpyHtoD(b->ple_pc, ppc, 160)); CK(cuMemHostGetDevicePointer(&b->ple_tab, pcompact ? ctab : ptab, 0)); b->ple_compact = pcompact;
			b->pg = dalloc(4 * 2560); b->pgq = dalloc(2 * 2560); b->pgs32 = dalloc(4 * 80); b->pgs16 = dalloc(4 * 160); b->pgas = dalloc(4 * 80);
			b->lk = dalloc(4u * (unsigned)s0.pkn); b->kn = dalloc(4 * 10240); b->qn = dalloc(4 * 10240); b->lv = dalloc(4u * (unsigned)s0.yn);
			b->pG = dalloc(4 * 10240); b->pgates = dalloc(16); b->px = dalloc(4 * 10240); b->prows = dalloc(64);
			dst_ple[d] = dalloc(4 * 10 * 10240);
		}
		b->hlo = dalloc(4u * (unsigned)s0.on); b->amo = dalloc(8); b->apart = dalloc(16 * 8); b->acnt = dalloc(4); b->tokout = dalloc(8); b->toks = dalloc(4 * ((size_t)T + 2));
		for (int bb = b0; bb < b1; bb++) {
			split_t *s = &sp[d][bb]; split_of(s, d, nd, bb);
			if (model_qsa) model_trace[d][bb] = dalloc(4 * 10240);
			dn_weights_t *w = &dw[d][bb]; int is_attn = BLK_IS_ATTN(bb);
			w->hc_attn_norm = dptr(d, bb, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, bb, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, bb, "hc_attn_up.weight");
			w->hc_ffn_norm = dptr(d, bb, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, bb, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, bb, "hc_ffn_up.weight");
			if (!is_attn) {
				w->attn_qkv = dptr(d, bb, "attn_qkv.weight"); w->attn_qkv_type = ttype(bb, "attn_qkv.weight");
				w->attn_gate = dptr(d, bb, "attn_gate.weight"); w->attn_gate_type = ttype(bb, "attn_gate.weight");
				w->ssm_out = dptr(d, bb, "ssm_out.weight"); w->ssm_out_type = ttype(bb, "ssm_out.weight");
				w->ssm_ab = dptr(d, bb, "ssm_ab.weight"); w->ssm_ab_bias = dptr(d, bb, "ssm_ab.bias");
				w->ssm_conv1d = dptr(d, bb, "ssm_conv1d.weight"); w->ssm_a = dptr(d, bb, "ssm_a"); w->ssm_norm = dptr(d, bb, "ssm_norm.weight");
				dst_dn[d][bb] = dalloc((size_t)s->hn * 16384 * 4); dst_conv[d][bb] = dalloc(4 * 4 * SPLIT_NLOC(s));
			} else {
				w->attn_q = dptr(d, bb, "attn_q.weight"); w->attn_q_type = ttype(bb, "attn_q.weight");
				w->attn_k = dptr(d, bb, "attn_k.weight"); w->attn_k_type = ttype(bb, "attn_k.weight");
				w->attn_v = dptr(d, bb, "attn_v.weight"); w->attn_v_type = ttype(bb, "attn_v.weight");
				w->attn_output = dptr(d, bb, "attn_output.weight"); w->attn_output_type = ttype(bb, "attn_output.weight");
				w->attn_q_norm = dptr(d, bb, "attn_q_norm.weight"); w->attn_k_norm = dptr(d, bb, "attn_k_norm.weight");
				if (model_qsa) {
					w->indexer_x = dptr(d, bb, "indexer_x.weight");
					w->indexer_qn = dptr(d, bb, "indexer.q_norm.weight");
					w->indexer_kn = dptr(d, bb, "indexer.k_norm.weight");
					if (attn_die_of(bb) < 0 || attn_die_of(bb) >= nd) {
						fprintf(stderr, "block %d attention owner %d outside %d test dies\n", bb, attn_die_of(bb), nd); return 2;
					}
					if (d == attn_die_of(bb)) model_qsa_dev_alloc(d, bb);
				} else {
					dst_kc[d][bb] = dalloc(d == attn_die() ? kcf * 4 : 256); dst_vc[d][bb] = dalloc(d == attn_die() ? vcf * 4 : 256);
				}
			}
			w->ffn_gate_inp_x = dptr(d, bb, "ffn_gate_inp_x.weight"); w->emap = dptr(d, bb, "emap");
			w->ffn_gate_exps = dptr(d, bb, "ffn_gate_exps.weight"); w->ffn_up_exps = dptr(d, bb, "ffn_up_exps.weight"); w->ffn_down_exps = dptr(d, bb, "ffn_down_exps.weight");
			w->ffn_gate_shexp = dptr(d, bb, "ffn_gate_shexp.weight"); w->ffn_gate_shexp_type = ttype(bb, "ffn_gate_shexp.weight");
			w->ffn_up_shexp = dptr(d, bb, "ffn_up_shexp.weight"); w->ffn_up_shexp_type = ttype(bb, "ffn_up_shexp.weight");
			w->ffn_down_shexp = dptr(d, bb, "ffn_down_shexp.weight"); w->ffn_down_shexp_type = ttype(bb, "ffn_down_shexp.weight");
			if (bb == 1) {
				w->ple_key = dptr(d, bb, "ple_key.weight"); w->ple_value = dptr(d, bb, "ple_value.weight");
				w->ple_norm_key = dptr(d, bb, "ple_norm_key.weight"); w->ple_norm_query = dptr(d, bb, "ple_norm_query.weight"); w->ple_norm_conv = dptr(d, bb, "ple_norm_conv.weight");
				w->ple_conv1d = dptr(d, bb, "ple_conv1d.weight");
			}
		}
		if (do_head) {
			dn_weights_t *w = &dw[d][0];
			w->output_hc_norm = dptr_raw(d, "output_hc_norm.weight"); w->output_hc_down = dptr_raw(d, "output_hc_down.weight"); w->output_hc_up = dptr_raw(d, "output_hc_up.weight"); w->output = dptr_raw(d, "output.weight");
			uint32_t ob[6]; for (int e = 0; e < 6; e++) ob[e] = split_bound(248320u, e, nd);
			w->obases = dalloc(24); CK(cuMemcpyHtoD(w->obases, ob, 24));
		}
		if (do_embed) dw[d][0].token_embd = model_qsa ? hold_ptr(&ar[d], "token_embd.weight") : dptr_raw(d, "token_embd.weight");
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * XSLOT); x->flags = dalloc(512); x->xtab = dalloc(128);   /* flags: u32[6 dies][16 CTAs] */
		uint32_t fl[96]; for (int e = 0; e < 96; e++) fl[e] = e / 16 < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		dPROG[d] = dalloc(128 * 4096); dCTL[d] = dalloc(0x100 + 4 * 4096);
		x->xgen = dCTL[d] + 0x20;   /* the exchange count lives in ctl[8] between launches */
	}
	if (model_qsa && do_embed) {
		for (int d = 0; d < nd; d++) if (dw[d][0].token_embd) { model_embed_die = d; break; }
		if (model_embed_die < 0) { fprintf(stderr, "no die holds token_embd.weight\n"); return 1; }
		printf("production embedding owner: die %d\n", model_embed_die);
	}
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { tab[d] = (uint64_t)xc[d < nd ? d : 0].inbox; tab[6 + d] = (uint64_t)xc[d < nd ? d : 0].flags; }   /* absent dies alias die 0: duplicate writes of the same bytes */
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab));
		/* the framework's exchange header in ctl: [3] me, [4] ndies, [16 + 2e] = &flags_e[me][0], [28] = my flags */
		uint32_t hdr[32] = { 0 }; hdr[3] = (uint32_t)d; hdr[4] = (uint32_t)nd;
		for (int e = 0; e < 6; e++) { uint64_t a = e < nd ? (uint64_t)xc[e].flags + 64u * (unsigned)d : (uint64_t)dCTL[d] + 0xc0; memcpy(&hdr[16 + 2 * e], &a, 8); }   /* absent dies: a scratch word, so the framework's six flag stores are unconditional */
		uint64_t mf = (uint64_t)xc[d].flags; memcpy(&hdr[28], &mf, 8);
		CK(cuMemsetD8(dCTL[d], 0, 0x100)); CK(cuMemcpyHtoD(dCTL[d], hdr, sizeof hdr));   /* ctl[32..47]: the grid barrier arrival flags */
	}
	/* the initial stream: the embedding of ids[0], or random when no embedding runs */
	if (!do_embed) { for (int i = 0; i < 10240; i++) hb->S[i] = 0.5f * frand(); for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].S, hb->S, 4 * 10240)); } }
	printf("%d die(s), blocks [%d, %d)%s%s%s, %d tokens (%d prompt)\n", nd, b0, b1, do_embed ? ", embed" : "", do_head ? ", head" : "", do_loop ? ", one launch" : "", T, nprompt);
	int fails = 0; unsigned tok_gpu = 0, tok_twin = 0; double tsum = 0;
	if (do_loop) {
		/* the twin runs every token first (its own greedy sequence) */
		unsigned *tw = calloc((size_t)T + 1, 4);
		double t0 = now();
		if (!notwin) for (int t = 0; t < T; t++) {
			unsigned idt = t < nprompt ? ids[t] : tok_twin;
			if (do_embed) embed_gpu(&hwb[0], hb, idt);
			for (int b = b0; b < b1; b++) {
				hb->dn_state = hs[b].dn_state; hb->kc = hs[b].kc; hb->vc = hs[b].vc;
				hb->qsa = model_qsa && BLK_IS_ATTN(b) ? model_qhost[b] : NULL;
				if (hs[b].conv_state) memcpy(hb->conv_state, hs[b].conv_state, sizeof hb->conv_state);
				if (hs[b].ple_state) memcpy(hb->ple_state, hs[b].ple_state, sizeof hb->ple_state);
				if (b == 1) ple_prefix_gpu(&hwb[b], hb, (unsigned)t, eps);
				if (BLK_IS_ATTN(b)) attn_block_gpu(&hwb[b], hb, (unsigned)t, eps, nd, b, 0); else deltanet_block_gpu(&hwb[b], hb, (unsigned)t, eps, nd, b, 0);
				if (model_qsa) memcpy(model_href[b], hb->S, 4 * 10240);
				if (hs[b].conv_state) memcpy(hs[b].conv_state, hb->conv_state, sizeof hb->conv_state);
				if (hs[b].ple_state) memcpy(hs[b].ple_state, hb->ple_state, sizeof hb->ple_state);
			}
			if (do_head) { head_gpu(&hwb[0], hb, eps, nd); tok_twin = hb->tok; tw[t] = hb->tok; if (t + 1 >= nprompt && t + 1 < T) ((uint32_t *)ids)[t + 1] = 0; }
		}
		double t1 = now();
		/* one program per die, token-relative, launched once for T tokens */
		prog_t P[MAXD]; unsigned nph[MAXD];
		for (int d = 0; d < nd; d++) {
			prog_init(&P[d]); P[d].loop = 1;
			if (do_embed) {
				if (model_qsa) prog_embed_bcast(&P[d], &xc[d], d, nd, model_embed_die, &dw[d][0], &db[d], 0);
				else prog_embed(&P[d], &dw[d][0], &db[d], 0);
			}
			for (int b = b0; b < b1; b++) {
				dn_bufs_t bb = db[d]; bb.dn_state = dst_dn[d][b]; bb.conv_state = dst_conv[d][b]; bb.kc = dst_kc[d][b]; bb.vc = dst_vc[d][b]; bb.ple_state = dst_ple[d];
				if (model_qsa && BLK_IS_ATTN(b)) model_qsa_bind(&bb, d, b);
				if (b == 1) prog_ple_prefix(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, 0, eps);
				if (BLK_IS_ATTN(b)) prog_attn_block(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, 0, eps);
				else prog_deltanet_block(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, 0, eps);
				if (model_qsa) model_layer_trace(&P[d], d, b, db[d].S);
			}
			if (do_head) prog_head(&P[d], &sp[d][b0], &xc[d], &dw[d][0], &db[d], eps, 0, (unsigned)nprompt);
			prog_token_end(&P[d], &xc[d], d, nd, db[d].S);
			if (model_qsa) model_assert_program(&P[d], d, b0, b1);
			nph[d] = (unsigned)P[d].n;
			CK(cuCtxSetCurrent(ctx[d]));
			CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
		}
		double t2 = now();
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			unsigned ntok = (unsigned)T; void *params[] = { &dPROG[d], &nph[d], &dCTL[d], &ntok };
			CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		}
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
		double t3 = now();
		printf("%u phases per token; twin %.2f s for %d tokens; dies %.3f ms total = %.3f ms per token\n", nph[0], t1 - t0, T, (t3 - t2) * 1e3, (t3 - t2) * 1e3 / T);
		if (getenv("MODEL_PROF")) {   /* die 0, the last token */
			CK(cuCtxSetCurrent(ctx[0]));
			uint32_t *ts = malloc(4 * (64 + nph[0])); CK(cuMemcpyDtoH(ts, dCTL[0], 4 * (64 + nph[0])));
			double byop[DIE_NOPS] = {0}; unsigned prev = ts[15], cnt[DIE_NOPS] = {0};
			for (unsigned i = 0; i < nph[0]; i++) { unsigned dd = ts[64 + i] - prev; prev = ts[64 + i]; byop[P[0].p[i].w[0]] += dd / 1e3; cnt[P[0].p[i].w[0]]++; }
			printf("  die 0 by op (us/count):");
			for (int o = 0; o < DIE_NOPS; o++) if (cnt[o]) printf(" %s %.0f/%u", die_opname[o], byop[o], cnt[o]);
			/* ts[15] is written once, at kernel entry, and the phase slots are overwritten every token: with many tokens in the launch this is the time since the
			   launch (mod 2^32 ns), NOT the last token, and the first op's bucket above absorbs every earlier token. Use the per-token launch path for a per-token profile. */
			printf("\n  since kernel entry %.1f us in %u phases (the last token only when the launch holds one token)\n", (ts[64 + nph[0] - 1] - ts[15]) / 1e3, nph[0]);
			free(ts);
		}
		unsigned *tg = calloc((size_t)T + 1, 4); int bad = 0;
		if (model_qsa && !notwin) bad += model_compare_layers(b0, b1, (unsigned)T - 1, do_head, hb, db, sp);
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			if (do_head) {
				CK(cuMemcpyDtoH(tg, db[d].toks, 4 * (size_t)T));
				int first = -1, nb = 0;
				for (int t = 0; t < T; t++) if (tg[t] != tw[t]) { if (first < 0) first = t; nb++; }
				if (!notwin) { if (nb) printf("  die %d tokens: %d of %d differ, first at %d (gpu %u twin %u)\n", d, nb, T, first, tg[first], tw[first]); else printf("  die %d tokens: all %d exact\n", d, T); }
				bad += nb;
			}
			if (!notwin) { char nm[16]; snprintf(nm, sizeof nm, "S.%d", d); if (!cmpbuf(nm, hb->S, db[d].S, 4 * 10240, 4)) printf("  %s exact\n", nm); else bad++; }
		}
		if (do_head) { printf("  tokens:"); for (int t = 0; t < T && t < 40; t++) printf(" %u", tg[t]); printf("%s\n", T > 40 ? " ..." : ""); }
		if (do_head && nprompt == T && getenv("MODEL_IDS")) {   /* teacher-forced: the prediction at t against the given id at t + 1 */
			int agree = 0, first = -1;
			for (int t = 0; t + 1 < T; t++) { if (tg[t] == ids[t + 1]) agree++; else if (first < 0) first = t; }
			printf("  teacher-forced: %d of %d next-token predictions match the given ids", agree, T - 1);
			if (first >= 0) printf("; first difference at %d (gpu %u given %u)", first, tg[first], ids[first + 1]);
			printf("\n");
		}
		printf("%s\n", notwin ? "ran" : bad ? "FAIL" : "OK: one launch, every token exact");
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); hold_close(&ar[d]); }
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
		return bad && !notwin ? 3 : 0;
	}
	for (int t = 0; t < T; t++) {
		unsigned id = t < nprompt ? ids[t] : tok_gpu;
		if (t >= nprompt) ids[t] = tok_gpu;   /* the PLE's context sees the generated token */
		if (pcompact && t >= nprompt) PLE_STAGE(t);   /* and its real PLE rows get staged */
		double t0 = now();
		if (!notwin) {
			unsigned idt = t < nprompt ? ids[t] : tok_twin;
			if (do_embed) embed_gpu(&hwb[0], hb, idt);
			for (int b = b0; b < b1; b++) {
				hb->dn_state = hs[b].dn_state; hb->kc = hs[b].kc; hb->vc = hs[b].vc;
				hb->qsa = model_qsa && BLK_IS_ATTN(b) ? model_qhost[b] : NULL;
				if (hs[b].conv_state) memcpy(hb->conv_state, hs[b].conv_state, sizeof hb->conv_state);
				if (hs[b].ple_state) memcpy(hb->ple_state, hs[b].ple_state, sizeof hb->ple_state);
				if (b == 1) ple_prefix_gpu(&hwb[b], hb, (unsigned)t, eps);
				if (BLK_IS_ATTN(b)) attn_block_gpu(&hwb[b], hb, (unsigned)t, eps, nd, b, 0); else deltanet_block_gpu(&hwb[b], hb, (unsigned)t, eps, nd, b, 0);
				if (model_qsa) memcpy(model_href[b], hb->S, 4 * 10240);
				if (hs[b].conv_state) memcpy(hs[b].conv_state, hb->conv_state, sizeof hb->conv_state);
				if (hs[b].ple_state) memcpy(hs[b].ple_state, hb->ple_state, sizeof hb->ple_state);
			}
			if (do_head) { head_gpu(&hwb[0], hb, eps, nd); tok_twin = hb->tok; }
			if (getenv("MODEL_DUMP_TWIN") && t == T - 1) {   /* the twin's intermediates of the last block (layer comparisons) */
				const char *dir = getenv("MODEL_DUMP_TWIN"); char fn[600];
				struct { const char *name; const void *p; size_t bytes; } bufs[] = {
					{ "S", hb->S, 4 * 10240 }, { "xh", hb->xh, 4 * 10240 }, { "z", hb->z, 4 * 2560 }, { "qkv", hb->qkv, 4 * 10240 }, { "gate", hb->gate, 4 * 6144 }, { "a", hb->a, 4 * 96 },
					{ "u", hb->u, 4 * 10240 }, { "o", hb->o, 4 * 6144 }, { "on", hb->on, 4 * 6144 }, { "y", hb->y, 4 * 2560 }, { "r", hb->r, 4 * 513 }, { "out", hb->out, 4 * 2560 }, { "t", hb->t, 4 * 324 }, { "up", hb->up, 4 * 10240 },
				};
				for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) { snprintf(fn, sizeof fn, "%s/%s.bin", dir, bufs[i].name); FILE *f = fopen(fn, "wb"); if (f) { fwrite(bufs[i].p, 1, bufs[i].bytes, f); fclose(f); } }
				printf("  twin buffers dumped to %s\n", dir);
			}
		}
		double t1 = now();
		prog_t P[MAXD]; unsigned nph[MAXD];
		for (int d = 0; d < nd; d++) {
			prog_init(&P[d]);
			if (do_embed) {
				CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].ids + 4u * (unsigned)t, &id, 4));
				if (model_qsa) prog_embed_bcast(&P[d], &xc[d], d, nd, model_embed_die, &dw[d][0], &db[d], (unsigned)t);
				else prog_embed(&P[d], &dw[d][0], &db[d], (unsigned)t);
			}
			for (int b = b0; b < b1; b++) {
				dn_bufs_t bb = db[d]; bb.dn_state = dst_dn[d][b]; bb.conv_state = dst_conv[d][b]; bb.kc = dst_kc[d][b]; bb.vc = dst_vc[d][b]; bb.ple_state = dst_ple[d];
				if (model_qsa && BLK_IS_ATTN(b)) model_qsa_bind(&bb, d, b);
				if (b == 1) prog_ple_prefix(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, (unsigned)t, eps);
				if (BLK_IS_ATTN(b)) prog_attn_block(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, (unsigned)t, eps);
				else prog_deltanet_block(&P[d], &sp[d][b], &xc[d], &dw[d][b], &bb, (unsigned)t, eps);
				if (model_qsa) model_layer_trace(&P[d], d, b, db[d].S);
			}
			if (do_head) prog_head(&P[d], &sp[d][b0], &xc[d], &dw[d][0], &db[d], eps, (unsigned)t, (unsigned)nprompt);
			if (model_qsa) model_assert_program(&P[d], d, b0, b1);
			nph[d] = (unsigned)P[d].n;
			if (getenv("MODEL_NPH") && (unsigned)atoi(getenv("MODEL_NPH")) < nph[d]) nph[d] = (unsigned)atoi(getenv("MODEL_NPH"));
			if (nph[d] > 4096) { fprintf(stderr, "program too long (%u phases)\n", nph[d]); return 1; }
			CK(cuCtxSetCurrent(ctx[d]));
			CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
		}
		double t2 = now();
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			unsigned ntok = 1; void *params[] = { &dPROG[d], &nph[d], &dCTL[d], &ntok };
			CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		}
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
		double t3 = now();
		tsum += t3 - t2;
		if (do_head) { CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(&tok_gpu, db[0].tokout, 4)); }
		model_capture_record((unsigned)t,(unsigned)T,ids,db,sp);
		if (getenv("MODEL_DUMP_S")) {   /* this position's stream: <path>.t<pos> */
			float *St = malloc(4 * 10240); CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(St, db[0].S, 4 * 10240));
			char fnt[600]; snprintf(fnt, sizeof fnt, "%s.t%d", getenv("MODEL_DUMP_S"), t); FILE *ft = fopen(fnt, "wb"); if (ft) { fwrite(St, 4, 10240, ft); fclose(ft); } free(St);
		}
		if (do_head && getenv("MODEL_TOPK")) {   /* the position's top-k logits, gathered from the dies' row slices (reference comparisons) */
			int k = atoi(getenv("MODEL_TOPK")); static float *lg = NULL; if (!lg) lg = malloc(4 * 248320);
			for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(lg + sp[d][b0].o0, db[d].hlo, 4u * (unsigned)sp[d][b0].on)); }
			printf("  pos %d top%d:", t, k);
			for (int j = 0; j < k; j++) { int bi = 0; for (int i = 1; i < 248320; i++) if (lg[i] > lg[bi]) bi = i; printf(" %d:%.4f", bi, lg[bi]); lg[bi] = -3.4e38f; }
			printf("\n");
		}
		int bad = 0;
		if (model_qsa && !notwin) bad += model_compare_layers(b0, b1, (unsigned)t, do_head, hb, db, sp);
		if (!notwin) {
			for (int d = 0; d < nd; d++) {
				CK(cuCtxSetCurrent(ctx[d]));
				char nm[16]; snprintf(nm, sizeof nm, "S.%d", d);
				bad += cmpbuf(nm, hb->S, db[d].S, 4 * 10240, 4);
				if (do_head) { uint32_t tk[2] = { hb->tok, 0 }; memcpy(&tk[1], &hb->tokmax, 4); snprintf(nm, sizeof nm, "tok.%d", d); bad += cmpbuf(nm, tk, db[d].tokout, 8, 4); }
			}
		}
		printf("token %3d: id %6u -> %6u%s %u phases; twin %.2f s, dies %.3f ms%s\n", t, id, tok_gpu, do_head ? "" : " (no head)", nph[0], t1 - t0, (t3 - t2) * 1e3, bad ? "  MISMATCH" : notwin ? "" : "  exact");
		if (bad) fails++;
		if (t == T - 1 && getenv("MODEL_PROF")) {
			CK(cuCtxSetCurrent(ctx[0]));
			uint32_t *ts = malloc(4 * (64 + nph[0])); CK(cuMemcpyDtoH(ts, dCTL[0], 4 * (64 + nph[0])));
			double byop[DIE_NOPS] = {0}; unsigned prev = ts[15], cnt[DIE_NOPS] = {0};
			for (unsigned i = 0; i < nph[0]; i++) { unsigned dd = ts[64 + i] - prev; prev = ts[64 + i]; byop[P[0].p[i].w[0]] += dd / 1e3; cnt[P[0].p[i].w[0]]++; }
			printf("  die 0 by op (us, count):");
			for (int o = 0; o < DIE_NOPS; o++) if (cnt[o]) printf(" %s %.0f/%u", die_opname[o], byop[o], cnt[o]);
			printf("\n  total %.1f us in %u phases\n", (ts[64 + nph[0] - 1] - ts[15]) / 1e3, nph[0]);
			free(ts);
		}
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
	}
	if (getenv("MODEL_DUMP_DIR")) {   /* die 0's whole-vector buffers after the last launch (with MODEL_NPH: a snapshot mid-block) */
		const char *dir = getenv("MODEL_DUMP_DIR"); char fn[600]; CK(cuCtxSetCurrent(ctx[0]));
		struct { const char *name; CUdeviceptr p; size_t bytes; } bufs[] = {
			{ "S", db[0].S, 4 * 10240 }, { "xh", db[0].xh, 4 * 10240 }, { "t", db[0].t, 4 * 324 }, { "r", db[0].r, 4 * 513 }, { "out", db[0].out, 4 * 2560 },
			{ "part", db[0].part, 4 * 2560 }, { "inbox0", xc[0].inbox, 4 * 10240 }, { "inbox1", xc[0].inbox + XSLOT, 4 * 10240 }, { "idx", db[0].idx, 40 }, { "w", db[0].w, 40 },
		};
		for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) {
			void *h = malloc(bufs[i].bytes); CK(cuMemcpyDtoH(h, bufs[i].p, bufs[i].bytes));
			snprintf(fn, sizeof fn, "%s/%s.bin", dir, bufs[i].name); FILE *f = fopen(fn, "wb"); if (f) { fwrite(h, 1, bufs[i].bytes, f); fclose(f); }
			free(h);
		}
		/* z: the quantised MoE/qkv input, dequantised (int16 codes x per-32 scale) */
		int16_t *q = malloc(2 * 2560); float *as = malloc(4 * 80), *z = malloc(4 * 2560);
		CK(cuMemcpyDtoH(q, db[0].zq, 2 * 2560)); CK(cuMemcpyDtoH(as, db[0].zas, 4 * 80));
		for (int i = 0; i < 2560; i++) z[i] = (float)q[i] * as[i / 32];
		snprintf(fn, sizeof fn, "%s/z.bin", dir); FILE *f = fopen(fn, "wb"); if (f) { fwrite(z, 4, 2560, f); fclose(f); }
		int16_t *oq = malloc(2 * 6144); float *oas = malloc(4 * 192), *o = malloc(4 * 6144);
		CK(cuMemcpyDtoH(oq, db[0].oq, 2 * 6144)); CK(cuMemcpyDtoH(oas, db[0].oas, 4 * 192));
		for (int i = 0; i < 6144; i++) o[i] = (float)oq[i] * oas[i / 32];
		snprintf(fn, sizeof fn, "%s/on.bin", dir); f = fopen(fn, "wb"); if (f) { fwrite(o, 4, 6144, f); fclose(f); }
		printf("  dumped die 0 buffers to %s\n", dir);
	}
	if (getenv("MODEL_DUMP_S")) {   /* die 0's stream after the last token (reference comparisons per block prefix); per-token files too */
		float *S = malloc(4 * 10240); CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(S, db[0].S, 4 * 10240));
		FILE *f = fopen(getenv("MODEL_DUMP_S"), "wb"); if (f) { fwrite(S, 4, 10240, f); fclose(f); }
		double ss = 0; for (int i = 0; i < 10240; i++) ss += (double)S[i] * S[i];
		printf("  S dumped to %s: lanes sum %.6g %.6g %.6g %.6g, |S| %.6g\n", getenv("MODEL_DUMP_S"), (double)S[0], (double)S[2560], (double)S[5120], (double)S[7680], ss);
		free(S);
	}
	printf("%s: %d of %d tokens %s; mean %.3f ms per token\n", fails ? "FAIL" : "OK", T - fails, T, notwin ? "run" : "exact", tsum / T * 1e3);
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); hold_close(&ar[d]); }
	return fails ? 3 : 0;
}
