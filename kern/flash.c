/* flash: the Flash-Next decoder front-end. Loads once (the dies' arenas from the holders'
 * manifests, the PLE table into pinned host memory, every buffer for FLASH_MAXT positions),
 * then serves responses from stdin, one persistent kernel launch per die per response:
 *
 *   gen <n_new> <id> [<id> ...]   run the prompt ids and greedily decode n_new tokens
 *                                 (n_new = 0: teacher-forced predictions for every position)
 *   cubin <path>                  reload the kernel (default die.cubin) on every die
 *   quit
 *
 * Output per response: "tokens: <toks[nprompt-1 .. T-1]>" (the prediction after each prompt
 * token's last position, then the generated ones; n_new = 0 prints toks[0 .. T-1]) and
 * "time: <ms> ms, <ms> per token". Env: SPLIT_SHARES, ATTN_DIE (ref/split.c), FLASH_MAXT
 * (positions, default 2048), FLASH_PLE_OFF (byte offset of the table in the PLE file,
 * 192 for shard 2 itself, 0 for an extracted copy). The PLE file "-" means no table (the
 * PLE rows read zeros: timing only). */
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

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "flash.c:%d: %s -> %s\n", __LINE__, #x, s_); exit(1); } } while (0)
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

#define MAXD 6
#define NB 48
static int nd;
static CUcontext ctx[MAXD];
static hold_arena ar[MAXD];
static char tname[128];
static const char *tn(int blk, const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static CUdeviceptr dptr_raw(int d, const char *name) { CUdeviceptr p = hold_ptr(&ar[d], name); if (!p) { fprintf(stderr, "die %d: no %s\n", d, name); exit(1); } return p; }
static CUdeviceptr dptr(int d, int blk, const char *suffix) { return dptr_raw(d, tn(blk, suffix)); }
static uint32_t ttype(int blk, const char *suffix) { const hold_tensor *t = hold_find(&ar[0], tn(blk, suffix)); if (!t) { fprintf(stderr, "no %s\n", tname); exit(1); } return t->type; }
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }

int main(int argc, char **argv) {
	if (argc < 3) { fprintf(stderr, "usage: flash <ple.bin | -> <manifest0> [manifest1 ...]\n"); return 2; }
	const char *plefile = argv[1]; nd = argc - 2; float eps = 1e-6f;
	if (nd > MAXD) { fprintf(stderr, "at most %d dies\n", MAXD); return 2; }
	int MAXT = getenv("FLASH_MAXT") ? atoi(getenv("FLASH_MAXT")) : 2048;
	const char *cubin = getenv("FLASH_CUBIN") ? getenv("FLASH_CUBIN") : "die.cubin";
	unsigned npos = ((unsigned)MAXT + 8192 + 64 + 31) / 32 * 32;
	size_t kcf = ATTN_KC_FLOATS(npos), vcf = (size_t)npos * 512;
	double ts0 = now();
	CK(cuInit(0));
	CUmodule mod[MAXD]; CUfunction fn[MAXD];
	for (int d = 0; d < nd; d++) {
		FILE *f = fopen(argv[2 + d], "r"); size_t bytes; int die;
		if (!f || fscanf(f, "arena %zu die %d", &bytes, &die) != 2) { fprintf(stderr, "bad manifest %s\n", argv[2 + d]); return 1; }
		fclose(f);
		CUdevice dev; if (hold_ordinal(die) < 0) { fprintf(stderr, "die %d is not in CUDA_VISIBLE_DEVICES\n", die); return 1; } CK(cuDeviceGet(&dev, hold_ordinal(die)));
		CK(cuCtxCreate_v2(&ctx[d], CU_CTX_SCHED_SPIN, dev));
		if (hold_open(&ar[d], argv[2 + d])) return 1;
		CK(cuModuleLoad(&mod[d], cubin)); CK(cuModuleGetFunction(&fn[d], mod[d], "die"));
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); for (int e = 0; e < nd; e++) if (e != d) CK(cuCtxEnablePeerAccess(ctx[e], 0)); }
	/* ---- the PLE table: pinned host memory, read from the file (CUDA 11.4 cannot pin a file mapping) ---- */
	uint8_t *ptab = NULL; uint8_t ppc[160]; uint64_t pmult[3]; uint32_t pabsent = 248044, pend = 248046;
	const size_t PLE_ROWS = 320001536, PLE_BYTES = PLE_ROWS * 90;
	memcpy(pmult, ple_multipliers, sizeof pmult);
	{
		/* no table: a zero compact table (16 rows per position) keeps plerows in bounds */
		size_t sz = strcmp(plefile, "-") ? PLE_BYTES : (size_t)MAXT * 16 * 90;
		CK(cuMemHostAlloc((void **)&ptab, sz + 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
		memset(ptab, 0, strcmp(plefile, "-") ? 4096 : sz + 4096);
		if (strcmp(plefile, "-")) {
			int fd = open(plefile, O_RDONLY); if (fd < 0) { perror(plefile); return 1; }
			off_t off = getenv("FLASH_PLE_OFF") ? (off_t)atoll(getenv("FLASH_PLE_OFF")) : 0;
			/* FLASH_PLE_CONSUME=1: the file is a tmpfs copy that does not fit in RAM next to the pinned
			   table; read it from the end and truncate as we go, so its pages are freed as ours fill */
			int consume = getenv("FLASH_PLE_CONSUME") != NULL; if (consume) { close(fd); fd = open(plefile, O_RDWR); if (fd < 0) { perror(plefile); return 1; } }
			size_t done = 0; double tp0 = now();
			while (done < PLE_BYTES) {
				size_t chunk = PLE_BYTES - done > (64u << 20) ? (64u << 20) : PLE_BYTES - done;
				size_t at = consume ? PLE_BYTES - done - chunk : done;   /* backwards when consuming */
				size_t got = 0;
				while (got < chunk) { ssize_t n = pread(fd, ptab + at + got, chunk - got, off + (off_t)(at + got)); if (n <= 0) { perror("pread ple"); return 1; } got += (size_t)n; }
				if (consume && ftruncate(fd, off + (off_t)at)) { perror("ftruncate ple"); return 1; }
				done += chunk;
				if ((done >> 30) != ((done - chunk) >> 30)) { fprintf(stderr, "\rPLE table: %zu of %zu GB (%.0f MB/s)", done >> 30, PLE_BYTES >> 30, done / 1e6 / (now() - tp0)); }
			}
			if (consume) unlink(plefile);
			close(fd);
			fprintf(stderr, "\rPLE table: %.2f GB pinned in %.0f s\n", PLE_BYTES / 1e9, now() - tp0);
		}
		memset(ppc, 0, sizeof ppc); memcpy(ppc, pmult, 24);
		for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)ple_head_offsets[h], v32 = (uint32_t)ple_head_vocab[h]; memcpy(ppc + 24 + 4 * h, &o32, 4); memcpy(ppc + 88 + 4 * h, &v32, 4); }
		memcpy(ppc + 152, &pabsent, 4); memcpy(ppc + 156, &pend, 4);
	}
	float *rcs = malloc(4 * 32 * (size_t)MAXT), *rsn = malloc(4 * 32 * (size_t)MAXT);
	for (int t = 0; t < MAXT; t++) rope_table_gpu((unsigned)t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	/* ---- the dies: weights, buffers, exchange, control ---- */
	dn_weights_t (*dw)[NB] = calloc(MAXD, sizeof *dw); dn_bufs_t db[MAXD]; xchg_t xc[MAXD]; split_t (*sp)[NB] = calloc(MAXD, sizeof *sp);
	CUdeviceptr dPROG[MAXD], dCTL[MAXD], (*dst_dn)[NB] = calloc(MAXD, sizeof *dst_dn), (*dst_conv)[NB] = calloc(MAXD, sizeof *dst_conv), (*dst_kc)[NB] = calloc(MAXD, sizeof *dst_kc), (*dst_vc)[NB] = calloc(MAXD, sizeof *dst_vc), dst_ple[MAXD];
	size_t dn_bytes[MAXD][NB], conv_bytes[MAXD][NB];
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
		b->rope_cs = dalloc(4 * 32 * (size_t)MAXT); b->rope_sn = dalloc(4 * 32 * (size_t)MAXT);
		CK(cuMemcpyHtoD(b->rope_cs, rcs, 4 * 32 * (size_t)MAXT)); CK(cuMemcpyHtoD(b->rope_sn, rsn, 4 * 32 * (size_t)MAXT));
		b->ids = dalloc(4 * ((size_t)MAXT + 2));
		b->ple_pc = dalloc(160); CK(cuMemcpyHtoD(b->ple_pc, ppc, 160)); CK(cuMemHostGetDevicePointer(&b->ple_tab, ptab, 0)); b->ple_compact = strcmp(plefile, "-") ? 0 : 1;
		b->pg = dalloc(4 * 2560); b->pgq = dalloc(2 * 2560); b->pgs32 = dalloc(4 * 80); b->pgs16 = dalloc(4 * 160); b->pgas = dalloc(4 * 80);
		b->lk = dalloc(4u * (unsigned)s0.pkn); b->kn = dalloc(4 * 10240); b->qn = dalloc(4 * 10240); b->lv = dalloc(4u * (unsigned)s0.yn);
		b->pG = dalloc(4 * 10240); b->pgates = dalloc(16); b->px = dalloc(4 * 10240); b->prows = dalloc(64);
		dst_ple[d] = dalloc(4 * 10 * 10240);
		b->hlo = dalloc(4u * (unsigned)s0.on); b->amo = dalloc(8); b->apart = dalloc(16 * 8); b->acnt = dalloc(4); b->tokout = dalloc(8); b->toks = dalloc(4 * ((size_t)MAXT + 2));
		for (int bb = 0; bb < NB; bb++) {
			split_t *s = &sp[d][bb]; split_of(s, d, nd, bb);
			dn_weights_t *w = &dw[d][bb]; int is_attn = BLK_IS_ATTN(bb);
			w->hc_attn_norm = dptr(d, bb, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, bb, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, bb, "hc_attn_up.weight");
			w->hc_ffn_norm = dptr(d, bb, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, bb, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, bb, "hc_ffn_up.weight");
			if (!is_attn) {
				w->attn_qkv = dptr(d, bb, "attn_qkv.weight"); w->attn_qkv_type = ttype(bb, "attn_qkv.weight");
				w->attn_gate = dptr(d, bb, "attn_gate.weight"); w->attn_gate_type = ttype(bb, "attn_gate.weight");
				w->ssm_out = dptr(d, bb, "ssm_out.weight"); w->ssm_out_type = ttype(bb, "ssm_out.weight");
				w->ssm_ab = dptr(d, bb, "ssm_ab.weight"); w->ssm_ab_bias = dptr(d, bb, "ssm_ab.bias");
				w->ssm_conv1d = dptr(d, bb, "ssm_conv1d.weight"); w->ssm_a = dptr(d, bb, "ssm_a"); w->ssm_norm = dptr(d, bb, "ssm_norm.weight");
				dn_bytes[d][bb] = (size_t)s->hn * 16384 * 4; conv_bytes[d][bb] = 4 * 4 * SPLIT_NLOC(s);
				dst_dn[d][bb] = dalloc(dn_bytes[d][bb]); dst_conv[d][bb] = dalloc(conv_bytes[d][bb]);
			} else {
				w->attn_q = dptr(d, bb, "attn_q.weight"); w->attn_q_type = ttype(bb, "attn_q.weight");
				w->attn_k = dptr(d, bb, "attn_k.weight"); w->attn_k_type = ttype(bb, "attn_k.weight");
				w->attn_v = dptr(d, bb, "attn_v.weight"); w->attn_v_type = ttype(bb, "attn_v.weight");
				w->attn_output = dptr(d, bb, "attn_output.weight"); w->attn_output_type = ttype(bb, "attn_output.weight");
				w->attn_q_norm = dptr(d, bb, "attn_q_norm.weight"); w->attn_k_norm = dptr(d, bb, "attn_k_norm.weight");
				dst_kc[d][bb] = dalloc(d == attn_die() ? kcf * 4 : 256); dst_vc[d][bb] = dalloc(d == attn_die() ? vcf * 4 : 256);
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
		dn_weights_t *w = &dw[d][0];
		w->output_hc_norm = dptr_raw(d, "output_hc_norm.weight"); w->output_hc_down = dptr_raw(d, "output_hc_down.weight"); w->output_hc_up = dptr_raw(d, "output_hc_up.weight"); w->output = dptr_raw(d, "output.weight");
		uint32_t ob[6]; for (int e = 0; e < 6; e++) ob[e] = split_bound(248320u, e, nd);
		w->obases = dalloc(24); CK(cuMemcpyHtoD(w->obases, ob, 24));
		w->token_embd = dptr_raw(d, "token_embd.weight");
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * XSLOT); x->flags = dalloc(512); x->xtab = dalloc(128);
		uint32_t fl[96]; for (int e = 0; e < 96; e++) fl[e] = e / 16 < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		dPROG[d] = dalloc(128 * 4096); dCTL[d] = dalloc(0x100 + 4 * 4096);
		x->xgen = dCTL[d] + 0x20;
	}
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { tab[d] = (uint64_t)xc[d < nd ? d : 0].inbox; tab[6 + d] = (uint64_t)xc[d < nd ? d : 0].flags; }
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab));
		uint32_t hdr[32] = { 0 }; hdr[3] = (uint32_t)d; hdr[4] = (uint32_t)nd;
		for (int e = 0; e < 6; e++) { uint64_t a = e < nd ? (uint64_t)xc[e].flags + 64u * (unsigned)d : (uint64_t)dCTL[d] + 0xc0; memcpy(&hdr[16 + 2 * e], &a, 8); }
		uint64_t mf = (uint64_t)xc[d].flags; memcpy(&hdr[28], &mf, 8);
		CK(cuMemsetD8(dCTL[d], 0, 0x100)); CK(cuMemcpyHtoD(dCTL[d], hdr, sizeof hdr));
	}
	fprintf(stderr, "flash: %d dies, %d blocks, up to %d positions, ready in %.0f s\n", nd, NB, MAXT, now() - ts0);
	printf("ready\n"); fflush(stdout);
	/* ---- responses ---- */
	uint32_t *ids = calloc((size_t)MAXT + 2, 4), *toks = calloc((size_t)MAXT + 2, 4);
	char *line = NULL; size_t cap = 0;
	while (getline(&line, &cap, stdin) > 0) {
		char *p = line; while (*p == ' ') p++;
		if (!strncmp(p, "quit", 4)) break;
		if (!strncmp(p, "cubin", 5)) {
			char path[512]; if (sscanf(p + 5, "%511s", path) != 1) { printf("error: cubin <path>\n"); fflush(stdout); continue; }
			for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleUnload(mod[d])); CK(cuModuleLoad(&mod[d], path)); CK(cuModuleGetFunction(&fn[d], mod[d], "die")); }
			printf("ok: %s\n", path); fflush(stdout); continue;
		}
		if (strncmp(p, "gen", 3)) { printf("error: gen <n_new> <id>... | cubin <path> | quit\n"); fflush(stdout); continue; }
		p += 3; char *end; long n_new = strtol(p, &end, 10); if (end == p || n_new < 0) { printf("error: gen <n_new> <id>...\n"); fflush(stdout); continue; }
		int nprompt = 0; p = end;
		for (;;) { long v = strtol(p, &end, 10); if (end == p) break; if (nprompt + 2 >= MAXT) break; ids[nprompt++] = (uint32_t)v; p = end; }
		int T = nprompt + (int)n_new;
		if (nprompt < 1 || T > MAXT) { printf("error: %d prompt ids + %ld new > %d positions\n", nprompt, n_new, MAXT); fflush(stdout); continue; }
		memset(ids + nprompt, 0, 4 * (size_t)(T + 2 - nprompt));
		double t0 = now();
		prog_t *P = calloc(MAXD, sizeof *P); unsigned nph[MAXD];
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			/* a fresh response: states, caches and the token counter start from zero */
			for (int bb = 0; bb < NB; bb++) {
				if (BLK_IS_ATTN(bb)) { CK(cuMemsetD8(dst_kc[d][bb], 0, d == attn_die() ? kcf * 4 : 256)); CK(cuMemsetD8(dst_vc[d][bb], 0, d == attn_die() ? vcf * 4 : 256)); }
				else { CK(cuMemsetD8(dst_dn[d][bb], 0, dn_bytes[d][bb])); CK(cuMemsetD8(dst_conv[d][bb], 0, conv_bytes[d][bb])); }
			}
			CK(cuMemsetD8(dst_ple[d], 0, 4 * 10 * 10240));
			CK(cuMemsetD32(dCTL[d] + 0x8, 0, 1));
			CK(cuMemcpyHtoD(db[d].ids, ids, 4 * ((size_t)T + 2)));
			CK(cuMemsetD8(db[d].toks, 0, 4 * ((size_t)T + 2)));
			prog_init(&P[d]); P[d].loop = 1;
			prog_embed(&P[d], &dw[d][0], &db[d], 0);
			for (int bb = 0; bb < NB; bb++) {
				dn_bufs_t bb_ = db[d]; bb_.dn_state = dst_dn[d][bb]; bb_.conv_state = dst_conv[d][bb]; bb_.kc = dst_kc[d][bb]; bb_.vc = dst_vc[d][bb]; bb_.ple_state = dst_ple[d];
				if (bb == 1) prog_ple_prefix(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
				if (BLK_IS_ATTN(bb)) prog_attn_block(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
				else prog_deltanet_block(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
			}
			prog_head(&P[d], &sp[d][0], &xc[d], &dw[d][0], &db[d], eps, 0, (unsigned)nprompt);
			prog_token_end(&P[d], &xc[d], d, nd, db[d].S);
			nph[d] = (unsigned)P[d].n;
			if (nph[d] > 4096) { fprintf(stderr, "program too long (%u phases)\n", nph[d]); return 1; }
			CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
		}
		double t1 = now();
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			unsigned ntok = (unsigned)T; void *params[] = { &dPROG[d], &nph[d], &dCTL[d], &ntok };
			CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		}
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
		double t2 = now();
		CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(toks, db[0].toks, 4 * (size_t)T));
		printf("tokens:"); for (int t = n_new ? nprompt - 1 : 0; t < T; t++) printf(" %u", toks[t]); printf("\n");
		printf("time: %.1f ms for %d positions (%.2f ms per position; %u phases; programs %.0f ms)\n", (t2 - t1) * 1e3, T, (t2 - t1) * 1e3 / T, nph[0], (t1 - t0) * 1e3);
		fflush(stdout);
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
		free(P);
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); hold_close(&ar[d]); }
	return 0;
}
