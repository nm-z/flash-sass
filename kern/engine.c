/* engine: the reloadable half of the Flash-Next decoder (libengine.so). flashcore.c keeps the
 * contexts, the weight arenas and the PLE table resident and dlopen()s this library; engine_init
 * builds every device buffer and program-independent table for FLASH_MAXT positions, then
 * engine_session() serves one client's commands, one persistent kernel launch per die per response
 * (see the command list at engine_session). `reload [path]` replaces the library in seconds without
 * touching the weights or the table. Env: SPLIT_SHARES, ATTN_DIE (ref/split.c), FLASH_CUBIN. */
#include "block.h"
#include "attn.h"
#include "ple.h"
#include "hold.h"
#include "prog.h"
#include "split.h"
#include "qsa.h"
#include <math.h>
#include "gguf.h"
#include "engine.h"
#include "core.h"
#include <setjmp.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
/* during engine_init a failure unwinds to it (the core and its weights and table stay up); later ones are fatal */
static jmp_buf init_jb, run_jb; static int in_init, in_run;
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "engine.c:%d: %s -> %s\n", __LINE__, #x, s_); if (in_init) longjmp(init_jb, 1); if (in_run) longjmp(run_jb, 1); exit(1); } } while (0)
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

#define MAXD CORE_MAXD
#define NB 48
static core_t *core;
#define nd (core->nd)
#define ctx (core->ctx)
#define ar (core->ar)
static char tname[128];
static const char *tn(int blk, const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static CUdeviceptr dptr_raw(int d, const char *name) { CUdeviceptr p = hold_ptr(&ar[d], name); if (!p) { fprintf(stderr, "die %d: no %s\n", d, name); if (in_init) longjmp(init_jb, 1); exit(1); } return p; }
static CUdeviceptr dptr(int d, int blk, const char *suffix) { return dptr_raw(d, tn(blk, suffix)); }
static uint32_t ttype(int blk, const char *suffix) { const hold_tensor *t = hold_find(&ar[0], tn(blk, suffix)); if (!t) { fprintf(stderr, "no %s\n", tname); if (in_init) longjmp(init_jb, 1); exit(1); } return t->type; }
/* every device allocation is recorded so engine_fini can free it */
typedef struct { CUdeviceptr p; CUcontext c; } allocrec;
static allocrec *allocs; static int nallocs, capallocs; static size_t alloc_bytes;
static CUdeviceptr dalloc(size_t bytes) {
	CUdeviceptr p; CUcontext c;
	CK(cuMemAlloc(&p, bytes)); CK(cuCtxGetCurrent(&c));
	if (nallocs == capallocs) { capallocs = capallocs ? 2 * capallocs : 1024; allocs = realloc(allocs, sizeof *allocs * (size_t)capallocs); }
	allocs[nallocs++] = (allocrec){ p, c }; alloc_bytes += bytes;
	CK(cuMemsetD8(p, 0, bytes));
	return p;
}

/* ---- the engine: everything main() loads once, used by every response ---- */
static int MAXT; static float eps = 1e-6f;
static CUmodule mod[MAXD]; static CUfunction fn[MAXD];
static uint32_t *s_ids, *s_toks;   /* engine_session scratch (positions each) */
static dn_weights_t (*dw)[NB]; static dn_bufs_t db[MAXD]; static xchg_t xc[MAXD]; static split_t (*sp)[NB];
static CUdeviceptr dPROG[MAXD], dCTL[MAXD], (*dst_dn)[NB], (*dst_conv)[NB], (*dst_kq)[NB], (*dst_ks)[NB], (*dst_vq)[NB], (*dst_vs)[NB], (*dst_pkt)[NB], (*dst_ring)[NB], (*snap_ring)[NB], dst_ple[MAXD];
static size_t dn_bytes[MAXD][NB], conv_bytes[MAXD][NB], kcf, vcf;
static unsigned nblk_qsa = QSA_NBLK;   /* blocks of 4 keys attended per layer and token (FLASH_NBLK: a small value exercises the selection on short prompts) */
static int has_attn[MAXD];             /* the die runs at least one attention layer */
static uint32_t *h_ids, *h_toks;    /* pinned: the prompt, and the tokens as die 0 writes them */
static float *h_rnd;                /* pinned: u[position] for the sampler */
static uint32_t *h_tok;             /* pinned: ctl[2] (count, bit 31 = the turn ended) */
static CUstream pollst[MAXD];       /* per die, non-blocking: the kernel runs on the legacy stream, these do not wait for it */
static uint32_t *h_abort;           /* pinned: the abort count written to every die */

void engine_fini(void);
int engine_positions(void) { return MAXT; }

/* ---- the prefix cache: a snapshot of every recurrent state (DeltaNet matrices, conv windows, the PLE ring)
   after snap_s positions of a prompt (and the indexer's raw key ring). The attention caches need none: positions >= snap_s are simply
   overwritten, and positions < snap_s are untouched by a response that resumes from the snapshot.
   A request whose first snap_s tokens are the same restores it and starts at position snap_s. ---- */
static CUdeviceptr (*snap_dn)[NB], (*snap_conv)[NB], snap_ple[MAXD];
static uint32_t *snap_ids; static int snap_s;
static uint32_t prof_op[4096]; static int prof_nph;   /* die 0 program of the last response, for the "prof" command */

static void state_copy(int to_snapshot) {
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int bb = 0; bb < NB; bb++) {
			if (BLK_IS_ATTN(bb)) {   /* the raw indexer keys of the unfinished block (the caches themselves are positional) */
				if (dst_ring[d][bb]) { if (to_snapshot) CK(cuMemcpyDtoD(snap_ring[d][bb], dst_ring[d][bb], 2048)); else CK(cuMemcpyDtoD(dst_ring[d][bb], snap_ring[d][bb], 2048)); }
				continue;
			}
			if (to_snapshot) { CK(cuMemcpyDtoD(snap_dn[d][bb], dst_dn[d][bb], dn_bytes[d][bb])); CK(cuMemcpyDtoD(snap_conv[d][bb], dst_conv[d][bb], conv_bytes[d][bb])); }
			else { CK(cuMemcpyDtoD(dst_dn[d][bb], snap_dn[d][bb], dn_bytes[d][bb])); CK(cuMemcpyDtoD(dst_conv[d][bb], snap_conv[d][bb], conv_bytes[d][bb])); }
		}
		if (to_snapshot) CK(cuMemcpyDtoD(snap_ple[d], dst_ple[d], 4 * 10 * 10240)); else CK(cuMemcpyDtoD(dst_ple[d], snap_ple[d], 4 * 10 * 10240));
	}
}

/* ---- one response is one or two launches of the persistent kernel: [start, snap_at) (only when a snapshot is
   taken) and [.., T) ---- */
typedef struct {
	int T, nprompt, ntok, sent, count, ended, aborted;
	double tfirst;
	const engine_cb *cb;
} runctx;

static void abort_run(runctx *r) {
	/* the client is gone: every die ends the turn when its token count reaches a value a few tokens ahead of all of
	   them (the dies are within one token of each other), so they stop together */
	r->aborted = 1; *h_abort = (uint32_t)r->count + 8;
	if (*h_abort < (uint32_t)r->ntok) for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoDAsync(dCTL[d] + 0x18, h_abort, 4, pollst[d])); CK(cuStreamSynchronize(pollst[d])); }
	CK(cuCtxSetCurrent(ctx[0]));
}

/* launch the program on every die for tokens up to ntok and wait for it, streaming the generated tokens */
static void run_segment(runctx *r, const unsigned *nph, int ntok_) {
	r->ntok = ntok_;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		unsigned ntok = (unsigned)ntok_; void *params[] = { &dPROG[d], (void *)&nph[d], &dCTL[d], &ntok };
		CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	}
	/* poll ctl[2] on die 0 through a non-blocking stream (the kernel runs on the legacy one), then the new tokens;
	   once the kernel is done, one last read */
	CK(cuCtxSetCurrent(ctx[0]));
	struct timespec nap = { 0, 500000 };
	double last_alive = now();
	for (int fin = 0; !fin; ) {
		CUresult q = cuStreamQuery(0);
		if (q != CUDA_SUCCESS && q != CUDA_ERROR_NOT_READY) CK(q);
		fin = q == CUDA_SUCCESS;
		CK(cuMemcpyDtoHAsync(h_tok, dCTL[0] + 0x8, 4, pollst[0])); CK(cuStreamSynchronize(pollst[0]));
		r->count = (int)(*h_tok & 0x7fffffffu); r->ended = (int)(*h_tok >> 31);
		if (r->count > r->T) r->count = r->T;
		if (r->count > r->sent) {
			CK(cuMemcpyDtoHAsync(h_toks + r->sent, db[0].toks + 4u * (unsigned)r->sent, 4 * (size_t)(r->count - r->sent), pollst[0])); CK(cuStreamSynchronize(pollst[0]));
			if (!r->tfirst) r->tfirst = now();
			for (; r->sent < r->count; r->sent++)
				if (r->cb && r->cb->token && r->cb->token(r->cb->arg, r->sent, h_toks[r->sent]) && !r->aborted && !r->ended) abort_run(r);
		}
		if (!r->aborted && !fin && r->cb && r->cb->alive && now() - last_alive > 0.02) { last_alive = now(); if (r->cb->alive(r->cb->arg) && !r->ended) abort_run(r); }
		if (!fin) nanosleep(&nap, NULL);
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
}

/* ---- images: the embeddings that replace placeholder tokens, and the M-RoPE positions of every token ----
   Image tokens take fp16 rows of the img table through imap; the attention layers rotate q / k by the position triple (t, y, x) of
   a token, one axis per rope pair (pair p takes axis p % 3, the interleaved sections of the model), so the RoPE tables of the
   dies that run attention are rebuilt from the positions of this response. Text tokens have (L, L, L) with L their logical
   position: an image consumes npos = max(nx, ny) logical positions, not one per token. */
static unsigned img_rows_max;          /* rows of the img table (FLASH_MAXIMG) */
static uint32_t *h_imap;               /* [MAXT + 2] */
static int imap_dirty, rope_dirty_T;   /* the device holds image rows / image-built RoPE rows of this many positions */
static void rope_upload(const uint32_t (*pos3)[3], int T) {
	float *cs = malloc(4 * 32 * (size_t)T), *sn = malloc(4 * 32 * (size_t)T);
	double theta[32]; for (int p = 0; p < 32; p++) theta[p] = pow((double)1e7f, -2.0 * p / 64.0);
	for (int t = 0; t < T; t++) for (int p = 0; p < 32; p++) { double ang = (double)pos3[t][p % 3] * theta[p]; cs[32 * t + p] = (float)cos(ang); sn[32 * t + p] = (float)sin(ang); }
	for (int d = 0; d < nd; d++) if (has_attn[d]) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].rope_cs, cs, 4 * 32 * (size_t)T)); CK(cuMemcpyHtoD(db[d].rope_sn, sn, 4 * 32 * (size_t)T)); }
	free(cs); free(sn);
}
static void rope_plain(int T) {
	uint32_t (*pos3)[3] = malloc(12 * (size_t)T);
	for (int t = 0; t < T; t++) pos3[t][0] = pos3[t][1] = pos3[t][2] = (uint32_t)t;
	rope_upload(pos3, T); free(pos3);
}
/* 0, or -3: a bad file, a segment outside the prompt or out of order, too many rows */
static int prepare_images(const engine_img *imgs, int nimg, int nprompt, int T) {
	if (!nimg) {   /* a response without images: undo what the last one left on the dies */
		if (imap_dirty) { for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemsetD32(db[d].imap, 0xffffffffu, (size_t)MAXT + 2)); } imap_dirty = 0; }
		if (rope_dirty_T) { rope_plain(rope_dirty_T); rope_dirty_T = 0; }
		return 0;
	}
	if (!img_rows_max) return -3;
	uint32_t (*pos3)[3] = malloc(12 * (size_t)T);
	uint16_t *h16 = malloc(2 * 2560 * (size_t)img_rows_max);
	for (int t = 0; t < T + 2; t++) h_imap[t] = 0xffffffffu;
	long L = 0; int t = 0, rows = 0, bad = 0;
	for (int i = 0; i < nimg && !bad; i++) {
		const engine_img *g = &imgs[i];
		FILE *f = fopen(g->path, "rb"); uint32_t hd[5];
		if (!f || fread(hd, 4, 5, f) != 5 || hd[4] != 2560 || g->s < t || g->s + (int)hd[0] > nprompt || rows + (int)hd[0] > (int)img_rows_max) { bad = 1; if (f) fclose(f); break; }
		for (; t < g->s; t++, L++) pos3[t][0] = pos3[t][1] = pos3[t][2] = (uint32_t)L;
		float *e = malloc(4 * 2560); uint32_t *rel = malloc(12 * (size_t)hd[0]);
		for (uint32_t k = 0; k < hd[0] && !bad; k++) {
			if (fread(e, 4, 2560, f) != 2560) { bad = 1; break; }
			for (int j = 0; j < 2560; j++) h16[(size_t)(rows + (int)k) * 2560 + j] = qsa_f2h(e[j]);
		}
		if (!bad && fread(rel, 12, hd[0], f) != hd[0]) bad = 1;
		for (uint32_t k = 0; k < hd[0] && !bad; k++) {
			h_imap[g->s + (int)k] = (uint32_t)(rows + (int)k);
			pos3[g->s + (int)k][0] = (uint32_t)L + rel[3 * k]; pos3[g->s + (int)k][1] = (uint32_t)L + rel[3 * k + 1]; pos3[g->s + (int)k][2] = (uint32_t)L + rel[3 * k + 2];
		}
		fclose(f); free(e); 
		if (!bad) { t = g->s + (int)hd[0]; L += (long)hd[3]; rows += (int)hd[0]; }
		free(rel);
	}
	if (!bad) {
		for (; t < T; t++, L++) pos3[t][0] = pos3[t][1] = pos3[t][2] = (uint32_t)L;
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].img, h16, 2 * 2560 * (size_t)rows)); CK(cuMemcpyHtoD(db[d].imap, h_imap, 4 * ((size_t)T + 2))); }
		rope_upload(pos3, T); imap_dirty = 1; if (T > rope_dirty_T) rope_dirty_T = T;
	}
	free(pos3); free(h16);
	return bad ? -3 : 0;
}

int engine_run(const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int snap_at, const engine_img *imgs, int nimg, const engine_cb *cb, run_stats *st) {
	int T = nprompt + n_new;
	if (nprompt < 1 || n_new < 0 || T > MAXT) return -1;
	/* a CUDA error from here on unwinds to this point: the contexts are no longer usable, the core rebuilds them (return -2) */
	in_run = 1;
	if (setjmp(run_jb)) { in_run = 0; snap_s = 0; return -2; }
	memcpy(h_ids, ids, 4 * (size_t)nprompt); memset(h_ids + nprompt, 0, 4 * (size_t)(T + 2 - nprompt));
	int irc = prepare_images(imgs, nimg, nprompt, T);
	if (irc) { in_run = 0; return irc; }
	double t0 = now();
	int samp_k = (sm && sm->temp > 0.f && sm->top_k >= 2) ? (sm->top_k > 32 ? 32 : sm->top_k) : 0;
	float samp_topp = sm && sm->topp > 0.f ? sm->topp : 1.0f, samp_invT = samp_k ? 1.0f / sm->temp : 1.0f;
	if (samp_k) {   /* u[t] in [0, 1): splitmix64 over the seed, 24 bits each */
		uint64_t z0 = sm->seed;
		for (int t = 0; t < T + 2; t++) { uint64_t z = (z0 += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31; h_rnd[t] = (float)(z >> 40) / 16777216.0f; }
	}
	/* the prefix cache: resume from the snapshot if the prompt starts with its tokens (at least one prompt token must
	   still run, to produce the logits of the last one) */
	int resume = (snap_s > 0 && snap_s <= nprompt - 1 && !memcmp(ids, snap_ids, 4 * (size_t)snap_s)) ? snap_s : 0;
	if (snap_at > nprompt - 1) snap_at = nprompt - 1;
	for (int i = 0; i < nimg; i++) if (snap_at > imgs[i].s) snap_at = imgs[i].s;   /* the snapshot would not know the image */
	if (snap_at <= resume) snap_at = 0;
	if (!resume) snap_s = 0;   /* this response rewrites the attention caches from position 0 */
	prog_t *P = calloc(MAXD, sizeof *P); unsigned nph[MAXD];
	if (resume) state_copy(0);
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		/* a fresh response: states and caches start from zero (not when resuming from the snapshot: the recurrent states
		   were restored, the attention caches below the snapshot are intact), and so do the end flag and the abort count */
		if (!resume) {
			for (int bb = 0; bb < NB; bb++) {
				/* the int8 caches, the indexer pool and the ring of an attention layer need no reset: positions are written before they are read */
				if (BLK_IS_ATTN(bb)) continue;
				else { CK(cuMemsetD8(dst_dn[d][bb], 0, dn_bytes[d][bb])); CK(cuMemsetD8(dst_conv[d][bb], 0, conv_bytes[d][bb])); }
			}
			CK(cuMemsetD8(dst_ple[d], 0, 4 * 10 * 10240));
		}
		CK(cuMemsetD32(dCTL[d] + 0x8, (unsigned)resume, 1)); CK(cuMemsetD32(dCTL[d] + 0x14, 0, 2));   /* token count, end flag, abort count */
		CK(cuMemcpyHtoD(db[d].ids, h_ids, 4 * ((size_t)T + 2)));
		CK(cuMemsetD8(db[d].toks, 0, 4 * ((size_t)T + 2)));
		prog_init(&P[d]); P[d].loop = 1;
		P[d].stop[0] = stop[0]; P[d].stop[1] = stop[1]; P[d].end = dCTL[d] + 0x14;
		P[d].samp_k = samp_k; P[d].samp_topp = samp_topp; P[d].samp_invT = samp_invT;
		if (samp_k) CK(cuMemcpyHtoD(db[d].rnd, h_rnd, 4 * ((size_t)T + 2)));
		prog_embed(&P[d], &dw[d][0], &db[d], 0);
		for (int bb = 0; bb < NB; bb++) {
			dn_bufs_t bb_ = db[d]; bb_.dn_state = dst_dn[d][bb]; bb_.conv_state = dst_conv[d][bb]; bb_.ple_state = dst_ple[d];
			bb_.qsa = 1; bb_.nblk = nblk_qsa; bb_.kq = dst_kq[d][bb]; bb_.ks = dst_ks[d][bb]; bb_.vq = dst_vq[d][bb]; bb_.vs = dst_vs[d][bb]; bb_.pkt = dst_pkt[d][bb]; bb_.ring = dst_ring[d][bb];
			if (bb == 1) prog_ple_prefix(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
			if (BLK_IS_ATTN(bb)) prog_attn_block(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
			else prog_deltanet_block(&P[d], &sp[d][bb], &xc[d], &dw[d][bb], &bb_, 0, eps);
		}
		prog_head(&P[d], &sp[d][0], &xc[d], &dw[d][0], &db[d], eps, 0, (unsigned)nprompt);
		prog_token_end(&P[d], &xc[d], d, nd, db[d].S);
		nph[d] = (unsigned)P[d].n;
		if (d == 0) { prof_nph = (int)nph[0] < 4096 ? (int)nph[0] : 4096; for (int i = 0; i < prof_nph; i++) prof_op[i] = P[0].p[i].w[0]; }
		if (nph[d] > 4096) { fprintf(stderr, "program too long (%u phases)\n", nph[d]); exit(1); }
		CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n));
	}
	double t1 = now();
	runctx r = { .T = T, .nprompt = nprompt, .sent = nprompt - 1, .cb = cb };
	if (snap_at) {   /* run the prompt up to the snapshot position, copy the states, go on */
		run_segment(&r, nph, snap_at);
		if (!r.aborted) {
			state_copy(1);
			memcpy(snap_ids, ids, 4 * (size_t)snap_at); snap_s = snap_at;
		}
	}
	if (!r.aborted) run_segment(&r, nph, T);
	double t2 = now();
	if (st) { st->ntok = r.count; st->ended = r.ended; st->nph = nph[0]; st->resumed = resume; st->snapshot = snap_at; st->build_ms = (t1 - t0) * 1e3; st->run_ms = (t2 - t1) * 1e3; st->first_ms = r.tfirst ? (r.tfirst - t1) * 1e3 : 0; }
	for (int d = 0; d < nd; d++) prog_free(&P[d]);
	free(P);
	in_run = 0;
	return 0;
}

/* ---- build and tear down all device state (the core keeps the contexts, the arenas and the PLE table) ---- */
int engine_init(core_t *c) {
	if (c->abi != CORE_ABI) { fprintf(stderr, "engine: the core speaks ABI %d, this library was built for %d\n", c->abi, CORE_ABI); return -1; }
	core = c; in_init = 1;
	if (setjmp(init_jb)) { in_init = 0; engine_fini(); return -1; }
	MAXT = c->maxt;
	const char *cubin = getenv("FLASH_CUBIN") ? getenv("FLASH_CUBIN") : "die.cubin";
	img_rows_max = getenv("FLASH_MAXIMG") ? (unsigned)atoi(getenv("FLASH_MAXIMG")) : 2048;   /* image tokens per response (fp16 rows of 2560 on every die) */
	h_imap = calloc((size_t)MAXT + 2, 4); imap_dirty = 0; rope_dirty_T = 0;
	if (getenv("FLASH_NBLK")) nblk_qsa = (unsigned)atoi(getenv("FLASH_NBLK"));
	if (nblk_qsa < 1 || nblk_qsa > 4096) { fprintf(stderr, "engine: FLASH_NBLK %u out of range\n", nblk_qsa); return -1; }
	unsigned cpos = (4u * (nblk_qsa + 1) + 2048 + 31) / 32 * 32;   /* compact caches: the attended keys (4 per block + the partial one) and the kernels' stray prefetch past them */
	kcf = ATTN_KC_FLOATS(cpos); vcf = (size_t)cpos * 512;
	unsigned maxp = ((unsigned)MAXT + 127) & ~127u;   /* positions of the int8 caches and the pool: whole groups of 32 blocks */
	memset(has_attn, 0, sizeof has_attn);
	for (int bb = 3; bb < NB; bb += 4) if (attn_die_of(bb) < nd) has_attn[attn_die_of(bb)] = 1;
	double ts0 = now();
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleLoad(&mod[d], cubin)); CK(cuModuleGetFunction(&fn[d], mod[d], "die")); }
	/* ---- the PLE constants block (plerows layout); the table itself is the core's ---- */
	uint8_t ppc[160]; uint64_t pmult[3]; uint32_t pabsent = 248044, pend = 248046;
	memcpy(pmult, ple_multipliers, sizeof pmult);
	memset(ppc, 0, sizeof ppc); memcpy(ppc, pmult, 24);
	for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)ple_head_offsets[h], v32 = (uint32_t)ple_head_vocab[h]; memcpy(ppc + 24 + 4 * h, &o32, 4); memcpy(ppc + 88 + 4 * h, &v32, 4); }
	memcpy(ppc + 152, &pabsent, 4); memcpy(ppc + 156, &pend, 4);
	float *rcs = malloc(4 * 32 * (size_t)MAXT), *rsn = malloc(4 * 32 * (size_t)MAXT);
	for (int t = 0; t < MAXT; t++) rope_table_gpu((unsigned)t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	/* ---- the dies: weights, buffers, exchange, control ---- */
	dw = calloc(MAXD, sizeof *dw); sp = calloc(MAXD, sizeof *sp);
	dst_dn = calloc(MAXD, sizeof *dst_dn); dst_conv = calloc(MAXD, sizeof *dst_conv); dst_kq = calloc(MAXD, sizeof *dst_kq); dst_ks = calloc(MAXD, sizeof *dst_ks); dst_vq = calloc(MAXD, sizeof *dst_vq); dst_vs = calloc(MAXD, sizeof *dst_vs); dst_pkt = calloc(MAXD, sizeof *dst_pkt); dst_ring = calloc(MAXD, sizeof *dst_ring); snap_ring = calloc(MAXD, sizeof *snap_ring);
	snap_dn = calloc(MAXD, sizeof *snap_dn); snap_conv = calloc(MAXD, sizeof *snap_conv); snap_ids = malloc(4 * ((size_t)MAXT + 2)); snap_s = 0;
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
		if (has_attn[d]) {   /* the attention layers of this die: RoPE tables, the indexer scratch, the compact caches of the attended keys */
			b->rope_cs = dalloc(4 * 32 * (size_t)MAXT); b->rope_sn = dalloc(4 * 32 * (size_t)MAXT);
			CK(cuMemcpyHtoD(b->rope_cs, rcs, 4 * 32 * (size_t)MAXT)); CK(cuMemcpyHtoD(b->rope_sn, rsn, 4 * 32 * (size_t)MAXT));
			b->iqk = dalloc(4 * 640); b->iq = dalloc(4 * 512); b->sc = dalloc(4 * ((size_t)maxp / 4 + 64)); b->hst = dalloc(4 * 1280); b->sel = dalloc(4 * ((size_t)nblk_qsa + 8));
			b->kc = dalloc(kcf * 4); b->vc = dalloc(vcf * 4);
		}
		b->ids = dalloc(4 * ((size_t)MAXT + 2));
		b->imap = dalloc(4 * ((size_t)MAXT + 2)); CK(cuMemsetD32(b->imap, 0xffffffffu, (size_t)MAXT + 2));
		if (img_rows_max) b->img = dalloc(2 * 2560 * (size_t)img_rows_max);
		b->ple_pc = dalloc(160); CK(cuMemcpyHtoD(b->ple_pc, ppc, 160)); CK(cuMemHostGetDevicePointer(&b->ple_tab, c->ple_host, 0)); b->ple_compact = (unsigned)c->ple_compact;
		b->pg = dalloc(4 * 2560); b->pgq = dalloc(2 * 2560); b->pgs32 = dalloc(4 * 80); b->pgs16 = dalloc(4 * 160); b->pgas = dalloc(4 * 80);
		b->lk = dalloc(4u * (unsigned)s0.pkn); b->kn = dalloc(4 * 10240); b->qn = dalloc(4 * 10240); b->lv = dalloc(4u * (unsigned)s0.yn);
		b->pG = dalloc(4 * 10240); b->pgates = dalloc(16); b->px = dalloc(4 * 10240); b->prows = dalloc(64);
		dst_ple[d] = dalloc(4 * 10 * 10240);
		snap_ple[d] = dalloc(4 * 10 * 10240);
		b->hlo = dalloc(4u * (unsigned)s0.on); b->amo = dalloc(8); b->apart = dalloc(16 * 8); b->acnt = dalloc(4); b->tokout = dalloc(8); b->toks = dalloc(4 * ((size_t)MAXT + 2));
		b->tkout = dalloc(256); b->tkpart = dalloc(16 * 256); b->tkcnt = dalloc(4); b->rnd = dalloc(4 * ((size_t)MAXT + 2));
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
				snap_dn[d][bb] = dalloc(dn_bytes[d][bb]); snap_conv[d][bb] = dalloc(conv_bytes[d][bb]);
			} else {
				w->attn_q = dptr(d, bb, "attn_q.weight"); w->attn_q_type = ttype(bb, "attn_q.weight");
				w->attn_k = dptr(d, bb, "attn_k.weight"); w->attn_k_type = ttype(bb, "attn_k.weight");
				w->attn_v = dptr(d, bb, "attn_v.weight"); w->attn_v_type = ttype(bb, "attn_v.weight");
				w->attn_output = dptr(d, bb, "attn_output.weight"); w->attn_output_type = ttype(bb, "attn_output.weight");
				w->attn_q_norm = dptr(d, bb, "attn_q_norm.weight"); w->attn_k_norm = dptr(d, bb, "attn_k_norm.weight");
				w->indexer_x = dptr(d, bb, "indexer_x.weight"); w->indexer_qn = dptr(d, bb, "indexer.q_norm.weight"); w->indexer_kn = dptr(d, bb, "indexer.k_norm.weight");
				if (d == attn_die_of(bb)) {   /* this layer runs here: int8 K / V with fp16 scales, the pooled indexer keys, the raw key ring */
					dst_kq[d][bb] = dalloc((size_t)maxp * 512); dst_vq[d][bb] = dalloc((size_t)maxp * 512);
					dst_ks[d][bb] = dalloc((size_t)maxp * 32); dst_vs[d][bb] = dalloc((size_t)maxp * 32);
					dst_pkt[d][bb] = dalloc((size_t)maxp / 128 * 8192); dst_ring[d][bb] = dalloc(2048); snap_ring[d][bb] = dalloc(2048);
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
	free(rcs); free(rsn);
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { tab[d] = (uint64_t)xc[d < nd ? d : 0].inbox; tab[6 + d] = (uint64_t)xc[d < nd ? d : 0].flags; }
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab));
		uint32_t hdr[32] = { 0 }; hdr[3] = (uint32_t)d; hdr[4] = (uint32_t)nd;
		for (int e = 0; e < 6; e++) { uint64_t a = e < nd ? (uint64_t)xc[e].flags + 64u * (unsigned)d : (uint64_t)dCTL[d] + 0xc0; memcpy(&hdr[16 + 2 * e], &a, 8); }
		uint64_t mf = (uint64_t)xc[d].flags; memcpy(&hdr[28], &mf, 8);
		CK(cuMemsetD8(dCTL[d], 0, 0x100)); CK(cuMemcpyHtoD(dCTL[d], hdr, sizeof hdr));
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuStreamCreate(&pollst[d], CU_STREAM_NON_BLOCKING)); }
	CK(cuCtxSetCurrent(ctx[0]));
	CK(cuMemAllocHost((void **)&h_ids, 4 * ((size_t)MAXT + 2))); CK(cuMemAllocHost((void **)&h_toks, 4 * ((size_t)MAXT + 2))); CK(cuMemAllocHost((void **)&h_rnd, 4 * ((size_t)MAXT + 2)));
	CK(cuMemAllocHost((void **)&h_tok, 64)); CK(cuMemHostAlloc((void **)&h_abort, 64, CU_MEMHOSTALLOC_PORTABLE));
	in_init = 0;
	fprintf(stderr, "engine: %d dies, %d blocks, up to %d positions, %.0f MB of buffers per die, ready in %.1f s\n", nd, NB, MAXT, (double)alloc_bytes / nd / 1e6, now() - ts0);
	return 0;
}

void engine_fini(void) {
	if (!core) return;
	for (int d = 0; d < nd; d++) { if (pollst[d]) { cuCtxSetCurrent(ctx[d]); cuStreamDestroy(pollst[d]); pollst[d] = 0; } }
	cuCtxSetCurrent(ctx[0]);
	if (h_ids) cuMemFreeHost(h_ids);
	if (h_toks) cuMemFreeHost(h_toks);
	if (h_rnd) cuMemFreeHost(h_rnd);
	if (h_tok) cuMemFreeHost(h_tok);
	if (h_abort) cuMemFreeHost(h_abort);
	h_ids = NULL; h_toks = NULL; h_rnd = NULL; h_tok = NULL; h_abort = NULL;
	for (int i = 0; i < nallocs; i++) { cuCtxSetCurrent(allocs[i].c); cuMemFree(allocs[i].p); }
	free(allocs); allocs = NULL; nallocs = capallocs = 0; alloc_bytes = 0;
	for (int d = 0; d < nd; d++) { if (mod[d]) { cuCtxSetCurrent(ctx[d]); cuModuleUnload(mod[d]); mod[d] = 0; } }
	cuCtxSetCurrent(ctx[0]);
	free(snap_dn); free(snap_conv); free(snap_ids); snap_dn = snap_conv = NULL; snap_ids = NULL; snap_s = 0; memset(snap_ple, 0, sizeof snap_ple);
	free(dw); free(sp); free(dst_dn); free(dst_conv); free(dst_kq); free(dst_ks); free(dst_vq); free(dst_vs); free(dst_pkt); free(dst_ring); free(snap_ring); dw = NULL; sp = NULL; dst_dn = dst_conv = dst_kq = dst_ks = dst_vq = dst_vs = dst_pkt = dst_ring = snap_ring = NULL;
	free(s_ids); free(s_toks); s_ids = s_toks = NULL;
	free(h_imap); h_imap = NULL; imap_dirty = rope_dirty_T = 0;
	memset(db, 0, sizeof db); memset(xc, 0, sizeof xc); memset(dPROG, 0, sizeof dPROG); memset(dCTL, 0, sizeof dCTL); memset(dst_ple, 0, sizeof dst_ple);
	core = NULL;
}

/* chat streaming: one "tok <id>" line per generated token, as die 0 produces it */
static int put_token(void *arg, int t, uint32_t tok) { (void)t; FILE *out = arg; fprintf(out, "tok %u\n", tok); return fflush(out) != 0 || ferror(out); }   /* nonzero: the client went away */
/* polled while the prompt runs too: has the client closed the connection? */
static int client_gone(void *arg) {
	struct pollfd p = { .fd = fileno((FILE *)arg), .events = 0 };
	return poll(&p, 1, 0) > 0 && (p.revents & (POLLHUP | POLLERR | POLLNVAL));
}

/* One client's commands (stdin, or one connection on FLASH_SOCK); returns 1 on quit, 0 at end of input.
 *   gen <n_new> <id>...      the whole response at once: "tokens: ..." and "time: ..."
 *   chat <n_new> <id>...     streamed: "tok <id>" per generated token, then
 *                            "done <generated> <ended> <prompt> <run_ms> <first_token_ms> <resumed> <snapshot>"
 *   stop [<id> [<id>]]       end-of-turn ids for the following responses (none by default)
 *   sample <seed> <temp> <top_k> <top_p>   sampling for the following responses (greedy by default;
 *                            temp <= 0 or top_k <= 1: greedy; top_k is capped at 32)
 *   img <path> <s>           the tokens from s on are an image (see engine.h), for the next response
 *   snap <s>                 take a prefix-cache snapshot after s positions of the next response; a later response
 *                            whose prompt starts with the same s tokens resumes there instead of from 0
 *   info                     "info <positions>"
 *   cubin <path>             reload the kernel on every die
 *   reload [path]            replace this library (the weights and the PLE table stay loaded)
 *   quit                     stop the whole process */
static engine_img pend[8]; static int npend;   /* "img" commands waiting for the next response */
static char reload_path[1024];
const char *engine_reload_path(void) { return reload_path[0] ? reload_path : NULL; }

int engine_session(FILE *in, FILE *out) {
	if (!s_ids) { s_ids = calloc((size_t)MAXT + 2, 4); s_toks = calloc((size_t)MAXT + 2, 4); }
	uint32_t *ids = s_ids, *toks = s_toks;
	uint32_t stop[2] = { 0xffffffffu, 0xffffffffu };
	sampling smp = { 0, 0.f, 1.0f, 0 };   /* greedy until the client sends "sample" */
	int snap_req = 0;                     /* "snap <s>": take a prefix-cache snapshot after s positions of the next response */
	char *line = NULL; size_t cap = 0; int quit = 0;
	while (getline(&line, &cap, in) > 0) {
		char *p = line; while (*p == ' ') p++;
		if (!strncmp(p, "quit", 4)) { quit = 1; break; }
		if (!strncmp(p, "reload", 6)) {   /* the core swaps the library after this returns */
			char path[1000]; reload_path[0] = 0;
			if (sscanf(p + 6, "%999s", path) == 1) snprintf(reload_path, sizeof reload_path, "%s", path);
			free(line); return 2;
		}
		if (!strncmp(p, "prof", 4)) {   /* where the last token of the last response spent its time, from the phase timestamps CTA 0 writes at each phase end */
			uint32_t *ts = malloc(4 * 4096); CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(ts, dCTL[0] + 0x100, 4 * (size_t)prof_nph));
			static double tot[OP_COUNT]; static int cnt[OP_COUNT]; memset(tot, 0, sizeof tot); memset(cnt, 0, sizeof cnt);
			double all = 0; int big[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
			for (int i = 1; i < prof_nph; i++) {
				double us = (uint32_t)(ts[i] - ts[i - 1]) / 1e3; if (prof_op[i] >= OP_COUNT) continue;
				tot[prof_op[i]] += us; cnt[prof_op[i]]++; all += us;
				for (int k = 0; k < 8; k++) if (big[k] < 0 || us > (uint32_t)(ts[big[k]] - ts[big[k] - 1]) / 1e3) { for (int j = 7; j > k; j--) big[j] = big[j - 1]; big[k] = i; break; }
			}
			fprintf(out, "prof: %d phases, %.2f ms between the first and last timestamp\n", prof_nph, all / 1e3);
			for (int rep = 0; rep < 12; rep++) { int bi = -1; for (int o = 0; o < OP_COUNT; o++) if (cnt[o] && (bi < 0 || tot[o] > tot[bi])) bi = o; if (bi < 0 || tot[bi] <= 0) break; fprintf(out, "  %-10s %4d phases %9.1f us (%6.1f us each)\n", die_opname[bi], cnt[bi], tot[bi], tot[bi] / cnt[bi]); tot[bi] = 0; }
			fprintf(out, "  longest:"); for (int k = 0; k < 8 && big[k] > 0; k++) fprintf(out, " #%d %s %.0f us;", big[k], die_opname[prof_op[big[k]]], (uint32_t)(ts[big[k]] - ts[big[k] - 1]) / 1e3);
			fprintf(out, "\n"); fflush(out); free(ts); continue;
		}
		if (!strncmp(p, "info", 4)) { fprintf(out, "info %d\n", MAXT); fflush(out); continue; }
		if (!strncmp(p, "cubin", 5)) {
			char path[512]; if (sscanf(p + 5, "%511s", path) != 1) { fprintf(out, "error: cubin <path>\n"); fflush(out); continue; }
			for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleUnload(mod[d])); CK(cuModuleLoad(&mod[d], path)); CK(cuModuleGetFunction(&fn[d], mod[d], "die")); }
			fprintf(out, "ok: %s\n", path); fflush(out); continue;
		}
		if (!strncmp(p, "sample", 6)) {
			unsigned long long seed = 0; float temp = 0.f, topp = 1.0f; int topk = 0;
			if (sscanf(p + 6, "%llu %f %d %f", &seed, &temp, &topk, &topp) != 4) { fprintf(out, "error: sample <seed> <temp> <top_k> <top_p>\n"); fflush(out); continue; }
			smp = (sampling){ seed, temp, topp, topk };
			fprintf(out, "ok: sample seed %llu temp %g top_k %d top_p %g%s\n", seed, temp, topk, topp, temp > 0.f && topk >= 2 ? "" : " (greedy)"); fflush(out); continue;
		}
		if (!strncmp(p, "img", 3) && (p[3] == 0x20)) {   /* img <path> <first token> */
			if (npend >= 8 || sscanf(p + 3, "%239s %d", pend[npend].path, &pend[npend].s) != 2) { fprintf(out, "error: img <path> <first token> (at most 8)\n"); fflush(out); continue; }
			npend++; fprintf(out, "ok: img %d\n", npend); fflush(out); continue;
		}
		if (!strncmp(p, "snap", 4)) { snap_req = atoi(p + 4); fprintf(out, "ok: snap %d\n", snap_req); fflush(out); continue; }
		if (!strncmp(p, "stop", 4)) {
			char *e2; p += 4; stop[0] = stop[1] = 0xffffffffu;
			for (int k = 0; k < 2; k++) { long v = strtol(p, &e2, 10); if (e2 == p) break; stop[k] = (uint32_t)v; p = e2; }
			fprintf(out, "ok: stop %d %d\n", (int)stop[0], (int)stop[1]); fflush(out); continue;
		}
		int chat = !strncmp(p, "chat", 4);
		if (!chat && strncmp(p, "gen", 3)) { fprintf(out, "error: gen|chat <n_new> <id>... | stop [<id> [<id>]] | info | cubin <path> | quit\n"); fflush(out); continue; }
		p += chat ? 4 : 3; char *end; long n_new = strtol(p, &end, 10); if (end == p || n_new < 0) { fprintf(out, "error: gen|chat <n_new> <id>...\n"); fflush(out); continue; }
		int nprompt = 0; p = end;
		for (;;) { long v = strtol(p, &end, 10); if (end == p) break; if (nprompt + 2 >= MAXT) break; ids[nprompt++] = (uint32_t)v; p = end; }
		run_stats st;
		engine_cb cb = { chat ? put_token : NULL, client_gone, out };
		int rc = engine_run(ids, nprompt, (int)n_new, stop, &smp, snap_req, pend, npend, &cb, &st);
		snap_req = 0; npend = 0;
		if (rc == -3) { fprintf(out, "error: bad image data (a file, a position or too many image tokens)\n"); fflush(out); continue; }
		if (rc == -2) { fprintf(out, "error: device fault; the core rebuilds the contexts and the engine\n"); fflush(out); free(line); return 3; }
		if (rc) { fprintf(out, "error: %d prompt ids + %ld new > %d positions\n", nprompt, n_new, MAXT); fflush(out); continue; }
		if (chat) {
			fprintf(out, "done %d %d %d %.1f %.1f %d %d\n", st.ntok - (nprompt - 1), st.ended, nprompt, st.run_ms, st.first_ms, st.resumed, st.snapshot);
		} else {
			CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(toks, db[0].toks, 4 * (size_t)st.ntok));
			fprintf(out, "tokens:"); for (int t = n_new ? nprompt - 1 : 0; t < st.ntok; t++) fprintf(out, " %u", toks[t]); fprintf(out, "\n");
			fprintf(out, "time: %.1f ms for %d positions (%.2f ms per position; first token %.1f ms; %u phases; programs %.0f ms)%s\n",
			        st.run_ms, st.ntok, st.run_ms / st.ntok, st.first_ms, st.nph, st.build_ms, st.ended ? "; end of turn" : "");
		}
		fflush(out);
	}
	free(line);
	return quit;
}
