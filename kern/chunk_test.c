/* chunk_test: the prefill chunk program's attention-side (Gated DeltaNet) hyper-block of one block, split over n dies, every
 * position of a chunk in one launch (kern/prog_chunk.c), against the per-position C twin (ref/block.c deltanet_block_gpu, stage 1)
 * run position by position with its states carried. The twin quantises its activations to int8 as the decode GEMVs do and
 * the chunk's GEMMs take fp32 activations, so the comparison is a tolerance: per position, the error of S after the block
 * relative to the scale of the block's change of S.
 *
 *   chunk_test <fullcache> <blk> <npos> <ldx> <manifest0> [manifest1 ...]
 *
 * fullcache: a holder's tile cache of the whole block (hold ... cache=DIR, no split); manifest_i: the running holder of die i's
 * slice (hold ... split=i/n). Positions 0 .. npos - 1 from zero states, a random stream per position. DIE_CUBIN: the die kernel.
 * CHUNK_STAGE=0: the whole block, the MoE half on the unit layout (kern/hold2.c's assignment of the block's 5120 units over the n dies,
 * cut from the cache's expert tiles); the twin runs stage 0. Default 1: the attention-side hyper-block only. */
#define _POSIX_C_SOURCE 200809L
#include "block.h"
#include "attn.h"
#include "quant.h"
#include "hold.h"
#include "prog.h"
#include "prog_chunk.h"
#include "split.h"
#include "gguf.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
static uint32_t rs = 4242;
static float frand(void) { rs = rs * 1664525u + 1013904223u; return ((rs >> 8) & 0xffff) / 32768.f - 1.f; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

#define MAXD 6
static int nd, blk;
static const char *fullcache;
static CUcontext ctx[MAXD];
static hold_arena ar[MAXD];
static char tname[128];
static const char *tn(const char *suffix) { snprintf(tname, sizeof tname, "blk.%d.%s", blk, suffix); return tname; }
static void *hostimg(const char *suffix) {
	char fn[512]; snprintf(fn, sizeof fn, "%s/%s", fullcache, tn(suffix));
	FILE *f = fopen(fn, "rb"); if (!f) { perror(fn); exit(1); }
	fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
	void *p = malloc((size_t)n);
	if (fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "short read %s\n", fn); exit(1); }
	fclose(f);
	return p;
}
static CUdeviceptr dptr(int d, const char *suffix) { CUdeviceptr p = hold_ptr(&ar[d], tn(suffix)); if (!p) { fprintf(stderr, "die %d: no %s\n", d, tname); exit(1); } return p; }
static uint32_t ttype(int d, const char *suffix) { const hold_tensor *t = hold_find(&ar[d], tn(suffix)); if (!t) { fprintf(stderr, "no %s\n", tname); exit(1); } return t->type; }
static CUdeviceptr dalloc(size_t bytes) { CUdeviceptr p; if (!bytes) return 0; CK(cuMemAlloc(&p, bytes)); CK(cuMemsetD8(p, 0, bytes)); return p; }

/* the send phase of fp16 exchange j (the last OP_SLAB_XH before its unpack) */
static unsigned xh_phase(const prog_t *P, int j) { unsigned k = P->xr[3 * j]; while (k > 0 && P->p[k].w[0] != OP_SLAB_XH) k--; return k; }

int main(int argc, char **argv) {
	if (argc < 6) { fprintf(stderr, "usage: chunk_test <fullcache> <blk> <npos> <ldx> <manifest0> [manifest1 ...]\n"); return 2; }
	fullcache = argv[1]; blk = atoi(argv[2]); int npos = atoi(argv[3]); unsigned ldx = (unsigned)atoi(argv[4]); nd = argc - 5; float eps = 1e-6f;
	if (nd > MAXD || npos < 1 || (unsigned)npos > ldx || ldx < 128 || (ldx & (ldx - 1))) { fprintf(stderr, "bad arguments\n"); return 2; }
	CK(cuInit(0));
	CUmodule mod[MAXD]; CUfunction fn[MAXD];
	for (int d = 0; d < nd; d++) {
		FILE *f = fopen(argv[5 + d], "r"); size_t bytes; int die;
		if (!f || fscanf(f, "arena %zu die %d", &bytes, &die) != 2) { fprintf(stderr, "bad manifest %s\n", argv[5 + d]); return 1; }
		fclose(f);
		CUdevice dev; if (hold_ordinal(die) < 0) { fprintf(stderr, "die %d is not in CUDA_VISIBLE_DEVICES\n", die); return 1; } CK(cuDeviceGet(&dev, hold_ordinal(die)));
		CK(cuCtxCreate_v2(&ctx[d], CU_CTX_SCHED_SPIN, dev));
		if (hold_open(&ar[d], argv[5 + d])) return 1;
		CK(cuModuleLoad(&mod[d], getenv("DIE_CUBIN") ? getenv("DIE_CUBIN") : "die.cubin")); CK(cuModuleGetFunction(&fn[d], mod[d], "die"));
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); for (int e = 0; e < nd; e++) if (e != d) CK(cuCtxEnablePeerAccess(ctx[e], 0)); }

	/* the twin's weights (stage 1: the hc read / write and the DeltaNet) */
	dn_hweights_t hw; memset(&hw, 0, sizeof hw);
	hw.hc_attn_norm = hostimg("hc_attn_norm.weight"); hw.hc_attn_dj = hostimg("hc_attn_dj.weight"); hw.hc_attn_up = hostimg("hc_attn_up.weight");
	int attn = BLK_IS_ATTN(blk), owner = attn ? attn_die_of(blk) : -1;
	if (attn) {   /* an attention block: projections on the layer's die (ATTN_MAP / ATTN_DIE), the output projection row-split */
		if (owner < 0 || owner >= nd) { fprintf(stderr, "the attention die %d is not among the %d dies\n", owner, nd); return 2; }
		hw.attn_q = hostimg("attn_q.weight"); hw.attn_q_type = ttype(owner, "attn_q.weight");
		hw.attn_k = hostimg("attn_k.weight"); hw.attn_k_type = ttype(owner, "attn_k.weight");
		hw.attn_v = hostimg("attn_v.weight"); hw.attn_v_type = ttype(owner, "attn_v.weight");
		hw.attn_output = hostimg("attn_output.weight"); hw.attn_output_type = ttype(0, "attn_output.weight");
		hw.attn_q_norm = hostimg("attn_q_norm.weight"); hw.attn_k_norm = hostimg("attn_k_norm.weight");
	} else {
		hw.attn_qkv = hostimg("attn_qkv.weight"); hw.attn_qkv_type = ttype(0, "attn_qkv.weight");
		hw.attn_gate = hostimg("attn_gate.weight"); hw.attn_gate_type = ttype(0, "attn_gate.weight");
		hw.ssm_out = hostimg("ssm_out.weight"); hw.ssm_out_type = ttype(0, "ssm_out.weight");
		hw.ssm_ab = hostimg("ssm_ab.weight"); hw.ssm_ab_bias = hostimg("ssm_ab.bias"); hw.ssm_conv1d = hostimg("ssm_conv1d.weight");
		hw.ssm_a = hostimg("ssm_a"); hw.ssm_norm = hostimg("ssm_norm.weight");
	}
	int full = getenv("CHUNK_STAGE") && !strcmp(getenv("CHUNK_STAGE"), "0");
	if (full) {
		hw.hc_ffn_norm = hostimg("hc_ffn_norm.weight"); hw.hc_ffn_dj = hostimg("hc_ffn_dj.weight"); hw.hc_ffn_up = hostimg("hc_ffn_up.weight");
		hw.ffn_gate_inp_x = hostimg("ffn_gate_inp_x.weight");
		hw.ffn_gate_exps = hostimg("ffn_gate_exps.weight"); hw.ffn_up_exps = hostimg("ffn_up_exps.weight"); hw.ffn_down_exps = hostimg("ffn_down_exps.weight");
		hw.ffn_gate_shexp = hostimg("ffn_gate_shexp.weight"); hw.ffn_gate_shexp_type = ttype(0, "ffn_gate_shexp.weight");
		hw.ffn_up_shexp = hostimg("ffn_up_shexp.weight"); hw.ffn_up_shexp_type = ttype(0, "ffn_up_shexp.weight");
		hw.ffn_down_shexp = hostimg("ffn_down_shexp.weight"); hw.ffn_down_shexp_type = ttype(0, "ffn_down_shexp.weight");
	}
	dn_hbufs_t *hb = calloc(1, sizeof *hb); hb->dn_state = calloc(48 * 16384, 4);
	if (attn) { hb->kc = calloc(ATTN_KC_FLOATS(npos), 4); hb->vc = calloc((size_t)npos * 512, 4); }
	float *aof = attn ? calloc((size_t)6144 * npos, 4) : NULL;   /* the twin's attention output (24 heads x 256) per position */
	float *qgf = attn ? calloc((size_t)12288 * npos, 4) : NULL, *kf = attn ? calloc((size_t)512 * npos, 4) : NULL, *vf = attn ? calloc((size_t)512 * npos, 4) : NULL;

	/* the chunk's input streams (a column each) and the twin's streams after the block */
	float *S0 = calloc((size_t)10240 * ldx, 4), *Sref = calloc((size_t)10240 * ldx, 4);
	for (int p = 0; p < npos; p++) for (int r = 0; r < 10240; r++) S0[(size_t)r * ldx + p] = 0.5f * frand();
	/* the MoE half's intermediates per position (stage 0): z, the router logits, the top ten, the MoE sum */
	float *zf = calloc((size_t)2560 * npos, 4), *rf = calloc((size_t)513 * npos, 4), *of = calloc((size_t)2560 * npos, 4);
	unsigned *xf = calloc((size_t)10 * npos, 4);
	float *sgf = calloc((size_t)640 * npos, 4), *suf = calloc((size_t)640 * npos, 4), *sdf = calloc((size_t)2560 * npos, 4), *swf = calloc((size_t)npos, 4);
	double t0 = now();
	for (int p = 0; p < npos; p++) {
		for (int r = 0; r < 10240; r++) hb->S[r] = S0[(size_t)r * ldx + p];
		moe_dbg_shd = sdf + (size_t)2560 * p;
		if (attn) {
			attn_block_gpu(&hw, hb, (unsigned)p, eps, nd, blk, full ? 0 : 1); memcpy(aof + (size_t)6144 * p, hb->aout, 4 * 6144);
			memcpy(qgf + (size_t)12288 * p, hb->qg, 4 * 12288); memcpy(kf + (size_t)512 * p, hb->k, 4 * 512); memcpy(vf + (size_t)512 * p, hb->v, 4 * 512);
		}
		else deltanet_block_gpu(&hw, hb, (unsigned)p, eps, nd, blk, full ? 0 : 1);
		memcpy(sgf + (size_t)640 * p, hb->g + 6400, 4 * 640); memcpy(suf + (size_t)640 * p, hb->gu + 6400, 4 * 640); swf[p] = hb->s;
		for (int r = 0; r < 10240; r++) Sref[(size_t)r * ldx + p] = hb->S[r];
		memcpy(zf + (size_t)2560 * p, hb->z, 4 * 2560); memcpy(rf + (size_t)513 * p, hb->r, 4 * 513);
		memcpy(of + (size_t)2560 * p, hb->out, 4 * 2560); memcpy(xf + (size_t)10 * p, hb->idx, 4 * 10);
	}
	printf("twin: %d positions in %.1f s\n", npos, now() - t0);

	/* the unit layout of the block (kern/hold2.c assign_units, frac 0.45 on the shared-expert die): unit u = 10 e + j, its die */
	/* ebatch takes at most 3 units of an expert per die, as hold2 hands out over 6 dies: with fewer dies (1, 2, 3) the units
	   are assigned over 6 virtual dies and die d runs the lists of the virtual dies v = d mod n in turn */
	static uint8_t assign[5120]; int units_of[MAXD] = { 0 };
	int vd = nd < 6 && 6 % nd == 0 ? 6 : nd;
	if (full) {
		double w[MAXD], sum = 0, x[MAXD], rem[MAXD]; long q[MAXD], qs = 0, cur[MAXD] = { 0 };
		for (int d = 0; d < vd; d++) { w[d] = (double)(split_bound(512, d + 1, vd) - split_bound(512, d, vd)) * (d == blk % vd ? 0.45 : 1.0); sum += w[d]; }
		for (int d = 0; d < vd; d++) { x[d] = 5120 * w[d] / sum; q[d] = (long)x[d]; rem[d] = x[d] - (double)q[d]; qs += q[d]; }
		while (qs < 5120) { int best = 0; for (int d = 1; d < vd; d++) if (rem[d] > rem[best]) best = d; q[best]++; rem[best] = -1; qs++; }
		for (int u = 0; u < 5120; u++) { int best = 0; for (int d = 0; d < vd; d++) { cur[d] += q[d]; if (cur[d] > cur[best]) best = d; } cur[best] -= 5120; assign[u] = (uint8_t)best; units_of[best]++; }
	}
	/* per die: weights, states, slabs, the chunk inbox (two slots) with the decode's flags and counter */
	dn_weights_t dw[MAXD]; dn_bufs_t db[MAXD]; xchg_t xc[MAXD]; split_t sp[MAXD]; chunk_bufs_t cb[MAXD];
	memset(cb, 0, sizeof cb);
	chunk_attn_t ca[MAXD]; memset(ca, 0, sizeof ca);
	CUdeviceptr dPROG[MAXD], dCTL[MAXD];
	unsigned cslot = (unsigned)CHUNK_SLOT_BYTES(ldx);
	size_t row = 4u * (size_t)ldx;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		split_of(&sp[d], d, nd, blk);
		const split_t *s = &sp[d];
		dn_weights_t *w = &dw[d]; memset(w, 0, sizeof *w);
		w->hc_attn_norm = dptr(d, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, "hc_attn_up.weight");
		if (attn) {
			/* the queries split by whole heads over the dies (all of them work: fewer than 3), copied from the attention die as
			   split_attention_queries does; K / V / the indexer on the attention die */
			const hold_tensor *q = hold_find(&ar[owner], tn("attn_q.weight"));
			if (!q) { fprintf(stderr, "no %s on die %d\n", tname, owner); return 1; }
			unsigned first = 24u * (unsigned)d / (unsigned)nd * 512u, rows = 24u * (unsigned)(d + 1) / (unsigned)nd * 512u - first;
			CUdeviceptr src = ar[owner].base + q->off + (size_t)first * q->stride;
			w->attn_q_split = 1; w->attn_q_first = first; w->attn_q_rows = rows; w->attn_q_type = hw.attn_q_type;
			if (d == owner) w->attn_q = src;
			else { w->attn_q = dalloc((size_t)rows * q->stride); CK(cuMemcpyPeer(w->attn_q, ctx[d], src, ctx[owner], (size_t)rows * q->stride)); }
			w->attn_q_norm = dptr(d, "attn_q_norm.weight"); w->attn_k_norm = dptr(d, "attn_k_norm.weight");
			w->attn_output = dptr(d, "attn_output.weight"); w->attn_output_type = hw.attn_output_type;
			if (d == owner) {
				w->attn_k = dptr(d, "attn_k.weight"); w->attn_k_type = hw.attn_k_type; w->attn_v = dptr(d, "attn_v.weight"); w->attn_v_type = hw.attn_v_type;
				w->indexer_x = dptr(d, "indexer_x.weight"); w->indexer_qn = dptr(d, "indexer.q_norm.weight"); w->indexer_kn = dptr(d, "indexer.k_norm.weight");
			}
		} else {
			w->attn_qkv = dptr(d, "attn_qkv.weight"); w->attn_qkv_type = hw.attn_qkv_type;
			w->attn_gate = dptr(d, "attn_gate.weight"); w->attn_gate_type = hw.attn_gate_type;
			w->ssm_out = dptr(d, "ssm_out.weight"); w->ssm_out_type = hw.ssm_out_type;
			w->ssm_ab = dptr(d, "ssm_ab.weight"); w->ssm_ab_bias = dptr(d, "ssm_ab.bias");
			w->ssm_conv1d = dptr(d, "ssm_conv1d.weight"); w->ssm_a = dptr(d, "ssm_a"); w->ssm_norm = dptr(d, "ssm_norm.weight");
		}
		if (full) {
			w->hc_ffn_norm = dptr(d, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, "hc_ffn_up.weight");
			w->ffn_gate_inp_x = dptr(d, "ffn_gate_inp_x.weight");
			w->ffn_gate_shexp = dptr(d, "ffn_gate_shexp.weight"); w->ffn_gate_shexp_type = hw.ffn_gate_shexp_type;
			w->ffn_up_shexp = dptr(d, "ffn_up_shexp.weight"); w->ffn_up_shexp_type = hw.ffn_up_shexp_type;
			w->ffn_down_shexp = dptr(d, "ffn_down_shexp.weight"); w->ffn_down_shexp_type = hw.ffn_down_shexp_type;
			if (!getenv("CHUNK_SHEXP1")) {   /* the split shared expert (CHUNK_SHEXP1: the whole one on its die) */
				char nm[64];
				snprintf(nm, sizeof nm, "blk.%d.ffn_gate_shexp_s.weight", blk); w->ffn_gate_shexp_s = hold_ptr(&ar[d], nm);
				snprintf(nm, sizeof nm, "blk.%d.ffn_up_shexp_s.weight", blk); w->ffn_up_shexp_s = hold_ptr(&ar[d], nm);
				snprintf(nm, sizeof nm, "blk.%d.ffn_down_shexp_s.weight", blk); w->ffn_down_shexp_s = hold_ptr(&ar[d], nm);
				if (!w->ffn_gate_shexp_s || !w->ffn_up_shexp_s || !w->ffn_down_shexp_s) w->ffn_gate_shexp_s = w->ffn_up_shexp_s = w->ffn_down_shexp_s = 0;
			}
			/* this die's units in (e, j) order: gate / up rows 64 j .. of expert e (tile rows of 720 B), down block j of each of the
			   expert's 2560 rows (K 640 tile rows of 192 B: scale at 2 j, codes at 32 + 16 j) as 2560 x 16 B codes then 2560 scales;
			   one list per virtual die v = d mod n */
			for (int v = d; v < vd; v += nd) {
				size_t gb = (size_t)units_of[v] * 64 * 720, dbs = (size_t)units_of[v] * 46080;
				uint8_t *ug = malloc(gb), *uu = malloc(gb), *ud = malloc(dbs); uint32_t emap[513]; int slot = 0;
				for (int e = 0; e < 512; e++) {
					int f0 = slot;
					for (int j = 0; j < 10; j++) {
						if (assign[10 * e + j] != v) continue;
						size_t src = ((size_t)e * 640 + 64u * (unsigned)j) * 720;
						memcpy(ug + (size_t)slot * 64 * 720, (const uint8_t *)hw.ffn_gate_exps + src, 64 * 720);
						memcpy(uu + (size_t)slot * 64 * 720, (const uint8_t *)hw.ffn_up_exps + src, 64 * 720);
						uint8_t *un = ud + (size_t)slot * 46080;
						for (int r = 0; r < 2560; r++) {
							const uint8_t *row = (const uint8_t *)hw.ffn_down_exps + ((size_t)e * 2560 + r) * 192;
							memcpy(un + 16 * r, row + 32 + 16 * j, 16); memcpy(un + 40960 + 2 * r, row + 2 * j, 2);
						}
						slot++;
					}
					emap[e] = ((uint32_t)f0 << 8) | (uint32_t)(slot - f0);
				}
				emap[512] = 0;
				CUdeviceptr g = dalloc(gb), u = dalloc(gb), dn = dalloc(dbs), em = dalloc(4 * 513);
				CK(cuMemcpyHtoD(g, ug, gb)); CK(cuMemcpyHtoD(u, uu, gb)); CK(cuMemcpyHtoD(dn, ud, dbs)); CK(cuMemcpyHtoD(em, emap, 4 * 513));
				free(ug); free(uu); free(ud);
				if (vd == nd) { w->ffn_gate_exps = g; w->ffn_up_exps = u; w->ffn_down_exps = dn; w->emap = em; }
				else { unsigned k = cb[d].nvu++; cb[d].vu_g[k] = g; cb[d].vu_u[k] = u; cb[d].vu_d[k] = dn; cb[d].vu_emap[k] = em; }
			}
		}
		unsigned nloc = SPLIT_NLOC(s), cols = 32u * (unsigned)s->cbn, hrows = 128u * (unsigned)s->hn;
		dn_bufs_t *b = &db[d]; memset(b, 0, sizeof *b);
		b->conv_state = dalloc(4 * 4 * nloc); b->dn_state = dalloc((size_t)s->hn * 16384 * 4);
		chunk_bufs_t *c = &cb[d]; c->ldx = ldx;
		c->S = dalloc(row * 10240); c->XH = dalloc(row * 10240); c->LDN = dalloc(row * (unsigned)s->djn); c->T = dalloc(row * 324);
		c->LUP = dalloc(row * 4 * cols); c->LZ = dalloc(row * cols); c->LQKV = dalloc(row * nloc); c->LU = dalloc(row * nloc);
		c->LGATE = dalloc(row * hrows); c->LO = dalloc(row * hrows); c->LON = dalloc(row * hrows); c->LA = dalloc(row * 2 * (unsigned)s->hn); c->LY = dalloc(row * (unsigned)s->yn);
		c->q6t_rows = 4u * (unsigned)s->yn > 1100u ? 4u * (unsigned)s->yn : 1100u; c->Q6T = dalloc(row * c->q6t_rows);
		if (full) {
			c->ZT = dalloc(row * 2560); c->R = dalloc(row * 513); c->IDX = dalloc(row * 10); c->RW = dalloc(row * 10); c->SW = dalloc(row);
			c->LP = dalloc(row * 10); c->LW = dalloc(row * 10); c->ITEMS = dalloc(512 * 16); c->NITEMS = dalloc(16);
			c->G = dalloc(row * 10 * 192); c->U = dalloc(row * 10 * 192); c->H = dalloc(row * 10 * 192); c->PART = dalloc(row * 2560);
			c->SG = dalloc(row * 640); c->SU = dalloc(row * 640); c->SH = dalloc(row * 640); c->SD = dalloc(row * 2560);
		}
		if (attn) {   /* the attention ops' slabs, the RoPE rows (row p + 3 = position p), the caches as the engine sizes them */
			unsigned nh = w->attn_q_rows / 512u, h0 = w->attn_q_first / 512u, maxp = ((unsigned)npos + 127u) & ~127u;
			c->LQ = dalloc(row * 512 * nh); c->LAO = dalloc(row * 256 * nh); c->LK = dalloc(row * 512); c->LV = dalloc(row * 512); c->LI = dalloc(row * 640);
			c->CB = dalloc(CHUNK_CB_BYTES(ldx)); c->IQ = dalloc(row * 512); c->CS = dalloc(128 * ((size_t)ldx + 3)); c->SN = dalloc(128 * ((size_t)ldx + 3));
			float *cs = calloc(64 * ((size_t)ldx + 3), 4), *sn = cs + 32 * ((size_t)ldx + 3);
			for (unsigned i = 3; i < (unsigned)npos + 3; i++) for (int pr = 0; pr < 32; pr++) {
				double ang = (double)(i - 3) * pow((double)1e7f, -2.0 * pr / 64.0); cs[32 * i + pr] = (float)cos(ang); sn[32 * i + pr] = (float)sin(ang);
			}
			CK(cuMemcpyHtoD(c->CS, cs, 128 * ((size_t)ldx + 3))); CK(cuMemcpyHtoD(c->SN, sn, 128 * ((size_t)ldx + 3))); free(cs);
			chunk_attn_t *a = &ca[d]; memset(a, 0, sizeof *a); a->nblk = QSA_NBLK;
			if (d == owner) {
				b->kq = dalloc((size_t)maxp * 512); b->vq = dalloc((size_t)maxp * 512); b->ks = dalloc((size_t)maxp * 32); b->vs = dalloc((size_t)maxp * 32);
				b->pkt = dalloc((size_t)maxp / 128 * 8192); b->ring = dalloc(2048); a->POOL = dalloc((size_t)128 * 8192 * 4);
			}
			for (unsigned hk = 0; hk < 2; hk++) if (nh && h0 + nh > 12u * hk && h0 < 12u * (hk + 1)) { a->R[hk] = dalloc((size_t)maxp * 512); a->RS[hk] = dalloc((size_t)maxp * 64); }
		}
		CK(cuMemcpyHtoD(c->S, S0, row * 10240));
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * (size_t)cslot); x->flags = dalloc(512); x->xtab = dalloc(128);
		uint32_t fl[96]; for (int e = 0; e < 96; e++) fl[e] = e / 16 < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		dPROG[d] = dalloc(128 * 512); dCTL[d] = dalloc(0x100 + 4 * 512);
		x->xgen = dCTL[d] + 0x20;
		/* CHUNK_XH=0: the f32 exchanges; CHUNK_RELAY=1: the fp16 exchanges' card relays (dies 2k, 2k + 1 a card) */
		x->unp[0] = x->unp[1] = 0; x->pinbox = 0; x->xd = d; x->xnd = nd; x->relay = getenv("CHUNK_RELAY") && atoi(getenv("CHUNK_RELAY"));
		if (!getenv("CHUNK_XH") || atoi(getenv("CHUNK_XH"))) { x->unp[0] = dalloc(cslot); x->unp[1] = dalloc(cslot); }
	}
	/* the phase beacon of a DIE_BEACON=1 cubin: beacon[d][cta] = the phase the CTA last dispatched (ctl[12..13]) */
	uint32_t *hbeacon; CK(cuMemHostAlloc((void **)&hbeacon, 64 * MAXD * 4, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE)); memset(hbeacon, 0xff, 64 * MAXD * 4);
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { tab[d] = (uint64_t)xc[d < nd ? d : 0].inbox; tab[6 + d] = (uint64_t)xc[d < nd ? d : 0].flags; }
	for (int d = 0; d < nd; d++) xc[d].pinbox = (d ^ 1) < nd ? xc[d ^ 1].inbox : 0;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(xc[d].xtab, tab, sizeof tab));
		uint32_t hdr[32] = { 0 }; hdr[3] = (uint32_t)d; hdr[4] = (uint32_t)nd;
		for (int e = 0; e < 6; e++) { uint64_t a = e < nd ? (uint64_t)xc[e].flags + 64u * (unsigned)d : (uint64_t)dCTL[d] + 0xc0; memcpy(&hdr[16 + 2 * e], &a, 8); }
		uint64_t mf = (uint64_t)xc[d].flags; memcpy(&hdr[28], &mf, 8);
		CUdeviceptr bp; CK(cuMemHostGetDevicePointer(&bp, hbeacon, 0)); bp += 64u * 4u * (unsigned)d; memcpy(&hdr[12], &bp, 8);
		CK(cuMemsetD8(dCTL[d], 0, 0x100)); CK(cuMemcpyHtoD(dCTL[d], hdr, sizeof hdr));
	}
	for (int d = 0; d < nd; d++) printf("die %d: dj %d+%d cb %d+%d h %d+%d k %d..%d y %d+%d\n", d, sp[d].dj0, sp[d].djn, sp[d].cb0, sp[d].cbn, sp[d].h0, sp[d].hn, sp[d].klo, sp[d].khi, sp[d].y0, sp[d].yn);

	prog_t P[MAXD]; unsigned nph[MAXD];
	for (int d = 0; d < nd; d++) {
		prog_init(&P[d]);
		if (attn) prog_chunk_attn_half(&P[d], &sp[d], &xc[d], &dw[d], &db[d], &cb[d], &ca[d], 0, (unsigned)npos, eps, cslot);
		else prog_chunk_deltanet_half(&P[d], &sp[d], &xc[d], &dw[d], &db[d], &cb[d], 0, (unsigned)npos, eps, cslot);
		if (full) prog_chunk_moe_half(&P[d], &sp[d], &xc[d], &dw[d], &db[d], &cb[d], (unsigned)npos, eps, cslot);
		nph[d] = (unsigned)P[d].n;
		if (getenv("CHUNK_NPH") && (unsigned)atoi(getenv("CHUNK_NPH")) < nph[d]) nph[d] = (unsigned)atoi(getenv("CHUNK_NPH"));
		if (getenv("CHUNK_STOP_APP")) for (int i = 0; i < P[d].n; i++) if (P[d].p[i].w[0] == OP_CATTN_APP) nph[d] = (unsigned)i + 1;   /* debugging: end after the appends */
	}
	prog_chunk_xh_patch(P, xc, nd);
	/* CHUNK_XHDBG=j (debugging): end after fp16 exchange j's unpack, then compare every die's unpacked slot with the senders' rows */
	int xhdbg = getenv("CHUNK_XHDBG") ? atoi(getenv("CHUNK_XHDBG")) : -1;
	int xhstop = getenv("CHUNK_XHSTOP") != NULL;   /* with CHUNK_XHDBG: end after the send instead, and compare the fp16 inboxes */
	if (xhdbg >= 0) for (int d = 0; d < nd; d++) if (xhdbg < P[d].nxr) nph[d] = P[d].xr[3 * xhdbg] + (xhstop ? 0 : 1);
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * P[d].n)); }
	double t1 = now();
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		unsigned ntok = 1; void *params[] = { &dPROG[d], &nph[d], &dCTL[d], &ntok };
		CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	}
	/* wait CHUNK_WAIT seconds at most (default 60); then the beacons say where every CTA of every die is */
	double limit = getenv("CHUNK_WAIT") ? atof(getenv("CHUNK_WAIT")) : 60.0;
	for (int left = nd; left; ) {
		left = 0;
		int fault = 0;
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CUresult q = cuStreamQuery(0); if (q == CUDA_ERROR_NOT_READY) left++; else if (q != CUDA_SUCCESS) { printf("chunk: die %d failed (%d)\n", d, (int)q); fault = 1; } }
		if (fault || (left && now() - t1 > limit)) {
			if (!fault) printf("chunk: not done after %.0f s; beacons:\n", limit);
			for (int d = 0; d < nd; d++) {
				printf(" die %d:", d);
				for (int c = 0; c < 16; c++) { uint32_t at = hbeacon[64 * d + c]; printf(" %u", at); }
				uint32_t at = hbeacon[64 * d];
				printf("  (cta 0 at %s)\n", at < (uint32_t)P[d].n && P[d].p[at].w[0] < OP_COUNT ? die_opname[P[d].p[at].w[0]] : "?");
			}
			fflush(stdout); _exit(5);
		}
	}
	double t2 = now();
	if (xhdbg >= 0 && xhdbg < P[0].nxr) {
		unsigned par = 0, end = 0;   /* the exchange's parity: the xh phase's slot */
		for (int e = 0; e < nd; e++) { unsigned b = P[e].xr[3 * xhdbg + 1] + P[e].xr[3 * xhdbg + 2]; if (b > end) end = b; }
		const phase_t *xp = &P[0].p[xh_phase(&P[0], xhdbg)]; par = xp->w[2 + (0x158 - 0x140) / 4] ? 1 : 0;
		float *ref = calloc(end / 4, 4), *gotu = malloc(end);
		for (int e = 0; e < nd; e++) {   /* the sender's source rows: the xh phase's SRC */
			const phase_t *sp2 = &P[e].p[xh_phase(&P[e], xhdbg)];
			CUdeviceptr src = (CUdeviceptr)sp2->w[2 + 2] | (CUdeviceptr)sp2->w[2 + 3] << 32;
			CK(cuCtxSetCurrent(ctx[e])); CK(cuMemcpyDtoH((char *)ref + P[e].xr[3 * xhdbg + 1], src, P[e].xr[3 * xhdbg + 2]));
		}
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			if (xhstop) {   /* the inbox's fp16 halves as floats */
				uint16_t *hh = malloc(end / 2); CK(cuMemcpyDtoH(hh, xc[d].inbox + (CUdeviceptr)par * cslot, end / 2));
				for (size_t i = 0; i < end / 4; i++) { int e = (hh[i] >> 10) & 31, m = hh[i] & 1023; double v = e == 31 ? NAN : e ? ldexp(1024 + m, e - 25) : ldexp(m, -24); gotu[i] = (float)((hh[i] & 0x8000) ? -v : v); }
				free(hh);
			} else CK(cuMemcpyDtoH(gotu, xc[d].unp[par], end));
			double w = 0, m = 1e-30; size_t at = 0, nbad = 0;
			for (size_t i = 0; i < end / 4; i++) { double e = isnan(gotu[i]) ? INFINITY : fabs(gotu[i] - ref[i]); m = fmax(m, fabs(ref[i])); if (e > w) { w = e; at = i; } if (e > 1e-2 * fabs(ref[i]) + 1e-6) nbad++; }
			printf("xh %d die %d: %u bytes (parity %u), worst %.3g of max %.3g at float %zu (gpu %.7g src %.7g), %zu off\n", xhdbg, d, end, par, w, m, at, gotu[at], ref[at], nbad);
			/* the off floats as runs of 16 B words */
			long run0 = -1, nr = 0;
			for (size_t i = 0; i <= end / 4 && nr < 24; i++) {
				int off = i < end / 4 && !(fabs(gotu[i] - ref[i]) <= 1e-2 * fabs(ref[i]) + 1e-6);
				if (off && run0 < 0) run0 = (long)i;
				if (!off && run0 >= 0) { printf("    off floats %ld..%zu (words %ld..%zu)\n", run0, i - 1, run0 / 4, (i - 1) / 4); run0 = -1; nr++; }
			}
		}
		free(ref); free(gotu);
	}
	printf("chunk: %u phases per die, %d positions in %.2f ms (%.1f us per position)\n", nph[0], npos, (t2 - t1) * 1e3, (t2 - t1) * 1e6 / npos);

	int bad = 0;
	float *got = malloc(row * 10240);
	if (full) for (int d = 0; d < nd; d++) {   /* where the MoE half first departs from the twin */
		CK(cuCtxSetCurrent(ctx[d]));
		float *zt = malloc(row * 2560), *rr = malloc(row * 513), *ly = malloc(row * (unsigned)sp[d].yn); unsigned *ix = malloc(row * 10);
		CK(cuMemcpyDtoH(zt, cb[d].ZT, row * 2560)); CK(cuMemcpyDtoH(rr, cb[d].R, row * 513)); CK(cuMemcpyDtoH(ix, cb[d].IDX, row * 10));
		CK(cuMemcpyDtoH(ly, cb[d].LY, row * (unsigned)sp[d].yn));
		double ez = 0, er = 0, eo = 0; int pz = 0, pr = 0, po = 0, nset = 0, first = -1;
		for (int p = 0; p < npos; p++) {
			double mz = 1e-30, mr = 1e-30, mo = 1e-30, e1 = 0, e2 = 0, e3 = 0;
			for (int r = 0; r < 2560; r++) { mz = fmax(mz, fabs(zf[(size_t)2560 * p + r])); e1 = fmax(e1, fabs(zt[(size_t)2560 * p + r] - zf[(size_t)2560 * p + r])); }
			for (int r = 0; r < 513; r++) { mr = fmax(mr, fabs(rf[(size_t)513 * p + r])); e2 = fmax(e2, fabs(rr[(size_t)r * ldx + p] - rf[(size_t)513 * p + r])); }
			for (int r = 0; r < sp[d].yn; r++) { double v = of[(size_t)2560 * p + sp[d].y0 + r]; mo = fmax(mo, fabs(v)); e3 = fmax(e3, fabs(ly[(size_t)r * ldx + p] - v)); }
			if (e1 / mz > ez) { ez = e1 / mz; pz = p; }
			if (e2 / mr > er) { er = e2 / mr; pr = p; }
			if (e3 / mo > eo) { eo = e3 / mo; po = p; }
			int miss = 0;
			for (int k = 0; k < 10; k++) { int hit = 0; for (int j = 0; j < 10; j++) hit |= ix[(size_t)10 * p + j] == xf[(size_t)10 * p + k]; miss += !hit; }
			if (miss) { nset++; if (first < 0) first = p; }
		}
		printf("die %d moe: z worst %.3g of max (position %d); router logits worst %.3g (position %d); top-ten sets differ at %d positions (first %d",
		       d, ez, pz, er, pr, nset, first);
		if (first >= 0) { printf(": gpu"); for (int k = 0; k < 10; k++) printf(" %u", ix[(size_t)10 * first + k]); printf(" twin"); for (int k = 0; k < 10; k++) printf(" %u", xf[(size_t)10 * first + k]); }
		printf("); MoE sum rows %d+%d worst %.3g of max (position %d)\n", sp[d].y0, sp[d].yn, eo, po);
		free(zt); free(rr); free(ly); free(ix);
	}
	if (full) {   /* the shared expert on its die, and the partials summed over the dies (before the reduce-scatter) */
		float *a = malloc(row * 2560), *sum = calloc((size_t)2560 * ldx, 4);
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d]));
			if (sp[d].shexp) {
				const char *nm[4] = { "gate", "up", "down", "gate weight" }; const float *ref[4] = { sgf, suf, sdf, swf }; CUdeviceptr src[4] = { cb[d].SG, cb[d].SU, cb[d].SD, cb[d].SW };
				unsigned n[4] = { 640, 640, 2560, 1 };
				for (int k = 0; k < 4; k++) {
					CK(cuMemcpyDtoH(a, src[k], row * n[k]));
					double worst = 0; int wp = 0;
					for (int p = 0; p < npos; p++) {
						double m = 1e-30, e = 0;
						for (unsigned r = 0; r < n[k]; r++) { double v = ref[k][(size_t)n[k] * p + r]; m = fmax(m, fabs(v)); e = fmax(e, fabs(a[(size_t)r * ldx + p] - v)); }
						if (e / m > worst) { worst = e / m; wp = p; }
					}
					printf("die %d shared expert %s: worst %.3g of max (position %d: gpu %.7g twin %.7g)\n", d, nm[k], worst, wp, a[wp], ref[k][(size_t)n[k] * wp]);
				}
			}
			int32_t *pi = malloc(row * 2560); CK(cuMemcpyDtoH(pi, cb[d].PART, row * 2560));   /* int32 fixed point, value 2^20 */
			for (size_t i = 0; i < (size_t)2560 * ldx; i++) sum[i] += (float)((double)pi[i] / 1048576.0);
			free(pi);
		}
		double worst = 0; int wp = 0, wr = 0;
		for (int p = 0; p < npos; p++) {
			double m = 1e-30, e = 0; int er = 0;
			for (int r = 0; r < 2560; r++) { double v = of[(size_t)2560 * p + r]; m = fmax(m, fabs(v)); double x = fabs(sum[(size_t)r * ldx + p] - v); if (x > e) { e = x; er = r; } }
			if (e / m > worst) { worst = e / m; wp = p; wr = er; }
		}
		printf("partials summed over the dies: worst %.3g of max (position %d row %d: gpu %.7g twin %.7g)\n", worst, wp, wr, sum[(size_t)wr * ldx + wp], of[(size_t)2560 * wp + wr]);
		free(a); free(sum);
	}
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(got, cb[d].S, row * 10240));
		double worst = 0, sum = 0; int wp = 0, wr = 0, nbad = 0;
		for (int p = 0; p < npos; p++) {
			double scale = 0;   /* the block's change of S at this position */
			for (int r = 0; r < 10240; r++) { double ch = fabs((double)Sref[(size_t)r * ldx + p] - S0[(size_t)r * ldx + p]); if (ch > scale) scale = ch; }
			double e = 0;
			for (int r = 0; r < 10240; r++) {
				double g = got[(size_t)r * ldx + p], v = Sref[(size_t)r * ldx + p], x = isnan(g) ? INFINITY : fabs(g - v) / (scale + 1e-30);
				if (x > e) { e = x; if (e > worst) { worst = e; wp = p; wr = r; } }
			}
			sum += e; if (e > 3e-2) nbad++;
		}
		printf("die %d S: worst position error %.3g of the block's change (position %d row %d: gpu %.7g twin %.7g input %.7g), mean %.3g; %d positions beyond 3e-2\n",
		       d, worst, wp, wr, got[(size_t)wr * ldx + wp], Sref[(size_t)wr * ldx + wp], S0[(size_t)wr * ldx + wp], sum / npos, nbad);
		bad += nbad;
		if (attn) {   /* the die's heads' attention output against the twin's */
			unsigned nh = dw[d].attn_q_rows / 512u, h0 = dw[d].attn_q_first / 512u;
			{   /* the projections: the die's query rows; on the attention die K and V */
				const char *nm[3] = { "q", "k", "v" }; CUdeviceptr src[3] = { cb[d].LQ, d == owner ? cb[d].LK : 0, d == owner ? cb[d].LV : 0 };
				unsigned n[3] = { 512 * nh, 512, 512 }, stride[3] = { 12288, 512, 512 }, off[3] = { 512 * h0, 0, 0 }; const float *ref[3] = { qgf, kf, vf };
				for (int k = 0; k < 3; k++) {
					if (!src[k]) continue;
					float *g = malloc(row * n[k]); CK(cuMemcpyDtoH(g, src[k], row * n[k]));
					double worst = 0; int wp = 0, wr = 0;
					for (int p = 0; p < npos; p++) {
						double m = 1e-30, e = 0; int er = 0;
						for (unsigned r = 0; r < n[k]; r++) { double v = ref[k][(size_t)stride[k] * p + off[k] + r]; m = fmax(m, fabs(v)); double x = fabs(g[(size_t)r * ldx + p] - v); if (x > e) { e = x; er = (int)r; } }
						if (e / m > worst) { worst = e / m; wp = p; wr = er; }
					}
					printf("die %d projection %s: worst %.3g of max (position %d row %d: gpu %.7g twin %.7g)\n", d, nm[k], worst, wp, wr, g[(size_t)wr * ldx + wp], ref[k][(size_t)stride[k] * wp + off[k] + wr]);
					free(g);
				}
			}
			if (d == owner) {   /* the owner's decode caches (int8, fp16 scales) and its chunk exchange buffer (CKV u8 = int8 + 128, CKS f32) */
				int8_t *kq = malloc((size_t)npos * 512), *vq = malloc((size_t)npos * 512); uint16_t *ks = malloc((size_t)npos * 32), *vs = malloc((size_t)npos * 32);
				uint8_t *ckv = malloc((size_t)npos * 1024); float *cks = malloc((size_t)npos * 128);
				CK(cuMemcpyDtoH(kq, db[d].kq, (size_t)npos * 512)); CK(cuMemcpyDtoH(vq, db[d].vq, (size_t)npos * 512));
				CK(cuMemcpyDtoH(ks, db[d].ks, (size_t)npos * 32)); CK(cuMemcpyDtoH(vs, db[d].vs, (size_t)npos * 32));
				CK(cuMemcpyDtoH(ckv, cb[d].CB, (size_t)npos * 1024)); CK(cuMemcpyDtoH(cks, cb[d].CB + 1024ull * npos, (size_t)npos * 128));
				double w4[4] = { 0 }; int p4[4] = { 0 }, j4[4] = { 0 }, h4[4] = { 0 };
				for (int p = 0; p < npos; p++) for (int hk = 0; hk < 2; hk++) for (int j = 0; j < 256; j++) {
					double kt = hb->kc[attn_kidx((unsigned)p, hk, j)], vt = hb->vc[(size_t)p * 512 + 256 * hk + j];
					double g[4] = { kq[(size_t)p * 512 + 256 * hk + j] * (double)fp16_to_fp32(ks[(size_t)p * 16 + 8 * hk + j / 32]),
					                vq[(size_t)p * 512 + 256 * hk + j] * (double)fp16_to_fp32(vs[(size_t)p * 16 + 8 * hk + j / 32]),
					                ((int)ckv[(size_t)p * 1024 + 512 * hk + j] - 128) * (double)cks[(size_t)p * 32 + 16 * hk + j / 32],
					                ((int)ckv[(size_t)p * 1024 + 512 * hk + 256 + j] - 128) * (double)cks[(size_t)p * 32 + 16 * hk + 8 + j / 32] };
					double t4[4] = { kt, vt, kt, vt };
					for (int k = 0; k < 4; k++) { double e = fabs(g[k] - t4[k]) / 1.0; if (e > w4[k]) { w4[k] = e; p4[k] = p; j4[k] = j; h4[k] = hk; } }
				}
				const char *nm[4] = { "decode K", "decode V", "exchange K", "exchange V" };
				for (int k = 0; k < 4; k++) printf("die %d %s: worst abs error %.3g (position %d head %d dim %d)\n", d, nm[k], w4[k], p4[k], h4[k], j4[k]);
				free(kq); free(vq); free(ks); free(vs); free(ckv); free(cks);
			}
			if (getenv("CHUNK_STOP_APP")) {   /* the program ended after the appends: the received exchange (slot 0) against the owner's buffer */
				size_t nb = (size_t)CHUNK_CB_BYTES(npos); uint8_t *a = malloc(nb), *o = malloc(nb);
				CK(cuMemcpyDtoH(a, xc[d].inbox, nb)); CK(cuCtxSetCurrent(ctx[owner])); CK(cuMemcpyDtoH(o, cb[owner].CB, nb)); CK(cuCtxSetCurrent(ctx[d]));
				size_t diff = 0, first = (size_t)-1; for (size_t i = 0; i < nb; i++) if (a[i] != o[i]) { diff++; if (first == (size_t)-1) first = i; }
				printf("die %d received exchange: %zu of %zu bytes differ from the owner's (first at %zd)\n", d, diff, nb, first == (size_t)-1 ? (ssize_t)-1 : (ssize_t)first);
				free(a); free(o);
			}
			for (int hk = 0; hk < 2; hk++) if (ca[d].R[hk]) {   /* the die's prompt cache of KV head hk (codes - 128 times the block's scale) against the twin's caches */
				uint8_t *rc = malloc((size_t)npos * 512); float *rs = malloc((size_t)npos * 64);
				CK(cuMemcpyDtoH(rc, ca[d].R[hk], (size_t)npos * 512)); CK(cuMemcpyDtoH(rs, ca[d].RS[hk], (size_t)npos * 64));
				double wk = 0, wv = 0; int pk = 0, pv = 0, dk = 0, dv = 0;
				for (int p = 0; p < npos; p++) {
					double mk = 1e-30, mv = 1e-30, ek = 0, ev = 0; int ak = 0, av = 0;
					for (int j = 0; j < 256; j++) {
						double kt = hb->kc[attn_kidx((unsigned)p, hk, j)], vt = hb->vc[(size_t)p * 512 + 256 * hk + j];
						double kg = ((int)rc[(size_t)p * 512 + j] - 128) * (double)rs[(size_t)p * 16 + j / 32];
						double vg = ((int)rc[(size_t)p * 512 + 256 + j] - 128) * (double)rs[(size_t)p * 16 + 8 + j / 32];
						mk = fmax(mk, fabs(kt)); mv = fmax(mv, fabs(vt));
						if (fabs(kg - kt) > ek) { ek = fabs(kg - kt); ak = j; }
						if (fabs(vg - vt) > ev) { ev = fabs(vg - vt); av = j; }
					}
					if (ek / mk > wk) { wk = ek / mk; pk = p; dk = ak; }
					if (ev / mv > wv) { wv = ev / mv; pv = p; dv = av; }
				}
				printf("die %d cache of KV head %d: K worst %.3g of max (position %d dim %d), V worst %.3g (position %d dim %d)\n", d, hk, wk, pk, dk, wv, pv, dv);
				free(rc); free(rs);
			}
			float *ao = malloc(row * 256 * nh); CK(cuMemcpyDtoH(ao, cb[d].LAO, row * 256 * nh));
			double worst = 0; int wp = 0, wr = 0, nbadp = 0;
			for (int p = 0; p < npos; p++) {
				double m = 1e-30, e = 0; int er = 0;
				for (unsigned r = 0; r < 256 * nh; r++) { double v = aof[(size_t)6144 * p + 256 * h0 + r]; m = fmax(m, fabs(v)); double x = isnan(ao[(size_t)r * ldx + p]) ? INFINITY : fabs(ao[(size_t)r * ldx + p] - v); if (x > e) { e = x; er = (int)r; } }
				if (e / m > 3e-2) {   /* the bad position and its bad heads */
					nbadp++;
					printf("    die %d position %d: heads", d, p);
					for (unsigned h = 0; h < nh; h++) {
						double eh = 0; for (unsigned r = 256 * h; r < 256 * (h + 1); r++) { double x = fabs(ao[(size_t)r * ldx + p] - aof[(size_t)6144 * p + 256 * h0 + r]); if (!(x <= eh)) eh = x; }
						if (eh / m > 3e-2) printf(" %u", h0 + h);
					}
					printf("\n");
				}
				if (e / m > worst) { worst = e / m; wp = p; wr = er; }
			}
			printf("die %d attention heads %u+%u: worst %.3g of max (position %d row %d: gpu %.7g twin %.7g); %d positions beyond 3e-2\n", d, h0, nh, worst, wp, wr,
			       ao[(size_t)wr * ldx + wp], aof[(size_t)6144 * wp + 256 * h0 + wr], nbadp);
			free(ao);
			continue;
		}
		/* the DeltaNet states of the die's heads after the chunk */
		const split_t *s = &sp[d];
		float *st = malloc((size_t)s->hn * 16384 * 4); CK(cuMemcpyDtoH(st, db[d].dn_state, (size_t)s->hn * 16384 * 4));
		double ss = 0, se = 0;
		for (size_t i = 0; i < (size_t)s->hn * 16384; i++) { double v = hb->dn_state[(size_t)s->h0 * 16384 + i]; ss = fmax(ss, fabs(v)); se = fmax(se, fabs(st[i] - v)); }
		printf("die %d state: max error %.3g of the max %.3g\n", d, se, ss);
		free(st);
	}
	if (getenv("CHUNK_PROF")) {   /* per phase: prefixes of k phases relaunched (best of 3), time(k) - time(k - 1); needs the dies' programs alike */
		int same = 1; for (int d = 1; d < nd; d++) same &= nph[d] == nph[0];
		if (!same) printf("prof: the dies' programs differ in length; skipped\n");
		double prev = 0;
		for (unsigned k = 1; same && k <= nph[0]; k++) {
			double best = 1e30;
			for (int rep = 0; rep < 3; rep++) {
				double ta = now();
				for (int d = 0; d < nd; d++) {
					CK(cuCtxSetCurrent(ctx[d]));
					unsigned ntok = 1, n = k; void *params[] = { &dPROG[d], &n, &dCTL[d], &ntok };
					CK(cuLaunchKernel(fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
				}
				for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
				double t = (now() - ta) * 1e3; if (t < best) best = t;
			}
			const uint32_t op = P[0].p[k - 1].w[0];
			printf("prof %3u %-14s units %4u  %8.3f ms  (prefix %8.3f)\n", k - 1, op < OP_COUNT ? die_opname[op] : "?", P[0].p[k - 1].w[1] & 0xffff, best - prev, best);
			prev = best;
		}
	}
	printf("%s\n", bad ? "MISMATCH" : "OK");
	return bad ? 3 : 0;
}
