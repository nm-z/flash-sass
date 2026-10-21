/* engine_chunk.c (included by engine.c): the chunked prefill.
 *
 * FLASH_CHUNK_PREFILL=<ldx> (a power of two, 128 .. 1024): the prompt runs in chunks of up to ldx positions, every op of a block over
 * the chunk's positions in one phase (kern/prog_chunk.c: GEMMs on k-major slabs, the slab / ebatch / cattn ops), one launch per chunk
 * and die. The positions [count, nprompt - 1) of a response go through it while at least CHUNK_MIN are left; the rest and the last
 * prompt position (its logits) run in the per-position programs, which find every state the chunks left (DeltaNet matrices and conv
 * windows, the attention layers' int8 caches, pooled keys and key ring, the PLE ring).
 *
 * The host stages each chunk: the token embeddings (token_embd's Q3_K rows, a host copy) and the PLE rows (staged by the PLE readers,
 * IQ4_NL) dequantised into [2560][ldx] slabs, the RoPE rows of the chunk's positions (plain text positions). The resident whole-expert
 * layout is viewed as four lists of 3/3/3/1 neuron blocks; its down op addresses the original 192-byte rows directly. It needs a
 * head-aligned attention Q split (split_attention_queries keeps one when FLASH_CHUNK_PREFILL is set) and a response
 * without images. FLASH_CHUNK_MAXT: the positions the dies' prompt caches of K / V (the attention layers' copies for their query heads)
 * hold, default 8448; a prompt longer than that runs the rest per position. */

#include "prog_chunk.h"
#include "quant.h"

#define CHUNK_MIN 32
static unsigned chunk_ldx, chunk_maxt;
static chunk_bufs_t cbuf[MAXD];
static xchg_t cxch[MAXD];
static CUdeviceptr cprog[MAXD];
static unsigned cprog_cap;
static chunk_attn_t cat[MAXD][NB];
static CUdeviceptr cple_alt[MAXD];       /* the PLE conv ring's second buffer (the decode's is dst_ple) */
static uint8_t *h_embd; static size_t h_embd_rows;
static float *h_slab, *h_rope;
static double chunk_ms;                  /* the last response's chunk launches, wall time */
static unsigned chunk_half_end[MAXD][2 * NB];   /* the phase count of die d's program after each half (chunk_build) */
static CUdeviceptr ctrace[MAXD];         /* FLASH_CHUNK_TRACE=1 with a DIE_TRACE=1 cubin: CTA 0's samples of the first chunk's phases */
static uint32_t *cbuild_ops[MAXD];       /* the op of every phase of the last uploaded chunk program */
static int chunk_positions;              /* and their positions */
static CUdeviceptr cunit_maps[MAXD][NB][4];
static CUmodule chunk_mod[MAXD];
static CUfunction chunk_fn[MAXD];
static char chunk_cubin_path[512];
static uint64_t chunk_cubin_fnv;
static uint64_t chunk_hash_ref[MAXD][NB + 2], chunk_hash_key;
static int chunk_hash_have;
static unsigned chunk_last_npos;
static char chunk_hash_report[384];

static uint64_t chunk_host_hash(const void *p, size_t n, uint64_t h) {
	const unsigned char *s = p;
	for (size_t i = 0; i < n; i++) { h ^= s[i]; h *= 0x100000001b3ull; }
	return h;
}

/* Diagnostic only: fingerprint the live prefix state before scalar generation.
 * Ignore unused columns of transposed pooled keys and padded slab positions. */
static void chunk_hash_prefix(unsigned positions) {
	const uint64_t seed = 0xcbf29ce484222325ull;
	uint64_t key = chunk_host_hash(h_ids, 4ull * positions, seed);
	key = chunk_host_hash(&positions, sizeof positions, key);
	int fresh = !chunk_hash_have || key != chunk_hash_key, changed = 0, first_layer = NB + 2, first_die = -1;
	size_t slab_bytes = 4ull * 10240 * chunk_ldx;
	unsigned char *slab = malloc(slab_bytes), pool[8192];
	if (!slab) { fprintf(stderr,"engine: prefix fingerprint allocation failed\n"); longjmp(run_jb,1); }
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int b = 0; b < nlayers_run + 2; b++) {
			uint64_t h = seed;
			int slot = b < nlayers_run ? b : b == nlayers_run ? NB : NB + 1;
			if (b == nlayers_run) h = device_fnv(dst_ple[d], 4 * 10 * 10240, h);
			else if (b == nlayers_run + 1) {
				CK(cuMemcpyDtoH(slab, cbuf[d].S, slab_bytes));
				for (unsigned row = 0; row < 10240; row++) h = chunk_host_hash(slab + 4ull * row * chunk_ldx, 4ull * chunk_last_npos, h);
			} else if (!BLK_IS_ATTN(b)) {
				h = device_fnv(dst_dn[d][b], dn_bytes[d][b], h);
				h = device_fnv(dst_conv[d][b], conv_bytes[d][b], h);
			} else if (dst_kq[d][b]) {
				h = device_fnv(dst_ring[d][b], 2048, h);
				h = device_fnv(dst_kq[d][b], 512ull * positions, h);
				h = device_fnv(dst_vq[d][b], 512ull * positions, h);
				h = device_fnv(dst_ks[d][b], 32ull * positions, h);
				h = device_fnv(dst_vs[d][b], 32ull * positions, h);
				unsigned groups = positions / 4;
				for (unsigned at = 0; at < groups; at += 32) {
					unsigned n = groups - at < 32 ? groups - at : 32;
					CK(cuMemcpyDtoH(pool, dst_pkt[d][b] + 8192ull * (at / 32), sizeof pool));
					for (unsigned row = 0; row < 128; row++) h = chunk_host_hash(pool + 64 * row, 2 * n, h);
				}
			}
			if (!fresh && h != chunk_hash_ref[d][slot]) {
				changed++;
				if (slot < first_layer) { first_layer = slot; first_die = d; }
			}
			if (fresh) chunk_hash_ref[d][slot] = h;
		}
	}
	free(slab); chunk_hash_key = key; chunk_hash_have = 1;
	snprintf(chunk_hash_report, sizeof chunk_hash_report,
	         "chunkhash positions=%u reference=%d changed_buffers=%d first_layer=%d first_die=%d input=%016llx",
	         positions, fresh, changed, changed ? first_layer : -1, first_die, (unsigned long long)key);
}

static int chunk_enabled(void) {
	const char *e = getenv("FLASH_CHUNK_PREFILL");
	return e && atoi(e) > 0;
}

/* the heads of block bb's attention on die d: [h0, h0 + nh) */
static void chunk_heads(int d, int bb, unsigned *h0, unsigned *nh) {
	const dn_weights_t *w = &dw[d][bb];
	if (w->attn_q_split) { *h0 = w->attn_q_first / 512u; *nh = w->attn_q_rows / 512u; }
	else if (dst_kq[d][bb]) { *h0 = 0; *nh = 24; }
	else { *h0 = 0; *nh = 0; }
}

static void chunk_init(void) {
	chunk_ldx = 0;
	if (!chunk_enabled()) return;
	if (!attn_q_split_mode) { fprintf(stderr, "engine: native chunk prefill requires FLASH_ATTN_Q_SPLIT=1\n"); longjmp(init_jb, 1); }
	unsigned ldx = (unsigned)atoi(getenv("FLASH_CHUNK_PREFILL"));
	if (ldx < 128 || ldx > 1024 || (ldx & (ldx - 1))) { fprintf(stderr, "FLASH_CHUNK_PREFILL must be 0 or a power of two 128..1024\n"); longjmp(init_jb, 1); }
	chunk_maxt = getenv("FLASH_CHUNK_MAXT") ? (unsigned)atoi(getenv("FLASH_CHUNK_MAXT")) : 8448u;
	if (chunk_maxt > (unsigned)MAXT) chunk_maxt = (unsigned)MAXT;
	if (chunk_maxt > 32768u) chunk_maxt = 32768u;
	const char *image = getenv("FLASH_CHUNK_CUBIN");
	if (image && *image) {
		if (strlen(image) >= sizeof chunk_cubin_path) { fprintf(stderr,"engine: chunk cubin path too long\n");longjmp(init_jb,1); }
		strcpy(chunk_cubin_path,image);chunk_cubin_fnv=file_fnv(image);
	}
	unsigned cslot = (unsigned)CHUNK_SLOT_BYTES(ldx);
	size_t row = 4u * (size_t)ldx;
	cprog_cap = 8192;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		if (image && *image) {
			CUfunction marker;
			CK(cuModuleLoad(&chunk_mod[d],image));
			CK(cuModuleGetFunction(&marker,chunk_mod[d],"flash_chunk_abi_v2"));
			CK(cuModuleGetFunction(&chunk_fn[d],chunk_mod[d],"die"));
		}
		for (int bb = 0; bb < nlayers_run; bb++) {
			uint32_t local[512], maps[4][512];
			CK(cuMemcpyDtoH(local, dw[d][bb].emap, sizeof local));
			for (int g = 0; g < 4; g++) for (int e = 0; e < 512; e++) {
				if (local[e] == UINT32_MAX) { maps[g][e] = 0; continue; }
				if (local[e] >= (unsigned)sp[d][bb].en) { fprintf(stderr, "engine: invalid resident expert map\n"); longjmp(init_jb, 1); }
				maps[g][e] = ((10u * local[e] + 3u * (unsigned)g) << 8) | (g == 3 ? 1u : 3u);
			}
			CUdeviceptr mp = dalloc(sizeof maps); CK(cuMemcpyHtoD(mp, maps, sizeof maps));
			for (int g = 0; g < 4; g++) cunit_maps[d][bb][g] = mp + (size_t)g * sizeof maps[0];
		}
		size_t before = alloc_bytes;
		unsigned djn = 0, cols = 0, nloc = 0, hrows = 0, hn = 0, yn = 0, nh = 0, pkn = (unsigned)sp[d][1].pkn;
		for (int bb = 0; bb < nlayers_run; bb++) {
			const split_t *s = &sp[d][bb];
			if ((unsigned)s->djn > djn) djn = (unsigned)s->djn;
			if (32u * (unsigned)s->cbn > cols) cols = 32u * (unsigned)s->cbn;
			if ((unsigned)s->yn > yn) yn = (unsigned)s->yn;
			if (BLK_IS_ATTN(bb)) { unsigned h0, n; chunk_heads(d, bb, &h0, &n); if (n > nh) nh = n; }
			else {
				if (SPLIT_NLOC(s) > nloc) nloc = SPLIT_NLOC(s);
				if ((unsigned)s->hn > hn) hn = (unsigned)s->hn;
			}
		}
		hrows = 128u * hn;
		chunk_bufs_t *c = &cbuf[d]; memset(c, 0, sizeof *c); c->ldx = ldx; c->nvu = 4;
		c->S = dalloc(row * 10240); c->XH = dalloc(row * 10240); c->LDN = dalloc(row * djn); c->T = dalloc(row * 324);
		c->LUP = dalloc(row * 4 * cols); c->LZ = dalloc(row * cols); c->LA = dalloc(row * 2 * hn); c->LY = dalloc(row * yn);
		c->R = dalloc(row * 513); c->IDX = dalloc(row * 10); c->RW = dalloc(row * 10); c->SW = dalloc(row);
		c->LP = dalloc(row * 10); c->LW = dalloc(row * 10); c->ITEMS = dalloc(512 * 16); c->NITEMS = dalloc(16);
		/* the DeltaNet and MoE scratch in one block: block 1's PLE prefix (before its halves, after block 0's) keeps its two 10240-row
		   slabs there */
		size_t srows = 2u * nloc + 3u * hrows + 2560 + 3 * 1920 + 2560 + 3 * 640 + 2560, at = 0;
		CUdeviceptr scr = dalloc(row * (srows > 20480 ? srows : 20480));
		#define CARVE(f, n) do { c->f = scr + row * at; at += (n); } while (0)
		CARVE(LQKV, nloc); CARVE(LU, nloc); CARVE(LGATE, hrows); CARVE(LO, hrows); CARVE(LON, hrows); CARVE(ZT, 2560);
		CARVE(G, 1920); CARVE(U, 1920); CARVE(H, 1920); CARVE(PART, 2560); CARVE(SG, 640); CARVE(SU, 640); CARVE(SH, 640); CARVE(SD, 2560);
		#undef CARVE
		c->PL1 = scr; c->PL2 = scr + row * 10240;
		c->LQ = nh ? dalloc(row * 512 * nh) : 0; c->LK = dalloc(row * 512); c->LV = dalloc(row * 512); c->LI = dalloc(row * 640);
		c->CB = dalloc(CHUNK_CB_BYTES(ldx)); c->IQ = dalloc(row * 512); c->LAO = nh ? dalloc(row * 256 * nh) : 0;
		c->CS = dalloc(128 * ((size_t)ldx + 3)); c->SN = dalloc(128 * ((size_t)ldx + 3)); c->q6t_rows = 4 * yn > 1100 ? 4 * yn : 1100; c->Q6T = dalloc(row * c->q6t_rows);
		c->PG = dalloc(row * 2560); c->LPK = dalloc(row * (pkn ? pkn : 1));
		xchg_t *x = &cxch[d];
		x->inbox = dalloc(2 * (size_t)cslot); x->xtab = dalloc(128); x->flags = xc[d].flags; x->xgen = xc[d].xgen;
		/* f32 exchanges by default; optional fp16 and card relays remain explicit experiments. */
		x->unp[0] = x->unp[1] = 0; x->pinbox = 0; x->xd = d; x->xnd = nd; x->relay = getenv("FLASH_CHUNK_RELAY") && atoi(getenv("FLASH_CHUNK_RELAY"));
		if (getenv("FLASH_CHUNK_XH") && atoi(getenv("FLASH_CHUNK_XH"))) { x->unp[0] = dalloc(cslot); x->unp[1] = dalloc(cslot); }
		cprog[d] = dalloc(sizeof(phase_t) * (size_t)cprog_cap);
		cple_alt[d] = dalloc(4 * 10 * 10240);
		for (int bb = 0; bb < nlayers_run; bb++) {
			chunk_attn_t *a = &cat[d][bb]; memset(a, 0, sizeof *a);
			if (!BLK_IS_ATTN(bb)) continue;
			a->nblk = nblk_qsa;
			if (dst_kq[d][bb]) a->POOL = dalloc((size_t)128 * 8192 * 4);
			unsigned h0, n; chunk_heads(d, bb, &h0, &n);
			for (int hk = 0; hk < 2; hk++) {
				if (!n || h0 + n <= 12u * (unsigned)hk || h0 >= 12u * (unsigned)(hk + 1)) continue;
				a->R[hk] = dalloc((size_t)chunk_maxt * 512); a->RS[hk] = dalloc((size_t)chunk_maxt * 64);
			}
		}
		fprintf(stderr, "engine: chunked prefill die %d: ldx %u, %u heads, %.1f MB\n", d, ldx, nh, (double)(alloc_bytes - before) / 1e6);
		if (getenv("FLASH_CHUNK_PROF")) {   /* the buffers the attention ops address, for fault reports */
			fprintf(stderr, "engine:   die %d LQ %#llx LAO %#llx CS %#llx SN %#llx inbox %#llx+2x%#x CB %#llx\n", d, (unsigned long long)c->LQ, (unsigned long long)c->LAO,
			        (unsigned long long)c->CS, (unsigned long long)c->SN, (unsigned long long)x->inbox, cslot, (unsigned long long)c->CB);
			for (int bb = 0; bb < NB; bb++) if (BLK_IS_ATTN(bb)) {
				const chunk_attn_t *a = &cat[d][bb]; unsigned h0, n; chunk_heads(d, bb, &h0, &n);
				fprintf(stderr, "engine:   die %d block %d heads %u+%u R %#llx %#llx RS %#llx %#llx POOL %#llx nblk %u qnorm %#llx\n", d, bb, h0, n,
				        (unsigned long long)a->R[0], (unsigned long long)a->R[1], (unsigned long long)a->RS[0], (unsigned long long)a->RS[1],
				        (unsigned long long)a->POOL, a->nblk, (unsigned long long)dw[d][bb].attn_q_norm);
			}
		}
	}
	uint64_t tab[12] = { 0 };
	for (int d = 0; d < 6; d++) { tab[d] = (uint64_t)cxch[d < nd ? d : 0].inbox; tab[6 + d] = (uint64_t)xc[d < nd ? d : 0].flags; }
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(cxch[d].xtab, tab, sizeof tab)); cxch[d].pinbox = (d ^ 1) < nd ? cxch[d ^ 1].inbox : 0; }
	/* the token embeddings on the host */
	const hold_tensor *te = hold_find(&ar[embed_die], "token_embd.weight");
	if (!te || te->type != GG_Q3_K || te->K != 2560) { fprintf(stderr, "engine: the chunked prefill needs token_embd as Q3_K rows of 2560\n"); longjmp(init_jb, 1); }
	h_embd_rows = te->bytes / 1100u;
	h_embd = malloc(te->bytes);
	if (!h_embd) { fprintf(stderr, "engine: no host memory for token_embd\n"); longjmp(init_jb, 1); }
	CK(cuCtxSetCurrent(ctx[embed_die])); CK(cuMemcpyDtoH(h_embd, ar[embed_die].base + te->off, te->bytes));
	h_slab = malloc(4 * 2560 * (size_t)ldx); h_rope = malloc(4 * 64 * ((size_t)ldx + 3));
	chunk_ldx = ldx;
}

static void chunk_fini(void) {
	chunk_hash_have = 0; chunk_hash_report[0] = 0;
	for (int d = 0; d < MAXD; d++) if (chunk_mod[d]) {
		cuCtxSetCurrent(ctx[d]);cuModuleUnload(chunk_mod[d]);chunk_mod[d]=NULL;chunk_fn[d]=NULL;
	}
	chunk_cubin_path[0]=0;chunk_cubin_fnv=0;
	free(h_embd); free(h_slab); free(h_rope); h_embd = NULL; h_slab = h_rope = NULL; chunk_ldx = 0;
	memset(ctrace, 0, sizeof ctrace);
	for (int d = 0; d < MAXD; d++) { free(cbuild_ops[d]); cbuild_ops[d] = NULL; }
}

/* the 16 PLE rows of position q (IQ4_NL, 90 B each), as the PLEROWS op reads them */
static void chunk_ple_rows(int q, const uint8_t *rows[16]) {
	if (ple_disk || core->ple_compact) { for (int h = 0; h < 16; h++) rows[h] = ple_host + ((size_t)q * 16 + (size_t)h) * 90; return; }
	uint64_t r[16]; uint32_t endid = ple_end_id;
	ple_rows_c(h_ids, (size_t)q, ple_absent_id, &endid, 1, ple_multipliers, ple_head_offsets, ple_head_vocab, r);
	for (int h = 0; h < 16; h++) rows[h] = ple_host + r[h] * 90;
}

/* the chunk program of die d for blocks [b0, b1): halves = 1 the attention-side half (and block 1's PLE prefix), 2 the MoE half */
static void chunk_build(prog_t *P, int d, int b0, int b1, int halves, unsigned t0, unsigned npos, int k) {
	unsigned cslot = (unsigned)CHUNK_SLOT_BYTES(chunk_ldx);
	CUdeviceptr RIN = (k & 1) ? cple_alt[d] : dst_ple[d], ROUT = (k & 1) ? dst_ple[d] : cple_alt[d];
	for (int bb = b0; bb < b1; bb++) {
		dn_bufs_t layer = db[d];
		layer.dn_state = dst_dn[d][bb]; layer.conv_state = dst_conv[d][bb]; layer.ple_state = dst_ple[d];
		layer.qsa = 1; layer.nblk = nblk_qsa;
		layer.kq = dst_kq[d][bb]; layer.ks = dst_ks[d][bb]; layer.vq = dst_vq[d][bb]; layer.vs = dst_vs[d][bb];
		layer.pkt = dst_pkt[d][bb]; layer.ring = dst_ring[d][bb];
		const split_t *s = &sp[d][bb]; const dn_weights_t *w = &dw[d][bb];
		for (int g = 0; g < 4; g++) {
			cbuf[d].vu_g[g] = w->ffn_gate_exps; cbuf[d].vu_u[g] = w->ffn_up_exps;
			cbuf[d].vu_d[g] = w->ffn_down_exps; cbuf[d].vu_emap[g] = cunit_maps[d][bb][g];
		}
		if (halves & 1) {
			if (bb == 1) prog_chunk_ple_prefix(P, s, &cxch[d], w, &cbuf[d], RIN, ROUT, t0, npos, eps, cslot);
			if (BLK_IS_ATTN(bb)) prog_chunk_attn_half(P, s, &cxch[d], w, &layer, &cbuf[d], &cat[d][bb], t0, npos, eps, cslot);
			else prog_chunk_deltanet_half(P, s, &cxch[d], w, &layer, &cbuf[d], t0, npos, eps, cslot);
		}
		chunk_half_end[d][2 * bb] = (unsigned)P->n;
		if (halves & 2) prog_chunk_moe_half(P, s, &cxch[d], w, &layer, &cbuf[d], npos, eps, cslot);
		chunk_half_end[d][2 * bb + 1] = (unsigned)P->n;
	}
}

/* launch every die's program (already uploaded to cprog) as one pass over the chunk; wait */
static double chunk_launch(unsigned t0, unsigned npos, const unsigned *nph) {
	double t1 = now();
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuMemsetD32(dCTL[d] + 0x24, npos, 1)); CK(cuMemsetD32(dCTL[d] + 0x8, t0, 1));
		unsigned ntok = t0 + npos, n = nph[d];
		void *params[] = { &cprog[d], &n, &dCTL[d], &ntok };
		CK(cuLaunchKernel(chunk_mod[d] ? chunk_fn[d] : fn[d], 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	}
	double limit = getenv("FLASH_CHUNK_WAIT") ? atof(getenv("FLASH_CHUNK_WAIT")) : 120.0;
	for (int left = nd; left; ) {
		left = 0;
		for (int d = 0; d < nd; d++) {
			CK(cuCtxSetCurrent(ctx[d])); CUresult q = cuStreamQuery(0);
			if (q == CUDA_ERROR_NOT_READY) left++;
			else if (q != CUDA_SUCCESS) { fprintf(stderr, "engine: die %d failed in the chunk at %u\n", d, t0); CK(q); }
		}
		if (left && now() - t1 > limit) {
			fprintf(stderr, "engine: chunk at %u not done after %.0f s\n", t0, limit);
			longjmp(run_jb, 1);
		}
	}
	return (now() - t1) * 1e3;
}

static void chunk_upload(prog_t *P, unsigned *nph) {
	prog_chunk_xh_patch(P, cxch, nd);
	for (int d = 0; d < nd; d++) {
		if ((unsigned)P[d].n > cprog_cap) { fprintf(stderr, "engine: chunk program of %d phases (capacity %u)\n", P[d].n, cprog_cap); longjmp(run_jb, 1); }
		nph[d] = (unsigned)P[d].n;
		CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(cprog[d], P[d].p, sizeof(phase_t) * (size_t)P[d].n));
		if (getenv("FLASH_CHUNK_OPS") && atoi(getenv("FLASH_CHUNK_OPS")) > d) {   /* debugging: every phase of the first FLASH_CHUNK_OPS dies' programs: op, units, flags, the parameters of the exchanges */
			for (int i = 0; i < P[d].n; i++) {
				const phase_t *ph = &P[d].p[i]; unsigned op = ph->w[0];
				fprintf(stderr, "engine: ops die %d %4d %-12s units %4u flags %#x", d, i, chunk_op_name(op), ph->w[1], ph->w[30]);
				if (op == OP_SLAB_XH || op == OP_SLAB_UH) for (int k = 0; k < 12; k++) fprintf(stderr, " %x", ph->w[2 + k]);
				fputc('\n', stderr);
			}
		}
		uint32_t *ops = realloc(cbuild_ops[d], 4 * (size_t)P[d].n);   /* the ops of the phases, for the chunk trace */
		if (ops) { for (int i = 0; i < P[d].n; i++) ops[i] = P[d].p[i].w[0]; cbuild_ops[d] = ops; }
	}
}

/* one chunk: positions [t0, t0 + npos) on every die; 0 when it ran */
static int chunk_one(runctx *r, unsigned t0, unsigned npos, int k) {
	unsigned ldx = chunk_ldx;
	chunk_last_npos = npos;
	size_t row = 4u * (size_t)ldx;
	/* the PLE rows of the chunk's positions */
	if (ple_disk) while (ple_next < (int)(t0 + npos)) {
		runctx staging = *r; staging.count = ple_next;
		int before = ple_next;
		ple_pump(&staging, (int)(t0 + npos) - ple_next);
		if (before == ple_next) { fprintf(stderr, "engine: chunk PLE staging made no progress\n"); longjmp(run_jb, 1); }
	}
	float tmp[2560];
	/* embeddings: S's four lanes */
	memset(h_slab, 0, 4 * 2560 * (size_t)ldx);
	for (unsigned p = 0; p < npos; p++) {
		uint32_t id = h_ids[t0 + p];
		if (id >= h_embd_rows) { fprintf(stderr, "engine: token %u outside token_embd\n", id); longjmp(run_jb, 1); }
		dequant(GG_Q3_K, h_embd + (size_t)id * 1100, tmp, 2560);
		for (int c = 0; c < 2560; c++) h_slab[(size_t)c * ldx + p] = tmp[c];
	}
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuMemcpyHtoD(cbuf[d].S, h_slab, row * 2560));
		for (int l = 1; l < 4; l++) CK(cuMemcpyDtoD(cbuf[d].S + row * 2560 * (size_t)l, cbuf[d].S, row * 2560));
	}
	/* PLE rows */
	memset(h_slab, 0, 4 * 2560 * (size_t)ldx);
	for (unsigned p = 0; p < npos; p++) {
		const uint8_t *rows[16]; chunk_ple_rows((int)(t0 + p), rows);
		for (int h = 0; h < 16; h++) dequant(GG_IQ4_NL, rows[h], tmp + 160 * h, 160);
		for (int c = 0; c < 2560; c++) h_slab[(size_t)c * ldx + p] = tmp[c];
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(cbuf[d].PG, h_slab, row * 2560)); }
	/* RoPE rows of positions t0 - 3 .. t0 + npos - 1: cos at [i][32], sin after them */
	float *cs = h_rope, *sn = h_rope + 32 * ((size_t)ldx + 3);
	memset(h_rope, 0, 4 * 64 * ((size_t)ldx + 3));
	for (unsigned i = 0; i < npos + 3; i++) {
		long pos = (long)t0 - 3 + (long)i;
		if (pos < 0) continue;
		for (int pr = 0; pr < 32; pr++) { double ang = (double)pos * pow((double)1e7f, -2.0 * pr / 64.0); cs[32 * i + pr] = (float)cos(ang); sn[32 * i + pr] = (float)sin(ang); }
	}
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuMemcpyHtoD(cbuf[d].CS, cs, 128 * ((size_t)npos + 3))); CK(cuMemcpyHtoD(cbuf[d].SN, sn, 128 * ((size_t)npos + 3)));
	}
unsigned nph[MAXD];
prog_t P[MAXD];
if (k == 0 && getenv("FLASH_CHUNK_PROF") && atoi(getenv("FLASH_CHUNK_PROF"))) {   /* the first chunk one half at a time, timed */
	double total = 0;
	/* FLASH_CHUNK_PROF_ORDER="b:h,b:h,...": only these halves (h 1 the attention side, 2 the MoE), in this order (debugging) */
	int order[4 * NB], no = 0;
	const char *po = getenv("FLASH_CHUNK_PROF_ORDER");
	if (po) for (const char *q = po; *q && no < 4 * NB; ) { char *e; long b = strtol(q, &e, 10); if (e == q || *e != ':') break; long h = strtol(e + 1, &e, 10); order[no++] = (int)(2 * b + h - 1); q = *e == ',' ? e + 1 : e; if (!*e) break; }
	else for (int bb = 0; bb < nlayers_run; bb++) { order[no++] = 2 * bb; order[no++] = 2 * bb + 1; }
	for (int oi = 0; oi < no; oi++) {
		int bb = order[oi] / 2, half = order[oi] % 2 + 1;
		for (int d = 0; d < nd; d++) { prog_init(&P[d]); chunk_build(&P[d], d, bb, bb + 1, half, t0, npos, k); }
		chunk_upload(P, nph);
		double ms = chunk_launch(t0, npos, nph);
		total += ms;
		fprintf(stderr, "engine: chunk prof block %d %s: %u phases, %.2f ms\n", bb, half == 1 ? (BLK_IS_ATTN(bb) ? "attn" : "deltanet") : "moe", nph[0], ms);
		for (int d = 0; d < nd; d++) prog_free(&P[d]);
		if (getenv("FLASH_CHUNK_CHECK")) {   /* every die's stream after the half: non-finite values stop the request (before they fault) */
			int bad = 0; float *hs = malloc(4 * 10240 * (size_t)chunk_ldx);
			for (int d = 0; d < nd; d++) {
				CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(hs, cbuf[d].S, 4 * 10240 * (size_t)chunk_ldx));
				long nf = 0; double mx = 0; int fr = -1, fp = -1;
				for (int r = 0; r < 10240; r++) for (unsigned p = 0; p < npos; p++) {
					float v = hs[(size_t)r * chunk_ldx + p];
					if (!isfinite(v)) { if (!nf++) { fr = r; fp = (int)p; } } else if (fabs(v) > mx) mx = fabs(v);
				}
				if (nf || d == 0) fprintf(stderr, "engine: chunk check block %d half %d die %d: %ld non-finite (first row %d position %d), max |S| %.3g\n", bb, half, d, nf, fr, fp, mx);
				if (nf) bad = 1;
			}
			free(hs);
			if (bad) { fprintf(stderr, "engine: chunk check: stopping the request\n"); longjmp(run_jb, 1); }
		}
	}
	fprintf(stderr, "engine: chunk prof: %u positions, %.1f ms in all\n", npos, total);
	chunk_ms += total;
} else {
	for (int d = 0; d < nd; d++) {
		prog_init(&P[d]); chunk_build(&P[d], d, 0, nlayers_run, 3, t0, npos, k);
		if (getenv("FLASH_DUMP_CHUNK")) { char path[600]; snprintf(path, sizeof path, "%s.c%u.d%d", getenv("FLASH_DUMP_CHUNK"), t0, d); FILE *pf = fopen(path, "wb"); if (pf) { fwrite(P[d].p, sizeof(phase_t), (size_t)P[d].n, pf); fclose(pf); } }
	}
	chunk_upload(P, nph);
	for (int d = 0; d < nd; d++) prog_free(&P[d]);
	int tr = k == 0 && getenv("FLASH_CHUNK_TRACE") && atoi(getenv("FLASH_CHUNK_TRACE"));
	if (tr) for (int d = 0; d < nd; d++) {   /* ctl+0xd0: records, first position, count, CTAs recorded (a DIE_TRACE=1 cubin) */
		CK(cuCtxSetCurrent(ctx[d]));
		if (!ctrace[d]) ctrace[d] = dalloc((size_t)cprog_cap * 64);
		CK(cuMemsetD8(ctrace[d], 0, (size_t)nph[d] * 64));
		uint32_t w5[5] = { (uint32_t)ctrace[d], (uint32_t)(ctrace[d] >> 32), t0, 1, 1 };
		CK(cuMemcpyHtoD(dCTL[d] + 0xd0, w5, sizeof w5));
	}
	chunk_ms += chunk_launch(t0, npos, nph);
	if (tr) {
		for (int d = 0; d < nd; d++) {
			uint64_t *s = malloc((size_t)nph[d] * 64);
			CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(s, ctrace[d], (size_t)nph[d] * 64)); CK(cuMemsetD32(dCTL[d] + 0xdc, 0, 1));
			double opn[CHUNK_OP_COUNT] = { 0 }, ot[CHUNK_OP_COUNT][4] = { { 0 } }, tot[4] = { 0 };
			for (unsigned i = 0; i < nph[d]; i++) {
				const uint64_t *x = s + 8 * (size_t)i; uint32_t op = cbuild_ops[d] ? cbuild_ops[d][i] : 0;
				if (!x[0] || !x[4]) continue;
				double v[4] = { (x[1] - x[0]) / 1e3, (x[2] - x[1]) / 1e3, (x[3] - x[2]) / 1e3, (x[4] - x[3]) / 1e3 };
				if (op < CHUNK_OP_COUNT) { opn[op]++; for (int j = 0; j < 4; j++) ot[op][j] += v[j]; }
				for (int j = 0; j < 4; j++) tot[j] += v[j];
			}
			fprintf(stderr, "engine: chunk trace die %d (%u positions, %u phases): dispatch %.0f us, exchange wait %.0f, body %.0f, barrier %.0f\n", d, npos, nph[d], tot[0], tot[1], tot[2], tot[3]);
			for (int op = 0; op < CHUNK_OP_COUNT; op++) if (opn[op])
				fprintf(stderr, "engine:   %-14s %4.0f  wait %8.0f  body %8.0f  barrier %7.0f us\n", chunk_op_name(op), opn[op], ot[op][1], ot[op][2], ot[op][3]);
			unsigned from = 0;
			fprintf(stderr, "engine:   halves (us):");
			for (int bb = 0; bb < nlayers_run; bb++) for (int h = 0; h < 2; h++) {
				unsigned to = chunk_half_end[d][2 * bb + h];
				if (to > from && to <= nph[d] && s[8 * (size_t)from] && s[8 * (size_t)(to - 1) + 4]) fprintf(stderr, " %d%c %.0f", bb, h ? 'm' : (BLK_IS_ATTN(bb) ? 'a' : 'd'), (s[8 * (size_t)(to - 1) + 4] - s[8 * (size_t)from]) / 1e3);
				from = to;
			}
			fputc('\n', stderr);
			free(s);
		}
	}
}
chunk_positions += (int)npos;
for (int d = 0; d < nd; d++) {
	uint32_t got; CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(&got, dCTL[d] + 0x8, 4));
	if ((got & 0x7fffffffu) != t0 + npos) { fprintf(stderr, "engine: die %d chunk at %u ended at %u, not %u\n", d, t0, got & 0x7fffffffu, t0 + npos); longjmp(run_jb, 1); }
}
r->count = (int)(t0 + npos);
	return 0;
}

/* positions [r->count, end) in chunks while at least CHUNK_MIN are left (and the prompt caches hold them) */
static void chunk_prefill(runctx *r, int end, int nimg) {
	chunk_ms = 0; chunk_positions = 0;
	chunk_hash_report[0] = 0;
	if (!chunk_ldx || nimg || r->aborted) return;
	int k = 0;
	while (!r->aborted && end - r->count >= CHUNK_MIN) {
		unsigned t0 = (unsigned)r->count, npos = (unsigned)(end - r->count) < chunk_ldx ? (unsigned)(end - r->count) : chunk_ldx;
		if (t0 + npos > chunk_maxt) break;
		if (r->cb && r->cb->alive && r->cb->alive(r->cb->arg)) { r->aborted = 1; break; }
		chunk_one(r, t0, npos, k);
		k++;
	}
	if (k & 1) for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoD(dst_ple[d], cple_alt[d], 4 * 10 * 10240)); }   /* the last chunk wrote the second ring */
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemsetD32(dCTL[d] + 0x24, 0, 1)); }
	if (chunk_positions && getenv("FLASH_PREFIX_HASH") && atoi(getenv("FLASH_PREFIX_HASH"))) chunk_hash_prefix((unsigned)r->count);
	if (chunk_positions) fprintf(stderr, "engine: chunk launches: %d positions in %.1f ms excluding host staging\n", chunk_positions, chunk_ms);
}
