/* Phase programs for the persistent kernel (die.pl): descriptor builders per op
 * and the block programs of one die of the n-die split (n = 1: the whole block on
 * one die, exchanges with itself). Parameter offsets are each op's standalone
 * CONSTANT_MAPPING (descriptor word 2 + (addr - 0x140) / 4, see die.pl). */
#ifndef FLASH_PROG_H
#define FLASH_PROG_H
#include "dieops.h"
#include <cuda.h>
#include <stdint.h>

typedef struct { uint32_t w[32]; } phase_t;
/* Logical phases retain the scalar parameter ABI. Grouped programs store one
 * descriptor per known position, with the group width in primary word 30.
 * Legacy B2 descriptors omit the width and retain bit 3 exactly as before. */
#define PROG_MAX_PHASES 4096u
#define PROG_MAX_POSITIONS 32u
#define PROG_MAX_DESCRIPTORS (PROG_MAX_PHASES * PROG_MAX_POSITIONS)
#define PROG_MAX_PROFILE_PHASES PROG_MAX_DESCRIPTORS
#define PHASE_GROUP_FLAG 8u
#define PHASE_FUSED_FLAG 16u
#define PHASE_WIDTH_SHIFT 8u
#define PHASE_WIDTH_MASK (255u << PHASE_WIDTH_SHIFT)
static inline unsigned phase_positions(const phase_t *p) {
    if (!(p->w[30] & PHASE_GROUP_FLAG)) return 1;
    unsigned n = (p->w[30] & PHASE_WIDTH_MASK) >> PHASE_WIDTH_SHIFT;
    return n ? n : 2;
}
typedef struct { phase_t *p; int n, cap; int xj; int loop; unsigned token_offset; int pending_wait; uint32_t stop[2]; CUdeviceptr end; int samp_k; float samp_topp, samp_invT; } prog_t;   /* stop[2], end: a generated token equal to stop[0] or stop[1] sets *end (ctl[5]) = 0x80000000 and the kernel exits after that token (0xffffffff: none) */
   /* pending_wait: the next phase consumes an exchange */   /* xj: exchanges issued (slot parity); loop: token-relative
 * parameters are marked for the kernel's token loop instead of taking the position literally */

void prog_init(prog_t *P);
/* Group consecutive known positions in phase-major order. Scalar fallbacks run
 * in causal order with one grid barrier and publication per logical phase.
 * For widths above two, the validated KVGATHER/ATTNPARTN/ATTNMERGE region runs
 * position-major, retaining its per-op barriers so compact scratch can alias.
 * out is initialized and unchanged on failure; no nested groups are accepted. */
int prog_group(prog_t *out, const prog_t *const *positions, unsigned count);
/* Wide arena lifetimes also serialize IDXQK..ATTNMERGE and
 * PLEGATE/RMS/PLECONV. Apply the same policy to wide-mode B2 tails. */
int prog_group_aliased(prog_t *out, const prog_t *const *positions, unsigned count);
/* Pair consecutive known positions. out is initialized; unchanged on failure. */
int prog_pair(prog_t *out, const prog_t *a, const prog_t *b);
/* Fuse validated BF16 pairs. disjoint_xout asserts that both peer-inbox tables
 * point to separate output storage, also disjoint from inputs and weights. */
int prog_pair_fuse_bf16(prog_t *P, int disjoint_xout);
/* Shared-weight BF16 bodies use eight positions. A B16/B32 phase is split into
 * two/four subgroups, with one initial wait and one final publication.
 * Unsupported operations retain their full-width causal scalar replay. */
int prog_batch_fuse_bf16(prog_t *P, int disjoint_xout);
/* Shared-weight Q2 B8 (B16/B32 split into independent eight-position bodies).
 * routed_disjoint asserts that routed weight arenas are
 * separate from every position's activations/outputs. Moesel supplies unique
 * local expert indices per position, NE<=10, in original rank order. */
int prog_batch_fuse_q2(prog_t *P, int routed_disjoint);
/* HC up shares one four-row BF16 tile across eight positions. The caller
 * guarantees all peer destinations are disjoint from inputs and local outputs,
 * and that position tables point to separate inbox storage. Wide phases retain
 * one initial wait and one final publication across their eight-position bodies. */
int prog_batch_fuse_hcup(prog_t *P, int disjoint_xout);
/* Opt-in grouped HCRX/W decomposition, before BF16 fusion. HCRX becomes RMS
 * plus HCX; HCRWX first uses HCWSUM(nd=1) as an out-of-place HCWRITE. Each
 * write_scratch position supplies10240 dead bytes for HCWSUM's checked Y copy.
 * The caller guarantees disjoint peer destinations; known local aliases are
 * checked here. B1/B2 descriptors stay unchanged. Returns phases decomposed,
 * or a negative value on allocation/capacity error without changing P. */
int prog_batch_decompose_hcr(prog_t *P, const CUdeviceptr *write_scratch, unsigned positions, int disjoint_xout);
void prog_free(prog_t *P);
void prog_nobar(prog_t *P);   /* no grid barrier after the last phase (independent next phase) */
phase_t *prog_add(prog_t *P, int op, unsigned units);
void ph_ptr(phase_t *p, int addr, CUdeviceptr v);
void ph_u32(phase_t *p, int addr, uint32_t v);
void ph_f32(phase_t *p, int addr, float v);
/* a token-relative u32 parameter: base + token_offset + tok (the die's token counter) when the program loops in the kernel, else value */
void ph_tokv(prog_t *P, phase_t *p, int addr, uint32_t base, uint32_t value);

/* ---- one phase per op (pointers are device addresses) ---- */
void ph_rms(prog_t *P, int op, CUdeviceptr X, CUdeviceptr W, CUdeviceptr Y, float eps, unsigned vectors);
void ph_bf16(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Y, unsigned rows, CUdeviceptr bias);
void ph_hc(prog_t *P, int op, CUdeviceptr P0, CUdeviceptr P1, CUdeviceptr P2, unsigned n, unsigned us, unsigned units);
void ph_actq(prog_t *P, int op, CUdeviceptr X, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors);
void ph_ffnact(prog_t *P, CUdeviceptr G, CUdeviceptr U, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors);
void ph_gemvk(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr AS);
void ph_q2(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS, CUdeviceptr WE, CUdeviceptr NE);
void ph_b32(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS);
void ph_conv1d(prog_t *P, CUdeviceptr X, CUdeviceptr STATE, CUdeviceptr W, CUdeviceptr U, unsigned t, unsigned n);
void ph_dnstep(prog_t *P, CUdeviceptr S, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr A, CUdeviceptr B, CUdeviceptr SSMA, CUdeviceptr O, float eps, unsigned hb, unsigned kb, unsigned heads);
void ph_dnnorm(prog_t *P, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, CUdeviceptr Y, float eps, unsigned heads);
void ph_router(prog_t *P, CUdeviceptr R, CUdeviceptr SH, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr S);
void ph_moesel(prog_t *P, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr MAP, CUdeviceptr LIDX, CUdeviceptr LW, CUdeviceptr NE, CUdeviceptr PART);
void ph_sumn(prog_t *P, CUdeviceptr SRC, unsigned stride, CUdeviceptr OUT, unsigned n, unsigned ndies);
/* a GEMV phase for any weight type by GGUF type and K (acc: y = fma(*scale, dot, y)) */
void ph_gemv_any(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr Xq, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr Y, unsigned rows, CUdeviceptr AS, int acc, CUdeviceptr scale);

/* the GEMV opcode for a GGUF type and K (0 if none), and whether the op is Q2_0 / b32 / K-quant */
int gemv_op(uint32_t type, int K, int acc);
int gemv_op_is_q2(int op);
int gemv_op_is_b32(int op);

/* ---- the exchange between dies ---- */
#define XSLOT (64u << 10)                 /* bytes per inbox slot (two slots per die) */
typedef struct {
	CUdeviceptr inbox;    /* 2 x XSLOT */
	CUdeviceptr flags;    /* u32[6]: last exchange completed by each die (absent dies 0xffffffff) */
	CUdeviceptr xgen;     /* u32: this die's exchange counter */
	CUdeviceptr xtab;     /* u64[12]: inbox base of die 0..5, flag base of die 0..5 (the same on every die) */
} xchg_t;
/* send this die's part (nbytes at src, at byte myoff of the slot) to every die and wait
 * for all parts: two phases; returns the slot's address in this die's inbox */
CUdeviceptr ph_xchg(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr src, unsigned nbytes, unsigned myoff, unsigned total);   /* total: bytes the slot holds once every part arrived */
CUdeviceptr ph_bf16x(prog_t *P, const xchg_t *x, int op, CUdeviceptr W, CUdeviceptr X, unsigned rows, unsigned myoff);   /* a GEMV_XOUT op: rows land in every die's slot */
CUdeviceptr ph_bf16rx(prog_t *P, const xchg_t *x, CUdeviceptr W, CUdeviceptr S, CUdeviceptr NW, CUdeviceptr XH, float eps, unsigned rows, unsigned myoff);   /* + RMSNorm(S) */
CUdeviceptr ph_hcup(prog_t *P, const xchg_t *x, CUdeviceptr W, CUdeviceptr DOWN, CUdeviceptr LUP, unsigned cols, CUdeviceptr XH, CUdeviceptr T, unsigned tn, unsigned myoff);   /* SiLU + hc up + mix -> slots */
CUdeviceptr ph_hcx(prog_t *P, const xchg_t *x, int op, CUdeviceptr P0, CUdeviceptr P1, unsigned n, unsigned us, unsigned units, unsigned myoff);
CUdeviceptr ph_dnnormx(prog_t *P, const xchg_t *x, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, float eps, unsigned heads, unsigned myoff);
/* end of a token program that loops in the kernel: an odd exchange count would make the
 * last exchange of one token and the first of the next share a slot, so pad with an empty one */
void prog_token_end(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr any);

/* ---- what die d of n owns in a block (expert split + row split, see SPEC) ---- */
typedef struct {
	int d, n;
	int dj0, djn;       /* hc down++inject rows [dj0, dj0+djn) of 324 */
	int cb0, cbn;       /* z column blocks (32 wide) [cb0, cb0+cbn) of 80; hc up rows l*2560 + 32 cb */
	int h0, hn;         /* value heads [h0, h0+hn) of 48 (gate rows 128 h, alpha/beta rows h, states) */
	int klo, khi, nk;   /* key heads (klo + j) % 16, j < nk (value head h uses key head h % 16): qkv rows q and k of these in that order, then v of [h0, h0+hn) */
	int y0, yn;         /* ssm_out / attn_output / ple_value rows (of 2560) */
	int pk0, pkn;       /* ple_key rows (of 10240) */
	int o0, on;         /* output.weight rows (of 248320) */
	int hd0, hdn;       /* output_hc_down rows (of 320) */
	int e0, en;         /* experts [e0, e0+en) of 512 */
	int shexp;          /* this die runs the shared expert (blk % n == d) */
} split_t;
void split_of(split_t *s, int d, int n, int blk);
/* local channel count of the die's qkv / conv1d: 128 (2 nk + hn) */
#define SPLIT_NLOC(s) (128u * (2u * (unsigned)(s)->nk + (unsigned)(s)->hn))

/* the die's tensors (the holder's per-die slices and merged rows: hc_*_dj = down ++
 * inject, ffn_gate_inp_x = router ++ shared gate, ssm_ab = alpha ++ beta with
 * ssm_ab_bias = dt_bias ++ zeros, emap = global expert -> local index or 0xffffffff) */
typedef struct {
	CUdeviceptr hc_attn_norm, hc_attn_dj, hc_attn_up;
	unsigned attn_q_split, attn_q_first, attn_q_rows;
	CUdeviceptr hc_ffn_norm, hc_ffn_dj, hc_ffn_up;
	CUdeviceptr attn_qkv, attn_gate, ssm_ab, ssm_ab_bias, ssm_conv1d, ssm_a, ssm_norm, ssm_out;
	uint32_t attn_qkv_type, attn_gate_type, ssm_out_type;
	CUdeviceptr ffn_gate_inp_x, ffn_gate_exps, ffn_up_exps, ffn_down_exps, emap;
	CUdeviceptr ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp;
	uint32_t ffn_gate_shexp_type, ffn_up_shexp_type, ffn_down_shexp_type;
	/* full-attention blocks (3, 7, ...): v1 keeps attn_q/k/v whole on die 0 only */
	CUdeviceptr attn_q, attn_k, attn_v, attn_q_norm, attn_k_norm, attn_output;
	/* QSA (the indexer of the layer): the q and k projections (BF16, 512 + 128 rows) and their norms (F32 128) */
	CUdeviceptr indexer_x, indexer_qn, indexer_kn;
	uint32_t attn_q_type, attn_k_type, attn_v_type, attn_output_type;
	/* block 1's PLE: ple_key Q2_0 (this die's rows of 10240), ple_value BF16 (rows of 2560), norms, conv F16 [10240][4] */
	CUdeviceptr ple_key, ple_value, ple_norm_key, ple_norm_query, ple_norm_conv, ple_conv1d;
	/* the head (output_hc read gate, output.weight Q5_K rows of this die) and the embedding (raw Q3_K rows, whole) */
	CUdeviceptr output_hc_norm, output_hc_down, output_hc_up, output, obases, token_embd;
} dn_weights_t;

/* activation / state buffers of one die (device); sizes in floats unless noted; the
 * l* buffers hold this die's part, the gathered vectors live in the inbox slots */
typedef struct {
	CUdeviceptr S;                    /* 10240 in/out stream (replicated) */
	CUdeviceptr S2;                   /* 10240 scratch stream: the attention side's write lands here (bf16hcrwx), the MoE end writes S */
	CUdeviceptr xh, t;                /* 10240; 324 = SiLU(0.25 down) ++ inj */
	CUdeviceptr ldown, lup, lz;       /* djn; 4 x 32 cbn; 32 cbn */
	CUdeviceptr zq, zs32, zs16, zas;  /* int16 2560, i32 80, i32 160, f32 80 */
	CUdeviceptr lqkv, lgate, la, lu, lo, lon;  /* nloc, 128 hn, 2 hn (a ++ b), nloc, 128 hn, 128 hn */
	CUdeviceptr oq, os32, os16, oas;  /* int16 6144, 192, 384, 192 */
	CUdeviceptr ly;                   /* yn */
	CUdeviceptr r, idx, w, s;         /* 513 (r ++ sh), u32 10, f32 10, f32 1 */
	CUdeviceptr lidx, lw, ne, part;   /* u32 10, f32 10, u32 1, 2560 (this die's partial) */
	CUdeviceptr g, gu;                /* 11 x 640 each */
	CUdeviceptr hq, hs32, hs16, has;  /* int16 11x640, i32 11x20, i32 11x40, f32 11x20 */
	CUdeviceptr out;                  /* 2560 (the reduced MoE output) */
	CUdeviceptr conv_state, dn_state; /* 4 x nloc ring, hn x 128 x 128 */
	/* attention (die 0): qg 24 x [q 256 | gate 256], k/v 512, qo 6144, caches (ref/attn.h
	 * layouts, readable >= n + 8192 + 64 positions; invalid rows need not be zero), partials pm/pd [24][8], po
	 * [24][8][256], aout 6144, RoPE tables cs/sn [positions][32] */
	CUdeviceptr qg, k, v, qo, kc, vc, pm, pd, po, aout, rope_cs, rope_sn;
	/* PLE (block 1): token ids (u32[positions]), the constants block (plerows PC layout),
	 * the host-mapped table, g 2560 (+ its quantisation), this die's K rows, kn/qn 10240,
	 * this die's V rows, G 10240, gates 4, conv input 10240, ring state [10][10240], rows debug */
	CUdeviceptr imap, img;   /* image input: imap[position] = a row of img (fp16 [row][2560]) or 0xffffffff for a token; 0: no images */
	CUdeviceptr ids, ple_pc, ple_tab, pg, pgq, pgs32, pgs16, pgas, lk, kn, qn, lv, pG, pgates, px, ple_state, prows;
	unsigned ple_compact;   /* 1: ple_tab holds the 16 rows of every position (16 pos + h), not the hashed table */
	CUdeviceptr ple_rdy; unsigned ple_rid;   /* ple_rid != 0: plerows waits until the u32 ple_rdy[pos] equals ple_rid (the host stages the rows of each position and then sets it: FLASH_PLE_DISK) */
	/* head: this die's logits, its argmax (idx, max bits) + partials/counter, the gathered pairs, the token (u32 idx, u32 max bits) */
	CUdeviceptr hlo, amo, apart, acnt, tokout, toks;   /* toks: u32[positions], the generated tokens */
	CUdeviceptr tkout, tkpart, tkcnt, rnd;   /* sampling: this die top-K list (32 x {idx, value bits}), the CTA lists, the counter, u[positions] floats */
	/* QSA attention (ref/qsa.h, ops attnqkq .. attnpartn): qsa = 1 makes prog_attn_block run the layer on the die whose kq is set.
	 * per layer: the int8 K / V caches kq, vq [position][2][256] with fp16 scales ks, vs [position][2][8], the indexer pool pkt
	 * (transposed fp16, 8192 B per 32 blocks), the raw key ring [4][128] f32; per die: iqk (640 f32), iq [4][128], sc (scores),
	 * hst (1280 u32 selection state), sel (u32 [nblk]); the compact caches kc, vc of the attended keys (ATTN_KC_FLOATS layout) are
	 * the ones above; nblk: blocks of 4 keys attended at most (QSA_NBLK, 512) */
	int qsa; unsigned nblk;
	CUdeviceptr kq, ks, vq, vs, pkt, ring, iqk, iq, sc, hst, sel;
	/* Validation-only raw pre-clamp maxima: u32[position][K/V][2 heads][8 groups].
	 * Null normally; only a QSA_KV_TRACE=1 cubin consumes this descriptor field. */
	CUdeviceptr kv_rawmax;
} dn_bufs_t;

void prog_deltanet_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps);
void prog_attn_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps);
/* the PLE prefix of block 1 (before its DeltaNet block program): S += PLE(S, n-gram rows of position t) */
void prog_ple_prefix(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps);
/* S = 4 lanes of token_embd[id] (every die) */
void prog_embed(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t);   /* S = token_embd[ids[t]] */
void prog_embed_bcast(prog_t *P, const xchg_t *x, int me, int ndies, int owner, const dn_weights_t *w, const dn_bufs_t *b, unsigned t);   /* token_embd on die `owner` only: it embeds and sends S, the others copy it */
/* the head: output_hc read gate -> z -> output.weight rows of this die -> argmax, the dies' (idx, max) pairs
 * gathered and combined identically everywhere into tokout = {idx, max bits} */
void prog_head(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, float eps, unsigned t, unsigned nprompt);   /* toks[t] = token; ids[t+1] = token once t + 1 >= nprompt */
/* attention blocks are every fourth block from 3 */
#define QSA_NBLK 512                 /* indexer.top_k (2048) / 4 keys per block */
#define BLK_IS_ATTN(blk) (((blk) & 3) == 3)
#endif
