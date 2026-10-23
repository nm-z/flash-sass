/* The prefill chunk program: every op of a block handles all npos positions of a chunk in one phase, on k-major slabs
 * [row][ldx] (a position per column, ldx a power of two >= 128: the GEMM contract of gen_gemm.lua, zero past npos), the
 * dense products as GEMMs (die ops gemm_*), the elementwise and per-position parts as slab ops (slab.rb) and dnstepc
 * (dnstep.sass DN_CHUNK). The split is the decode program's (split_t); the exchanges gather slab rows through a chunk
 * inbox of two slots of cslot bytes (xchg_t with its own inbox and xtab, the decode's flags and counter). */
#ifndef FLASH_PROG_CHUNK_H
#define FLASH_PROG_CHUNK_H
#include "prog.h"
#include "chunk_ops.h"

typedef struct {
	unsigned ldx;               /* slab row length (positions), a power of two >= 128 */
	CUdeviceptr S;              /* [10240][ldx] the stream, every die's own copy */
	CUdeviceptr XH;             /* [10240][ldx] RMS-normalised stream lanes */
	CUdeviceptr LDN;            /* [djn][ldx] this die's rows of hc down ++ inject */
	CUdeviceptr T;              /* [324][ldx] SiLU(0.25 down) ++ the 4 inject rows */
	CUdeviceptr LUP;            /* [4 cols][ldx] this die's 4 lane rows of hc up */
	CUdeviceptr LZ;             /* [cols][ldx] this die's z columns */
	CUdeviceptr LQKV, LU;       /* [nloc][ldx] q / k / v rows of the die's heads, then after conv1d + SiLU */
	CUdeviceptr LGATE, LO, LON; /* [hrows][ldx] the output gate, the DeltaNet output, its norm */
	CUdeviceptr LA;             /* [2 hn][ldx] alpha ++ beta */
	CUdeviceptr LY;             /* [yn][ldx] this die's rows of ssm_out (and of the MoE sum) */
	/* the MoE half (unit layout, kern/hold2.c) */
	CUdeviceptr ZT;             /* [ldx][2560] z position-major */
	CUdeviceptr R;              /* [513][ldx] the router logits ++ the shared-expert gate logit */
	CUdeviceptr IDX, RW, SW;    /* [ldx][10] u32 top ten, [ldx][10] their weights, [ldx] the shared-expert gate */
	CUdeviceptr LP, LW;         /* [10 ldx] the pairs grouped by expert: position, weight */
	CUdeviceptr ITEMS, NITEMS;  /* [512][4] u32 the die's expert work items, u32 count */
	CUdeviceptr G, U, H;        /* [10 ldx][192] the pairs' rows of the die's units: gate, up, SiLU(gate) up */
	CUdeviceptr PART;           /* [2560][ldx] int32 (value 2^20) this die's MoE partial */
	CUdeviceptr SG, SU, SH;     /* [640][ldx] the shared expert (its die only) */
	CUdeviceptr SD;             /* [2560][ldx] its down rows */
	unsigned nvu;               /* 0: the die's own units (dn_weights_t); else nvu unit lists run in turn (tests on fewer dies) */
	CUdeviceptr vu_g[6], vu_u[6], vu_d[6], vu_emap[6];   /* up to 6 (one die running all six lists) */
	/* the attention layers (cattn.rb) */
	CUdeviceptr LQ;             /* [nh 512][ldx] the die's query heads: q 256 then gate 256 each */
	CUdeviceptr LK, LV, LI;     /* [512][ldx] K, V, [640][ldx] the indexer (the layer's owner) */
	CUdeviceptr CB;             /* the owner's exchange: CKV [npos][1024] ++ CKS [npos][128] ++ CSEL [npos][1024] (CHUNK_CB_BYTES) */
	CUdeviceptr IQ;             /* [ldx][512] the indexer queries (owner) */
	CUdeviceptr LAO;            /* [nh 256][ldx] the die's heads' attention output */
	CUdeviceptr CS, SN;         /* [ldx + 3][32] the chunk's RoPE rows: row p + 3 = position t0 + p */
	CUdeviceptr Q6T;            /* [q6t_rows][ldx] the partial products of the sliced Q6_K K=6144 GEMMs and the K-split BF16 ones */
	unsigned q6t_rows;
	/* block 1's PLE prefix: PG the chunk's PLE rows (host-filled), LPK this die's ple_key rows, PL1 / PL2 [10240][ldx] (QN, G; KN and
	 * the conv input use XH) */
	CUdeviceptr PG, LPK, PL1, PL2;
} chunk_bufs_t;

/* an attention layer's chunk state on one die: the prompt cache of the KV heads its query heads use (cattn_app), the owner's
 * pooled keys */
typedef struct {
	CUdeviceptr R[2], RS[2];    /* KV head h: [T][512] codes, [T][16] f32 scales (0: the die has no head of it) */
	CUdeviceptr POOL;           /* [128][8192] f32 (owner) */
	unsigned nblk;
} chunk_attn_t;
#define CHUNK_CB_BYTES(npos) ((unsigned long long)(npos) * 2176ull)

/* bytes of the largest slab the chunk program gathers (ON: 6144 rows; the MoE reduce-scatter's nd x maxr rows are fewer), one chunk inbox slot */
#define CHUNK_SLOT_BYTES(ldx) (6144ull * 4ull * (unsigned long long)(ldx))

/* C[rows][ldx] = W X for a weight tile of any GGUF type (the holder's layout) against Xt[K][ldx] */
void ph_gemm(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr X, CUdeviceptr C, unsigned rows, unsigned npos, unsigned ldx);
/* the attention-side (Gated DeltaNet) hyper-block of a block over positions [t0, t0 + npos): S += the block's write */
void prog_chunk_deltanet_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                              unsigned t0, unsigned npos, float eps, unsigned cslot);
/* the MoE (FFN-side) hyper-block on the unit layout over the chunk's npos positions: S += the block's MoE write */
void prog_chunk_moe_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                         unsigned npos, float eps, unsigned cslot);
/* the attention-side hyper-block of an attention layer (QSA, ref/qsa.h) over positions [t0, t0 + npos): S += the block's write.
 * The owner (b->kq != 0) projects K / V / the indexer and runs cattn kv / idx / sel; its chunk K / V and selections go to every
 * die in one exchange; the dies with query heads (the head-aligned Q split, w->attn_q_first / rows) append their KV heads' K / V
 * to their prompt caches and attend. */
void prog_chunk_attn_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, const chunk_bufs_t *c,
                          const chunk_attn_t *a, unsigned t0, unsigned npos, float eps, unsigned cslot);
/* block 1's PLE prefix over positions [t0, t0 + npos) (kern/prog.c prog_ple_prefix on the chunk): S += PLE(S, the chunk's PLE rows
 * in c->PG); the conv ring is read from RIN and written to ROUT ([10][10240], kept apart: the chunk's ring writes would race
 * its reads) */
void prog_chunk_ple_prefix(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const chunk_bufs_t *c,
                           CUdeviceptr RIN, CUdeviceptr ROUT, unsigned t0, unsigned npos, float eps, unsigned cslot);
/* after all nd dies' programs P[d] (exchanges x[d]) are built: the fp16 exchanges' unpack extents and relayed ranges */
void prog_chunk_xh_patch(prog_t *P, const xchg_t *x, int ndies);
#endif
