/* engine: the reloadable half of the Flash-Next decoder (libengine.so). flashcore.c keeps the
 * contexts, the weight arenas and the PLE table resident and dlopen()s this library; engine_init
 * builds every device buffer and program-independent table for FLASH_MAXT positions, then
 * engine_session() serves one client's commands, one persistent kernel launch per die per response
 * (see the command list at engine_session). `reload [path]` replaces the library in seconds without
 * touching the weights or the table. Env: SPLIT_SHARES, ATTN_DIE (ref/split.c), FLASH_CUBIN. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "block.h"
#include "attn.h"
#include "ple.h"
#include "hold.h"
#include "prog.h"
#include "split.h"
#include "qsa.h"
#include "qsa_state.h"
#include <math.h>
#include <fenv.h>
#include "gguf.h"
#include "engine.h"
#include "vision.h"
#include "prefill.h"
#include "timings.h"
static __thread flash_timing last_timing;
static __thread int last_timing_valid;
static __thread double timing_request_start, timing_run_start;
#include "core.h"
#include "ple_read_pool.h"
#include "ple_read_reload.h"
#include <setjmp.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fcntl.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <limits.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>

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
typedef struct { CUdeviceptr p; CUcontext c; size_t bytes; } allocrec;
static allocrec *allocs; static int nallocs, capallocs; static size_t alloc_bytes;
static CUdeviceptr dalloc(size_t bytes) {
	CUdeviceptr p; CUcontext c;
	CK(cuMemAlloc(&p, bytes)); CK(cuCtxGetCurrent(&c));
	if (nallocs == capallocs) { capallocs = capallocs ? 2 * capallocs : 1024; allocs = realloc(allocs, sizeof *allocs * (size_t)capallocs); }
	allocs[nallocs++] = (allocrec){ p, c, bytes }; alloc_bytes += bytes;
	CK(cuMemsetD8(p, 0, bytes));
	return p;
}
/* Clone an activation allocation's shape, leaving the new position zero-filled.
 * Cached weights, positional KV, recurrent state and token arrays are shared. */
static size_t scratch_bytes(CUdeviceptr p,CUcontext context) {
    if (!p) return 0;
	for (int i = 0; i < nallocs; i++) if (allocs[i].p == p && allocs[i].c == context) return allocs[i].bytes;
	fprintf(stderr, "engine: unregistered paired activation allocation\n");
	longjmp(init_jb, 1);
}

/* ---- the engine: everything main() loads once, used by every response ---- */
static int MAXT; static float eps = 1e-6f;
static CUmodule mod[MAXD]; static CUfunction fn[MAXD];
static uint32_t *s_ids, *s_toks;   /* engine_session scratch (positions each) */
static dn_weights_t (*dw)[NB]; static dn_bufs_t db[MAXD]; static xchg_t xc[MAXD]; static split_t (*sp)[NB];
static dn_bufs_t db_pair[MAXD]; static xchg_t xc_pair[MAXD];
#ifdef FLASH_EXACT_BATCH
#include "exactbatch/prog_group.h"
static unsigned exact_width;
static dn_bufs_t db_exact[7][MAXD];
static xchg_t xc_exact[7][MAXD];
static CUmodule exact_mod[MAXD];
static CUfunction exact_fn[MAXD];
static char exact_cubin_path[512];
static uint64_t exact_cubin_fnv;
static int last_batched_positions;
#endif
#ifdef FLASH_VERIFY_NATIVE
static unsigned verify_accept;
static CUdeviceptr verify_dn[7][MAXD][NB],verify_conv[7][MAXD][NB],verify_ring[7][MAXD][NB],verify_ple[7][MAXD];
#endif
static unsigned program_capacity = 4096, active_program_positions = 1;
static unsigned allocated_positions = 1;
static int producer_quant,group_matrices,residual_fp8;
static int gr_replicate_ffn;
static uint32_t *phase_watch;
#include "phase/ownership.h"
#include "phase/gr_replica.h"
static unsigned gr_fusions;
static CUdeviceptr embedding_tables[8];
static CUdeviceptr producer_tables[8][MAXD][2],residual_encoded[8][MAXD][2];
static CUdeviceptr producer_local_tables[8][MAXD];
static CUdeviceptr phase_gr_sum_cache[MAXD];
static int gr_write_trace;
#ifdef FLASH_DFLASH_NATIVE
static int dflash_enabled;
static CUdeviceptr dflash_capture[2][MAXD][5];
static const int dflash_layers[5]={5,15,25,35,45};
#endif
static prog_t *request_programs[8];
static void release_request_programs(void) {
	for (unsigned i = 0; i < 8; i++) if (request_programs[i]) {
		for (int d = 0; d < MAXD; d++) prog_free(&request_programs[i][d]);
		free(request_programs[i]); request_programs[i] = NULL;
	}
}
static prog_t *new_request_programs(void) {
	for (unsigned i = 0; i < 8; i++) if (!request_programs[i]) {
		return request_programs[i] = calloc(MAXD, sizeof(prog_t));
	}
	return NULL;
}
static int two_position, prefill_chunk;       /* opt-in causal forward over two known prompt ids */
static int last_positions, last_paired_positions, last_prefill_chunks;
static CUdeviceptr dPROG[MAXD], dCTL[MAXD], (*dst_dn)[NB], (*dst_conv)[NB], (*dst_kq)[NB], (*dst_ks)[NB], (*dst_vq)[NB], (*dst_vs)[NB], (*dst_pkt)[NB], (*dst_ring)[NB], (*snap_ring)[NB], dst_ple[MAXD];
static size_t dn_bytes[MAXD][NB], conv_bytes[MAXD][NB], kcf, vcf;
static int kv_host = 1;                  /* FLASH_KV_HOST=0: the int8 KV caches in VRAM (small contexts) instead of pinned host memory */
static void *h_kv;                      /* the host-resident KV caches of all attention layers */
/* Validation-only raw group maxima. The ordinary cubin has no trace instructions. */
static int kv_trace, kv_trace_start, kv_trace_complete;
static uint32_t *h_kv_rawmax;
static CUdeviceptr d_kv_rawmax[MAXD];
#define KV_TRACE_WORDS 32u
static int nlayers_run = NB;   /* FLASH_LAYERS: run only the first blocks (bisecting a failure; the answers are meaningless) */
static unsigned nblk_qsa = QSA_NBLK;   /* blocks of 4 keys attended per layer and token (FLASH_NBLK: a small value exercises the selection on short prompts) */
static int has_attn[MAXD];             /* the die runs at least one attention layer */
static int rope_owner = -1;            /* optional shared table, addressed through enabled peer mappings */
static unsigned attn_q_split_mask;
static unsigned attn_q_split_mode;
#ifndef FLASH_REV
#define FLASH_REV "unknown"             /* the Makefile passes the commit (and -dirty) this library was built from */
#endif
static char cubin_path[512]; static uint64_t cubin_fnv;   /* identity of the loaded kernel image: FNV-1a 64 of the cubin file */
static const char *profile_op_name(unsigned op) {
    if(op<OP_COUNT)return die_opname[op];
    const char *names[]={"grwrite_read","dnnormq_x","hcupq_x","router_select","dnstep_normq","gdn_full","router_full","attnmergeq_x","qsa_select","qsa_score_select","qsa_part_merge","plegate_norm","ple_rowsq"};
    return op>=94 && op<107?names[op-94]:"extended";
}
static void check_fused_moe_module(CUmodule module) {
	CUfunction marker;
    if(getenv("FLASH_FUSED_GR") && atoi(getenv("FLASH_FUSED_GR")))CK(cuModuleGetFunction(&marker,module,"flash_gr_read_sum_abi_v1"));
    if(producer_quant)CK(cuModuleGetFunction(&marker,module,"flash_phase_fused_abi_v6"));
    if(producer_quant)CK(cuModuleGetFunction(&marker,module,"flash_phase_exchange_join_abi_v1"));
    if(producer_quant)CK(cuModuleGetFunction(&marker,module,"flash_phase_gr_cache_abi_v1"));
    if(producer_quant)CK(cuModuleGetFunction(&marker,module,"flash_phase_amax_schedule_abi_v1"));
    if(getenv("FLASH_XTRACE") && atoi(getenv("FLASH_XTRACE")))CK(cuModuleGetFunction(&marker,module,"flash_exchange_trace_abi_v2"));
    if(gr_write_trace)CK(cuModuleGetFunction(&marker,module,"flash_gr_write_trace_abi_v1"));
    if(gr_replicate_ffn)CK(cuModuleGetFunction(&marker,module,"flash_gr_local_abi_v1"));
    if(residual_fp8)CK(cuModuleGetFunction(&marker,module,"flash_fp8_residual_abi_v1"));
#ifdef FLASH_CHUNK_NATIVE
	const char *chunk_image = getenv("FLASH_CHUNK_CUBIN");
	if (getenv("FLASH_CHUNK_PREFILL") && atoi(getenv("FLASH_CHUNK_PREFILL")) > 0 && !(chunk_image && *chunk_image))
		CK(cuModuleGetFunction(&marker, module, "flash_chunk_abi_v2"));
#endif
	if (!getenv("FLASH_FUSED_MOE") || !atoi(getenv("FLASH_FUSED_MOE"))) return;
	CK(cuModuleGetFunction(&marker, module, "flash_fused_moe_abi_v1"));
}
static uint64_t file_fnv(const char *path) {
	FILE *f = fopen(path, "rb"); if (!f) return 0;
	uint64_t h = 0xcbf29ce484222325ull; unsigned char buf[65536]; size_t k;
	while ((k = fread(buf, 1, sizeof buf, f)) > 0) for (size_t i = 0; i < k; i++) { h ^= buf[i]; h *= 0x100000001b3ull; }
	fclose(f); return h;
}
static uint32_t *h_ids, *h_toks;    /* pinned: the prompt, and the tokens as die 0 writes them */
/* ---- FLASH_PLE_DISK: the 28.8 GB PLE table stays in its file. A token needs 16 rows of 90 B; the host reads them into the pinned compact table (16 rows per position) just
   ahead of the position that runs and then stores this response's id in a ready flag, which plerows (compact mode) waits for. ---- */
static int ple_disk, ple_fd = -1; static off_t ple_off; static uint8_t *ple_host; static size_t ple_rdy_off;
static char ple_cache_path[1024];
static off_t ple_source_bytes;
static int ple_cache_active;
/* A cache descriptor lives for one synchronous row batch. A RAM-cache owner
 * can then exit under pressure without this engine retaining the whole file. */
static int ple_cache_open(void) {
	if (!ple_cache_path[0]) return -1;
	struct stat copy;
	int fd=open(ple_cache_path,O_RDONLY|O_CLOEXEC);
	if (fd<0 || fstat(fd,&copy) || !S_ISREG(copy.st_mode) || copy.st_size!=ple_source_bytes) {
		if (fd>=0) close(fd);
		if (ple_cache_active) fprintf(stderr,"engine: PLE cache unavailable; using the original source\n");
		ple_cache_active=0;return -1;
	}
	posix_fadvise(fd,0,0,POSIX_FADV_RANDOM);
	ple_cache_active=1;return fd;
}
static struct ple_read_pool ple_reads;
static unsigned ple_readers = 1;
static uint32_t *h_rdy;                    /* the u32 ready flag of each position (pinned) */
static uint32_t ple_rid;                   /* library-local response id; core-owned ready flags outlive reloads */
static unsigned ple_ready_collisions,ple_ready_cleared;
static uint32_t ple_absent_id, ple_end_id; /* the constants block's absent / end ids: the hash of a position needs the same ones */
static uint32_t *ple_seq;                  /* the ids of the response so far: the prompt, then each generated token as it arrives */
static int ple_next, ple_start;            /* the first position whose rows are not staged yet; the position the response starts at */
static unsigned long long ple_short_reads; /* rows that could not be read (staged as zeros; the flag is set anyway so that no die waits for ever) */
#define PLE_AHEAD 128                      /* positions staged ahead of the one that runs */
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
static int embed_die;   /* the die that holds token_embd (the lowest one when several do) */
static CUdeviceptr dTRACE[MAXD]; static unsigned trace_n, trace_first, trace_valid;
static CUdeviceptr dXTRACE[MAXD];static unsigned xtrace_n,xtrace_first,xtrace_count;
static uint64_t trace_request;
static const char *trace_status = "none";
static uint64_t trace_requested;
static unsigned trace_completed, trace_aborted, trace_ended;
/* A malformed run command must replace the preceding trace identity even
 * though it never reaches engine_run. Valid runs advance their own identity. */
static void trace_reject_arguments(void) {
    trace_request++;
    trace_first = trace_valid = trace_completed = 0;
    trace_aborted = trace_ended = 0;
    trace_requested = 0;
    trace_status = "argument_error";
}
static int prof_layer[MAXD][4096];
typedef struct {
    uint64_t observed, callback_start, callback_end, staging_start, staging_end, ready;
    unsigned bytes, short_reads;
} host_trace;
static host_trace trace_host[256];
static int trace_slot(int p) { return p >= 0 && (unsigned)p >= trace_first && (unsigned)p - trace_first < trace_n ? (int)((unsigned)p - trace_first) : -1; }
static uint64_t host_ns(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec; }
static uint32_t prof_units[MAXD][4096], prof_flags[MAXD][4096];
static uint32_t prof_op[MAXD][4096], prof_index[MAXD][4096];
static unsigned prof_producer_fusions[MAXD],prof_matrix_groups[MAXD],prof_gr_fusions[MAXD],prof_local_fusions[MAXD],prof_publications[MAXD];
static int prof_nph[MAXD]; static unsigned prof_desc[MAXD];   /* every die's program of the last response, for the "prof" command */

/* Post-request determinism diagnosis: distinguish GPU-die disagreement from
 * streamed-token visibility. No model operation or sampling decision changes. */
static void token_audit(FILE *out) {
    unsigned start=last_timing.prompt>0?(unsigned)last_timing.prompt-1:0;
    unsigned end=last_timing.completed>0?(unsigned)last_timing.completed:0;
    if(!last_timing.counts_final || last_timing.error_code || end<=start || end>(unsigned)MAXT) {
        fprintf(out,"error: tokenaudit requires a completed request\n");return;
    }
    unsigned n=end-start;uint32_t *tokens=malloc((size_t)nd*n*4);
    if(!tokens){fprintf(out,"error: tokenaudit allocation\n");return;}
    for(int d=0;d<nd;d++){CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(tokens+(size_t)d*n,db[d].toks+4ull*start,4ull*n));}
    unsigned stream=0,peer=0,input=0;
    uint32_t *ids=malloc((size_t)n*4);
    if(!ids){free(tokens);fprintf(out,"error: tokenaudit input allocation\n");return;}
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(ids,db[d].ids+4ull*(start+1),4ull*n));
        for(unsigned i=0;i<n;i++)if(ids[i]!=tokens[(size_t)d*n+i]) {
            if(input<4)fprintf(out,"tokenaudit input position=%u die=%d generated=%u next_id=%u\n",start+i,d,tokens[(size_t)d*n+i],ids[i]);
            input++;
        }
    }
    free(ids);
    for(unsigned i=0;i<n;i++) {
        if(tokens[i]!=h_toks[start+i]) {if(stream<4)fprintf(out,"tokenaudit stream position=%u delivered=%u device=%u\n",start+i,h_toks[start+i],tokens[i]);stream++;}
        for(int d=1;d<nd;d++)if(tokens[(size_t)d*n+i]!=tokens[i]) {
            if(peer<4)fprintf(out,"tokenaudit peer position=%u die=%d gpu0=%u gpu_die=%u\n",start+i,d,tokens[i],tokens[(size_t)d*n+i]);
            peer++;
        }
    }
    for(int d=0;d<nd;d++) {
        uint64_t hash=UINT64_C(0xcbf29ce484222325);
        for(unsigned i=0;i<n;i++){hash^=tokens[(size_t)d*n+i];hash*=UINT64_C(0x100000001b3);}
        fprintf(out,"tokenaudit die=%d count=%u hash=%016llx\n",d,n,(unsigned long long)hash);
    }
    if(ple_disk)fprintf(out,"pleaudit rid=%u cleared=%u prior_equal=%u\n",ple_rid,ple_ready_cleared,ple_ready_collisions);
    fprintf(out,"tokenaudit_end count=%u stream_mismatches=%u peer_mismatches=%u input_mismatches=%u\n",n,stream,peer,input);free(tokens);
}

/* A validation-only readback, requested explicitly with `statehash`. Hash each
 * named buffer separately so paired/single passes can be compared beyond tokens. */
static uint64_t device_fnv(CUdeviceptr p, size_t bytes, uint64_t h) {
	unsigned char buf[65536];
	for (size_t at = 0; at < bytes; ) {
		size_t n = bytes - at < sizeof buf ? bytes - at : sizeof buf;
		CK(cuMemcpyDtoH(buf, p + at, n));
		for (size_t i = 0; i < n; i++) { h ^= buf[i]; h *= 0x100000001b3ull; }
		at += n;
	}
	return h;
}
static void state_hashes(FILE *out) {
	const uint64_t seed = 0xcbf29ce484222325ull;
	fprintf(out, "statehash positions=%d paired_positions=%d prefill_chunks=%d\n", last_positions, last_paired_positions, last_prefill_chunks);
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int bb = 0; bb < nlayers_run; bb++) {
			uint64_t h = seed;
			if (BLK_IS_ATTN(bb)) {
				if (!dst_kq[d][bb]) continue;
				h = device_fnv(dst_ring[d][bb], 2048, h);
				h = device_fnv(dst_kq[d][bb], (size_t)last_positions * 512, h);
				h = device_fnv(dst_vq[d][bb], (size_t)last_positions * 512, h);
				h = device_fnv(dst_ks[d][bb], (size_t)last_positions * 32, h);
				h = device_fnv(dst_vs[d][bb], (size_t)last_positions * 32, h);
				/* Full transposed tiles, including their finite zero/stale padding. */
				h = device_fnv(dst_pkt[d][bb], ((size_t)last_positions / 4 + 31) / 32 * 8192, h);
			} else {
				h = device_fnv(dst_dn[d][bb], dn_bytes[d][bb], h);
				h = device_fnv(dst_conv[d][bb], conv_bytes[d][bb], h);
			}
			fprintf(out, "statehash die=%d layer=%d state=%016llx\n", d, bb, (unsigned long long)h);
		}
		uint64_t ple = device_fnv(dst_ple[d], 4 * 10 * 10240, seed);
		uint64_t logits = device_fnv(db[d].hlo, 4u * (unsigned)sp[d][0].on, seed);
		fprintf(out, "statehash die=%d ple=%016llx logits=%016llx\n", d, (unsigned long long)ple, (unsigned long long)logits);
	}
	CK(cuCtxSetCurrent(ctx[0]));
	fprintf(out, "statehash done\n"); fflush(out);
}

/* Explicit validation readback; counts refer to 32-component groups, not values. */
static void kv_stats(FILE *out) {
	if (!kv_trace || !h_kv_rawmax || last_positions <= 0) {
		fprintf(out, "kvstats error=disabled_or_no_completed_positions\nkvstats done status=FAIL rows=0 positions=%d start=%d\n", last_positions, kv_trace_start);
		fflush(out); return;
	}
	size_t nscale = (size_t)last_positions * 16u;
	uint16_t *scales = malloc(nscale * sizeof *scales);
	if (!scales) {
		fprintf(out, "kvstats error=allocation\nkvstats done status=FAIL rows=0 positions=%d start=%d\n", last_positions, kv_trace_start);
		fflush(out); return;
	}
	int rows = 0, all_ok = kv_trace_complete && kv_trace_start == 0 && fegetround() == FE_TONEAREST;
	fprintf(out, "kvstats begin rev=%s cubin_fnv=%016llx positions=%d start=%d complete=%d rounding=%s counts=groups raw=pre_clamp_pre_floor\n",
	        FLASH_REV, (unsigned long long)cubin_fnv, last_positions, kv_trace_start, kv_trace_complete,
	        fegetround() == FE_TONEAREST ? "nearest_even" : "unsupported");
	for (int bb = 3; bb < nlayers_run; bb += 4) {
		int d = runtime_attn_owner(bb);
		if (d < 0 || d >= nd || !dst_ks[d][bb] || !dst_vs[d][bb]) { all_ok = 0; continue; }
		CK(cuCtxSetCurrent(ctx[d]));
		const uint32_t *raw = h_kv_rawmax + (size_t)((bb - 3) / 4) * (size_t)MAXT * KV_TRACE_WORDS;
		for (int kind = 0; kind < 2; kind++) {
			CUdeviceptr sptr = kind ? dst_vs[d][bb] : dst_ks[d][bb];
			CK(cuMemcpyDtoH(scales, sptr, nscale * sizeof *scales));
			unsigned long long groups = 0, missing = 0, zero = 0, underflow = 0, subnormal = 0;
			unsigned long long would_overflow = 0, clipped = 0, nonfinite = 0, bad_scales = 0, mismatches = 0;
			uint32_t max_bits = 0; int max_pos = 0, max_head = 0, max_group = 0;
			for (int t = 0; t < last_positions; t++) for (int g = 0; g < 16; g++) {
				uint32_t bits = raw[(size_t)t * KV_TRACE_WORDS + (unsigned)kind * 16u + (unsigned)g];
				if (bits == 0xffffffffu) { missing++; continue; }
				groups++;
				uint16_t s16 = scales[(size_t)t * 16u + (unsigned)g];
				if ((s16 & 0x7c00u) == 0x7c00u) bad_scales++;
				if (bits >= 0x7f800000u) { nonfinite++; continue; }
				float a; memcpy(&a, &bits, sizeof a);
				if (bits > max_bits) { max_bits = bits; max_pos = t; max_head = g / 8; max_group = g % 8; }
				if (!bits) zero++;
				if (bits && s16 == 0) underflow++;
				if (s16 > 0 && s16 < 0x0400u) subnormal++;
				if (a > QSA_KV_MAX_ABS) clipped++;
				const float tiny = 7.8886090522101180541e-31f;
				float raw_g = a < tiny ? tiny : a;
				uint16_t unbounded = qsa_f2h(raw_g * (1.0f / 127.0f));
				if ((unbounded & 0x7c00u) == 0x7c00u) would_overflow++;
				float bounded_g = a > QSA_KV_MAX_ABS ? QSA_KV_MAX_ABS : a;
				if (bounded_g < tiny) bounded_g = tiny;
				uint16_t expected = qsa_f2h(bounded_g * (1.0f / 127.0f));
				if (s16 != expected) mismatches++;
			}
			float max_abs; memcpy(&max_abs, &max_bits, sizeof max_abs);
			int ok = kv_trace_complete && kv_trace_start == 0 && fegetround() == FE_TONEAREST
			    && !missing && !nonfinite && !bad_scales && !mismatches;
			all_ok = all_ok && ok; rows++;
			fprintf(out, "kvstats layer=%d kind=%s die=%d groups=%llu expected_groups=%zu max_abs=%.9g max_hex=%a max_bits=%08x max_position=%d max_head=%d max_group=%d zero_groups=%llu underflow_groups=%llu subnormal_scale_groups=%llu would_overflow_groups=%llu clipped_groups=%llu nonfinite_input_groups=%llu nonfinite_stored_scales=%llu scale_mismatches=%llu missing_slots=%llu status=%s\n",
			        bb, kind ? "V" : "K", d, groups, nscale, (double)max_abs, (double)max_abs, max_bits,
			        max_pos, max_head, max_group, zero, underflow, subnormal, would_overflow, clipped,
			        nonfinite, bad_scales, mismatches, missing, ok ? "PASS" : "FAIL");
		}
	}
	CK(cuCtxSetCurrent(ctx[0])); free(scales);
	if (rows != 2 * (nlayers_run / 4)) all_ok = 0;
	fprintf(out, "kvstats done status=%s rows=%d positions=%d start=%d\n", all_ok ? "PASS" : "FAIL", rows, last_positions, kv_trace_start);
	fflush(out);
}

static dn_bufs_t *position_buf(int d, unsigned position) {
#ifdef FLASH_EXACT_BATCH
	if (position && exact_width) return &db_exact[position - 1][d];
#endif
	return position ? &db_pair[d] : &db[d];
}
static xchg_t *position_xchg(int d, unsigned position) {
#ifdef FLASH_EXACT_BATCH
	if (position && exact_width) return &xc_exact[position - 1][d];
#endif
	return position ? &xc_pair[d] : &xc[d];
}
static void clone_position(int d, dn_bufs_t *b, xchg_t *x, int alias_attention) {
	*b = db[d];
	*x = xc[d];
	struct { CUdeviceptr *destination; size_t bytes; } fields[96];
	unsigned count=0;size_t total=0;
#define CLONE(ptr) do { if(*(ptr)) { size_t bytes=scratch_bytes(*(ptr),ctx[d]); fields[count].destination=(ptr);fields[count++].bytes=bytes; total=((total+255)&~(size_t)255)+bytes+256; } } while(0)
#define SECOND(field) CLONE(&b->field)
	SECOND(S); SECOND(S2); SECOND(xh); SECOND(t);
	SECOND(ldown); SECOND(lup); SECOND(lz);
	SECOND(zq); SECOND(zs32); SECOND(zs16); SECOND(zas);
	SECOND(lqkv); SECOND(lgate); SECOND(la); SECOND(lu); SECOND(lo); SECOND(lon);
	SECOND(oq); SECOND(os32); SECOND(os16); SECOND(oas); SECOND(ly);
	SECOND(r); SECOND(idx); SECOND(w); SECOND(s);
	SECOND(lidx); SECOND(lw); SECOND(ne); SECOND(part);
	SECOND(g); SECOND(gu); SECOND(hq); SECOND(hs32); SECOND(hs16); SECOND(has); SECOND(out);
	SECOND(qg); SECOND(k); SECOND(v); SECOND(qo); SECOND(aout);
	/* Wide headless programs serialize gather/partial/merge per position. */
	if (!alias_attention) { SECOND(kc); SECOND(vc); SECOND(pm); SECOND(pd); SECOND(po); }
	SECOND(iqk); SECOND(iq); SECOND(sc); SECOND(hst); SECOND(sel);
	SECOND(pg); SECOND(pgq); SECOND(pgs32); SECOND(pgs16); SECOND(pgas);
	SECOND(lk); SECOND(kn); SECOND(qn); SECOND(lv); SECOND(pG); SECOND(pgates); SECOND(px); SECOND(prows);
	int copy_head = !alias_attention;
#ifdef FLASH_VERIFY_NATIVE
	copy_head |= verify_accept != 0;
#endif
	if (copy_head) {
		SECOND(hlo); SECOND(amo); SECOND(apart); SECOND(acnt); SECOND(tokout);
		SECOND(tkout); SECOND(tkpart); SECOND(tkcnt);
	}
#undef SECOND
	CLONE(&x->inbox);CLONE(&x->xtab);
#undef CLONE
	/* One arena avoids allocator granularity costs for dozens of small buffers
	 * per position. Keep zeroed guard space between logical activations. */
	CUdeviceptr arena=dalloc(total);size_t offset=0;
	for(unsigned i=0;i<count;i++) {
		offset=(offset+255)&~(size_t)255;*fields[i].destination=arena+offset;
		offset+=fields[i].bytes+256;
	}
	/* Each position has both exchange slots, while a logical paired exchange has
	 * one generation and one publication to the common peer flags. */
}
static void alloc_position(int d, unsigned position) {
	clone_position(d, position_buf(d, position), position_xchg(d, position), position >= 2);
}

/* A fresh sequence starts from allocation-equivalent scratch and cache padding.
 * Keep weights, pointer tables, image rows and RoPE tables intact. In particular,
 * speculative/vector loads must not observe a previous response's padding. */
static void reset_fresh_sequence(int d) {
    for(unsigned p=0;p<allocated_positions;p++) {
        dn_bufs_t *b=position_buf(d,p);xchg_t *x=position_xchg(d,p);
#define ZERO(field) do { if(b->field)CK(cuMemsetD8(b->field,0,scratch_bytes(db[d].field,ctx[d]))); } while(0)
        ZERO(S);ZERO(S2);ZERO(xh);ZERO(t);ZERO(ldown);ZERO(lup);ZERO(lz);
        ZERO(zq);ZERO(zs32);ZERO(zs16);ZERO(zas);
        ZERO(lqkv);ZERO(lgate);ZERO(la);ZERO(lu);ZERO(lo);ZERO(lon);
        ZERO(oq);ZERO(os32);ZERO(os16);ZERO(oas);ZERO(ly);
        ZERO(r);ZERO(idx);ZERO(w);ZERO(s);ZERO(lidx);ZERO(lw);ZERO(ne);ZERO(part);
        ZERO(g);ZERO(gu);ZERO(hq);ZERO(hs32);ZERO(hs16);ZERO(has);ZERO(out);
        ZERO(qg);ZERO(k);ZERO(v);ZERO(qo);ZERO(aout);ZERO(kc);ZERO(vc);ZERO(pm);ZERO(pd);ZERO(po);
        ZERO(iqk);ZERO(iq);ZERO(sc);ZERO(hst);ZERO(sel);
        ZERO(pg);ZERO(pgq);ZERO(pgs32);ZERO(pgs16);ZERO(pgas);
        ZERO(lk);ZERO(kn);ZERO(qn);ZERO(lv);ZERO(pG);ZERO(pgates);ZERO(px);ZERO(prows);
        ZERO(hlo);ZERO(amo);ZERO(apart);ZERO(acnt);ZERO(tokout);ZERO(tkout);ZERO(tkpart);ZERO(tkcnt);
#undef ZERO
        CK(cuMemsetD8(x->inbox,0,scratch_bytes(xc[d].inbox,ctx[d])));
    }
    if(phase_gr_sum_cache[d])CK(cuMemsetD8(phase_gr_sum_cache[d],0,16u*2560u*4u));
    for(int bb=0;bb<nlayers_run;bb++)if(BLK_IS_ATTN(bb) && dst_ring[d][bb]) {
        CK(cuMemsetD8(dst_ring[d][bb],0,scratch_bytes(dst_ring[d][bb],ctx[d])));
        CK(cuMemsetD8(dst_pkt[d][bb],0,scratch_bytes(dst_pkt[d][bb],ctx[d])));
        if(!kv_host) {
            CUdeviceptr cache[]={dst_kq[d][bb],dst_vq[d][bb],dst_ks[d][bb],dst_vs[d][bb]};
            for(unsigned k=0;k<4;k++)CK(cuMemsetD8(cache[k],0,scratch_bytes(cache[k],ctx[d])));
        }
    }
}

static void state_copy(int to_snapshot) {
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int bb = 0; bb < nlayers_run; bb++) {
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

#include "fp8/storage.h"
static CUdeviceptr gr_pointer(const phase_t *p,unsigned word) { uint64_t v;memcpy(&v,&p->w[word],8);return v; }
#include "phase/plan.h"
static unsigned fuse_gr_write_read(prog_t *p,int *layers,int die) {
    unsigned fused=0;
#ifdef FLASH_EXACT_BATCH
    if(!getenv("FLASH_FUSED_GR") || !atoi(getenv("FLASH_FUSED_GR")))return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_HCWSUM || b->w[0]!=OP_BF16HCRX || a->w[8]!=2560 || a->w[9]!=10240 || a->w[11]!=6 || gr_pointer(a,6)!=gr_pointer(b,4))continue;
        phase_t merged=*b;merged.w[0]=EXACT_OP_COUNT;
        ph_ptr(&merged,0x148,gr_pointer(a,14));ph_ptr(&merged,0x160,gr_pointer(a,6));
        ph_ptr(&merged,0x1a0,gr_pointer(a,4));ph_ptr(&merged,0x1a8,gr_pointer(a,2));
        if(producer_quant)ph_ptr(&merged,0x180,phase_gr_sum_cache[die]);
        merged.w[30]|=a->w[30]&2u;p->p[i]=merged;
        if(layers){layers[i]=layers[i+1];memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);}
        memmove(p->p+i+1,p->p+i+2,(size_t)(p->n-i-2)*sizeof *p->p);p->n--;fused++;
    }
#else
    (void)p;(void)layers;(void)die;
#endif
    return fused;
}

/* Both positions use the same layer state and positional caches, with separate
 * activations, router selections, compact gathers and exchange inboxes. */
static void model_program(prog_t *P, int d, unsigned position, unsigned nprompt,
                          const uint32_t stop[2], int samp_k, float topp, float invT,
                          int include_head, int checkpoints) {
	const dn_bufs_t *b = position_buf(d, position);
	const xchg_t *x = position_xchg(d, position);
	if (!position && !checkpoints) for (int p = 0; p < 4096; p++) prof_layer[d][p] = -1;
	prog_init(P); P->loop = 1; P->token_offset = position;
	P->stop[0] = stop[0]; P->stop[1] = stop[1]; P->end = dCTL[d] + 0x14;
	P->samp_k = samp_k; P->samp_topp = topp; P->samp_invT = invT;
	prog_embed_bcast(P, x, d, nd, embed_die, &dw[d][0], b, 0);
	for (int bb = 0; bb < nlayers_run; bb++) {
		int phase_first = P->n;
		dn_bufs_t layer = *b;
		layer.dn_state = dst_dn[d][bb]; layer.conv_state = dst_conv[d][bb]; layer.ple_state = dst_ple[d];
		qsa_bind_layer(&layer, nblk_qsa, dst_kq[d][bb], dst_ks[d][bb], dst_vq[d][bb], dst_vs[d][bb], dst_pkt[d][bb], dst_ring[d][bb]);
#ifdef FLASH_VERIFY_NATIVE
		if (checkpoints) {
			layer.checkpoint_copy_op = OP_VERIFY_COPY;
			layer.dn_checkpoint = position < 7 ? verify_dn[position][d][bb] : 0;
			layer.conv_checkpoint = position < 7 ? verify_conv[position][d][bb] : layer.conv_state;
			layer.ple_checkpoint = position < 7 ? verify_ple[position][d] : layer.ple_state;
			layer.ring_checkpoint = position < 7 ? verify_ring[position][d][bb] : layer.ring;
		}
#endif
		layer.kv_rawmax = kv_trace && layer.kq
		    ? d_kv_rawmax[d] + (size_t)((bb - 3) / 4) * (size_t)MAXT * KV_TRACE_WORDS * sizeof(uint32_t) : 0;
		if (bb == 1) prog_ple_prefix(P, &sp[d][bb], x, &dw[d][bb], &layer, 0, eps);
		if (BLK_IS_ATTN(bb)) prog_attn_block(P, &sp[d][bb], x, &dw[d][bb], &layer, 0, eps);
		else prog_deltanet_block(P, &sp[d][bb], x, &dw[d][bb], &layer, 0, eps);
#ifdef FLASH_DFLASH_NATIVE
		if(dflash_enabled)for(unsigned tap=0;tap<5;tap++)if(bb==dflash_layers[tap]) {
			phase_t *capture=prog_add(P,OP_XCOPY,10);
			ph_ptr(capture,0x140,b->S);ph_ptr(capture,0x148,dflash_capture[position][d][tap]);
		}
#endif
		if (!position && !checkpoints) for (int p = phase_first; p < P->n && p < 4096; p++) prof_layer[d][p] = bb;
	}
	if (include_head) prog_head(P, &sp[d][0], x, &dw[d][0], b, eps, 0, nprompt);
	prog_token_end(P, x, d, nd, b->S);
    int *layers=!position && !checkpoints?prof_layer[d]:NULL;
    gr_fusions=fuse_gr_write_read(P,layers,d);
    phase_producer_fusions=producer_fuse(P,position,d,layers);
    phase_local_fuse(P,layers);phase_attention_fuse(P,position,d,layers);
    phase_publish_fuse(P,layers);phase_gdn_fuse(P,layers);phase_router_fuse(P,layers);
    phase_q6_fuse(P);phase_shared_down_fuse(P);phase_shared_gate_group(P,b->r);
    phase_selection_fuse(P,layers);phase_score_selection_fuse(P);phase_part_merge_fuse(P);
    phase_qsa_empty_publish(P,layers);phase_qsa_independent_group(P);phase_ple_fuse(P,b,layers);
    phase_ple_rows_fuse(P,layers);
    phase_edges_fuse(P,position,d,layers);
    phase_matrix_groups=matrix_group(P);
    phase_reduce_waits(P);
    /* Diagnostic isolation of inter-phase overlap. The complete merged bodies
     * and 499 dispatches remain; only local join elision is disabled. */
    if(producer_quant && getenv("FLASH_STRICT_PHASE_JOINS") && atoi(getenv("FLASH_STRICT_PHASE_JOINS")))
        for(int i=0;i<P->n;i++)P->p[i].w[30]&=~(4u|PHASE_EXCHANGE_JOIN_FLAG);
    if(gr_write_trace) {
        unsigned slot=0;
        for(int i=0;i<P->n;i++)if(P->p[i].w[0]==94 || P->p[i].w[0]==OP_BF16HCRWX)P->p[i].w[21]=++slot;
        if(slot>128){fprintf(stderr,"GR trace exceeds128 records per position\n");longjmp(run_jb,1);}
    }
    if(residual_fp8){CUdeviceptr work[2]={b->S,b->S2};fp8_program(P,work,residual_encoded[position][d],2,10240,0,layers);}
}

#ifdef FLASH_EXACT_BATCH
static int fuse_exact_program(prog_t *out,int d);
static int build_exact_program(prog_t *out, const prog_t *first, int d,
                               unsigned nprompt, const uint32_t stop[2],
                               int samp_k, float topp, float invT) {
	prog_t extra[7] = {0};
	const prog_t *positions[8] = { first };
	for (unsigned p = 1; p < exact_width; p++) {
		model_program(&extra[p - 1], d, p, nprompt, stop, samp_k, topp, invT, getenv("FLASH_MTP_DRAFTS") && atoi(getenv("FLASH_MTP_DRAFTS"))>1, 0);
		positions[p] = &extra[p - 1];
	}
	/* Preserve position order through PLE and shared attention scratch. */
	prog_init(out);
	int rc = prog_group_aliased(out, positions, exact_width);
	for (unsigned p = 1; p < exact_width; p++) prog_free(&extra[p - 1]);
	if (rc) return rc;
	(void)fuse_exact_program(out,d);
	return 0;
}
static int fuse_exact_program(prog_t *out,int d) {
	int bf = prog_batch_fuse_bf16(out, 1);
	/* Production-input component timings show the routed B8 scan and large
	 * K2560 rows regress. Keep those grouped scalar operations until replaced. */
	int q2 = prog_batch_fuse_q2(out, 0);
	for (int i = 0; i < out->n; i += (int)phase_descriptors(out->p + i)) {
		phase_t *a = out->p + i;
		if (a->w[0] == OP_Q2K25608 && a->w[10] > 1024) {
			a->w[0] = OP_Q2K2560; a->w[30] &= ~PHASE_FUSED_FLAG; q2--;
		}
	}
	int dn = prog_batch_fuse_dnstep(out), act = prog_batch_fuse_actq(out);
	fprintf(stderr, "engine: exact batch die=%d width=%u descriptors=%d bf16=%d q2=%d dn=%d act=%d\n",
	        d, exact_width, out->n, bf, q2, dn, act);
	return dn;
}
#endif

static void upload_programs(const prog_t *P, unsigned *nph, unsigned positions) {
	active_program_positions = positions;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		nph[d] = (unsigned)P[d].n;
		if (nph[d] > program_capacity) { fprintf(stderr, "program too long (%u descriptors)\n", nph[d]); longjmp(run_jb, 1); }
        prof_desc[d] = nph[d]; prof_nph[d] = 0;
        prof_producer_fusions[d]=prof_matrix_groups[d]=prof_gr_fusions[d]=prof_local_fusions[d]=prof_publications[d]=0;
        for (int i = 0; i < P[d].n; ) {
#ifdef FLASH_EXACT_BATCH
            unsigned step = phase_descriptors(&P[d].p[i]);
#else
            unsigned step = (P[d].p[i].w[30] & 8u) ? 2u : 1u;
#endif
            if (step > (unsigned)(P[d].n - i) || prof_nph[d] >= 4096) { fprintf(stderr,"invalid logical phase span\n");longjmp(run_jb,1); }
            int j = prof_nph[d]++;
            prof_index[d][j] = (uint32_t)i; prof_op[d][j] = P[d].p[i].w[0];
            prof_units[d][j] = P[d].p[i].w[1]; prof_flags[d][j] = P[d].p[i].w[30];
            unsigned opcode=P[d].p[i].w[0],flags=P[d].p[i].w[30];
            prof_producer_fusions[d]+=opcode==95 || opcode==96 || opcode==98 || opcode==99 || opcode==101;
            prof_matrix_groups[d]+=(flags&8u) && !(flags&(16u|32u));
            prof_gr_fusions[d]+=opcode==94;prof_local_fusions[d]+=opcode==97 || opcode==98 || opcode==99 || opcode==100;
            prof_publications[d]+=(flags&128u)!=0;
            i += (int)step;
        }
        if(producer_quant && (positions!=1 || prof_nph[d]>=500)) {
            fprintf(stderr,"phase candidate not runnable: die %d has %d phases, %u positions; require single-position production decode below 500 phases on every die\n",d,prof_nph[d],positions);
            longjmp(run_jb,1);
        }
		CK(cuMemcpyHtoD(dPROG[d], P[d].p, sizeof(phase_t) * (size_t)P[d].n));
		CK(cuMemsetD32(dCTL[d] + 0x24, positions, 1));
	}
}

/* ---- persistent launches bounded by snapshots and known-prefix tails ---- */
typedef struct {
	int T, nprompt, ntok, sent, count, ended, aborted;
	int paired_positions, prefill_chunks, batched_positions;
	double tfirst;
	const engine_cb *cb;
} runctx;

static void abort_run(runctx *r) {
	/* the client is gone: every die ends the turn when its token count reaches a value a few tokens ahead of all of
	   them (the dies are within one token of each other), so they stop together */
	last_timing.aborted = 1;
	r->aborted = 1; trace_aborted = 1; *h_abort = (uint32_t)r->count + 8;
	if (*h_abort < (uint32_t)r->ntok) for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoDAsync(dCTL[d] + 0x18, h_abort, 4, pollst[d])); CK(cuStreamSynchronize(pollst[d])); }
	CK(cuCtxSetCurrent(ctx[0]));
}

/* The callback span ends at its return. Cancellation propagation can synchronize
   several device streams, so it belongs to run/completion time, not delivery. */
static void deliver_token(runctx *r, int position, uint32_t token) {
	int ti = trace_slot(position);
	if (ti >= 0) trace_host[ti].callback_start = host_ns();
	int cancel = r->cb && r->cb->token ? r->cb->token(r->cb->arg, position, token) : 0;
	if (ti >= 0) trace_host[ti].callback_end = host_ns();
	if (cancel && !r->aborted && !r->ended) abort_run(r);
}

/* stage the PLE rows of the positions that are known and close ([ple_next, min(known, finished + PLE_AHEAD)), at most `budget` of them): position q is known when
   q < nprompt or the token q - 1 has arrived (q <= count); its 16 row numbers are the hash of ids[q-2..q].
   WILLNEED hints precede a batch of parallel row reads; the flag is stored after every reader completes. */
static void ple_pump(runctx *r, int budget) {
	if (!ple_disk) return;
	int c = r->count < ple_start ? ple_start : r->count;
	int known = r->nprompt > c + 1 ? r->nprompt : c + 1;
	if (known > r->T) known = r->T;
	int lim = c + PLE_AHEAD < known ? c + PLE_AHEAD : known;
	for (; ple_next < lim && budget > 0; ple_next++, budget--) {
		int q = ple_next;
		int ti = trace_slot(q); if (ti >= 0) trace_host[ti].staging_start = host_ns();
		if (q >= r->nprompt) ple_seq[q] = h_toks[q - 1];
		uint64_t rows[16]; uint32_t endid = ple_end_id;
		ple_rows_c(ple_seq, (size_t)q, ple_absent_id, &endid, 1, ple_multipliers, ple_head_offsets, ple_head_vocab, rows);
		uint8_t *dst = ple_host + (size_t)q * 16 * 90;
		off_t offsets[PLE_READ_HEADS]; struct ple_read_result results[PLE_READ_HEADS];
		int cache_fd=ple_cache_open(), read_fd=cache_fd>=0?cache_fd:ple_fd;
		for (int h = 0; h < PLE_READ_HEADS; h++) {
			offsets[h] = ple_off + (off_t)(rows[h] * 90);
			posix_fadvise(read_fd, offsets[h], PLE_READ_ROW_BYTES, POSIX_FADV_WILLNEED);
		}
		ple_read_batch(&ple_reads, read_fd, offsets, dst, results);
		if (cache_fd>=0) {
			int bad=0;for (int h=0;h<PLE_READ_HEADS;h++) bad|=results[h].error || results[h].got!=PLE_READ_ROW_BYTES;
			close(cache_fd);
			if (bad) {
				ple_cache_active=0;
				ple_read_batch(&ple_reads,ple_fd,offsets,dst,results);
			}
		}
		for (int h = 0; h < 16; h++) {
			size_t got = results[h].got;
			if (ti >= 0) { trace_host[ti].bytes += (unsigned)got; trace_host[ti].short_reads += got < 90; }
			if (got < 90) { memset(dst + h * 90 + got, 0, 90 - got); if (!ple_short_reads++) fprintf(stderr, "engine: PLE row read failed at position %d (head %d)\n", q, h); }
		}
		__atomic_store_n(&h_rdy[q], ple_rid, __ATOMIC_RELEASE);
		if (ti >= 0) trace_host[ti].ready = trace_host[ti].staging_end = host_ns();
	}
}

/* launch the program on every die for tokens up to ntok and wait for it, streaming the generated tokens */
static void run_segment(runctx *r, const unsigned *nph, int ntok_) {
	r->ntok = ntok_;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuMemsetD32(dCTL[d]+0x2c,0,1));
		unsigned ntok = (unsigned)ntok_; void *params[] = { &dPROG[d], (void *)&nph[d], &dCTL[d], &ntok };
		CUfunction kernel = fn[d];
#ifdef FLASH_EXACT_BATCH
		if (exact_width && active_program_positions > 1) kernel = exact_fn[d];
#endif
		CK(cuLaunchKernel(kernel, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	}
	/* poll ctl[2] on die 0 through a non-blocking stream (the kernel runs on the legacy one), then the new tokens;
	   once the kernel is done, one last read */
	CK(cuCtxSetCurrent(ctx[0]));
	struct timespec nap = { 0, ple_disk ? 50000 : 500000 };   /* with the PLE rows staged on demand a new token is learned of as soon as possible: the next position's rows wait on it */
	double last_alive = now();
	int warned = 0, phase_reported = 0;
	double last_chk = now(), last_move = now(), watchdog_s = getenv("FLASH_WATCHDOG_S") ? atof(getenv("FLASH_WATCHDOG_S")) : 600.0; int last_count = -1;
	for (int fin = 0; !fin; ) {
		CUresult q = cuStreamQuery(0);
		if (q != CUDA_SUCCESS && q != CUDA_ERROR_NOT_READY) CK(q);
		fin = q == CUDA_SUCCESS;
		CK(cuMemcpyDtoHAsync(h_tok, dCTL[0] + 0x8, 4, pollst[0])); CK(cuStreamSynchronize(pollst[0]));
		r->count = (int)(*h_tok & 0x7fffffffu); r->ended = (int)(*h_tok >> 31);
		if (r->count > r->T) r->count = r->T;
        /* TLS checkpoints survive CUDA longjmp; counters are a lower bound until final sync. */
        last_timing.completed=r->count;last_timing.stop_known=0;
        last_timing.aborted=r->aborted;last_timing.counts_known=1;
        last_timing.run_ms=(now()-timing_run_start)*1e3;
		trace_completed = (unsigned)r->count; trace_ended = r->ended != 0;
		if (r->count > r->sent) {
			CK(cuMemcpyDtoHAsync(h_toks + r->sent, db[0].toks + 4u * (unsigned)r->sent, 4 * (size_t)(r->count - r->sent), pollst[0])); CK(cuStreamSynchronize(pollst[0]));
			if (!r->tfirst) r->tfirst = now();
            last_timing.first_valid=1;last_timing.first_ms=(r->tfirst-timing_run_start)*1e3;
			uint64_t observed = trace_n ? host_ns() : 0;
			for (; r->sent < r->count; r->sent++) {
				int ti = trace_slot(r->sent);
				if (ti >= 0) trace_host[ti].observed = observed;
				deliver_token(r, r->sent, h_toks[r->sent]);
			}
		}
		if (!fin) ple_pump(r, 8);
		if (!r->aborted && !fin && r->cb && r->cb->alive && now() - last_alive > 0.02) { last_alive = now(); if (r->cb->alive(r->cb->arg) && !r->ended) abort_run(r); }
		/* a die whose kernel failed (an illegal address, a trap) ends its launch while the others spin in an exchange: look at all of them, and give
		   up when the token count has not moved for FLASH_WATCHDOG_S seconds (a deadlock) */
		if (!fin && now() - last_chk > 0.05) {
			last_chk = now();
			for (int d = 1; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CUresult qd = cuStreamQuery(0); if (qd != CUDA_SUCCESS && qd != CUDA_ERROR_NOT_READY) { fprintf(stderr, "engine: die %d failed\n", d); CK(qd); } if (qd == CUDA_SUCCESS && !r->aborted && !warned) { warned = 1; fprintf(stderr, "engine: die %d finished its launch (token count %d of %d)\n", d, r->count, r->ntok); } }
			CK(cuCtxSetCurrent(ctx[0]));
			if (r->count != last_count) { last_count = r->count; last_move = now(); }
            if(phase_watch && !phase_reported && now()-last_move>2.0) {
                phase_reported=1;
                for(int d=0;d<nd;d++) {
                    CK(cuCtxSetCurrent(ctx[d]));
                    CK(cuMemcpyDtoHAsync(phase_watch,dCTL[d]+0x100,4u*nph[d],pollst[d]));
                    CK(cuStreamSynchronize(pollst[d]));
                    int finished=-1;
                    for(int j=0;j<prof_nph[d];j++)if(phase_watch[prof_index[d][j]])finished=j;
                    fprintf(stderr,"phasewatch die=%d token=%d last_completed=%d phases=%d next=%s\n",d,r->count,finished,prof_nph[d],finished+1<prof_nph[d]?profile_op_name(prof_op[d][finished+1]):"token-boundary");
                }
                CK(cuCtxSetCurrent(ctx[0]));
            }
			if (now() - last_move > watchdog_s) { fprintf(stderr, "engine: no progress for %.0f s at token %d: giving up\n", watchdog_s, r->count); longjmp(run_jb, 1); }
		}
		if (!fin) nanosleep(&nap, NULL);
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
	for (int d=0;d<nd;d++) {
		uint32_t limit=0;CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(&limit,dCTL[d]+0x2c,4));
		if (limit) {fprintf(stderr,"engine: die %d reached the kernel time limit\n",d);longjmp(run_jb,1);}
	}
	if (getenv("FLASH_VERIFY_STREAM") && atoi(getenv("FLASH_VERIFY_STREAM")) && r->count >= r->nprompt) {
		unsigned start = (unsigned)r->nprompt - 1, n = (unsigned)r->count - start, bad = 0;
		uint32_t *final = malloc((size_t)n * 4);
		if (!final) { fprintf(stderr,"engine: stream verification allocation failed\n"); longjmp(run_jb,1); }
		CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(final, db[0].toks + 4ull * start, (size_t)n * 4));
		for (unsigned i = 0; i < n; i++) if (final[i] != h_toks[start+i]) {
			if (bad < 3) fprintf(stderr,"engine: stream mismatch position=%u delivered=%u final_device=%u\n",start+i,h_toks[start+i],final[i]);
			bad++;
		}
		fprintf(stderr,"engine: streamcheck positions=%u mismatches=%u\n",n,bad);
		free(final);
	}
}

/* The paired program may only consume known ids. Stop before the last prompt
 * position, whose head produces the first generated token. */
static void run_known_prefix(runctx *r, const prog_t *single, const prog_t *pair, const prog_t *batch, int end) {
	unsigned nph[MAXD];
	if (r->aborted || r->count >= end) return;
#ifdef FLASH_EXACT_BATCH
	if (batch && exact_width && end - r->count >= (int)exact_width) {
		int start = r->count, limit = end - (end - start) % (int)exact_width;
		upload_programs(batch, nph, exact_width); run_segment(r, nph, limit);
		r->batched_positions += r->count - start;
	}
#else
	(void)batch;
#endif
	if (pair && end - r->count >= 2) {
		int start = r->count;
		int paired_end = end - ((end - r->count) & 1);
		upload_programs(pair, nph, 2);
		run_segment(r, nph, paired_end);
		r->paired_positions += r->count - start;
	}
	if (!r->aborted && r->count < end) {
		upload_programs(single, nph, 1);
		run_segment(r, nph, end);
	}
}

/* Bound headless prompt work by both the requested chunk and snapshot end.
 * Poll at every host boundary, even if a fast GPU launch finished before its
 * in-flight poll ran. No generated ID is admitted to this program. */
static void run_prefill_prefix(runctx *r, const prog_t *full,
                               const prog_t *prefix, const prog_t *pair, const prog_t *batch, int end) {
    if (!prefix) { run_known_prefix(r, full, pair, NULL, end); return; }
    while (!r->aborted && r->count < end) {
        if (r->cb && r->cb->alive && r->cb->alive(r->cb->arg)) {
            r->aborted = 1;
            break;
        }
        prefill_chunk_t chunk;
        if (prefill_plan_chunk(r->count, r->nprompt, end, prefill_chunk,
                               pair != NULL, &chunk) != 1) {
            fprintf(stderr, "engine: invalid prefill boundary at %d of %d\n", r->count, end);
            longjmp(run_jb, 1);
        }
        run_known_prefix(r, prefix, pair, batch, chunk.end);
        if (r->count == chunk.end) r->prefill_chunks++;
        else if (!r->aborted) {
            fprintf(stderr, "engine: prefill stopped at %d before %d\n", r->count, chunk.end);
            longjmp(run_jb, 1);
        }
    }
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
	for (int d = 0; d < nd; d++) if (has_attn[d] && (rope_owner < 0 || d == rope_owner)) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(db[d].rope_cs, cs, 4 * 32 * (size_t)T)); CK(cuMemcpyHtoD(db[d].rope_sn, sn, 4 * 32 * (size_t)T)); }
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

#ifdef FLASH_CHUNK_NATIVE
#include "chunk/engine_chunk.c"
#endif
#ifdef FLASH_MTP_NATIVE
#include "mtp/engine_mtp.c"
#endif
#ifdef FLASH_DFLASH_NATIVE
#include "dflash/engine_dflash.c"
#endif
#ifdef FLASH_VERIFY_NATIVE
#include "exactbatch/engine_verify.c"
#endif

int engine_run(const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int snap_at, const engine_img *imgs, int nimg, const engine_cb *cb, run_stats *st) {
    kv_trace_complete = 0;
    timing_request_start=now();timing_run_start=0;
    trace_request++;
    last_timing=(flash_timing){.request_id=trace_request,.prompt=nprompt,.error_code=-1,.build_ms=NAN,.run_ms=NAN};
    last_timing_valid=1;
    trace_first=nprompt>0?(unsigned)(nprompt-1):0;
    trace_valid=trace_completed=trace_aborted=trace_ended=0;
    trace_requested=nprompt>0 && n_new>=0?(uint64_t)(unsigned)nprompt+(unsigned)n_new:0;
    trace_status="argument_error";
    memset(trace_host,0,sizeof trace_host);
    if(!flash_timing_request_fits(nprompt,n_new,MAXT))return -1;
    int T=nprompt+n_new;
    trace_status="running";
	/* a CUDA error from here on unwinds to this point: the contexts are no longer usable, the core rebuilds them (return -2) */
	in_run = 1;
    if(setjmp(run_jb)) {
        release_request_programs();
        trace_status="device_error";in_run=0;snap_s=0;last_timing.error_code=-2;last_timing.counts_final=0;
        if(timing_run_start)last_timing.run_ms=(now()-timing_run_start)*1e3;
        else last_timing.build_ms=(now()-timing_request_start)*1e3;
        return -2;
    }
	if (kv_trace) {
		for (int l = 0; l < 12; l++)
			memset(h_kv_rawmax + (size_t)l * (size_t)MAXT * KV_TRACE_WORDS, 0xff,
			       (size_t)T * KV_TRACE_WORDS * sizeof(uint32_t));
	}
	memcpy(h_ids, ids, 4 * (size_t)nprompt); memset(h_ids + nprompt, 0, 4 * (size_t)(T + 2 - nprompt));
	int irc = prepare_images(imgs, nimg, nprompt, T);
	if (irc) { trace_status="image_error";in_run = 0; last_timing.error_code=irc;last_timing.build_ms=(now()-timing_request_start)*1e3;return irc; }
	int samp_k = (sm && sm->temp > 0.f && sm->top_k >= 2) ? (sm->top_k > 32 ? 32 : sm->top_k) : 0;
#ifdef FLASH_VERIFY_NATIVE
	if (verify_accept && (nimg || snap_at || snap_s || samp_k || nprompt < 25)) {
		fprintf(stderr,"engine: rollback audit requires at least25 prompt tokens, greedy text, no snapshots\n");
		trace_status="argument_error";in_run=0;last_timing.error_code=-1;
		last_timing.build_ms=(now()-timing_request_start)*1e3;return -1;
	}
#endif
#ifdef FLASH_MTP_NATIVE
	if (mtp_enabled && (nimg || snap_at || snap_s || samp_k)) {
		fprintf(stderr,"engine: MTP shadow mode supports greedy text without snapshots\n");
		trace_status="argument_error";in_run=0;last_timing.error_code=-1;
		last_timing.build_ms=(now()-timing_request_start)*1e3;return -1;
	}
#endif
	float samp_topp = sm && sm->topp > 0.f ? sm->topp : 1.0f, samp_invT = samp_k ? 1.0f / sm->temp : 1.0f;
	if (samp_k) {   /* u[t] in [0, 1): splitmix64 over the seed, 24 bits each */
		uint64_t z0 = sm->seed;
		for (int t = 0; t < T + 2; t++) { uint64_t z = (z0 += 0x9e3779b97f4a7c15ULL); z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL; z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL; z ^= z >> 31; h_rnd[t] = (float)(z >> 40) / 16777216.0f; }
	}
	/* the prefix cache: resume from the snapshot if the prompt starts with its tokens (at least one prompt token must
	   still run, to produce the logits of the last one) */
	int resume = (snap_s > 0 && snap_s <= nprompt - 1 && !memcmp(ids, snap_ids, 4 * (size_t)snap_s)) ? snap_s : 0;
    last_timing.resumed=resume;last_timing.completed=resume;last_timing.counts_known=1;last_timing.error_code=0;
	kv_trace_start = resume;
	if (snap_at > nprompt - 1) snap_at = nprompt - 1;
	for (int i = 0; i < nimg; i++) if (snap_at > imgs[i].s) snap_at = imgs[i].s;   /* the snapshot would not know the image */
	if (snap_at <= resume) snap_at = 0;
	if (!resume) snap_s = 0;   /* this response rewrites the attention caches from position 0 */
	if (ple_disk) {   /* a new id for this response (never 0); the rows of the positions it runs are staged from `resume` on */
		ple_rid = ple_rid + 1 ? ple_rid + 1 : 1; ple_start = ple_next = resume; ple_short_reads = 0;
        /* The core's pinned table survives dlopen/reload, but ple_rid does not.
         * Invalidate the executed span before launching any GPU reader so an
         * old response with the same id cannot satisfy this response's wait. */
        ple_ready_collisions=0;ple_ready_cleared=(unsigned)(T-resume);
        for(int q=resume;q<T;q++)ple_ready_collisions+=h_rdy[q]==ple_rid;
        memset(h_rdy+resume,0,4u*(size_t)(T-resume));
        __atomic_thread_fence(__ATOMIC_RELEASE);
		memcpy(ple_seq, ids, 4 * (size_t)nprompt);
		for (int d = 0; d < nd; d++) for (unsigned p = 0; p < allocated_positions; p++) position_buf(d, p)->ple_rid = ple_rid;
	}
	prog_t *P = new_request_programs(); unsigned nph[MAXD];
	prog_t *pair = two_position ? new_request_programs() : NULL;
    /* gen 0 is the existing teacher-forced, full-head diagnostic interface. */
    int need_prefix = prefill_chunk && (n_new > 0 || (cb && cb->token));
#ifdef FLASH_VERIFY_NATIVE
    if (verify_accept) need_prefix = 1;
#endif
#ifdef FLASH_MTP_NATIVE
    if (mtp_enabled && (n_new > 0 || (cb && cb->token))) need_prefix = 1;
#endif
    prog_t *prefix = need_prefix ? new_request_programs() : NULL;
    prog_t *batch = NULL;
    int need_batch = 0;
#ifdef FLASH_EXACT_BATCH
    need_batch = exact_width && need_prefix && !nimg && nprompt - 1 - resume >= (int)exact_width;
#ifdef FLASH_VERIFY_NATIVE
    if (verify_accept) need_batch = 0;
#endif
    if (need_batch) batch = new_request_programs();
    last_batched_positions = 0;
#endif
    prog_t *verification = NULL;int need_verification=0;
#ifdef FLASH_VERIFY_NATIVE
    need_verification=verify_accept!=0;
    if(need_verification)verification=new_request_programs();
    verify_executed=verify_replayed=verify_changed=0;
#endif
    if (!P || (two_position && !pair) || (need_prefix && !prefix) || (need_batch && !batch) || (need_verification && !verification)) {
        release_request_programs();
        /* No snapshot has been restored or current-request counter observed.
           Allocation failure must not retain the provisional success record. */
        last_timing.error_code = -4;
        last_timing.counts_known = last_timing.counts_final = 0;
        last_timing.build_ms = (now() - timing_request_start) * 1e3;
        trace_status = "allocation_error";
        in_run = 0; return -4;
    }
	if (resume) state_copy(0);
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		/* A fresh response resets recurrent state; attention rows are overwritten
		   before becoming visible. Snapshot resume restores recurrent state and the
		   unfinished indexer ring, preserving prefix attention rows. Reset the end
		   flag and abort count in either case. */
		if (!resume) {
            reset_fresh_sequence(d);
			for (int bb = 0; bb < nlayers_run; bb++) {
				/* the int8 caches, the indexer pool and the ring of an attention layer need no reset: positions are written before they are read */
				if (BLK_IS_ATTN(bb)) continue;
				else { CK(cuMemsetD8(dst_dn[d][bb], 0, dn_bytes[d][bb])); CK(cuMemsetD8(dst_conv[d][bb], 0, conv_bytes[d][bb])); }
			}
			CK(cuMemsetD8(dst_ple[d], 0, 4 * 10 * 10240));
		}
		CK(cuMemsetD32(dCTL[d] + 0x8, (unsigned)resume, 1)); CK(cuMemsetD32(dCTL[d] + 0x14, 0, 2));   /* token count, end flag, abort count */
		CK(cuMemcpyHtoD(db[d].ids, h_ids, 4 * ((size_t)T + 2)));
		CK(cuMemsetD8(db[d].toks, 0, 4 * ((size_t)T + 2)));
		if (samp_k) CK(cuMemcpyHtoD(db[d].rnd, h_rnd, 4 * ((size_t)T + 2)));
		model_program(&P[d], d, 0, (unsigned)nprompt, stop, samp_k, samp_topp, samp_invT, 1, 0);
		if (prefix) model_program(&prefix[d], d, 0, (unsigned)nprompt, stop, samp_k, samp_topp, samp_invT, 0, 0);
		if (pair) {
			prog_t second;
			model_program(&second, d, 1, (unsigned)nprompt, stop, samp_k, samp_topp, samp_invT, prefix == NULL, 0);
			prog_init(&pair[d]);
			if (prog_pair(&pair[d], prefix ? &prefix[d] : &P[d], &second)) { fprintf(stderr, "engine: incompatible position programs on die %d\n", d); prog_free(&second); longjmp(run_jb, 1); }
			/* The two positions have disjoint scratch/inboxes. */
			prog_pair_fuse_bf16(&pair[d], 1);
			prog_free(&second);
		}
#ifdef FLASH_EXACT_BATCH
		if (batch && build_exact_program(&batch[d], &prefix[d], d, (unsigned)nprompt, stop, samp_k, samp_topp, samp_invT)) {
			fprintf(stderr, "engine: incompatible exact position programs on die %d\n", d);
			longjmp(run_jb, 1);
		}
#endif
#ifdef FLASH_VERIFY_NATIVE
		if (verification && build_verify_program(&verification[d],d)) {
			fprintf(stderr,"engine: incompatible verifier program die%d\n",d);longjmp(run_jb,1);
		}
#endif
	}
	upload_programs(P, nph, 1);
    /* Paired known-prefix passes end before trace_first. Generated positions
       always restore the scalar program and its physical descriptor stride. */
    if (trace_n) {
        unsigned count = (unsigned)(T - nprompt + 1);
        if (count > trace_n) count = trace_n;
        for (int d = 0; d < nd; d++) {
            CK(cuCtxSetCurrent(ctx[d]));
            uint32_t tr[5] = {(uint32_t)dTRACE[d], (uint32_t)(dTRACE[d] >> 32),
                              trace_first, count, 1};
            CK(cuMemsetD8(dTRACE[d], 0, (size_t)count * nph[d] * 64));
            CK(cuMemcpyHtoD(dCTL[d] + 0xd0, tr, sizeof tr));
        }
    }

    if(xtrace_n) {
        xtrace_count=(unsigned)(T-nprompt+1);if(xtrace_count>xtrace_n)xtrace_count=xtrace_n;
        xtrace_first=(unsigned)T-xtrace_count;
        const char *first=getenv("FLASH_XTRACE_FIRST");
        if(first && *first) {
            unsigned start=(unsigned)strtoul(first,NULL,10);
            if(start>=(unsigned)T){fprintf(stderr,"trace start is beyond this request\n");longjmp(run_jb,1);}
            xtrace_first=start;xtrace_count=(unsigned)T-start;if(xtrace_count>xtrace_n)xtrace_count=xtrace_n;
        }
        for(int d=0;d<nd;d++) {
            CK(cuCtxSetCurrent(ctx[d]));
            uint32_t header[4]={(uint32_t)dXTRACE[d],(uint32_t)(dXTRACE[d]>>32),xtrace_first,xtrace_count};
            CK(cuMemsetD8(dXTRACE[d],0,(size_t)xtrace_count*(gr_write_trace?128:4096)*2048));
            CK(cuMemcpyHtoD(dCTL[d]+0xe8,header,sizeof header));
        }
    }
	double t1 = now();
    timing_run_start=t1;last_timing.build_ms=(t1-timing_request_start)*1e3;last_timing.run_ms=0;
	runctx r = { .T = T, .nprompt = nprompt, .sent = nprompt - 1, .count = resume, .cb = cb };
	/* Prime the first position, then overlap subsequent staging with GPU work.
	 * A paired pass waits on its second ready flag while the host's normal
	 * in-flight pump fills it. Reading the entire lookahead before launch
	 * serialized cold prompt I/O with all of the prompt's compute. */
	ple_pump(&r, 1);
#ifdef FLASH_VERIFY_NATIVE
	if (verification) verify_audit_run(&r,P,prefix,verification);
	else {
#endif
#ifdef FLASH_MTP_NATIVE
#ifdef FLASH_DFLASH_NATIVE
	if(dflash_enabled) dflash_run(&r,P,prefix,stop);
	else
#endif
	if (mtp_enabled) mtp_audit_run(&r,P,prefix,stop);
	else {
#endif
#ifdef FLASH_CHUNK_NATIVE
	/* Chunk attention caches are local to this fresh prompt. Snapshot resumes
	 * retain the scalar path until their prefix caches are imported as well. */
	if (!resume && !snap_at && (n_new > 0 || (cb && cb->token))) chunk_prefill(&r, nprompt - 1, nimg);
#endif
	if (snap_at) {   /* run the prompt up to the snapshot position, copy the states, go on */
		run_prefill_prefix(&r, P, prefix, pair, batch, snap_at);
		if (!r.aborted && r.count == snap_at) {
			state_copy(1);
			memcpy(snap_ids, ids, 4 * (size_t)snap_at); snap_s = snap_at;
		}
	}
	if ((pair || prefix) && !r.aborted) run_prefill_prefix(&r, P, prefix, pair, batch, nprompt - 1);
	if (!r.aborted) { upload_programs(P, nph, 1); run_segment(&r, nph, T); }
#ifdef FLASH_MTP_NATIVE
	}
#endif
#ifdef FLASH_VERIFY_NATIVE
	}
#endif
	double t2 = now();
	trace_completed = (unsigned)r.count; trace_aborted = r.aborted != 0; trace_ended = r.ended != 0;
	trace_status = r.aborted ? "aborted" : "completed";
	last_positions = r.count; last_paired_positions = r.paired_positions; last_prefill_chunks = r.prefill_chunks;
#ifdef FLASH_EXACT_BATCH
	last_batched_positions = r.batched_positions;
#endif
	kv_trace_complete = !r.aborted && r.count == T && resume == 0;
	if (trace_n && r.count > (int)trace_first) { trace_valid = (unsigned)r.count - trace_first; if (trace_valid > trace_n) trace_valid = trace_n; }
	if (st) { st->ntok = r.count; st->ended = r.ended; st->nph = nph[0]; st->resumed = resume; st->snapshot = snap_at; st->build_ms = (t1 - timing_request_start) * 1e3; st->run_ms = (t2 - t1) * 1e3; st->first_ms = r.tfirst ? (r.tfirst - t1) * 1e3 : 0; st->paired_positions = r.paired_positions; st->prefill_chunks = r.prefill_chunks; }
    uint32_t model_stop_flag=0;CK(cuCtxSetCurrent(ctx[0]));
    CK(cuMemcpyDtoH(&model_stop_flag,dCTL[0]+0x14,sizeof model_stop_flag));
    trace_ended=!!(model_stop_flag&0x80000000u);
    if(st)st->ended=(int)trace_ended;
    last_timing=(flash_timing){.request_id=trace_request,.prompt=nprompt,.resumed=resume,.completed=r.count,
        .ended=!!(model_stop_flag&0x80000000u),.stop_known=1,.aborted=r.aborted,.first_valid=(r.tfirst!=0),
        .counts_known=1,.counts_final=1,.error_code=0,
        .build_ms=(t1-timing_request_start)*1e3,.run_ms=(t2-t1)*1e3,
        .first_ms=r.tfirst?(r.tfirst-t1)*1e3:0};
    last_timing_valid=1;
    /* Optional CTA-0 trace: all events below are die 0 SR_GLOBALTIMER.
       Publication of the final token is the last phase's stage 7. */
    if(trace_valid && prof_desc[0]>0) {
        uint64_t begin[8],first[8],last[8];size_t phases=(size_t)prof_desc[0];
        CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(begin,dTRACE[0],sizeof begin));
        CK(cuMemcpyDtoH(first,dTRACE[0]+(phases-1)*64,sizeof first));
        CK(cuMemcpyDtoH(last,dTRACE[0]+((size_t)trace_valid*phases-1)*64,sizeof last));
        flash_timing_device(&last_timing,begin[0],first[7],last[7],trace_valid);
    }

	release_request_programs();
	in_run = 0;
	return 0;
}

/* ---- build and tear down all device state (the core keeps the contexts, the arenas and the PLE table) ---- */
/* Reuse the resident attention weights, copying only each worker's query rows.
 * The owner handles K/V/indexer while other workers compute Q. Plan after all normal serving allocations
 * so context/image capacity and paired scratch are included in the budget. */
static void split_attention_queries(void) {
	attn_q_split_mask = 0;
	attn_q_split_mode = 0;
	const char *option = getenv("FLASH_ATTN_Q_SPLIT");
	if (!option || !strcmp(option, "0")) return;
	if (strcmp(option, "1")) { fprintf(stderr, "FLASH_ATTN_Q_SPLIT must be 0 or 1\n"); longjmp(init_jb, 1); }
	size_t room[MAXD] = {0}, needed[MAXD] = {0};
	unsigned first[MAXD][NB] = {{0}}, rows[MAXD][NB] = {{0}}, mask = 0;
	const size_t reserve = 8u << 20;
	for (int d = 0; d < nd; d++) {
		size_t avail, total; CK(cuCtxSetCurrent(ctx[d])); CK(cuMemGetInfo(&avail, &total));
		room[d] = avail > reserve ? avail - reserve : 0;
		if (room[d]) mask |= 1u << d;
	}
	for (;;) {
		unsigned available = 0;
		for (int d = 0; d < nd; d++) available += !!(mask & (1u << d));
		if (available < 2) { fprintf(stderr, "engine: insufficient room for attention Q split\n"); longjmp(init_jb, 1); }
		memset(needed, 0, sizeof needed);
		memset(first, 0, sizeof first); memset(rows, 0, sizeof rows);
		for (int bb = 0; bb < NB; bb++) if (BLK_IS_ATTN(bb)) {
			int owner = runtime_attn_owner(bb), source=attn_die_of(bb);
			unsigned participants = mask;
			/* Reserve the owner for K/V/indexer when at least two other workers fit. */
			int reserve_owner=available>=3;
#ifdef FLASH_CHUNK_NATIVE
            if(chunk_enabled() && available==6)reserve_owner=0;
#endif
            if(reserve_owner)participants &= ~(1u<<owner);
			unsigned workers = 0, rank = 0;
			for (int d = 0; d < nd; d++) workers += !!(participants & (1u << d));
			const hold_tensor *q = hold_find(&ar[source], tn(bb, "attn_q.weight"));
			if (!q || q->K != 2560 || q->rows != 12288 || q->stride <= 0 || q->bytes < (size_t)q->rows * q->stride) {
				fprintf(stderr, "engine: unsupported resident Q tensor at block %d\n", bb); longjmp(init_jb, 1);
			}
			for (int d = 0; d < nd; d++) if (participants & (1u << d)) {
				unsigned quantum = 32;
#ifdef FLASH_CHUNK_NATIVE
				if (chunk_enabled()) quantum = 512; /* complete Q/gate heads */
#endif
				first[d][bb] = ((12288u / quantum) * rank / workers) * quantum;
				rows[d][bb] = ((12288u / quantum) * (rank + 1) / workers) * quantum - first[d][bb]; rank++;
				if (d != source) needed[d] += ((size_t)rows[d][bb] * q->stride + 65535) & ~(size_t)65535;
			}
		}
		int remove = -1;
		for (int d = 0; d < nd; d++) if ((mask & (1u << d)) && needed[d] > room[d] && (remove < 0 || room[d] < room[remove])) remove = d;
		if (remove < 0) break;
		mask &= ~(1u << remove);
	}
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		for (int bb = 0; bb < NB; bb++) if (BLK_IS_ATTN(bb)) {
			int source=attn_die_of(bb);
			dn_weights_t *w = &dw[d][bb];
			w->attn_q_split = 1;
			w->attn_q_first = first[d][bb]; w->attn_q_rows = rows[d][bb];
			if (!rows[d][bb]) continue;
			const hold_tensor *q = hold_find(&ar[source], tn(bb, "attn_q.weight"));
			CUdeviceptr src = ar[source].base + q->off + (size_t)first[d][bb] * q->stride;
			if (d == source) w->attn_q = src;
			else {
				size_t bytes = (size_t)rows[d][bb] * q->stride;
				w->attn_q = dalloc(bytes);
				CK(cuMemcpyPeer(w->attn_q, ctx[d], src, ctx[source], bytes));
			}
		}
		fprintf(stderr, "engine: Q split die=%d copied_bytes=%zu\n", d, needed[d]);
	}
	attn_q_split_mask = mask;
	attn_q_split_mode = 1;
}

int engine_init(core_t *c) {
	trace_n = getenv("FLASH_TRACE") ? (unsigned)strtoul(getenv("FLASH_TRACE"), NULL, 10) : 0;
	if (trace_n > 256) { fprintf(stderr, "FLASH_TRACE must be 0..256\n"); return -1; }
    gr_write_trace=getenv("FLASH_GR_WRITE_TRACE") && atoi(getenv("FLASH_GR_WRITE_TRACE"));
    xtrace_n=getenv("FLASH_XTRACE")?(unsigned)strtoul(getenv("FLASH_XTRACE"),NULL,10):0;
    if(xtrace_n>(gr_write_trace?256u:16u) || (gr_write_trace && !xtrace_n)){fprintf(stderr,"invalid trace window\n");return -1;}
	if (c->abi != CORE_ABI) { fprintf(stderr, "engine: the core speaks ABI %d, this library was built for %d\n", c->abi, CORE_ABI); return -1; }
	core = c; in_init = 1;
	if (setjmp(init_jb)) { in_init = 0; engine_fini(); return -1; }
	MAXT = c->maxt;
	last_positions = last_paired_positions = last_prefill_chunks = 0;
	const char *trace_env = getenv("FLASH_KV_TRACE");
	if (trace_env && strcmp(trace_env, "0") && strcmp(trace_env, "1")) {
		fprintf(stderr, "engine: FLASH_KV_TRACE must be 0 or 1\n");
		in_init = 0; return -1;
	}
	kv_trace = trace_env && !strcmp(trace_env, "1");
	kv_trace_start = kv_trace_complete = 0;
	if (prefill_parse_chunk(getenv("FLASH_PREFILL_CHUNK"), &prefill_chunk)) {
		fprintf(stderr, "engine: FLASH_PREFILL_CHUNK must be 0..%d\n", FLASH_PREFILL_CHUNK_MAX);
		in_init = 0; return -1;
	}
	const char *pair_env = getenv("FLASH_TWO_POSITION");
	if (pair_env && strcmp(pair_env, "0") && strcmp(pair_env, "1")) {
		fprintf(stderr, "engine: FLASH_TWO_POSITION must be 0 or 1\n");
		in_init = 0; return -1;
	}
	two_position = pair_env ? !strcmp(pair_env, "1") : prefill_chunk > 1;
    producer_quant=getenv("FLASH_PRODUCER_QUANT") && atoi(getenv("FLASH_PRODUCER_QUANT"));
    group_matrices=getenv("FLASH_MATRIX_GROUP") && atoi(getenv("FLASH_MATRIX_GROUP"));
    residual_fp8=getenv("FLASH_RESIDUAL_FP8") && atoi(getenv("FLASH_RESIDUAL_FP8"));
    gr_replicate_ffn=getenv("FLASH_GR_REPLICATE_FFN")?atoi(getenv("FLASH_GR_REPLICATE_FFN")):0;
    if(gr_replicate_ffn<0 || gr_replicate_ffn>48 || (gr_replicate_ffn && (!producer_quant || getenv("PROG_NOHCUP") || getenv("PROG_NORMSFUSE")))) {
        fprintf(stderr,"engine: GR replication requires production quantization, standard GR kernels, and a layer count 0..48\n");in_init=0;return -1;
    }
    if(group_matrices || producer_quant)two_position=0;
#ifdef FLASH_MTP_NATIVE
	if(getenv("FLASH_MTP_AUDIT") && !strcmp(getenv("FLASH_MTP_AUDIT"),"3"))two_position=1;
#endif
#ifdef FLASH_DFLASH_NATIVE
	dflash_enabled=getenv("FLASH_MTP_AUDIT") && !strcmp(getenv("FLASH_MTP_AUDIT"),"4");
	if(dflash_enabled)two_position=1;
#endif
#ifdef FLASH_VERIFY_NATIVE
	const char *verify_env=getenv("FLASH_VERIFY_AUDIT");verify_accept=0;
	if(verify_env && strcmp(verify_env,"0")) {
		char *end;unsigned long n=strtoul(verify_env,&end,10);
		if(!*verify_env || *end || n<1 || n>8) { fprintf(stderr,"engine: FLASH_VERIFY_AUDIT must be0..8\n");longjmp(init_jb,1); }
		verify_accept=(unsigned)n;
	}
#endif
	program_capacity = 4096;
	allocated_positions = two_position ? 2 : 1;
#ifdef FLASH_EXACT_BATCH
	const char *batch_env = getenv("FLASH_PHASE_BATCH");
	if (batch_env && strcmp(batch_env, "0") && strcmp(batch_env, "5") && strcmp(batch_env, "8")) {
		fprintf(stderr, "engine: FLASH_PHASE_BATCH must be 0, 5 or 8\n");
		longjmp(init_jb, 1);
	}
	exact_width = batch_env ? (unsigned)atoi(batch_env) : 0;
	exact_cubin_path[0] = 0; exact_cubin_fnv = 0;
	if (exact_width) {
		const char *image = getenv("FLASH_EXACT_CUBIN");
		if (!image || !*image || strlen(image) >= sizeof exact_cubin_path ||
		    (getenv("FLASH_FUSED_MOE") && atoi(getenv("FLASH_FUSED_MOE"))) ||
		    (getenv("FLASH_FAST_EXCHANGE") && atoi(getenv("FLASH_FAST_EXCHANGE")))) {
			fprintf(stderr, "engine: exact batch requires FLASH_EXACT_CUBIN and ordinary exchange flags\n");
			longjmp(init_jb, 1);
		}
		strcpy(exact_cubin_path, image); exact_cubin_fnv = file_fnv(image);
		program_capacity *= exact_width; allocated_positions = exact_width;
		if (!prefill_chunk) prefill_chunk = 64;
	}
#endif
	if (ple_read_parse_readers(getenv("FLASH_PLE_READERS"), &ple_readers)) {
		fprintf(stderr, "engine: FLASH_PLE_READERS must be a decimal integer from 1 to 16\n");
		longjmp(init_jb, 1);
	}
	if (getenv("FLASH_ENGINE_MAXT")) { int m = atoi(getenv("FLASH_ENGINE_MAXT")); if (m > 0 && m <= c->maxt) MAXT = m; }   /* a library reloaded with FLASH_ENGINE_MAXT=n serves fewer positions than the core was started for (the KV caches shrink) */
	const char *cubin = getenv("FLASH_CUBIN") ? getenv("FLASH_CUBIN") : "die.cubin";
	img_rows_max = getenv("FLASH_MAXIMG") ? (unsigned)atoi(getenv("FLASH_MAXIMG")) : (unsigned)(MAXT < FLASH_IMAGE_MAX_TOKENS ? MAXT : FLASH_IMAGE_MAX_TOKENS);   /* image tokens per response (fp16 rows of 2560 on every die) */
	h_imap = calloc((size_t)MAXT + 2, 4); imap_dirty = 0; rope_dirty_T = 0;
	kv_host = getenv("FLASH_KV_HOST") ? atoi(getenv("FLASH_KV_HOST")) : 1;
	nlayers_run = getenv("FLASH_LAYERS") ? atoi(getenv("FLASH_LAYERS")) : NB; if (nlayers_run < 1 || nlayers_run > NB) nlayers_run = NB;
	if (getenv("FLASH_NBLK")) nblk_qsa = (unsigned)atoi(getenv("FLASH_NBLK"));
	if (nblk_qsa < 1 || nblk_qsa > 4096) { fprintf(stderr, "engine: FLASH_NBLK %u out of range\n", nblk_qsa); return -1; }
	unsigned cpos = qsa_compact_positions(nblk_qsa);   /* compact caches: the attended keys (4 per block + the partial one) and the kernels' stray prefetch past them */
	kcf = ATTN_KC_FLOATS(cpos); vcf = (size_t)cpos * 512;
	unsigned maxp = qsa_cache_positions((unsigned)MAXT);   /* positions of the int8 caches and the pool: whole groups of 32 blocks */
	size_t kv_layer = qsa_layer_cache_bytes(maxp);   /* K, V int8 [pos][2][256] and their fp16 scales [pos][2][8] */
	if (kv_host) {   /* DEVICEMAP | PORTABLE: every die reads/writes over PCIe. Initially zeroed; reuse/rollback correctness relies on logical masks, not tail contents. */
		CK(cuCtxSetCurrent(ctx[0])); CK(cuMemHostAlloc(&h_kv, 12 * kv_layer, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE)); memset(h_kv, 0, 12 * kv_layer);
	}
	if (kv_trace) {
		CK(cuCtxSetCurrent(ctx[0]));
		size_t bytes = 12u * (size_t)MAXT * KV_TRACE_WORDS * sizeof(uint32_t);
		CK(cuMemHostAlloc((void **)&h_kv_rawmax, bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
		memset(h_kv_rawmax, 0xff, bytes);
	}
	memset(has_attn, 0, sizeof has_attn);
	for (int bb = 3; bb < nlayers_run; bb += 4) if (runtime_attn_owner(bb) >= 0 && runtime_attn_owner(bb) < nd) has_attn[runtime_attn_owner(bb)] = 1;
	double ts0 = now();
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleLoad(&mod[d], cubin)); check_fused_moe_module(mod[d]); CK(cuModuleGetFunction(&fn[d], mod[d], "die")); }
#ifdef FLASH_EXACT_BATCH
	if (exact_width) for (int d = 0; d < nd; d++) {
		CUfunction marker;
		CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleLoad(&exact_mod[d], exact_cubin_path));
		CK(cuModuleGetFunction(&marker, exact_mod[d], "flash_exact_batch_abi_v1"));
		CK(cuModuleGetFunction(&exact_fn[d], exact_mod[d], "die"));
	}
#endif
	snprintf(cubin_path, sizeof cubin_path, "%s", cubin); cubin_fnv = file_fnv(cubin);
	/* ---- the PLE constants block (plerows layout); the table itself is the core's ---- */
	uint8_t ppc[160]; uint64_t pmult[3]; uint32_t pabsent = PLE_EOS_TOKEN, pend = PLE_EOS_TOKEN;   /* qwen4exp.ple.eos_token_id for both the fill and the reset */
	memcpy(pmult, ple_multipliers, sizeof pmult);
	memset(ppc, 0, sizeof ppc); memcpy(ppc, pmult, 24);
	for (int h = 0; h < 16; h++) { uint32_t o32 = (uint32_t)ple_head_offsets[h], v32 = (uint32_t)ple_head_vocab[h]; memcpy(ppc + 24 + 4 * h, &o32, 4); memcpy(ppc + 88 + 4 * h, &v32, 4); }
	memcpy(ppc + 152, &pabsent, 4); memcpy(ppc + 156, &pend, 4);
	ple_disk = c->ple_disk; ple_fd = c->ple_fd; ple_off = c->ple_off; ple_host = c->ple_host; ple_rdy_off = c->ple_rdy_off;
	ple_cache_path[0]=0;ple_cache_active=0;ple_source_bytes=0;
	const char *ple_file=getenv("FLASH_PLE_FILE");
	if (ple_disk && ple_file && *ple_file) {
		struct stat source;
		/* The core already opened this immutable checkpoint. Its cached size
		 * lets a validated local copy remain usable while NFS is unavailable. */
		struct statx sx;
		if (!statx(ple_fd,"",AT_EMPTY_PATH|AT_STATX_DONT_SYNC,STATX_SIZE|STATX_TYPE,&sx) &&
		    (sx.stx_mask & (STATX_SIZE|STATX_TYPE)) == (STATX_SIZE|STATX_TYPE)) {
			memset(&source,0,sizeof source); source.st_size=(off_t)sx.stx_size; source.st_mode=sx.stx_mode;
		} else if (fstat(ple_fd,&source)) {perror("PLE source stat");longjmp(init_jb,1);}
		ple_source_bytes=source.st_size;
		if (strlen(ple_file)>=sizeof ple_cache_path) {fprintf(stderr,"PLE cache path too long\n");longjmp(init_jb,1);}
		strcpy(ple_cache_path,ple_file);
		int fd=ple_cache_open();
		if (fd>=0) close(fd);
		fprintf(stderr,"engine: PLE cache %s is %s; original source remains available\n",ple_cache_path,ple_cache_active?"available":"unavailable");
	}
	if (ple_disk) {
		int err = ple_read_pool_init(&ple_reads, ple_readers);
		if (err) { fprintf(stderr, "engine: starting PLE readers failed: %s\n", strerror(err)); longjmp(init_jb, 1); }
	}
	h_rdy = ple_disk ? (uint32_t *)(c->ple_host + c->ple_rdy_off) : NULL;
	ple_absent_id = pabsent; ple_end_id = pend;
	free(ple_seq); ple_seq = ple_disk ? calloc((size_t)MAXT + 2, 4) : NULL; ple_next = ple_start = 0;
	float *rcs = malloc(4 * 32 * (size_t)MAXT), *rsn = malloc(4 * 32 * (size_t)MAXT);
	for (int t = 0; t < MAXT; t++) rope_table_gpu((unsigned)t, 1e7f, rcs + 32 * t, rsn + 32 * t);
	/* Each attention layer reads just 32 entries per position. A shared copy
	 * releases two full-capacity tables on the other attention dies, allowing
	 * paired activation scratch without reducing context or image capacity.
	 * Keep one allocation record per pointer; engine_fini frees its owner.
	 * flashcore enables all context-to-context peer mappings before init. */
	rope_owner = -1;
	CUdeviceptr shared_cs = 0, shared_sn = 0;
	const char *rope_env = getenv("FLASH_ROPE_OWNER");
	if (rope_env && strcmp(rope_env, "-1")) {
		char *end;
		long owner = strtol(rope_env, &end, 10);
		if (!*rope_env || *end || owner < 0 || owner >= nd || !has_attn[owner]) {
			fprintf(stderr, "FLASH_ROPE_OWNER must name an attention die or -1\n");
			free(rcs); free(rsn); longjmp(init_jb, 1);
		}
		rope_owner = (int)owner;
		CK(cuCtxSetCurrent(ctx[rope_owner]));
		shared_cs = dalloc(4 * 32 * (size_t)MAXT);
		shared_sn = dalloc(4 * 32 * (size_t)MAXT);
		CK(cuMemcpyHtoD(shared_cs, rcs, 4 * 32 * (size_t)MAXT));
		CK(cuMemcpyHtoD(shared_sn, rsn, 4 * 32 * (size_t)MAXT));
	}
	/* ---- the dies: weights, buffers, exchange, control ---- */
	dw = calloc(MAXD, sizeof *dw); sp = calloc(MAXD, sizeof *sp);
	dst_dn = calloc(MAXD, sizeof *dst_dn); dst_conv = calloc(MAXD, sizeof *dst_conv); dst_kq = calloc(MAXD, sizeof *dst_kq); dst_ks = calloc(MAXD, sizeof *dst_ks); dst_vq = calloc(MAXD, sizeof *dst_vq); dst_vs = calloc(MAXD, sizeof *dst_vs); dst_pkt = calloc(MAXD, sizeof *dst_pkt); dst_ring = calloc(MAXD, sizeof *dst_ring); snap_ring = calloc(MAXD, sizeof *snap_ring);
	snap_dn = calloc(MAXD, sizeof *snap_dn); snap_conv = calloc(MAXD, sizeof *snap_conv); snap_ids = malloc(4 * ((size_t)MAXT + 2)); snap_s = 0;
	for (int d = 0; d < nd; d++) {
		CK(cuCtxSetCurrent(ctx[d]));
		split_t s0; split_of(&s0, d, nd, 0);
		if (kv_trace) CK(cuMemHostGetDevicePointer(&d_kv_rawmax[d], h_kv_rawmax, 0));
		dn_bufs_t *b = &db[d]; memset(b, 0, sizeof *b);
		unsigned nlocmax = 128u * (2u * (unsigned)s0.nk + (unsigned)s0.hn)   /* q, k of the die's key heads, v of its value heads */, cols = 32u * (unsigned)s0.cbn, hrows = 128u * (unsigned)s0.hn;
		b->S = dalloc(4 * 10240); b->S2 = dalloc(4 * 10240); b->xh = dalloc(4 * 10240); b->t = dalloc(4 * 324);
		b->ldown = dalloc(4 * 324); b->lup = dalloc(4 * 4 * (gr_replicate_ffn?2560:cols)); b->lz = dalloc(4 * (gr_replicate_ffn?2560:cols));
        if(gr_replicate_ffn) {
            CUdeviceptr down[6],up[6];for(int e=0;e<6;e++){down[e]=b->ldown;up[e]=b->lz;}
            b->gr_local_down_table=dalloc(sizeof down);b->gr_local_up_table=dalloc(sizeof up);
            CK(cuMemcpyHtoD(b->gr_local_down_table,down,sizeof down));CK(cuMemcpyHtoD(b->gr_local_up_table,up,sizeof up));
        }
		b->zq = dalloc(2 * 2560); b->zs32 = dalloc(4 * 80); b->zs16 = dalloc(4 * 160); b->zas = dalloc(4 * 80);
		b->lqkv = dalloc(4 * nlocmax); b->lgate = dalloc(4 * hrows); b->la = dalloc(4 * 2u * (unsigned)s0.hn); b->lu = dalloc(4 * nlocmax); b->lo = dalloc(4 * hrows); b->lon = dalloc(4 * hrows);
		b->oq = dalloc(2 * 6144); b->os32 = dalloc(4 * 192); b->os16 = dalloc(4 * 384); b->oas = dalloc(4 * 192);
		b->ly = dalloc(4u * (unsigned)s0.yn); b->r = dalloc(4 * (513+16)); b->idx = dalloc(64); b->w = dalloc(64); b->s = dalloc(16);
		b->lidx = dalloc(64); b->lw = dalloc(64); b->ne = dalloc(16); b->part = dalloc(4 * 2560);
		b->g = dalloc(4 * 7040); b->gu = dalloc(4 * 7040); b->hq = dalloc(2 * 7040); b->hs32 = dalloc(4 * 220); b->hs16 = dalloc(4 * 440); b->has = dalloc(4 * 220);
		b->out = dalloc(4 * 2560);
		b->qg = dalloc(4 * 12288); b->k = dalloc(4 * 512); b->v = dalloc(4 * 512); b->qo = dalloc(4 * 6144); b->aout = dalloc(4 * 6144);
		b->pm = dalloc(4 * 24 * 8); b->pd = dalloc(4 * (24 * 8+16)); b->po = dalloc(4 * 24 * 8 * 256);
		if (has_attn[d]) {   /* the attention layers of this die: RoPE tables, the indexer scratch, the compact caches of the attended keys */
			if (rope_owner >= 0) { b->rope_cs = shared_cs; b->rope_sn = shared_sn; }
			else {
				b->rope_cs = dalloc(4 * 32 * (size_t)MAXT); b->rope_sn = dalloc(4 * 32 * (size_t)MAXT);
				CK(cuMemcpyHtoD(b->rope_cs, rcs, 4 * 32 * (size_t)MAXT)); CK(cuMemcpyHtoD(b->rope_sn, rsn, 4 * 32 * (size_t)MAXT));
			}
			b->iqk = dalloc(4 * 640); b->iq = dalloc(4 * 512); b->sc = dalloc(4 * ((size_t)maxp / 4 + 64)); b->hst = dalloc(4 * 1280); b->sel = dalloc(4 * ((size_t)nblk_qsa + 8));
			b->kc = dalloc(kcf * 4); b->vc = dalloc(vcf * 4);
		}
		b->ids = dalloc(4 * ((size_t)MAXT + 2));
		b->imap = dalloc(4 * ((size_t)MAXT + 2)); CK(cuMemsetD32(b->imap, 0xffffffffu, (size_t)MAXT + 2));
		if (img_rows_max) b->img = dalloc(2 * 2560 * (size_t)img_rows_max);
		b->ple_pc = dalloc(160); CK(cuMemcpyHtoD(b->ple_pc, ppc, 160)); CK(cuMemHostGetDevicePointer(&b->ple_tab, c->ple_host, 0)); b->ple_compact = (unsigned)c->ple_compact;
		b->ple_rdy = c->ple_disk ? b->ple_tab + c->ple_rdy_off : 0; b->ple_rid = 0;   /* ple_rid is set per response (engine_run) */
		b->pg = dalloc(4 * 2560); b->pgq = dalloc(2 * 2560); b->pgs32 = dalloc(4 * 80); b->pgs16 = dalloc(4 * 160); b->pgas = dalloc(4 * 80);
		b->lk = dalloc(4u * (unsigned)s0.pkn); b->kn = dalloc(4 * 10240); b->qn = dalloc(4 * 10240); b->lv = dalloc(4u * (unsigned)s0.yn);
		b->pG = dalloc(4 * 10240); b->pgates = dalloc(16); b->px = dalloc(4 * 10240); b->prows = dalloc(64);
		dst_ple[d] = dalloc(4 * 10 * 10240);
		snap_ple[d] = dalloc(4 * 10 * 10240);
		b->hlo = dalloc(4u * (unsigned)s0.on); b->amo = dalloc(8); b->apart = dalloc(16 * 8); b->acnt = dalloc(4); b->tokout = dalloc(8); b->toks = dalloc(4 * ((size_t)MAXT + 2));
		b->tkout = dalloc(256); b->tkpart = dalloc(16 * 256); b->tkcnt = dalloc(4); b->rnd = dalloc(4 * ((size_t)MAXT + 2));
		/* FLASH_LAYERS fixtures need weights and state only for executed blocks. */
		for (int bb = 0; bb < nlayers_run; bb++) {
			split_t *s = &sp[d][bb]; split_of(s, d, nd, bb);
			dn_weights_t *w = &dw[d][bb]; int is_attn = BLK_IS_ATTN(bb);
			w->hc_attn_norm = dptr(d, bb, "hc_attn_norm.weight"); w->hc_attn_dj = dptr(d, bb, "hc_attn_dj.weight"); w->hc_attn_up = dptr(d, bb, "hc_attn_up.weight");
			w->hc_ffn_norm = dptr(d, bb, "hc_ffn_norm.weight"); w->hc_ffn_dj = dptr(d, bb, "hc_ffn_dj.weight"); w->hc_ffn_up = dptr(d, bb, "hc_ffn_up.weight");
            if(bb<gr_replicate_ffn){w->hc_ffn_dj=replicate_gr_matrix(d,bb,0);w->hc_ffn_up=replicate_gr_matrix(d,bb,1);w->hc_ffn_replicated=1;}
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
                if(!getenv("FLASH_ATTN_Q_SPLIT") || strcmp(getenv("FLASH_ATTN_Q_SPLIT"),"1"))
                    w->attn_q=runtime_attn_tensor(d,bb,"attn_q.weight");
				w->attn_k = runtime_attn_tensor(d, bb, "attn_k.weight"); w->attn_k_type = ttype(bb, "attn_k.weight");
				w->attn_v = runtime_attn_tensor(d, bb, "attn_v.weight"); w->attn_v_type = ttype(bb, "attn_v.weight");
				w->attn_output = dptr(d, bb, "attn_output.weight"); w->attn_output_type = ttype(bb, "attn_output.weight");
				w->attn_q_norm = runtime_attn_tensor(d, bb, "attn_q_norm.weight"); w->attn_k_norm = runtime_attn_tensor(d, bb, "attn_k_norm.weight");
				w->indexer_x = runtime_attn_tensor(d, bb, "indexer_x.weight"); w->indexer_qn = runtime_attn_tensor(d, bb, "indexer.q_norm.weight"); w->indexer_kn = runtime_attn_tensor(d, bb, "indexer.k_norm.weight");
				if (d == runtime_attn_owner(bb)) {   /* this layer runs here: int8 K / V with fp16 scales, the pooled indexer keys, the raw key ring */
					if (kv_host) {
						CUdeviceptr base; CK(cuMemHostGetDevicePointer(&base, h_kv, 0)); base += (size_t)((bb - 3) / 4) * kv_layer;
						dst_kq[d][bb] = base; dst_vq[d][bb] = base + (size_t)maxp * 512; dst_ks[d][bb] = base + (size_t)maxp * 1024; dst_vs[d][bb] = base + (size_t)maxp * 1024 + (size_t)maxp * 32;
					} else {
						dst_kq[d][bb] = dalloc((size_t)maxp * 512); dst_vq[d][bb] = dalloc((size_t)maxp * 512);
						dst_ks[d][bb] = dalloc((size_t)maxp * 32); dst_vs[d][bb] = dalloc((size_t)maxp * 32);
					}
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
		w->token_embd = hold_ptr(&ar[d], "token_embd.weight");   /* only the die that holds it embeds (see embed_die); the others get S through an exchange */
		xchg_t *x = &xc[d];
		x->inbox = dalloc(2 * XSLOT); x->flags = dalloc(512); x->xtab = dalloc(128);
		uint32_t fl[96]; for (int e = 0; e < 96; e++) fl[e] = e / 16 < nd ? 0u : 0xffffffffu;
		CK(cuMemcpyHtoD(x->flags, fl, sizeof fl));
		if (trace_n) dTRACE[d] = dalloc((size_t)trace_n * 4096 * 64);
        if(xtrace_n)dXTRACE[d]=dalloc((size_t)xtrace_n*(gr_write_trace?128:4096)*2048);
		dPROG[d] = dalloc(sizeof(phase_t) * program_capacity);
		dCTL[d] = dalloc(0x100 + 4 * (program_capacity < 8192 ? 8192 : program_capacity));
		x->xgen = dCTL[d] + 0x20;
		for (unsigned p = 1; p < allocated_positions; p++) alloc_position(d, p);
#ifdef FLASH_DFLASH_NATIVE
		if(dflash_enabled)for(unsigned p=0;p<2;p++)for(unsigned t=0;t<5;t++)dflash_capture[p][d][t]=dalloc(4*10240);
#endif
	}
	embed_die = -1; for (int d = 0; d < nd; d++) if (dw[d][0].token_embd) { embed_die = d; break; }
	if (embed_die < 0) { fprintf(stderr, "engine: no die holds token_embd\n"); if (in_init) longjmp(init_jb, 1); exit(1); }
	free(rcs); free(rsn);
	split_attention_queries();
    if(producer_quant)for(unsigned p=0;p<allocated_positions;p++)for(unsigned kind=0;kind<2;kind++) {
        CUdeviceptr table[6][4];
        for(int e=0;e<6;e++){const dn_bufs_t *b=position_buf(e,p);table[e][0]=kind?b->oq:b->zq;table[e][1]=kind?b->os32:b->zs32;table[e][2]=kind?b->os16:b->zs16;table[e][3]=kind?b->oas:b->zas;}
        for(int d=0;d<nd;d++){CK(cuCtxSetCurrent(ctx[d]));producer_tables[p][d][kind]=dalloc(sizeof table);CK(cuMemcpyHtoD(producer_tables[p][d][kind],table,sizeof table));}
    }
    if(gr_replicate_ffn)for(unsigned p=0;p<allocated_positions;p++)for(int d=0;d<nd;d++) {
        CUdeviceptr table[6][4];const dn_bufs_t *b=position_buf(d,p);
        for(int e=0;e<6;e++){table[e][0]=b->zq;table[e][1]=b->zs32;table[e][2]=b->zs16;table[e][3]=b->zas;}
        CK(cuCtxSetCurrent(ctx[d]));producer_local_tables[p][d]=dalloc(sizeof table);CK(cuMemcpyHtoD(producer_local_tables[p][d],table,sizeof table));
    }
    if(producer_quant)for(unsigned p=0;p<allocated_positions;p++) {
        CUdeviceptr targets[6];for(int e=0;e<6;e++)targets[e]=position_buf(e,p)->S;
        CK(cuCtxSetCurrent(ctx[embed_die]));embedding_tables[p]=dalloc(sizeof targets);
        CK(cuMemcpyHtoD(embedding_tables[p],targets,sizeof targets));
    }
    if(residual_fp8)for(unsigned p=0;p<allocated_positions;p++)for(int d=0;d<nd;d++){CK(cuCtxSetCurrent(ctx[d]));for(unsigned k=0;k<2;k++)residual_encoded[p][d][k]=dalloc(fp8_bytes(10240));}
    if(producer_quant)for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));phase_gr_sum_cache[d]=dalloc(16u*2560u*sizeof(float));
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
	for (unsigned p = 1; p < allocated_positions; p++) {
		for (int d = 0; d < 6; d++) tab[d] = (uint64_t)position_xchg(d < nd ? d : 0, p)->inbox;
		for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(position_xchg(d, p)->xtab, tab, sizeof tab)); }
	}
	for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuStreamCreate(&pollst[d], CU_STREAM_NON_BLOCKING)); }
	CK(cuCtxSetCurrent(ctx[0]));
	CK(cuMemAllocHost((void **)&h_ids, 4 * ((size_t)MAXT + 2))); CK(cuMemAllocHost((void **)&h_toks, 4 * ((size_t)MAXT + 2))); CK(cuMemAllocHost((void **)&h_rnd, 4 * ((size_t)MAXT + 2)));
	CK(cuMemAllocHost((void **)&h_tok, 64)); CK(cuMemHostAlloc((void **)&h_abort, 64, CU_MEMHOSTALLOC_PORTABLE));
    if(getenv("FLASH_PHASE_WATCH") && atoi(getenv("FLASH_PHASE_WATCH")))CK(cuMemAllocHost((void **)&phase_watch,4u*program_capacity));
#ifdef FLASH_CHUNK_NATIVE
	chunk_init();
#endif
#ifdef FLASH_VERIFY_NATIVE
	verify_init();
#endif
#ifdef FLASH_MTP_NATIVE
	mtp_init();
#endif
#ifdef FLASH_DFLASH_NATIVE
	if(dflash_enabled)dflash_init();
#endif
	in_init = 0;
	fprintf(stderr, "engine: %d dies, %d/%d blocks, up to %d positions, %.0f MB of buffers per die, ready in %.1f s\n", nd, nlayers_run, NB, MAXT, (double)alloc_bytes / nd / 1e6, now() - ts0);
	return 0;
}

void engine_fini(void) {
	release_request_programs();
#ifdef FLASH_DFLASH_NATIVE
	dflash_fini();
#endif
#ifdef FLASH_VERIFY_NATIVE
	verify_fini();
#endif
#ifdef FLASH_MTP_NATIVE
	mtp_fini();
#endif
#ifdef FLASH_CHUNK_NATIVE
	chunk_fini();
#endif
	ple_read_pool_destroy(&ple_reads);
	ple_cache_path[0]=0;ple_cache_active=0;
	if (!core) return;
	for (int d = 0; d < nd; d++) { if (pollst[d]) { cuCtxSetCurrent(ctx[d]); cuStreamDestroy(pollst[d]); pollst[d] = 0; } }
	cuCtxSetCurrent(ctx[0]);
	if (h_kv) { cuCtxSetCurrent(ctx[0]); cuMemFreeHost(h_kv); h_kv = NULL; }
	if (h_kv_rawmax) { cuCtxSetCurrent(ctx[0]); cuMemFreeHost(h_kv_rawmax); h_kv_rawmax = NULL; }
	memset(d_kv_rawmax, 0, sizeof d_kv_rawmax);
	kv_trace = kv_trace_start = kv_trace_complete = 0;
	free(ple_seq); ple_seq = NULL; h_rdy = NULL; ple_disk = 0;
	if (h_ids) cuMemFreeHost(h_ids);
	if (h_toks) cuMemFreeHost(h_toks);
	if (h_rnd) cuMemFreeHost(h_rnd);
	if (h_tok) cuMemFreeHost(h_tok);
    if(phase_watch){cuMemFreeHost(phase_watch);phase_watch=NULL;}
	if (h_abort) cuMemFreeHost(h_abort);
	h_ids = NULL; h_toks = NULL; h_rnd = NULL; h_tok = NULL; h_abort = NULL;
	for (int i = 0; i < nallocs; i++) { cuCtxSetCurrent(allocs[i].c); cuMemFree(allocs[i].p); }
	free(allocs); allocs = NULL; nallocs = capallocs = 0; alloc_bytes = 0;
	for (int d = 0; d < nd; d++) { if (mod[d]) { cuCtxSetCurrent(ctx[d]); cuModuleUnload(mod[d]); mod[d] = 0; } }
#ifdef FLASH_EXACT_BATCH
	for (int d = 0; d < nd; d++) if (exact_mod[d]) {
		cuCtxSetCurrent(ctx[d]); cuModuleUnload(exact_mod[d]); exact_mod[d] = 0; exact_fn[d] = 0;
	}
	memset(db_exact, 0, sizeof db_exact); memset(xc_exact, 0, sizeof xc_exact);
	exact_width = 0; exact_cubin_fnv = 0; exact_cubin_path[0] = 0;
#endif
	cuCtxSetCurrent(ctx[0]);
	free(snap_dn); free(snap_conv); free(snap_ids); snap_dn = snap_conv = NULL; snap_ids = NULL; snap_s = 0; memset(snap_ple, 0, sizeof snap_ple);
	free(dw); free(sp); free(dst_dn); free(dst_conv); free(dst_kq); free(dst_ks); free(dst_vq); free(dst_vs); free(dst_pkt); free(dst_ring); free(snap_ring); dw = NULL; sp = NULL; dst_dn = dst_conv = dst_kq = dst_ks = dst_vq = dst_vs = dst_pkt = dst_ring = snap_ring = NULL;
	free(s_ids); free(s_toks); s_ids = s_toks = NULL;
	free(h_imap); h_imap = NULL; imap_dirty = rope_dirty_T = 0;
	memset(db, 0, sizeof db); memset(xc, 0, sizeof xc); memset(dPROG, 0, sizeof dPROG); memset(dCTL, 0, sizeof dCTL); memset(dst_ple, 0, sizeof dst_ple);
	memset(dTRACE, 0, sizeof dTRACE); trace_n = trace_valid = 0;
    memset(dXTRACE,0,sizeof dXTRACE);xtrace_n=xtrace_first=xtrace_count=0;
    gr_write_trace=0;
    memset(producer_local_tables,0,sizeof producer_local_tables);gr_replicate_ffn=0;
	memset(db_pair, 0, sizeof db_pair); memset(xc_pair, 0, sizeof xc_pair); two_position = prefill_chunk = 0;
	allocated_positions = active_program_positions = 1; program_capacity = 4096;
    memset(embedding_tables,0,sizeof embedding_tables);memset(producer_tables,0,sizeof producer_tables);memset(residual_encoded,0,sizeof residual_encoded);
    memset(phase_gr_sum_cache,0,sizeof phase_gr_sum_cache);
    producer_quant=group_matrices=residual_fp8=0;
	core = NULL;
}

/* chat streaming: one "tok <id>" line per generated token, as die 0 produces it */
typedef struct { FILE *out; int cancel_on_eof; } session_client;
static int put_token(void *arg, int t, uint32_t tok) { (void)t; FILE *out = ((session_client *)arg)->out; fprintf(out, "tok %u\n", tok); return fflush(out) != 0 || ferror(out); }   /* nonzero: the client went away */
/* polled while the prompt runs too: has the client closed the connection? */
static int client_gone(void *arg) {
    session_client *c = arg;
    /* Ordinary CLI clients may close their command pipe and still read output.
       HTTP explicitly negotiates half-close cancellation and drains the reply. */
    short eof = c->cancel_on_eof ? POLLRDHUP : 0;
	struct pollfd p = { .fd = fileno(c->out), .events = eof };
	return poll(&p, 1, 0) > 0 && (p.revents & (eof | POLLHUP | POLLERR | POLLNVAL));
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
 *   info                     "info <positions> <image token capacity>"
 *   timings [0|1]            retrieve the last record or disable/enable per-run records
 *   cancel_on_eof <0|1>      opt in to socket half-close cancellation, retaining the reply
 *   cubin <path>             reload the kernel on every die
 *   reload [path]            replace this library (the weights and the PLE table stay loaded)
 *   quit                     stop the whole process */
static engine_img pend[8]; static int npend;   /* "img" commands waiting for the next response */
static char reload_path[1024];
const char *engine_reload_path(void) { return reload_path[0] ? reload_path : NULL; }

#include "trace_host.inc"
#include "phase/exchange_host.inc"

int engine_session(FILE *in, FILE *out) {
	last_timing_valid=0;
	if (!s_ids) { s_ids = calloc((size_t)MAXT + 2, 4); s_toks = calloc((size_t)MAXT + 2, 4); }
	uint32_t *ids = s_ids, *toks = s_toks;
	uint32_t stop[2] = { 0xffffffffu, 0xffffffffu };
	sampling smp = { 0, 0.f, 1.0f, 0 };   /* greedy until the client sends "sample" */
	int emit_timing = 0;
    session_client client = { .out = out };
	int snap_req = 0;                     /* "snap <s>": take a prefix-cache snapshot after s positions of the next response */
	char *line = NULL; size_t cap = 0; int quit = 0;
	while (getline(&line, &cap, in) > 0) {
		char *p = line; while (*p == ' ') p++;
        if(!strncmp(p,"chat",4)||!strncmp(p,"gen",3))last_timing_valid=0;
        if(!strcmp(p,"timings 1\n")||!strcmp(p,"timings 1\r\n")||!strcmp(p,"timings on\n")||!strcmp(p,"timings on\r\n")){
            emit_timing=1;fprintf(out,"ok: timings 1\n");fflush(out);continue;
        }
        if(!strcmp(p,"timings 0\n")||!strcmp(p,"timings 0\r\n")||!strcmp(p,"timings off\n")||!strcmp(p,"timings off\r\n")){
            emit_timing=0;fprintf(out,"ok: timings 0\n");fflush(out);continue;
        }
        if(!strncmp(p,"cancel_on_eof ",14) && (p[14]=='0'||p[14]=='1')
                && strspn(p+15," \t\r\n")==strlen(p+15)) {
            client.cancel_on_eof=p[14]=='1';
            fprintf(out,"ok: cancel_on_eof %d\n",client.cancel_on_eof);fflush(out);continue;
        }
        if(!strncmp(p,"timings",7)&&strspn(p+7," \t\r\n")==strlen(p+7)){
            char json[4096];
            if(!last_timing_valid)fprintf(out,"timings null\n");
            else if(!flash_timing_json(json,sizeof json,&last_timing))fprintf(out,"timings %s\n",json);
            else fprintf(out,"error: timing JSON overflow\n");
            fflush(out);continue;
        }

		if (!strncmp(p, "quit", 4)) { quit = 1; break; }
		if (!strncmp(p, "reload", 6)) {   /* the core swaps the library after this returns */
			/* Reject every invalid reader override before applying settings
			 * or unloading, including duplicate keys. */
			int preflight = ple_read_reload_preflight(p + 6);
			if (preflight) {
				fprintf(out, "error: reload rejected: %s\n", preflight == EINVAL
				        ? "FLASH_PLE_READERS must be a decimal integer from 1 to 16"
				        : "could not validate reload settings");
				fflush(out);
				continue;
			}
			char path[1000]; reload_path[0] = 0;
			if (sscanf(p + 6, "%999s", path) == 1 && !strchr(path, '=')) snprintf(reload_path, sizeof reload_path, "%s", path);
			/* reload [<lib>] [NAME=VALUE ...]: the settings go into the core's environment before the new engine reads them (FLASH_ENGINE_MAXT, FLASH_KV_HOST, FLASH_NBLK, ...) */
			for (char *tok = strtok(p + 6, " \t\n"); tok; tok = strtok(NULL, " \t\n")) { char *eq = strchr(tok, '='); if (eq && !strncmp(tok, "FLASH_", 6)) { *eq = 0; setenv(tok, eq + 1, 1); } }
			free(line); return 2;
		}
		if (!strncmp(p, "trace", 5)) { trace_dump(out, atoi(p + 5)); fflush(out); continue; }
		if (!strncmp(p, "hosttrace", 9)) { host_trace_dump(out); fflush(out); continue; }
        if(!strncmp(p,"xtrace ",7)){exchange_trace_dump(out,atoi(p+7));fflush(out);continue;}
		if (!strncmp(p, "prof", 4)) {   /* where the last token of the last response spent its time, from the phase timestamps CTA 0 writes at each phase end */
			int pd = atoi(p + 4); if (pd < 0 || pd >= nd) pd = 0; const int nph_ = prof_nph[pd]; const uint32_t *pop = prof_op[pd];   /* prof [die] */
			uint32_t *ts = malloc(4 * (size_t)(prof_desc[pd] ? prof_desc[pd] : 1));
			if (!ts) { fprintf(out, "error: profile allocation\n"); fflush(out); continue; }
			CK(cuCtxSetCurrent(ctx[pd])); CK(cuMemcpyDtoH(ts, dCTL[pd] + 0x100, 4 * (size_t)prof_desc[pd]));
			/* Secondary descriptors have no timestamp; compact logical phase ends. */
			for (int i = 0; i < nph_; i++) ts[i] = ts[prof_index[pd][i]];
			static double tot[128]; static int cnt[128]; memset(tot, 0, sizeof tot); memset(cnt, 0, sizeof cnt);
			double all = 0; int big[8] = { -1, -1, -1, -1, -1, -1, -1, -1 };
			for (int i = 1; i < nph_; i++) {
				double us = (uint32_t)(ts[i] - ts[i - 1]) / 1e3; if (pop[i] >= 128) continue;
				tot[pop[i]] += us; cnt[pop[i]]++; all += us;
				for (int k = 0; k < 8; k++) if (big[k] < 0 || us > (uint32_t)(ts[big[k]] - ts[big[k] - 1]) / 1e3) { for (int j = 7; j > k; j--) big[j] = big[j - 1]; big[k] = i; break; }
			}
			fprintf(out, "prof: %d phases, %.2f ms between the first and last timestamp\n", nph_, all / 1e3);
			fprintf(out,"  structural: producer_quant_fusions=%u matrix_groups=%u residual_fusions=%u local_fusions=%u producer_publications=%u\n",prof_producer_fusions[pd],prof_matrix_groups[pd],prof_gr_fusions[pd],prof_local_fusions[pd],prof_publications[pd]);
            unsigned grid_joins=0,peer_waits=0,joined_publications=0;
            for(int j=0;j<nph_;j++) {
                unsigned flags=prof_flags[pd][j];
                grid_joins+=!(flags&(4u|PHASE_EXCHANGE_JOIN_FLAG));
                peer_waits+=!!(flags&2u);joined_publications+=!!(flags&PHASE_EXCHANGE_JOIN_FLAG);
            }
            fprintf(out,"  waits: local_grid_joins=%u peer_acquires=%u joined_publications=%u\n",grid_joins,peer_waits,joined_publications);
            const int all_kinds = strstr(p, "all") != NULL;
			for (int rep = 0; rep < (all_kinds ? 128 : 12); rep++) { int bi = -1; for (int o = 0; o < 128; o++) if (cnt[o] && (bi < 0 || tot[o] > tot[bi])) bi = o; if (bi < 0 || tot[bi] <= 0) break; fprintf(out, "  %-10s %4d phases %9.1f us (%6.1f us each)\n", profile_op_name((unsigned)bi), cnt[bi], tot[bi], tot[bi] / cnt[bi]); tot[bi] = 0; }
			fprintf(out, "  longest:"); for (int k = 0; k < 8 && big[k] > 0; k++) fprintf(out, " #%d %s %.0f us;", big[k], profile_op_name(pop[big[k]]), (uint32_t)(ts[big[k]] - ts[big[k] - 1]) / 1e3);
			fprintf(out, "\n"); fflush(out); free(ts); continue;
		}
		if (!strncmp(p, "statehash", 9)) { state_hashes(out); continue; }
        if(!strncmp(p,"tokenaudit",10)){token_audit(out);fflush(out);continue;}
#ifdef FLASH_VERIFY_NATIVE
		if (!strncmp(p,"verifystats",11)) { verify_stats(out);continue; }
		if (!strncmp(p,"verifykeep ",11)) {
			char *end;unsigned long kept=strtoul(p+11,&end,10);
			if(!verify_accept || end==p+11 || strspn(end," \t\r\n")!=strlen(end) || kept<1 || kept>8)
				fprintf(out,"error: verifykeep requires an enabled verifier and a value1..8\n");
			else { verify_accept=(unsigned)kept;fprintf(out,"ok: verifykeep %u\n",verify_accept); }
			fflush(out);continue;
		}
#endif
#ifdef FLASH_MTP_NATIVE
		if (!strncmp(p,"mtpstats",8)) {
#ifdef FLASH_DFLASH_NATIVE
			if(dflash_enabled)dflash_stats(out);else
#endif
			{ mtp_stats(out); }
			continue;
		}
#endif
#ifdef FLASH_EXACT_BATCH
		if (!strncmp(p, "batchstats", 10)) {
			fprintf(out, "batchstats width=%u positions=%d paired=%d total=%d\n", exact_width, last_batched_positions, last_paired_positions, last_positions);
			fflush(out); continue;
		}
#endif
#ifdef FLASH_CHUNK_NATIVE
		if (!strncmp(p, "chunkhash", 9)) {
			fprintf(out, "%s\n", chunk_hash_report[0] ? chunk_hash_report : "chunkhash unavailable"); fflush(out); continue;
		}
#endif
		if (!strncmp(p, "kvstats", 7)) { kv_stats(out); continue; }
		if (!strncmp(p, "info", 4)) { fprintf(out, "info %d %u\n", MAXT, img_rows_max); fflush(out); continue; }
		if (!strncmp(p, "config", 6)) {   /* what this engine is: the identity to attach to a measurement (the commit it was built from, where the KV lives, the layers it runs, the kernel image) */
			char ad[64] = ""; for (int d = 0; d < nd; d++) if (has_attn[d]) { char one[8]; snprintf(one, sizeof one, "%s%d", ad[0] ? "," : "", d); strncat(ad, one, sizeof ad - strlen(ad) - 1); }
			const char *sh = getenv("SPLIT_SHARES"), *am = getenv("ATTN_MAP");
            char runtime_map[96]="";for(int bb=3;bb<NB;bb+=4){char one[8];snprintf(one,sizeof one,"%s%d",bb==3?"":",",runtime_attn_owner(bb));strcat(runtime_map,one);}
			fprintf(out, "config rev=%s kv_host=%d kv_trace=%d ple_disk=%d ple_readers=%u ple_file=%s ple_cache=%s maxt=%d layers=%d/%d nblk=%u dies=%d attn_dies=%s split=%s attn_map=%s img_rows=%u rope_owner=%d trace_tokens=%u two_position=%d prefill_chunk=%d attn_q_split=%u attn_q_mask=%x cubin=%s cubin_fnv=%016llx",
			        FLASH_REV, kv_host, kv_trace, ple_disk, ple_disk ? ple_readers : 0, ple_cache_active ? ple_cache_path : "core", ple_cache_path[0] ? ple_cache_path : "core", MAXT, nlayers_run, NB, nblk_qsa, nd, ad, sh ? sh : "default", am ? am : "default", img_rows_max, rope_owner, trace_n, two_position, prefill_chunk, attn_q_split_mode, attn_q_split_mask, cubin_path, (unsigned long long)cubin_fnv);
            fprintf(out," runtime_attn_map=%s balanced=%d",runtime_map,phase_attention_balanced());
            fprintf(out," residual_storage=%s fused_gr=%d producer_quant=%d matrix_group=%d gr_replica_ffn=%d xtrace=%u",residual_fp8?"fp8":"fp32",getenv("FLASH_FUSED_GR")?atoi(getenv("FLASH_FUSED_GR")):0,producer_quant,group_matrices,gr_replicate_ffn,xtrace_n);
#ifdef FLASH_CHUNK_NATIVE
			fprintf(out, " native_chunk=%u chunk_maxt=%u prefix_hash=%d chunk_cubin=%s chunk_cubin_fnv=%016llx", chunk_ldx, chunk_ldx ? chunk_maxt : 0, getenv("FLASH_PREFIX_HASH") && atoi(getenv("FLASH_PREFIX_HASH")), chunk_ldx ? (chunk_mod[0] ? chunk_cubin_path : cubin_path) : "none", (unsigned long long)(chunk_ldx ? (chunk_mod[0] ? chunk_cubin_fnv : cubin_fnv) : 0));
#endif
#ifdef FLASH_EXACT_BATCH
			fprintf(out, " phase_batch=%u exact_cubin=%s exact_cubin_fnv=%016llx", exact_width, exact_width ? exact_cubin_path : "none", (unsigned long long)exact_cubin_fnv);
#endif
#ifdef FLASH_MTP_NATIVE
			fprintf(out," mtp_shadow=%d mtp_mode=%d mtp_rollout=%d mtp_model=%s mtp_cubin=%s mtp_cubin_fnv=%016llx",mtp_enabled==1,mtp_enabled,mtp_enabled==2,mtp_enabled?mtp_model_path:"none",mtp_enabled?mtp_cubin_path:"none",(unsigned long long)mtp_cubin_fnv);
#endif
#ifdef FLASH_DFLASH_NATIVE
			fprintf(out," dflash2=%d dflash_block=%d dflash_device=6 dflash_model=%s",dflash_enabled,dflash_enabled?DF_B:0,dflash_enabled?getenv("FLASH_DFLASH_MODEL"):"none");
#endif
#ifdef FLASH_VERIFY_NATIVE
			fprintf(out," verify_audit=%u verify_state_check=%d",verify_accept,verify_check_state);
#endif
			fputc('\n', out);
			fflush(out); continue;
		}
		if (!strncmp(p, "cubin", 5)) {
			char path[512]; if (sscanf(p + 5, "%511s", path) != 1) { fprintf(out, "error: cubin <path>\n"); fflush(out); continue; }
			for (int d = 0; d < nd; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuModuleUnload(mod[d])); CK(cuModuleLoad(&mod[d], path)); check_fused_moe_module(mod[d]); CK(cuModuleGetFunction(&fn[d], mod[d], "die")); }
			snprintf(cubin_path, sizeof cubin_path, "%s", path); cubin_fnv = file_fnv(path);
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
		p += chat ? 4 : 3; char *end; errno = 0;
		long n_new = strtol(p, &end, 10);
		if (end == p || errno == ERANGE || n_new < 0 || n_new > INT_MAX) {
			trace_reject_arguments();
			last_timing = (flash_timing){ .request_id = trace_request, .prompt = 0, .error_code = -1,
				.build_ms = NAN, .run_ms = NAN };
			last_timing_valid = 1;
			if (emit_timing) {
				char json[4096];
				if (!flash_timing_json(json, sizeof json, &last_timing))
					fprintf(out, "timings %s\n", json);
			}
			fprintf(out, "error: gen|chat <n_new> <id>...\n");
			fflush(out); continue;
		}
		int nprompt = 0; p = end;
		for (;;) { long v = strtol(p, &end, 10); if (end == p) break; if (nprompt + 2 >= MAXT) break; ids[nprompt++] = (uint32_t)v; p = end; }
		run_stats st;
		engine_cb cb = { chat ? put_token : NULL, client_gone, &client };
		int rc = engine_run(ids, nprompt, (int)n_new, stop, &smp, snap_req, pend, npend, &cb, &st);
        /* Emit before done/error: a device fault closes this session for a rebuild. */
        if(emit_timing && last_timing_valid){char json[4096];if(!flash_timing_json(json,sizeof json,&last_timing))fprintf(out,"timings %s\n",json);}
		snap_req = 0; npend = 0;
		if (rc == -3) { fprintf(out, "error: bad image data (a file, a position or too many image tokens)\n"); fflush(out); continue; }
		if (rc == -4) { fprintf(out, "error: host program allocation failed\n"); fflush(out); continue; }
		if (rc == -2) {
		fprintf(out, "error: device fault; the core rebuilds the contexts and the engine; request=%llu", (unsigned long long)trace_request);
		trace_status_fields(out); fputc('\n', out); fflush(out); free(line); return 3;
		}
		if (rc) { fprintf(out, "error: %d prompt ids + %ld new > %d positions\n", nprompt, n_new, MAXT); fflush(out); continue; }
		if (chat) {
			fprintf(out, "done %d %d %d %.1f %.1f %d %d\n", flash_timing_derive(&last_timing).generated, last_timing.ended, nprompt, st.run_ms, st.first_ms, st.resumed, st.snapshot);
		} else {
			CK(cuCtxSetCurrent(ctx[0])); CK(cuMemcpyDtoH(toks, db[0].toks, 4 * (size_t)st.ntok));
			fprintf(out, "tokens:"); for (int t = n_new ? nprompt - 1 : 0; t < st.ntok; t++) fprintf(out, " %u", toks[t]); fprintf(out, "\n");
			/* st.ntok is the final position count; the first st.resumed positions came from the snapshot and were not run, so they are not in the average */
			fprintf(out, "time: %.1f ms for %d positions run (%d from the snapshot; %.2f ms per position run; first token %.1f ms; %u phases; programs %.0f ms; %d paired positions; %d prefill chunks)%s\n",
			        st.run_ms, st.ntok - st.resumed, st.resumed, st.run_ms / (st.ntok - st.resumed > 0 ? st.ntok - st.resumed : 1), st.first_ms, st.nph, st.build_ms, st.paired_positions, st.prefill_chunks, st.ended ? "; end of turn" : "");
		}
		fflush(out);
	}
	free(line);
	return quit;
}
