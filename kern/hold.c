/* hold: the resident weight holder. Loads the tensors of blocks [b0, b1) (plus the
 * non-block tensors when asked) of the GGUF shard onto one die, repacked into the
 * kernels' tile layouts, in a single arena; exports the arena with a CUDA IPC
 * handle and writes a manifest (handle, then one line per tensor: name, offset,
 * bytes, type, rows, K, tile stride); then sleeps until killed, so kernel tests
 * map the arena (hold_open in hold.h) instead of reloading from NFS.
 *
 *   hold <gguf> <die> <b0> <b1> <manifest> [token_embd] [output] [cache=DIR]
 *     token_embd / output also load those tensors; cache=DIR writes every device
 *     image (tiles or raw) to DIR/<name> so the C references read local disk, not NFS
 *
 * Layout rules (what the kernels expect):
 *   Q2_0, Q4_K, Q5_K, Q3_K, Q6_K, IQ4_XS, Q4_0/Q5_0/Q8_0/IQ4_NL: repack_* tiles,
 *     rows = ne1 (output rows), K = ne0; 3-d expert tensors are repacked per
 *     expert, contiguously (expert stride = tile stride x rows).
 *   BF16 / F16 / F32: raw, as in the file.
 */
#define _POSIX_C_SOURCE 200809L
#include "gguf.h"
#include "quant.h"
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

static int wanted(const char *name, int b0, int b1, int extra_embd, int extra_out) {
	if (!strncmp(name, "blk.", 4)) { int b = atoi(name + 4); return b >= b0 && b < b1; }
	if (!strncmp(name, "token_embd", 10)) return extra_embd;
	if (!strncmp(name, "output", 6)) return extra_out;
	return 0;
}

int main(int argc, char **argv) {
	if (argc < 6) { fprintf(stderr, "usage: hold <gguf> <die> <b0> <b1> <manifest> [token_embd] [output]\n"); return 2; }
	const char *path = argv[1], *mpath = argv[5];
	int die = atoi(argv[2]), b0 = atoi(argv[3]), b1 = atoi(argv[4]), xe = 0, xo = 0;
	const char *cache = NULL;
	for (int i = 6; i < argc; i++) { if (!strcmp(argv[i], "token_embd")) xe = 1; if (!strcmp(argv[i], "output")) xo = 1; if (!strncmp(argv[i], "cache=", 6)) cache = argv[i] + 6; }
	gguf_file g;
	if (gguf_open(&g, path)) { fprintf(stderr, "cannot open %s\n", path); return 1; }
	/* pass 1: sizes */
	size_t total = 0; int count = 0;
	for (int i = 0; i < g.ntensors; i++) {
		const gguf_tensor *t = &g.t[i];
		if (!wanted(t->name, b0, b1, xe, xo)) continue;
		int K = (int)t->dims[0], rows = (int)t->dims[1], experts = t->nd > 2 ? (int)t->dims[2] : 1;
		int st = tile_stride(t->type, K);
		size_t bytes = st ? (size_t)st * rows * experts : t->nbytes;
		total = (total + 255) & ~(size_t)255;
		total += bytes; count++;
	}
	fprintf(stderr, "hold: %d tensors, %.2f GB on die %d\n", count, total / 1e9, die);
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUdeviceptr arena;
	CK(cuDeviceGet(&dev, die));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_BLOCKING_SYNC, dev));
	CK(cuMemAlloc(&arena, total));
	CUipcMemHandle h;
	CK(cuIpcGetMemHandle(&h, arena));
	FILE *m = fopen(mpath, "w");
	if (!m) { perror(mpath); return 1; }
	fprintf(m, "arena %zu die %d handle ", total, die);
	for (size_t i = 0; i < sizeof h; i++) fprintf(m, "%02x", ((unsigned char *)&h)[i]);
	fprintf(m, "\n");
	/* pass 2: load */
	size_t off = 0; size_t cap = 0; uint8_t *buf = NULL;
	for (int i = 0; i < g.ntensors; i++) {
		const gguf_tensor *t = &g.t[i];
		if (!wanted(t->name, b0, b1, xe, xo)) continue;
		int K = (int)t->dims[0], rows = (int)t->dims[1], experts = t->nd > 2 ? (int)t->dims[2] : 1;
		int st = tile_stride(t->type, K);
		size_t bytes = st ? (size_t)st * rows * experts : t->nbytes;
		off = (off + 255) & ~(size_t)255;
		const uint8_t *src = t->data;
		if (!src) { fprintf(stderr, "%s: not mapped\n", t->name); return 1; }
		const uint8_t *img = src;
		if (st) {
			if (bytes > cap) { free(buf); cap = bytes; buf = malloc(cap); }
			size_t per = t->nbytes / experts;
			for (int e = 0; e < experts; e++) repack(t->type, src + per * e, rows, K, buf + (size_t)st * rows * e);
			img = buf;
		}
		CK(cuMemcpyHtoD(arena + off, img, bytes));
		if (cache) {
			char fn[512]; snprintf(fn, sizeof fn, "%s/%s", cache, t->name);
			FILE *cf = fopen(fn, "wb");
			if (!cf || fwrite(img, 1, bytes, cf) != bytes) { perror(fn); return 1; }
			fclose(cf);
		}
		fprintf(m, "%s %zu %zu %u %d %d %d %d\n", t->name, off, bytes, t->type, rows, K, st, experts);
		fprintf(stderr, "  %-40s %8.1f MB  %s%s\n", t->name, bytes / 1e6, gguf_type_name(t->type), st ? " tiles" : "");
		off += bytes;
	}
	fclose(m);
	gguf_close(&g);
	fprintf(stderr, "hold: ready (%s); holding until SIGTERM\n", mpath);
	signal(SIGTERM, onsig); signal(SIGINT, onsig);
	while (!stop) pause();
	CK(cuMemFree(arena));
	return 0;
}
