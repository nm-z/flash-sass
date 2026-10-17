/* memtest: fill (almost) all free memory of one die with patterns and read it back through
 * the copy engine, so weak cells raise their ECC errors now (Xid 48 -> the page is retired at
 * the next reset) instead of in the middle of a response.
 *
 *   memtest <die> [passes] [chunk=MiB] [skip=RANGES] [only=RANGES]
 *     chunk   size of each allocation, 2..128 MiB (default 128); the chunks are numbered in allocation order (the driver hands out
 *             memory bottom-up, the same way every time on a freshly reset die), so a chunk number names a place on the die
 *     skip    chunks that are allocated but never touched (the fence around known bad memory); RANGES: 5,7-9,12
 *     only    touch only these chunks
 *   Each chunk is innocent until it fails on its own: a failing call names the chunk it was on.
 *   exit 0: clean; 3: an uncorrectable ECC error (context lost); 4: a value mismatch without an ECC report; 1: setup error */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);

#define MAXCH 4096

static void ranges(const char *s, char *set, char v) {
	while (*s) {
		char *e; long a = strtol(s, &e, 10); if (e == s) break;
		long b = a; if (*e == '-') { s = e + 1; b = strtol(s, &e, 10); }
		for (long i = a; i <= b; i++) if (i >= 0 && i < MAXCH) set[i] = v;
		s = *e == ',' ? e + 1 : e;
	}
}

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: memtest <die> [passes] [chunk=MiB] [skip=RANGES] [only=RANGES]\n"); return 2; }
	int die = atoi(argv[1]), passes = argc > 2 && argv[2][0] >= '0' && argv[2][0] <= '9' ? atoi(argv[2]) : 1;
	size_t CHUNK = (size_t)128 << 20;
	static char skip[MAXCH], only[MAXCH]; int have_only = 0;
	for (int a = 2; a < argc; a++) {
		if (!strncmp(argv[a], "chunk=", 6)) { long m = atol(argv[a] + 6); if (m < 2 || m > 128) { fprintf(stderr, "memtest: chunk=2..128\n"); return 2; } CHUNK = (size_t)m << 20; }
		else if (!strncmp(argv[a], "skip=", 5)) ranges(argv[a] + 5, skip, 1);
		else if (!strncmp(argv[a], "only=", 5)) { ranges(argv[a] + 5, only, 1); have_only = 1; }
	}
	CUdevice dev; CUcontext ctx;
	if (cuInit(0) || cuDeviceGet(&dev, die)) { fprintf(stderr, "memtest: no die %d\n", die); return 1; }
	CUresult r = cuCtxCreate_v2(&ctx, 0, dev);
	if (r == CUDA_ERROR_ECC_UNCORRECTABLE) { printf("die %d: context refused (uncorrectable ECC pending)\n", die); return 3; }
	if (r) { fprintf(stderr, "memtest: context %d\n", (int)r); return 1; }
	static CUdeviceptr ch[MAXCH]; int n = 0, cur = -1;   /* cur: the chunk being written or read when a call fails */
	while (n < MAXCH && cuMemAlloc(&ch[n], CHUNK) == CUDA_SUCCESS) n++;
	if (have_only) for (int i = 0; i < MAXCH; i++) if (!only[i]) skip[i] = 1;
	size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot);
	int ntest = 0; for (int i = 0; i < n; i++) ntest += !skip[i];
	printf("die %d: %d x %zu MB allocated (%.2f of %.2f GiB; %zu MB left), %d touched\n", die, n, CHUNK >> 20, (double)n * CHUNK / (1u << 30), (double)tot / (1u << 30), fr >> 20, ntest);
	uint32_t *h = malloc(CHUNK);
	const uint32_t pat[4] = { 0x00000000u, 0xffffffffu, 0xaaaaaaaau, 0x55555555u };
	for (int p = 0; p < passes; p++)
		for (int k = 0; k < 4; k++) {
			for (int i = 0; i < n; i++) { if (skip[i]) continue; cur = i; if ((r = cuMemsetD32(ch[i], pat[k], CHUNK / 4)) || (r = cuCtxSynchronize())) goto fail; }
			for (int i = 0; i < n; i++) {
				if (skip[i]) continue;
				cur = i; if ((r = cuMemcpyDtoH(h, ch[i], CHUNK))) goto fail;
				for (size_t j = 0; j < CHUNK / 4; j++)
					if (h[j] != pat[k]) { printf("die %d: chunk %d word %zu reads %08x, wrote %08x\n", die, i, j, h[j], pat[k]); return 4; }
			}
			printf("die %d: pass %d pattern %08x clean\n", die, p, pat[k]); fflush(stdout);
		}
	printf("die %d: clean\n", die);
	return 0;
fail:;
	const char *s = "?"; cuGetErrorString(r, &s);
	printf("die %d: %s (while on chunk %d = %.4f-%.4f GiB of the allocation order, pattern pass)\n", die, s, cur, (double)cur * CHUNK / (1u << 30), (double)(cur + 1) * CHUNK / (1u << 30));
	return r == CUDA_ERROR_ECC_UNCORRECTABLE ? 3 : 1;
}
