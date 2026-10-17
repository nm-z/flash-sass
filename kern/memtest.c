/* memtest: fill (almost) all free memory of one die with patterns and read it back through
 * the copy engine, so weak cells raise their ECC errors now (Xid 48 -> the page is retired at
 * the next reset) instead of in the middle of a response.
 *
 *   memtest <die> [passes] [skip=N,M,..]   skip: chunks (128 MiB, in allocation order) that are allocated but never touched,
 *                                the fence around a known bad band; exit 0: clean; 3: an uncorrectable ECC error (context lost);
 *                                4: a value mismatch without an ECC report; 1: setup error */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);

#define CHUNK ((size_t)128 << 20)

int main(int argc, char **argv) {
	if (argc < 2) { fprintf(stderr, "usage: memtest <die> [passes]\n"); return 2; }
	int die = atoi(argv[1]), passes = argc > 2 ? atoi(argv[2]) : 1;
	CUdevice dev; CUcontext ctx;
	if (cuInit(0) || cuDeviceGet(&dev, die)) { fprintf(stderr, "memtest: no die %d\n", die); return 1; }
	CUresult r = cuCtxCreate_v2(&ctx, 0, dev);
	if (r == CUDA_ERROR_ECC_UNCORRECTABLE) { printf("die %d: context refused (uncorrectable ECC pending)\n", die); return 3; }
	if (r) { fprintf(stderr, "memtest: context %d\n", (int)r); return 1; }
	CUdeviceptr ch[256]; int n = 0, cur = -1; char skip[256] = { 0 };
	for (int a = 3; a < argc; a++) if (!strncmp(argv[a], "skip=", 5)) for (char *s = argv[a] + 5; *s;) { char *e; long v = strtol(s, &e, 10); if (e == s) break; if (v >= 0 && v < 256) skip[v] = 1; s = *e == ',' ? e + 1 : e; }   /* cur: the chunk being written or read when a call fails (chunks come from the driver bottom-up) */
	while (n < 256 && cuMemAlloc(&ch[n], CHUNK) == CUDA_SUCCESS) n++;
	size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot);
	printf("die %d: %d x %zu MB allocated (%.2f of %.2f GiB; %zu MB left)\n", die, n, CHUNK >> 20, (double)n * CHUNK / (1u << 30), (double)tot / (1u << 30), fr >> 20);
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
	printf("die %d: %s (while on chunk %d = %.3f-%.3f GiB of the allocation order, pattern pass)\n", die, s, cur, (double)cur * CHUNK / (1u << 30), (double)(cur + 1) * CHUNK / (1u << 30));
	return r == CUDA_ERROR_ECC_UNCORRECTABLE ? 3 : 1;
}
