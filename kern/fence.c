/* fence: keep known-bad blocks of a die's memory allocated so that nothing else is ever given them.
 *
 *   fence <die> <block MiB> <bad blocks: RANGES> [tail]      e.g.  fence 2 2 2785-2800,3328-3711
 *
 * The blocks are numbered in allocation order, the way memtest numbers its chunks: on a freshly reset die with no other process the driver
 * hands out memory bottom-up, the same way every time, so block i names the same physical place as in the memtest/blockmap run that found it.
 * fence allocates blocks until the die is full, frees every block that is not listed and then waits for SIGTERM (writing /tmp/fence-d<die>.ready
 * once the good blocks are free). Memory that is later allocated on the die (the holder's arena, the engine's buffers) can only come from the
 * freed blocks: the GPU's page tables do not need physically contiguous memory, so one big allocation is built from the good blocks around the fence.

 * The top of the die: once the good blocks are freed, one or two more blocks become allocatable above the last one the first sweep could get (the driver keeps
 * a reserve while the die is full); nobody has tested them and they sit next to the top of the die. After freeing, fence sweeps the free memory again and keeps
 * the last <tail> blocks it gets (default 2), so those are fenced too.
 * Start it before anything else touches the die, and keep it running as long as the die is in use. */
#include <cuda.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);

#define MAXB 8192

static void ranges(const char *s, char *set) {
	while (*s) {
		char *e; long a = strtol(s, &e, 10); if (e == s) break;
		long b = a; if (*e == '-') { s = e + 1; b = strtol(s, &e, 10); }
		for (long i = a; i <= b; i++) if (i >= 0 && i < MAXB) set[i] = 1;
		s = *e == ',' ? e + 1 : e;
	}
}
static void on_term(int sig) { (void)sig; _exit(0); }

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: fence <die> <block MiB> <bad blocks: RANGES> [tail]\n"); return 2; }
	int die = atoi(argv[1]); long mib = atol(argv[2]); static char bad[MAXB];
	if (mib < 2 || mib > 128) { fprintf(stderr, "fence: block MiB 2..128\n"); return 2; }
	ranges(argv[3], bad);
	size_t B = (size_t)mib << 20;
	CUdevice dev; CUcontext ctx;
	if (cuInit(0) || cuDeviceGet(&dev, die)) { fprintf(stderr, "fence: no die %d\n", die); return 1; }
	CUresult r = cuCtxCreate_v2(&ctx, 0, dev);
	if (r) { fprintf(stderr, "fence: context on die %d: %d\n", die, (int)r); return 1; }
	static CUdeviceptr p[MAXB]; int n = 0;
	while (n < MAXB && cuMemAlloc(&p[n], B) == CUDA_SUCCESS) n++;
	int top = -1; for (int i = 0; i < MAXB; i++) if (bad[i]) top = i;
	if (top >= n) { fprintf(stderr, "fence: die %d gave only %d blocks of %ld MiB, the list reaches block %d: the die is not in the state the list was measured in\n", die, n, mib, top); return 1; }
	int kept = 0; for (int i = 0; i < n; i++) { if (bad[i]) kept++; else cuMemFree(p[i]); }
	int tail = argc > 4 ? atoi(argv[4]) : 2; static CUdeviceptr q[MAXB]; int nq = 0;
	while (nq < MAXB && cuMemAlloc(&q[nq], B) == CUDA_SUCCESS) nq++;
	for (int i = 0; i < nq; i++) { if (i >= nq - tail) kept++; else cuMemFree(q[i]); }
	size_t fr = 0, tot = 0; cuMemGetInfo(&fr, &tot);
	fprintf(stderr, "fence: die %d: %d blocks of %ld MiB allocated (+%d on the second sweep, the last %d of them kept), %d kept (%.1f MiB fenced), %.2f GiB free for others\n", die, n, mib, nq, tail < nq ? tail : nq, kept, (double)kept * mib, (double)fr / (1u << 30));
	char rp[64]; snprintf(rp, sizeof rp, "/tmp/fence-d%d.ready", die);
	FILE *f = fopen(rp, "w"); if (f) { fprintf(f, "%d %d\n", n, kept); fclose(f); }
	signal(SIGTERM, on_term); signal(SIGINT, on_term);
	for (;;) pause();
}
