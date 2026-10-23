/* phase_floor: the per-phase fixed cost of the persistent kernel. One die image (FLASH_CUBIN), one die (16 CTAs), a program of N phases of
 * one op with zero units (every CTA falls straight through the op to the grid barrier), timed with the host clock over several launches:
 *   phase_floor [N] [op] [chain]    N phases (default 2000), op id (default OP_ACTQ2560), chain=1 marks the phases independent (no barrier)
 * Prints microseconds per phase for the dispatch + barrier path with a warm instruction cache (nothing else runs in between), the lower bound of
 * the 3.1 us that the decode trace shows for the same path interleaved with real ops. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dieops.h"

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); exit(2); } } while (0)
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
	unsigned n = argc > 1 ? (unsigned)atoi(argv[1]) : 2000;
	int op = argc > 2 ? atoi(argv[2]) : OP_ACTQ2560;
	int nobar = argc > 3 ? atoi(argv[3]) : 0;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	const char *cubin = getenv("FLASH_CUBIN");
	CK(cuModuleLoad(&mod, cubin ? cubin : "die.cubin")); CK(cuModuleGetFunction(&fn, mod, "die"));
	uint32_t *prog = calloc(n, 128);
	for (unsigned i = 0; i < n; i++) { prog[32 * i] = (uint32_t)op; prog[32 * i + 1] = 0; if (nobar) prog[32 * i + 30] |= 4; }
	CUdeviceptr dPROG, dCTL;
	CK(cuMemAlloc(&dPROG, (size_t)n * 128)); CK(cuMemAlloc(&dCTL, 8192));
	CK(cuMemcpyHtoD(dPROG, prog, (size_t)n * 128));
	unsigned ntok = 1; void *params[] = { &dPROG, &n, &dCTL, &ntok };
	double best = 1e9;
	for (int run = 0; run < 6; run++) {
		CK(cuMemsetD8(dCTL, 0, 8192));
		unsigned width = 0; CK(cuMemcpyHtoD(dCTL + 0x24, &width, 4));
		double t0 = now();
		CK(cuLaunchKernel(fn, 16, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double t = now() - t0;
		if (run && t < best) best = t;
	}
	printf("%u phases of op %d (units 0%s): %.3f us per phase\n", n, op, nobar ? ", nobar" : "", best / n * 1e6);
	return 0;
}
