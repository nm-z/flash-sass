/* icache_probe: time N straight-line instructions per loop pass (icache_probe.sass built per N) for 1 CTA of W warps, or one CTA per SM (16).
 *   icache_probe N_KIB... : the cubins icp_<n>.cubin (n instructions) in the current directory
 * Prints ns per instruction per warp: the knee is the instruction cache capacity, the plateau above it the fetch cost. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CK(x) do { CUresult r_ = (x); if (r_) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); exit(2); } } while (0)
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

int main(int argc, char **argv) {
	int threads = getenv("ICP_THREADS") ? atoi(getenv("ICP_THREADS")) : 32, ctas = getenv("ICP_CTAS") ? atoi(getenv("ICP_CTAS")) : 1;
	unsigned reps = getenv("ICP_R") ? (unsigned)atoi(getenv("ICP_R")) : 200;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate_v2(&ctx, 0, dev));
	CUdeviceptr out; CK(cuMemAlloc(&out, 4096));
	for (int i = 1; i < argc; i++) {
		int n = atoi(argv[i]); const char *colon = strchr(argv[i], ':'); if (colon) reps = (unsigned)atoi(colon + 1);   /* "n:r": n instructions, r passes */
		char name[64]; snprintf(name, sizeof name, "icp_%d.cubin", n);
		CUmodule mod; CUfunction fn; CK(cuModuleLoad(&mod, name)); CK(cuModuleGetFunction(&fn, mod, "icache_probe"));
		void *params[] = { &out };
		CK(cuLaunchKernel(fn, ctas, 1, 1, threads, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		double t0 = now();
		CK(cuLaunchKernel(fn, ctas, 1, 1, threads, 1, 1, 0, 0, params, NULL)); CK(cuCtxSynchronize());
		double t = now() - t0;
		printf("%7d instr (%7.1f KB): %8.3f ns per instruction per warp  (%.2f us total)\n", n, n * 32.0 / 3 / 1024.0, t / ((double)n * reps) * 1e9, t * 1e6);
		fflush(stdout);
		CK(cuModuleUnload(mod));
	}
	return 0;
}
