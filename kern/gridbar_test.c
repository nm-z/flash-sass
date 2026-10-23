/*
 * gridbar_test: the MaxAs software grid barrier, checked for visibility (each CTA
 * publishes the iteration number before the barrier and reads its neighbour's
 * after it) and timed.
 *
 *   gridbar_test [iters]        runs grids of 16, 32, 64 and 128 CTAs
 */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
	unsigned iters = argc > 1 ? (unsigned)atoi(argv[1]) : 20000;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, "gridbar.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "gridbar"));
	int nregs, maxb, sms;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 256, 0));
	CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
	printf("gridbar: %d registers, %d CTAs/SM resident, %d SMs\n", nregs, maxb, sms);
	CUdeviceptr dctr, dgen, dout, ddata;
	CK(cuMemAlloc(&dctr, 256));
	CK(cuMemAlloc(&dgen, 256));
	CK(cuMemAlloc(&dout, 4 * 1024));
	CK(cuMemAlloc(&ddata, 128 * 1024));
	int fails = 0;
	for (unsigned grid = 16; grid <= 128; grid *= 2) {
		if ((int)grid > maxb * sms) { printf("grid %u: not co-resident, skipped\n", grid); continue; }
		CK(cuMemsetD32(dctr, 0, 64));
		CK(cuMemsetD32(dgen, 0, 64));
		CK(cuMemsetD32(dout, 0xffffffffu, 1024));
		CK(cuMemsetD32(ddata, 0xffffffffu, 32 * 1024));
		void *params[] = { &dctr, &dgen, &grid, &iters, &dout, &ddata };
		double t0 = now();
		CK(cuLaunchKernel(fn, grid, 1, 1, 256, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		double t1 = now();
		unsigned out[1024];
		CK(cuMemcpyDtoH(out, dout, 4 * grid));
		unsigned bad = 0;
		for (unsigned c = 0; c < grid; c++) bad += out[c];
		printf("grid %3u: %u iterations, %u visibility mismatches, %.2f us per barrier\n", grid, iters, bad, (t1 - t0) / iters * 1e6);
		if (bad) fails++;
	}
	return fails ? 3 : 0;
}
