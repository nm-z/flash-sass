/* Runs xmadbench cubins (kernel "semtest", one per XB_FORM) and reports issue rate.
 *   xmadbench CUBIN [iters]
 * grid = 2 CTAs per SM x 1024 threads so the SM is full (16 warps/SM at <= 32 regs... here 42 regs -> 3 CTAs fit). */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + 1e-9 * t.tv_nsec; }

int main(int argc, char **argv) {
	const char *cubin = argc > 1 ? argv[1] : "xmadbench0.cubin";
	unsigned iters = argc > 2 ? (unsigned)atoi(argv[2]) : 4096;
	unsigned unroll = argc > 3 ? (unsigned)atoi(argv[3]) : 1;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, 0, dev));
	int sms, khz;
	CK(cuDeviceGetAttribute(&sms, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev));
	CK(cuDeviceGetAttribute(&khz, CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev));
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "semtest"));
	int nregs, maxb;
	CK(cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, fn));
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, fn, 1024, 0));
	unsigned grid = 2 * sms, threads = 1024;
	CUdeviceptr din, dout;
	CK(cuMemAlloc(&din, 64));
	CK(cuMemAlloc(&dout, (size_t)grid * threads * 4));
	CK(cuMemcpyHtoD(din, &iters, 4));
	void *params[] = { &din, &dout };
	CK(cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	double t0 = now();
	CK(cuLaunchKernel(fn, grid, 1, 1, threads, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	double t1 = now();
	double ops = (double)grid * threads * iters * 64 * unroll;
	double per_clk_sm = ops / ((t1 - t0) * khz * 1e3) / sms;
	printf("%s unroll %u: %d regs, %d CTAs/SM at 1024 thr, %.1f lane-ops/clk/SM at nominal %d MHz (%.2f ms)\n", cubin, unroll, nregs, maxb, per_clk_sm, khz / 1000, (t1 - t0) * 1e3);
	return 0;
}
