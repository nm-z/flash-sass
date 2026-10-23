/* allgather_test: time a 6-die all-gather (P2P stores + flags) per iteration.
 * usage: allgather_test [iters] [ndies] [cubin]   (dies 0..ndies-1; every pair peer-enabled) */
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
	unsigned iters = argc > 1 ? (unsigned)atoi(argv[1]) : 2000;
	int N = argc > 2 ? atoi(argv[2]) : 6;
	const char *cubin = argc > 3 ? argv[3] : "allgather.cubin";
	size_t slice = 4096 * (strstr(cubin, "ag4") ? 4 : 1);
	int check = !strstr(cubin, "flags") && !strstr(cubin, "nover");
	CK(cuInit(0));
	CUdevice dev[8]; CUcontext ctx[8]; CUmodule mod[8]; CUfunction fn[8];
	CUdeviceptr pay[8], flg[8], out[8], ptab[8], ftab[8];
	for (int d = 0; d < N; d++) {
		CK(cuDeviceGet(&dev[d], d));
		CK(cuCtxCreate_v2(&ctx[d], CU_CTX_SCHED_SPIN, dev[d]));
		CK(cuModuleLoad(&mod[d], cubin)); CK(cuModuleGetFunction(&fn[d], mod[d], "allgather"));
		CK(cuMemAlloc(&pay[d], 12 * slice)); CK(cuMemAlloc(&flg[d], 64)); CK(cuMemAlloc(&out[d], 64));
		CK(cuMemAlloc(&ptab[d], 64)); CK(cuMemAlloc(&ftab[d], 64));
		CK(cuMemsetD32(pay[d], 0, 12 * slice / 4)); CK(cuMemsetD32(flg[d], 0, 16)); CK(cuMemsetD32(out[d], 0, 16));
		CK(cuCtxSynchronize());
	}
	for (int a = 0; a < N; a++) { CK(cuCtxSetCurrent(ctx[a])); for (int b = 0; b < N; b++) if (a != b) { int can; CK(cuDeviceCanAccessPeer(&can, dev[a], dev[b])); if (!can) { printf("no peer access %d -> %d\n", a, b); return 2; } CK(cuCtxEnablePeerAccess(ctx[b], 0)); } }
	unsigned long long pt[6] = { 0 }, ft[6] = { 0 };
	for (int d = 0; d < N; d++) { pt[d] = pay[d]; ft[d] = flg[d]; }
	for (int d = N; d < 6; d++) { pt[d] = pay[0]; ft[d] = flg[0]; }   /* unused dies point somewhere harmless */
	for (int d = 0; d < N; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyHtoD(ptab[d], pt, 48)); CK(cuMemcpyHtoD(ftab[d], ft, 48)); CK(cuCtxSynchronize()); }
	double t0 = now();
	for (int d = 0; d < N; d++) {
		unsigned me = d;
		void *params[] = { &ptab[d], &ftab[d], &me, &iters, &out[d] };
		CK(cuCtxSetCurrent(ctx[d]));
		CK(cuLaunchKernel(fn[d], 1, 1, 1, 256, 1, 1, 0, 0, params, NULL));
	}
	for (int d = 0; d < N; d++) { CK(cuCtxSetCurrent(ctx[d])); CK(cuCtxSynchronize()); }
	double dt = now() - t0;
	unsigned long long badtot = 0;
	for (int d = 0; d < N; d++) { unsigned b; CK(cuCtxSetCurrent(ctx[d])); CK(cuMemcpyDtoH(&b, out[d], 4)); badtot += b; }
	printf("%d dies, slice %zu B (%zu B gathered), %u iterations: %.2f us per all-gather, mismatches %llu\n", N, slice, slice * N, iters, dt / iters * 1e6, badtot);
	return (check && badtot) ? 3 : 0;
}
