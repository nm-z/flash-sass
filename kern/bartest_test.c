/* bartest_test: count stale registers after one barrier wait over N slow loads. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

int main(int argc, char **argv) {
	const char *cubin = argc > 1 ? argv[1] : "bartest.cubin";
	int N = argc > 2 ? atoi(argv[2]) : 20;
	int host = argc > 3 ? atoi(argv[3]) : 1;
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, CU_CTX_SCHED_SPIN, dev));
	CK(cuModuleLoad(&mod, cubin));
	CK(cuModuleGetFunction(&fn, mod, "bartest"));
	size_t bytes = 512 * (size_t)N;
	unsigned *h;
	CUdeviceptr dsrc, ddst;
	if (host) {
		void *p;
		CK(cuMemHostAlloc(&p, bytes, CU_MEMHOSTALLOC_DEVICEMAP));
		h = p;
		CK(cuMemHostGetDevicePointer(&dsrc, p, 0));
	} else {
		h = malloc(bytes);
		CK(cuMemAlloc(&dsrc, bytes));
	}
	for (size_t i = 0; i < bytes / 4; i++) h[i] = (unsigned)i;
	if (!host) CK(cuMemcpyHtoD(dsrc, h, bytes));
	CK(cuMemAlloc(&ddst, bytes));
	CK(cuMemsetD32(ddst, 0x11111111u, bytes / 4));
	unsigned un = (unsigned)N;
	void *params[] = { &dsrc, &ddst, &un };
	int stale_total = 0, runs = 20;
	int stale_per_load[64] = { 0 };
	for (int r = 0; r < runs; r++) {
		CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
		CK(cuCtxSynchronize());
		unsigned *out = malloc(bytes);
		CK(cuMemcpyDtoH(out, ddst, bytes));
		for (int i = 0; i < N; i++) {
			int bad = 0;
			for (int l = 0; l < 32; l++) for (int k = 0; k < 4; k++) if (out[128 * i + 4 * l + k] != (unsigned)(128 * i + 4 * l + k)) bad++;
			if (bad) { stale_per_load[i]++; stale_total++; }
		}
		free(out);
	}
	printf("N=%d %s: loads with stale data in %d of %d runs; per load:", N, host ? "host-mapped" : "device", stale_total ? 1 : 0, runs);
	for (int i = 0; i < N; i++) printf(" %d", stale_per_load[i]);
	printf("\n");
	return 0;
}
