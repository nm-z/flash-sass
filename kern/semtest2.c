/* Runs semtest2.cubin (kernel "semtest") and reports which butterfly formulation holds. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

int main(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, 0, dev));
	CK(cuModuleLoad(&mod, "semtest2.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "semtest"));
	float out[192];
	memset(out, 0, sizeof out);
	CUdeviceptr din, dout;
	CK(cuMemAlloc(&din, 64));
	CK(cuMemAlloc(&dout, sizeof out));
	CK(cuMemcpyHtoD(dout, out, sizeof out));
	void *params[] = { &din, &dout };
	CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(out, dout, sizeof out));
	const char *names[6] = { "A gemv-style", "B fresh regs", "C shfl stall 6", "D read barrier", "E shfl stall 2", "F shfl stall 3" };
	for (int v = 0; v < 6; v++) {
		int wrong = 0;
		for (int t = 0; t < 32; t++) wrong += out[32 * v + t] != (t < 16 ? 120.f : 376.f);
		printf("%-16s %2d lanes wrong; lanes 0..7:", names[v], wrong);
		for (int t = 0; t < 8; t++) printf(" %g", out[32 * v + t]);
		printf("\n");
	}
	return 0;
}
