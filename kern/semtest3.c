#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)
int main(void) {
	CK(cuInit(0)); CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0)); CK(cuCtxCreate_v2(&ctx, 0, dev));
	CK(cuModuleLoad(&mod, "semtest3.cubin")); CK(cuModuleGetFunction(&fn, mod, "semtest"));
	unsigned in[2] = { 0xfffffff0u, 0x20u }, out[8]; memset(out, 0xee, sizeof out);
	CUdeviceptr din, dout; CK(cuMemAlloc(&din, 64)); CK(cuMemAlloc(&dout, 64));
	CK(cuMemcpyHtoD(din, in, 8)); CK(cuMemcpyHtoD(dout, out, 32));
	void *p[] = { &din, &dout };
	CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, p, NULL)); CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(out, dout, 32));
	printf("a+b low 0x%08x (expect 0x00000010) high 0x%08x (expect 0x11111112)\n", out[0], out[1]);
	unsigned base = (unsigned)dout;
	for (int k = 0; k < 4; k++) printf("bump %d: 0x%08x expect 0x%08x %s\n", k + 1, out[2 + k], base + 0x1000u * (k + 1), out[2 + k] == base + 0x1000u * (k + 1) ? "ok" : "WRONG");
	return 0;
}
