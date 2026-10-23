/* Runs semtest.cubin with known inputs and prints every result next to what the
 * intended semantics predict. a = 0x8005ffff, b = 0x00070002, c = 5, h = fp16 1.5. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

static float asf(unsigned u) { float f; memcpy(&f, &u, 4); return f; }

int main(void) {
	CK(cuInit(0));
	CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
	CK(cuDeviceGet(&dev, 0));
	CK(cuCtxCreate_v2(&ctx, 0, dev));
	CK(cuModuleLoad(&mod, "semtest.cubin"));
	CK(cuModuleGetFunction(&fn, mod, "semtest"));
	unsigned in[4] = { 0x8005ffffu, 0x00070002u, 5u, 0x3e00u };
	unsigned out[80];
	memset(out, 0xee, sizeof out);
	CUdeviceptr din, dout;
	CK(cuMemAlloc(&din, sizeof in));
	CK(cuMemAlloc(&dout, sizeof out));
	CK(cuMemcpyHtoD(din, in, sizeof in));
	CK(cuMemcpyHtoD(dout, out, sizeof out));
	void *params[] = { &din, &dout };
	CK(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 0, 0, params, NULL));
	CK(cuCtxSynchronize());
	CK(cuMemcpyDtoH(out, dout, sizeof out));
	struct { const char *what; unsigned expect; int isfloat; } exp[16] = {
		{ "XMAD.U16.S16 a.H0*b.H0  (65535*2)", 131070u, 0 },
		{ "XMAD.U16.S16 a.H1*b.H0  (32773*2)", 65546u, 0 },
		{ "XMAD.U16.S16 a.H0*b.H1  (65535*7)", 458745u, 0 },
		{ "XMAD.U16.S16 a.H1*b.H1  (32773*7)", 229411u, 0 },
		{ "XMAD          a.H0*b.H0  (65535*2)", 131070u, 0 },
		{ "XMAD.S16.U16 a.H0*b.H0  (-1*2)", (unsigned)-2, 0 },
		{ "LOP32I.AND a, 0x000c000c", 0x0004000cu, 0 },
		{ "SHR b, 2 (arith)", 0x00070002u >> 2, 0 },
		{ "I2F.F32.S32 b", 0, 1 },
		{ "F2F.F32.F16 h (1.5)", 0, 1 },
		{ "XMAD.U16.S16 acc (131070+131070)", 262140u, 0 },
		{ "IADD3 -c", (unsigned)-5, 0 },
		{ "LDG.64 pair a+b", 0x8005ffffu + 0x00070002u, 0 },
		{ "LDG.128 c+h", 5u + 0x3e00u, 0 },
		{ "ISCADD (c<<4)+b", (5u << 4) + 0x00070002u, 0 },
		{ "LDG.U16 low half of c", 5u, 0 },
	};
	float fexp[16] = { 0 };
	fexp[8] = (float)(int)0x00070002u;
	fexp[9] = 1.5f;
	int bad = 0;
	for (int i = 0; i < 16; i++) {
		if (exp[i].isfloat) {
			int ok = asf(out[i]) == fexp[i];
			printf("%-40s got %g expect %g %s\n", exp[i].what, asf(out[i]), fexp[i], ok ? "ok" : "WRONG");
			bad += !ok;
		} else {
			int ok = out[i] == exp[i].expect;
			printf("%-40s got 0x%08x (%d) expect 0x%08x %s\n", exp[i].what, out[i], (int)out[i], exp[i].expect, ok ? "ok" : "WRONG");
			bad += !ok;
		}
	}
	int sh = 0, shf = 0;
	for (int t = 0; t < 32; t++) { sh += out[16 + t] != (unsigned)(t ^ 1); shf += asf(out[48 + t]) != (float)(t + (t ^ 8)); }
	printf("SHFL.BFLY xor1: %d wrong; SHFL.BFLY xor8 + FADD: %d wrong (t0: %g)\n", sh, shf, asf(out[48]));
	return bad + sh + shf ? 1 : 0;
}
