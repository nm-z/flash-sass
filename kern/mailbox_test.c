/*
 * mailbox_test: die-to-die mailbox over P2P (40 KB payload + sequence word).
 *
 *   mailbox_test A B [iters]     sender on die A, receiver on die B
 *
 * Prints the round trip (payload A->B, ack B->A) per iteration and the payload
 * mismatch counts, which must be 0.
 */
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
	if (argc < 3) { fprintf(stderr, "usage: mailbox_test A B [iters]\n"); return 2; }
	int a = atoi(argv[1]), b = atoi(argv[2]);
	unsigned iters = argc > 3 ? (unsigned)atoi(argv[3]) : 10000;
	CK(cuInit(0));
	CUdevice da, db; CUcontext ca, cb; CUmodule ma, mb; CUfunction fa, fb; CUstream sa, sb;
	CK(cuDeviceGet(&da, a));
	CK(cuDeviceGet(&db, b));
	int can_ab = 0, can_ba = 0;
	CK(cuDeviceCanAccessPeer(&can_ab, da, db));
	CK(cuDeviceCanAccessPeer(&can_ba, db, da));
	if (!can_ab || !can_ba) { fprintf(stderr, "no peer access between %d and %d\n", a, b); return 1; }
	CK(cuCtxCreate_v2(&ca, CU_CTX_SCHED_SPIN, da));
	CK(cuModuleLoad(&ma, "mailbox.cubin"));
	CK(cuModuleGetFunction(&fa, ma, "mailbox"));
	CK(cuStreamCreate(&sa, CU_STREAM_NON_BLOCKING));
	CK(cuCtxCreate_v2(&cb, CU_CTX_SCHED_SPIN, db));
	CK(cuModuleLoad(&mb, "mailbox.cubin"));
	CK(cuModuleGetFunction(&fb, mb, "mailbox"));
	CK(cuStreamCreate(&sb, CU_STREAM_NON_BLOCKING));
	CK(cuCtxSetCurrent(ca));
	CK(cuCtxEnablePeerAccess(cb, 0));
	CK(cuCtxSetCurrent(cb));
	CK(cuCtxEnablePeerAccess(ca, 0));

	/* per die: sequence word (own page), payload 40 KB, out word */
	CUdeviceptr seq_a, pay_a, out_a, seq_b, pay_b, out_b;
	CK(cuCtxSetCurrent(ca));
	CK(cuMemAlloc(&seq_a, 4096)); CK(cuMemAlloc(&pay_a, 40960)); CK(cuMemAlloc(&out_a, 256));
	CK(cuMemsetD32(seq_a, 0, 1024)); CK(cuMemsetD32(pay_a, 0xdeadbeefu, 10240)); CK(cuMemsetD32(out_a, 0, 64));
	CK(cuCtxSynchronize());
	CK(cuCtxSetCurrent(cb));
	CK(cuMemAlloc(&seq_b, 4096)); CK(cuMemAlloc(&pay_b, 40960)); CK(cuMemAlloc(&out_b, 256));
	CK(cuMemsetD32(seq_b, 0, 1024)); CK(cuMemsetD32(pay_b, 0xdeadbeefu, 10240)); CK(cuMemsetD32(out_b, 0, 64));
	CK(cuCtxSynchronize());

	unsigned role_s = 0, role_r = 1;
	/* sender on A: local seq = seq_a (ack arrives here), remote seq = seq_b, remote payload = pay_b */
	void *pa[] = { &seq_a, &seq_b, &pay_a, &pay_b, &iters, &role_s, &out_a };
	/* receiver on B: local seq = seq_b, remote seq = seq_a (ack), local payload = pay_b */
	void *pb[] = { &seq_b, &seq_a, &pay_b, &pay_a, &iters, &role_r, &out_b };
	double t0 = now();
	CK(cuCtxSetCurrent(cb));
	CK(cuLaunchKernel(fb, 1, 1, 1, 256, 1, 1, 0, sb, pb, NULL));
	CK(cuCtxSetCurrent(ca));
	CK(cuLaunchKernel(fa, 1, 1, 1, 256, 1, 1, 0, sa, pa, NULL));
	CK(cuStreamSynchronize(sa));
	CK(cuCtxSetCurrent(cb));
	CK(cuStreamSynchronize(sb));
	double t1 = now();
	unsigned bad_a = 0, bad_b = 0;
	CK(cuMemcpyDtoH(&bad_b, out_b, 4));
	CK(cuCtxSetCurrent(ca));
	CK(cuMemcpyDtoH(&bad_a, out_a, 4));
	printf("mailbox %d -> %d: %u iterations of 40 KB + seq, %.2f us per round trip, payload mismatches %u (sender-side counter %u)\n",
	       a, b, iters, (t1 - t0) / iters * 1e6, bad_b, bad_a);
	return bad_b ? 3 : 0;
}
