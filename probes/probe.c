/*
 * Phase-0 probes for the Flash-Next persistent kernel on archy's Tesla M60s.
 * Driver API only (libcuda, driver 470). Kernels come from probes.cubin.
 *
 *   probe p2p          peer-access matrix
 *   probe pp           die-to-die flag latency, P2P flags and host-mapped flags
 *   probe row [GiB]    n-gram row reads from host-mapped RAM vs device memory
 *   probe chase [GiB]  dependent pointer chasing through host-mapped RAM
 *   probe xmad         XMAD and FFMA issue rates per SM
 *   probe bar          software grid barrier cost at 1, 2, 4 CTAs per SM
 *   probe pin GiB      time to pin GiB of host memory (the n-gram table path)
 *   probe all          everything except pin
 */
#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); \
	fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__, #x, s_); exit(1); } } while (0)

/* Driver 470 exports the v2 context creation; CUDA 12.9 headers map cuCtxCreate to a v4 it lacks. */
CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);

#define MAXDEV 8
static int ndev;
static CUdevice dev[MAXDEV];
static CUcontext ctx[MAXDEV];
static CUmodule mod[MAXDEV];
static CUstream strm[MAXDEV];
static int sms[MAXDEV], khz[MAXDEV];
static int peer[MAXDEV][MAXDEV];

static double now(void) {
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static CUfunction fn(int d, const char *name) {
	CUfunction f;
	CK(cuCtxSetCurrent(ctx[d]));
	CK(cuModuleGetFunction(&f, mod[d], name));
	return f;
}

static uint64_t rng_state = 0x9e3779b97f4a7c15ull;
static uint64_t rng(void) {
	uint64_t x = rng_state;
	x ^= x << 13; x ^= x >> 7; x ^= x << 17;
	return rng_state = x;
}

static void init(void) {
	CK(cuInit(0));
	CK(cuDeviceGetCount(&ndev));
	if (ndev > MAXDEV) ndev = MAXDEV;
	for (int i = 0; i < ndev; i++) {
		char name[64];
		size_t bytes;
		CK(cuDeviceGet(&dev[i], i));
		CK(cuDeviceGetName(name, sizeof name, dev[i]));
		CK(cuDeviceTotalMem(&bytes, dev[i]));
		CK(cuDeviceGetAttribute(&sms[i], CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, dev[i]));
		CK(cuDeviceGetAttribute(&khz[i], CU_DEVICE_ATTRIBUTE_CLOCK_RATE, dev[i]));
		CK(cuCtxCreate_v2(&ctx[i], CU_CTX_SCHED_SPIN, dev[i]));
		CK(cuModuleLoad(&mod[i], "probes.cubin"));
		CK(cuStreamCreate(&strm[i], CU_STREAM_NON_BLOCKING));
		printf("die %d: %s %zu MiB %d SMs %d MHz\n", i, name, bytes >> 20, sms[i], khz[i] / 1000);
	}
	for (int i = 0; i < ndev; i++)
		for (int j = 0; j < ndev; j++) {
			peer[i][j] = 0;
			if (i == j) continue;
			CK(cuDeviceCanAccessPeer(&peer[i][j], dev[i], dev[j]));
			if (peer[i][j]) {
				CK(cuCtxSetCurrent(ctx[i]));
				CUresult r = cuCtxEnablePeerAccess(ctx[j], 0);
				if (r != CUDA_SUCCESS && r != CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED) {
					const char *s = "?";
					cuGetErrorString(r, &s);
					printf("enable peer %d->%d: %s\n", i, j, s);
					peer[i][j] = 0;
				}
			}
		}
}

static void probe_p2p(void) {
	printf("\n== peer access (row can read/write column) ==\n   ");
	for (int j = 0; j < ndev; j++) printf(" %d", j);
	printf("\n");
	for (int i = 0; i < ndev; i++) {
		printf("%d: ", i);
		for (int j = 0; j < ndev; j++) printf(" %c", i == j ? '.' : peer[i][j] ? 'Y' : '-');
		printf("\n");
	}
}

/* Flag ping-pong between die a and die b. host=1 puts both flags in pinned host memory. */
static void pingpong(int a, int b, int host, unsigned iters) {
	CUdeviceptr fa, fb, oa, ob, fa_b, fb_a;
	void *h = NULL;
	if (host) {
		CK(cuCtxSetCurrent(ctx[a]));
		CK(cuMemHostAlloc(&h, 8192, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
		memset(h, 0, 8192);
		CK(cuMemHostGetDevicePointer(&fa, h, 0));
		fb = fa + 4096;
		CK(cuCtxSetCurrent(ctx[b]));
		CK(cuMemHostGetDevicePointer(&fa_b, h, 0));
		fb_a = fb; fb = fa_b + 4096; fa_b = fa_b; /* same UVA address on both dies */
	} else {
		if (!peer[a][b] || !peer[b][a]) { printf("pp %d<->%d dev: no peer access\n", a, b); return; }
		CK(cuCtxSetCurrent(ctx[a]));
		CK(cuMemAlloc(&fa, 4096));
		CK(cuMemsetD32(fa, 0, 1024));
		CK(cuCtxSetCurrent(ctx[b]));
		CK(cuMemAlloc(&fb, 4096));
		CK(cuMemsetD32(fb, 0, 1024));
		fa_b = fa; fb_a = fb;
	}
	CK(cuCtxSetCurrent(ctx[a]));
	CK(cuMemAlloc(&oa, 16));
	CK(cuCtxSetCurrent(ctx[b]));
	CK(cuMemAlloc(&ob, 16));
	CK(cuCtxSynchronize());
	CK(cuCtxSetCurrent(ctx[a]));
	CK(cuCtxSynchronize());

	CUfunction ka = fn(a, "pingpong"), kb = fn(b, "pingpong");
	unsigned zero = 0, one = 1;
	CUdeviceptr r_local_b = fb, r_remote_b = fa, r_local_a = fa, r_remote_a = fb;
	void *pb[] = { &r_local_b, &r_remote_b, &iters, &zero, &ob };
	void *pa[] = { &r_local_a, &r_remote_a, &iters, &one, &oa };
	double t0 = now();
	CK(cuCtxSetCurrent(ctx[b]));
	CK(cuLaunchKernel(kb, 1, 1, 1, 1, 1, 1, 0, strm[b], pb, NULL));
	CK(cuCtxSetCurrent(ctx[a]));
	CK(cuLaunchKernel(ka, 1, 1, 1, 1, 1, 1, 0, strm[a], pa, NULL));
	CK(cuStreamSynchronize(strm[a]));
	CK(cuCtxSetCurrent(ctx[b]));
	CK(cuStreamSynchronize(strm[b]));
	double t1 = now();
	uint64_t cyc = 0;
	CK(cuCtxSetCurrent(ctx[a]));
	CK(cuMemcpyDtoH(&cyc, oa, 8));
	printf("pp %d<->%d %-4s round trip %.2f us (wall), %.2f us (die %d clock), %u iters\n", a, b,
	       host ? "host" : "p2p", (t1 - t0) / iters * 1e6, (double)cyc / iters / khz[a] * 1e3, a, iters);
	CK(cuMemFree(oa));
	CK(cuCtxSetCurrent(ctx[b]));
	CK(cuMemFree(ob));
	if (host) {
		CK(cuCtxSetCurrent(ctx[a]));
		CK(cuMemFreeHost(h));
	} else {
		CK(cuCtxSetCurrent(ctx[a]));
		CK(cuMemFree(fa));
		CK(cuCtxSetCurrent(ctx[b]));
		CK(cuMemFree(fb));
	}
	(void)fb_a; (void)fa_b;
}

static void probe_pp(void) {
	printf("\n== die-to-die flag latency ==\n");
	int pairs[][2] = { {0, 1}, {2, 3}, {0, 2}, {1, 2}, {0, 7}, {3, 4} };
	for (unsigned k = 0; k < sizeof pairs / sizeof pairs[0]; k++)
		if (pairs[k][0] < ndev && pairs[k][1] < ndev) pingpong(pairs[k][0], pairs[k][1], 0, 20000);
	int hpairs[][2] = { {0, 1}, {0, 2}, {0, 7} };
	for (unsigned k = 0; k < sizeof hpairs / sizeof hpairs[0]; k++)
		if (hpairs[k][0] < ndev && hpairs[k][1] < ndev) pingpong(hpairs[k][0], hpairs[k][1], 1, 5000);
}

/* Fill a host buffer with pseudo-random words. */
static void fill(uint32_t *w, size_t words) {
	for (size_t i = 0; i < words; i++) w[i] = (uint32_t)rng();
}

/* rowread on die d over a region (device pointer) of `bytes`; rows are `words` u32 wide. */
static void rowread(int d, CUdeviceptr region, size_t bytes, unsigned nrows, unsigned words, unsigned reps, const char *tag) {
	CK(cuCtxSetCurrent(ctx[d]));
	uint32_t *off = malloc(nrows * 4);
	size_t slots = bytes / 1440;
	for (unsigned i = 0; i < nrows; i++) off[i] = (uint32_t)((rng() % slots) * 1440);
	CUdeviceptr doff, dout;
	CK(cuMemAlloc(&doff, nrows * 4));
	CK(cuMemcpyHtoD(doff, off, nrows * 4));
	CK(cuMemAlloc(&dout, (size_t)nrows * 16));
	CK(cuMemsetD8(dout, 0, (size_t)nrows * 16));
	CUfunction k = fn(d, "rowread");
	void *p[] = { &region, &doff, &nrows, &words, &reps, &dout };
	CK(cuCtxSynchronize());
	double t0 = now();
	CK(cuLaunchKernel(k, nrows, 1, 1, 32, 1, 1, 0, strm[d], p, NULL));
	CK(cuStreamSynchronize(strm[d]));
	double t1 = now();
	uint64_t *out = malloc((size_t)nrows * 16);
	CK(cuMemcpyDtoH(out, dout, (size_t)nrows * 16));
	double maxc = 0, sumc = 0;
	for (unsigned i = 0; i < nrows; i++) { double c = (double)out[2 * i]; sumc += c; if (c > maxc) maxc = c; }
	double bytes_moved = (double)nrows * words * 4 * reps;
	printf("row %-6s die %d rows %5u x %3u B reps %5u: %.0f ns/read (mean), %.0f ns (slowest row), %.2f GB/s, %.1f us/rep for all rows\n",
	       tag, d, nrows, words * 4, reps, sumc / nrows / reps / khz[d] * 1e6, maxc / reps / khz[d] * 1e6,
	       bytes_moved / (t1 - t0) / 1e9, (t1 - t0) / reps * 1e6);
	free(out); free(off);
	CK(cuMemFree(doff));
	CK(cuMemFree(dout));
}

static void probe_row(double gib) {
	size_t bytes = (size_t)(gib * (1ull << 30));
	printf("\n== n-gram row reads (region %.1f GiB) ==\n", gib);
	CK(cuCtxSetCurrent(ctx[0]));
	void *h;
	double t0 = now();
	CK(cuMemHostAlloc(&h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
	double t1 = now();
	fill(h, bytes / 4);
	printf("cuMemHostAlloc %.1f GiB: %.2f s\n", gib, t1 - t0);
	CUdeviceptr hd;
	CK(cuMemHostGetDevicePointer(&hd, h, 0));
	/* IQ4_NL rows: 160 values = 5 blocks x 18 B = 90 B -> 23 words. 16 rows per token. */
	rowread(0, hd, bytes, 16, 23, 2000, "host");
	rowread(0, hd, bytes, 256, 23, 200, "host");
	rowread(0, hd, bytes, 4096, 23, 20, "host");
	rowread(0, hd, bytes, 4096, 32, 20, "host");
	/* same from device memory for comparison */
	size_t dbytes = bytes < (1ull << 30) ? bytes : (1ull << 30);
	CUdeviceptr dd;
	CK(cuMemAlloc(&dd, dbytes));
	CK(cuMemcpyHtoD(dd, h, dbytes));
	rowread(0, dd, dbytes, 16, 23, 2000, "device");
	rowread(0, dd, dbytes, 4096, 23, 20, "device");
	CK(cuMemFree(dd));
	/* host-mapped from a far die too */
	if (ndev > 7) {
		CK(cuCtxSetCurrent(ctx[7]));
		CUdeviceptr hd7;
		CK(cuMemHostGetDevicePointer(&hd7, h, 0));
		rowread(7, hd7, bytes, 16, 23, 2000, "host");
	}
	/* write-combined variant */
	void *hw;
	CK(cuCtxSetCurrent(ctx[0]));
	CK(cuMemHostAlloc(&hw, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_WRITECOMBINED));
	memcpy(hw, h, bytes);
	CUdeviceptr hwd;
	CK(cuMemHostGetDevicePointer(&hwd, hw, 0));
	rowread(0, hwd, bytes, 16, 23, 2000, "hostWC");
	rowread(0, hwd, bytes, 4096, 23, 20, "hostWC");
	CK(cuMemFreeHost(hw));
	/* cuMemHostRegister on an anonymous mapping (the mmap'd-file path) */
	void *m = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_POPULATE, -1, 0);
	if (m == MAP_FAILED) { perror("mmap"); } else {
		memcpy(m, h, bytes);
		t0 = now();
		CUresult r = cuMemHostRegister(m, bytes, CU_MEMHOSTREGISTER_PORTABLE | CU_MEMHOSTREGISTER_DEVICEMAP);
		t1 = now();
		if (r != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r, &s); printf("cuMemHostRegister: %s\n", s); }
		else {
			printf("cuMemHostRegister %.1f GiB: %.2f s\n", gib, t1 - t0);
			CUdeviceptr md;
			CK(cuMemHostGetDevicePointer(&md, m, 0));
			rowread(0, md, bytes, 16, 23, 2000, "reg");
			rowread(0, md, bytes, 4096, 23, 20, "reg");
			CK(cuMemHostUnregister(m));
		}
		munmap(m, bytes);
	}
	CK(cuMemFreeHost(h));
}

static void probe_chase(double gib) {
	size_t bytes = (size_t)(gib * (1ull << 30));
	printf("\n== dependent pointer chase (region %.1f GiB, 1440 B stride) ==\n", gib);
	CK(cuCtxSetCurrent(ctx[0]));
	uint32_t *h;
	CK(cuMemHostAlloc((void **)&h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
	size_t slots = bytes / 1440;
	uint32_t *perm = malloc(slots * 4);
	for (size_t i = 0; i < slots; i++) perm[i] = (uint32_t)i;
	for (size_t i = slots - 1; i > 0; i--) { size_t j = rng() % i; uint32_t t = perm[i]; perm[i] = perm[j]; perm[j] = t; }
	for (size_t i = 0; i < slots; i++) h[(size_t)perm[i] * 360] = perm[(i + 1) % slots] * 360;
	free(perm);
	CUdeviceptr hd, dout;
	CK(cuMemHostGetDevicePointer(&hd, h, 0));
	CK(cuMemAlloc(&dout, 16));
	unsigned start = 0, steps = 20000;
	CUfunction k = fn(0, "chase");
	void *p[] = { &hd, &start, &steps, &dout };
	double t0 = now();
	CK(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, strm[0], p, NULL));
	CK(cuStreamSynchronize(strm[0]));
	double t1 = now();
	uint64_t out[2];
	CK(cuMemcpyDtoH(out, dout, 16));
	printf("chase host: %.0f ns/read (clock), %.0f ns/read (wall)\n", (double)out[0] / steps / khz[0] * 1e6, (t1 - t0) / steps * 1e9);
	CUdeviceptr dd;
	size_t dbytes = bytes < (1ull << 30) ? bytes : (1ull << 30);
	if (dbytes == bytes) {
		CK(cuMemAlloc(&dd, dbytes));
		CK(cuMemcpyHtoD(dd, h, dbytes));
		void *q[] = { &dd, &start, &steps, &dout };
		CK(cuLaunchKernel(k, 1, 1, 1, 1, 1, 1, 0, strm[0], q, NULL));
		CK(cuStreamSynchronize(strm[0]));
		CK(cuMemcpyDtoH(out, dout, 16));
		printf("chase device: %.0f ns/read (clock)\n", (double)out[0] / steps / khz[0] * 1e6);
		CK(cuMemFree(dd));
	}
	CK(cuMemFree(dout));
	CK(cuMemFreeHost(h));
}

static void tput(int d, const char *name, unsigned ctas_per_sm, unsigned threads, unsigned iters) {
	CK(cuCtxSetCurrent(ctx[d]));
	unsigned grid = ctas_per_sm * sms[d];
	CUdeviceptr dout;
	CK(cuMemAlloc(&dout, (size_t)grid * 8 + 16));
	CK(cuMemsetD8(dout, 0, (size_t)grid * 8 + 16));
	CUfunction k = fn(d, name);
	void *p[] = { &iters, &dout };
	double t0 = now();
	CK(cuLaunchKernel(k, grid, 1, 1, threads, 1, 1, 0, strm[d], p, NULL));
	CK(cuStreamSynchronize(strm[d]));
	double t1 = now();
	uint64_t *out = malloc((size_t)grid * 8);
	CK(cuMemcpyDtoH(out, dout, (size_t)grid * 8));
	double maxc = 0;
	for (unsigned i = 0; i < grid; i++) if ((double)out[i] > maxc) maxc = (double)out[i];
	double ops = (double)grid * threads * iters * 64;
	printf("%-9s die %d grid %u x %u: %.1f ops/clk/SM (per-CTA clock), %.2f Tops/s wall, %.0f MHz effective\n", name, d, grid, threads,
	       ops / maxc / sms[d], ops / (t1 - t0) / 1e12, ops / (t1 - t0) / (128.0 * sms[d]) / 1e6);
	free(out);
	CK(cuMemFree(dout));
}

static void probe_xmad(void) {
	printf("\n== issue rates (peak is 128/clk/SM) ==\n");
	tput(0, "xmad_tput", 2, 1024, 2048);
	tput(0, "ffma_tput", 2, 1024, 2048);
	tput(0, "xmad_tput", 4, 256, 2048);
	if (ndev > 7) tput(7, "xmad_tput", 2, 1024, 2048);
}

static void probe_bar(void) {
	printf("\n== software grid barrier ==\n");
	CK(cuCtxSetCurrent(ctx[0]));
	CUfunction k = fn(0, "gridbar");
	int maxb;
	CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&maxb, k, 128, 0));
	printf("resident CTAs/SM at 128 threads: %d\n", maxb);
	for (unsigned per = 1; per <= 4; per *= 2) {
		if ((int)per > maxb) break;
		unsigned grid = per * sms[0], iters = 20000;
		CUdeviceptr sync, dout;
		CK(cuMemAlloc(&sync, 256));
		CK(cuMemsetD8(sync, 0, 256));
		CK(cuMemAlloc(&dout, (size_t)grid * 8));
		CUdeviceptr ctr = sync, gen = sync + 128;
		void *p[] = { &ctr, &gen, &grid, &iters, &dout };
		double t0 = now();
		CK(cuLaunchKernel(k, grid, 1, 1, 128, 1, 1, 0, strm[0], p, NULL));
		CK(cuStreamSynchronize(strm[0]));
		double t1 = now();
		uint64_t *out = malloc((size_t)grid * 8);
		CK(cuMemcpyDtoH(out, dout, (size_t)grid * 8));
		double maxc = 0;
		for (unsigned i = 0; i < grid; i++) if ((double)out[i] > maxc) maxc = (double)out[i];
		printf("gridbar %3u CTAs (%u/SM): %.2f us per barrier (clock), %.2f us (wall)\n", grid, per, maxc / iters / khz[0] * 1e3, (t1 - t0) / iters * 1e6);
		free(out);
		CK(cuMemFree(sync));
		CK(cuMemFree(dout));
	}
}

static void probe_bw(void);
static void probe_pin(double gib) {
	size_t bytes = (size_t)(gib * (1ull << 30));
	printf("\n== pin %.1f GiB of host memory ==\n", gib);
	CK(cuCtxSetCurrent(ctx[0]));
	void *h;
	double t0 = now();
	CUresult r = cuMemHostAlloc(&h, bytes, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP);
	double t1 = now();
	if (r != CUDA_SUCCESS) { const char *s = "?"; cuGetErrorString(r, &s); printf("cuMemHostAlloc: %s\n", s); return; }
	printf("cuMemHostAlloc: %.2f s\n", t1 - t0);
	memset(h, 1, bytes);
	double t2 = now();
	printf("first touch: %.2f s (%.1f GB/s)\n", t2 - t1, (double)bytes / (t2 - t1) / 1e9);
	CUdeviceptr hd;
	CK(cuMemHostGetDevicePointer(&hd, h, 0));
	rowread(0, hd, bytes, 16, 23, 2000, "host");
	rowread(0, hd, bytes, 4096, 23, 20, "host");
	CK(cuMemFreeHost(h));
}

int main(int argc, char **argv) {
	const char *what = argc > 1 ? argv[1] : "all";
	double gib = argc > 2 ? atof(argv[2]) : 1.0;
	init();
	int all = !strcmp(what, "all");
	if (all || !strcmp(what, "p2p")) probe_p2p();
	if (all || !strcmp(what, "pp")) probe_pp();
	if (all || !strcmp(what, "row")) probe_row(gib);
	if (all || !strcmp(what, "chase")) probe_chase(gib);
	if (all || !strcmp(what, "xmad")) probe_xmad();
	if (all || !strcmp(what, "bar")) probe_bar();
	if (!strcmp(what, "pin")) probe_pin(gib);
	if (!strcmp(what, "bw")) probe_bw();
	return 0;
}

/* appended: device streaming bandwidth with 16 B per lane per load */
static void probe_bw(void) {
	printf("\n== device streaming read bandwidth (1 GiB, 16 B per lane) ==\n");
	CK(cuCtxSetCurrent(ctx[0]));
	size_t bytes = 1ull << 30;
	CUdeviceptr d, dout;
	CK(cuMemAlloc(&d, bytes));
	CK(cuMemsetD32(d, 0x01020304, bytes / 4));
	const char *names[] = { "stream", "stream_nc" };
	unsigned grids[] = { 16, 32, 64, 128 };
	for (int k = 0; k < 2; k++) {
		CUfunction f = fn(0, names[k]);
		for (unsigned g = 0; g < 4; g++) {
			unsigned grid = grids[g] * sms[0], threads = 256;
			CK(cuMemAlloc(&dout, (size_t)grid * threads * 4));
			unsigned n = (unsigned)(bytes / 16);
			void *p[] = { &d, &n, &dout };
			CK(cuLaunchKernel(f, grid, 1, 1, threads, 1, 1, 0, strm[0], p, NULL));
			CK(cuStreamSynchronize(strm[0]));
			double t0 = now();
			for (int r = 0; r < 3; r++) CK(cuLaunchKernel(f, grid, 1, 1, threads, 1, 1, 0, strm[0], p, NULL));
			CK(cuStreamSynchronize(strm[0]));
			double t1 = now();
			printf("%-10s grid %5u x %u: %.1f GB/s\n", names[k], grid, threads, 3.0 * bytes / (t1 - t0) / 1e9);
			CK(cuMemFree(dout));
		}
	}
	CK(cuMemFree(d));
}
