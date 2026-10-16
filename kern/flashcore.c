/* flashcore: the long-lived process of the Flash-Next engine. It owns what is slow to build (the
 * dies' contexts and weight arenas, the PLE table pinned in host RAM) and runs the engine library
 * (libengine.so) that does everything else, reloading it on request without disturbing them.
 *
 *   flashcore <ple.bin | -> <libengine.so> <manifest0> [manifest1 ...]
 *
 * The clients speak the line protocol of engine_session() (see engine.c) on stdin/stdout, or, with
 * FLASH_SOCK=<path>, one at a time on that Unix socket (flashd). `reload [path]` swaps the library; a CUDA error
 * in the engine rebuilds the die contexts and the engine (the table is owned by an anchor context on another die).
 * Env: FLASH_MAXT (positions, default 2048), FLASH_PLE_OFF (byte offset of the table in the PLE file,
 * 192 for shard 2 itself, 0 for an extracted copy; the file "-" means no table: the PLE rows read
 * zeros, timing only), FLASH_PLE_CONSUME (see below), plus the engine's own (SPLIT_SHARES, ATTN_DIE,
 * FLASH_CUBIN). */
#define _GNU_SOURCE
#include "core.h"
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { const char *s_ = "?"; cuGetErrorString(r_, &s_); fprintf(stderr, "flashcore.c:%d: %s -> %s\n", __LINE__, #x, s_); exit(1); } } while (0)
static double now(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }

static core_t core;
static void *lib;
static char libpath[1024];
static int (*e_init)(core_t *);
static void (*e_fini)(void);
static int (*e_session)(FILE *, FILE *);
static const char *(*e_reload_path)(void);

/* load the engine library and initialise it; 0 on success. A library that fails to load or
 * initialise is unloaded again so the caller can fall back. */
static int engine_load(const char *path) {
	void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
	if (!h) { fprintf(stderr, "flashcore: %s\n", dlerror()); return -1; }
	int (*i)(core_t *) = dlsym(h, "engine_init");
	void (*f)(void) = dlsym(h, "engine_fini");
	int (*s)(FILE *, FILE *) = dlsym(h, "engine_session");
	const char *(*rp)(void) = dlsym(h, "engine_reload_path");
	if (!i || !f || !s || !rp) { fprintf(stderr, "flashcore: %s lacks engine_init/engine_fini/engine_session/engine_reload_path\n", path); dlclose(h); return -1; }
	double t0 = now();
	if (i(&core)) { fprintf(stderr, "flashcore: engine_init failed in %s\n", path); f(); dlclose(h); return -1; }
	lib = h; e_init = i; e_fini = f; e_session = s; e_reload_path = rp;
	snprintf(libpath, sizeof libpath, "%s", path);
	fprintf(stderr, "flashcore: engine %s ready in %.1f s\n", path, now() - t0);
	return 0;
}
static void engine_unload(void) {
	if (!lib) return;
	e_fini(); dlclose(lib); lib = NULL;
}
/* replace the library (path NULL: reload the current file); on failure bring the old one back */
static int engine_reload(const char *path) {
	char old[1024]; snprintf(old, sizeof old, "%s", libpath);
	char next[1024]; snprintf(next, sizeof next, "%s", path && *path ? path : libpath);
	engine_unload();
	if (!engine_load(next)) return 0;
	fprintf(stderr, "flashcore: falling back to %s\n", old);
	if (engine_load(old)) { fprintf(stderr, "flashcore: cannot bring %s back; exiting\n", old); exit(1); }
	return -1;
}

static char *manifest[CORE_MAXD];
static CUcontext anchor;

/* The die contexts, peer access and weight arenas; 0 on success. A device fault leaves the contexts unusable:
   close_dies + open_dies rebuild them without touching the PLE table, which belongs to the anchor context. */
static int open_dies(void) {
	for (int d = 0; d < core.nd; d++) {
		FILE *f = fopen(manifest[d], "r"); size_t bytes; int die;
		if (!f || fscanf(f, "arena %zu die %d", &bytes, &die) != 2) { fprintf(stderr, "bad manifest %s\n", manifest[d]); if (f) fclose(f); return -1; }
		fclose(f);
		CUdevice dev; if (hold_ordinal(die) < 0) { fprintf(stderr, "die %d is not in CUDA_VISIBLE_DEVICES\n", die); return -1; }
		if (cuDeviceGet(&dev, hold_ordinal(die)) || cuCtxCreate_v2(&core.ctx[d], CU_CTX_SCHED_SPIN, dev)) { fprintf(stderr, "cannot create a context on die %d\n", die); return -1; }
		if (hold_open(&core.ar[d], manifest[d])) return -1;
	}
	for (int d = 0; d < core.nd; d++) { if (cuCtxSetCurrent(core.ctx[d])) return -1; for (int e = 0; e < core.nd; e++) if (e != d && cuCtxEnablePeerAccess(core.ctx[e], 0)) return -1; }
	return 0;
}
static void close_dies(void) {
	for (int d = 0; d < core.nd; d++) {
		if (core.ctx[d]) { cuCtxSetCurrent(core.ctx[d]); hold_close(&core.ar[d]); cuCtxDestroy(core.ctx[d]); core.ctx[d] = NULL; }
	}
}

int main(int argc, char **argv) {
	if (argc < 4) { fprintf(stderr, "usage: flashcore <ple.bin | -> <libengine.so> <manifest0> [manifest1 ...]\n"); return 2; }
	const char *plefile = argv[1], *libfile = argv[2];
	core.abi = CORE_ABI; core.nd = argc - 3;
	if (core.nd > CORE_MAXD) { fprintf(stderr, "at most %d dies\n", CORE_MAXD); return 2; }
	core.maxt = getenv("FLASH_MAXT") ? atoi(getenv("FLASH_MAXT")) : 2048;
	double ts0 = now();
	signal(SIGPIPE, SIG_IGN);
	CK(cuInit(0));
	for (int d = 0; d < core.nd; d++) manifest[d] = argv[3 + d];
	/* the anchor: a context on a die the engine never runs on (FLASH_ANCHOR_DIE, default 7; the first die if that one is not
	   visible) owns the pinned table, so destroying the die contexts after a device fault does not free it */
	{
		const char *ad = getenv("FLASH_ANCHOR_DIE"); int die = ad ? atoi(ad) : 7, ord = hold_ordinal(die);
		if (ord < 0) { fprintf(stderr, "flashcore: anchor die %d is not visible, anchoring on the first die\n", die); ord = 0; }
		CUdevice dev; CK(cuDeviceGet(&dev, ord)); CK(cuCtxCreate_v2(&anchor, 0, dev));
	}
	if (open_dies()) return 1;
	/* ---- the PLE table: pinned host memory, read from the file (CUDA 11.4 cannot pin a file mapping) ---- */
	CK(cuCtxSetCurrent(anchor));
	const size_t PLE_ROWS = 320001536, PLE_BYTES = PLE_ROWS * 90;
	int have = strcmp(plefile, "-") != 0;
	size_t sz = have ? PLE_BYTES : (size_t)core.maxt * 16 * 90;   /* no table: a zero compact table (16 rows per position) keeps plerows in bounds */
	CK(cuMemHostAlloc((void **)&core.ple_host, sz + 4096, CU_MEMHOSTALLOC_PORTABLE | CU_MEMHOSTALLOC_DEVICEMAP));
	memset(core.ple_host, 0, have ? 4096 : sz + 4096);
	core.ple_compact = !have;
	if (have) {
		int fd = open(plefile, O_RDONLY); if (fd < 0) { perror(plefile); return 1; }
		off_t off = getenv("FLASH_PLE_OFF") ? (off_t)atoll(getenv("FLASH_PLE_OFF")) : 0;
		/* FLASH_PLE_CONSUME=1: the file is a tmpfs copy that does not fit in RAM next to the pinned
		   table; read it from the end and truncate as we go, so its pages are freed as ours fill */
		int consume = getenv("FLASH_PLE_CONSUME") != NULL; if (consume) { close(fd); fd = open(plefile, O_RDWR); if (fd < 0) { perror(plefile); return 1; } }
		size_t done = 0; double tp0 = now();
		while (done < PLE_BYTES) {
			size_t chunk = PLE_BYTES - done > (64u << 20) ? (64u << 20) : PLE_BYTES - done;
			size_t at = consume ? PLE_BYTES - done - chunk : done;   /* backwards when consuming */
			size_t got = 0;
			while (got < chunk) { ssize_t n = pread(fd, core.ple_host + at + got, chunk - got, off + (off_t)(at + got)); if (n <= 0) { perror("pread ple"); return 1; } got += (size_t)n; }
			if (consume && ftruncate(fd, off + (off_t)at)) { perror("ftruncate ple"); return 1; }
			done += chunk;
			if ((done >> 30) != ((done - chunk) >> 30)) fprintf(stderr, "\rPLE table: %zu of %zu GB (%.0f MB/s)", done >> 30, PLE_BYTES >> 30, done / 1e6 / (now() - tp0));
		}
		if (consume) unlink(plefile);
		close(fd);
		fprintf(stderr, "\rPLE table: %.2f GB pinned in %.0f s\n", PLE_BYTES / 1e9, now() - tp0);
	}
	fprintf(stderr, "flashcore: %d dies, weights and PLE table resident after %.0f s\n", core.nd, now() - ts0);
	if (engine_load(libfile)) fprintf(stderr, "flashcore: starting without an engine; \"reload <path>\" loads one (the weights and the table stay resident)\n");
	printf("ready\n"); fflush(stdout);

	const char *sock = getenv("FLASH_SOCK");
	int ls = -1;
	if (sock) {
		ls = socket(AF_UNIX, SOCK_STREAM, 0); struct sockaddr_un sa = { .sun_family = AF_UNIX };
		snprintf(sa.sun_path, sizeof sa.sun_path, "%s", sock); unlink(sock);
		if (ls < 0 || bind(ls, (struct sockaddr *)&sa, sizeof sa) || listen(ls, 4)) { perror(sock); return 1; }
		fprintf(stderr, "flashcore: listening on %s\n", sock);
	}
	for (;;) {
		FILE *in = stdin, *out = stdout;
		if (ls >= 0) {
			int c = accept(ls, NULL, NULL); if (c < 0) { perror("accept"); continue; }
			in = fdopen(c, "r"); out = fdopen(dup(c), "w");
		}
		int q;
		if (lib) q = e_session(in, out);
		else {   /* no engine: only reload and quit are understood */
			char buf[1024]; q = 0;
			while (fgets(buf, sizeof buf, in)) {
				if (!strncmp(buf, "quit", 4)) { q = 1; break; }
				if (!strncmp(buf, "reload", 6)) {
					char path[1000] = ""; sscanf(buf + 6, "%999s", path);
					int rc = engine_load(*path ? path : libfile);
					fprintf(out, rc ? "error: could not load the engine\n" : "ok: engine loaded\n"); fflush(out);
					if (!rc) break;
				} else { fprintf(out, "error: no engine loaded; send \"reload <path>\"\n"); fflush(out); }
			}
		}
		if (q == 3) {   /* the engine hit a CUDA error: the contexts are unusable, the table (anchor) is not */
			fprintf(stderr, "flashcore: device fault; rebuilding the die contexts and the engine\n");
			char keep[1024]; snprintf(keep, sizeof keep, "%s", libpath);
			engine_unload(); close_dies();
			int rc = open_dies() ? -1 : engine_load(keep);
			fprintf(out, rc ? "error: could not recover from the device fault; the process exits\n" : "ok: contexts rebuilt, engine reloaded\n"); fflush(out);
			if (rc) { if (ls >= 0) { fclose(in); fclose(out); } exit(1); }
		}
		if (q == 2) {
			const char *next = e_reload_path();
			char keep[1024]; snprintf(keep, sizeof keep, "%s", next ? next : "");
			int rc = engine_reload(keep);
			fprintf(out, rc ? "error: reload failed, the previous engine is back\n" : "ok: engine reloaded\n"); fflush(out);
		}
		if (ls >= 0) { fclose(in); fclose(out); } else if (q != 2) break;
		if (q == 1) break;
	}
	engine_unload();
	close_dies();
	if (ls >= 0) { close(ls); unlink(sock); }
	return 0;
}
