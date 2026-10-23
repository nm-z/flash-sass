/*
 * Read-only PLE I/O benchmark. Never evicts cache pages.
 * Unprimed arms use disjoint shuffled 1 MiB regions; warm arms reuse the same rows.
 * mincore reports the client page cache, not NFS server or physical-media state.
 */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif
static uint64_t ns(void) {
	struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000u + (uint64_t)t.tv_nsec;
}
#ifndef PLE_BENCH_PREAD
#define PLE_BENCH_PREAD pread
#endif
#ifndef PLE_BENCH_REFERENCE_PREAD
#define PLE_BENCH_REFERENCE_PREAD pread
#endif
#ifndef PLE_BENCH_FADVISE
#define PLE_BENCH_FADVISE posix_fadvise
#endif
#ifndef PLE_BENCH_MINCORE
#define PLE_BENCH_MINCORE mincore
#endif
#ifdef PLE_READ_AUDIT
#define PLE_BENCH_INSTRUMENTATION "pread_audit"
#else
#define PLE_BENCH_INSTRUMENTATION "none"
#endif
enum {
	PLE_BENCH_FAIL_HINT = 1u,
	PLE_BENCH_FAIL_READ = 2u,
	PLE_BENCH_FAIL_REFERENCE = 4u,
	PLE_BENCH_FAIL_PARITY = 8u,
	PLE_BENCH_FAIL_RESIDENCY = 16u,
	PLE_BENCH_FAIL_PAIR = 32u,
	PLE_BENCH_FAIL_AUDIT = 64u
};
struct bench_case {
	const char *phase;
	unsigned trial, sample_set, repeat, order;
	/* First warm arm saves bytes; second compares those exact bytes. */
	const uint8_t *expected;
	uint8_t *copy;
};
static const char bench_csv_header[] =
	"phase,trial,readers,resident_pages,pages,hint_ns,read_ns,total_ns,"
	"hash,bytes,errors,hint_error,offsets,schema_version,instrumentation,"
	"comparison,sample_set,repeat,order,cache_class,valid,failure_mask,"
	"residency_error,reference_errors,parity_errors,audit_recorded,audit_dropped";
static const char *cache_class(unsigned resident, unsigned pages, int error) {
	if (error || !pages || resident > pages) return "unknown";
	if (!resident) return "client_absent";
	return resident == pages ? "client_resident" : "client_mixed";
}
/* Each fixed row set receives both arm orders across four repetitions. */
static inline unsigned warm_arm(unsigned sample_set, unsigned repeat, unsigned order) {
	return order ^ ((sample_set + repeat) & 1u);
}

#ifdef PLE_READ_AUDIT
struct read_event { long tid; off_t offset; uint64_t begin, end; ssize_t got; };
static struct read_event events[128];
static unsigned nevents, active, peak;
static ssize_t audit_pread(int fd, void *dst, size_t count, off_t offset) {
	unsigned i = __atomic_fetch_add(&nevents, 1, __ATOMIC_RELAXED);
	long tid = syscall(SYS_gettid);
	uint64_t begin = ns();
	unsigned n = __atomic_add_fetch(&active, 1, __ATOMIC_RELAXED), old = __atomic_load_n(&peak, __ATOMIC_RELAXED);
	while (old < n && !__atomic_compare_exchange_n(&peak, &old, n, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {}
	ssize_t got = PLE_BENCH_PREAD(fd, dst, count, offset); int saved_errno = errno;
	uint64_t end = ns();
	__atomic_sub_fetch(&active, 1, __ATOMIC_RELAXED);
	if (i < 128) events[i] = (struct read_event){tid, offset, begin, end, got};
	errno = saved_errno;
	return got;
}
#define PLE_READ_PREAD audit_pread
#else
#define PLE_READ_PREAD PLE_BENCH_PREAD
#endif
#include "ple_read_pool.h"

static uint64_t rng_state;
static uint64_t random64(void) {
	uint64_t z = (rng_state += UINT64_C(0x9e3779b97f4a7c15));
	z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
	return z ^ (z >> 31);
}
static void fail(const char *what) { perror(what); exit(1); }
static uint64_t number(const char *s) {
	char *end; errno = 0;
	if (!*s || *s == '-' || *s == '+') { fprintf(stderr, "invalid number: %s\n", s); exit(2); }
	uint64_t n = strtoull(s, &end, 10);
	if (errno || *end) { fprintf(stderr, "invalid number: %s\n", s); exit(2); }
	return n;
}
static int residency(uint8_t *mapping, long page, const off_t offsets[16],
                     unsigned *resident, unsigned *total) {
	int error = 0;
	*resident = *total = 0;
	for (int h = 0; h < 16; h++) {
		size_t start = (size_t)offsets[h] / (size_t)page * (size_t)page;
		size_t n = ((size_t)offsets[h] - start + 89) / (size_t)page + 1;
		unsigned char vec[2] = {0};
		*total += (unsigned)n;
		if (PLE_BENCH_MINCORE(mapping + start, n * (size_t)page, vec)) {
			if (!error) error = errno ? errno : EIO;
			continue;
		}
		for (size_t j = 0; j < n; j++) *resident += (vec[j] & 1) != 0;
	}
	return error;
}
static int measure(FILE *out, FILE *audit, int fd, uint8_t *mapping, long page,
                   struct ple_read_pool *pool, const struct bench_case *bc,
                   const off_t offsets[16]) {
	uint8_t dst[16 * 90], reference[16 * 90];
	struct ple_read_result results[16];
	unsigned resident, pages;
	int residency_error = residency(mapping, page, offsets, &resident, &pages);
	memset(dst, 0xa5, sizeof dst);
#ifdef PLE_READ_AUDIT
	nevents = active = peak = 0;
#endif
	uint64_t t0 = ns();
	int hint_error = 0;
	for (int h = 0; h < 16; h++) {
		int rc = PLE_BENCH_FADVISE(fd, offsets[h], 90, POSIX_FADV_WILLNEED);
		if (rc && !hint_error) hint_error = rc;
	}
	uint64_t t1 = ns();
	ple_read_batch(pool, fd, offsets, dst, results);
	uint64_t t2 = ns();
	unsigned bytes = 0, errors = 0;
	for (int h = 0; h < 16; h++) {
		bytes += (unsigned)results[h].got;
		errors += results[h].got != 90 || results[h].error != 0;
	}
	/* Verification is outside [t0,t2]. Failures retain timings and offsets
	 * in a raw record, but cannot be reported as valid. */
	unsigned reference_errors = 0, parity_errors = 0, pair_errors = 0;
	if (!errors) {
		for (int h = 0; h < 16; h++) {
			size_t got = 0;
			while (got < 90) {
				ssize_t n = PLE_BENCH_REFERENCE_PREAD(fd, reference + h * 90 + got,
				                                    90 - got, offsets[h] + (off_t)got);
				if (n < 0 && errno == EINTR) continue;
				if (n <= 0) {
					int error = n < 0 ? errno : 0;
					fprintf(audit, "reference_error,phase=%s,trial=%u,head=%d,got=%zu,errno=%d\n",
					        bc->phase, bc->trial, h, got, error);
					reference_errors++;
					break;
				}
				got += (size_t)n;
			}
		}
		if (!reference_errors) parity_errors = memcmp(dst, reference, sizeof dst) != 0;
		if (bc->expected) pair_errors = memcmp(dst, bc->expected, sizeof dst) != 0;
	}
	unsigned audit_recorded = 0, audit_dropped = 0;
#ifdef PLE_READ_AUDIT
	audit_recorded = nevents < 128 ? nevents : 128;
	audit_dropped = nevents - audit_recorded;
#endif
	unsigned failure =
		(hint_error ? PLE_BENCH_FAIL_HINT : 0u) |
		(errors ? PLE_BENCH_FAIL_READ : 0u) |
		(reference_errors ? PLE_BENCH_FAIL_REFERENCE : 0u) |
		(parity_errors ? PLE_BENCH_FAIL_PARITY : 0u) |
		(residency_error ? PLE_BENCH_FAIL_RESIDENCY : 0u) |
		(pair_errors ? PLE_BENCH_FAIL_PAIR : 0u) |
		(audit_dropped ? PLE_BENCH_FAIL_AUDIT : 0u);
	int valid = failure == 0;
	uint64_t hash = 0;
	if (!errors) {
		hash = UINT64_C(0xcbf29ce484222325);
		for (unsigned i = 0; i < sizeof dst; i++) {
			hash ^= dst[i]; hash *= UINT64_C(0x100000001b3);
		}
	}
	if (valid && bc->copy) memcpy(bc->copy, dst, sizeof dst);
	/* Publish after read, hint, reference, byte, and audit validation. */
	fprintf(out, "%s,%u,%u,%u,%u,%" PRIu64 ",%" PRIu64 ",%" PRIu64
	        ",%016" PRIx64 ",%u,%u,%d,\"",
	        bc->phase, bc->trial, pool->readers, resident, pages,
	        t1-t0, t2-t1, t2-t0, hash, bytes, errors, hint_error);
	for (int h = 0; h < 16; h++) fprintf(out, "%s%lld", h ? ";" : "", (long long)offsets[h]);
	fprintf(out, "\",2,%s,%s,%u,%u,%u,%s,%d,%u,%d,%u,%u,%u,%u\n",
	        PLE_BENCH_INSTRUMENTATION,
	        !strcmp(bc->phase, "warm") ? "same_inputs" : "disjoint_inputs",
	        bc->sample_set, bc->repeat, bc->order, cache_class(resident, pages, residency_error),
	        valid, failure, residency_error, reference_errors, parity_errors,
	        audit_recorded, audit_dropped);
#ifdef PLE_READ_AUDIT
	fprintf(audit, "audit_batch,%s,%u,%u,peak_call_regions=%u,calls=%u,"
	        "events_recorded=%u,events_dropped=%u,valid=%d\n",
	        bc->phase, bc->trial, pool->readers, peak, nevents, audit_recorded, audit_dropped, valid);
	for (unsigned i = 0; i < audit_recorded; i++)
		fprintf(audit, "audit_read,%s,%u,%u,%ld,%lld,%" PRIu64 ",%" PRIu64 ",%zd\n",
		        bc->phase, bc->trial, pool->readers, events[i].tid,
		        (long long)events[i].offset, events[i].begin, events[i].end, events[i].got);
#endif
	if (!valid) fprintf(audit, "failed_batch,phase=%s,trial=%u,readers=%u,failure_mask=%u\n",
	                    bc->phase, bc->trial, pool->readers, failure);
	if (fflush(out) || ferror(out) || fflush(audit) || ferror(audit)) return -1;
	return valid ? 0 : -1;
}

int main(int argc, char **argv) {
	if (argc < 5 || argc > 6) {
		fprintf(stderr, "usage: %s <PLE-file> <table-offset> <batches-per-arm 1..512> <seed> [readers 2..16]\n", argv[0]); return 2;
	}
	uint64_t base = number(argv[2]), trials64 = number(argv[3]), seed = number(argv[4]);
	unsigned readers;
	if (trials64 < 1 || trials64 > 512 || ple_read_parse_readers(argc == 6 ? argv[5] : "16", &readers) || readers < 2) {
		fprintf(stderr, "batches must be 1..512 and parallel readers 2..16\n"); return 2;
	}
	unsigned trials = (unsigned)trials64;
	int fd = open(argv[1], O_RDONLY); if (fd < 0) fail(argv[1]);
	struct stat st; if (fstat(fd, &st)) fail("fstat");
	if (!S_ISREG(st.st_mode) || st.st_size <= 0 || base >= (uint64_t)st.st_size) { fprintf(stderr, "invalid regular file/table offset\n"); return 2; }
	const uint64_t region = 1u << 20;
	size_t nregions = ((uint64_t)st.st_size - base) / region;
	if (nregions < (size_t)trials * 2 * 16) { fprintf(stderr, "file too small for disjoint samples\n"); return 2; }
	long page = sysconf(_SC_PAGESIZE); if (page < 90) { fprintf(stderr, "unsupported page size\n"); return 2; }
	uint8_t *mapping = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	if (mapping == MAP_FAILED) fail("mmap");
	int rc = posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM);
	if (rc) { errno = rc; fail("POSIX_FADV_RANDOM"); }
	size_t *regions = malloc(nregions * sizeof *regions);
	off_t (*sets)[16] = calloc((size_t)trials * 2, sizeof *sets);
	if (!regions || !sets) fail("allocate samples");
	rng_state = seed;
	for (size_t i = 0; i < nregions; i++) regions[i] = i;
	for (size_t i = nregions; i > 1; i--) {
		size_t j = random64() % i, tmp = regions[i-1]; regions[i-1] = regions[j]; regions[j] = tmp;
	}
	for (unsigned i = 0; i < trials * 2; i++) for (unsigned h = 0; h < 16; h++) {
		uint64_t byte = regions[(size_t)i * 16 + h] * region + 8192 + random64() % (region - 16384 - 90);
		sets[i][h] = (off_t)(base + (byte / 90) * 90);
	}
	free(regions);
	struct ple_read_pool pools[2];
	rc = ple_read_pool_init(&pools[0], 1); if (rc) { errno = rc; fail("serial reader init"); }
	rc = ple_read_pool_init(&pools[1], readers); if (rc) { errno = rc; fail("parallel reader init"); }
	char host[256] = ""; gethostname(host, sizeof host - 1);
	fprintf(stderr, "revision=%s hostname=%s file=%s offset=%" PRIu64 " seed=%" PRIu64 " trials=%u readers=%u page_size=%ld\n",
	        FLASH_REV, host, argv[1], base, seed, trials, readers, page);
	puts(bench_csv_header);
	struct rusage before, after;
	getrusage(RUSAGE_SELF, &before);
	int failed = 0;
	for (unsigned i = 0; i < trials && !failed; i++) {
		for (unsigned order = 0; order < 2; order++) {
			unsigned arm = order ^ (i & 1);
			unsigned set = i * 2 + arm;
			struct bench_case bc = {
				.phase = "unprimed", .trial = i, .sample_set = set,
				.repeat = 0, .order = order
			};
			if (measure(stdout, stderr, fd, mapping, page, &pools[arm], &bc, sets[set])) {
				failed = 1; break;
			}
		}
	}
#ifndef PLE_READ_AUDIT
	for (unsigned i = 0; i < trials * 8 && !failed; i++) {
		unsigned set = i % (trials * 2), repeat = i / (trials * 2);
		uint8_t first_bytes[16 * 90];
		for (unsigned order = 0; order < 2; order++) {
			unsigned arm = warm_arm(set, repeat, order);
			struct bench_case bc = {
				.phase = "warm", .trial = i, .sample_set = set,
				.repeat = repeat, .order = order,
				.expected = order ? first_bytes : NULL, .copy = order ? NULL : first_bytes
			};
			if (measure(stdout, stderr, fd, mapping, page, &pools[arm], &bc, sets[set])) {
				failed = 1; break;
			}
		}
	}
#endif
	getrusage(RUSAGE_SELF, &after);
	double user = after.ru_utime.tv_sec-before.ru_utime.tv_sec + (after.ru_utime.tv_usec-before.ru_utime.tv_usec)*1e-6;
	double system = after.ru_stime.tv_sec-before.ru_stime.tv_sec + (after.ru_stime.tv_usec-before.ru_stime.tv_usec)*1e-6;
	if (fflush(stdout) || ferror(stdout)) failed = 1;
	fprintf(stderr, "pass=%d user_s=%.6f system_s=%.6f voluntary_switches=%ld involuntary_switches=%ld max_rss_kib=%ld\n",
	        !failed, user, system, after.ru_nvcsw-before.ru_nvcsw, after.ru_nivcsw-before.ru_nivcsw, after.ru_maxrss);
	ple_read_pool_destroy(&pools[1]); ple_read_pool_destroy(&pools[0]);
	free(sets); munmap(mapping, (size_t)st.st_size); close(fd);
	if (fflush(stderr) || ferror(stderr)) failed = 1;
	return failed;
}
