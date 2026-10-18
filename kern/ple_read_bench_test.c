/* CPU-only regression of benchmark validation and raw record semantics. */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>
static ssize_t fixture_pread(int, void *, size_t, off_t);
static ssize_t fixture_reference_pread(int, void *, size_t, off_t);
static int fixture_fadvise(int, off_t, off_t, int);
static int fixture_mincore(void *, size_t, unsigned char *);
#define PLE_BENCH_PREAD fixture_pread
#define PLE_BENCH_REFERENCE_PREAD fixture_reference_pread
#define PLE_BENCH_FADVISE fixture_fadvise
#define PLE_BENCH_MINCORE fixture_mincore
#define main ple_read_benchmark_program_main
#include "ple_read_bench.c"
#undef main
enum {
	FIXTURE_OK, FIXTURE_READ_ERROR, FIXTURE_REFERENCE_ERROR,
	FIXTURE_REFERENCE_MISMATCH, FIXTURE_HINT_ERROR,
	FIXTURE_RESIDENCY_ERROR, FIXTURE_ONE_BYTE_READS
};
static int fixture_fault;
static off_t fixture_fault_at;
static unsigned char fixture_bytes[8192];
static ssize_t fixture_pread(int fd, void *dst, size_t count, off_t offset) {
	if (fixture_fault == FIXTURE_READ_ERROR &&
	    offset >= fixture_fault_at && offset < fixture_fault_at + 90) {
		errno = EIO; return -1;
	}
	if (fixture_fault == FIXTURE_ONE_BYTE_READS && count > 1) count = 1;
	return pread(fd, dst, count, offset);
}
static ssize_t fixture_reference_pread(int fd, void *dst, size_t count, off_t offset) {
	if (fixture_fault == FIXTURE_REFERENCE_ERROR &&
	    offset >= fixture_fault_at && offset < fixture_fault_at + 90) {
		errno = EIO; return -1;
	}
	ssize_t n = pread(fd, dst, count, offset);
	if (fixture_fault == FIXTURE_REFERENCE_MISMATCH && offset == fixture_fault_at && n > 0)
		((unsigned char *)dst)[0] ^= 1;
	return n;
}
static int fixture_fadvise(int fd, off_t offset, off_t length, int advice) {
	(void)fd; (void)offset; (void)length;
	return fixture_fault == FIXTURE_HINT_ERROR && advice == POSIX_FADV_WILLNEED ? EIO : 0;
}
static int fixture_mincore(void *addr, size_t length, unsigned char *vec) {
	if (fixture_fault == FIXTURE_RESIDENCY_ERROR) { errno = ENOSYS; return -1; }
	return mincore(addr, length, vec);
}
static void check_case(int fd, uint8_t *mapping, long page,
                       struct ple_read_pool *pool, const off_t offsets[16],
                       int fault, unsigned expected_failure,
                       const uint8_t *expected_bytes, uint8_t *copy) {
	FILE *out = tmpfile(), *audit = tmpfile();
	assert(out && audit);
	fixture_fault = fault;
	struct bench_case bc = {
		.phase = "warm", .trial = 7, .sample_set = 3, .repeat = 2,
		.order = expected_bytes ? 1u : 0u, .expected = expected_bytes, .copy = copy
	};
	int rc = measure(out, audit, fd, mapping, page, pool, &bc, offsets);
	assert((rc != 0) == (expected_failure != 0));
	assert(!fseek(out, 0, SEEK_SET));
	char line[4096];
	assert(fgets(line, sizeof line, out));
	assert(fgetc(out) == EOF); /* Exactly one retained row, even on failure. */
	char *field[32], *next = line;
	unsigned nfield = 0;
	while (next) {
		assert(nfield < sizeof field / sizeof *field);
		field[nfield++] = strsep(&next, ",");
	}
	assert(nfield == 27);
	assert(!strcmp(field[0], "warm"));
	assert(!strcmp(field[13], "2"));
	assert(!strcmp(field[14], PLE_BENCH_INSTRUMENTATION));
	assert(!strcmp(field[15], "same_inputs"));
	assert(!strcmp(field[16], "3") && !strcmp(field[17], "2"));
	assert(strtoul(field[20], NULL, 10) == !expected_failure);
	assert(strtoul(field[21], NULL, 10) == expected_failure);
	assert(strtoull(field[7], NULL, 10) ==
	       strtoull(field[5], NULL, 10) + strtoull(field[6], NULL, 10));
	if (expected_failure & PLE_BENCH_FAIL_REFERENCE) {
		assert(!strcmp(field[10], "0"));
		assert(strtoul(field[23], NULL, 10) > 0);
	}
	if (expected_failure & PLE_BENCH_FAIL_PARITY) assert(strtoul(field[24], NULL, 10) > 0);
	if (expected_failure & PLE_BENCH_FAIL_HINT) assert(strtol(field[11], NULL, 10) == EIO);
	if (expected_failure & PLE_BENCH_FAIL_RESIDENCY) {
		assert(!strcmp(field[19], "unknown"));
		assert(strtol(field[22], NULL, 10) == ENOSYS);
	}
#ifdef PLE_READ_AUDIT
	if (expected_failure & PLE_BENCH_FAIL_AUDIT) {
		assert(strtoul(field[25], NULL, 10) == 128);
		assert(strtoul(field[26], NULL, 10) == 16 * 90 - 128);
	} else {
		assert(strtoul(field[25], NULL, 10) == 16);
		assert(strtoul(field[26], NULL, 10) == 0);
	}
#else
	assert(strtoul(field[25], NULL, 10) == 0 && strtoul(field[26], NULL, 10) == 0);
#endif
	assert(!fclose(out) && !fclose(audit));
}
int main(void) {
	assert(!strcmp(cache_class(0, 16, 0), "client_absent"));
	assert(!strcmp(cache_class(7, 16, 0), "client_mixed"));
	assert(!strcmp(cache_class(16, 16, 0), "client_resident"));
	assert(!strcmp(cache_class(16, 16, ENOSYS), "unknown"));
	assert(!strcmp(cache_class(17, 16, 0), "unknown"));
	for (unsigned set = 0; set < 8; set++) {
		unsigned first[2] = {0, 0};
		for (unsigned repeat = 0; repeat < 4; repeat++) {
			unsigned a = warm_arm(set, repeat, 0), b = warm_arm(set, repeat, 1);
			assert(a < 2 && b < 2 && a != b);
			first[a]++;
		}
		assert(first[0] == 2 && first[1] == 2);
	}
	char path[] = "/tmp/flash-ple-bench-test-XXXXXX";
	int fd = mkstemp(path); assert(fd >= 0 && !unlink(path));
	for (size_t i = 0; i < sizeof fixture_bytes; i++)
		fixture_bytes[i] = (unsigned char)(i * 37 + i / 251);
	for (size_t at = 0; at < sizeof fixture_bytes;) {
		ssize_t n = write(fd, fixture_bytes + at, sizeof fixture_bytes - at);
		if (n < 0 && errno == EINTR) continue;
		assert(n > 0); at += (size_t)n;
	}
	uint8_t *mapping = mmap(NULL, sizeof fixture_bytes, PROT_READ, MAP_SHARED, fd, 0);
	assert(mapping != MAP_FAILED);
	long page = sysconf(_SC_PAGESIZE); assert(page >= 90);
	off_t offsets[16];
	for (unsigned h = 0; h < 16; h++) offsets[h] = (off_t)(h * 256);
	fixture_fault_at = offsets[3];
	for (unsigned readers = 1; readers <= 16; readers += 15) {
		struct ple_read_pool pool; assert(!ple_read_pool_init(&pool, readers));
		uint8_t first[16 * 90];
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_OK, 0, NULL, first);
		for (unsigned h = 0; h < 16; h++) assert(!memcmp(first + h * 90, fixture_bytes + offsets[h], 90));
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_OK, 0, first, NULL);
		first[0] ^= 1;
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_OK, PLE_BENCH_FAIL_PAIR, first, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_READ_ERROR, PLE_BENCH_FAIL_READ, NULL, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_REFERENCE_ERROR, PLE_BENCH_FAIL_REFERENCE, NULL, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_REFERENCE_MISMATCH, PLE_BENCH_FAIL_PARITY, NULL, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_HINT_ERROR, PLE_BENCH_FAIL_HINT, NULL, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_RESIDENCY_ERROR, PLE_BENCH_FAIL_RESIDENCY, NULL, NULL);
		check_case(fd, mapping, page, &pool, offsets, FIXTURE_ONE_BYTE_READS,
#ifdef PLE_READ_AUDIT
		           PLE_BENCH_FAIL_AUDIT,
#else
		           0,
#endif
		           NULL, NULL);
		ple_read_pool_destroy(&pool);
	}
	assert(!munmap(mapping, sizeof fixture_bytes) && !close(fd));
	puts("ple_read_bench: failed-row retention, scalar and paired byte parity, hint/cache failures, per-set order balance, and audit completeness passed");
	return 0;
}
