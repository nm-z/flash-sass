/* Host-only regression of the same reader used by engine.c. */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static ssize_t test_pread(int, void *, size_t, off_t);
static int test_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
static int test_join(pthread_t, void **);
#define PLE_READ_PREAD test_pread
#define pthread_create test_create
#define pthread_join test_join
#include "ple_read_pool.h"
#undef pthread_create
#undef pthread_join

enum { NORMAL, PARTIAL, INTERRUPTED, END_OF_FILE, IO_ERROR };
static int fault, injected, fail_create = -1, create_calls, join_calls;
static off_t fault_at;

static ssize_t test_pread(int fd, void *buf, size_t count, off_t offset) {
	if (fault && offset >= fault_at && offset < fault_at + 90) {
		size_t got = (size_t)(offset - fault_at);
		if (got < 17) { if (count > 17 - got) count = 17 - got; }
		else if (fault == PARTIAL) { if (count > 7) count = 7; }
		else if (fault == INTERRUPTED && !__atomic_exchange_n(&injected, 1, __ATOMIC_RELAXED)) { errno = EINTR; return -1; }
		else if (fault == END_OF_FILE) return 0;
		else if (fault == IO_ERROR) { errno = EIO; return -1; }
	}
	return pread(fd, buf, count, offset);
}
static int test_create(pthread_t *t, const pthread_attr_t *a, void *(*fn)(void *), void *arg) {
	if (create_calls++ == fail_create) return EAGAIN;
	return pthread_create(t, a, fn, arg);
}
static int test_join(pthread_t t, void **out) { join_calls++; return pthread_join(t, out); }

static unsigned char fixture[131072];
static void check_rows(int fd, struct ple_read_pool *pool, const off_t offsets[16]) {
	unsigned char guarded[16 * 90 + 32];
	struct ple_read_result results[16];
	memset(guarded, 0xa5, sizeof guarded);
	ple_read_batch(pool, fd, offsets, guarded + 16, results);
	for (int h = 0; h < 16; h++) {
		assert(results[h].got == 90 && results[h].error == 0);
		assert(!memcmp(guarded + 16 + h * 90, fixture + offsets[h], 90));
	}
	for (int i = 0; i < 16; i++) assert(guarded[i] == 0xa5 && guarded[16 + 16 * 90 + i] == 0xa5);
}
int main(void) {
	unsigned readers;
	assert(!ple_read_parse_readers(NULL, &readers) && readers == 16);
	const char *bad[] = {"", "0", "17", "-1", "+1", " 1", "1 ", "1x", "999999999999999999"};
	for (unsigned i = 0; i < sizeof bad / sizeof *bad; i++) assert(ple_read_parse_readers(bad[i], &readers) == EINVAL);
	char path[] = "/tmp/flash-ple-read-test-XXXXXX";
	int fd = mkstemp(path); assert(fd >= 0); assert(!unlink(path));
	for (size_t i = 0; i < sizeof fixture; i++) fixture[i] = (unsigned char)(i * 37 + i / 251);
	for (size_t done = 0; done < sizeof fixture; ) {
		ssize_t n = write(fd, fixture + done, sizeof fixture - done); assert(n > 0); done += (size_t)n;
	}
	struct ple_read_pool pool = {0};
	ple_read_pool_destroy(&pool);
	assert(ple_read_pool_init(&pool, 0) == EINVAL);
	assert(ple_read_pool_init(&pool, 17) == EINVAL);
	for (readers = 1; readers <= 16; readers++) {
		char number[8]; snprintf(number, sizeof number, "%u", readers);
		unsigned parsed; assert(!ple_read_parse_readers(number, &parsed) && parsed == readers);
		assert(!ple_read_pool_init(&pool, readers));
		for (unsigned generation = 0; generation < 128; generation++) {
			off_t offsets[16];
			for (unsigned h = 0; h < 16; h++) offsets[h] = (off_t)((generation * 977 + h * 4003) % (sizeof fixture - 90));
			offsets[0] = 0; offsets[1] = sizeof fixture - 90;
			offsets[2] = 4090; offsets[3] = offsets[7]; /* page straddle and duplicate row */
			check_rows(fd, &pool, offsets);
		}
		off_t offsets[16]; for (int h = 0; h < 16; h++) offsets[h] = h * 1000;
		fault_at = offsets[3];
		for (fault = PARTIAL; fault <= INTERRUPTED; fault++) { injected = 0; check_rows(fd, &pool, offsets); }
		for (fault = END_OF_FILE; fault <= IO_ERROR; fault++) {
			unsigned char data[16 * 90]; struct ple_read_result results[16]; memset(data, 0xa5, sizeof data);
			ple_read_batch(&pool, fd, offsets, data, results);
			for (int h = 0; h < 16; h++) {
				size_t expected = h == 3 ? 17 : 90;
				assert(results[h].got == expected);
				assert(results[h].error == (h == 3 && fault == IO_ERROR ? EIO : 0));
				assert(!memcmp(data + h * 90, fixture + offsets[h], expected));
				for (size_t b = expected; b < 90; b++) assert(data[h * 90 + b] == 0xa5);
			}
		}
		fault = NORMAL;
		unsigned char data[16 * 90]; struct ple_read_result results[16];
		ple_read_batch(&pool, -1, offsets, data, results);
		for (int h = 0; h < 16; h++) assert(results[h].got == 0 && results[h].error == EBADF);
		offsets[5] = sizeof fixture - 11;
		ple_read_batch(&pool, fd, offsets, data, results);
		assert(results[5].got == 11 && results[5].error == 0);
		ple_read_pool_destroy(&pool); ple_read_pool_destroy(&pool);
	}
	/* Fail each possible thread start, including after other workers are waiting. */
	for (int fail = 0; fail < 16; fail++) {
		fail_create = fail; create_calls = join_calls = 0;
		assert(ple_read_pool_init(&pool, 16) == EAGAIN);
		assert(join_calls == fail && !pool.sync_ready && !pool.started);
		ple_read_pool_destroy(&pool);
	}
	fail_create = -1;
	assert(!ple_read_pool_init(&pool, 16));
	off_t offsets[16] = {0}; check_rows(fd, &pool, offsets);
	ple_read_pool_destroy(&pool);
	assert(!close(fd));
	puts("ple_read_pool: readers 1..16, 2048 reused batches, byte/guard parity, partial/EINTR/EOF/EIO/EBADF, and 16 startup-failure points passed");
	return 0;
}
