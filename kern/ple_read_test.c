/* Host-only regression of the same reader used by engine.c. */
#define _GNU_SOURCE
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

static ssize_t test_pread(int, void *, size_t, off_t);
static int test_create(pthread_t *, const pthread_attr_t *, void *(*)(void *), void *);
static int test_join(pthread_t, void **);
#define PLE_READ_PREAD test_pread
#define pthread_create test_create
#define pthread_join test_join
#include "ple_read_pool.h"
#undef pthread_create
#undef pthread_join
#include "ple_read_reload.h"

enum { NORMAL, PARTIAL, INTERRUPTED, END_OF_FILE, IO_ERROR };
static int fault, injected, fail_create = -1, create_calls, join_calls;
static off_t fault_at;

/* A timed barrier makes a serial replacement fail instead of passing byte
 * comparisons or hanging. All gate state is protected by gate_mutex. */
#define GATE_ALL ((1u << PLE_READ_HEADS) - 1u)
static pthread_mutex_t gate_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t gate_cond = PTHREAD_COND_INITIALIZER;
static int gate_enabled, gate_returned, gate_premature;
static unsigned gate_entered, gate_finished, gate_release;
static off_t gate_offsets[PLE_READ_HEADS];
static pthread_t gate_threads[PLE_READ_HEADS];
static struct timespec gate_deadline;
static struct timespec future_ms(long milliseconds) {
	struct timespec t;
	assert(!clock_gettime(CLOCK_REALTIME, &t));
	t.tv_sec += milliseconds / 1000;
	t.tv_nsec += milliseconds % 1000 * 1000000L;
	if (t.tv_nsec >= 1000000000L) { t.tv_nsec -= 1000000000L; t.tv_sec++; }
	return t;
}
static void gate_wait(void) {
	int rc = pthread_cond_timedwait(&gate_cond, &gate_mutex, &gate_deadline);
	assert(rc == 0);
}

static ssize_t test_pread(int fd, void *buf, size_t count, off_t offset) {
	int gated_head = -1;
	assert(!pthread_mutex_lock(&gate_mutex));
	if (gate_enabled) {
		for (unsigned h = 0; h < PLE_READ_HEADS; h++)
			if (offset == gate_offsets[h]) { gated_head = (int)h; break; }
		assert(gated_head >= 0 && count == PLE_READ_ROW_BYTES);
		unsigned bit = 1u << (unsigned)gated_head;
		assert(!(gate_entered & bit));
		gate_threads[gated_head] = pthread_self();
		gate_entered |= bit;
		assert(!pthread_cond_broadcast(&gate_cond));
		while (!(gate_release & bit)) gate_wait();
	}
	assert(!pthread_mutex_unlock(&gate_mutex));
	if (fault && offset >= fault_at && offset < fault_at + 90) {
		size_t got = (size_t)(offset - fault_at);
		if (got < 17) { if (count > 17 - got) count = 17 - got; }
		else if (fault == PARTIAL) { if (count > 7) count = 7; }
		else if (fault == INTERRUPTED && !__atomic_exchange_n(&injected, 1, __ATOMIC_RELAXED)) { errno = EINTR; return -1; }
		else if (fault == END_OF_FILE) return 0;
		else if (fault == IO_ERROR) { errno = EIO; return -1; }
	}
	ssize_t got = pread(fd, buf, count, offset);
	if (gated_head >= 0) {
		assert(!pthread_mutex_lock(&gate_mutex));
		gate_finished |= 1u << (unsigned)gated_head;
		assert(!pthread_cond_broadcast(&gate_cond));
		assert(!pthread_mutex_unlock(&gate_mutex));
	}
	return got;
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
struct batch_probe {
	struct ple_read_pool *pool;
	int fd;
	off_t offsets[PLE_READ_HEADS];
	unsigned char guarded[PLE_READ_HEADS * PLE_READ_ROW_BYTES + 32];
	struct ple_read_result results[PLE_READ_HEADS];
};
static void *probe_batch(void *opaque) {
	struct batch_probe *probe = opaque;
	ple_read_batch(probe->pool, probe->fd, probe->offsets,
	               probe->guarded + 16, probe->results);
	assert(!pthread_mutex_lock(&gate_mutex));
	gate_premature = gate_finished != GATE_ALL;
	gate_returned = 1;
	assert(!pthread_cond_broadcast(&gate_cond));
	assert(!pthread_mutex_unlock(&gate_mutex));
	return NULL;
}
static void check_concurrency(int fd) {
	struct ple_read_pool pool;
	pthread_t previous[PLE_READ_HEADS];
	assert(!ple_read_pool_init(&pool, PLE_READ_HEADS));
	int starts = create_calls;
	for (unsigned generation = 0; generation < 2; generation++) {
		struct batch_probe probe = {.pool = &pool, .fd = fd};
		memset(probe.guarded, 0xa5, sizeof probe.guarded);
		for (unsigned h = 0; h < PLE_READ_HEADS; h++)
			probe.offsets[h] = (off_t)(500 + generation * 2000 + h * 6000);
		assert(!pthread_mutex_lock(&gate_mutex));
		gate_enabled = 1;
		gate_returned = gate_premature = 0;
		gate_entered = gate_finished = gate_release = 0;
		memcpy(gate_offsets, probe.offsets, sizeof gate_offsets);
		gate_deadline = future_ms(3000);
		assert(!pthread_mutex_unlock(&gate_mutex));
		pthread_t producer;
		assert(!pthread_create(&producer, NULL, probe_batch, &probe));
		assert(!pthread_mutex_lock(&gate_mutex));
		while (gate_entered != GATE_ALL) gate_wait();
		for (unsigned h = 0; h < PLE_READ_HEADS; h++) {
			assert(!pthread_equal(gate_threads[h], producer));
			for (unsigned k = 0; k < h; k++)
				assert(!pthread_equal(gate_threads[h], gate_threads[k]));
			if (generation) assert(pthread_equal(gate_threads[h], previous[h]));
			previous[h] = gate_threads[h];
		}
		assert(!gate_returned);
		unsigned delayed = generation ? 13 : 7;
		gate_release = GATE_ALL ^ (1u << delayed);
		assert(!pthread_cond_broadcast(&gate_cond));
		while (gate_finished != gate_release) {
			assert(!gate_returned);
			gate_wait();
		}
		/* Fifteen completed rows must not release the producer. */
		struct timespec quiet = future_ms(25);
		while (!gate_returned) {
			int rc = pthread_cond_timedwait(&gate_cond, &gate_mutex, &quiet);
			if (rc == ETIMEDOUT) break;
			assert(rc == 0);
		}
		assert(!gate_returned && !gate_premature);
		gate_release = GATE_ALL;
		assert(!pthread_cond_broadcast(&gate_cond));
		while (!gate_returned) gate_wait();
		assert(gate_finished == GATE_ALL && !gate_premature);
		gate_enabled = 0;
		assert(!pthread_mutex_unlock(&gate_mutex));
		assert(!pthread_join(producer, NULL));
		assert(create_calls == starts);
		for (unsigned h = 0; h < PLE_READ_HEADS; h++) {
			assert(probe.results[h].got == PLE_READ_ROW_BYTES);
			assert(!probe.results[h].error);
			assert(!memcmp(probe.guarded + 16 + h * PLE_READ_ROW_BYTES,
			               fixture + probe.offsets[h], PLE_READ_ROW_BYTES));
		}
		for (unsigned i = 0; i < 16; i++)
			assert(probe.guarded[i] == 0xa5 &&
			       probe.guarded[16 + PLE_READ_HEADS * PLE_READ_ROW_BYTES + i] == 0xa5);
	}
	ple_read_pool_destroy(&pool);
}
static void check_reload_preflight(void) {
	const char *valid[] = {
		"", " \t\n", "libengine.so FLASH_OTHER=17",
		"FLASH_PLE_READERS=1", "FLASH_PLE_READERS=16",
		"FLASH_PLE_READERS=00016", "FLASH_PLE_READERS=2 FLASH_PLE_READERS=16",
		"XFLASH_PLE_READERS=17 FLASH_PLE_READERS_EXTRA=17"
	};
	const char *invalid[] = {
		"FLASH_PLE_READERS=", "FLASH_PLE_READERS=0", "FLASH_PLE_READERS=17",
		"FLASH_PLE_READERS=-1", "FLASH_PLE_READERS=+1", "FLASH_PLE_READERS=x",
		"FLASH_PLE_READERS=1x", "FLASH_PLE_READERS=1\r",
		"FLASH_PLE_READERS=999999999999999999",
		"FLASH_OTHER=9 FLASH_PLE_READERS=2 FLASH_PLE_READERS=17",
		"FLASH_PLE_READERS=17 FLASH_PLE_READERS=2"
	};
	const char *current = getenv("FLASH_PLE_READERS");
	char *saved = current ? strdup(current) : NULL;
	assert(!current || saved);
	assert(!setenv("FLASH_PLE_READERS", "6", 1));
	for (unsigned group = 0; group < 2; group++) {
		const char **cases = group ? invalid : valid;
		size_t count = group ? sizeof invalid / sizeof *invalid : sizeof valid / sizeof *valid;
		for (size_t i = 0; i < count; i++) {
			char *command = strdup(cases[i]);
			assert(command);
			assert(ple_read_reload_preflight(command) == (group ? EINVAL : 0));
			assert(!strcmp(command, cases[i]));
			assert(!strcmp(getenv("FLASH_PLE_READERS"), "6"));
			free(command);
		}
	}
	if (saved) { assert(!setenv("FLASH_PLE_READERS", saved, 1)); free(saved); }
	else { assert(!unsetenv("FLASH_PLE_READERS")); }
}

static void check_reload_readiness(void) {
	enum { CORE_POSITIONS = 33, SMALL_ENGINE_POSITIONS = 7 };
	struct {
		unsigned char rows[CORE_POSITIONS * 16 * PLE_READ_ROW_BYTES];
		uint32_t ready[CORE_POSITIONS + 2];
		unsigned char guard[32];
	} table;
	memset(&table, 0xa5, sizeof table);
	uint32_t *ready = table.ready;
	for (unsigned reload = 0; reload < 3; reload++) {
		/* Old requests leave IDs 1 and 2 ready across the entire core. A new
		 * library would accept these IDs before restaging its own row data. */
		for (unsigned q = 0; q < CORE_POSITIONS + 2; q++) ready[q] = 1 + (q & 1);
		assert(ready[0] == 1 && ready[CORE_POSITIONS] == 2);
		ready = ple_read_reset_ready(&table, offsetof(__typeof__(table), ready), CORE_POSITIONS);
		assert(ready == table.ready);
		for (unsigned q = 0; q < CORE_POSITIONS + 2; q++) assert(ready[q] == 0);
		assert(ready[SMALL_ENGINE_POSITIONS + 1] == 0);
		for (size_t i = 0; i < sizeof table.rows; i++) assert(table.rows[i] == 0xa5);
		for (size_t i = 0; i < sizeof table.guard; i++) assert(table.guard[i] == 0xa5);
		/* Only the position that has been restaged becomes visible for the
		 * restarted request ID; subsequent positions must still wait. */
		table.rows[0] = (unsigned char)reload;
		__atomic_store_n(&ready[0], 1, __ATOMIC_RELEASE);
		assert(__atomic_load_n(&ready[0], __ATOMIC_ACQUIRE) == 1);
		assert(table.rows[0] == reload && ready[1] == 0);
		table.rows[0] = 0xa5;
	}
}

int main(void) {
	check_reload_preflight();
	check_reload_readiness();
	unsigned readers;
	assert(!ple_read_parse_readers(NULL, &readers) && readers == 1);
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
	check_concurrency(fd);
	assert(!close(fd));
	puts("ple_read_pool: readers 1..16, 2048 reused batches, byte/guard parity, partial/EINTR/EOF/EIO/EBADF, 16 startup-failure points, reload preflight/readiness, and 16 concurrent persistent workers with delayed completion passed");
	return 0;
}
