#ifndef FLASH_PLE_READ_POOL_H
#define FLASH_PLE_READ_POOL_H
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#define PLE_READ_HEADS 16
#define PLE_READ_ROW_BYTES 90
#ifndef PLE_READ_PREAD
#define PLE_READ_PREAD pread
#endif

struct ple_read_result {
	size_t got;
	int error;                  /* errno on failure; 0 on success or EOF */
};
struct ple_read_pool;
struct ple_read_worker {
	struct ple_read_pool *pool;
	unsigned head;
};
struct ple_read_pool {
	pthread_mutex_t mutex;
	pthread_cond_t work, done;
	pthread_t threads[PLE_READ_HEADS];
	struct ple_read_worker workers[PLE_READ_HEADS];
	unsigned readers, started, remaining;
	int sync_ready, stopping;
	uint64_t generation;
	/* Published under mutex; immutable until the batch completes. */
	int fd;
	const off_t *offsets;
	uint8_t *dst;
	struct ple_read_result *results;
};

/* Absent selects 16; empty, malformed, or out-of-range settings fail. */
static int ple_read_parse_readers(const char *s, unsigned *readers) {
	unsigned n = 0;
	if (!s) { *readers = PLE_READ_HEADS; return 0; }
	if (!*s) return EINVAL;
	for (; *s; s++) {
		if (*s < '0' || *s > '9') return EINVAL;
		n = n * 10 + (unsigned)(*s - '0');
		if (n > PLE_READ_HEADS) return EINVAL;
	}
	if (!n) return EINVAL;
	*readers = n;
	return 0;
}

/* Complete positive short reads and retry EINTR; leave failed tails untouched. */
static void ple_read_one(int fd, off_t offset, uint8_t *dst, struct ple_read_result *result) {
	size_t got = 0;
	int error = 0;
	while (got < PLE_READ_ROW_BYTES) {
		ssize_t n = PLE_READ_PREAD(fd, dst + got, PLE_READ_ROW_BYTES - got, offset + (off_t)got);
		if (n > 0) { got += (size_t)n; continue; }
		if (n < 0 && errno == EINTR) continue;
		if (n < 0) error = errno;
		break;
	}
	result->got = got;
	result->error = error;
}

static void *ple_read_worker_main(void *opaque) {
	struct ple_read_worker *worker = opaque;
	struct ple_read_pool *pool = worker->pool;
	uint64_t seen = 0;
	pthread_mutex_lock(&pool->mutex);
	for (;;) {
		while (!pool->stopping && pool->generation == seen)
			pthread_cond_wait(&pool->work, &pool->mutex);
		if (pool->stopping) break;
		seen = pool->generation;
		int fd = pool->fd;
		unsigned readers = pool->readers;
		const off_t *offsets = pool->offsets;
		uint8_t *dst = pool->dst;
		struct ple_read_result *results = pool->results;
		pthread_mutex_unlock(&pool->mutex);
		for (unsigned h = worker->head; h < PLE_READ_HEADS; h += readers)
			ple_read_one(fd, offsets[h], dst + h * PLE_READ_ROW_BYTES, &results[h]);
		pthread_mutex_lock(&pool->mutex);
		if (--pool->remaining == 0) pthread_cond_signal(&pool->done);
	}
	pthread_mutex_unlock(&pool->mutex);
	return NULL;
}

/* The sole batch producer must have stopped. Safe on zeroed/destroyed state. */
static void ple_read_pool_destroy(struct ple_read_pool *pool) {
	if (pool->sync_ready) {
		pthread_mutex_lock(&pool->mutex);
		pool->stopping = 1;
		pthread_cond_broadcast(&pool->work);
		pthread_mutex_unlock(&pool->mutex);
		for (unsigned i = 0; i < pool->started; i++) pthread_join(pool->threads[i], NULL);
		pthread_cond_destroy(&pool->done);
		pthread_cond_destroy(&pool->work);
		pthread_mutex_destroy(&pool->mutex);
	}
	memset(pool, 0, sizeof *pool);
}

/* Initialize fresh/destroyed state. The fd and buffers remain caller-owned. */
static int ple_read_pool_init(struct ple_read_pool *pool, unsigned readers) {
	int error;
	memset(pool, 0, sizeof *pool);
	if (readers < 1 || readers > PLE_READ_HEADS) return EINVAL;
	pool->readers = readers;
	if (readers == 1) return 0;
	error = pthread_mutex_init(&pool->mutex, NULL);
	if (error) return error;
	error = pthread_cond_init(&pool->work, NULL);
	if (error) { pthread_mutex_destroy(&pool->mutex); return error; }
	error = pthread_cond_init(&pool->done, NULL);
	if (error) {
		pthread_cond_destroy(&pool->work); pthread_mutex_destroy(&pool->mutex);
		return error;
	}
	pool->sync_ready = 1;
	for (unsigned i = 0; i < readers; i++) {
		pool->workers[i].pool = pool;
		pool->workers[i].head = i;
		error = pthread_create(&pool->threads[i], NULL, ple_read_worker_main, &pool->workers[i]);
		if (error) { ple_read_pool_destroy(pool); return error; }
		pool->started++;
	}
	return 0;
}

/* One producer, synchronous completion; all row writes precede the return. */
static void ple_read_batch(struct ple_read_pool *pool, int fd,
                          const off_t offsets[PLE_READ_HEADS], void *dst,
                          struct ple_read_result results[PLE_READ_HEADS]) {
	if (pool->readers == 1) {
		for (unsigned h = 0; h < PLE_READ_HEADS; h++)
			ple_read_one(fd, offsets[h], (uint8_t *)dst + h * PLE_READ_ROW_BYTES, &results[h]);
		return;
	}
	pthread_mutex_lock(&pool->mutex);
	pool->fd = fd; pool->offsets = offsets; pool->dst = dst; pool->results = results;
	pool->remaining = pool->started;
	pool->generation++;
	pthread_cond_broadcast(&pool->work);
	while (pool->remaining) pthread_cond_wait(&pool->done, &pool->mutex);
	pool->offsets = NULL; pool->dst = NULL; pool->results = NULL;
	pthread_mutex_unlock(&pool->mutex);
}
#endif
