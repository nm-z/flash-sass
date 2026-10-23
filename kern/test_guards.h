/* Test-only canaries around CUDA and mapped-host allocations.
 * Guards detect writes across allocation boundaries; the documented prefetch
 * derivation, not these canaries, establishes the readable load extent. */
#ifndef FLASH_TEST_GUARDS_H
#define FLASH_TEST_GUARDS_H
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TEST_GUARD_BYTES 256
#define TEST_GUARD_VALUE 0xa5
typedef struct {
	CUdeviceptr base, data;
	void *host;
	size_t bytes;
	const char *name;
} test_guard_allocation;
static test_guard_allocation test_guard_allocations[256];
static unsigned test_guard_count;

static CUdeviceptr test_guard_alloc(const char *name, size_t bytes, int host) {
	if (!bytes) bytes = 4;
	if (test_guard_count == 256 || bytes > SIZE_MAX - 2 * TEST_GUARD_BYTES) {
		fprintf(stderr, "guard allocation limit: %s\n", name); exit(2);
	}
	test_guard_allocation *a = &test_guard_allocations[test_guard_count++];
	a->bytes = bytes; a->name = name;
	if (host) {
		CK(cuMemHostAlloc(&a->host, bytes + 2 * TEST_GUARD_BYTES,
			CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
		memset(a->host, TEST_GUARD_VALUE, bytes + 2 * TEST_GUARD_BYTES);
		CK(cuMemHostGetDevicePointer(&a->base, a->host, 0));
	} else {
		CK(cuMemAlloc(&a->base, bytes + 2 * TEST_GUARD_BYTES));
		CK(cuMemsetD8(a->base, TEST_GUARD_VALUE, bytes + 2 * TEST_GUARD_BYTES));
	}
	a->data = a->base + TEST_GUARD_BYTES;
	CK(cuMemsetD8(a->data, 0, bytes));
	return a->data;
}

static int test_guard_check(void) {
	unsigned char expected[TEST_GUARD_BYTES], got[TEST_GUARD_BYTES];
	memset(expected, TEST_GUARD_VALUE, sizeof expected);
	int bad = 0;
	CK(cuCtxSynchronize());
	if (getenv("TEST_GUARD_INJECT") && test_guard_count)
		CK(cuMemsetD8(test_guard_allocations[0].base, 0, 1));
	for (unsigned i = 0; i < test_guard_count; i++) {
		test_guard_allocation *a = &test_guard_allocations[i];
		for (unsigned side = 0; side < 2; side++) {
			CUdeviceptr p = side ? a->data + a->bytes : a->base;
			CK(cuMemcpyDtoH(got, p, sizeof got));
			if (memcmp(expected, got, sizeof got)) {
				fprintf(stderr, "guard mismatch: allocation=%u name=%s side=%s placement=%s bytes=%zu\n",
					i, a->name, side ? "suffix" : "prefix",
					a->host ? "host" : "device", a->bytes);
				bad++;
			}
		}
	}
	printf("guards: %s (%u allocations, %u bytes per boundary)\n",
		bad ? "FAIL" : "PASS", test_guard_count, TEST_GUARD_BYTES);
	return bad;
}

static void test_guard_free(void) {
	for (unsigned i = test_guard_count; i > 0; i--) {
		test_guard_allocation *a = &test_guard_allocations[i - 1];
		if (a->host) CK(cuMemFreeHost(a->host));
		else CK(cuMemFree(a->base));
	}
	test_guard_count = 0;
}
#endif
