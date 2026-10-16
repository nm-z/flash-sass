/* Client side of the weight holder (kern/hold.c): map the arena a running `hold`
 * exported and look tensors up by name. */
#ifndef FLASH_HOLD_H
#define FLASH_HOLD_H
#include <cuda.h>
#include <stddef.h>
#include <stdint.h>

typedef struct {
	char name[96];
	size_t off, bytes;
	uint32_t type;
	int rows, K, stride, experts;
} hold_tensor;

typedef struct {
	CUdeviceptr base;
	size_t bytes;
	int die, n;
	hold_tensor *t;
} hold_arena;

/* 0 on success; the current context must be on the holder's die */
int hold_open(hold_arena *a, const char *manifest);
const hold_tensor *hold_find(const hold_arena *a, const char *name);   /* NULL if absent */
CUdeviceptr hold_ptr(const hold_arena *a, const char *name);           /* 0 if absent */
void hold_close(hold_arena *a);
#endif
