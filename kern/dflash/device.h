#ifndef FLASH_DFLASH_DEVICE_H
#define FLASH_DFLASH_DEVICE_H
#include <stdint.h>

typedef struct {
    uint32_t context,anchor,append_start,append_count;
} dflash_device_control;
typedef struct { float value;uint32_t token; } dflash_candidate;
typedef struct {
    uint64_t data;
    uint32_t first_group,groups,stride,raw_q3;
} dflash_embedding_shard;

/* Raw GGUF Q4_K/Q6_K for the draft; the target head remains in its existing
 * sub-block-major Q5_K device layout. No expanded matrix copy is required. */
#define DFLASH_Q4_K 12u
#define DFLASH_Q6_K 14u
#define DFLASH_TARGET_Q5_TILE 1013u
#define DFLASH_DEVICE_BLOCK 4u
#define DFLASH_DEVICE_WINDOW 2048u
#define DFLASH_ATTENTION_PARTS 4u
#endif
