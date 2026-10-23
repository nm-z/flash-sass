#ifndef FLASH_VISION_H
#define FLASH_VISION_H

/* Qwen3.8-Flash-Next: patch 16, spatial merge 2, 65,536..16,777,216
 * pixels. One output token represents a 32 x 32 pixel region. These limits
 * match kern/vision/preprocessor_config.json, rather than libmtmd's fallback. */
#define FLASH_IMAGE_MIN_TOKENS 64
#define FLASH_IMAGE_MAX_TOKENS 16384

/* Image bytes alone do not identify embeddings made under the old 4096-token
 * preprocessing limit. Keep those entries out of the current cache namespace. */
#define FLASH_IMAGE_CACHE_PREFIX "16mp-"

#endif
