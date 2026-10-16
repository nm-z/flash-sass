/* PLE (n-gram) pieces exactly as the kernels compute them. */
#ifndef FLASH_PLE_H
#define FLASH_PLE_H
#include <stddef.h>
#include <stdint.h>

/* Row hash of per_layer_token_embd (GGUF qwen4exp.ple.*, Flash-Next values):
 *   ngram 3, 8 heads per order -> 16 rows per token.
 *   context[back] = ids[pos - back] if back <= pos and no id in ids[pos-back, pos)
 *                   is an end id, else `absent`
 *   order n = 2, 3: mixed_n = XOR_{i < n} context[i] * mult[i]   (wrapping u64)
 *   head h in [(n-2) 8, (n-1) 8): row_h = offsets[h] + mixed_n % vocab[h]
 * (cx-flash RowHash::rows_at.) */
#define PLE_NGRAM 3
#define PLE_HEADS 16
#define PLE_ROW_VALUES 160
extern const uint64_t ple_multipliers[PLE_NGRAM];
extern const uint64_t ple_head_offsets[PLE_HEADS];
extern const uint64_t ple_head_vocab[PLE_HEADS];
void ple_rows(const uint32_t *ids, size_t pos, uint32_t absent, const uint32_t *ends, int nends, uint64_t rows[PLE_HEADS]);
#endif
