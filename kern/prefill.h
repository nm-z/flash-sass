#ifndef FLASH_PREFILL_H
#define FLASH_PREFILL_H

/* Known positions per host chunk; device programs remain B1 or B2. */
#define FLASH_PREFILL_CHUNK_MAX 4096
#define FLASH_PREFILL_AUTO_THRESHOLD FLASH_PREFILL_CHUNK_MAX
typedef struct {
    int begin;
    int end;      /* exclusive; excludes the final prompt position */
    int pair_end; /* [begin,pair_end) uses B2; an odd tail uses B1 */
} prefill_chunk_t;

/* Absence and zero parse as size 0; the caller selects the default policy. */
static inline int prefill_parse_chunk(const char *text, int *size) {
    if (!size) return -1;
    if (!text) { *size = 0; return 0; }
    if (!*text) return -1;
    int value = 0;
    for (const char *p = text; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        int digit = *p - '0';
        if (value > (FLASH_PREFILL_CHUNK_MAX - digit) / 10) return -1;
        value = value * 10 + digit;
    }
    *size = value;
    return 0;
}

/* Keep an unchunked kernel bounded for long uncached prompts. An explicit
 * chunk size always wins; an explicit zero disables the automatic policy. */
static inline int prefill_effective_chunk(int configured, int automatic,
                                          int pending_positions) {
    if (configured > 0) return configured;
    return automatic && pending_positions > FLASH_PREFILL_AUTO_THRESHOLD
        ? FLASH_PREFILL_CHUNK_MAX : 0;
}

/* Plan [position,min(until,nprompt-1)), stopping exactly at snapshots.
 * Difference-before-addition avoids overflow near INT_MAX.
 * Return 1 for work, 0 at the boundary, -1 for invalid inputs.
 * Cancellation is checked by the caller before submitting the next chunk. */
static inline int prefill_plan_chunk(int position, int nprompt, int until,
                                     int chunk_size, int paired,
                                     prefill_chunk_t *chunk) {
    if (!chunk || position < 0 || nprompt < 1 || position > nprompt ||
        until < position || until > nprompt || chunk_size < 1 ||
        chunk_size > FLASH_PREFILL_CHUNK_MAX) return -1;
    int limit = until < nprompt - 1 ? until : nprompt - 1;
    chunk->begin = chunk->end = chunk->pair_end = position;
    if (position >= limit) return 0;
    int count = limit - position;
    if (count > chunk_size) count = chunk_size;
    chunk->end = position + count;
    chunk->pair_end = paired ? position + (count / 2) * 2 : position;
    return 1;
}
#endif
