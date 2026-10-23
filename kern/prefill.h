#ifndef FLASH_PREFILL_H
#define FLASH_PREFILL_H

/* Known positions per host chunk. B8 replay is separately opt-in; the legacy
 * device programs remain B1/B2 when FLASH_PREFILL_BATCH is absent or zero. */
#define FLASH_PREFILL_CHUNK_MAX 4096
typedef struct {
    int begin;
    int end;      /* exclusive; excludes the final prompt position */
    int pair_end; /* [begin,pair_end) uses B2; an odd tail uses B1 */
} prefill_chunk_t;

static inline int prefill_parse_batch(const char *text, int *size) {
    if (!size) return -1;
    if (!text || (text[0] == '0' && !text[1])) { *size = 0; return 0; }
    if (text[0] == '8' && !text[1]) { *size = 8; return 0; }
    return -1;
}

/* Largest whole batch before an exclusive known-prefix or snapshot boundary.
 * Difference-before-addition also handles boundaries near INT_MAX. */
static inline int prefill_batch_end(int position, int end, int width) {
    if (position < 0 || end < position || (width != 1 && width != 2 && width != 8)) return -1;
    return position + ((end - position) / width) * width;
}

/* Absent/zero disables. Malformed settings must not silently select a path. */
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
