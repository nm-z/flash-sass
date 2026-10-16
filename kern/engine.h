/* flash.h: the engine flash.c exposes to its serving front-end (serve.c). */
#ifndef FLASH_ENGINE_H
#define FLASH_ENGINE_H
#include <stdint.h>

/* temp <= 0 or top_k <= 1: greedy. Otherwise top-k (<= 32) -> top-p -> temperature, the draw for position t from seed (kern/sample.sass) */
typedef struct { uint64_t seed; float temp, topp; int top_k; } sampling;
typedef struct { int ntok; int ended; unsigned nph; double build_ms, run_ms, first_ms; } run_stats;
/* Run one response: positions 0 .. nprompt + n_new - 1, greedy. on_token(arg, t, tok) gets every
 * generated token as soon as die 0 has it (t = its position, from nprompt - 1), the stop token
 * included; stop[2] are the end-of-turn ids (0xffffffff: none). Returns -1 on bad sizes. */
int engine_run(const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int (*on_token)(void *, int, uint32_t), void *arg, run_stats *st);
int engine_positions(void);          /* FLASH_MAXT: prompt + reply positions per response */

#endif
