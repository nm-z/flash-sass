/* engine.h: the interface of the engine inside libengine.so (engine.c) */
#ifndef FLASH_ENGINE_H
#define FLASH_ENGINE_H
#include <stdint.h>

/* temp <= 0 or top_k <= 1: greedy. Otherwise top-k (<= 32) -> top-p -> temperature, the draw for position t from seed (kern/sample.sass) */
typedef struct { uint64_t seed; float temp, topp; int top_k; } sampling;
typedef struct { int ntok; int ended; unsigned nph; int resumed, snapshot; double build_ms, run_ms, first_ms; } run_stats;
/* token: every generated token as soon as die 0 has it (t = its position, from nprompt - 1; nonzero return: the
 * client is gone); alive: polled every ~20 ms, also while the prompt runs (nonzero: the client is gone) */
typedef struct { int (*token)(void *arg, int t, uint32_t tok); int (*alive)(void *arg); void *arg; } engine_cb;
/* Run one response: positions 0 .. nprompt + n_new - 1. The generated tokens (the stop token included) go to
 * cb->token; stop[2] are the end-of-turn ids (0xffffffff: none). snap_at > 0: take a prefix-cache snapshot after that
 * many positions (the response then runs as two launches; snap_at <= nprompt - 1), and a later response whose prompt
 * starts with the same tokens resumes from it. Returns -1 on bad sizes, -2 after a CUDA error (the contexts need
 * rebuilding). */
int engine_run(const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int snap_at, const engine_cb *cb, run_stats *st);
int engine_positions(void);          /* FLASH_MAXT: prompt + reply positions per response */

#endif
