/* engine.h: the interface of the engine inside libengine.so (engine.c) */
#ifndef FLASH_ENGINE_H
#define FLASH_ENGINE_H
#include <stdint.h>

/* temp <= 0 or top_k <= 1: greedy. Otherwise top-k (<= 32) -> top-p -> temperature, the draw for position t from seed (kern/sample.sass) */
typedef struct { uint64_t seed; float temp, topp; int top_k; } sampling;
typedef struct { int slot; int ntok; int ended; unsigned nph; int resumed, snapshot; double build_ms, run_ms, first_ms; int paired_positions, prefill_chunks; } run_stats;
/* all_tokens: optional engine_positions()-sized caller buffer, copied on completion.
 * token: every generated token as soon as die 0 has it (t = its position, from nprompt - 1; nonzero return: the
 * client is gone); alive: polled every ~20 ms, also while the prompt runs (nonzero: the client is gone) */
typedef struct { int (*token)(void *arg, int t, uint32_t tok); int (*alive)(void *arg); void *arg; uint32_t *all_tokens; } engine_cb;
/* An image in the prompt: the tokens [s, s + ntok) (placeholders in ids) take their embeddings from `path`, flashvis' answer
 * written to a file: u32 ntok, nx, ny, npos, dim (2560), then f32 [ntok][dim], then u32 [ntok][3] (t, y, x of each token relative to the
 * first). The positions of the tokens after it continue npos after its first one, and the tokens of the image get M-RoPE positions
 * (t, y, x) in the attention layers (the interleaved sections of rope.dimension_sections). */
typedef struct { int s; char path[240]; } engine_img;
/* Run one response: positions 0 .. nprompt + n_new - 1. The generated tokens (the stop token included) go to
 * cb->token; stop[2] are the end-of-turn ids (0xffffffff: none). snap_at > 0: take a prefix-cache snapshot after that
 * many positions (the response then runs as two launches; snap_at <= nprompt - 1), and a later response whose prompt
 * starts with the same tokens resumes from it. Returns -1 on bad sizes, -2 after a CUDA error (the contexts need
 * rebuilding), -3 on bad image data, and -4 if host program allocation fails.
 * Images: snapshots stop before the first one. */
int engine_run(const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int snap_at, const engine_img *imgs, int nimg, const engine_cb *cb, run_stats *st);
/* wanted=-1: free slot; otherwise wait for the numbered slot. Invalid slot: -1;
 * cancelled while waiting: -5; host program allocation failure: -4. */
int engine_run_slot(int wanted, const uint32_t *ids, int nprompt, int n_new, const uint32_t stop[2], const sampling *sm, int snap_at, const engine_img *imgs, int nimg, const engine_cb *cb, run_stats *st);
int engine_positions(void);          /* FLASH_MAXT: prompt + reply positions per response */

#endif
