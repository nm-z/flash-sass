/* Test-only CPU oracle. This is a separate executable, not a service mode. */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "../../kern/vision.h"
#include <math.h>
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Diagnostic CPU promotion experiment. It did not remove the observed
 * whole-model CPU/GPU difference and is not the GPU FA acceptance oracle.
 * gpu_reference.c supplies the matched GPU numerical reference. */
struct ggml_tensor *ggml_flash_attn_ext(struct ggml_context *ctx,
        struct ggml_tensor *q, struct ggml_tensor *k, struct ggml_tensor *v,
        struct ggml_tensor *mask, float scale, float bias, float softcap) {
    typedef struct ggml_tensor *(*builder)(struct ggml_context *, struct ggml_tensor *,
        struct ggml_tensor *, struct ggml_tensor *, struct ggml_tensor *, float, float, float);
    builder real = (builder)dlsym(RTLD_NEXT, "ggml_flash_attn_ext");
    if (!real) abort();
    return real(ctx, q, ggml_cast(ctx, k, GGML_TYPE_F32), ggml_cast(ctx, v, GGML_TYPE_F32), mask, scale, bias, softcap);
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr, "usage: reference PROJECTOR TEXT IMAGE OUTPUT BACKEND_DIR\n"); return 2; }
    ggml_backend_load_all_from_path(argv[5]);
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.vocab_only = true;
    struct llama_model *model = llama_model_load_from_file(argv[2], mp);
    if (!model) return 1;
    struct mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = false; cp.device = NULL; cp.n_threads = 10;
    cp.warmup = false; cp.print_timings = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.image_min_tokens = FLASH_IMAGE_MIN_TOKENS;
    cp.image_max_tokens = FLASH_IMAGE_MAX_TOKENS;
    mtmd_context *ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) return 1;
    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) return 1;
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text text = {.text = mtmd_default_marker(), .text_len = 0};
    text.text_len = __builtin_strlen(text.text);
    const mtmd_bitmap *images[] = {bw.bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, images, 1)) return 1;
    const mtmd_input_chunk *img = NULL;
    for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk *part = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(part) == MTMD_INPUT_CHUNK_TYPE_IMAGE) img = part;
    }
    if (!img || mtmd_encode_chunk(ctx, img)) return 1;
    size_t tokens = mtmd_input_chunk_get_n_tokens(img), count = tokens * 2560;
    float *emb = mtmd_get_output_embd(ctx);
    for (size_t i = 0; i < count; i++) if (!isfinite(emb[i])) return 1;
    FILE *out = fopen(argv[4], "wb");
    if (!out || fwrite(emb, sizeof(float), count, out) != count || fclose(out)) return 1;
    printf("{\"event\":\"cpu_reference\",\"tokens\":%zu,\"dimension\":2560,\"attention\":\"F32 accumulation, K/V rounded through F16\"}\n", tokens);
    mtmd_input_chunks_free(chunks); mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx); llama_model_free(model); llama_backend_free();
    return 0;
}
