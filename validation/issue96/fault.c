/* Test-only libmtmd interposer. Never load in the service. */
#define _GNU_SOURCE
#include "mtmd.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* A standard-attention CPU reference separates backend arithmetic from the
 * F16 K/V conversion used by the production CPU Flash Attention path. */
mtmd_context *mtmd_init_from_file(const char *path, const struct llama_model *model,
                                const struct mtmd_context_params initial) {
    struct mtmd_context_params params = initial;
    const char *standard = getenv("FLASHVIS_TEST_CPU_STANDARD");
    if (!params.use_gpu && standard && !strcmp(standard, "1")) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        fprintf(stderr, "issue96-test: standard-attention CPU reference\n");
    }
    mtmd_context *(*real_init)(const char *, const struct llama_model *, const struct mtmd_context_params) =
        dlsym(RTLD_NEXT, "mtmd_init_from_file");
    if (!real_init) abort();
    return real_init(path, model, params);
}

int32_t mtmd_encode_chunk(mtmd_context *ctx, const mtmd_input_chunk *chunk) {
    static int calls;
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (calls++ == 0 && mode && !strcmp(mode, "error")) {
        fprintf(stderr, "issue96-test: injecting one reported GPU encoding failure\n");
        return 1;
    }
    int32_t (*real_encode)(mtmd_context *, const mtmd_input_chunk *) = dlsym(RTLD_NEXT, "mtmd_encode_chunk");
    if (!real_encode) abort();
    return real_encode(ctx, chunk);
}

float *mtmd_get_output_embd(mtmd_context *ctx) {
    static int calls;
    float *(*real_output)(mtmd_context *) = dlsym(RTLD_NEXT, "mtmd_get_output_embd");
    if (!real_output) abort();
    float *output = real_output(ctx);
    const char *mode = getenv("FLASHVIS_TEST_FAULT");
    if (calls++ == 0 && output && mode && !strcmp(mode, "nan")) {
        fprintf(stderr, "issue96-test: injecting one non-finite GPU embedding\n");
        output[0] = NAN;
    }
    return output;
}
