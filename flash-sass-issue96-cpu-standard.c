/* Diagnostic only: match standard attention on the CPU reference. */
#define _GNU_SOURCE
#include "mtmd.h"
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

mtmd_context *mtmd_init_from_file(const char *path, const struct llama_model *model,
                                const struct mtmd_context_params initial) {
    struct mtmd_context_params params = initial;
    if (!params.use_gpu) {
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
        fprintf(stderr, "issue96-test: standard-attention CPU reference\n");
    }
    mtmd_context *(*real_init)(const char *, const struct llama_model *, const struct mtmd_context_params) =
        dlsym(RTLD_NEXT, "mtmd_init_from_file");
    if (!real_init) abort();
    return real_init(path, model, params);
}
