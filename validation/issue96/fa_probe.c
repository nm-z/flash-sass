/* Diagnostic interposer: force GPU FA and inspect its actual graph tensors.
 * Not linked into the service. Run under gpuq with LD_PRELOAD. */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include "mtmd.h"
#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

static int inspect(const struct ggml_tensor *t, const char *role) {
    size_t bytes = ggml_nbytes(t), bad = 0;
    unsigned char *data = malloc(bytes);
    if (!data) abort();
    ggml_backend_tensor_get(t, data, 0, bytes);
    float maximum = 0;
    for (int64_t w = 0; w < t->ne[3]; w++)
    for (int64_t z = 0; z < t->ne[2]; z++)
    for (int64_t y = 0; y < t->ne[1]; y++)
    for (int64_t x = 0; x < t->ne[0]; x++) {
        size_t offset = x*t->nb[0] + y*t->nb[1] + z*t->nb[2] + w*t->nb[3];
        float v = t->type == GGML_TYPE_F32 ? *(float *)(data + offset) : ggml_fp16_to_fp32(*(ggml_fp16_t *)(data + offset));
        if (!isfinite(v)) bad++;
        else if (fabsf(v) > maximum) maximum = fabsf(v);
    }
    fprintf(stderr, "fa-probe role=%s name=%s buffer=%s type=%s shape=%lld,%lld,%lld,%lld strides=%zu,%zu,%zu,%zu max_abs=%g nonfinite=%zu\n",
            role, t->name, t->buffer ? ggml_backend_buffer_name(t->buffer) : "view",
            ggml_type_name(t->type), (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2], (long long)t->ne[3],
            t->nb[0], t->nb[1], t->nb[2], t->nb[3], maximum, bad);
    free(data);
    return bad != 0;
}

static bool observe(struct ggml_tensor *t, bool ask, void *unused) {
    (void)unused;
    if (ask) return t->op == GGML_OP_FLASH_ATTN_EXT;
    int bad = 0;
    for (int i = 0; i < 3; i++) bad |= inspect(t->src[i], i == 0 ? "Q" : i == 1 ? "K" : "V");
    bad |= inspect(t, "FA");
    if (bad) { fflush(stderr); _Exit(42); }
    return true;
}

mtmd_context *mtmd_init_from_file(const char *path, const struct llama_model *model,
                                 const struct mtmd_context_params initial) {
    struct mtmd_context_params params = initial;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.cb_eval = observe;
    params.cb_eval_user_data = NULL;
    mtmd_context *(*real_init)(const char *, const struct llama_model *, const struct mtmd_context_params) =
        dlsym(RTLD_NEXT, "mtmd_init_from_file");
    if (!real_init) abort();
    return real_init(path, model, params);
}
