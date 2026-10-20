/* Isolate the GPU FA operator without a model. Constant V must stay constant. */
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    ggml_backend_load_all_from_path(argc > 1 ? argv[1] : "/home/nate/claude/llama.cpp-cuda/build/bin");
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA6");
    if (!dev) return 2;
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    const int sizes[] = {1024, 2048, 2304, 2408, 2432, 4096};
    int failed = 0;
    for (int d = 72; d <= 72; d += 8) for (size_t i = 0; i < (argc > 2 ? 1 : sizeof sizes / sizeof *sizes); i++) {
        int n = argc > 3 ? atoi(argv[3]) : sizes[i], nk = argc > 2 ? atoi(argv[2]) : n, h = 16;
        struct ggml_context *ctx = ggml_init((struct ggml_init_params){.mem_size = 4 << 20, .no_alloc = true});
        struct ggml_tensor *q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, d, h, n);
        q = ggml_permute(ctx, q, 0, 2, 1, 3);
        struct ggml_tensor *k = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, nk, h);
        struct ggml_tensor *v = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, d, nk, h);
        struct ggml_tensor *out = ggml_flash_attn_ext(ctx, q, k, v, NULL, 1.0f/sqrtf(d), 0, 0);
        ggml_prec_set_acc(out, GGML_PREC_F32);
        struct ggml_cgraph *graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, out);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
        size_t count = (size_t)d * n * h;
        float *data = calloc(count, sizeof *data);
        size_t kv_count = (size_t)d * nk * h;
        ggml_fp16_t *half = calloc(kv_count, sizeof *half);
        ggml_backend_tensor_set(q, data, 0, count * 4);
        ggml_backend_tensor_set(k, half, 0, kv_count * 2);
        for (size_t j = 0; j < kv_count; j++) half[j] = ggml_fp32_to_fp16(1.0f);
        ggml_backend_tensor_set(v, half, 0, kv_count * 2);
        enum ggml_status status = ggml_backend_graph_compute(backend, graph);
        ggml_backend_tensor_get(out, data, 0, count * 4);
        size_t bad = 0, nan = 0;
        for (size_t j = 0; j < count; j++) {
            if (!isfinite(data[j])) nan++;
            if (!isfinite(data[j]) || fabsf(data[j]-1) > 1e-4f) bad++;
        }
        printf("{\"event\":\"fa_constant\",\"head\":%d,\"tokens\":%d,\"keys\":%d,\"status\":%d,\"nonfinite\":%zu,\"bad\":%zu,\"total\":%zu}\n", d, n, nk, status, nan, bad, count);
        fflush(stdout);
        failed |= bad != 0 || status != GGML_STATUS_SUCCESS;
        free(data); free(half);
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    ggml_backend_free(backend);
    return failed;
}
