/* Installed CUDA backend FA validation: aligned keys, no custom interposer.
 * One native GPU FA graph per shape, sampled float64 reference at every head.
 * compute_wall_ms is host monotonic graph execution plus synchronization;
 * it is not a kernel-only timing and excludes transfers and CPU reference.
 */
#define _POSIX_C_SOURCE 200809L
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static uint32_t rng = 96;
static float random_value(void) {
    rng = rng * 1664525u + 1013904223u;
    return ((int)(rng >> 16) - 32768) / 32768.0f;
}

static double seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) { perror("clock_gettime"); exit(2); }
    return (double)t.tv_sec + (double)t.tv_nsec * 1e-9;
}

static int check(ggml_backend_t backend, ggml_backend_dev_t dev, unsigned nq, unsigned nk) {
    const unsigned dim = 72, heads = 16;
    const float scale = 1.0f / sqrtf((float)dim);
    const size_t qc = (size_t)nq * heads * dim, kc = (size_t)nk * heads * dim;
    float *q = malloc(qc * sizeof *q), *out = malloc(qc * sizeof *out);
    ggml_fp16_t *k = malloc(kc * sizeof *k), *v = malloc(kc * sizeof *v);
    double *scores = malloc((size_t)nk * sizeof *scores);
    if (!q || !out || !k || !v || !scores) { fprintf(stderr, "host allocation failed\n"); exit(2); }
    for (size_t i = 0; i < qc; i++) q[i] = random_value();
    for (unsigned d = 0; d < dim; d++) q[d] *= 1024.0f;
    for (size_t i = 0; i < kc; i++) {
        k[i] = ggml_fp32_to_fp16(random_value());
        v[i] = ggml_fp32_to_fp16(random_value());
    }

    struct ggml_context *ctx = ggml_init((struct ggml_init_params){
        .mem_size = 4 << 20, .no_alloc = true});
    if (!ctx) { fprintf(stderr, "ggml_init failed\n"); exit(2); }
    struct ggml_tensor *tq = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, dim, heads, nq);
    tq = ggml_permute(ctx, tq, 0, 2, 1, 3);
    struct ggml_tensor *tk = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, nk, heads);
    struct ggml_tensor *tv = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, dim, nk, heads);
    struct ggml_tensor *to = ggml_flash_attn_ext(ctx, tq, tk, tv, NULL, scale, 0, 0);
    ggml_prec_set_acc(to, GGML_PREC_F32);
    if (!ggml_backend_dev_supports_op(dev, to)) {
        fprintf(stderr, "CUDA6 does not support the requested native FA op\n"); exit(2);
    }
    struct ggml_cgraph *graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, to);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer || ggml_backend_buffer_is_host(buffer) ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(buffer)) != dev) {
        fprintf(stderr, "native FA graph did not allocate on CUDA6\n"); exit(2);
    }
    ggml_backend_tensor_set(tq, q, 0, qc * sizeof *q);
    ggml_backend_tensor_set(tk, k, 0, kc * sizeof *k);
    ggml_backend_tensor_set(tv, v, 0, kc * sizeof *v);
    ggml_backend_synchronize(backend);
    const double start = seconds();
    enum ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    const double elapsed_ms = (seconds() - start) * 1000.0;
    ggml_backend_tensor_get(to, out, 0, qc * sizeof *out);
    size_t nonfinite = 0;
    for (size_t i = 0; i < qc; i++) nonfinite += !isfinite(out[i]);

    const unsigned sample[] = {0, nq > 1 ? 1 : 0, nq / 2, nq - 1};
    double max_error = 0;
    unsigned max_head = 0, max_query = 0, max_dim = 0;
    for (unsigned h = 0; h < heads; h++) for (unsigned s = 0; s < 4; s++) {
        const unsigned qi = sample[s];
        double maximum = -INFINITY, sum = 0, reference[72] = {0};
        for (unsigned j = 0; j < nk; j++) {
            double dot = 0;
            for (unsigned d = 0; d < dim; d++)
                dot += (double)q[((size_t)qi * heads + h) * dim + d] *
                    ggml_fp16_to_fp32(k[((size_t)h * nk + j) * dim + d]);
            scores[j] = dot * scale;
            if (scores[j] > maximum) maximum = scores[j];
        }
        for (unsigned j = 0; j < nk; j++) {
            const double p = exp(scores[j] - maximum);
            sum += p;
            for (unsigned d = 0; d < dim; d++)
                reference[d] += p * ggml_fp16_to_fp32(v[((size_t)h * nk + j) * dim + d]);
        }
        for (unsigned d = 0; d < dim; d++) {
            const double actual = out[((size_t)qi * heads + h) * dim + d];
            const double error = isfinite(actual) ? fabs(actual - reference[d] / sum) : INFINITY;
            if (error > max_error) {
                max_error = error; max_head = h; max_query = qi; max_dim = d;
            }
        }
    }
    const int failed = status != GGML_STATUS_SUCCESS || nonfinite || max_error > 2e-4;
    printf("{\"event\":\"fa_native_reference\",\"backend\":\"%s\","
           "\"implementation\":\"installed_ggml_cuda_flash_attn_ext\","
           "\"queries\":%u,\"keys\":%u,\"dim\":72,\"heads\":16,"
           "\"sampled_queries_per_head\":4,\"large_q_scale\":1024,"
           "\"status\":%d,\"nonfinite\":%zu,\"max_abs_error\":%.12g,"
           "\"max_head\":%u,\"max_query\":%u,\"max_dim\":%u,"
           "\"compute_wall_ms\":%.6f,\"result\":\"%s\"}\n",
           ggml_backend_dev_name(dev), nq, nk, (int)status, nonfinite,
           max_error, max_head, max_query, max_dim, elapsed_ms, failed ? "fail" : "pass");
    fflush(stdout);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    free(q); free(out); free(k); free(v); free(scores);
    return failed;
}

int main(int argc, char **argv) {
    if (!getenv("GPUQ_JOB_ID")) { fprintf(stderr, "run through gpuq\n"); return 2; }
    if (setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1)) { perror("setenv"); return 2; }
    ggml_backend_load_all_from_path(argc > 1 ? argv[1] : "/home/nate/claude/llama.cpp-cuda/build/bin");
    ggml_backend_dev_t dev = ggml_backend_dev_by_name("CUDA6");
    if (!dev || ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        fprintf(stderr, "CUDA6 GPU backend unavailable\n"); return 2;
    }
    ggml_backend_t backend = ggml_backend_dev_init(dev, NULL);
    if (!backend) { fprintf(stderr, "CUDA6 initialization failed\n"); return 2; }
    int failed = 0;
    failed |= check(backend, dev, 13, 256);
    failed |= check(backend, dev, 512, 512);
    failed |= check(backend, dev, 4096, 4096);
    failed |= check(backend, dev, 16384, 16384);
    failed |= check(backend, dev, 13, 65536);
    ggml_backend_free(backend);
    return failed;
}
