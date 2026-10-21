/* C adapter for the pinned ggml CUDA backend's FA dispatch entry point.
 * This is linked into flashvis and exported to its loaded CUDA backend.
 * The C++ ABI spelling belongs to the dependency; this file is C11.
 * The public graph callback verifies that dispatch actually reached our
 * GPU kernel. A missing/changed ABI is an explicit request failure.
 */
#include "vis_ggml.h"
#include "vis_fa.h"
#include <dlfcn.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static _Thread_local ggml_backend_dev_t selected;
static _Thread_local unsigned launched, observed, gpu_ops;
static _Thread_local const char *failure;
static _Thread_local double kernel_ms;
static _Thread_local uint64_t peak_used;

/* The pinned M60 tile kernel fails on a partial KV tile. Its aligned path
 * passes the same float64 oracle and is faster on large images. Select the
 * two GPU FlashAttention kernels by shape, before executing either one.
 * Errors never select another implementation or retry the request. */
static int native_aligned(void *opaque, struct ggml_tensor *dst, struct vis_fa_stats *stats) {
    typedef void (*dispatch_fn)(void *, struct ggml_tensor *);
    static _Thread_local dispatch_fn dispatch;
    if (!dispatch) dispatch = (dispatch_fn)dlsym(RTLD_NEXT,
        "_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");
    if (!dispatch) return -1;
    CUevent start = NULL, end = NULL;
    size_t available, total;
    memset(stats, 0, sizeof *stats);
    stats->kernel_name = "ggml_cuda_flash_attn_ext_tile_aligned";
#define CHECK(call) do { if ((call) != CUDA_SUCCESS) goto fail; } while (0)
    CHECK(cuCtxSynchronize());
    CHECK(cuMemGetInfo(&available, &total));
    stats->free_before = available; stats->total_bytes = total;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&end, CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    CHECK(cuEventSynchronize(start));
    dispatch(opaque, dst);
    CHECK(cuCtxSynchronize());
    CHECK(cuEventRecord(end, NULL));
    CHECK(cuEventSynchronize(end));
    CHECK(cuEventElapsedTime(&stats->elapsed_ms, start, end));
    CHECK(cuMemGetInfo(&available, &total));
    stats->free_after = available;
    cuEventDestroy(start); cuEventDestroy(end);
    return 0;
fail:
    if (start) cuEventDestroy(start);
    if (end) cuEventDestroy(end);
    return -1;
#undef CHECK
}

void vis_ggml_begin(ggml_backend_dev_t device) {
    selected = device; launched = observed = gpu_ops = 0;
    failure = NULL; kernel_ms = 0; peak_used = 0;
}

void vis_ggml_dispatch(void *opaque, struct ggml_tensor *dst)
    __asm__("_Z24ggml_cuda_flash_attn_extR25ggml_backend_cuda_contextP11ggml_tensor");

void vis_ggml_dispatch(void *opaque, struct ggml_tensor *dst) {
    if (failure) return;
    float scale, bias, softcap;
    memcpy(&scale, (char *)dst->op_params + 0, 4);
    memcpy(&bias, (char *)dst->op_params + 4, 4);
    memcpy(&softcap, (char *)dst->op_params + 8, 4);
    const struct ggml_tensor *q = dst->src[0], *k = dst->src[1], *v = dst->src[2];
    if (!selected || !q || !k || !v || dst->src[3] || dst->src[4] || bias != 0 || softcap != 0 || !isfinite(scale) ||
        q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 ||
        q->ne[0] != 72 || k->ne[0] != 72 || v->ne[0] != 72 ||
        q->ne[2] != 16 || k->ne[2] != 16 || v->ne[2] != 16 ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 ||
        q->ne[1] <= 0 || k->ne[1] <= 0 || k->ne[1] != v->ne[1]) {
        failure = "unsupported GPU Flash Attention graph";
        return;
    }
    struct vis_fa_stats stats;
    int rc;
    int aligned = k->ne[1] % 256 == 0;
    if (aligned) rc = native_aligned(opaque, dst, &stats);
    else { rc = vis_fa_run(q, k, v, dst, scale); if (!rc) stats = *vis_fa_last_stats(); }
    if (rc) {
        failure = "GPU Flash Attention kernel failed";
        return;
    }
    launched++; kernel_ms += stats.elapsed_ms;
    uint64_t used = stats.total_bytes - stats.free_after;
    if (used > peak_used) peak_used = used;
    fprintf(stderr, "flashvis-fa: kernel=%s backend=%s queries=%lld keys=%lld heads=%lld ms=%.3f ",
            stats.kernel_name, ggml_backend_dev_name(selected),
            (long long)dst->src[0]->ne[1], (long long)dst->src[1]->ne[1],
            (long long)dst->src[0]->ne[2], stats.elapsed_ms);
    if (aligned) fprintf(stderr, "workspace=library_managed shared=unmeasured registers=unmeasured ");
    else fprintf(stderr, "workspace=%llu shared=%u registers=%u ",
                 (unsigned long long)stats.workspace_bytes, stats.shared_bytes_per_block, stats.registers_per_thread);
    fprintf(stderr, "device_used=%llu\n", (unsigned long long)used);
}

bool vis_ggml_observe(struct ggml_tensor *tensor, bool ask, void *unused) {
    (void)unused;
    if (ask) return tensor->op != GGML_OP_NONE && tensor->op != GGML_OP_VIEW &&
                    tensor->op != GGML_OP_RESHAPE && tensor->op != GGML_OP_PERMUTE &&
                    tensor->op != GGML_OP_TRANSPOSE;
    if (failure) return false;
    if (!tensor->buffer || ggml_backend_buffer_is_host(tensor->buffer) ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(tensor->buffer)) != selected) {
        failure = "vision computation was assigned to a different backend";
        return false;
    }
    gpu_ops++;
    if (tensor->op != GGML_OP_FLASH_ATTN_EXT) return true;
    observed++;
    if (observed != launched) {
        failure = "required GPU Flash Attention dispatch was not used";
        return false;
    }
    return true;
}

const char *vis_ggml_result(void) {
    if (failure) return failure;
    if (!launched || launched != observed) return "required GPU Flash Attention did not execute";
    fprintf(stderr, "flashvis-fa: completed=%u observed_gpu=%u gpu_compute_ops=%u kernel_ms=%.3f sampled_device_peak_bytes=%llu\n",
            launched, observed, gpu_ops, kernel_ms, (unsigned long long)peak_used);
    return NULL;
}
