/* Standalone C CUDA-driver launcher. No CUDA runtime or CPU attention path. */
#include "vis_fa.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#ifndef VIS_FA_NO_GGML
#include "ggml.h"
#endif

#ifndef VIS_FA_PTX_PATH
#define VIS_FA_PTX_PATH "kern/vis_fa.ptx"
#endif

static CUresult report(CUresult rc, const char *where) {
    if (rc != CUDA_SUCCESS) {
        const char *name = "unknown", *message = "unknown";
        (void)cuGetErrorName(rc, &name);
        (void)cuGetErrorString(rc, &message);
        fprintf(stderr, "vis_fa: %s: %s (%d): %s\n", where, name, (int)rc, message);
    }
    return rc;
}

static CUresult invalid(const char *why) {
    fprintf(stderr, "vis_fa: unsupported/invalid input: %s\n", why);
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult vis_fa_init(struct vis_fa_module *m, const char *path) {
    if (!m) return invalid("NULL module");
    memset(m, 0, sizeof *m);
    CUresult rc = cuCtxGetCurrent(&m->context);
    if (rc != CUDA_SUCCESS) return report(rc, "cuCtxGetCurrent");
    if (!m->context) return report(CUDA_ERROR_INVALID_CONTEXT, "no current context");
    CUdevice dev;
    int major = 0, minor = 0;
    rc = cuCtxGetDevice(&dev);
    if (rc != CUDA_SUCCESS) return report(rc, "cuCtxGetDevice");
    rc = cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev);
    if (rc != CUDA_SUCCESS) return report(rc, "compute capability major");
    rc = cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev);
    if (rc != CUDA_SUCCESS) return report(rc, "compute capability minor");
    if (major < 5 || (major == 5 && minor < 2)) return invalid("requires sm_52 or later");
    rc = cuModuleLoad(&m->module, path ? path : VIS_FA_PTX_PATH);
    if (rc != CUDA_SUCCESS) return report(rc, "cuModuleLoad " VIS_FA_PTX_PATH);
    rc = cuModuleGetFunction(&m->kernel, m->module, VIS_FA_KERNEL);
    if (rc != CUDA_SUCCESS) {
        report(rc, "cuModuleGetFunction " VIS_FA_KERNEL);
        (void)cuModuleUnload(m->module);
        m->module = NULL;
        return rc;
    }
    return CUDA_SUCCESS;
}

CUresult vis_fa_destroy(struct vis_fa_module *m) {
    if (!m) return invalid("NULL module");
    if (!m->module) return CUDA_SUCCESS;
    CUcontext current = NULL;
    CUresult rc = cuCtxGetCurrent(&current);
    if (rc != CUDA_SUCCESS) return report(rc, "cuCtxGetCurrent");
    if (current != m->context) return report(CUDA_ERROR_INVALID_CONTEXT, "module context mismatch");
    rc = cuModuleUnload(m->module);
    if (rc == CUDA_SUCCESS) memset(m, 0, sizeof *m);
    return report(rc, "cuModuleUnload");
}

/* Check arithmetic/alignment without guessing allocation extents. The caller
 * owns the buffers and is responsible for their size and non-overlap. */
static int tensor_ok(CUdeviceptr base, const uint64_t s[3], uint32_t nt, unsigned align) {
    const uint64_t extent[3] = {71, (uint64_t)nt - 1, 15};
    uint64_t last = align - 1;
    if (!base || base % align) return 0;
    for (unsigned i = 0; i < 3; i++) {
        if (s[i] < align || s[i] % align) return 0;
        if (extent[i] && s[i] > (UINT64_MAX - last) / extent[i]) return 0;
        last += extent[i] * s[i];
    }
    return base <= UINT64_MAX - last;
}

CUresult vis_fa_launch(const struct vis_fa_module *m,
                     const struct vis_fa_desc *d, CUstream stream) {
    if (!m || !m->module || !m->kernel || !d) return invalid("uninitialized module/descriptor");
    /* Leave room for rounded-up tiles in the PTX u32 indexing. */
    if (d->dim != 72 || d->heads != 16 || d->nq == 0 || d->nk == 0 ||
        d->nq > 0x7ffffff0u || d->nk > 0x7fffffe0u || !isfinite(d->scale))
        return invalid("requires D72/H16, positive lengths, finite scale");
    if (!tensor_ok(d->q, d->qs, d->nq, 4) || !tensor_ok(d->k, d->ks, d->nk, 2) ||
        !tensor_ok(d->v, d->vs, d->nk, 2) || !tensor_ok(d->o, d->os, d->nq, 4))
        return invalid("unaligned/overflowing tensor addresses or byte strides");
    CUcontext current = NULL;
    CUresult rc = cuCtxGetCurrent(&current);
    if (rc != CUDA_SUCCESS) return report(rc, "cuCtxGetCurrent");
    if (current != m->context) return report(CUDA_ERROR_INVALID_CONTEXT, "module context mismatch");
    CUdeviceptr q = d->q, k = d->k, v = d->v, o = d->o;
    uint64_t qs[3], ks[3], vs[3], os[3];
    memcpy(qs, d->qs, sizeof qs); memcpy(ks, d->ks, sizeof ks);
    memcpy(vs, d->vs, sizeof vs); memcpy(os, d->os, sizeof os);
    uint32_t nq = d->nq, nk = d->nk;
    float scale = d->scale;
    void *args[] = {&q, &k, &v, &o,
        &qs[0], &qs[1], &qs[2], &ks[0], &ks[1], &ks[2],
        &vs[0], &vs[1], &vs[2], &os[0], &os[1], &os[2], &nq, &nk, &scale};
    return report(cuLaunchKernel(m->kernel, (nq + 15u) / 16u, 16, 1,
                                128, 1, 1, 0, stream, args, NULL), "cuLaunchKernel " VIS_FA_KERNEL);
}

/* Modules stay associated with their owning context. A small per-thread
 * cache avoids locks and repeated JIT/module loads in the synchronous ggml
 * adapter; explicit module API is available to callers managing lifetimes.
 * The cache is retained until process exit. Contexts must outlive this use.
 */
static _Thread_local struct vis_fa_module cache[16];
static _Thread_local struct vis_fa_stats last_stats;

const struct vis_fa_stats *vis_fa_last_stats(void) { return &last_stats; }

int vis_fa_execute(const struct vis_fa_desc *d, struct vis_fa_stats *stats) {
    CUcontext context = NULL;
    CUevent start = NULL, end = NULL;
    CUresult rc;
    struct vis_fa_module *m = NULL;
    struct vis_fa_stats s = {.kernel_name = VIS_FA_KERNEL,
                            .shared_bytes_per_block = VIS_FA_SHARED_BYTES};
    size_t free_bytes, total_bytes;
    int registers = 0;
    memset(&last_stats, 0, sizeof last_stats);
    if (stats) memset(stats, 0, sizeof *stats);
#define CHECK(call) do { rc = (call); if (rc != CUDA_SUCCESS) { report(rc, #call); goto fail; } } while (0)
    CHECK(cuCtxGetCurrent(&context));
    if (!context) { report(CUDA_ERROR_INVALID_CONTEXT, "no current context"); goto fail; }
    /* ggml may have produced inputs on a non-default runtime stream. */
    CHECK(cuCtxSynchronize());
    for (unsigned i = 0; i < sizeof cache / sizeof *cache; i++)
        if (cache[i].module && cache[i].context == context) { m = &cache[i]; break; }
    if (!m) {
        for (unsigned i = 0; i < sizeof cache / sizeof *cache; i++)
            if (!cache[i].module) { m = &cache[i]; break; }
        if (!m) { invalid("per-thread context cache full; use explicit module API"); goto fail; }
        CHECK(vis_fa_init(m, NULL));
    }
    CHECK(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, m->kernel));
    s.registers_per_thread = (uint32_t)registers;
    CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
    s.free_before = free_bytes; s.total_bytes = total_bytes;
    CHECK(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK(cuEventCreate(&end, CU_EVENT_DEFAULT));
    CHECK(cuEventRecord(start, NULL));
    CHECK(vis_fa_launch(m, d, NULL));
    CHECK(cuEventRecord(end, NULL));
    CHECK(cuCtxSynchronize());
    CHECK(cuEventElapsedTime(&s.elapsed_ms, start, end));
    CHECK(cuMemGetInfo(&free_bytes, &total_bytes));
    s.free_after = free_bytes;
    CHECK(cuEventDestroy(start)); start = NULL;
    CHECK(cuEventDestroy(end)); end = NULL;
    last_stats = s;
    if (stats) *stats = s;
    return 0;
fail:
    if (start) (void)cuEventDestroy(start);
    if (end) (void)cuEventDestroy(end);
    return -1;
#undef CHECK
}

#ifndef VIS_FA_NO_GGML
int vis_fa_run(const struct ggml_tensor *q, const struct ggml_tensor *k,
               const struct ggml_tensor *v, struct ggml_tensor *dst, float scale) {
    if (!q || !k || !v || !dst) { invalid("NULL ggml tensor"); return -1; }
    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 ||
        v->type != GGML_TYPE_F16 || dst->type != GGML_TYPE_F32 ||
        q->ne[0] != 72 || k->ne[0] != 72 || v->ne[0] != 72 || dst->ne[0] != 72 ||
        q->ne[2] != 16 || k->ne[2] != 16 || v->ne[2] != 16 || dst->ne[1] != 16 ||
        q->ne[3] != 1 || k->ne[3] != 1 || v->ne[3] != 1 || dst->ne[3] != 1 ||
        q->ne[1] != dst->ne[2] || k->ne[1] != v->ne[1] ||
        q->ne[1] <= 0 || k->ne[1] <= 0 ||
        q->ne[1] > 0x7ffffff0 || k->ne[1] > 0x7fffffe0) {
        invalid("ggml layout/type/shape outside dense D72/H16 support"); return -1;
    }
    if (dst->op != GGML_OP_FLASH_ATTN_EXT || dst->src[3] || dst->src[4]) {
        invalid("ggml op must be unmasked FLASH_ATTN_EXT without sinks"); return -1;
    }
    float max_bias, softcap;
    int32_t n_kv_max;
    memcpy(&max_bias, (const char *)dst->op_params + 4, sizeof max_bias);
    memcpy(&softcap, (const char *)dst->op_params + 8, sizeof softcap);
    memcpy(&n_kv_max, (const char *)dst->op_params + 16, sizeof n_kv_max);
    if (max_bias != 0 || softcap != 0 || n_kv_max != 0) {
        invalid("bias/softcap/sparse attention parameters"); return -1;
    }
    struct vis_fa_desc d = {
        .q = (CUdeviceptr)(uintptr_t)q->data, .k = (CUdeviceptr)(uintptr_t)k->data,
        .v = (CUdeviceptr)(uintptr_t)v->data, .o = (CUdeviceptr)(uintptr_t)dst->data,
        .qs = {q->nb[0], q->nb[1], q->nb[2]},
        .ks = {k->nb[0], k->nb[1], k->nb[2]},
        .vs = {v->nb[0], v->nb[1], v->nb[2]},
        .os = {dst->nb[0], dst->nb[2], dst->nb[1]},
        .nq = (uint32_t)q->ne[1], .nk = (uint32_t)k->ne[1],
        .heads = 16, .dim = 72, .scale = scale
    };
    return vis_fa_execute(&d, NULL);
}
#endif
