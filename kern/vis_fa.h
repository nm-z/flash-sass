/* GPU-only dense FlashAttention, D=72, H=16. C CUDA-driver interface.
 * Build: cc -std=c11 -O2 -I/opt/cuda/include -I.../ggml/include
 *        -c kern/vis_fa.c; link -lcuda -lm.
 * PTX can be JIT loaded, or assembled with CUDA 11.4 ptxas -arch=sm_52.
 */
#ifndef VIS_FA_H
#define VIS_FA_H

#include <cuda.h>
#include <stdint.h>

#define VIS_FA_KERNEL "vis_fa_d72_f32"
#define VIS_FA_SHARED_BYTES 25376u
#define VIS_FA_QUERY_TILE 16u
#define VIS_FA_KEY_TILE 32u

/* Each tensor is indexed [dimension, token, head], using BYTE strides.
 * q/o are F32, k/v are F16. Lengths need not be multiples of a tile.
 * Dense, unmasked attention: o=softmax(scale*q*k^T)*v, all keys visited.
 * No causal mask, bias, softcap, batches, or head broadcasting.
 * Buffers must belong to the current CUDA context, be valid for their
 * extents and not overlap output. Strides must describe distinct elements.
 */
struct vis_fa_desc {
    CUdeviceptr q, k, v, o;
    uint64_t qs[3], ks[3], vs[3], os[3];
    uint32_t nq, nk, heads, dim;
    float scale;
};

struct vis_fa_module {
    CUmodule module;
    CUfunction kernel;
    CUcontext context;
};

/* These are measurements of this launcher, not a device-wide peak sampler.
 * No global workspace/device allocations are made by the FA launch itself.
 * Module/JIT driver overhead is included in the device free-memory samples.
 */
struct vis_fa_stats {
    const char *kernel_name;
    float elapsed_ms;                 /* CUDA event clock, kernel only */
    uint64_t free_before, free_after, total_bytes;
    uint64_t workspace_bytes;          /* exactly zero */
    uint32_t shared_bytes_per_block, registers_per_thread;
};

/* Caller owns a current context. path accepts PTX or a cubin. NULL uses
 * VIS_FA_PTX_PATH build macro (kern/vis_fa.ptx).
 * Module init/destroy and launch must run with that context current.
 * Destroy only after work using this module has completed.
 * CUDA results are propagated; diagnostics go to stderr; no fallback.
 */
CUresult vis_fa_init(struct vis_fa_module *module, const char *path);
CUresult vis_fa_destroy(struct vis_fa_module *module);
CUresult vis_fa_launch(const struct vis_fa_module *module,
                     const struct vis_fa_desc *desc, CUstream stream);

/* Convenience interface: per-thread/per-context module cache, synchronizes
 * current context before and after launch, measures with CUDA events.
 * Returns zero on success, -1 on any validation/CUDA error.
 */
int vis_fa_execute(const struct vis_fa_desc *desc, struct vis_fa_stats *stats);
const struct vis_fa_stats *vis_fa_last_stats(void);

/* ggml adapter; define VIS_FA_NO_GGML to build the raw driver interface
 * without ggml headers. Validates the supported shape/types/op parameters.
 * Output layout ggml [72,16,Nq,1], input [72,N,16,1].
 */
struct ggml_tensor;
int vis_fa_run(const struct ggml_tensor *q, const struct ggml_tensor *k,
               const struct ggml_tensor *v, struct ggml_tensor *dst,
               float scale);

#endif
