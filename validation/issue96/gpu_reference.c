/* Independent GPU dense-attention numerical reference, test executable only.
 * This executable is not linked into flashvis and exposes no service mode. */
#define _GNU_SOURCE
#include "ggml-backend.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
#include "../../kern/vision.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static ggml_backend_dev_t selected;
static unsigned gpu_nodes, kq_nodes, pv_nodes;
static int observer_failed;
static bool observe(struct ggml_tensor *t, bool ask, void *unused) {
    (void)unused;
    if (ask) return t->op != GGML_OP_NONE && t->op != GGML_OP_VIEW &&
                    t->op != GGML_OP_RESHAPE && t->op != GGML_OP_PERMUTE && t->op != GGML_OP_TRANSPOSE;
    if (!t->buffer || ggml_backend_buffer_is_host(t->buffer) ||
        ggml_backend_buft_get_device(ggml_backend_buffer_get_type(t->buffer)) != selected) {
        observer_failed = 1;
        return false;
    }
    gpu_nodes++;
    if (!strcmp(t->name, "reference_kq")) kq_nodes++;
    if (!strcmp(t->name, "reference_pv")) pv_nodes++;
    return true;
}

/* Keep exactly the production F16 K/V values, then use separate F32 matrix
 * products and softmax as an independent oracle for the FA implementation. */
struct ggml_tensor *ggml_flash_attn_ext(struct ggml_context *ctx,
        struct ggml_tensor *q, struct ggml_tensor *k, struct ggml_tensor *v,
        struct ggml_tensor *mask, float scale, float bias, float softcap) {
    if (mask || bias != 0 || softcap != 0) abort();
    k = ggml_cast(ctx, k, GGML_TYPE_F32);
    v = ggml_cast(ctx, v, GGML_TYPE_F32);
    struct ggml_tensor *kq = ggml_mul_mat(ctx, k, q);
    ggml_prec_set_acc(kq, GGML_PREC_F32);
    ggml_set_name(kq, "reference_kq");
    struct ggml_tensor *p = ggml_soft_max_ext(ctx, kq, NULL, scale, 0);
    struct ggml_tensor *vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 0, 2, 3));
    struct ggml_tensor *pv = ggml_mul_mat(ctx, vt, p);
    ggml_prec_set_acc(pv, GGML_PREC_F32);
    ggml_set_name(pv, "reference_pv");
    return ggml_cont(ctx, ggml_permute(ctx, pv, 0, 2, 1, 3));
}

int main(int argc, char **argv) {
    if (argc != 6) { fprintf(stderr,"usage: gpu_reference PROJECTOR TEXT IMAGE OUTPUT BACKEND_DIR\n");return 2; }
    if (!getenv("GPUQ_JOB_ID")) { fprintf(stderr,"run through gpuq\n");return 2; }
    setenv("GGML_CUDA_DISABLE_GRAPHS", "1", 1);
    ggml_backend_load_all_from_path(argv[5]);
    llama_backend_init();
    selected = ggml_backend_dev_by_name("CUDA6");
    if (!selected) return 2;
    struct llama_model_params mp = llama_model_default_params();mp.vocab_only = true;
    struct llama_model *model = llama_model_load_from_file(argv[2], mp);
    if (!model) return 1;
    struct mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = true;cp.device = selected;cp.n_threads = 10;
    cp.warmup = false;cp.print_timings = false;cp.cb_eval = observe;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    cp.image_min_tokens = FLASH_IMAGE_MIN_TOKENS;cp.image_max_tokens = FLASH_IMAGE_MAX_TOKENS;
    mtmd_context *ctx = mtmd_init_from_file(argv[1], model, cp);
    if (!ctx) return 1;
    struct mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, argv[3], false, mtmd_helper_init_opt_default());
    if (!bw.bitmap) return 1;
    mtmd_input_chunks *chunks = mtmd_input_chunks_init();
    mtmd_input_text text = {.text = mtmd_default_marker()};text.text_len = strlen(text.text);
    const mtmd_bitmap *images[] = {bw.bitmap};
    if (mtmd_tokenize(ctx, chunks, &text, images, 1)) return 1;
    const mtmd_input_chunk *img = NULL;
    for (size_t i=0;i<mtmd_input_chunks_size(chunks);i++) {
        const mtmd_input_chunk *part = mtmd_input_chunks_get(chunks,i);
        if (mtmd_input_chunk_get_type(part)==MTMD_INPUT_CHUNK_TYPE_IMAGE)img=part;
    }
    if (!img || mtmd_input_chunk_get_n_tokens(img)>1024) {
        fprintf(stderr,"dense test reference limited to 1024 image tokens\n");return 2;
    }
    if (mtmd_encode_chunk(ctx,img) || observer_failed || kq_nodes!=27 || pv_nodes!=27) {
        fprintf(stderr,"GPU reference failed: observer=%d GPU nodes=%u KQ=%u PV=%u\n",observer_failed,gpu_nodes,kq_nodes,pv_nodes);
        return 1;
    }
    size_t tokens=mtmd_input_chunk_get_n_tokens(img),count=tokens*2560;
    float *emb=mtmd_get_output_embd(ctx);
    for(size_t i=0;i<count;i++)if(!isfinite(emb[i]))return 1;
    FILE *out=fopen(argv[4],"wb");
    if(!out || fwrite(emb,4,count,out)!=count || fclose(out))return 1;
    printf("{\"event\":\"gpu_dense_reference\",\"tokens\":%zu,\"dimension\":2560,\"gpu_compute_ops\":%u,\"kq_ops\":%u,\"pv_ops\":%u}\n",tokens,gpu_nodes,kq_nodes,pv_nodes);
    mtmd_input_chunks_free(chunks);mtmd_bitmap_free(bw.bitmap);
    mtmd_free(ctx);llama_model_free(model);llama_backend_free();return 0;
}
