#ifndef FLASH_DFLASH_GGUF_MODEL_H
#define FLASH_DFLASH_GGUF_MODEL_H
#include "gguf.h"

#define DFLASH_LAYERS 5u
#define DFLASH_HIDDEN 2560u
#define DFLASH_FEATURES (5u*DFLASH_HIDDEN)
#define DFLASH_INTERMEDIATE 7680u
#define DFLASH_HEADS 32u
#define DFLASH_KV_HEADS 8u
#define DFLASH_HEAD_DIM 128u
#define DFLASH_WINDOW 2048u
#define DFLASH_BLOCK 4u
#define DFLASH_VOCAB 248320u

typedef struct {
    const gguf_tensor *attn_norm,*attn_conv_base,*attn_conv_projection;
    const gguf_tensor *q,*k,*v,*q_norm,*k_norm,*attention_output;
    const gguf_tensor *ffn_norm,*ffn_conv_base,*ffn_conv_projection;
    const gguf_tensor *gate,*up,*down;
} dflash_layer_weights;

typedef struct {
    gguf_file file;
    dflash_layer_weights layer[DFLASH_LAYERS];
    const gguf_tensor *feature_projection,*feature_norm,*output_norm;
    const gguf_tensor *selector_hidden,*selector_predecessor,*selector_successor;
    unsigned target_layer[DFLASH_LAYERS]; /* zero-based completed target blocks */
    unsigned capacity,window,mask_token;
    float epsilon,rope_theta;
    size_t packed_bytes,matrix_bytes,vector_bytes;
    unsigned q4_tensors,q6_tensors,f32_tensors;
} dflash_gguf_model;

/* Map and describe the selected Q4_K_M checkpoint. Matrix bytes remain Q4_K
 * or Q6_K; this API never expands the backbone into BF16/F32 matrices. */
int dflash_gguf_open(dflash_gguf_model *,const char *,char *,size_t);
void dflash_gguf_close(dflash_gguf_model *);
size_t dflash_tensor_bytes(const gguf_tensor *);
#endif
