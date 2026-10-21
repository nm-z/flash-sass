#ifndef FLASH_MTP_LOAD_H
#define FLASH_MTP_LOAD_H
#include "mtp/model.h"
#include "gguf.h"
typedef CUdeviceptr (*mtp_alloc_fn)(size_t);
/* The current CUDA context is the destination die. alloc owns every allocation
 * for normal engine teardown, including partial initialization failures.
 * A NULL allocator validates shapes and returns the required byte count only. */
CUresult mtp_load_weights(mtp_weights_t *,const gguf_file *,const split_t *,
                          const dn_weights_t *,int,mtp_alloc_fn,size_t *);
const gguf_tensor *mtp_embedding_tensor(const gguf_file *);
int mtp_embedding_row(const gguf_tensor *,uint32_t,float [2560]);
#endif
