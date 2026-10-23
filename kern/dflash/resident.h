#ifndef FLASH_DFLASH_RESIDENT_H
#define FLASH_DFLASH_RESIDENT_H
#include <cuda.h>
#include <stddef.h>
#include "device.h"

typedef struct {
    CUcontext context;
    CUdeviceptr output_weights,embedding;
    unsigned output_rows,output_first,output_stride;
    unsigned embedding_first_group,embedding_groups,embedding_stride,embedding_raw_q3;
    size_t embedding_bytes;
} dflash_target_binding;
typedef struct dflash_resident dflash_resident;

/* Borrow one target context for the drafter; the target head remains sharded
 * over all six contexts. Only the final four input IDs cross to the host. */
int dflash_resident_open(dflash_resident **,const char *gguf,const char *module,
                        const dflash_target_binding target[6],unsigned draft_rank,
                        CUdeviceptr feature_ring,char *error,size_t error_size);
int dflash_resident_begin(dflash_resident *,unsigned processed,char *,size_t);
int dflash_resident_propose(dflash_resident *,uint32_t anchor,uint32_t inputs[4],char *,size_t);
int dflash_resident_append(dflash_resident *,unsigned start,unsigned retained,char *,size_t);
int dflash_resident_finish(dflash_resident *,char *,size_t);
void dflash_resident_target_layers(const dflash_resident *,unsigned layers[5]);
void dflash_resident_close(dflash_resident *);
#endif
