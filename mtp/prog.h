#ifndef FLASH_MTP_PROG_H
#define FLASH_MTP_PROG_H
#include "../prog.h"
#include "mtp_ops.h"
/* FP32 activation inputs; Q8 weights in repack_b32 layout. count may be zero
 * (a dense phase), otherwise slot >= *count leaves output untouched. */
int mtp_ph_q8(prog_t *,unsigned,CUdeviceptr,CUdeviceptr,CUdeviceptr,unsigned,unsigned,CUdeviceptr,unsigned,CUdeviceptr);
void mtp_ph_swiglu(prog_t *,CUdeviceptr,CUdeviceptr,CUdeviceptr,unsigned,CUdeviceptr);
void mtp_ph_weighted_sum(prog_t *,CUdeviceptr,CUdeviceptr,CUdeviceptr,CUdeviceptr,CUdeviceptr,CUdeviceptr);
void mtp_ph_eh_concat(prog_t *,CUdeviceptr,CUdeviceptr,CUdeviceptr);
#endif
