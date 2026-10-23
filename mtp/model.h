#ifndef FLASH_MTP_MODEL_H
#define FLASH_MTP_MODEL_H
#include "mtp/prog.h"
typedef struct {
    /* Q8 projections in repack_b32 layout, original F32 norms, BF16 indexer.
     * Router ffn_gate_inp_x is converted from checkpoint F32 to BF16 when loaded.
     * block.output and obases refer to the resident target's Q5 output head. */
    dn_weights_t block;
    CUdeviceptr enorm,hnorm,eh_proj;
    CUdeviceptr head_norm,head_down,head_up;
} mtp_weights_t;
typedef struct {
    dn_bufs_t base; /* independent MTP activations and one QSA attention cache */
    CUdeviceptr embedding; /* normalized separately from the target residual */
    CUdeviceptr eh_concat; /* 4 x 5120 FP32 */
    CUdeviceptr expert_rows; /* 11 x 2560 FP32 */
} mtp_buffers_t;
/* Build one genuine draft step. hidden is target S for an initial proposal or
 * the preceding draft S for subsequent proposals. Caller stages embedding of
 * the consumed token. Output token and next draft residual remain on device.
 * This does not accept drafts, alter target KV/recurrent state, or implement
 * rollback; target verification must precede publishing any proposed token. */
int mtp_program(prog_t *,const split_t *,const xchg_t *,const mtp_weights_t *,
                const mtp_buffers_t *,CUdeviceptr,unsigned,unsigned,float,int);
#endif
