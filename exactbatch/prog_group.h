#ifndef FLASH_EXACT_GROUP_H
#define FLASH_EXACT_GROUP_H
#include "prog.h"
#include "exact_ops.h"
#define PROG_MAX_PHASES 4096u
#define PROG_MAX_POSITIONS 32u
#define PROG_MAX_DESCRIPTORS (PROG_MAX_PHASES * PROG_MAX_POSITIONS)
#define PHASE_GROUP_FLAG 8u
#define PHASE_FUSED_FLAG 16u
#define PHASE_WIDTH_SHIFT 8u
#define PHASE_WIDTH_MASK (255u << PHASE_WIDTH_SHIFT)
#define PHASE_CHAIN_FLAG 32u
#define PHASE_CHAIN_BAR_FLAG 64u
/* The verification module appends a bounded copy after the B8 opcode table. */
#define OP_VERIFY_COPY EXACT_OP_COUNT
int prog_verify_copy(prog_t *,CUdeviceptr,CUdeviceptr,size_t);
static inline unsigned phase_descriptors(const phase_t *p) {
    if (!(p->w[30] & PHASE_GROUP_FLAG)) return 1;
    unsigned n=(p->w[30]&PHASE_WIDTH_MASK)>>PHASE_WIDTH_SHIFT;
    return n?n:2;
}
static inline unsigned phase_positions(const phase_t *p) {
    return p->w[30]&PHASE_CHAIN_FLAG?1:phase_descriptors(p);
}
int prog_group(prog_t *, const prog_t *const *, unsigned);
int prog_group_aliased(prog_t *, const prog_t *const *, unsigned);
int prog_batch_fuse_bf16(prog_t *, int);
int prog_batch_fuse_q2(prog_t *, int);
int prog_batch_fuse_q3(prog_t *);
int prog_batch_fuse_hcup(prog_t *, int);
int prog_batch_fuse_dnstep(prog_t *);
int prog_batch_fuse_actq(prog_t *);
int prog_batch_fuse_rms(prog_t *);
int prog_batch_decompose_hcr(prog_t *, const CUdeviceptr *, unsigned, int);
#endif
