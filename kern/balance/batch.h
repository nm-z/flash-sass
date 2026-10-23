#ifndef FLASH_BALANCE_BATCH_H
#define FLASH_BALANCE_BATCH_H
#include "../prog.h"
#define OP_BALANCED_BATCH 108u
#define OP_BALANCED_HCUP5 109u
#define BALANCED_BATCH_MAX 5u
typedef struct {
    unsigned positions;
    unsigned outer_phases;
    unsigned causal_phase_replays;
    unsigned inner_grid_joins;
    unsigned publications;
    unsigned peer_acquires;
    unsigned native_hcup_phases;
} balance_batch_work;
/* Every input is a complete, already-fused balanced position program. The
 * wrapper preserves complete physical-phase bundles, including their child
 * descriptors. out must be initialized and is unchanged on failure.
 * Counters distinguish outer publications from causal replays/local joins;
 * speculative accounting must divide total work by committed tokens only. */
int prog_balance_batch(prog_t *out,const prog_t *const *positions,
                       unsigned count,balance_batch_work *work);
#endif
