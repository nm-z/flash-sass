#ifndef FLASH_BALANCE_LAYOUT_H
#define FLASH_BALANCE_LAYOUT_H
#include "../prog.h"

#define FLASH_BALANCE_LAYOUT_MARKER "__flash_balance_rows_v1"
#define FLASH_BALANCE_DIES 6u
#define FLASH_BALANCE_EXPERT_GRAIN 8u
#define PHASE_BALANCED_ROW8 0x80000u
#define PHASE_BALANCED_PUBLISH 0x100000u
#define PHASE_BALANCED_PREPARE 0x200000u
#define PHASE_BALANCED_ASSEMBLED 0x400000u
#define PHASE_BALANCED_REDUCE 0x800000u
#define PHASE_BALANCED_HEAD_OUTPUT 0x1000000u
#define OP_BALANCED_SELECT 107u

/* Output rows, not whole experts, are striped across all six dies. Every
 * selected expert has the same local row count, independent of its ID. For
 * 640 rows this is104/112 rows (within5% of the mean);32-row stripes can be
 * 96/128 rows and miss the per-phase10% requirement. No weight is replicated. */
static inline unsigned balance_row_count(unsigned rows,unsigned die,unsigned dies,unsigned grain) {
    unsigned groups=rows/grain,tail=rows%grain;
    unsigned n=groups/dies+(die<groups%dies);
    return n*grain+(die==groups%dies?tail:0);
}
static inline unsigned balance_global_row(unsigned local,unsigned die,unsigned dies,unsigned grain) {
    return (local/grain*dies+die)*grain+local%grain;
}
static inline unsigned balance_local_row(unsigned row,unsigned dies,unsigned grain) {
    return row/(dies*grain)*grain+row%grain;
}
static inline unsigned balance_row_owner(unsigned row,unsigned dies,unsigned grain) {
    return row/grain%dies;
}
static inline unsigned balance_boundary(unsigned groups,unsigned die,unsigned dies) {
    return (unsigned)((uint64_t)groups*die/dies);
}
static inline unsigned balance_attention_first(unsigned die,unsigned dies) {
    return balance_boundary(24,die,dies);
}
static inline unsigned balance_attention_heads(unsigned die,unsigned dies) {
    return balance_attention_first(die+1,dies)-balance_attention_first(die,dies);
}
static inline unsigned balance_attention_kv_head(unsigned die,unsigned dies) {
    return balance_attention_first(die,dies)/12;
}

/* The holder and program builder must use this same policy. Nonexpert rows
 * remain contiguous, with32-column boundaries where activation quantization
 * requires complete groups. All routed experts and the shared expert are
 * present as row shards; global expert IDs remain unchanged. */
static inline void balance_split(split_t *s,unsigned die,unsigned dies) {
    unsigned cb0=balance_boundary(80,die,dies),cb1=balance_boundary(80,die+1,dies);
    s->d=(int)die;s->n=(int)dies;
    s->dj0=(int)balance_boundary(324,die,dies);
    s->djn=(int)balance_boundary(324,die+1,dies)-s->dj0;
    s->cb0=(int)cb0;s->cbn=(int)(cb1-cb0);
    s->h0=(int)balance_boundary(48,die,dies);
    s->hn=(int)balance_boundary(48,die+1,dies)-s->h0;
    s->klo=s->h0%16;s->nk=s->hn<16?s->hn:16;s->khi=(s->klo+s->nk-1)%16;
    s->y0=(int)(32*cb0);s->yn=(int)(32*(cb1-cb0));
    s->pk0=(int)(32*balance_boundary(320,die,dies));
    s->pkn=(int)(32*balance_boundary(320,die+1,dies))-s->pk0;
    s->o0=(int)(32*balance_boundary(7760,die,dies));
    s->on=(int)(32*balance_boundary(7760,die+1,dies))-s->o0;
    s->hd0=(int)balance_boundary(320,die,dies);
    s->hdn=(int)balance_boundary(320,die+1,dies)-s->hd0;
    s->e0=0;s->en=512;s->shexp=1;
}
#endif
