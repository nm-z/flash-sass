#ifndef FLASH_UNIQUE_LAYOUT_H
#define FLASH_UNIQUE_LAYOUT_H
#include "../balance/layout.h"

/* Scalar placement whose matrix rows have one owner. GDN value heads sharing
 * a key head stay together; publishing restores the model's global head order.
 * The distinct holder/module markers keep this layout separate from the older
 * replicated-Q/K balanced experiment. */
#define FLASH_UNIQUE_LAYOUT_MARKER "__flash_unique_rows_v1"
#define FLASH_PAIRED_LAYOUT_MARKER "__flash_paired_columns_v1"
#define PHASE_UNIQUE_ROUTE 0x4000000u
#define OP_UNIQUE_ROUTER 111u
#define OP_PAIRED_DOWN 112u
#define OP_PAIRED_DOWN4 123u
#define UNIQUE_ROUTE_PACKET_WORDS 32u
#define UNIQUE_ROUTE_CANDIDATES 10u

/* Ten indivisible 64-channel quantization blocks. Rank two stays light to
 * respect the existing fenced die; the second light rank rotates per layer.
 * Gate/up rows and down columns use this identical contiguous assignment. */
static inline unsigned paired_light_rank(unsigned layer) {
    static const unsigned ranks[5]={0,1,3,4,5};return ranks[layer%5];
}
static inline unsigned paired_blocks(unsigned die,unsigned layer) {
    return die==2 || die==paired_light_rank(layer)?1:2;
}
static inline unsigned paired_first_block(unsigned die,unsigned layer) {
    unsigned first=0;for(unsigned d=0;d<die;d++)first+=paired_blocks(d,layer);return first;
}
static inline unsigned paired_channels(unsigned die,unsigned layer) {
    return 64*paired_blocks(die,layer);
}

static inline unsigned unique_key_first(unsigned die,unsigned dies) {
    return balance_boundary(16,die,dies);
}
static inline unsigned unique_key_count(unsigned die,unsigned dies) {
    return unique_key_first(die+1,dies)-unique_key_first(die,dies);
}
static inline unsigned unique_value_head(unsigned local,unsigned die,unsigned dies) {
    unsigned keys=unique_key_count(die,dies);
    return unique_key_first(die,dies)+local%keys+16*(local/keys);
}
static inline unsigned unique_router_first(unsigned die,unsigned dies) {
    return balance_boundary(513,die,dies);
}
static inline unsigned unique_router_rows(unsigned die,unsigned dies) {
    return unique_router_first(die+1,dies)-unique_router_first(die,dies);
}
static inline void unique_split(split_t *s,unsigned die,unsigned dies) {
    balance_split(s,die,dies);
    s->klo=(int)unique_key_first(die,dies);
    s->nk=(int)unique_key_count(die,dies);s->khi=s->klo+s->nk-1;
    s->h0=s->klo;s->hn=3*s->nk;
}
#endif
