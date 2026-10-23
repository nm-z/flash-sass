#ifndef FLASH_PAIRED_COLUMNS_H
#define FLASH_PAIRED_COLUMNS_H
#include "gguf.h"
#include <stddef.h>
#include <stdint.h>
#include <string.h>

/* One local 64-channel block contains all output rows. Its scales are a
 * separate plane, followed by naturally aligned code rows. This keeps Q2 at
 * exactly 18 bytes/output-row/block without a padded 32/48-byte row stride.
 * The manifest's stride is a byte count per output row, not an AoS stride. */
static inline unsigned paired_scale_bytes(uint32_t type) {return type==GG_Q2_0?2:4;}
static inline unsigned paired_code_bytes(uint32_t type) {
    switch(type) {
    case GG_Q2_0:return 16;
    case GG_Q4_0:case GG_IQ4_NL:return 32;
    case GG_Q5_0:return 40;
    case GG_Q8_0:return 64;
    default:return 0;
    }
}
static inline unsigned paired_record_bytes(uint32_t type) {
    unsigned codes=paired_code_bytes(type);return codes?paired_scale_bytes(type)+codes:0;
}
static inline void paired_pack_down(uint32_t type,const uint8_t *source,
                                    unsigned source_block,unsigned row,unsigned rows,
                                    uint8_t *destination) {
    unsigned scales=paired_scale_bytes(type),codes=paired_code_bytes(type);
    uint8_t *scale=destination+(size_t)row*scales;
    uint8_t *code=destination+(size_t)rows*scales+(size_t)row*codes;
    if(type==GG_Q2_0) {
        memcpy(scale,source+2*source_block,2);
        memcpy(code,source+32+16*source_block,16);
    } else {
        unsigned group=2*source_block;
        memcpy(scale,source+2*group,4);
        if(type==GG_Q8_0)memcpy(code,source+48+32*group,64);
        else {
            if(type==GG_Q5_0)code=destination+(size_t)rows*4+(size_t)row*32;
            memcpy(code,source+48+16*group,32);
            if(type==GG_Q5_0)memcpy(destination+(size_t)rows*36+(size_t)row*8,
                                   source+48+20*16+4*group,8);
        }
    }
}
#endif
