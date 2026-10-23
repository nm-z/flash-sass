#ifndef FLASH_BALANCE_EMBEDDING_H
#define FLASH_BALANCE_EMBEDDING_H
#include <stdint.h>
#include <string.h>

/* One32-column Q3_K group, preserving the original integer codes, six-bit
 * scales and FP16 block multiplier. There is no requantization:
 * [8B low-two-bit codes][4B high mask][2B scales][2B multiplier].
 * The balanced SASS decoder must retain (d * (scale-32)) * signed_code. */
static inline void balance_pack_embedding_group(const uint8_t *row,unsigned first,uint8_t out[16]) {
    const uint8_t *block=row+(first/256)*110;
    uint32_t scales[4];memcpy(scales,block+96,12);
    const uint32_t low=0x0f0f0f0f,high=0x03030303;
    uint32_t tail=scales[2];
    scales[2]=((scales[0]>>4)&low)|(((tail>>4)&high)<<4);
    scales[3]=((scales[1]>>4)&low)|(((tail>>6)&high)<<4);
    scales[0]=(scales[0]&low)|(((tail>>0)&high)<<4);
    scales[1]=(scales[1]&low)|(((tail>>2)&high)<<4);
    const uint8_t *scale=(const uint8_t*)scales;
    uint64_t codes=0;uint32_t mask=0;
    for(unsigned lane=0;lane<32;lane++) {
        unsigned v=(first%256)+lane,nn=v>>7,j=(v>>5)&3,which=(v>>4)&1,l=v&15;
        unsigned q=(block[32+32*nn+16*which+l]>>(2*j))&3;
        unsigned h=(block[16*which+l]>>(4*nn+j))&1;
        codes|=(uint64_t)q<<(2*lane);mask|=h<<lane;
    }
    memcpy(out,&codes,8);memcpy(out+8,&mask,4);
    unsigned group=(first%256)/16;
    out[12]=scale[group];out[13]=scale[group+1];
    memcpy(out+14,block+108,2);
}
#endif
