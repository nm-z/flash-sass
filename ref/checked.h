/* Checked bounds wrappers. Return 0 on success, 1 on rejection; outputs are
 * unchanged on rejection. Attention decode/partials also reject invalid counts;
 * their other pointer and capacity preconditions remain caller duties.
 * This header defines dimensions, not the nonfinite/padding policies. */
#ifndef FLASH_CHECKED_H
#define FLASH_CHECKED_H
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#define QSA_CHECKED_MIN_BUDGET 1u
#define QSA_CHECKED_MAX_BUDGET 4096u
/* Strict decimal digits; no signs, whitespace, suffixes or overflow. */
static inline int checked_parse_unsigned(const char*s,unsigned lo,unsigned hi,unsigned*out){
 if(!s||!*s||!out||lo>hi)return 1;unsigned v=0;
 for(;*s;s++){if(*s<'0'||*s>'9')return 1;unsigned d=(unsigned)(*s-'0');if(v>(UINT_MAX-d)/10u)return 1;v=v*10u+d;}
 if(v<lo||v>hi)return 1;*out=v;return 0;
}
int attn_cache_sizes_checked(unsigned n,size_t*kc_floats,size_t*vc_floats);
/* qo/out:6144 floats; gate:12288. Sizes count floats, not bytes.
 * Scalar layout only; vector prefetch padding is a separate GPU contract. */
int attn_decode_checked(const float*qo,const float*kc,size_t kcf,const float*vc,size_t vcf,unsigned n,const float*gate,float*out);
/* position is zero based; capacity counts physically available positions. */
int qsa_nkeys_checked(unsigned position,unsigned capacity,unsigned budget,unsigned*nkeys);
/* Source K/V codes:capacity*512 bytes each; scales:capacity*16 uint16 each.
 * The unfinished 4-position source block must be fully allocated.
 * If complete blocks exceed budget, IDs must be strictly ascending, unique,
 * valid complete blocks; otherwise selection is ignored.
 * Destination sizes count floats and cover every physically gathered row. */
int qsa_gather_checked(const int8_t*kq,const uint16_t*ks,const int8_t*vq,const uint16_t*vs,unsigned capacity,const uint32_t*sel,size_t selection_count,unsigned position,unsigned budget,float*kc,size_t kcf,float*vc,size_t vcf,unsigned*nkeys);
#endif
