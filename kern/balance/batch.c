#include "batch.h"
#include "../exactbatch/prog_group.h"
#include "layout.h"
#include "../unique/layout.h"
#include "../dflash/target.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

int prog_balance_batch(prog_t *out,const prog_t *const *positions,
                       unsigned count,balance_batch_work *work,CUdeviceptr local_quant_ready) {
    if(!out || !positions || count<2 || count>BALANCED_BATCH_MAX)return -1;
    const prog_t *first=positions[0];
    if(!first || first->n<=0 || !first->p || first->xr || first->nxr ||
       !first->loop || first->token_offset>UINT_MAX-count+1)return -1;
    for(unsigned p=0;p<count;p++) {
        const prog_t *q=positions[p];
        if(!q || q==out || !q->p || q->n!=first->n || q->xr || q->nxr ||
           q->loop!=first->loop || q->xj!=first->xj ||
           q->pending_wait!=first->pending_wait ||
           q->token_offset!=first->token_offset+p ||
           q->end!=first->end || q->samp_k!=first->samp_k ||
           q->samp_topp!=first->samp_topp || q->samp_invT!=first->samp_invT ||
           memcmp(q->stop,first->stop,sizeof q->stop))return -1;
    }
    balance_batch_work stats={.positions=count};
    const char *strict_env=getenv("FLASH_BALANCED_BATCH_STRICT");
    int strict=strict_env && atoi(strict_env)!=0;
    const char *hcup_env=getenv("FLASH_BALANCED_HCUP5");
    int hcup5=hcup_env && atoi(hcup_env)!=0 && count==5;
    const char *hcup4_env=getenv("FLASH_BALANCED_HCUP4");
    const char *proposer=getenv("FLASH_BALANCED_SPECULATE");
    int hcup4=count==4 && ((hcup4_env && atoi(hcup4_env)) || (proposer && !strcmp(proposer,"dflash2")));
    int native_hcup=hcup5 || hcup4;
    const char *localq_env=getenv("FLASH_BALANCED_HCUP5_LOCALQ");
    int localq=localq_env && atoi(localq_env)!=0 && hcup5;
    if(localq && !local_quant_ready)return -1;
    if(native_hcup && strict)return -1;
    int preceding_hcup=0;
    for(int i=0;i<first->n;) {
        const phase_t *a=first->p+i;
        unsigned span=phase_descriptors(a);
        if(!span || span>(unsigned)(first->n-i) || 1+span*count>255 ||
           (a->w[0]>=OP_BALANCED_BATCH && a->w[0]!=OP_UNIQUE_ROUTER && a->w[0]!=OP_DFLASH_FEATURES))return -1;
        for(unsigned p=1;p<count;p++)for(unsigned j=0;j<span;j++) {
            const phase_t *x=a+j,*y=positions[p]->p+i+j;
            if(x->w[0]!=y->w[0] || x->w[1]!=y->w[1] ||
               x->w[30]!=y->w[30] || x->w[31]!=y->w[31])return -1;
        }
        if(native_hcup && a->w[0]==96u) {
            if(span!=1 || a->w[1]!=16 || a->w[31] || !a->w[8] || a->w[8]>2048 || a->w[8]%128 ||
               (a->w[20]!=320 && a->w[20]!=324))return -1;
            for(unsigned p=0;p<count;p++) {
                const phase_t *x=positions[p]->p+i;
                if(x->w[2]!=a->w[2] || x->w[3]!=a->w[3] || x->w[8]!=a->w[8] ||
                   x->w[12]!=a->w[12] || x->w[20]!=a->w[20])return -1;
            }
            stats.native_hcup_phases++;
        }
        stats.outer_phases++;
        if(hcup4 && a->w[0]==OP_DNSTEP && span==1 && (a->w[24]|a->w[25]) && (a->w[26]|a->w[27]))
            stats.native_dn_phases++;
        if(localq && preceding_hcup) {
            if(!(a->w[30]&2u) || a->w[0]==96u)return -1;
            stats.local_quant_phases++;stats.inner_grid_joins++;
        }
        stats.causal_phase_replays+=count;
        stats.publications+=(a->w[30]&1u)!=0;
        stats.peer_acquires+=(a->w[30]&2u)!=0;
        /* QSA owns shared rolling indexer state and possibly aliased gather
         * scratch. Every CTA must finish one position before another starts. */
        if(strict || a->w[0]==OP_BALANCED_SELECT)stats.inner_grid_joins+=count;
        preceding_hcup=native_hcup && a->w[0]==96u;
        i+=(int)span;
    }
    if(localq && preceding_hcup)return -1;
    size_t n=(size_t)first->n*count+stats.outer_phases;
    if(n>INT_MAX || n>SIZE_MAX/sizeof(phase_t))return -2;
    phase_t *image=malloc(n*sizeof *image);if(!image)return -2;
    size_t at=0;preceding_hcup=0;
    for(int i=0;i<first->n;) {
        const phase_t *a=first->p+i;
        unsigned span=phase_descriptors(a);
        phase_t *wrap=image+at++;memset(wrap,0,sizeof *wrap);
        wrap->w[0]=OP_BALANCED_BATCH;wrap->w[1]=16;
        if(native_hcup && a->w[0]==96u)wrap->w[0]=OP_BALANCED_HCUP5;
        if(hcup4 && a->w[0]==OP_DNSTEP && span==1 && (a->w[24]|a->w[25]) && (a->w[26]|a->w[27]))
            wrap->w[0]=OP_DFLASH_DN4;
        if(localq && preceding_hcup) {
            wrap->w[0]=OP_BALANCED_PREQUANT;
            wrap->w[6]=(uint32_t)local_quant_ready;wrap->w[7]=(uint32_t)(local_quant_ready>>32);
        }
        if(localq && a->w[0]==96u)wrap->w[6]=1;
        wrap->w[2]=count;wrap->w[3]=span;wrap->w[4]=a->w[0];
        wrap->w[5]=strict || a->w[0]==OP_BALANCED_SELECT;
        wrap->w[30]=(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|
                       PHASE_CHAIN_FLAG|PHASE_WIDTH_MASK))|
                       PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|((1+count*span)<<8);
        for(unsigned p=0;p<count;p++) {
            memcpy(image+at,positions[p]->p+i,span*sizeof *image);
            /* All positions consume the wrapper's one incoming acquire and
             * publish together at its final system release. Native child
             * chains and their internal dependencies remain untouched. */
            image[at].w[30]=(image[at].w[30]&~3u)|4u;
            if(strict || a->w[0]==OP_BALANCED_SELECT)
                image[at].w[30]&=~(4u|0x20000u);
            at+=span;
        }
        preceding_hcup=native_hcup && a->w[0]==96u;i+=(int)span;
    }
    if(at!=n){free(image);return -3;}
    free(out->p);free(out->xr);*out=*first;
    out->p=image;out->n=out->cap=(int)n;out->xr=NULL;out->nxr=out->xrcap=0;
    if(work)*work=stats;
    return 0;
}
