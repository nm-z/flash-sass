#include "batch.h"
#include "../exactbatch/prog_group.h"
#include "layout.h"
#include "../unique/layout.h"
#include "../dflash/target.h"
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static unsigned native_matrix(const phase_t *a,unsigned span,unsigned *k) {
    if(span!=1 || a->w[31] || (a->w[30]&PHASE_BALANCED_ROW8 && a->w[23]!=1))return 0;
    switch(a->w[0]) {
    case OP_Q4KK2560:*k=2560;return OP_DFLASH_Q4K4;
    case OP_Q4KK6144:*k=6144;return OP_DFLASH_Q4K4;
    case OP_Q5KK2560:*k=2560;return OP_DFLASH_Q5K4;
    case OP_Q5KK6144:*k=6144;return OP_DFLASH_Q5K4;
    case OP_Q2K2560:*k=2560;return OP_DFLASH_Q2_4;
    case OP_Q2K640:*k=640;return OP_DFLASH_Q2_4;
    case OP_Q3KK2560:*k=2560;return OP_DFLASH_Q3K4;
    case OP_Q3KK6144:*k=6144;return OP_DFLASH_Q3K4;
    case OP_Q6KK2560:*k=2560;return OP_DFLASH_Q6K4;
    case OP_Q6KK1536:case OP_Q6KK1536ACC:*k=1536;return OP_DFLASH_Q6K4;
    case OP_IQ4K2560:*k=2560;return OP_DFLASH_IQ4_4;
    case OP_IQ4K6144:*k=6144;return OP_DFLASH_IQ4_4;
    case OP_BF16BIAS:case OP_BF16K2560:*k=2560;return OP_DFLASH_BF16_4;
    case OP_BF16K320:*k=320;return OP_DFLASH_BF16_4;
    default:return 0;
    }
}

/* Generic children of one wrapper run back to back without a grid join. A
 * recurrent operation whose position p+1 reads state another CTA wrote for
 * position p (or a retained bank an earlier position still reads) must join
 * between positions. FLASH_B4_JOIN_OPS overrides the default opcode set. */
static int position_join(unsigned op) {
    static unsigned ops[32],n;static int loaded;
    if(!loaded) {
        const char *list=getenv("FLASH_B4_JOIN_OPS");
        /* conv1d, dnstep, pleconv, dnnormq_x, dnstep_normq, dflash_features:
         * with these joined, all-generic B4 logits are bit-identical to scalar. */
        if(!list)list="15,16,24,95,98,113";
        for(const char *p=list;*p && n<32;) {
            char *end;unsigned long v=strtoul(p,&end,0);if(end==p)break;
            ops[n++]=(unsigned)v;p=*end==','?end+1:end;
        }
        loaded=1;
    }
    if(op==OP_BALANCED_SELECT)return 1;
    for(unsigned i=0;i<n;i++)if(ops[i]==op)return 1;
    return 0;
}
static int balance_batch_impl(prog_t *out,const prog_t *const *positions,
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
    /* FLASH_B4_GENERIC: diagnostic bitmask forcing generic per-position
     * replay for a native B4 family (1 matrix, 2 hcup, 4 dn, 8 gr, 16 moe,
     * 32 router; 64 skips bundle transposition). */
    const char *generic_env=getenv("FLASH_B4_GENERIC");
    unsigned generic=generic_env?(unsigned)strtoul(generic_env,NULL,0):0u;
    int nat_matrix=hcup4 && !(generic&1u),nat_dn=hcup4 && !(generic&4u),nat_gr=hcup4 && !(generic&8u);
    int nat_moe=hcup4 && !(generic&16u),nat_router=hcup4 && !(generic&32u);
    if(hcup4 && (generic&2u))native_hcup=0;
    if(hcup4 && !local_quant_ready)return -1;
    const char *localq_env=getenv("FLASH_BALANCED_HCUP5_LOCALQ");
    int localq=localq_env && atoi(localq_env)!=0 && hcup5;
    if(localq && !local_quant_ready)return -1;
    if(native_hcup && strict)return -1;
    int preceding_hcup=0;
    for(int i=0;i<first->n;) {
        const phase_t *a=first->p+i;
        unsigned span=phase_descriptors(a);
        if(!span || span>(unsigned)(first->n-i) || 1+span*count>255 ||
           (a->w[0]>=OP_BALANCED_BATCH && a->w[0]!=OP_UNIQUE_ROUTER && a->w[0]!=OP_PAIRED_DOWN && a->w[0]!=OP_DFLASH_FEATURES))return -1;
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
        unsigned matrix_k=0,matrix_op=nat_matrix?native_matrix(a,span,&matrix_k):0;
        if(matrix_op) {
            for(unsigned p=1;p<count;p++) {
                const phase_t *q=positions[p]->p+i;
                unsigned rows_word=matrix_op==OP_DFLASH_BF16_4?8:10;
                if(q->w[2]!=a->w[2] || q->w[3]!=a->w[3] || q->w[rows_word]!=a->w[rows_word])return -1;
                if(a->w[0]==OP_BF16BIAS && (q->w[10]!=a->w[10] || q->w[11]!=a->w[11]))return -1;
            }
            stats.native_matrix_phases++;
        }
        if(nat_dn && a->w[0]==OP_DNSTEP && span==1 && (a->w[24]|a->w[25]) && (a->w[26]|a->w[27]))
            stats.native_dn_phases++;
        if(nat_moe && a->w[0]==OP_PAIRED_DOWN) {
            if(span!=1 || a->w[8]<1 || a->w[8]>2)return -1;
            stats.native_moe_phases++;stats.inner_grid_joins++;
        }
        if(nat_router && a->w[0]==OP_UNIQUE_ROUTER){if(span!=1)return -1;stats.native_matrix_phases++;}
        if(nat_gr && span==1 && (a->w[0]==OP_BF16HCRX || a->w[0]==OP_BF16HCRWX || a->w[0]==94u)) {
            if(!(a->w[30]&1u) || a->w[31] || a->w[1]!=16 || !a->w[8])return -1;
            for(unsigned p=1;p<count;p++) {
                const phase_t *q=positions[p]->p+i;
                if(q->w[2]!=a->w[2] || q->w[3]!=a->w[3] || q->w[8]!=a->w[8] || q->w[12]!=a->w[12])return -1;
            }
            stats.native_gr_phases++;stats.inner_grid_joins++;
        }
        if(localq && preceding_hcup) {
            if(!(a->w[30]&2u) || a->w[0]==96u)return -1;
            stats.local_quant_phases++;stats.inner_grid_joins++;
        }
        stats.causal_phase_replays+=count;
        stats.publications+=(a->w[30]&1u)!=0;
        stats.peer_acquires+=(a->w[30]&2u)!=0;
        /* QSA owns shared rolling indexer state and possibly aliased gather
         * scratch. Every CTA must finish one position before another starts. */
        if(strict || position_join(a->w[0]))stats.inner_grid_joins+=count;
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
        unsigned matrix_k=0,matrix_op=nat_matrix?native_matrix(a,span,&matrix_k):0;
        if(matrix_op) {
            wrap->w[0]=matrix_op;wrap->w[8]=matrix_k;
            wrap->w[9]=matrix_op==OP_DFLASH_Q2_4?((matrix_k/32+15u)&~15u)+matrix_k/4:
                (matrix_op==OP_DFLASH_Q4K4?144u:176u)*(matrix_k/256u);
            wrap->w[10]=matrix_k>=6144?5:matrix_k==640?2:4;
            wrap->w[12]=(a->w[30]&PHASE_BALANCED_ROW8)?3:0;
            if(matrix_op==OP_DFLASH_Q3K4 || matrix_op==OP_DFLASH_Q6K4 || matrix_op==OP_DFLASH_IQ4_4) {
                unsigned block=matrix_op==OP_DFLASH_Q3K4?112:matrix_op==OP_DFLASH_Q6K4?208:136;
                wrap->w[9]=(matrix_k>=6144?64:32)+block*(matrix_k/256);
                wrap->w[11]=a->w[0]==OP_Q6KK1536ACC;
            }
            if(matrix_op==OP_DFLASH_BF16_4) {
                wrap->w[9]=matrix_k*2;wrap->w[10]=a->w[0]==OP_BF16BIAS;
                wrap->w[12]=matrix_k==320?3:5;
                wrap->w[11]=(a->w[30]&PHASE_BALANCED_ROW8)?3:8-wrap->w[12];
            }
        }
        if(native_hcup && a->w[0]==96u)wrap->w[0]=OP_BALANCED_HCUP5;
        if(nat_moe && a->w[0]==OP_PAIRED_DOWN)wrap->w[0]=OP_PAIRED_DOWN4;
        if(nat_router && a->w[0]==OP_UNIQUE_ROUTER)wrap->w[0]=OP_DFLASH_ROUTER4;
        if(nat_dn && a->w[0]==OP_DNSTEP && span==1 && (a->w[24]|a->w[25]) && (a->w[26]|a->w[27]))
            wrap->w[0]=OP_DFLASH_DN4;
        if(nat_gr && span==1 && (a->w[0]==OP_BF16HCRX || a->w[0]==OP_BF16HCRWX || a->w[0]==94u)) {
            wrap->w[0]=OP_DFLASH_GR4;
            wrap->w[6]=(uint32_t)local_quant_ready;wrap->w[7]=(uint32_t)(local_quant_ready>>32);
        }
        if(localq && preceding_hcup) {
            wrap->w[0]=OP_BALANCED_PREQUANT;
            wrap->w[6]=(uint32_t)local_quant_ready;wrap->w[7]=(uint32_t)(local_quant_ready>>32);
        }
        if(localq && a->w[0]==96u)wrap->w[6]=1;
        wrap->w[2]=count;wrap->w[3]=span;wrap->w[4]=a->w[0];
        wrap->w[5]=strict || position_join(a->w[0]);
        wrap->w[30]=(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|
                       PHASE_CHAIN_FLAG|PHASE_WIDTH_MASK))|
                       PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|((1+count*span)<<8);
        for(unsigned p=0;p<count;p++) {
            memcpy(image+at,positions[p]->p+i,span*sizeof *image);
            /* All positions consume the wrapper's one incoming acquire and
             * publish together at its final system release. Native child
             * chains and their internal dependencies remain untouched. */
            image[at].w[30]=(image[at].w[30]&~3u)|4u;
            if(strict || position_join(a->w[0]))
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

/* Transpose independent matrix bundles into matrix-major B4 operations. The
 * old bundle executes (QKV, gate, alpha/beta) for one position at a time. Each
 * resulting operation can now reuse one matrix over all four positions. Only
 * the last member publishes the original exchange epoch; no peer handoff is
 * added. Dependent fused bodies and stateful composites retain their layout. */
static int independent_matrix(unsigned op) {
    switch(op) {
    case OP_BF16BIAS:case OP_BF16K2560:case OP_BF16K320:
    case OP_Q2K2560:case OP_Q2K640:case OP_Q2MOEGN:
    case OP_Q3KK2560:case OP_Q3KK6144:case OP_Q4KK2560:case OP_Q4KK6144:
    case OP_Q5KK2560:case OP_Q5KK6144:case OP_Q6KK2560:
    case OP_IQ4K2560:case OP_IQ4K6144:
    case OP_B32Q40:case OP_B32Q50:case OP_B32Q80:case OP_B32IQ4NL:return 1;
    default:return 0;
    }
}
static int transpose_matrix_bundles(prog_t *p) {
    for(int i=0;i<p->n;) {
        phase_t *a=p->p+i;unsigned span=phase_descriptors(a);
        if(!span || span>(unsigned)(p->n-i))return -1;
        int split=span>1 && !(a->w[30]&PHASE_FUSED_FLAG);
        for(unsigned j=0;j<span;j++)split&=independent_matrix(a[j].w[0]) && !a[j].w[31];
        int slices=span==4 && !(a->w[30]&PHASE_FUSED_FLAG) && a->w[0]==OP_Q6KK1536;
        for(unsigned j=0;j<span && slices;j++)slices&=a[j].w[0]==(j?OP_Q6KK1536ACC:OP_Q6KK1536) && !a[j].w[31];
        split|=slices;
        if(split) {
            unsigned flags=a->w[30],publish=flags&1u;
            for(unsigned j=0;j<span;j++)publish|=a[j].w[30]&1u;
            for(unsigned j=0;j<span;j++) {
                a[j].w[30]&=~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|PHASE_CHAIN_FLAG|
                               PHASE_CHAIN_BAR_FLAG|PHASE_WIDTH_MASK|0x80020007u);
                if(!j)a[j].w[30]|=flags&2u;
                a[j].w[30]|=j+1==span?publish|(flags&4u):slices?0u:4u;
            }
        }
        i+=(int)span;
    }
    return 0;
}
int prog_balance_batch(prog_t *out,const prog_t *const *positions,
                       unsigned count,balance_batch_work *work,CUdeviceptr ready) {
    const char *mode=getenv("FLASH_BALANCED_SPECULATE");
    if(count!=4 || !mode || strcmp(mode,"dflash2"))return balance_batch_impl(out,positions,count,work,ready);
    if(!out || !positions)return -1;
    prog_t copies[4]={{0}};const prog_t *inputs[4]={copies,copies+1,copies+2,copies+3};int result=-1;
    for(unsigned p=0;p<4;p++) {
        if(!positions[p] || !positions[p]->p || positions[p]->n<=0 || positions[p]->xr)goto done;
        copies[p]=*positions[p];copies[p].p=malloc((size_t)copies[p].n*sizeof(phase_t));
        if(!copies[p].p){result=-2;goto done;}
        memcpy(copies[p].p,positions[p]->p,(size_t)copies[p].n*sizeof(phase_t));
        const char *generic_env=getenv("FLASH_B4_GENERIC");
        if(!(generic_env && (strtoul(generic_env,NULL,0)&64u)) && transpose_matrix_bundles(copies+p))goto done;
    }
    result=balance_batch_impl(out,inputs,count,work,ready);
done:
    for(unsigned p=0;p<4;p++)free(copies[p].p);
    return result;
}
