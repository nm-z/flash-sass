/* Imported scalar-compatible grouped-phase builders from the earlier B8 implementation. */
#include "prog_group.h"
#include "quant.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <limits.h>
int prog_verify_copy(prog_t *p,CUdeviceptr source,CUdeviceptr destination,size_t bytes) {
    if(!p || !source || !destination || !bytes || (bytes&15u) || bytes/16>UINT32_MAX) return -1;
    phase_t *a=prog_add(p,OP_VERIFY_COPY,(unsigned)((bytes+4095)/4096));
    ph_ptr(a,0x140,source);ph_ptr(a,0x148,destination);ph_u32(a,0x150,(uint32_t)(bytes/16));return 0;
}
/* Return a strictly validated position-major region, or one grouped phase.
 * Every region retains grid barriers. Its first operation may consume a prior
 * publication, but no operation publishes or advances the exchange count. */
static int serial_span(const prog_t *p, int i, unsigned count, int aliased) {
    static const int indexer[]={OP_IDXQK,OP_IDXSCORE,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE,OP_VERIFY_COPY};
    static const int ple[]={OP_PLEGATE,OP_RMS2560,OP_PLECONV,OP_VERIFY_COPY};
    static const int conv[]={OP_CONV1D,OP_VERIFY_COPY};
    static const int compact[]={OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE};
    /* a request whose positions never exceed the block budget has no selection passes (prog.h qsa_tmax) */
    static const int unselected[]={OP_IDXQK,OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE,OP_VERIFY_COPY};
    const int *ops=NULL;int n=1;
    if(aliased && p->p[i].w[0]==OP_IDXQK && i+1<p->n && p->p[i+1].w[0]==OP_KVGATHER){ops=unselected;n=4;}
    else if(aliased && p->p[i].w[0]==OP_IDXQK){ops=indexer;n=10;}
    else if(aliased && p->p[i].w[0]==OP_PLEGATE){ops=ple;n=3;}
    else if((aliased || count>2) && p->p[i].w[0]==OP_KVGATHER){ops=compact;n=3;}
    else if(p->p[i].w[0]==OP_CONV1D && i+1<p->n && p->p[i+1].w[0]==OP_VERIFY_COPY){ops=conv;n=2;}
    if(!ops)return 1;
    if((ops==ple || ops==indexer || ops==unselected) && i+n<p->n && p->p[i+n].w[0]==OP_VERIFY_COPY)n++;
    if(i+n>p->n)return -1;
    for(int j=0;j<n;j++) {
        if(p->p[i+j].w[0]!=(unsigned)ops[j] ||
            (p->p[i+j].w[30] & ~((ops==ple && j==0)?2u:0u)))return -1;
    }
    /* Selection passes must retain their increasing radix pass parameters. */
    if(ops==indexer)for(int j=2;j<7;j++)if(p->p[i+j].w[10]!=(unsigned)j)return -1;
    return n;
}
static int prog_group_impl(prog_t *out, const prog_t *const *positions, unsigned count, int aliased) {
    if (!out || !positions || count < 2 || count > PROG_MAX_POSITIONS) return -1;
    const prog_t *a = positions[0];
    if (!a || a->n <= 0 || !a->p || a->xr || a->nxr || a->n > INT_MAX / (int)count ||
        (a->loop && a->token_offset > UINT32_MAX - (count - 1u))) return -1;
    /* Validate every position before allocating or changing out. */
    for (unsigned k = 0; k < count; k++) {
        const prog_t *b = positions[k];
        if (!b || out == b || a->n != b->n || !b->p || b->xr || b->nxr || a->xj != b->xj ||
            a->loop != b->loop || a->pending_wait != b->pending_wait ||
            a->stop[0] != b->stop[0] || a->stop[1] != b->stop[1] || a->end != b->end ||
            a->samp_k != b->samp_k || a->samp_topp != b->samp_topp ||
            a->samp_invT != b->samp_invT ||
            (a->loop && b->token_offset != a->token_offset + k)) return -1;
        for (int i = 0; i < a->n; i++) {
            const phase_t *x = &a->p[i], *y = &b->p[i];
            if (x->w[0] != y->w[0] || x->w[1] != y->w[1] || x->w[30] != y->w[30] ||
                (x->w[30] & (PHASE_GROUP_FLAG | PHASE_FUSED_FLAG | PHASE_WIDTH_MASK))) return -1;
        }
    }
    /* Validate every shared-lifetime region before allocation. No region may
     * publish or skip a grid barrier; the PLE region may consume its incoming
     * publication. Reject changed shapes rather than reuse live scratch. */
    for (int i = 0; i < a->n; ) {
        int span=serial_span(a,i,count,aliased);
        if(span<0)return -1;
        for(unsigned k=1;k<count;k++)if(serial_span(positions[k],i,count,aliased)!=span)return -1;
        i+=span;
    }
    size_t n = (size_t)a->n * count;
    if (n > SIZE_MAX / sizeof(phase_t)) return -2;
    phase_t *p = malloc(n * sizeof *p);
    if (!p) return -2;
    size_t next = 0;
    for (int i = 0; i < a->n; ) {
        int span=serial_span(a,i,count,aliased);
        if (span > 1) {
            for (unsigned k = 0; k < count; k++)
                for (int j = 0; j < span; j++) p[next++] = positions[k]->p[i + j];
            i += span;
        } else {
            size_t primary = next;
            for (unsigned k = 0; k < count; k++) p[next++] = positions[k]->p[i];
            p[primary].w[30] |= PHASE_GROUP_FLAG;
            if (count != 2) p[primary].w[30] |= count << PHASE_WIDTH_SHIFT;
            i++;
        }
    }
    free(out->p); free(out->xr);
    *out = *a;
    out->p = p; out->n = out->cap = (int)n;
    return 0;
}
int prog_group(prog_t *out, const prog_t *const *positions, unsigned count) { return prog_group_impl(out,positions,count,0); }
int prog_group_aliased(prog_t *out, const prog_t *const *positions, unsigned count) { return prog_group_impl(out,positions,count,1); }
static uint64_t desc_ptr(const phase_t *p, int word) {
    return p->w[word] | ((uint64_t)p->w[word + 1] << 32);
}
static int overlaps(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) {
    return an && bn && (a <= b ? b - a < an : a - b < bn);
}
static int prog_fuse_bf16(prog_t *P, unsigned width, int disjoint_xout) {
    int fused = 0;
    for (int i = 0; i + 1 < P->n; ) {
        phase_t *a = &P->p[i];
        unsigned count = phase_positions(a);
        if (count > (unsigned)(P->n - i)) break;
        i += (int)count;
        if ((width==8 ? (count!=8 && count!=16 && count!=32) : count!=width) || (a->w[30] & 24u) != 8u) continue;
        int compatible = !a->w[31];
        for (unsigned j = 1; j < count; j++) {
            const phase_t *other = a + j;
            if (a->w[0] != other->w[0] || a->w[1] != other->w[1] ||
                desc_ptr(a, 2) != desc_ptr(other, 2) || a->w[8] != other->w[8] ||
                other->w[30] != (a->w[30] & ~(PHASE_GROUP_FLAG | PHASE_FUSED_FLAG | PHASE_WIDTH_MASK)) ||
                other->w[31]) compatible = 0;
        }
        if (!compatible) continue;
        int op, k = 2560, slices = 1, bias = 0, xout = 0;
        switch (a->w[0]) {
        case OP_BF16BIAS: op = width == 8 ? OP_BF16BIAS8 : OP_BF16BIASP; bias = 1; break;
        case OP_BF16K2560: op = width == 8 ? OP_BF16K25608 : OP_BF16K2560P; break;
        case OP_BF16K320: op = width == 8 ? OP_BF16K3208 : OP_BF16K320P; k = 320; break;
        case OP_BF16HC: op = width == 8 ? OP_BF16HC8 : OP_BF16HCP; slices = 4; break;
        case OP_BF16HCX: op = width == 8 ? OP_BF16HCX8 : OP_BF16HCXP; slices = 4; xout = 1; break;
        default: continue;
        }
        if (xout && !disjoint_xout) continue;
        if (!xout) {
            uint64_t yn = 4ull * a->w[8], xn = 4ull * k * slices;
            uint64_t wn = 2ull * k * slices * a->w[8];
            int alias = 0;
            for (unsigned j = 0; j < count; j++) {
                uint64_t y = desc_ptr(a + j, 6);
                alias |= overlaps(y, yn, desc_ptr(a, 2), wn);
                for (unsigned kpos = 0; kpos < count; kpos++) {
                    if (j != kpos) alias |= overlaps(y, yn, desc_ptr(a + kpos, 6), yn);
                    alias |= overlaps(y, yn, desc_ptr(a + kpos, 4), xn);
                    if (bias) alias |= overlaps(y, yn, desc_ptr(a + kpos, 10), yn);
                }
            }
            if (alias) continue;
        }
        if(width==8) {
            uint32_t flags=a->w[30] & ~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|PHASE_WIDTH_MASK);
            for(unsigned first=0;first<count;first+=8) {
                uint32_t part_flags=flags;
                if(first)part_flags&=~2u; /* only the initial subgroup waits */
                if(first+8<count)part_flags=(part_flags&~1u)|4u;
                a[first].w[0]=(uint32_t)op;
                a[first].w[30]=part_flags|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
                fused++;
            }
        } else { a->w[0] = (uint32_t)op; a->w[30] |= 16u; fused++; }
    }
    return fused;
}
int prog_batch_fuse_bf16(prog_t *P, int disjoint_xout) { return prog_fuse_bf16(P, 8, disjoint_xout); }
int prog_batch_fuse_q2(prog_t *P, int routed_disjoint) {
    int fused=0;
    for (int i=0;i<P->n;) {
        phase_t *a=&P->p[i];unsigned width=phase_positions(a);
        if (!width || width>(unsigned)(P->n-i)) break;
        i+=(int)width;
        if ((width!=8 && width!=16 && width!=32) || (a->w[30]&24u)!=8u || a->w[31] || !a->w[1]) continue;
        int op=0,k=0,moe=0,acc=0;
        switch(a->w[0]) {
        case OP_Q2K2560: op=OP_Q2K25608;k=2560;break;
        case OP_Q2K640: op=OP_Q2K6408;k=640;break;
        case OP_Q2ACC640: op=OP_Q2ACC6408;k=640;acc=1;break;
        case OP_Q2MOEGN: op=OP_Q2MOEGN8;k=2560;moe=1;break;
        case OP_Q2MOEDN: op=OP_Q2MOEDN8;k=640;moe=2;break;
        default: continue;
        }
        if (!a->w[10] || (moe && (!routed_disjoint || a->w[10]!=(moe==1?640u:2560u)))) continue;
        int valid=1;
        for (unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if (b->w[0]!=a->w[0] || b->w[1]!=a->w[1] || b->w[10]!=a->w[10] || b->w[31] ||
                b->w[30]!=(p?(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|PHASE_WIDTH_MASK)):a->w[30]) ||
                desc_ptr(b,2)!=desc_ptr(a,2) || !desc_ptr(b,4) || !desc_ptr(b,6) ||
                !desc_ptr(b,8) || !desc_ptr(b,14) || (moe&&(!desc_ptr(b,12)||!desc_ptr(b,18))) ||
                ((acc||moe==2)&&!desc_ptr(b,16))) valid=0;
        }
        uint64_t yn=4ull*a->w[10]*(moe==1?10u:1u);
        uint64_t xn=2ull*(unsigned)k*(moe==2?10u:1u),sn=2*xn/32;
        uint64_t wn=(uint64_t)a->w[10]*(unsigned)q2_0_tile_stride(k);
        for (unsigned p=0;p<width;p++) for (unsigned q=0;q<width;q++) {
            uint64_t y=desc_ptr(a+p,8);
            valid&=p==q || !overlaps(y,yn,desc_ptr(a+q,8),yn);
            valid&=!overlaps(y,yn,desc_ptr(a+q,4),xn);
            valid&=!overlaps(y,yn,desc_ptr(a+q,6),sn);
            valid&=!overlaps(y,yn,desc_ptr(a+q,14),sn);
            if(moe){valid&=!overlaps(y,yn,desc_ptr(a+q,12),40);valid&=!overlaps(y,yn,desc_ptr(a+q,18),4);}
            if(acc||moe==2) valid&=!overlaps(y,yn,desc_ptr(a+q,16),moe==2?40:4);
            if(!moe) valid&=!overlaps(y,yn,desc_ptr(a,2),wn);
        }
        if(valid){
            uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|PHASE_WIDTH_MASK);
            for(unsigned first=0;first<width;first+=8){
                uint32_t part=flags;
                if(first)part&=~2u;
                if(first+8<width)part=(part&~1u)|4u;
                a[first].w[0]=(uint32_t)op;
                a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
                fused++;
            }
        }
    }
    return fused;
}
int prog_batch_fuse_q3(prog_t *P) {
    int fused=0;
    if(!P || P->n<0 || (P->n && !P->p))return 0;
    for(int i=0;i<P->n;) {
        phase_t *a=P->p+i;unsigned width=phase_positions(a);
        if(width>(unsigned)(P->n-i))break;
        i+=(int)width;
        if((width!=8 && width!=16 && width!=32) ||
            (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
            (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|7u)) ||
            a->w[31] || !a->w[1] || !a->w[10])continue;
        unsigned k,op;
        if(a->w[0]==OP_Q3KK2560){k=2560;op=OP_Q3KK25608;}
        else if(a->w[0]==OP_Q3KK6144){k=6144;op=OP_Q3KK61448;}
        else continue;
        int valid=1;
        uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(b->w[0]!=a->w[0] || b->w[1]!=a->w[1] || b->w[10]!=a->w[10] ||
                b->w[31] || b->w[30]!=(p?flags:a->w[30]) ||
                !desc_ptr(b,2) || desc_ptr(b,2)!=desc_ptr(a,2) ||
                !desc_ptr(b,4) || !desc_ptr(b,8) || !desc_ptr(b,14))valid=0;
        }
        uint64_t yn=4ull*a->w[10],xn=2ull*k,sn=4ull*(k/32);
        uint64_t wn=(uint64_t)a->w[10]*(unsigned)q3_k_tile_stride((int)k);
        for(unsigned p=0;p<width;p++)for(unsigned q=0;q<width;q++) {
            uint64_t y=desc_ptr(a+p,8);
            valid&=p==q || !overlaps(y,yn,desc_ptr(a+q,8),yn);
            valid&=!overlaps(y,yn,desc_ptr(a+q,4),xn);
            valid&=!overlaps(y,yn,desc_ptr(a+q,14),sn);
            valid&=!overlaps(y,yn,desc_ptr(a,2),wn);
        }
        if(!valid)continue;
        for(unsigned first=0;first<width;first+=8) {
            uint32_t part=flags;
            if(first)part&=~2u;
            if(first+8<width)part=(part&~1u)|4u;
            a[first].w[0]=op;
            a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            fused++;
        }
    }
    return fused;
}
int prog_batch_fuse_hcup(prog_t *P, int disjoint_xout) {
    int fused=0;
    for(int i=0;i<P->n;) {
        phase_t *a=P->p+i;unsigned width=phase_positions(a);
        if(!width || width>(unsigned)(P->n-i))break;
        i+=(int)width;
        if(!disjoint_xout || (width!=8 && width!=16 && width!=32) ||
            a->w[0]!=OP_BF16HCUP || a->w[1]!=16 || a->w[31] ||
            (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
            (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|3u)) || !(a->w[30]&1u) ||
            !a->w[8] || a->w[8]>4u*2560u || a->w[8]%4u ||
            a->w[12]%4u || a->w[12]>=2u*XSLOT || a->w[12]%XSLOT+a->w[8]>XSLOT ||
            (a->w[20]!=320 && a->w[20]!=324))continue;
        int valid=1;
        uint64_t un=4ull*a->w[8],tn=4ull*a->w[20],wn=640ull*a->w[8];
        uint64_t xhn=4ull*(3u*2560u+a->w[8]/4u);
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(b->w[0]!=OP_BF16HCUP || b->w[1]!=16 || b->w[31] ||
                b->w[8]!=a->w[8] || b->w[12]!=a->w[12] || b->w[20]!=a->w[20] ||
                b->w[30]!=(p?(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK)):a->w[30]) ||
                !desc_ptr(b,2) || desc_ptr(b,2)!=desc_ptr(a,2) || !desc_ptr(b,4) ||
                !desc_ptr(b,6) || !desc_ptr(b,14) || !desc_ptr(b,16) || !desc_ptr(b,18))valid=0;
            for(unsigned q=0;q<width;q++) {
                const phase_t *c=a+q;
                uint64_t outputs[]={desc_ptr(b,6),desc_ptr(b,18)},sizes[]={un,tn};
                for(unsigned j=0;j<2;j++) {
                    valid&=!overlaps(outputs[j],sizes[j],desc_ptr(c,2),wn);
                    valid&=!overlaps(outputs[j],sizes[j],desc_ptr(c,4),tn);
                    valid&=!overlaps(outputs[j],sizes[j],desc_ptr(c,16),xhn);
                    valid&=!overlaps(outputs[j],sizes[j],desc_ptr(c,14),96);
                    if(p!=q)for(unsigned k=0;k<2;k++)valid&=!overlaps(outputs[j],sizes[j],desc_ptr(c,k?18:6),sizes[k]);
                }
                if(p!=q)valid&=!overlaps(desc_ptr(b,14),96,desc_ptr(c,14),96);
            }
            valid&=!overlaps(desc_ptr(b,6),un,desc_ptr(b,18),tn);
        }
        if(!valid)continue;
        uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
        for(unsigned first=0;first<width;first+=8) {
            uint32_t part=flags;
            if(first)part&=~2u;
            if(first+8<width)part=(part&~1u)|4u;
            a[first].w[0]=OP_BF16HCUP8;
            a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            fused++;
        }
    }
    return fused;
}
int prog_batch_fuse_dnstep(prog_t *P) {
    int fused=0;
    if(!P || P->n<0 || (P->n && !P->p))return 0;
    for(int i=0;i<P->n;) {
        phase_t *a=P->p+i;unsigned width=phase_positions(a);
        if(width>(unsigned)(P->n-i))break;
        i+=(int)width;
        if((width!=8 && width!=16 && width!=32) || a->w[0]!=OP_DNSTEP ||
            (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
            (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|7u)) ||
            a->w[31] || !a->w[1] || a->w[1]>48)continue;
        uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
        uint64_t state=desc_ptr(a,2),sa=desc_ptr(a,14);
        uint64_t state_n=65536ull*a->w[1],head_n=4ull*a->w[1];
        uint64_t out_n=512ull*a->w[1],key_n=512ull*(a->w[1]<16?a->w[1]:16u);
        int valid=state && sa && !(state&15u) && !(sa&3u);
        valid&=!overlaps(state,state_n,sa,head_n);
        const unsigned input_words[]={4,6,8,10,12};
        const uint64_t input_n[]={key_n,key_n,out_n,head_n,head_n};
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(b->w[0]!=OP_DNSTEP || b->w[1]!=a->w[1] || b->w[31] ||
                b->w[30]!=(p?flags:a->w[30]) || desc_ptr(b,2)!=state ||
                desc_ptr(b,14)!=sa || b->w[18]!=a->w[18] ||
                b->w[19]!=a->w[19] || b->w[20]!=a->w[20])valid=0;
            uint64_t out=desc_ptr(b,16);
            if(!out || (out&15u) || overlaps(out,out_n,state,state_n) ||
                overlaps(out,out_n,sa,head_n))valid=0;
            for(unsigned j=0;j<5;j++) {
                uint64_t input=desc_ptr(b,input_words[j]);
                if(!input || (input&(j<3?15u:3u)) ||
                    overlaps(state,state_n,input,input_n[j]))valid=0;
                for(unsigned q=0;q<width;q++)
                    if(overlaps(desc_ptr(a+q,16),out_n,input,input_n[j]))valid=0;
            }
            for(unsigned q=0;q<p;q++)
                if(overlaps(out,out_n,desc_ptr(a+q,16),out_n))valid=0;
            /* Optional verification state capture (word24/25). The checkpoint
             * module validates this ABI before a client requests captures. */
            uint64_t checkpoint=desc_ptr(b,24);
            if(checkpoint) {
                if((checkpoint&15u) || overlaps(checkpoint,state_n,state,state_n) ||
                    overlaps(checkpoint,state_n,sa,head_n))valid=0;
                for(unsigned q=0;q<width;q++) {
                    if(overlaps(checkpoint,state_n,desc_ptr(a+q,16),out_n))valid=0;
                    uint64_t other=desc_ptr(a+q,24);
                    if(q!=p && other && overlaps(checkpoint,state_n,other,state_n))valid=0;
                    for(unsigned j=0;j<5;j++)
                        if(overlaps(checkpoint,state_n,desc_ptr(a+q,input_words[j]),input_n[j]))valid=0;
                }
            }
        }
        if(!valid)continue;
        for(unsigned first=0;first<width;first+=8) {
            uint32_t part=flags;
            if(first)part&=~2u;
            if(first+8<width)part=(part&~1u)|4u;
            a[first].w[0]=OP_DNSTEP8;
            a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            fused++;
        }
    }
    return fused;
}
int prog_batch_fuse_actq(prog_t *P) {
    int fused=0;
    for(int i=0;i<P->n;) {
        phase_t *a=P->p+i;unsigned width=phase_positions(a);
        if(width>(unsigned)(P->n-i))break;
        i+=(int)width;
        if(a->w[0]!=OP_ACTQ256 || (width!=8 && width!=16 && width!=32) ||
            (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
            (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|7u)) || a->w[31] ||
            !a->w[1] || a->w[1]>65535 || a->w[12]!=256)continue;
        unsigned units=a->w[1];int valid=1;
        uint64_t bytes[]={512ull*units,32ull*units,64ull*units,32ull*units};
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(b->w[0]!=OP_ACTQ256 || b->w[1]!=units || b->w[12]!=256 || b->w[31] ||
                b->w[30]!=(p?(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK)):a->w[30]) || !desc_ptr(b,2))valid=0;
            for(unsigned j=0;j<4;j++) {
                uint64_t out=desc_ptr(b,4+2*j);if(!out)valid=0;
                for(unsigned q=0;q<width;q++) {
                    const phase_t *c=a+q;
                    valid&=!overlaps(out,bytes[j],desc_ptr(c,2),1024ull*units);
                    for(unsigned k=0;k<4;k++)if(p!=q || j!=k)
                        valid&=!overlaps(out,bytes[j],desc_ptr(c,4+2*k),bytes[k]);
                }
            }
        }
        if(!valid)continue;
        uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
        for(unsigned first=0;first<width;first+=8) {
            uint32_t part=flags;
            if(first)part&=~2u;
            if(first+8<width)part=(part&~1u)|4u;
            a[first].w[0]=OP_ACTQ2568;
            a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            fused++;
        }
    }
    return fused;
}
int prog_batch_fuse_rms(prog_t *P) {
    int fused=0;
    for(int i=0;i<P->n;) {
        phase_t *a=P->p+i;unsigned width=phase_positions(a);
        if(width>(unsigned)(P->n-i))break;
        i+=(int)width;
        if(a->w[0]!=OP_RMS2560 || (width!=8 && width!=16 && width!=32) ||
            (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
            (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|7u)) || a->w[31] ||
            !a->w[1] || a->w[1]>65535)continue;
        uint64_t bytes=10240ull*a->w[1];int valid=1;
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(b->w[0]!=OP_RMS2560 || b->w[1]!=a->w[1] || b->w[31] ||
                b->w[30]!=(p?(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK)):a->w[30]) ||
                !desc_ptr(b,2) || !desc_ptr(b,4) || !desc_ptr(b,6))valid=0;
            for(unsigned q=0;q<width;q++) {
                const phase_t *c=a+q;
                valid&=!overlaps(desc_ptr(b,6),bytes,desc_ptr(c,2),bytes);
                valid&=!overlaps(desc_ptr(b,6),bytes,desc_ptr(c,4),bytes);
                if(p!=q)valid&=!overlaps(desc_ptr(b,6),bytes,desc_ptr(c,6),bytes);
            }
        }
        if(!valid)continue;
        uint32_t flags=a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
        for(unsigned first=0;first<width;first+=8) {
            uint32_t part=flags;
            if(first)part&=~2u;
            if(first+8<width)part=(part&~1u)|4u;
            a[first].w[0]=OP_RMS25608;
            a[first].w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            fused++;
        }
    }
    return fused;
}
static unsigned hcr_parts(const phase_t *a,unsigned width,const CUdeviceptr *scratch,unsigned positions,int disjoint_xout) {
    int rw=a->w[0]==OP_BF16HCRWX;
    if(!disjoint_xout || (a->w[0]!=OP_BF16HCRX && !rw) ||
        (width!=8 && width!=16 && width!=32) || a->w[1]!=16 || a->w[31] ||
        (a->w[30]&(PHASE_GROUP_FLAG|PHASE_FUSED_FLAG))!=PHASE_GROUP_FLAG ||
        (a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK|3u)) || !(a->w[30]&1u) ||
        !a->w[8] || a->w[8]>324 || a->w[12]%4u || a->w[12]>=2u*XSLOT ||
        a->w[12]%XSLOT+4u*a->w[8]>XSLOT || (rw && (!scratch || positions<width)))return 0;
    uint64_t wn=20480ull*a->w[8];
    for(unsigned p=0;p<width;p++) {
        const phase_t *b=a+p;
        if(b->w[0]!=a->w[0] || b->w[1]!=16 || b->w[31] || b->w[8]!=a->w[8] ||
            b->w[12]!=a->w[12] || b->w[24]!=a->w[24] ||
            b->w[30]!=(p?(a->w[30]&~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK)):a->w[30]) ||
            !desc_ptr(b,2) || desc_ptr(b,2)!=desc_ptr(a,2) || !desc_ptr(b,4) ||
            !desc_ptr(b,6) || !desc_ptr(b,16) || !desc_ptr(b,22) ||
            desc_ptr(b,22)!=desc_ptr(a,22) ||
            (rw && (!desc_ptr(b,10) || !desc_ptr(b,26) || !desc_ptr(b,28) || !scratch[p])))return 0;
        uint64_t outputs[]={desc_ptr(b,16),rw?desc_ptr(b,10):0,rw?scratch[p]:0};
        uint64_t sizes[]={40960,40960,10240};unsigned nout=rw?3:1;
        for(unsigned q=0;q<width;q++) {
            const phase_t *c=a+q;
            uint64_t reads[]={desc_ptr(c,2),desc_ptr(c,4),desc_ptr(c,6),desc_ptr(c,22),rw?desc_ptr(c,26):0,rw?desc_ptr(c,28):0};
            uint64_t bytes[]={wn,40960,96,40960,10240,16};
            uint64_t other[]={desc_ptr(c,16),rw?desc_ptr(c,10):0,rw?scratch[q]:0};
            for(unsigned j=0;j<nout;j++) {
                for(unsigned r=0;r<(rw?6u:4u);r++)if(overlaps(outputs[j],sizes[j],reads[r],bytes[r]))return 0;
                for(unsigned k=0;k<nout;k++)if((p!=q || j!=k) && overlaps(outputs[j],sizes[j],other[k],sizes[k]))return 0;
            }
            if(p!=q && overlaps(desc_ptr(b,6),96,desc_ptr(c,6),96))return 0;
        }
    }
    return rw?3:2;
}
static void hcr_group_flags(phase_t *out,unsigned width,uint32_t flags) {
    for(unsigned p=0;p<width;p++)out[p].w[30]=flags|(p?0:PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT));
}
int prog_batch_decompose_hcr(prog_t *P,const CUdeviceptr *scratch,unsigned positions,int disjoint_xout) {
    if(!P || P->n<0 || (P->n && !P->p))return -1;
    size_t count=(size_t)P->n;int changed=0;
    for(int i=0;i<P->n;) {
        unsigned width=phase_positions(P->p+i);
        if(width>(unsigned)(P->n-i))return -1;
        unsigned parts=hcr_parts(P->p+i,width,scratch,positions,disjoint_xout);
        if(parts){count+=(parts-1u)*width;changed++;}
        i+=(int)width;
    }
    if(!changed)return 0;
    if(count>PROG_MAX_DESCRIPTORS || count>INT_MAX || count>SIZE_MAX/sizeof(phase_t))return -1;
    phase_t *expanded=calloc(count,sizeof *expanded);if(!expanded)return -1;
    size_t next=0;
    for(int i=0;i<P->n;) {
        const phase_t *a=P->p+i;unsigned width=phase_positions(a);
        unsigned parts=hcr_parts(a,width,scratch,positions,disjoint_xout);
        if(!parts){memcpy(expanded+next,a,width*sizeof *a);next+=width;i+=(int)width;continue;}
        phase_t *write=parts==3?expanded+next:NULL;
        phase_t *rms=expanded+next+(parts==3?width:0),*hc=rms+width;
        for(unsigned p=0;p<width;p++) {
            const phase_t *b=a+p;
            if(write) {
                write[p].w[0]=OP_HCWSUM;write[p].w[1]=16;
                ph_ptr(write+p,0x140,desc_ptr(b,28));ph_ptr(write+p,0x148,desc_ptr(b,26));ph_ptr(write+p,0x150,desc_ptr(b,10));
                ph_u32(write+p,0x158,2560);ph_u32(write+p,0x15c,10240);ph_u32(write+p,0x164,1);
                ph_ptr(write+p,0x168,scratch[p]);ph_ptr(write+p,0x170,desc_ptr(b,4));
            }
            rms[p].w[0]=OP_RMS2560;rms[p].w[1]=4;
            ph_ptr(rms+p,0x140,desc_ptr(b,write?10:4));ph_ptr(rms+p,0x148,desc_ptr(b,22));ph_ptr(rms+p,0x150,desc_ptr(b,16));rms[p].w[8]=b->w[24];
            hc[p].w[0]=OP_BF16HCX;hc[p].w[1]=16;
            ph_ptr(hc+p,0x140,desc_ptr(b,2));ph_ptr(hc+p,0x148,desc_ptr(b,16));ph_ptr(hc+p,0x150,desc_ptr(b,6));hc[p].w[8]=b->w[8];hc[p].w[12]=b->w[12];
        }
        uint32_t wait=a->w[30]&2u;
        if(write)hcr_group_flags(write,width,wait);
        hcr_group_flags(rms,width,write?0:wait);hcr_group_flags(hc,width,1);
        next+=parts*width;i+=(int)width;
    }
    free(P->p);P->p=expanded;P->n=P->cap=(int)count;return changed;
}
