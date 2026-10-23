#include "prog.h"
#include "split.h"
#include "gguf.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>

void prog_init(prog_t *P) { P->cap = 64; P->n = 0; P->xj = 0; P->loop = 0; P->token_offset = 0; P->pending_wait = 0; P->stop[0] = P->stop[1] = 0xffffffffu; P->end = 0; P->samp_k = 0; P->samp_topp = 1.0f; P->samp_invT = 1.0f; P->p = malloc(sizeof(phase_t) * P->cap); }
void prog_free(prog_t *P) { free(P->p); P->p = NULL; P->n = P->cap = 0; }
/* Return a strictly validated position-major region, or one grouped phase.
 * Every region retains grid barriers. Its first operation may consume a prior
 * publication, but no operation publishes or advances the exchange count. */
static int serial_span(const prog_t *p, int i, unsigned count, int aliased) {
    static const int indexer[]={OP_IDXQK,OP_IDXSCORE,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE};
    static const int ple[]={OP_PLEGATE,OP_RMS2560,OP_PLECONV};
    static const int compact[]={OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE};
    const int *ops=NULL;int n=1;
    if(aliased && p->p[i].w[0]==OP_IDXQK){ops=indexer;n=10;}
    else if(aliased && p->p[i].w[0]==OP_PLEGATE){ops=ple;n=3;}
    else if((aliased || count>2) && p->p[i].w[0]==OP_KVGATHER){ops=compact;n=3;}
    if(!ops)return 1;
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
    if (!a || a->n <= 0 || !a->p || a->n > INT_MAX / (int)count ||
        (a->loop && a->token_offset > UINT32_MAX - (count - 1u))) return -1;
    /* Validate every position before allocating or changing out. */
    for (unsigned k = 0; k < count; k++) {
        const prog_t *b = positions[k];
        if (!b || out == b || a->n != b->n || !b->p || a->xj != b->xj ||
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
    free(out->p);
    *out = *a;
    out->p = p; out->n = out->cap = (int)n;
    return 0;
}
int prog_group(prog_t *out, const prog_t *const *positions, unsigned count) { return prog_group_impl(out,positions,count,0); }
int prog_group_aliased(prog_t *out, const prog_t *const *positions, unsigned count) { return prog_group_impl(out,positions,count,1); }
int prog_pair(prog_t *out, const prog_t *a, const prog_t *b) {
    const prog_t *positions[] = { a, b };
    return prog_group(out, positions, 2);
}
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
int prog_pair_fuse_bf16(prog_t *P, int disjoint_xout) { return prog_fuse_bf16(P, 2, disjoint_xout); }
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
phase_t *prog_add(prog_t *P, int op, unsigned units) {
	if (P->n == P->cap) { P->cap *= 2; P->p = realloc(P->p, sizeof(phase_t) * P->cap); }
	phase_t *p = &P->p[P->n++];
	memset(p, 0, sizeof *p);
	p->w[0] = (uint32_t)op; p->w[1] = units;
	if (P->pending_wait) { p->w[30] |= 2; P->pending_wait = 0; }   /* wait for the exchange before running */
	return p;
}
/* the phase just added needs no grid barrier after it: the next phase neither reads what it
   writes nor writes what it reads (independent GEMVs on the same input). Not for sends. */
void prog_nobar(prog_t *P) {
	if (getenv("PROG_NONOBAR") || !P->n) return;
	phase_t *p = &P->p[P->n - 1];
	if (p->w[30] & 1) { fprintf(stderr, "prog_nobar on a send phase\n"); exit(1); }
	p->w[30] |= 4;
}
void ph_ptr(phase_t *p, int addr, CUdeviceptr v) { int w = 2 + (addr - 0x140) / 4; p->w[w] = (uint32_t)v; p->w[w + 1] = (uint32_t)(v >> 32); }
void ph_u32(phase_t *p, int addr, uint32_t v) { p->w[2 + (addr - 0x140) / 4] = v; }
void ph_f32(phase_t *p, int addr, float v) { uint32_t u; memcpy(&u, &v, 4); ph_u32(p, addr, u); }
void ph_tokv(prog_t *P, phase_t *p, int addr, uint32_t base, uint32_t value) {
	if (P->loop) { ph_u32(p, addr, base + P->token_offset); p->w[31] |= 1u << (2 + (addr - 0x140) / 4); } else ph_u32(p, addr, value);   /* descriptor word 2 + i */
}

void ph_rms(prog_t *P, int op, CUdeviceptr X, CUdeviceptr W, CUdeviceptr Y, float eps, unsigned vectors) {
	phase_t *p = prog_add(P, op, vectors);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, Y); ph_f32(p, 0x158, eps);
}
void ph_bf16(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr Y, unsigned rows, CUdeviceptr bias) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, Y); ph_u32(p, 0x158, rows); ph_ptr(p, 0x160, bias);
}
void ph_hc(prog_t *P, int op, CUdeviceptr P0, CUdeviceptr P1, CUdeviceptr P2, unsigned n, unsigned us, unsigned units) {
	phase_t *p = prog_add(P, op, units);
	ph_ptr(p, 0x140, P0); ph_ptr(p, 0x148, P1); ph_ptr(p, 0x150, P2); ph_u32(p, 0x158, n); ph_u32(p, 0x15c, us);
}
void ph_actq(prog_t *P, int op, CUdeviceptr X, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors) {
	phase_t *p = prog_add(P, op, vectors);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, Q); ph_ptr(p, 0x150, S32); ph_ptr(p, 0x158, S16); ph_ptr(p, 0x160, AS); ph_u32(p, 0x168, n);
}
void ph_ffnact(prog_t *P, CUdeviceptr G, CUdeviceptr U, CUdeviceptr Q, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr AS, unsigned n, unsigned vectors) {
	phase_t *p = prog_add(P, OP_FFNACT, vectors);
	ph_ptr(p, 0x140, G); ph_ptr(p, 0x148, U); ph_ptr(p, 0x150, Q); ph_ptr(p, 0x158, S32); ph_ptr(p, 0x160, S16); ph_ptr(p, 0x168, AS); ph_u32(p, 0x170, n);
}
void ph_gemvk(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr AS) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f); ph_ptr(p, 0x170, AS);
}
void ph_q2(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS, CUdeviceptr WE, CUdeviceptr NE) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS); ph_ptr(p, 0x178, WE); ph_ptr(p, 0x180, NE);
}
void ph_b32(prog_t *P, int op, CUdeviceptr W, CUdeviceptr X, CUdeviceptr S, CUdeviceptr Y, unsigned rows, CUdeviceptr D, CUdeviceptr AS) {
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, S); ph_ptr(p, 0x158, Y); ph_u32(p, 0x160, rows); ph_f32(p, 0x164, 1.0f);
	ph_ptr(p, 0x168, D); ph_ptr(p, 0x170, AS);
}
void ph_conv1d(prog_t *P, CUdeviceptr X, CUdeviceptr STATE, CUdeviceptr W, CUdeviceptr U, unsigned t, unsigned n) {
	phase_t *p = prog_add(P, OP_CONV1D, (n + 1023) / 1024);
	ph_ptr(p, 0x140, X); ph_ptr(p, 0x148, STATE); ph_ptr(p, 0x150, W); ph_ptr(p, 0x158, U); ph_tokv(P, p, 0x160, 0, t); ph_u32(p, 0x164, n);
}
void ph_dnstep(prog_t *P, CUdeviceptr S, CUdeviceptr Q, CUdeviceptr K, CUdeviceptr V, CUdeviceptr A, CUdeviceptr B, CUdeviceptr SSMA, CUdeviceptr O, float eps, unsigned hb, unsigned kb, unsigned heads) {
	phase_t *p = prog_add(P, OP_DNSTEP, heads);
	ph_ptr(p, 0x140, S); ph_ptr(p, 0x148, Q); ph_ptr(p, 0x150, K); ph_ptr(p, 0x158, V); ph_ptr(p, 0x160, A); ph_ptr(p, 0x168, B);
	ph_ptr(p, 0x170, SSMA); ph_ptr(p, 0x178, O); ph_f32(p, 0x180, eps); ph_u32(p, 0x184, hb); ph_u32(p, 0x188, kb);
}
void ph_dnnorm(prog_t *P, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, CUdeviceptr Y, float eps, unsigned heads) {
	phase_t *p = prog_add(P, OP_DNNORM, heads);
	ph_ptr(p, 0x140, O); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, G); ph_ptr(p, 0x158, Y); ph_f32(p, 0x160, eps);
}
void ph_router(prog_t *P, CUdeviceptr R, CUdeviceptr SH, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr S) {
	phase_t *p = prog_add(P, OP_ROUTER, 1);
	ph_ptr(p, 0x140, R); ph_ptr(p, 0x148, SH); ph_ptr(p, 0x150, IDX); ph_ptr(p, 0x158, W); ph_ptr(p, 0x160, S);
}
void ph_moesel(prog_t *P, CUdeviceptr IDX, CUdeviceptr W, CUdeviceptr MAP, CUdeviceptr LIDX, CUdeviceptr LW, CUdeviceptr NE, CUdeviceptr PART) {
	phase_t *p = prog_add(P, OP_MOESEL, 1);
	ph_ptr(p, 0x140, IDX); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, MAP); ph_ptr(p, 0x158, LIDX); ph_ptr(p, 0x160, LW); ph_ptr(p, 0x168, NE); ph_ptr(p, 0x170, PART);
}
void ph_sumn(prog_t *P, CUdeviceptr SRC, unsigned stride, CUdeviceptr OUT, unsigned n, unsigned ndies) {
	phase_t *p = prog_add(P, OP_SUMN, (n + 255) / 256);
	ph_ptr(p, 0x140, SRC); ph_u32(p, 0x148, stride); ph_ptr(p, 0x150, OUT); ph_u32(p, 0x158, n); ph_u32(p, 0x15c, ndies);
}

int gemv_op(uint32_t type, int K, int acc) {
	switch (type) {
	case GG_Q2_0: return K == 2560 ? OP_Q2K2560 : acc ? OP_Q2ACC640 : OP_Q2K640;
	case GG_Q4_K: return K == 2560 ? OP_Q4KK2560 : OP_Q4KK6144;
	case GG_Q5_K: return K == 2560 ? OP_Q5KK2560 : OP_Q5KK6144;
	case GG_Q3_K: return K == 2560 ? OP_Q3KK2560 : OP_Q3KK6144;
	case GG_Q6_K: return K == 2560 ? OP_Q6KK2560 : getenv("FAKE_Q6K") ? OP_Q5KK6144 : 0;   /* FAKE_Q6K: timing runs only (wrong numbers) until the K=6144 Q6_K kernel exists */
	case GG_IQ4_XS: return K == 2560 ? OP_IQ4K2560 : OP_IQ4K6144;
	case GG_Q4_0: return acc ? OP_B32Q40ACC : OP_B32Q40;
	case GG_Q5_0: return acc ? OP_B32Q50ACC : OP_B32Q50;
	case GG_Q8_0: return acc ? OP_B32Q80ACC : OP_B32Q80;
	case GG_IQ4_NL: return acc ? OP_B32IQ4NLACC : OP_B32IQ4NL;
	default: return 0;
	}
}
int gemv_op_is_q2(int op) { return op == OP_Q2K2560 || op == OP_Q2K640 || op == OP_Q2ACC640 || op == OP_Q2MOEG || op == OP_Q2MOED || op == OP_Q2MOEGN || op == OP_Q2MOEDN; }
int gemv_op_is_b32(int op) { return op == OP_B32Q40 || op == OP_B32Q50 || op == OP_B32Q80 || op == OP_B32IQ4NL || op == OP_B32Q40ACC || op == OP_B32Q50ACC || op == OP_B32Q80ACC || op == OP_B32IQ4NLACC; }

void ph_gemv_any(prog_t *P, uint32_t type, int K, CUdeviceptr W, CUdeviceptr Xq, CUdeviceptr S32, CUdeviceptr S16, CUdeviceptr Y, unsigned rows, CUdeviceptr AS, int acc, CUdeviceptr scale) {
	if (type == GG_Q6_K && K == 6144) {   /* four K = 1536 column slices (the holder's layout), y = ((p0 + p1) + p2) + p3 */
		size_t sl = (size_t)rows * (size_t)q6_k_tile_stride(1536);
		for (unsigned s = 0; s < 4; s++) ph_gemvk(P, s ? OP_Q6KK1536ACC : OP_Q6KK1536, W + sl * s, Xq + 2u * 1536u * s, S16 + 4u * 96u * s, Y, rows, AS + 4u * 48u * s);
		return;
	}
	int op = gemv_op(type, K, acc);
	if (!op) { fprintf(stderr, "prog: no GEMV op for type %u K %d\n", type, K); exit(1); }
	if (gemv_op_is_q2(op)) ph_q2(P, op, W, Xq, S32, Y, rows, 0, AS, scale, 0);
	else if (gemv_op_is_b32(op)) ph_b32(P, op, W, Xq, S32, Y, rows, scale, AS);
	else ph_gemvk(P, op, W, Xq, type == GG_Q6_K ? S16 : S32, Y, rows, AS);   /* Q6_K takes the per-16 sums */
}

/* ---- exchange ---- */
/* the fused hc up (bf16hcup): t = SiLU(0.25 DOWN) (tn values, the last tn - 320 copied), the die's 4 lane rows
   of up (u -> LUP, rows l cols + c), z_c = mix of the 4 lanes into every die's slot at myoff + 4 c */
CUdeviceptr ph_hcup(prog_t *P, const xchg_t *x, CUdeviceptr W, CUdeviceptr DOWN, CUdeviceptr LUP, unsigned cols, CUdeviceptr XH, CUdeviceptr T, unsigned tn, unsigned myoff) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_BF16HCUP, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, DOWN); ph_ptr(p, 0x150, LUP); ph_u32(p, 0x158, 4 * cols);
	ph_u32(p, 0x168, slot + myoff); ph_ptr(p, 0x170, x->xtab); ph_ptr(p, 0x178, XH); ph_ptr(p, 0x180, T); ph_u32(p, 0x188, tn);
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}
/* hcmix / dnnorm variants whose output lands in every die's inbox slot (HC_XOUT / DN_XOUT ops) */
CUdeviceptr ph_hcx(prog_t *P, const xchg_t *x, int op, CUdeviceptr P0, CUdeviceptr P1, unsigned n, unsigned us, unsigned units, unsigned myoff) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, op, units);
	ph_ptr(p, 0x140, P0); ph_ptr(p, 0x148, P1); ph_ptr(p, 0x150, x->xtab); ph_u32(p, 0x158, n); ph_u32(p, 0x15c, us); ph_u32(p, 0x160, slot + myoff);
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}
CUdeviceptr ph_dnnormx(prog_t *P, const xchg_t *x, CUdeviceptr O, CUdeviceptr W, CUdeviceptr G, float eps, unsigned heads, unsigned myoff) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_DNNORMX, heads);
	ph_ptr(p, 0x140, O); ph_ptr(p, 0x148, W); ph_ptr(p, 0x150, G); ph_ptr(p, 0x158, x->xtab); ph_f32(p, 0x160, eps); ph_u32(p, 0x168, slot + myoff);
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}

/* a GEMV whose rows go straight into every die's inbox slot (op built with GEMV_XOUT): the
   phase is a send, the next phase waits; returns the gathered slot */
/* the hc down GEMV with the RMS norm of S fused (bf16hcrx): every CTA normalises S into shared memory, the
   rows go to every die's slot, xh (the normalised stream) is stored for the up / mix that follows */
CUdeviceptr ph_bf16rx(prog_t *P, const xchg_t *x, CUdeviceptr W, CUdeviceptr S, CUdeviceptr NW, CUdeviceptr XH, float eps, unsigned rows, unsigned myoff) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_BF16HCRX, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, S); ph_ptr(p, 0x150, x->xtab); ph_u32(p, 0x158, rows); ph_u32(p, 0x168, slot + myoff);
	ph_ptr(p, 0x178, XH); ph_ptr(p, 0x190, NW); ph_f32(p, 0x198, eps);
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}
CUdeviceptr ph_bf16x(prog_t *P, const xchg_t *x, int op, CUdeviceptr W, CUdeviceptr X, unsigned rows, unsigned myoff) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, op, 16);
	ph_ptr(p, 0x140, W); ph_ptr(p, 0x148, X); ph_ptr(p, 0x150, x->xtab); ph_u32(p, 0x158, rows); ph_u32(p, 0x168, slot + myoff);
	p->w[30] |= 1; P->pending_wait = 1; P->xj++;
	return x->inbox + slot;
}

CUdeviceptr ph_xchg(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr src, unsigned nbytes, unsigned myoff, unsigned total) {
	unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
	phase_t *p = prog_add(P, OP_XSEND, 16);
	ph_ptr(p, 0x140, x->xtab); ph_ptr(p, 0x148, src); ph_u32(p, 0x150, nbytes); ph_u32(p, 0x154, myoff); ph_u32(p, 0x158, slot); ph_u32(p, 0x15c, (unsigned)ndies); ph_ptr(p, 0x160, x->xgen); ph_u32(p, 0x168, (unsigned)me);
	p->w[30] |= 1;             /* a send (every CTA publishes its flag itself; the bit is informational) ... */
	P->pending_wait = 1;       /* ... and the next phase polls for every die's and CTA's flag first */
	(void)total;
	P->xj++;
	return x->inbox + slot;
}

void prog_token_end(prog_t *P, const xchg_t *x, int me, int ndies, CUdeviceptr any) {
	if (P->loop && (P->xj & 1)) ph_xchg(P, x, me, ndies, any, 0, 0, 16);
}

/* ---- the split ---- */
void split_of(split_t *s, int d, int n, int blk) {
	s->d = d; s->n = n;
	s->dj0 = split_bound(324, d, n);  s->djn = split_bound(324, d + 1, n) - s->dj0;
	s->cb0 = split_bound(80, d, n);   s->cbn = split_bound(80, d + 1, n) - s->cb0;
	s->h0 = split_bound(48, d, n);    s->hn = split_bound(48, d + 1, n) - s->h0;
	/* value head h pairs with key head h mod 16 (ggml_repeat_4d / the fused GDN kernel: head h uses q/k
	   head h % neq1 — not h / 3, which cost 2 of 7 reference tokens): the die holds key heads
	   (klo + j) % 16, j < nk, in that order, so local value head j reads local key head j % 16 */
	s->klo = s->h0 % 16;   s->nk = s->hn < 16 ? s->hn : 16; s->khi = (s->klo + s->nk - 1) % 16;
	s->y0 = split_bound(2560, d, n);  s->yn = split_bound(2560, d + 1, n) - s->y0;
	s->pk0 = split_bound(10240, d, n); s->pkn = split_bound(10240, d + 1, n) - s->pk0;
	s->o0 = split_bound(248320, d, n); s->on = split_bound(248320, d + 1, n) - s->o0;
	s->hd0 = split_bound(320, d, n);  s->hdn = split_bound(320, d + 1, n) - s->hd0;
	s->e0 = split_bound(512, d, n);   s->en = split_bound(512, d + 1, n) - s->e0;
	s->shexp = (blk % n) == d;
}

/* hyper-connection read: xh = per-lane RMS(S) * norm (whole, on every die); this die's
 * rows of down ++ inject, gathered; t = SiLU(0.25 down) ++ inj; this die's rows of
 * up (the 4 lanes of its z columns); z columns mixed locally, gathered; z quantised.
 * Returns the gathered z (an inbox slot). */
/* Y / inj: the pending stream write of the attention side (gathered y, inject values) or 0: with them
   the read runs bf16hcrwx, which first applies the write (S + 2 sigmoid(inj / 4) y) into S2 */
static CUdeviceptr ph_hc_read(prog_t *P, const split_t *s, const xchg_t *x, CUdeviceptr norm, CUdeviceptr dj, CUdeviceptr up, const dn_bufs_t *b, float eps, CUdeviceptr Y, CUdeviceptr inj) {
	unsigned cols = 32u * (unsigned)s->cbn;
	CUdeviceptr DOWN;
	if (Y) {
		unsigned slot = (unsigned)(P->xj & 1) * XSLOT;
		phase_t *p = prog_add(P, OP_BF16HCRWX, 16);
		ph_ptr(p, 0x140, dj); ph_ptr(p, 0x148, b->S); ph_ptr(p, 0x150, x->xtab); ph_u32(p, 0x158, (unsigned)s->djn); ph_ptr(p, 0x160, b->S2);
		ph_u32(p, 0x168, slot + 4u * (unsigned)s->dj0); ph_ptr(p, 0x178, b->xh); ph_ptr(p, 0x190, norm); ph_f32(p, 0x198, eps); ph_ptr(p, 0x1a0, Y); ph_ptr(p, 0x1a8, inj);
		p->w[30] |= 1; P->pending_wait = 1; P->xj++;
		DOWN = x->inbox + slot;
	} else if (getenv("PROG_NORMSFUSE")) {
		ph_rms(P, OP_RMS2560, b->S, norm, b->xh, eps, 4);
		DOWN = ph_bf16x(P, x, OP_BF16HCX, dj, b->xh, (unsigned)s->djn, 4u * (unsigned)s->dj0);   /* down ++ inject rows straight into every die's slot */
	} else DOWN = ph_bf16rx(P, x, dj, b->S, norm, b->xh, eps, (unsigned)s->djn, 4u * (unsigned)s->dj0);   /* + the RMS norm */
	CUdeviceptr Z = getenv("PROG_NOHCUP") ? 0 : ph_hcup(P, x, up, DOWN, b->lup, cols, b->xh + 4u * 32u * (unsigned)s->cb0, b->t, 324, 4u * 32u * (unsigned)s->cb0);
	if (!Z) {
		ph_hc(P, OP_HCSILU, DOWN, b->t, 0, 324, 320, 2);
		ph_bf16(P, OP_BF16K320, up, b->t, b->lup, 4 * cols, 0);
		Z = ph_hcx(P, x, OP_HCMIXX, b->lup, b->xh + 4u * 32u * (unsigned)s->cb0, cols, 4 * cols, (cols + 255) / 256, 4u * 32u * (unsigned)s->cb0);
	}
	ph_actq(P, OP_ACTQ256, Z, b->zq, b->zs32, b->zs16, b->zas, 256, 10);
	return Z;
}

/* the MoE hyper-block (expert split: this die's routed experts, partials reduced) */
static void ph_moe_half(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, float eps, CUdeviceptr Y) {
	CUdeviceptr inj = b->t + 320 * 4;
	/* Y: the attention side's gathered output, its write fused into this read (the stream goes to S2 and
	   the MoE end writes S from S2); 0: the write already happened (S2 = S, in place) */
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_ffn_norm, w->hc_ffn_dj, w->hc_ffn_up, b, eps, Y, inj);
	CUdeviceptr SIN = Y ? b->S2 : b->S;
	ph_bf16(P, OP_BF16K2560, w->ffn_gate_inp_x, Z, b->r, 513, 0);
	ph_router(P, b->r, b->r + 512 * 4, b->idx, b->w, b->s);
	ph_moesel(P, b->idx, b->w, w->emap, b->lidx, b->lw, b->ne, b->part);
	ph_q2(P, OP_Q2MOEGN, w->ffn_gate_exps, b->zq, b->zs32, b->g, 640, b->lidx, b->zas, 0, b->ne);
	prog_nobar(P);   /* expert gate / up and shared gate / up: four GEMVs of z into disjoint rows */
	ph_q2(P, OP_Q2MOEGN, w->ffn_up_exps, b->zq, b->zs32, b->gu, 640, b->lidx, b->zas, 0, b->ne);
	if (s->shexp) {
		prog_nobar(P);
		ph_gemv_any(P, w->ffn_gate_shexp_type, 2560, w->ffn_gate_shexp, b->zq, b->zs32, b->zs16, b->g + 6400 * 4, 640, b->zas, 0, 0);
		prog_nobar(P);
		ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->zs16, b->gu + 6400 * 4, 640, b->zas, 0, 0);
		if (getenv("PROG_DUP")) ph_gemv_any(P, w->ffn_up_shexp_type, 2560, w->ffn_up_shexp, b->zq, b->zs32, b->zs16, b->gu + 6400 * 4, 640, b->zas, 0, 0);
	}
	ph_ffnact(P, b->g, b->gu, b->hq, b->hs32, b->hs16, b->has, 640, s->shexp ? 11 : 10);
	ph_q2(P, OP_Q2MOEDN, w->ffn_down_exps, b->hq, b->hs32, b->part, 2560, b->lidx, b->has, b->lw, b->ne);
	if (s->shexp) ph_gemv_any(P, w->ffn_down_shexp_type, 640, w->ffn_down_shexp, b->hq + 10 * 640 * 2, b->hs32 + 10 * 20 * 4, b->hs16 + 10 * 40 * 4, b->part, 2560, b->has + 10 * 20 * 4, 1, b->s);
	CUdeviceptr PARTS = ph_xchg(P, x, s->d, s->n, b->part, 2560 * 4, 2560u * 4u * (unsigned)s->d, 2560u * 4u * (unsigned)s->n);
	/* out = the fixed-order sum of the parts (sumn's order), fused into the stream write */
	phase_t *p = prog_add(P, OP_HCWSUM, 16);   /* 160 columns per CTA: all sixteen SMs */
	ph_ptr(p, 0x140, inj); ph_ptr(p, 0x148, PARTS); ph_ptr(p, 0x150, b->S); ph_u32(p, 0x158, 2560); ph_u32(p, 0x15c, 2560 * 4); ph_u32(p, 0x164, (unsigned)s->n); ph_ptr(p, 0x168, b->out); ph_ptr(p, 0x170, SIN);
}

/* QSA attention (ref/qsa.h): the q/k/v projections of z and the indexer projections (Z, the gathered fp32 z), the int8 KV append,
 * the indexer (q norm + rope, key ring, pooled keys, scores, 5 selection passes), the gather of the attended blocks into the compact
 * caches and flash-decoding over them (n keys from the token position, see ref/qsa.h qsa_nkeys), then the merge with the output gate */
static void ph_attn_kvi(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, CUdeviceptr Z) {
	ph_gemv_any(P, w->attn_k_type, 2560, w->attn_k, b->zq, b->zs32, b->zs16, b->k, 512, b->zas, 0, 0);
	prog_nobar(P);
	ph_gemv_any(P, w->attn_v_type, 2560, w->attn_v, b->zq, b->zs32, b->zs16, b->v, 512, b->zas, 0, 0);
	prog_nobar(P);
	ph_bf16(P, OP_BF16K2560, w->indexer_x, Z, b->iqk, 640, 0);   /* iq (512) ++ ik (128) */
}
static void ph_attn_qsa(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps, CUdeviceptr Z, CUdeviceptr QG) {
	if (!QG) {
		ph_gemv_any(P, w->attn_q_type, 2560, w->attn_q, b->zq, b->zs32, b->zs16, b->qg, 12288, b->zas, 0, 0);
		prog_nobar(P);
		ph_attn_kvi(P, w, b, Z);
		QG = b->qg;
	}
	phase_t *p = prog_add(P, OP_ATTNQKQ, 26);
	ph_ptr(p, 0x140, QG); ph_ptr(p, 0x148, b->k); ph_ptr(p, 0x150, b->v); ph_ptr(p, 0x158, w->attn_q_norm); ph_ptr(p, 0x160, w->attn_k_norm);
	ph_ptr(p, 0x168, b->rope_cs); ph_ptr(p, 0x170, b->rope_sn); ph_ptr(p, 0x178, b->qo); ph_ptr(p, 0x180, b->kq); ph_ptr(p, 0x188, b->vq);
	ph_tokv(P, p, 0x190, 0, t); ph_f32(p, 0x194, eps); ph_ptr(p, 0x198, b->ks); ph_ptr(p, 0x1a0, b->vs);
	ph_ptr(p, 0x1a8, b->kv_rawmax);
	prog_nobar(P);   /* the indexer does not read what the cache append writes */
	p = prog_add(P, OP_IDXQK, 6);
	ph_ptr(p, 0x140, b->iqk); ph_ptr(p, 0x148, w->indexer_qn); ph_ptr(p, 0x150, w->indexer_kn); ph_ptr(p, 0x158, b->rope_cs); ph_ptr(p, 0x160, b->rope_sn); ph_ptr(p, 0x168, b->iq);
	ph_ptr(p, 0x170, b->ring); ph_ptr(p, 0x178, b->pkt); ph_ptr(p, 0x180, b->hst); ph_tokv(P, p, 0x188, 0, t); ph_f32(p, 0x18c, eps);
	p = prog_add(P, OP_IDXSCORE, 16);
	ph_ptr(p, 0x140, b->pkt); ph_ptr(p, 0x148, b->iq); ph_ptr(p, 0x150, b->sc); ph_ptr(p, 0x158, b->hst); ph_tokv(P, p, 0x160, 0, t); ph_u32(p, 0x164, b->nblk);
	for (unsigned pass = 2; pass <= 6; pass++) {
		p = prog_add(P, OP_IDXSEL, 16);
		ph_ptr(p, 0x140, b->sc); ph_ptr(p, 0x148, b->hst); ph_ptr(p, 0x150, b->sel); ph_tokv(P, p, 0x158, 0, t); ph_u32(p, 0x15c, b->nblk); ph_u32(p, 0x160, pass);
	}
	p = prog_add(P, OP_KVGATHER, b->nblk + 1);
	ph_ptr(p, 0x140, b->kq); ph_ptr(p, 0x148, b->ks); ph_ptr(p, 0x150, b->vq); ph_ptr(p, 0x158, b->vs); ph_ptr(p, 0x160, b->kc); ph_ptr(p, 0x168, b->vc);
	ph_ptr(p, 0x170, b->sel); ph_tokv(P, p, 0x178, 0, t); ph_u32(p, 0x17c, b->nblk);
	p = prog_add(P, OP_ATTNPARTN, 32);
	ph_ptr(p, 0x140, b->qo); ph_ptr(p, 0x148, b->kc); ph_ptr(p, 0x150, b->vc); ph_ptr(p, 0x158, b->pm); ph_ptr(p, 0x160, b->pd); ph_ptr(p, 0x168, b->po); ph_tokv(P, p, 0x170, 1, t + 1); ph_u32(p, 0x174, b->nblk);
	p = prog_add(P, OP_ATTNMERGE, 24);
	ph_ptr(p, 0x140, b->pm); ph_ptr(p, 0x148, b->pd); ph_ptr(p, 0x150, b->po); ph_ptr(p, 0x158, QG); ph_ptr(p, 0x160, b->aout);
}

/* attention proper on die 0 (v1): q/k/v GEMVs, per-head norm + RoPE + cache append,
 * flash-decoding partials over the t+1 keys, merge with the output gate */
static void ph_attn_core(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	ph_gemv_any(P, w->attn_q_type, 2560, w->attn_q, b->zq, b->zs32, b->zs16, b->qg, 12288, b->zas, 0, 0);
	prog_nobar(P);
	ph_gemv_any(P, w->attn_k_type, 2560, w->attn_k, b->zq, b->zs32, b->zs16, b->k, 512, b->zas, 0, 0);
	prog_nobar(P);   /* q, k, v GEMVs of z into separate buffers */
	ph_gemv_any(P, w->attn_v_type, 2560, w->attn_v, b->zq, b->zs32, b->zs16, b->v, 512, b->zas, 0, 0);
	phase_t *p = prog_add(P, OP_ATTNQK, 26);
	ph_ptr(p, 0x140, b->qg); ph_ptr(p, 0x148, b->k); ph_ptr(p, 0x150, b->v); ph_ptr(p, 0x158, w->attn_q_norm); ph_ptr(p, 0x160, w->attn_k_norm);
	ph_ptr(p, 0x168, b->rope_cs); ph_ptr(p, 0x170, b->rope_sn); ph_ptr(p, 0x178, b->qo); ph_ptr(p, 0x180, b->kc); ph_ptr(p, 0x188, b->vc);
	ph_tokv(P, p, 0x190, 0, t); ph_f32(p, 0x194, eps);
	p = prog_add(P, OP_ATTNPART, 32);
	ph_ptr(p, 0x140, b->qo); ph_ptr(p, 0x148, b->kc); ph_ptr(p, 0x150, b->vc); ph_ptr(p, 0x158, b->pm); ph_ptr(p, 0x160, b->pd); ph_ptr(p, 0x168, b->po); ph_tokv(P, p, 0x170, 1, t + 1);
	p = prog_add(P, OP_ATTNMERGE, 24);
	ph_ptr(p, 0x140, b->pm); ph_ptr(p, 0x148, b->pd); ph_ptr(p, 0x150, b->po); ph_ptr(p, 0x158, b->qg); ph_ptr(p, 0x160, b->aout);
}

void prog_attn_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	CUdeviceptr inj = b->t + 320 * 4;
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps, 0, 0);
	int here = b->qsa ? b->kq != 0 : s->d == attn_die();   /* the die that runs this layer: the one holding its KV cache (QSA), else attn_die() */
	CUdeviceptr QG = 0;
	if (b->qsa && w->attn_q_split) {
		if (w->attn_q_rows) {
			ph_gemv_any(P, w->attn_q_type, 2560, w->attn_q, b->zq, b->zs32, b->zs16, b->qg, w->attn_q_rows, b->zas, 0, 0);
			if (here) prog_nobar(P);
		}
		/* The owner computes K/V/indexer while other dies finish their Q rows.
		 * Z is in the previous inbox slot; this exchange writes the other one. */
		if (here) ph_attn_kvi(P, w, b, Z);
		QG = ph_xchg(P, x, s->d, s->n, b->qg, 4*w->attn_q_rows, 4*w->attn_q_first, 4*12288);
	}
	if (here) { if (b->qsa) ph_attn_qsa(P, w, b, t, eps, Z, QG); else ph_attn_core(P, w, b, t, eps); }
	CUdeviceptr AOUT = ph_xchg(P, x, s->d, s->n, b->aout, here ? 6144 * 4 : 0, 0, 6144 * 4);
	ph_actq(P, OP_ACTQ256, AOUT, b->oq, b->os32, b->os16, b->oas, 256, 24);
	ph_gemv_any(P, w->attn_output_type, 6144, w->attn_output, b->oq, b->os32, b->os16, b->ly, (unsigned)s->yn, b->oas, 0, 0);
	CUdeviceptr Y = ph_xchg(P, x, s->d, s->n, b->ly, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	if (getenv("PROG_NORMSW")) { ph_hc(P, OP_HCWRITE, inj, Y, b->S, 2560, 0, 10); ph_moe_half(P, s, x, w, b, eps, 0); }
	else ph_moe_half(P, s, x, w, b, eps, Y);   /* the write is fused into the MoE side's read */
}

void prog_embed(prog_t *P, const dn_weights_t *w, const dn_bufs_t *b, unsigned t) {
	phase_t *p = prog_add(P, OP_EMBROW, 10);
	ph_ptr(p, 0x140, w->token_embd); ph_ptr(p, 0x148, b->S); ph_tokv(P, p, 0x150, 0, t); ph_ptr(p, 0x158, b->ids);
	ph_ptr(p, 0x160, b->imap); ph_ptr(p, 0x168, b->img);
}

/* The token embedding with token_embd on one die only: the owner decodes the row into S (the four lanes, 40,960 bytes) and sends it, the other dies
   send nothing and copy what arrived (inbox slot 0, the program starts at an even exchange count) into their own S. */
void prog_embed_bcast(prog_t *P, const xchg_t *x, int me, int ndies, int owner, const dn_weights_t *w, const dn_bufs_t *b, unsigned t) {
	if (me == owner) prog_embed(P, w, b, t);
	CUdeviceptr in = ph_xchg(P, x, me, ndies, b->S, me == owner ? 4u * 10240u : 0u, 0, 4u * 10240u);
	if (me != owner) { phase_t *p = prog_add(P, OP_XCOPY, 10); ph_ptr(p, 0x140, in); ph_ptr(p, 0x148, b->S); }
}

void prog_head(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, float eps, unsigned t, unsigned nprompt) {
	unsigned cols = 32u * (unsigned)s->cbn;
	/* the read gate without an inject: 320 down rows, nothing copied behind t */
	CUdeviceptr DOWN = ph_bf16rx(P, x, w->output_hc_down, b->S, w->output_hc_norm, b->xh, eps, (unsigned)s->hdn, 4u * (unsigned)s->hd0);
	CUdeviceptr Z = ph_hcup(P, x, w->output_hc_up, DOWN, b->lup, cols, b->xh + 4u * 32u * (unsigned)s->cb0, b->t, 320, 4u * 32u * (unsigned)s->cb0);
	ph_actq(P, OP_ACTQ256, Z, b->zq, b->zs32, b->zs16, b->zas, 256, 10);
	ph_gemv_any(P, GG_Q5_K, 2560, w->output, b->zq, b->zs32, b->zs16, b->hlo, (unsigned)s->on, b->zas, 0, 0);
	phase_t *p;
	if (P->samp_k >= 2) {   /* top-k -> top-p -> temperature, the dies agree on the token (kern/sample.sass) */
		p = prog_add(P, OP_TOPK, 16);
		ph_ptr(p, 0x140, b->hlo); ph_ptr(p, 0x148, b->tkout); ph_ptr(p, 0x150, b->tkpart); ph_ptr(p, 0x158, b->tkcnt); ph_u32(p, 0x160, (unsigned)s->on); ph_u32(p, 0x164, 16); ph_u32(p, 0x168, (unsigned)P->samp_k);
		CUdeviceptr LISTS = ph_xchg(P, x, s->d, s->n, b->tkout, 256, 256u * (unsigned)s->d, 256u * (unsigned)s->n);
		p = prog_add(P, OP_SAMPLE, 1);
		ph_ptr(p, 0x140, LISTS); ph_ptr(p, 0x148, w->obases); ph_u32(p, 0x150, (unsigned)s->n); ph_u32(p, 0x154, (unsigned)P->samp_k); ph_f32(p, 0x158, P->samp_topp); ph_f32(p, 0x15c, P->samp_invT);
		ph_ptr(p, 0x160, b->ids); ph_ptr(p, 0x168, b->toks); ph_tokv(P, p, 0x170, 0, t); ph_u32(p, 0x174, nprompt);
		ph_u32(p, 0x178, P->stop[0]); ph_u32(p, 0x17c, P->stop[1]); ph_ptr(p, 0x180, P->end); ph_ptr(p, 0x188, b->rnd);
		return;
	}
	p = prog_add(P, OP_ARGMAX, 16);
	ph_ptr(p, 0x140, b->hlo); ph_ptr(p, 0x148, b->amo); ph_ptr(p, 0x150, b->apart); ph_ptr(p, 0x158, b->acnt); ph_u32(p, 0x160, (unsigned)s->on); ph_u32(p, 0x164, 16);
	CUdeviceptr PAIRS = ph_xchg(P, x, s->d, s->n, b->amo, 8, 8u * (unsigned)s->d, 8u * (unsigned)s->n);
	p = prog_add(P, OP_AMAX, 1);
	ph_ptr(p, 0x140, PAIRS); ph_ptr(p, 0x148, w->obases); ph_u32(p, 0x150, (unsigned)s->n); ph_ptr(p, 0x158, b->tokout);
	ph_ptr(p, 0x160, b->ids); ph_ptr(p, 0x168, b->toks); ph_tokv(P, p, 0x170, 0, t); ph_u32(p, 0x174, nprompt);
	ph_u32(p, 0x178, P->stop[0]); ph_u32(p, 0x17c, P->stop[1]); ph_ptr(p, 0x180, P->end);
}

void prog_ple_prefix(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	phase_t *p = prog_add(P, OP_PLEROWS, 1);   /* every die hashes and gathers the 16 rows itself */
	ph_ptr(p, 0x140, b->ids); ph_ptr(p, 0x148, b->ple_pc); ph_ptr(p, 0x150, b->ple_tab); ph_ptr(p, 0x158, b->pg); ph_ptr(p, 0x160, b->prows); ph_tokv(P, p, 0x168, 0, t); ph_u32(p, 0x16c, b->ple_compact);
	ph_ptr(p, 0x170, b->ple_rdy); ph_u32(p, 0x178, b->ple_rid);
	ph_actq(P, OP_ACTQ256, b->pg, b->pgq, b->pgs32, b->pgs16, b->pgas, 256, 10);
	ph_gemv_any(P, GG_Q2_0, 2560, w->ple_key, b->pgq, b->pgs32, b->pgs16, b->lk, (unsigned)s->pkn, b->pgas, 0, 0);
	CUdeviceptr K = ph_xchg(P, x, s->d, s->n, b->lk, 4u * (unsigned)s->pkn, 4u * (unsigned)s->pk0, 4 * 10240);
	ph_rms(P, OP_RMS2560, K, w->ple_norm_key, b->kn, eps, 4);
	ph_rms(P, OP_RMS2560, b->S, w->ple_norm_query, b->qn, eps, 4);
	ph_bf16(P, OP_BF16K2560, w->ple_value, b->pg, b->lv, (unsigned)s->yn, 0);
	CUdeviceptr V = ph_xchg(P, x, s->d, s->n, b->lv, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	p = prog_add(P, OP_PLEGATE, 4);
	ph_ptr(p, 0x140, b->kn); ph_ptr(p, 0x148, b->qn); ph_ptr(p, 0x150, V); ph_ptr(p, 0x158, b->pG); ph_ptr(p, 0x160, b->pgates);
	ph_rms(P, OP_RMS2560, b->pG, w->ple_norm_conv, b->px, eps, 4);
	p = prog_add(P, OP_PLECONV, 10);
	ph_ptr(p, 0x140, b->px); ph_ptr(p, 0x148, b->ple_state); ph_ptr(p, 0x150, w->ple_conv1d); ph_ptr(p, 0x158, b->pG); ph_ptr(p, 0x160, b->S); ph_tokv(P, p, 0x168, 0, t);
}

void prog_deltanet_block(prog_t *P, const split_t *s, const xchg_t *x, const dn_weights_t *w, const dn_bufs_t *b, unsigned t, float eps) {
	unsigned nloc = SPLIT_NLOC(s), hrows = 128u * (unsigned)s->hn;
	CUdeviceptr inj = b->t + 320 * 4;
	/* ---- attention-side hyper-block: Gated DeltaNet ---- */
	CUdeviceptr Z = ph_hc_read(P, s, x, w->hc_attn_norm, w->hc_attn_dj, w->hc_attn_up, b, eps, 0, 0);
	ph_gemv_any(P, w->attn_qkv_type, 2560, w->attn_qkv, b->zq, b->zs32, b->zs16, b->lqkv, nloc, b->zas, 0, 0);
	prog_nobar(P);
	ph_gemv_any(P, w->attn_gate_type, 2560, w->attn_gate, b->zq, b->zs32, b->zs16, b->lgate, hrows, b->zas, 0, 0);
	prog_nobar(P);   /* qkv, gate and the alpha/beta rows all read z only */
	ph_bf16(P, OP_BF16BIAS, w->ssm_ab, Z, b->la, 2u * (unsigned)s->hn, w->ssm_ab_bias);
	ph_conv1d(P, b->lqkv, b->conv_state, w->ssm_conv1d, b->lu, t, nloc);
	ph_dnstep(P, b->dn_state, b->lu, b->lu + 512u * (unsigned)s->nk, b->lu + 1024u * (unsigned)s->nk, b->la, b->la + 4u * (unsigned)s->hn, w->ssm_a, b->lo, eps, (unsigned)s->h0, (unsigned)s->klo, (unsigned)s->hn);
	CUdeviceptr ON = ph_dnnormx(P, x, b->lo, w->ssm_norm, b->lgate, eps, (unsigned)s->hn, 4u * 128u * (unsigned)s->h0);
	ph_actq(P, OP_ACTQ256, ON, b->oq, b->os32, b->os16, b->oas, 256, 24);
	ph_gemv_any(P, w->ssm_out_type, 6144, w->ssm_out, b->oq, b->os32, b->os16, b->ly, (unsigned)s->yn, b->oas, 0, 0);
	CUdeviceptr Y = ph_xchg(P, x, s->d, s->n, b->ly, 4u * (unsigned)s->yn, 4u * (unsigned)s->y0, 4 * 2560);
	if (getenv("PROG_NORMSW")) { ph_hc(P, OP_HCWRITE, inj, Y, b->S, 2560, 0, 10); ph_moe_half(P, s, x, w, b, eps, 0); }
	else ph_moe_half(P, s, x, w, b, eps, Y);   /* the write is fused into the MoE side's read */
}
