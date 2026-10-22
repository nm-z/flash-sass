#define PHASE_EXCHANGE_JOIN_FLAG 0x20000u
static unsigned phase_producer_fusions,phase_matrix_groups;
static unsigned producer_fuse(prog_t *p,unsigned position,int die,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;unsigned kind;
        if(b->w[0]!=OP_ACTQ256)continue;
        if(a->w[0]==OP_DNNORMX && b->w[1]==24)kind=1;
        else if(a->w[0]==OP_BF16HCUP && b->w[1]==10 && a->w[8]%128==0)kind=0;
        else continue;
        if(kind) { a->w[0]=95;ph_ptr(a,0x158,producer_tables[position][die][1]); }
        else { a->w[0]=96;ph_ptr(a,0x190,(a->w[30]&0x40000u)?producer_local_tables[position][die]:producer_tables[position][die][0]); }
        if(i+2<p->n)p->p[i+2].w[30]|=b->w[30]&2u;
        memmove(p->p+i+1,p->p+i+2,(size_t)(p->n-i-2)*sizeof *p->p);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}
static int matrix_opcode(unsigned op) {
    switch(op) {
    case OP_BF16BIAS:case OP_BF16K2560:case OP_BF16K320:
    case OP_Q2K2560:case OP_Q2MOEGN:case OP_Q3KK2560:
    case OP_Q4KK2560:case OP_Q5KK2560:case OP_Q6KK2560:case OP_IQ4K2560:
    case OP_B32Q40:case OP_B32Q50:case OP_B32Q80:case OP_B32IQ4NL:return 1;
    default:return 0;
    }
}
static unsigned matrix_group(prog_t *p) {
    unsigned groups=0;if(!group_matrices)return 0;
    for(int i=0;i<p->n;i++) {
        phase_t *a=p->p+i;if(!matrix_opcode(a->w[0]) || (a->w[30]&~6u) || ! (a->w[30]&4u) || a->w[31])continue;
        unsigned n=1;
        while(i+(int)n<p->n && n<8) {
            phase_t *b=a+n,*prior=b-1;
            /* prog_nobar is the production dependency declaration. Mixed weight
             * formats and the FP32 alpha/beta projection remain independent. */
            if(!(prior->w[30]&4u) || !matrix_opcode(b->w[0]) || b->w[31] || (b->w[30]&~133u))break;
            n++;
        }
        if(n<2)continue;
        unsigned tail=a[n-1].w[30]&5u;
        for(unsigned j=0;j<n;j++)a[j].w[30]&=~4u;
        a->w[30]|=8u|tail;if(n!=2)a->w[30]|=n<<8;
        groups++;i+=(int)n-1;
    }
    return groups;
}

/* Fuse dependencies whose producer and consumer have the same owner. These
 * descriptors contain one real dispatch and one grid barrier, not a chain of
 * operations with inner grid barriers. */
static unsigned phase_local_fuse(prog_t *p,int *layers) {
    unsigned count=0;
    if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[30]&~2u || b->w[30]&~1u || a->w[31] || b->w[31])continue;
        if(a->w[0]==OP_ROUTER && b->w[0]==OP_MOESEL && a->w[1]==1 && b->w[1]==1 &&
           gr_pointer(a,6)==gr_pointer(b,2) && gr_pointer(a,8)==gr_pointer(b,4)) {
            a->w[0]=97;
            memcpy(a->w+12,b->w+6,10*sizeof(uint32_t));
        } else if(a->w[0]==OP_DNSTEP && b->w[0]==95 && a->w[1]==b->w[1] &&
                  gr_pointer(a,16)==gr_pointer(b,2) && a->w[18]==b->w[10] &&
                  !gr_pointer(a,24)) {
            a->w[0]=98;
            memcpy(a->w+22,b->w+4,6*sizeof(uint32_t));
            a->w[28]=b->w[12];
            a->w[30]|=b->w[30]&1u;
        } else continue;
        memmove(b,b+1,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

/* All participating CTAs retain exactly the original GEMV row ownership. The
 * publication tail handles zero routed experts as well: MOESEL's initialized
 * partials are still the correct output in that case. */
static unsigned phase_publish_fuse(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        switch(a->w[0]) {
        case OP_Q3KK6144:case OP_Q4KK6144:case OP_Q5KK6144:
        case OP_BF16K2560:case OP_IQ4K6144:case OP_Q2MOEDN:case OP_Q2ACC640:
        case OP_Q2K2560:case OP_Q3KK2560:case OP_Q4KK2560:case OP_Q5KK2560:
        case OP_Q6KK2560:case OP_IQ4K2560:case OP_Q6KK1536ACC:
        case OP_B32Q40ACC:case OP_B32Q50ACC:case OP_B32Q80ACC:case OP_B32IQ4NLACC:break;
        default:continue;
        }
        if(a->w[1]!=16 || a->w[30]&~2u || a->w[31] || b->w[0]!=OP_XSEND ||
           b->w[30]!=1 || b->w[31] || b->w[9]!=6 ||
           gr_pointer(a,a->w[0]==OP_BF16K2560?6:8)!=gr_pointer(b,4) || b->w[6]!=a->w[a->w[0]==OP_BF16K2560?8:10]*4u)continue;
        memcpy(a->w+22,b->w+2,2*sizeof(uint32_t));
        a->w[24]=b->w[7]+b->w[8];
        a->w[30]|=129u;
        memmove(b,b+1,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

static unsigned phase_gdn_fuse(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant || !p->loop)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_CONV1D || b->w[0]!=98 || !b->w[1] || b->w[1]>16 ||
           a->w[30]&~2u || b->w[30]!=1 || a->w[31]!=(1u<<10) || b->w[31] ||
           a->w[11]!=3u*128u*b->w[1] || gr_pointer(a,8)!=gr_pointer(b,4) ||
           gr_pointer(b,6)!=gr_pointer(b,4)+512u*b->w[1] ||
           gr_pointer(b,8)!=gr_pointer(b,4)+1024u*b->w[1])continue;
        phase_t conv=*a;
        *a=*b;a->w[0]=99;a->w[30]|=24u|(conv.w[30]&2u);
        /* Group width two, fused body: dispatcher skips the auxiliary descriptor
         * after one actual operation. The body loads its token-relative word. */
        *b=conv;b->w[30]=0;
        if(layers)layers[i]=layers[i+1];
        count++;i++;
    }
    return count;
}

static unsigned phase_router_fuse(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant || !p->loop)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_BF16K2560 || a->w[1]!=16 || a->w[8]!=513 ||
           a->w[30]&~2u || a->w[31] || b->w[0]!=97 || b->w[1]!=1 ||
           b->w[30] || b->w[31] || gr_pointer(a,6)!=gr_pointer(b,2))continue;
        a->w[0]=100;a->w[10]=++count;
        a->w[30]|=24u;
        /* Word10 is unused by unbiased BF16K2560. It supplies the per-layer
         * epoch within the shared router-ready array appended to r[513]. */
        if(layers)layers[i]=layers[i+1];
        i++;
    }
    return count;
}

/* Q6 K6144 is physically stored as four K1536 slices. The same CTA/warp
 * owns every slice of a given output row, so three grid barriers are redundant.
 * Preserve the original slice addition order and only synchronize that CTA. */
static unsigned phase_q6_fuse(prog_t *p) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+3<p->n;i++) {
        phase_t *a=p->p+i;
        if(a->w[0]!=OP_Q6KK1536 || a->w[1]!=16 || a->w[30]&~2u || a->w[31])continue;
        int ok=1;
        for(unsigned k=1;k<4;k++) {
            phase_t *b=a+k;
            if(b->w[0]!=OP_Q6KK1536ACC || b->w[1]!=a->w[1] || b->w[10]!=a->w[10] ||
               (b->w[30] && !(k==3 && b->w[30]==129u)) || b->w[31] || gr_pointer(b,8)!=gr_pointer(a,8))ok=0;
        }
        if(!ok)continue;
        a->w[30]|=8u|32u|(4u<<8)|0x80000000u|(a[3].w[30]&1u);
        count++;i+=3;
    }
    return count;
}

/* Routed and shared down projections own the same groups of64 output rows
 * per CTA. Preserve routed-then-shared addition, and publish from the final
 * consumer without a grid-wide boundary between these operations. */
static unsigned phase_shared_down_fuse(prog_t *p) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_Q2MOEDN || a->w[1]!=16 || a->w[30]&~2u || a->w[31])continue;
        switch(b->w[0]) {
        case OP_Q2ACC640:case OP_B32Q40ACC:case OP_B32Q50ACC:
        case OP_B32Q80ACC:case OP_B32IQ4NLACC:break;
        default:continue;
        }
        if(b->w[1]!=16 || b->w[30]!=129u || b->w[31] ||
           b->w[10]!=a->w[10] || gr_pointer(b,8)!=gr_pointer(a,8))continue;
        a->w[30]|=8u|32u|0x80000000u|1u;
        count++;i++;
    }
    return count;
}

static unsigned phase_attention_fuse(prog_t *p,unsigned position,int die,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *send=p->p+i,*q=send+1;
        if(send->w[0]!=OP_XSEND || (send->w[30]&~2u)!=1 || send->w[31] ||
           send->w[7] || (send->w[6] && send->w[6]!=6144*4) ||
           q->w[0]!=OP_ACTQ256 || q->w[1]!=24 || q->w[30]!=2 || q->w[31] ||
           gr_pointer(q,2)!=position_xchg(die,position)->inbox+send->w[8])continue;
        if(send->w[6]) {
            if(!i)continue;
            phase_t *merge=send-1;
            if(merge->w[0]!=OP_ATTNMERGE || merge->w[1]!=24 ||
               merge->w[30]&~2u || merge->w[31] || gr_pointer(merge,10)!=gr_pointer(send,4))continue;
            merge->w[0]=101;merge->w[30]|=1u;
            ph_ptr(merge,0x168,producer_tables[position][die][1]);
            if(i+2<p->n)q[1].w[30]|=2u;
            memmove(send,q+1,(size_t)(p->n-i-2)*sizeof *send);
            if(layers)memmove(layers+i,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
            p->n-=2;count+=2;i--;
        } else {
            /* This die only participates in the exchange-generation protocol;
             * the owner publishes codes/scales directly to its oq buffers. */
            if(i+2<p->n)q[1].w[30]|=2u;
            memmove(q,q+1,(size_t)(p->n-i-2)*sizeof *q);
            if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
            p->n--;count++;
        }
    }
    return count;
}

/* The shared expert's gate/up projections are independent of the routed
 * gate/up/activation body. prog_nobar already states this dependence contract. */
static unsigned phase_shared_gate_group(prog_t *p,CUdeviceptr router_logits) {
    unsigned count=0,layer=0;if(!producer_quant)return 0;
    for(int i=0;i+2<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1,*c=a+2;
        if(a->w[0]==OP_Q2MOEGN)layer++;
        if(a->w[0]!=OP_Q2MOEGN || a->w[30]!=36u || a->w[31] ||
           !matrix_opcode(b->w[0]) || !matrix_opcode(c->w[0]) ||
           b->w[30]!=4u || c->w[30] || b->w[31] || c->w[31])continue;
        unsigned width=3;
        if(i+3<p->n && a[3].w[0]==OP_FFNACT && a[3].w[1]==1 && !a[3].w[30] && !a[3].w[31]) {
            width=4;c->w[30]=0x10000u;
            ph_ptr(c,0x190,router_logits+513u*4u);c->w[24]=layer+49u;
        }
        a->w[30]=8u|32u|(width<<8);b->w[30]=0;
        count++;i+=(int)width-1;
    }
    return count;
}

static unsigned phase_selection_fuse(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+4<p->n;i++) {
        phase_t *a=p->p+i;
        if(a->w[0]!=OP_IDXSEL || a->w[1]!=16 || a->w[10]!=2)continue;
        int ok=1;
        for(unsigned k=1;k<5;k++) {
            phase_t *b=a+k;
            if(b->w[0]!=OP_IDXSEL || b->w[1]!=16 || b->w[10]!=k+2 ||
               b->w[30] || b->w[31]!=a->w[31] || memcmp(a->w+2,b->w+2,8*sizeof(uint32_t)))ok=0;
        }
        if(!ok)continue;
        a->w[0]=102;a->w[1]=1;
        memmove(a+1,a+5,(size_t)(p->n-i-5)*sizeof *a);
        if(layers)memmove(layers+i+1,layers+i+5,(size_t)(p->n-i-5)*sizeof *layers);
        p->n-=4;count++;
    }
    return count;
}

static unsigned phase_qsa_independent_group(prog_t *p) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_ATTNQKQ || (a->w[30]&~2u)!=4u ||
           b->w[0]!=OP_IDXQK || b->w[30])continue;
        /* Cache append and indexer update are already declared independent.
         * The grouped loader applies each descriptor's own token-offset mask. */
        a->w[30]=(a->w[30]&2u)|8u;count++;i++;
    }
    return count;
}
static unsigned phase_qsa_empty_publish(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+2<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1,*c=a+2;
        if(a->w[0]!=OP_BF16K2560 || a->w[8]!=640 || a->w[30] || a->w[31] ||
           b->w[0]!=OP_XSEND || b->w[6] || b->w[30]!=1 || b->w[31] ||
           c->w[0]!=OP_ATTNQKQ)continue;
        /* The attention owner has no Q rows to send. Its indexer producer can
         * publish that empty participation flag at its existing phase end. */
        a->w[30]|=1u;
        memmove(b,c,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

static unsigned phase_score_selection_fuse(prog_t *p,unsigned request_epoch_base) {
    unsigned count=0;if(!producer_quant || !p->loop)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_IDXSCORE || a->w[1]!=16 || a->w[30] || a->w[31]!=(1u<<10) ||
           b->w[0]!=102 || b->w[1]!=1 || b->w[30] || b->w[31]!=(1u<<8) ||
           gr_pointer(a,6)!=gr_pointer(b,2) || gr_pointer(a,8)!=gr_pointer(b,4) ||
           a->w[10]!=b->w[8] || a->w[11]!=b->w[9])continue;
        a->w[0]=103;a->w[30]=24u;
        /* HST scratch is reused between attention layers and tokens. A ready
         * word must identify this scoring generation, not just contain1. */
        a->w[12]=request_epoch_base+(++count);
        if(count>=16){fprintf(stderr,"too many local QSA generations\n");exit(1);}
        i++;
    }
    return count;
}

static unsigned phase_part_merge_fuse(prog_t *p) {
    unsigned count=0;if(!producer_quant || !p->loop)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_ATTNPARTN || a->w[1]!=32 || a->w[30] || a->w[31]!=(1u<<14) ||
           b->w[0]!=101 || b->w[1]!=24 || b->w[30]!=1 || b->w[31] ||
           gr_pointer(a,8)!=gr_pointer(b,2) || gr_pointer(a,10)!=gr_pointer(b,4) ||
           gr_pointer(a,12)!=gr_pointer(b,6))continue;
        a->w[0]=104;a->w[30]=25u;a->w[16]=++count;
        i++;
    }
    return count;
}

static unsigned phase_ple_fuse(prog_t *p,const dn_bufs_t *buffers,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[31] || b->w[31] || a->w[30]&~2u || b->w[30] || a->w[1]!=4 || b->w[1]!=4)continue;
        if(a->w[0]==OP_RMS2560 && b->w[0]==OP_RMS2560 &&
           gr_pointer(a,6)==buffers->kn && gr_pointer(b,6)==buffers->qn && gr_pointer(b,2)==buffers->S) {
            a->w[30]|=8u;count++;i++;continue;
        }
        if(a->w[0]!=OP_PLEGATE || b->w[0]!=OP_RMS2560 || gr_pointer(a,8)!=gr_pointer(b,2))continue;
        a->w[0]=105;
        memcpy(a->w+12,b->w+4,4*sizeof(uint32_t));a->w[16]=b->w[8];
        memmove(b,b+1,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

static unsigned phase_ple_rows_fuse(prog_t *p,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        if(a->w[0]!=OP_PLEROWS || a->w[1]!=1 || b->w[0]!=OP_ACTQ256 || b->w[1]!=10 ||
           b->w[30] || b->w[31] || gr_pointer(a,8)!=gr_pointer(b,2))continue;
        a->w[0]=106;memcpy(a->w+20,b->w+4,8*sizeof(uint32_t));a->w[28]=b->w[12];
        if(row_balance)a->w[1]=16; /* one complete160-value n-gram head per CTA */
        memmove(b,b+1,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

static unsigned phase_edges_fuse(prog_t *p,unsigned position,int die,int *layers) {
    unsigned count=0;if(!producer_quant)return 0;
    if(p->n>=2 && p->p[0].w[0]==OP_EMBROW && p->p[1].w[0]==OP_XSEND &&
       p->p[1].w[6]==40960 && p->p[1].w[30]==1 && gr_pointer(p->p,4)==gr_pointer(p->p+1,4)) {
        p->p[0].w[30]|=129u;ph_ptr(p->p,0x170,embedding_tables[position]);
        memmove(p->p+1,p->p+2,(size_t)(p->n-2)*sizeof *p->p);
        if(layers)memmove(layers+1,layers+2,(size_t)(p->n-2)*sizeof *layers);
        p->n--;count++;
    } else if(p->n>=3 && p->p[0].w[0]==OP_XSEND && !p->p[0].w[6] && p->p[0].w[30]==1 &&
              p->p[1].w[0]==OP_XCOPY && p->p[1].w[30]==2 &&
              gr_pointer(p->p+1,4)==position_buf(die,position)->S) {
        p->p[2].w[30]|=2u;
        memmove(p->p+1,p->p+2,(size_t)(p->n-2)*sizeof *p->p);
        if(layers)memmove(layers+1,layers+2,(size_t)(p->n-2)*sizeof *layers);
        p->n--;count++;
    }
    for(int i=0;i+1<p->n;i++) {
        phase_t *a=p->p+i,*b=a+1;
        /* Sixteen CTA winners form one128-byte payload. Keep its copy phase
         * so the sender coalesces the payload after the producer grid joins,
         * instead of scattering96 tiny peer writes from the argmax CTAs. */
        if(row_balance && a->w[0]==OP_ARGMAX)continue;
        if((a->w[0]!=OP_ARGMAX && a->w[0]!=OP_TOPK) || a->w[1]!=16 ||
           a->w[30] || a->w[31] || b->w[0]!=OP_XSEND || b->w[30]!=1 || b->w[31] ||
           b->w[6]!=(a->w[0]==OP_ARGMAX?(row_balance?128u:8u):256u) || gr_pointer(a,4)!=gr_pointer(b,4))continue;
        ph_ptr(a,0x170,gr_pointer(b,2));a->w[16]=b->w[7]+b->w[8];a->w[30]|=129u;
        memmove(b,b+1,(size_t)(p->n-i-2)*sizeof *b);
        if(layers)memmove(layers+i+1,layers+i+2,(size_t)(p->n-i-2)*sizeof *layers);
        p->n--;count++;
    }
    return count;
}

/* The following exchange acquire includes this die's own sixteen producer
 * flags. Do not also poll the local grid immediately after publishing those
 * same flags. The footer still fences stores, publishes every block's flag,
 * advances GEN/XG, and hands the descriptor state to the next operation. */
static void phase_reduce_waits(prog_t *p) {
    if(!producer_quant)return;
    for(int i=0;i<p->n;) {
        phase_t *a=p->p+i;
        unsigned encoded=(a->w[30]>>8)&255u;
        unsigned width=(a->w[30]&8u)?(encoded?encoded:2u):1u;
        int next=i+(int)width;
        if(next<p->n && (a->w[30]&1u) && !(a->w[30]&4u) &&
           (p->p[next].w[30]&2u))a->w[30]|=PHASE_EXCHANGE_JOIN_FLAG;
        /* PLE key/query RMS does not feed the following raw-row value GEMV.
         * Its block is joined by that GEMV's subsequent exchange acquire. */
        if(next<p->n && a->w[0]==OP_RMS2560 && width==2 &&
           a[1].w[0]==OP_RMS2560 && p->p[next].w[0]==OP_BF16K2560 &&
           (p->p[next].w[30]&1u))a->w[30]|=4u;
        i=next;
    }
}
