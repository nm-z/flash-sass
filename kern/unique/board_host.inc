/* Exact producer-CTA forwarding plans for ordinary allgathers. Each sender
 * writes both local-board dies and one die on each remote board. The two
 * redundant destination slots alias its own local destination; they perform
 * local identical-value stores, not additional interboard writes. */
#include "../dflash/target.h"
typedef struct {CUdeviceptr table,base[6][4];unsigned die,columns;} board_table_record;
typedef struct {CUdeviceptr source,destination;uint32_t bytes,padding;} board_transfer;
typedef struct {CUdeviceptr transfers;uint32_t count,active;} board_cta_plan;
typedef struct {board_transfer *v;size_t n,cap;} board_list;
static board_table_record *board_tables;static size_t board_table_count,board_table_capacity;
static CUdeviceptr board_flags[6],board_stage_targets[6];
static struct {const prog_t *source;CUdeviceptr image[6];size_t capacity[6];} board_cache[9];
static void board_error(const char *what) {
    fprintf(stderr,"engine: board transport: %s\n",what);
    if(in_init)longjmp(init_jb,1);longjmp(run_jb,1);
}
static CUdeviceptr board_pointer(const phase_t *p,unsigned word) {
    return (CUdeviceptr)p->w[word]|((CUdeviceptr)p->w[word+1]<<32);
}
static unsigned board_span(const phase_t *p) {
    return !(p->w[30]&8u)?1:((p->w[30]>>8)&255u)?((p->w[30]>>8)&255u):2;
}
static void board_table_upload(unsigned d,CUdeviceptr address,const CUdeviceptr *original,unsigned columns) {
    if(!paired_columns){CK(cuMemcpyHtoD(address,original,6u*columns*sizeof *original));return;}
    if(columns!=1 && columns!=4)board_error("unsupported destination table width");
    if(board_table_count==board_table_capacity) {
        size_t n=board_table_capacity?board_table_capacity*2:128;
        void *p=realloc(board_tables,n*sizeof *board_tables);if(!p)board_error("table registry allocation");
        board_tables=p;board_table_capacity=n;
    }
    board_table_record *r=board_tables+board_table_count++;memset(r,0,sizeof *r);
    r->table=address;r->die=d;r->columns=columns;CUdeviceptr folded[24];
    for(unsigned peer=0;peer<6;peer++)for(unsigned c=0;c<columns;c++) {
        r->base[peer][c]=original[peer*columns+c];
        folded[peer*columns+c]=original[((peer&1u) && peer/2!=d/2?d:peer)*columns+c];
    }
    CK(cuMemcpyHtoD(address,folded,6u*columns*sizeof *folded));
}
static const board_table_record *board_find(unsigned d,CUdeviceptr address) {
    for(size_t i=0;i<board_table_count;i++)if(board_tables[i].die==d && board_tables[i].table==address)return board_tables+i;
    board_error("publication has no registered destination table");return NULL;
}
static void board_append(board_list *l,board_transfer t) {
    if(!t.bytes)return;
    if((t.source|t.destination|t.bytes)&3u)board_error("unaligned forwarding range");
    if(l->n && l->v[l->n-1].source+l->v[l->n-1].bytes==t.source &&
       l->v[l->n-1].destination+l->v[l->n-1].bytes==t.destination) {
        l->v[l->n-1].bytes+=t.bytes;return;
    }
    if(l->n==l->cap) {
        size_t n=l->cap?l->cap*2:16;void *p=realloc(l->v,n*sizeof *l->v);
        if(!p)board_error("forwarding list allocation");l->v=p;l->cap=n;
    }
    l->v[l->n++]=t;
}
static void board_range(board_list list[3][16],unsigned sender,unsigned cta,
                        CUdeviceptr table,unsigned field,unsigned offset,unsigned bytes) {
    if(cta>=16)board_error("producer CTA outside resident grid");
    const board_table_record *r=board_find(sender,table);
    if(field>=r->columns)board_error("publication field outside destination table");
    for(unsigned board=0;board<3;board++)if(board!=sender/2)
        board_append(&list[board][cta],(board_transfer){r->base[board*2][field]+offset,r->base[board*2+1][field]+offset,bytes,0});
}
static void board_quant(board_list list[3][16],unsigned sender,unsigned cta,CUdeviceptr table,unsigned first,unsigned n) {
    if((first|n)&31u)board_error("partial quantization group in publication");
    board_range(list,sender,cta,table,0,first*2,n*2);
    board_range(list,sender,cta,table,1,first/32*4,n/32*4);
    board_range(list,sender,cta,table,2,first/16*4,n/16*4);
    board_range(list,sender,cta,table,3,first/32*4,n/32*4);
}
static unsigned board_matrix_tile(unsigned op) {
    switch(op) {
    case OP_BF16K2560:return 8;
    case OP_Q2K2560:case OP_Q3KK2560:case OP_Q4KK2560:case OP_Q5KK2560:
    case OP_Q6KK2560:case OP_Q6KK1536ACC:case OP_IQ4K2560:return 16;
    case OP_Q3KK6144:case OP_Q4KK6144:case OP_Q5KK6144:case OP_IQ4K6144:return 8;
    default:return 0;
    }
}
static void board_operation(board_list list[3][16],unsigned d,const phase_t *a,int native) {
    unsigned op=a->w[0],flags=a->w[30];
    if(op==OP_PAIRED_DOWN || op==OP_PAIRED_DOWN4)return; /* own tiled hierarchy */
    if(op==OP_EMBROW) {
        for(unsigned c=0;c<a->w[1];c++)for(unsigned branch=0;branch<4;branch++)
            board_range(list,d,c,board_pointer(a,14),0,4*(a->w[16]+c*32+branch*2560),128);
        return;
    }
    if(op==OP_UNIQUE_ROUTER) {
        board_range(list,d,0,board_pointer(a,12),0,a->w[15]*128,84);return;
    }
    if(op==OP_BF16HCRX || op==OP_BF16HCRWX || op==94) {
        for(unsigned c=0;c<16;c++)for(unsigned row=c*4;row<a->w[8];row+=64) {
            unsigned n=a->w[8]-row;if(n>4)n=4;
            board_range(list,d,c,board_pointer(a,6),0,a->w[12]+row*4,n*4);
        }
        return;
    }
    if(op==96) {
        unsigned cols=a->w[8]/4;
        for(unsigned c=0;c*32<cols;c++) {
            board_range(list,d,c,board_pointer(a,14),0,a->w[12]+c*128,128);
            board_quant(list,d,c,board_pointer(a,22),(a->w[12]&65535u)/4+c*32,32);
        }
        return;
    }
    if(op==95 || op==98) {
        unsigned keys=a->w[29]>>8,first=a->w[29]&255u;
        if(!keys)board_error("GDN unique head layout absent");
        for(unsigned c=0;c<a->w[1];c++) {
            unsigned head=first+c%keys+16*(c/keys);
            board_quant(list,d,c,board_pointer(a,op==95?8:26),head*128,128);
        }
        return;
    }
    if(op==101) {
        for(unsigned c=0;c<a->w[1];c++)board_quant(list,d,c,board_pointer(a,12),a->w[14]+c*256,256);
        return;
    }
    if(op==OP_XSEND) {
        for(unsigned c=0;c<16;c++)for(unsigned offset=c*1024;offset<a->w[6];offset+=16384) {
            unsigned n=a->w[6]-offset;if(n>1024)n=1024;
            board_range(list,d,c,board_pointer(a,2),0,a->w[7]+a->w[8]+offset,n);
        }
        return;
    }
    if(flags&PHASE_BALANCED_PUBLISH) {
        unsigned rows=a->w[op==OP_BF16K2560?8:10];
        unsigned tile=(flags&PHASE_BALANCED_TILE32)?32:8;
        if(native)tile=8;
        for(unsigned c=0;c<16;c++)for(unsigned start=c*tile;start<rows;start+=16*tile)
            for(unsigned row=start;row<start+tile && row<rows;row++) {
                unsigned global=balance_global_row(row,d,6,8)+a->w[25];
                if(a->w[23]!=1)board_error("routed intermediate unexpectedly published");
                board_range(list,d,c,board_pointer(a,20),0,global*4,4);
            }
        return;
    }
    if(flags&128u) {
        unsigned tile=board_matrix_tile(op),rows=a->w[op==OP_BF16K2560?8:10];
        if(!tile)board_error("matrix publication geometry unavailable");
        for(unsigned c=0;c<16;c++)for(unsigned row=c*tile;row<rows;row+=16*tile) {
            unsigned n=rows-row;if(n>tile)n=tile;
            board_range(list,d,c,board_pointer(a,22),0,a->w[24]+row*4,n*4);
        }
        return;
    }
    /* Head winners are read directly from their original owner by AMAX.
     * Other independent matrix members have no publication of their own. */
}
static void board_bundle(board_list list[3][16],unsigned d,const phase_t *a,unsigned span,int native) {
    unsigned op=a->w[0];
    if(op==OP_PAIRED_DOWN4)return;
    if(op==108 || op==109 || op==110 || op==OP_DFLASH_ROUTER4 || (op>=OP_DFLASH_DN4 && op<=OP_DFLASH_BF16_4)) {
        unsigned count=a->w[2],step=a->w[3];
        if(!count || !step || span!=1+count*step)board_error("native B4 descriptor span");
        for(unsigned p=0;p<count;p++)board_bundle(list,d,a+1+p*step,step,op!=108 && op!=110);
    } else if(op==107) {
        if(span!=8 || a[7].w[0]!=101)board_error("QSA publication descriptor span");
        board_operation(list,d,a+7,0);
    } else if(span>1 && !(a->w[30]&16u)) {
        for(unsigned j=0;j<span;j++)board_operation(list,d,a+j,native);
    } else board_operation(list,d,a,native);
}
static void board_bind_programs(const prog_t *P,unsigned positions) {
    if(!paired_columns)return;
    if(positions<1 || positions>8)board_error("program position capacity");
    if(board_cache[positions].source!=P) {
        size_t descriptors=(size_t)P[0].n,header_bytes=descriptors*16*sizeof(board_cta_plan);
        board_cta_plan *headers[6]={0};board_list records[3]={{0}},phase[3][16]={{{0}}};
        for(unsigned d=0;d<6;d++) {
            if(P[d].n!=P[0].n)board_error("phase geometry differs across dies");
            headers[d]=calloc(descriptors*16,sizeof(board_cta_plan));if(!headers[d])board_error("plan headers allocation");
        }
        for(unsigned index=0;index<descriptors;) {
            unsigned span=board_span(P[0].p+index);
            if(!span || span>descriptors-index)board_error("program span outside descriptors");
            if(P[0].p[index].w[30]&1u) {
                for(unsigned d=0;d<6;d++) {
                    if(board_span(P[d].p+index)!=span || !(P[d].p[index].w[30]&1u))board_error("publication geometry differs across dies");
                    board_bundle(phase,d,P[d].p+index,span,0);
                }
                int active=0;for(unsigned b=0;b<3;b++)for(unsigned c=0;c<16;c++)active|=phase[b][c].n!=0;
                for(unsigned d=0;d<6;d++)for(unsigned c=0;c<16;c++)headers[d][index*16+c].active=(unsigned)active;
                for(unsigned b=0;b<3;b++)for(unsigned c=0;c<16;c++) {
                    board_cta_plan *h=headers[b*2]+index*16+c;
                    h->transfers=header_bytes+records[b].n*sizeof(board_transfer);h->count=(uint32_t)phase[b][c].n;
                    /* Do not merge across CTA lists: their readiness differs. */
                    size_t total=records[b].n+phase[b][c].n;
                    if(total>records[b].cap) {
                        size_t cap=total*2+64;void *p=realloc(records[b].v,cap*sizeof(board_transfer));
                        if(!p)board_error("plan records allocation");records[b].v=p;records[b].cap=cap;
                    }
                    if(phase[b][c].n)memcpy(records[b].v+records[b].n,phase[b][c].v,phase[b][c].n*sizeof(board_transfer));
                    records[b].n=total;phase[b][c].n=0;
                }
            }
            index+=span;
        }
        for(unsigned d=0;d<6;d++) {
            size_t bytes=header_bytes+((d&1)?0:records[d/2].n*sizeof(board_transfer));
            CK(cuCtxSetCurrent(ctx[d]));
            if(board_cache[positions].capacity[d]<bytes) {
                board_cache[positions].image[d]=dalloc(bytes);board_cache[positions].capacity[d]=bytes;
            }
            CUdeviceptr base=board_cache[positions].image[d];
            for(size_t i=0;i<descriptors*16;i++)if(headers[d][i].count)headers[d][i].transfers+=base;
            CK(cuMemcpyHtoD(base,headers[d],header_bytes));free(headers[d]);
            if(!(d&1) && records[d/2].n)CK(cuMemcpyHtoD(base+header_bytes,records[d/2].v,records[d/2].n*sizeof(board_transfer)));
        }
        for(unsigned b=0;b<3;b++){free(records[b].v);for(unsigned c=0;c<16;c++)free(phase[b][c].v);}
        board_cache[positions].source=P;
    }
    for(unsigned d=0;d<6;d++){CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(dCTL[d]+0xe0,&board_cache[positions].image[d],8));}
}
static void board_init(void) {
    if(!paired_columns)return;
    for(unsigned d=0;d<6;d++){CK(cuCtxSetCurrent(ctx[d]));board_flags[d]=dalloc(512);}
    for(unsigned d=0;d<6;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CUdeviceptr targets[3];
        for(unsigned b=0;b<3;b++)targets[b]=board_flags[2*b]+d*64;
        board_stage_targets[d]=dalloc(sizeof targets);CK(cuMemcpyHtoD(board_stage_targets[d],targets,sizeof targets));
        CUdeviceptr control[3]={board_flags[d],board_stage_targets[d],board_flags[d&1?d:d+1]+384};
        CK(cuMemcpyHtoD(dCTL[d]+0xe8,control,sizeof control));
    }
}
static void board_forget_programs(void) {for(unsigned p=0;p<9;p++)board_cache[p].source=NULL;}
static void board_fini(void) {
    free(board_tables);board_tables=NULL;board_table_count=board_table_capacity=0;
    memset(board_cache,0,sizeof board_cache);memset(board_flags,0,sizeof board_flags);memset(board_stage_targets,0,sizeof board_stage_targets);
}
