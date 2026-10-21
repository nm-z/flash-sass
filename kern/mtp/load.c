#include "mtp/load.h"
#include "quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct { unsigned first,count; } rows_t;
typedef struct { const gguf_file *file;mtp_alloc_fn allocate;size_t bytes; } loader;
static const gguf_tensor *tensor(const gguf_file *g,const char *suffix,unsigned type,
                                  unsigned k,unsigned rows,unsigned experts) {
    char name[128];snprintf(name,sizeof name,"blk.48.%s",suffix);
    const gguf_tensor *t=gguf_find(g,name);
    unsigned rank=experts>1?3:rows>1?2:1;
    size_t rowbytes=type==GG_Q8_0?(size_t)k/32*34:(size_t)k*(type==GG_F32?4:2);
    if(!t || !t->data || t->type!=type || t->dims[0]!=k ||
       (rank>=2 && (t->nd<2 || t->dims[1]!=rows)) ||
       (rank==3 && (t->nd!=3 || t->dims[2]!=experts)) ||
       (rank==1 && t->elems!=k) || t->elems!=(uint64_t)k*rows*experts ||
       t->nbytes<rowbytes*rows*experts) {
        fprintf(stderr,"mtp: missing or incompatible tensor %s (type=%u shape=%ux%ux%u)\n",name,type,k,rows,experts);
        return NULL;
    }
    return t;
}

/* Reuse the production Q8 repacker. A bounded host buffer streams selected rows
 * directly into the engine-owned allocation, including concatenated matrices. */
static CUresult selected(loader *l,CUdeviceptr *out,const gguf_tensor *a,const gguf_tensor *b,
                          const rows_t *ranges,unsigned nr,unsigned destination_type) {
    if(!a || (b && (a->type!=b->type || a->dims[0]!=b->dims[0]))) return CUDA_ERROR_INVALID_VALUE;
    unsigned k=(unsigned)a->dims[0],source_type=a->type;
    size_t arows=(size_t)(a->elems/k),brows=b?(size_t)(b->elems/k):0,nrows=0;
    size_t source_stride=source_type==GG_Q8_0?(size_t)k/32*34:(size_t)k*(source_type==GG_F32?4:2);
    size_t stride=destination_type==GG_Q8_0?(size_t)b32_tile_stride(GG_Q8_0,(int)k):(size_t)k*(destination_type==GG_F32?4:2);
    if(!stride || (source_type!=destination_type && !(source_type==GG_F32 && destination_type==GG_BF16))) return CUDA_ERROR_INVALID_VALUE;
    for(unsigned r=0;r<nr;r++) {
        if(ranges[r].first>arows+brows || ranges[r].count>arows+brows-ranges[r].first) return CUDA_ERROR_INVALID_VALUE;
        nrows+=ranges[r].count;
    }
    if(!nrows) { *out=0;return CUDA_SUCCESS; }
    size_t bytes=nrows*stride,slack=destination_type==GG_BF16?64*stride:0;
    l->bytes+=bytes+slack;
    if(!l->allocate) { *out=0;return CUDA_SUCCESS; }
    *out=l->allocate(bytes+slack);
    if(!*out) return CUDA_ERROR_OUT_OF_MEMORY;
    const size_t chunk_rows=512;
    uint8_t *buffer=malloc(chunk_rows*stride);
    if(!buffer) return CUDA_ERROR_OUT_OF_MEMORY;
    size_t offset=0;CUresult rc=CUDA_SUCCESS;
    for(unsigned r=0;r<nr && rc==CUDA_SUCCESS;r++) {
        unsigned done=0;
        while(done<ranges[r].count) {
            unsigned n=ranges[r].count-done;if(n>chunk_rows)n=(unsigned)chunk_rows;
            for(unsigned j=0;j<n;j++) {
                size_t row=(size_t)ranges[r].first+done+j;
                const uint8_t *source=row<arows?a->data+row*source_stride:b->data+(row-arows)*source_stride;
                uint8_t *dest=buffer+(size_t)j*stride;
                if(destination_type==GG_Q8_0) {
                    if(repack_b32(GG_Q8_0,source,1,(int)k,dest)<0) { rc=CUDA_ERROR_INVALID_VALUE;break; }
                } else if(source_type==GG_F32 && destination_type==GG_BF16) {
                    for(unsigned i=0;i<k;i++) {
                        uint32_t bits;memcpy(&bits,source+4*i,4);
                        uint16_t rounded=(uint16_t)((bits+0x7fffu+((bits>>16)&1u))>>16);
                        memcpy(dest+2*i,&rounded,2);
                    }
                } else memcpy(dest,source,stride);
            }
            if(rc!=CUDA_SUCCESS) break;
            rc=cuMemcpyHtoD(*out+offset,buffer,(size_t)n*stride);
            if(rc!=CUDA_SUCCESS) break;
            offset+=(size_t)n*stride;done+=n;
        }
    }
    free(buffer);return rc;
}
static CUresult whole(loader *l,CUdeviceptr *out,const char *name,unsigned type,unsigned k,unsigned rows) {
    rows_t range={0,rows};return selected(l,out,tensor(l->file,name,type,k,rows,1),NULL,&range,1,type);
}
#define LOAD(x) do { CUresult result=(x);if(result!=CUDA_SUCCESS)return result; } while(0)
CUresult mtp_load_weights(mtp_weights_t *mw,const gguf_file *g,const split_t *s,
                          const dn_weights_t *target,int owner,mtp_alloc_fn allocate,size_t *bytes) {
    if(!mw||!g||!s||!target||s->n!=6||owner<0||owner>=6) return CUDA_ERROR_INVALID_VALUE;
    memset(mw,0,sizeof *mw);loader l={g,allocate,0};dn_weights_t *w=&mw->block;
    rows_t down={(unsigned)s->dj0,(unsigned)s->djn},head={(unsigned)s->hd0,(unsigned)s->hdn};
    rows_t up[4];for(unsigned lane=0;lane<4;lane++) up[lane]=(rows_t){lane*2560+32u*(unsigned)s->cb0,32u*(unsigned)s->cbn};
    const char *sides[]={"attn","ffn"};
    for(unsigned side=0;side<2;side++) {
        char norm[64],dn[64],inject[64],un[64];
        snprintf(norm,sizeof norm,"hc_%s_norm.weight",sides[side]);
        snprintf(dn,sizeof dn,"hc_%s_down.weight",sides[side]);
        snprintf(inject,sizeof inject,"hc_%s_inject.weight",sides[side]);
        snprintf(un,sizeof un,"hc_%s_up.weight",sides[side]);
        CUdeviceptr *wn=side?&w->hc_ffn_norm:&w->hc_attn_norm;
        CUdeviceptr *wd=side?&w->hc_ffn_dj:&w->hc_attn_dj;
        CUdeviceptr *wu=side?&w->hc_ffn_up:&w->hc_attn_up;
        LOAD(whole(&l,wn,norm,GG_F32,10240,1));
        LOAD(selected(&l,wd,tensor(g,dn,GG_Q8_0,10240,320,1),tensor(g,inject,GG_Q8_0,10240,4,1),&down,1,GG_Q8_0));
        LOAD(selected(&l,wu,tensor(g,un,GG_Q8_0,320,10240,1),NULL,up,4,GG_Q8_0));
    }
    LOAD(whole(&l,&mw->hnorm,"nextn.hnorm.weight",GG_F32,10240,1));
    LOAD(whole(&l,&mw->enorm,"nextn.enorm.weight",GG_F32,2560,1));
    LOAD(whole(&l,&mw->eh_proj,"nextn.eh_proj.weight",GG_Q8_0,5120,2560));
    LOAD(whole(&l,&mw->head_norm,"nextn.hc_head_norm.weight",GG_F32,10240,1));
    LOAD(selected(&l,&mw->head_down,tensor(g,"nextn.hc_head_down.weight",GG_Q8_0,10240,320,1),NULL,&head,1,GG_Q8_0));
    LOAD(selected(&l,&mw->head_up,tensor(g,"nextn.hc_head_up.weight",GG_Q8_0,320,10240,1),NULL,up,4,GG_Q8_0));
    rows_t router={0,513};
    LOAD(selected(&l,&w->ffn_gate_inp_x,tensor(g,"ffn_gate_inp.weight",GG_F32,2560,512,1),tensor(g,"ffn_gate_inp_shexp.weight",GG_F32,2560,1,1),&router,1,GG_BF16));
    const char *experts[]={"ffn_gate_exps.weight","ffn_up_exps.weight","ffn_down_exps.weight"};
    CUdeviceptr *dest[]={&w->ffn_gate_exps,&w->ffn_up_exps,&w->ffn_down_exps};
    for(unsigned i=0;i<3;i++) {
        unsigned k=i==2?640:2560,rows=i==2?2560:640;
        rows_t range={(unsigned)s->e0*rows,(unsigned)s->en*rows};
        LOAD(selected(&l,dest[i],tensor(g,experts[i],GG_Q8_0,k,rows,512),NULL,&range,1,GG_Q8_0));
    }
    uint32_t map[512];for(unsigned i=0;i<512;i++) map[i]=i>=(unsigned)s->e0 && i<(unsigned)(s->e0+s->en)?i-(unsigned)s->e0:0xffffffffu;
    l.bytes+=sizeof map;
    if(allocate) {
        w->emap=allocate(sizeof map);
        if(!w->emap)return CUDA_ERROR_OUT_OF_MEMORY;
        LOAD(cuMemcpyHtoD(w->emap,map,sizeof map));
    }
    if(s->shexp) {
        LOAD(whole(&l,&w->ffn_gate_shexp,"ffn_gate_shexp.weight",GG_Q8_0,2560,640));
        LOAD(whole(&l,&w->ffn_up_shexp,"ffn_up_shexp.weight",GG_Q8_0,2560,640));
        LOAD(whole(&l,&w->ffn_down_shexp,"ffn_down_shexp.weight",GG_Q8_0,640,2560));
    }
    if(s->d==owner) {
        LOAD(whole(&l,&w->attn_q,"attn_q.weight",GG_Q8_0,2560,12288));
        LOAD(whole(&l,&w->attn_k,"attn_k.weight",GG_Q8_0,2560,512));
        LOAD(whole(&l,&w->attn_v,"attn_v.weight",GG_Q8_0,2560,512));
        LOAD(whole(&l,&w->attn_q_norm,"attn_q_norm.weight",GG_F32,256,1));
        LOAD(whole(&l,&w->attn_k_norm,"attn_k_norm.weight",GG_F32,256,1));
        LOAD(whole(&l,&w->indexer_qn,"indexer.q_norm.weight",GG_F32,128,1));
        LOAD(whole(&l,&w->indexer_kn,"indexer.k_norm.weight",GG_F32,128,1));
        rows_t index={0,640};
        LOAD(selected(&l,&w->indexer_x,tensor(g,"indexer.q_proj.weight",GG_BF16,2560,512,1),tensor(g,"indexer.k_proj.weight",GG_BF16,2560,128,1),&index,1,GG_BF16));
    }
    rows_t output={(unsigned)s->y0,(unsigned)s->yn};
    LOAD(selected(&l,&w->attn_output,tensor(g,"attn_output.weight",GG_Q8_0,6144,2560,1),NULL,&output,1,GG_Q8_0));
    w->output=target->output;w->obases=target->obases;
    if(bytes)*bytes=l.bytes;
    return CUDA_SUCCESS;
}
#undef LOAD

const gguf_tensor *mtp_embedding_tensor(const gguf_file *g) {
    const gguf_tensor *t=gguf_find(g,"token_embd.weight");
    return t && t->data && t->type==GG_Q8_0 && t->nd==2 && t->dims[0]==2560 &&
           t->dims[1]==248320 && t->nbytes>=248320ull*2720?t:NULL;
}
int mtp_embedding_row(const gguf_tensor *t,uint32_t token,float dst[2560]) {
    if(!t || token>=248320) return -1;
    return dequant(GG_Q8_0,t->data+(size_t)token*2720,dst,2560);
}
