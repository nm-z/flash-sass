#include "gguf_model.h"
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int error(char *text,size_t size,const char *format,...) {
    va_list ap;va_start(ap,format);vsnprintf(text,size,format,ap);va_end(ap);return -1;
}
size_t dflash_tensor_bytes(const gguf_tensor *t) {
    uint32_t elements,bytes;gguf_block(t->type,&elements,&bytes);
    if(!elements || !bytes || t->elems%elements || t->elems/elements>SIZE_MAX/bytes)return 0;
    return (size_t)(t->elems/elements)*bytes;
}
static int integer(const gguf_file *g,const char *key,int64_t expected,char *err,size_t n) {
    int64_t value;
    if(gguf_get_int(g,key,&value) || value!=expected)return error(err,n,"%s must be %lld",key,(long long)expected);
    return 0;
}
static int real(const gguf_file *g,const char *key,float *out,char *err,size_t n) {
    const gguf_kv *v=gguf_get(g,key);
    if(!v || v->type!=6)return error(err,n,"missing float metadata %s",key);
    memcpy(out,v->val,4);return 0;
}
static const gguf_tensor *tensor(dflash_gguf_model *m,const char *name,unsigned k,unsigned rows,
                               unsigned third,int matrix,char *err,size_t n) {
    const gguf_tensor *t=gguf_find(&m->file,name);
    unsigned dimensions=third?3:rows?2:1;
    if(!t || t->nd!=dimensions || t->dims[0]!=k ||
       (rows && t->dims[1]!=rows) || (third && t->dims[2]!=third)) {
        error(err,n,"DFlash2 tensor shape: %s",name);return NULL;
    }
    if((matrix && t->type!=GG_Q4_K && t->type!=GG_Q6_K) || (!matrix && t->type!=GG_F32)) {
        error(err,n,"DFlash2 tensor format: %s type=%u",name,t->type);return NULL;
    }
    size_t bytes=dflash_tensor_bytes(t);
    if(!bytes || !t->data || bytes>t->nbytes) {error(err,n,"DFlash2 tensor extent: %s",name);return NULL;}
    m->packed_bytes+=bytes;
    if(matrix)m->matrix_bytes+=bytes;else m->vector_bytes+=bytes;
    m->q4_tensors+=t->type==GG_Q4_K;m->q6_tensors+=t->type==GG_Q6_K;m->f32_tensors+=t->type==GG_F32;
    return t;
}
void dflash_gguf_close(dflash_gguf_model *m) {
    if(m->file.map)gguf_close(&m->file);
    memset(m,0,sizeof *m);m->file.fd=-1;
}
int dflash_gguf_open(dflash_gguf_model *m,const char *path,char *err,size_t n) {
    memset(m,0,sizeof *m);m->file.fd=-1;
    if(gguf_open(&m->file,path))return error(err,n,"cannot map DFlash2 GGUF: %s",path);
    uint64_t length;const char *architecture=gguf_get_str(&m->file,"general.architecture",&length);
    if(!architecture || length!=6 || memcmp(architecture,"dflash",6)) {
        error(err,n,"selected draft is not a dflash GGUF");goto fail;
    }
    struct {const char *key;int64_t value;} shape[]={
        {"dflash.block_count",5},{"dflash.embedding_length",2560},{"dflash.feed_forward_length",7680},
        {"dflash.attention.head_count",32},{"dflash.attention.head_count_kv",8},
        {"dflash.attention.key_length",128},{"dflash.attention.value_length",128},
        {"dflash.attention.causal",0},{"dflash.block_size",8},{"dflash.conv_kernel_size",2},
        {"dflash.conv_group_size",16},{"dflash.selector_rank",256},{"dflash.selector_top_k",16},
        {"dflash.attention.sliding_window",2048}
    };
    for(unsigned i=0;i<sizeof shape/sizeof *shape;i++)if(integer(&m->file,shape[i].key,shape[i].value,err,n))goto fail;
    if(real(&m->file,"dflash.attention.layer_norm_rms_epsilon",&m->epsilon,err,n) ||
       real(&m->file,"dflash.rope.freq_base",&m->rope_theta,err,n))goto fail;
    m->capacity=8;m->window=2048;m->mask_token=248077;
    const uint8_t *values;uint32_t type;uint64_t count;
    if(gguf_get_arr(&m->file,"dflash.target_layers",&type,&count,&values) || count!=5 || (type!=4 && type!=5)) {
        error(err,n,"DFlash2 target-layer metadata");goto fail;
    }
    for(unsigned i=0;i<5;i++) {
        uint32_t layer;memcpy(&layer,values+4*i,4);
        if(!layer || layer>48 || (i && layer<=m->target_layer[i-1]+1)) {
            error(err,n,"DFlash2 completed target-layer index");goto fail;
        }
        m->target_layer[i]=layer-1; /* GGUF hidden-state index includes embeddings. */
    }
#define GET(field,name,k,rows,third,matrix) do { (field)=tensor(m,name,k,rows,third,matrix,err,n);if(!(field))goto fail; }while(0)
    GET(m->feature_projection,"fc.weight",12800,2560,0,1);
    GET(m->feature_norm,"enc.output_norm.weight",2560,0,0,0);
    GET(m->output_norm,"output_norm.weight",2560,0,0,0);
    GET(m->selector_hidden,"selector_hidden.weight",2560,256,0,1);
    GET(m->selector_predecessor,"selector_predecessor.weight",256,248320,0,1);
    GET(m->selector_successor,"selector_successor.weight",256,248320,0,1);
    for(unsigned i=0;i<5;i++) {
        dflash_layer_weights *w=m->layer+i;char key[96];
#define LGET(field,suffix,k,rows,third,matrix) do { snprintf(key,sizeof key,"blk.%u.%s",i,suffix);GET(w->field,key,k,rows,third,matrix); }while(0)
        LGET(attn_norm,"attn_norm.weight",2560,0,0,0);
        LGET(attn_conv_base,"attn_conv_base",2560,2,2,0);
        LGET(attn_conv_projection,"attn_conv_proj.weight",2560,640,0,1);
        LGET(q,"attn_q.weight",2560,4096,0,1);LGET(k,"attn_k.weight",2560,1024,0,1);
        LGET(v,"attn_v.weight",2560,1024,0,1);
        LGET(q_norm,"attn_q_norm.weight",128,0,0,0);LGET(k_norm,"attn_k_norm.weight",128,0,0,0);
        LGET(attention_output,"attn_output.weight",4096,2560,0,1);
        LGET(ffn_norm,"ffn_norm.weight",2560,0,0,0);
        LGET(ffn_conv_base,"ffn_conv_base",2560,2,2,0);
        LGET(ffn_conv_projection,"ffn_conv_proj.weight",2560,640,0,1);
        LGET(gate,"ffn_gate.weight",2560,7680,0,1);LGET(up,"ffn_up.weight",2560,7680,0,1);
        LGET(down,"ffn_down.weight",7680,2560,0,1);
#undef LGET
    }
#undef GET
    if(m->q4_tensors+m->q6_tensors+m->f32_tensors!=m->file.ntensors) {
        error(err,n,"unmapped DFlash2 tensors");goto fail;
    }
    return 0;
fail:
    dflash_gguf_close(m);return -1;
}
