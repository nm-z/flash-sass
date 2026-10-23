#include "resident.h"
#include "gguf_model.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { MATRIX,NORM,CONV,ADD,SWIGLU,ROPE,CACHE,ATTENTION,ATTENTION_MERGE,
       FEATURES,EMBEDDING,TOPK_LOGITS,TOPK_PAIRS,HEAD_MERGE,SELECT,NFUNCTIONS };
static const char *names[NFUNCTIONS]={"df4_matrix","df4_norm","df4_conv","df4_add","df4_swiglu",
    "df4_rope","df4_cache_put","df4_attention_parts","df4_attention_merge","df4_features",
    "df4_embedding","df4_topk_logits","df4_topk_pairs","df4_head_merge","df4_select"};
struct dflash_resident {
    dflash_gguf_model model;
    dflash_target_binding target[6];
    unsigned rank,context;
    CUmodule module[6];CUfunction fn[6][NFUNCTIONS];CUstream stream[6];
    CUmodule q4k_module;CUfunction q4k;   /* optional PTX four-row Q4_K matrix */
    CUmodule head_module[6];CUfunction q5t[6];   /* same PTX module: three-position head */
    CUevent hidden_ready,head_ready[6];
    CUgraph backbone_graph,append_graph,head_graph[6],select_graph;
    CUgraphExec backbone,append,head[6],select;
    struct {unsigned rank;CUdeviceptr pointer;} allocation[128];unsigned allocations;
    CUdeviceptr arena,control,embedding_table,frequency,feature_ring;
    CUdeviceptr x,norm,input,coeff,projected,temp,q,k,v,attention,gate,up;
    CUdeviceptr features,selector,partial,statistics,tokens,candidates,head_sources;
    CUdeviceptr cache_k[5],cache_v[5];
    CUdeviceptr head_input[6],logits[6],top_a[6],top_b[6],top_final[6];
    dflash_device_control *host_control;uint32_t *host_tokens;
};
static int failure(CUresult code,const char *operation,char *err,size_t n) {
    const char *message="CUDA error";cuGetErrorString(code,&message);
    snprintf(err,n,"DFlash2 %s: %s",operation,message);return -1;
}
#define TRY(call) do { CUresult result=(call);if(result!=CUDA_SUCCESS)return failure(result,#call,err,nerr); }while(0)
static int allocate(dflash_resident *r,unsigned rank,CUdeviceptr *out,size_t bytes,char *err,size_t nerr) {
    if(r->allocations>=128){snprintf(err,nerr,"DFlash2 allocation table capacity");return -1;}
    TRY(cuCtxSetCurrent(r->target[rank].context));TRY(cuMemAlloc(out,bytes));
    r->allocation[r->allocations].rank=rank;r->allocation[r->allocations++].pointer=*out;return 0;
}
static CUdeviceptr weight(dflash_resident *r,const gguf_tensor *t) {return r->arena+t->off;}
static int launch(dflash_resident *r,unsigned rank,unsigned fn,unsigned gx,unsigned gy,unsigned threads,void **args,char *err,size_t nerr) {
    TRY(cuLaunchKernel(r->fn[rank][fn],gx,gy,1,threads,1,1,0,r->stream[rank],args,NULL));return 0;
}
static int matrix_reference(dflash_resident *r,const gguf_tensor *t,CUdeviceptr input,CUdeviceptr output,char *err,size_t nerr) {
    CUdeviceptr w=weight(r,t);unsigned rows=(unsigned)t->dims[1],k=(unsigned)t->dims[0],count=4,type=t->type;
    unsigned stride=(unsigned)(dflash_tensor_bytes(t)/rows);
    void *args[]={&w,&input,&output,&rows,&k,&count,&type,&stride};
    return launch(r,r->rank,MATRIX,(rows+3)/4,1,128,args,err,nerr);
}
/* Each 256-thread CTA covers 32 rows: eight warps of four rows. */
static int matrix(dflash_resident *r,const gguf_tensor *t,CUdeviceptr input,CUdeviceptr output,char *err,size_t nerr) {
    if(!r->q4k || t->type!=DFLASH_Q4_K || t->dims[0]%256)return matrix_reference(r,t,input,output,err,nerr);
    CUdeviceptr w=weight(r,t);unsigned rows=(unsigned)t->dims[1],k=(unsigned)t->dims[0];
    unsigned stride=(unsigned)(dflash_tensor_bytes(t)/rows);
    void *args[]={&w,&input,&output,&rows,&k,&stride};
    TRY(cuLaunchKernel(r->q4k,(rows+31)/32,1,1,256,1,1,0,r->stream[r->rank],args,NULL));return 0;
}
/* FLASH_DFLASH_MATRIX_CHECK=1: compare both kernels on one input per Q4_K
 * matrix of layer 0 before capturing graphs. Diagnostic only. */
static int matrix_check(dflash_resident *r,char *err,size_t nerr) {
    if(!r->q4k || !getenv("FLASH_DFLASH_MATRIX_CHECK") || !atoi(getenv("FLASH_DFLASH_MATRIX_CHECK")))return 0;
    dflash_layer_weights *w=r->model.layer;
    const gguf_tensor *list[]={w->q,w->k,w->gate,w->down,w->attention_output,w->attn_conv_projection};
    TRY(cuCtxSetCurrent(r->target[r->rank].context));
    for(unsigned i=0;i<sizeof list/sizeof *list;i++) {
        const gguf_tensor *t=list[i];if(t->type!=DFLASH_Q4_K)continue;
        unsigned rows=(unsigned)t->dims[1],k=(unsigned)t->dims[0];
        float *x=malloc((size_t)4*k*4),*a=malloc((size_t)4*rows*4),*b=malloc((size_t)4*rows*4);
        if(!x||!a||!b){free(x);free(a);free(b);snprintf(err,nerr,"DFlash2 check allocation");return -1;}
        uint32_t s=12345u;for(size_t j=0;j<(size_t)4*k;j++){s=s*1664525u+1013904223u;x[j]=((float)(s>>8)/16777216.f-.5f);}
        CUdeviceptr dx,da,db;TRY(cuMemAlloc(&dx,(size_t)4*k*4));TRY(cuMemAlloc(&da,(size_t)4*rows*4));TRY(cuMemAlloc(&db,(size_t)4*rows*4));
        TRY(cuMemcpyHtoD(dx,x,(size_t)4*k*4));
        if(matrix_reference(r,t,dx,da,err,nerr) || matrix(r,t,dx,db,err,nerr))return -1;
        TRY(cuStreamSynchronize(r->stream[r->rank]));
        TRY(cuMemcpyDtoH(a,da,(size_t)4*rows*4));TRY(cuMemcpyDtoH(b,db,(size_t)4*rows*4));
        double worst=0,scale=0;for(size_t j=0;j<(size_t)4*rows;j++){double e=fabs((double)a[j]-b[j]);if(e>worst)worst=e;if(fabs(a[j])>scale)scale=fabs(a[j]);}
        fprintf(stderr,"dflash_matrix_check tensor=%u rows=%u k=%u max_abs=%.6g scale=%.6g\n",i,rows,k,worst,scale);
        cuMemFree(dx);cuMemFree(da);cuMemFree(db);free(x);free(a);free(b);
    }
    return 0;
}
/* Same comparison for the three-position target-head shard on the draft die. */
static int head_check(dflash_resident *r,char *err,size_t nerr) {
    unsigned d=r->rank;
    if(!r->q5t[d] || !getenv("FLASH_DFLASH_MATRIX_CHECK") || !atoi(getenv("FLASH_DFLASH_MATRIX_CHECK")))return 0;
    TRY(cuCtxSetCurrent(r->target[d].context));
    unsigned rows=r->target[d].output_rows,k=2560,count=3,type=DFLASH_TARGET_Q5_TILE,stride=r->target[d].output_stride;
    float *x=malloc((size_t)3*k*4),*a=malloc((size_t)3*rows*4),*b=malloc((size_t)3*rows*4);
    if(!x||!a||!b){free(x);free(a);free(b);snprintf(err,nerr,"DFlash2 head check allocation");return -1;}
    uint32_t s=777u;for(size_t j=0;j<(size_t)3*k;j++){s=s*1664525u+1013904223u;x[j]=((float)(s>>8)/16777216.f-.5f);}
    CUdeviceptr dx,da,db;TRY(cuMemAlloc(&dx,(size_t)3*k*4));TRY(cuMemAlloc(&da,(size_t)3*rows*4));TRY(cuMemAlloc(&db,(size_t)3*rows*4));
    TRY(cuMemcpyHtoD(dx,x,(size_t)3*k*4));
    void *old[]={&r->target[d].output_weights,&dx,&da,&rows,&k,&count,&type,&stride};
    if(launch(r,d,MATRIX,(rows+3)/4,1,128,old,err,nerr))return -1;
    void *now[]={&r->target[d].output_weights,&dx,&db,&rows,&k,&stride};
    TRY(cuLaunchKernel(r->q5t[d],(rows+31)/32,1,1,256,1,1,0,r->stream[d],now,NULL));
    TRY(cuStreamSynchronize(r->stream[d]));
    TRY(cuMemcpyDtoH(a,da,(size_t)3*rows*4));TRY(cuMemcpyDtoH(b,db,(size_t)3*rows*4));
    double worst=0,scale=0;for(size_t j=0;j<(size_t)3*rows;j++){double e=fabs((double)a[j]-b[j]);if(e>worst)worst=e;if(fabs(a[j])>scale)scale=fabs(a[j]);}
    fprintf(stderr,"dflash_head_check die=%u rows=%u max_abs=%.6g scale=%.6g\n",d,rows,worst,scale);
    cuMemFree(dx);cuMemFree(da);cuMemFree(db);free(x);free(a);free(b);
    return 0;
}
static int norm(dflash_resident *r,CUdeviceptr input,CUdeviceptr output,const gguf_tensor *t,unsigned width,unsigned vectors,char *err,size_t nerr) {
    CUdeviceptr w=weight(r,t);float epsilon=r->model.epsilon;
    void *args[]={&input,&output,&w,&width,&vectors,&epsilon};
    return launch(r,r->rank,NORM,vectors,1,256,args,err,nerr);
}
static int conv(dflash_resident *r,CUdeviceptr input,CUdeviceptr output,const gguf_tensor *t,unsigned side,char *err,size_t nerr) {
    CUdeviceptr base=weight(r,t);unsigned count=4;
    void *args[]={&input,&output,&r->coeff,&base,&side,&count};
    return launch(r,r->rank,CONV,40,1,256,args,err,nerr);
}
static int add(dflash_resident *r,char *err,size_t nerr) {
    unsigned count=4*2560;void *args[]={&r->x,&r->temp,&count};
    return launch(r,r->rank,ADD,40,1,256,args,err,nerr);
}
static int rope(dflash_resident *r,CUdeviceptr input,unsigned heads,unsigned append,char *err,size_t nerr) {
    unsigned count=4;void *args[]={&input,&r->frequency,&r->control,&heads,&count,&append};
    return launch(r,r->rank,ROPE,(count*heads*64+255)/256,1,256,args,err,nerr);
}
static int graph_end(dflash_resident *r,unsigned rank,CUgraph *graph,CUgraphExec *executable,char *err,size_t nerr) {
    TRY(cuStreamEndCapture(r->stream[rank],graph));TRY(cuGraphInstantiateWithFlags(executable,*graph,0));return 0;
}
#define DO(call) do {if((call))return -1;}while(0)
static int backbone(dflash_resident *r,char *err,size_t nerr) {
    unsigned rank=r->rank,n=4,mask=r->model.mask_token;
    TRY(cuCtxSetCurrent(r->target[rank].context));
    TRY(cuStreamBeginCapture(r->stream[rank],CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    void *embed[]={&r->embedding_table,&r->x,&r->control,&mask};
    DO(launch(r,rank,EMBEDDING,40,1,256,embed,err,nerr));
    for(unsigned l=0;l<5;l++) {
        dflash_layer_weights *w=r->model.layer+l;
        DO(norm(r,r->x,r->norm,w->attn_norm,2560,4,err,nerr));
        DO(matrix(r,w->attn_conv_projection,r->norm,r->coeff,err,nerr));
        DO(conv(r,r->norm,r->input,w->attn_conv_base,0,err,nerr));
        DO(matrix(r,w->q,r->input,r->q,err,nerr));DO(matrix(r,w->k,r->input,r->k,err,nerr));DO(matrix(r,w->v,r->input,r->v,err,nerr));
        DO(norm(r,r->q,r->q,w->q_norm,128,4*32,err,nerr));DO(norm(r,r->k,r->k,w->k_norm,128,4*8,err,nerr));
        DO(rope(r,r->q,32,0,err,nerr));DO(rope(r,r->k,8,0,err,nerr));
        void *attention[]={&r->q,&r->k,&r->v,&r->cache_k[l],&r->cache_v[l],&r->partial,&r->statistics,&r->control,&n};
        DO(launch(r,rank,ATTENTION,8*4,1,128,attention,err,nerr));
        void *merge[]={&r->partial,&r->statistics,&r->attention,&n};
        DO(launch(r,rank,ATTENTION_MERGE,4*32,1,32,merge,err,nerr));
        DO(matrix(r,w->attention_output,r->attention,r->projected,err,nerr));
        DO(conv(r,r->projected,r->temp,w->attn_conv_base,1,err,nerr));DO(add(r,err,nerr));
        DO(norm(r,r->x,r->norm,w->ffn_norm,2560,4,err,nerr));
        DO(matrix(r,w->ffn_conv_projection,r->norm,r->coeff,err,nerr));
        DO(conv(r,r->norm,r->input,w->ffn_conv_base,0,err,nerr));
        DO(matrix(r,w->gate,r->input,r->gate,err,nerr));DO(matrix(r,w->up,r->input,r->up,err,nerr));
        unsigned elements=4*7680;void *activation[]={&r->gate,&r->up,&elements};
        DO(launch(r,rank,SWIGLU,(elements+255)/256,1,256,activation,err,nerr));
        DO(matrix(r,w->down,r->gate,r->projected,err,nerr));
        DO(conv(r,r->projected,r->temp,w->ffn_conv_base,1,err,nerr));DO(add(r,err,nerr));
    }
    DO(norm(r,r->x,r->x,r->model.output_norm,2560,4,err,nerr));
    DO(matrix(r,r->model.selector_hidden,r->x,r->selector,err,nerr));
    return graph_end(r,rank,&r->backbone_graph,&r->backbone,err,nerr);
}
static int append_graph(dflash_resident *r,char *err,size_t nerr) {
    unsigned rank=r->rank;TRY(cuCtxSetCurrent(r->target[rank].context));
    TRY(cuStreamBeginCapture(r->stream[rank],CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    void *features[]={&r->feature_ring,&r->features,&r->control};
    DO(launch(r,rank,FEATURES,200,1,256,features,err,nerr));
    DO(matrix(r,r->model.feature_projection,r->features,r->projected,err,nerr));
    DO(norm(r,r->projected,r->projected,r->model.feature_norm,2560,4,err,nerr));
    for(unsigned l=0;l<5;l++) {
        dflash_layer_weights *w=r->model.layer+l;
        DO(matrix(r,w->k,r->projected,r->k,err,nerr));DO(matrix(r,w->v,r->projected,r->v,err,nerr));
        DO(norm(r,r->k,r->k,w->k_norm,128,4*8,err,nerr));DO(rope(r,r->k,8,1,err,nerr));
        void *cache[]={&r->k,&r->v,&r->cache_k[l],&r->cache_v[l],&r->control};
        DO(launch(r,rank,CACHE,16,1,256,cache,err,nerr));
    }
    return graph_end(r,rank,&r->append_graph,&r->append,err,nerr);
}
static int head_graph(dflash_resident *r,unsigned d,char *err,size_t nerr) {
    TRY(cuCtxSetCurrent(r->target[d].context));
    TRY(cuStreamBeginCapture(r->stream[d],CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    unsigned rows=r->target[d].output_rows,k=2560,count=3,type=DFLASH_TARGET_Q5_TILE,stride=r->target[d].output_stride;
    void *matrix_args[]={&r->target[d].output_weights,&r->head_input[d],&r->logits[d],&rows,&k,&count,&type,&stride};
    if(r->q5t[d]) {
        void *args[]={&r->target[d].output_weights,&r->head_input[d],&r->logits[d],&rows,&k,&stride};
        TRY(cuLaunchKernel(r->q5t[d],(rows+31)/32,1,1,256,1,1,0,r->stream[d],args,NULL));
    } else DO(launch(r,d,MATRIX,(rows+3)/4,1,128,matrix_args,err,nerr));
    unsigned first=r->target[d].output_first,blocks=(rows+255)/256;
    void *top[]={&r->logits[d],&r->top_a[d],&rows,&first};
    DO(launch(r,d,TOPK_LOGITS,blocks,3,256,top,err,nerr));
    CUdeviceptr input=r->top_a[d],output=r->top_b[d];unsigned entries=blocks*16;
    while(entries>16) {
        blocks=(entries+255)/256;void *args[]={&input,&output,&entries};
        DO(launch(r,d,TOPK_PAIRS,blocks,3,256,args,err,nerr));
        entries=blocks*16;CUdeviceptr swap=input;input=output;output=swap;
    }
    r->top_final[d]=input;
    return graph_end(r,d,&r->head_graph[d],&r->head[d],err,nerr);
}
static int select_graph(dflash_resident *r,char *err,size_t nerr) {
    TRY(cuCtxSetCurrent(r->target[r->rank].context));
    TRY(cuMemcpyHtoD(r->head_sources,r->top_final,sizeof r->top_final));
    TRY(cuStreamBeginCapture(r->stream[r->rank],CU_STREAM_CAPTURE_MODE_THREAD_LOCAL));
    void *merge[]={&r->head_sources,&r->candidates};
    DO(launch(r,r->rank,HEAD_MERGE,3,1,256,merge,err,nerr));
    const gguf_tensor *a=r->model.selector_predecessor,*b=r->model.selector_successor;
    CUdeviceptr before=weight(r,a),after=weight(r,b);
    unsigned ta=a->type,tb=b->type,sa=(unsigned)(dflash_tensor_bytes(a)/248320),sb=(unsigned)(dflash_tensor_bytes(b)/248320);
    void *args[]={&before,&after,&ta,&tb,&sa,&sb,&r->selector,&r->candidates,&r->control,&r->tokens};
    DO(launch(r,r->rank,SELECT,1,1,256,args,err,nerr));
    return graph_end(r,r->rank,&r->select_graph,&r->select,err,nerr);
}
static int initialize(dflash_resident *r,const char *model,const char *module,char *err,size_t nerr) {
    if(dflash_gguf_open(&r->model,model,err,nerr))return -1;
    for(unsigned d=0;d<6;d++) {
        if(!r->target[d].context || !r->target[d].output_weights || !r->target[d].output_rows || r->target[d].output_stride!=1760) {
            snprintf(err,nerr,"DFlash2 target head binding %u",d);return -1;
        }
        TRY(cuCtxSetCurrent(r->target[d].context));TRY(cuModuleLoad(&r->module[d],module));
        for(unsigned f=0;f<NFUNCTIONS;f++)TRY(cuModuleGetFunction(&r->fn[d][f],r->module[d],names[f]));
        TRY(cuStreamCreate(&r->stream[d],CU_STREAM_NON_BLOCKING));TRY(cuEventCreate(&r->head_ready[d],CU_EVENT_DISABLE_TIMING));
    }
    unsigned rank=r->rank;
#define ALLOC(field,bytes) DO(allocate(r,rank,&r->field,bytes,err,nerr))
    ALLOC(arena,r->model.file.len-r->model.file.data_start);
    TRY(cuMemcpyHtoD(r->arena,r->model.file.map+r->model.file.data_start,r->model.file.len-r->model.file.data_start));
    ALLOC(control,sizeof(dflash_device_control));ALLOC(embedding_table,6*sizeof(dflash_embedding_shard));
    ALLOC(frequency,64*sizeof(float));float frequency[64];for(unsigned j=0;j<64;j++)frequency[j]=powf(r->model.rope_theta,-(float)j/64.f);
    TRY(cuMemcpyHtoD(r->frequency,frequency,sizeof frequency));
    ALLOC(x,4*2560*4);ALLOC(norm,4*2560*4);ALLOC(input,4*2560*4);ALLOC(coeff,4*640*4);
    ALLOC(projected,4*2560*4);ALLOC(temp,4*2560*4);ALLOC(q,4*4096*4);ALLOC(k,4*1024*4);ALLOC(v,4*1024*4);
    ALLOC(attention,4*4096*4);ALLOC(gate,4*7680*4);ALLOC(up,4*7680*4);ALLOC(features,4*12800*4);
    ALLOC(selector,4*256*4);ALLOC(partial,4*4*32*128*4);ALLOC(statistics,4*4*32*2*4);
    ALLOC(tokens,4*sizeof(uint32_t));ALLOC(candidates,3*16*sizeof(dflash_candidate));ALLOC(head_sources,6*sizeof(CUdeviceptr));
    for(unsigned l=0;l<5;l++) {ALLOC(cache_k[l],2048*1024*2);ALLOC(cache_v[l],2048*1024*2);}
    dflash_embedding_shard embedding[6]={{0}};
    for(unsigned d=0;d<6;d++)if(r->target[d].embedding && r->target[d].embedding_bytes) {
        CUdeviceptr p;DO(allocate(r,rank,&p,r->target[d].embedding_bytes,err,nerr));
        if(d==rank)TRY(cuMemcpyDtoD(p,r->target[d].embedding,r->target[d].embedding_bytes));
        else TRY(cuMemcpyPeer(p,r->target[rank].context,r->target[d].embedding,r->target[d].context,r->target[d].embedding_bytes));
        embedding[d]=(dflash_embedding_shard){p,r->target[d].embedding_first_group,r->target[d].embedding_groups,
                                             r->target[d].embedding_stride,r->target[d].embedding_raw_q3};
    }
    TRY(cuMemcpyHtoD(r->embedding_table,embedding,sizeof embedding));
    TRY(cuEventCreate(&r->hidden_ready,CU_EVENT_DISABLE_TIMING));
    TRY(cuMemAllocHost((void **)&r->host_control,sizeof *r->host_control));TRY(cuMemAllocHost((void **)&r->host_tokens,16));
    memset(r->host_control,0,sizeof *r->host_control);TRY(cuMemcpyHtoD(r->control,r->host_control,sizeof *r->host_control));
    for(unsigned d=0;d<6;d++) {
        unsigned rows=r->target[d].output_rows,blocks=(rows+255)/256;
        DO(allocate(r,d,&r->head_input[d],3*2560*4,err,nerr));DO(allocate(r,d,&r->logits[d],(size_t)3*rows*4,err,nerr));
        DO(allocate(r,d,&r->top_a[d],(size_t)3*blocks*16*sizeof(dflash_candidate),err,nerr));
        DO(allocate(r,d,&r->top_b[d],(size_t)3*blocks*16*sizeof(dflash_candidate),err,nerr));
    }
#undef ALLOC
    const char *q4k=getenv("FLASH_DFLASH_Q4K_CUBIN");
    if(q4k && *q4k) {
        TRY(cuCtxSetCurrent(r->target[rank].context));
        TRY(cuModuleLoad(&r->q4k_module,q4k));TRY(cuModuleGetFunction(&r->q4k,r->q4k_module,"df4_q4k_matrix"));
        for(unsigned d=0;d<6;d++) {
            TRY(cuCtxSetCurrent(r->target[d].context));
            TRY(cuModuleLoad(&r->head_module[d],q4k));TRY(cuModuleGetFunction(&r->q5t[d],r->head_module[d],"df3_q5t_matrix"));
        }
    }
    DO(head_check(r,err,nerr));
    DO(matrix_check(r,err,nerr));
    DO(backbone(r,err,nerr));DO(append_graph(r,err,nerr));
    for(unsigned d=0;d<6;d++)DO(head_graph(r,d,err,nerr));
    return select_graph(r,err,nerr);
}
int dflash_resident_open(dflash_resident **out,const char *model,const char *module,
                        const dflash_target_binding target[6],unsigned rank,
                        CUdeviceptr feature_ring,char *err,size_t nerr) {
    *out=NULL;if(rank>=6 || !feature_ring){snprintf(err,nerr,"DFlash2 context/feature binding");return -1;}
    dflash_resident *r=calloc(1,sizeof *r);if(!r){snprintf(err,nerr,"DFlash2 host allocation");return -1;}
    memcpy(r->target,target,sizeof r->target);r->rank=rank;r->feature_ring=feature_ring;
    if(initialize(r,model,module,err,nerr)){dflash_resident_close(r);return -1;}
    *out=r;return 0;
}
int dflash_resident_append(dflash_resident *r,unsigned start,unsigned retained,char *err,size_t nerr) {
    if(start!=r->context || !retained || retained>4){snprintf(err,nerr,"DFlash2 retained-prefix position");return -1;}
    unsigned d=r->rank;TRY(cuCtxSetCurrent(r->target[d].context));
    /* The same pinned control block is reused; retire its previous copy first.
     * This is a block boundary, never a per-matrix synchronization. */
    TRY(cuStreamSynchronize(r->stream[d]));
    r->host_control->append_start=start;r->host_control->append_count=retained;
    r->context+=retained;r->host_control->context=r->context;
    TRY(cuMemcpyHtoDAsync(r->control,r->host_control,sizeof *r->host_control,r->stream[d]));
    TRY(cuGraphLaunch(r->append,r->stream[d]));return 0;
}
int dflash_resident_begin(dflash_resident *r,unsigned processed,char *err,size_t nerr) {
    r->context=processed>2048?processed-2048:0;
    while(r->context<processed) {
        unsigned count=processed-r->context;if(count>4)count=4;
        DO(dflash_resident_append(r,r->context,count,err,nerr));
    }
    return dflash_resident_finish(r,err,nerr);
}
/* FLASH_DFLASH_TIMING=1 serializes each draft stage to time it on the host.
 * Diagnostic only: the synchronizations remove the normal overlap. */
static double stage_now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int dflash_resident_propose(dflash_resident *r,uint32_t anchor,uint32_t inputs[4],char *err,size_t nerr) {
    if(anchor>=248320){snprintf(err,nerr,"DFlash2 anchor outside target vocabulary");return -1;}
    static int timing=-1;if(timing<0)timing=getenv("FLASH_DFLASH_TIMING") && atoi(getenv("FLASH_DFLASH_TIMING"));
    double t0=stage_now(),t1=t0,t2=t0;
    unsigned owner=r->rank;TRY(cuCtxSetCurrent(r->target[owner].context));
    TRY(cuStreamSynchronize(r->stream[owner]));
    double t_ready=stage_now();
    r->host_control->context=r->context;r->host_control->anchor=anchor;
    TRY(cuMemcpyHtoDAsync(r->control,r->host_control,sizeof *r->host_control,r->stream[owner]));
    TRY(cuGraphLaunch(r->backbone,r->stream[owner]));TRY(cuEventRecord(r->hidden_ready,r->stream[owner]));
    if(timing){TRY(cuStreamSynchronize(r->stream[owner]));t1=stage_now();}
    CUdeviceptr hidden=r->x+4*2560;
    for(unsigned d=0;d<6;d++) {
        TRY(cuCtxSetCurrent(r->target[d].context));TRY(cuStreamWaitEvent(r->stream[d],r->hidden_ready,0));
        if(d==owner)TRY(cuMemcpyDtoDAsync(r->head_input[d],hidden,3*2560*4,r->stream[d]));
        else TRY(cuMemcpyPeerAsync(r->head_input[d],r->target[d].context,hidden,r->target[owner].context,3*2560*4,r->stream[d]));
        TRY(cuGraphLaunch(r->head[d],r->stream[d]));TRY(cuEventRecord(r->head_ready[d],r->stream[d]));
    }
    if(timing){for(unsigned d=0;d<6;d++){TRY(cuCtxSetCurrent(r->target[d].context));TRY(cuStreamSynchronize(r->stream[d]));}t2=stage_now();}
    TRY(cuCtxSetCurrent(r->target[owner].context));
    for(unsigned d=0;d<6;d++)TRY(cuStreamWaitEvent(r->stream[owner],r->head_ready[d],0));
    TRY(cuGraphLaunch(r->select,r->stream[owner]));
    TRY(cuMemcpyDtoHAsync(r->host_tokens,r->tokens,16,r->stream[owner]));TRY(cuStreamSynchronize(r->stream[owner]));
    if(timing)fprintf(stderr,"draft_timing prior_ms=%.3f backbone_ms=%.3f head_ms=%.3f select_ms=%.3f total_ms=%.3f\n",
        (t_ready-t0)*1e3,(t1-t_ready)*1e3,(t2-t1)*1e3,(stage_now()-t2)*1e3,(stage_now()-t0)*1e3);
    memcpy(inputs,r->host_tokens,16);return 0;
}
int dflash_resident_finish(dflash_resident *r,char *err,size_t nerr) {
    TRY(cuCtxSetCurrent(r->target[r->rank].context));TRY(cuStreamSynchronize(r->stream[r->rank]));return 0;
}
void dflash_resident_target_layers(const dflash_resident *r,unsigned layers[5]) {
    memcpy(layers,r->model.target_layer,5*sizeof *layers);
}
void dflash_resident_close(dflash_resident *r) {
    if(!r)return;
    for(unsigned d=0;d<6;d++)if(r->target[d].context) {
        cuCtxSetCurrent(r->target[d].context);if(r->stream[d])cuStreamSynchronize(r->stream[d]);
        if(r->head[d])cuGraphExecDestroy(r->head[d]);if(r->head_graph[d])cuGraphDestroy(r->head_graph[d]);
        if(r->head_ready[d])cuEventDestroy(r->head_ready[d]);
    }
    if(r->target[r->rank].context)cuCtxSetCurrent(r->target[r->rank].context);
    if(r->backbone)cuGraphExecDestroy(r->backbone);if(r->backbone_graph)cuGraphDestroy(r->backbone_graph);
    if(r->append)cuGraphExecDestroy(r->append);if(r->append_graph)cuGraphDestroy(r->append_graph);
    if(r->select)cuGraphExecDestroy(r->select);if(r->select_graph)cuGraphDestroy(r->select_graph);
    if(r->q4k_module)cuModuleUnload(r->q4k_module);
    for(unsigned d=0;d<6;d++)if(r->head_module[d]){cuCtxSetCurrent(r->target[d].context);cuModuleUnload(r->head_module[d]);}
    if(r->hidden_ready)cuEventDestroy(r->hidden_ready);
    if(r->host_control)cuMemFreeHost(r->host_control);if(r->host_tokens)cuMemFreeHost(r->host_tokens);
    for(unsigned i=r->allocations;i;i--) {unsigned d=r->allocation[i-1].rank;cuCtxSetCurrent(r->target[d].context);cuMemFree(r->allocation[i-1].pointer);}
    for(unsigned d=0;d<6;d++)if(r->target[d].context) {
        cuCtxSetCurrent(r->target[d].context);if(r->stream[d])cuStreamDestroy(r->stream[d]);if(r->module[d])cuModuleUnload(r->module[d]);
    }
    dflash_gguf_close(&r->model);free(r);
}
