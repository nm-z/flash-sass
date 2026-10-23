/* Included by engine.c. DFlash2 backbone with the existing SM52 BF16 SASS
 * GEMM, C elementwise/attention/selector, and the resident target's SASS head.
 * First end-to-end path: four-token proposal blocks, no acceptance/recovery. */
#include "json.h"
#include "chunk_ops.h"
#include <sys/mman.h>
#include <fcntl.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define DF_H 2560
#define DF_F 12800
#define DF_B 4
#define DF_WIN 2048
typedef struct { const char *name; const uint16_t *data; CUdeviceptr gpu,narrow[5]; size_t count; unsigned rows,cols; } df_tensor;
static df_tensor df_t[128];static unsigned df_nt;
static arena *df_arena;static void *df_map;static size_t df_map_bytes;
static CUcontext df_ctx;static CUmodule df_module;static CUfunction df_fn;
static CUdeviceptr df_allocations[160],df_dp,df_ctl,df_x,df_y;
static unsigned df_nalloc;
static CUdeviceptr df_head_dp[MAXD],df_head_ctl[MAXD],df_embed_ids,df_embed_out;
static prog_t df_head_program[MAXD],df_embed_program;
static float *df_hx,*df_hy,*df_k[5],*df_v[5],*df_feature_batch;
static unsigned df_context,df_blocks,df_outputs,df_captured;
static double df_ms;
static void df_error(const char *s) { fprintf(stderr,"engine: DFlash2 %s\n",s);if(in_init)longjmp(init_jb,1);longjmp(run_jb,1); }
static float df_bf(uint16_t v) { uint32_t u=(uint32_t)v<<16;float f;memcpy(&f,&u,4);return f; }
static df_tensor *df_weight(const char *name) {
    for(unsigned i=0;i<df_nt;i++)if(!strcmp(name,df_t[i].name))return &df_t[i];
    df_error(name);return NULL;
}
static df_tensor *df_lw(unsigned layer,const char *suffix) {
    char key[160];snprintf(key,sizeof key,"layers.%u.%s",layer,suffix);return df_weight(key);
}
static CUdeviceptr df_alloc(size_t n,const void *src) {
    CUdeviceptr p;CK(cuMemAlloc(&p,n+256));df_allocations[df_nalloc++]=p;
    CK(cuMemsetD8(p,0,n+256));if(src)CK(cuMemcpyHtoD(p,src,n));return p;
}
static void df_gemm(df_tensor *w,const float *input,float *output,unsigned n) {
    unsigned k=w->cols,rows=w->rows;
    CK(cuCtxSetCurrent(df_ctx));
    if(n<=DF_B) {
        /* The four-token draft need not execute 128 padded GEMM columns.
         * Reuse production BF16 paired GEMV for the common hidden width. */
        unsigned slices=(k+2559)/2560;
        for(unsigned slice=0;slice<slices;slice++) {
            memset(df_hx,0,(size_t)n*2560*4);
            unsigned columns=k-slice*2560;if(columns>2560)columns=2560;
            for(unsigned j=0;j<n;j++)memcpy(df_hx+(size_t)j*2560,input+(size_t)j*k+slice*2560,4u*columns);
            CK(cuMemcpyHtoD(df_x,df_hx,(size_t)n*2560*4));
            prog_t p;prog_init(&p);
            for(unsigned j=0;j<n;j++) {
                ph_bf16(&p,OP_BF16K2560,w->narrow[slice],df_x+(size_t)j*2560*4,df_y+(size_t)j*rows*4,rows,0);
                if(j&1u) { p.p[j-1].w[0]=OP_BF16K2560P;p.p[j-1].w[30]=24; }
            }
            CK(cuMemcpyHtoD(df_dp,p.p,(size_t)p.n*sizeof(phase_t)));CK(cuMemsetD8(df_ctl,0,8192));
            unsigned phases=(unsigned)p.n,end=1;void *args[]={&df_dp,&phases,&df_ctl,&end};
            CK(cuLaunchKernel(df_fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
            CK(cuMemcpyDtoH(df_hy,df_y,(size_t)n*rows*4));prog_free(&p);
            if(!slice)memcpy(output,df_hy,(size_t)n*rows*4);
            else for(size_t j=0;j<(size_t)n*rows;j++)output[j]+=df_hy[j];
        }
        return;
    }
    for(unsigned i=0;i<k;i++)for(unsigned p=0;p<n;p++)df_hx[(size_t)i*n+p]=input[(size_t)p*k+i];
    CUDA_MEMCPY2D copy={0};copy.srcMemoryType=CU_MEMORYTYPE_HOST;copy.srcHost=df_hx;copy.srcPitch=4*n;
    copy.dstMemoryType=CU_MEMORYTYPE_DEVICE;copy.dstDevice=df_x;copy.dstPitch=4*128;
    copy.WidthInBytes=4*n;copy.Height=k;CK(cuMemcpy2D(&copy));
    prog_t p;prog_init(&p);phase_t *a=prog_add(&p,OP_GEMM_BF16,16);
    ph_ptr(a,0x140,w->gpu);ph_ptr(a,0x148,df_x);ph_ptr(a,0x150,df_y);
    ph_u32(a,0x158,rows);ph_u32(a,0x15c,n);ph_u32(a,0x160,k);ph_u32(a,0x164,128);ph_u32(a,0x188,1);
    CK(cuMemcpyHtoD(df_dp,p.p,sizeof(phase_t)));CK(cuMemsetD8(df_ctl,0,8192));
    unsigned phases=1,end=1;void *args[]={&df_dp,&phases,&df_ctl,&end};
    CK(cuLaunchKernel(df_fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());prog_free(&p);
    memset(&copy,0,sizeof copy);copy.srcMemoryType=CU_MEMORYTYPE_DEVICE;copy.srcDevice=df_y;copy.srcPitch=4*128;
    copy.dstMemoryType=CU_MEMORYTYPE_HOST;copy.dstHost=df_hy;copy.dstPitch=4*n;copy.WidthInBytes=4*n;copy.Height=rows;
    CK(cuMemcpy2D(&copy));
    for(unsigned r=0;r<rows;r++)for(unsigned j=0;j<n;j++)output[(size_t)j*rows+r]=df_hy[(size_t)r*n+j];
}
static void df_norm(const float *x,float *y,const df_tensor *w,unsigned n,unsigned width) {
    for(unsigned p=0;p<n;p++) {
        float sum=0;for(unsigned j=0;j<width;j++)sum+=x[(size_t)p*width+j]*x[(size_t)p*width+j];
        float scale=1.0f/sqrtf(sum/width+1e-6f);
        for(unsigned j=0;j<width;j++)y[(size_t)p*width+j]=x[(size_t)p*width+j]*scale*df_bf(w->data[j]);
    }
}
static void df_rope(float *x,unsigned n,unsigned heads,unsigned start) {
    for(unsigned p=0;p<n;p++)for(unsigned h=0;h<heads;h++)for(unsigned j=0;j<64;j++) {
        float angle=(start+p)*powf(10000.f,-(float)j/64.f),c=cosf(angle),s=sinf(angle);
        float *v=x+((size_t)p*heads+h)*128;float a=v[j],b=v[j+64];v[j]=a*c-b*s;v[j+64]=b*c+a*s;
    }
}
static void df_conv(const float *x,float *y,const float *coeff,const df_tensor *base,unsigned side,unsigned n) {
    for(unsigned p=0;p<n;p++)for(unsigned c=0;c<DF_H;c++) {
        unsigned group=c/16;size_t delta=(size_t)p*640+side*320+group;
        float a=df_bf(base->data[side*2*DF_H+c])+coeff[delta];
        float b=df_bf(base->data[(side*2+1)*DF_H+c])+coeff[delta+160];
        y[(size_t)p*DF_H+c]=a*x[(size_t)p*DF_H+c]+(p?b*x[(size_t)(p-1)*DF_H+c]:0);
    }
}
static void df_load(void) {
    const char *path=getenv("FLASH_DFLASH_MODEL");if(!path)df_error("requires FLASH_DFLASH_MODEL");
    int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0||fstat(fd,&st)||st.st_size<16)df_error("cannot open safetensors");
    df_map_bytes=(size_t)st.st_size;df_map=mmap(NULL,df_map_bytes,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    if(df_map==MAP_FAILED) { df_map=NULL;df_error("mmap failed"); }
    uint64_t header;memcpy(&header,df_map,8);if(header>df_map_bytes-8)df_error("invalid header");
    df_arena=arena_new();const char *error=NULL;
    jv *root=json_parse(df_arena,(const char *)df_map+8,(size_t)header,&error);
    if(!root||root->k!=J_DICT)df_error("invalid tensor metadata");
    for(int i=0;i<root->o.n;i++) {
        const char *name=root->o.key[i];if(!strcmp(name,"__metadata__"))continue;
        jv *entry=root->o.val[i],*dtype=jv_getz(entry,"dtype"),*shape=jv_getz(entry,"shape"),*offset=jv_getz(entry,"data_offsets");
        if(!dtype||dtype->k!=J_STR||strcmp(dtype->s.s,"BF16")||!shape||!offset||df_nt>=128)df_error("unsupported tensor");
        size_t elements=1;for(int d=0;d<shape->l.n;d++)elements*=(size_t)shape->l.a[d]->i;
        size_t begin=(size_t)offset->l.a[0]->i,end=(size_t)offset->l.a[1]->i;
        if(end<begin||end-begin!=elements*2||end>df_map_bytes-8-header)df_error("tensor bounds");
        df_tensor *t=&df_t[df_nt++];t->name=name;t->count=elements;t->data=(const uint16_t *)((const char *)df_map+8+header+begin);
        if(shape->l.n==2 && strstr(name,".weight")) {
            t->rows=(unsigned)shape->l.a[0]->i;t->cols=(unsigned)shape->l.a[1]->i;t->gpu=df_alloc(elements*2,t->data);
            if(t->cols==2560)t->narrow[0]=t->gpu;
            else {
                if(t->cols>DF_F || t->rows>DF_F)df_error("unsupported matrix dimensions");
                uint16_t *tile=calloc((size_t)t->rows*2560,2);if(!tile)df_error("weight staging allocation");
                for(unsigned slice=0;slice<(t->cols+2559)/2560;slice++) {
                    unsigned columns=t->cols-slice*2560;if(columns>2560)columns=2560;
                    memset(tile,0,(size_t)t->rows*2560*2);
                    for(unsigned row=0;row<t->rows;row++)memcpy(tile+(size_t)row*2560,t->data+(size_t)row*t->cols+slice*2560,2u*columns);
                    t->narrow[slice]=df_alloc((size_t)t->rows*2560*2,tile);
                }
                free(tile);
            }
        }
    }
}
static void dflash_init(void) {
    CUdevice device;int ordinal=getenv("FLASH_DFLASH_DEVICE")?atoi(getenv("FLASH_DFLASH_DEVICE")):6;
    CK(cuDeviceGet(&device,ordinal));CK(cuCtxCreate_v2(&df_ctx,CU_CTX_SCHED_BLOCKING_SYNC,device));
    const char *module=getenv("FLASH_DFLASH_CUBIN");if(!module)df_error("requires FLASH_DFLASH_CUBIN");
    CK(cuModuleLoad(&df_module,module));CK(cuModuleGetFunction(&df_fn,df_module,"die"));df_load();
    df_dp=df_alloc(8192,NULL);df_ctl=df_alloc(8192,NULL);
    df_x=df_alloc(4ull*DF_F*128,NULL);df_y=df_alloc(4ull*DF_F*128,NULL);
    df_hx=calloc(128*DF_F,4);df_hy=calloc(128*DF_F,4);df_feature_batch=calloc(128*DF_F,4);
    for(unsigned l=0;l<5;l++) { df_k[l]=calloc(DF_WIN*1024,4);df_v[l]=calloc(DF_WIN*1024,4); }
    if(!df_hx||!df_hy||!df_feature_batch)df_error("host allocation");
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));df_head_dp[d]=dalloc(8192);df_head_ctl[d]=dalloc(8192);
        prog_t *p=&df_head_program[d];prog_init(p);
        dn_bufs_t *b=&db[d];ph_actq(p,OP_ACTQ256,b->xh,b->zq,b->zs32,b->zs16,b->zas,256,10);
        ph_gemv_any(p,GG_Q5_K,2560,dw[d][0].output,b->zq,b->zs32,b->zs16,b->hlo,(unsigned)sp[d][0].on,b->zas,0,0);
        phase_t *a=prog_add(p,OP_TOPK,16);ph_ptr(a,0x140,b->hlo);ph_ptr(a,0x148,b->tkout);ph_ptr(a,0x150,b->tkpart);
        ph_ptr(a,0x158,b->tkcnt);ph_u32(a,0x160,(unsigned)sp[d][0].on);ph_u32(a,0x164,16);ph_u32(a,0x168,16);
        CK(cuMemcpyHtoD(df_head_dp[d],p->p,(size_t)p->n*sizeof(phase_t)));
    }
    CK(cuCtxSetCurrent(ctx[embed_die]));df_embed_ids=dalloc(16);df_embed_out=dalloc(4*10240);
    dn_bufs_t embedding=db[embed_die];embedding.ids=df_embed_ids;embedding.S=df_embed_out;
    prog_init(&df_embed_program);prog_embed(&df_embed_program,&dw[embed_die][0],&embedding,0);
    CK(cuCtxSetCurrent(ctx[0]));
    fprintf(stderr,"engine: DFlash2 ready tensors=%u device=%d block=4 completed_layers=5,15,25,35,45 hc_mean=4 target_layers=48\n",df_nt,ordinal);
}
static void dflash_fini(void) {
    for(int d=0;d<MAXD;d++)prog_free(&df_head_program[d]);
    prog_free(&df_embed_program);
    if(df_ctx) {
        cuCtxSetCurrent(df_ctx);for(unsigned i=0;i<df_nalloc;i++)cuMemFree(df_allocations[i]);
        if(df_module)cuModuleUnload(df_module);
        cuCtxDestroy(df_ctx);
    }
    df_ctx=NULL;df_module=NULL;df_nalloc=df_nt=0;
    free(df_hx);free(df_hy);free(df_feature_batch);df_hx=df_hy=df_feature_batch=NULL;
    for(unsigned l=0;l<5;l++) { free(df_k[l]);free(df_v[l]);df_k[l]=df_v[l]=NULL; }
    if(df_map)munmap(df_map,df_map_bytes);
    df_map=NULL;arena_free(df_arena);df_arena=NULL;
    dflash_enabled=0;memset(dflash_capture,0,sizeof dflash_capture);
}
static void df_capture(unsigned slot,float *features) {
    float residual[10240];CK(cuCtxSetCurrent(ctx[0]));
    for(unsigned tap=0;tap<5;tap++) {
        CK(cuMemcpyDtoH(residual,dflash_capture[slot][0][tap],sizeof residual));
        for(unsigned j=0;j<DF_H;j++)features[tap*DF_H+j]=.25f*(residual[j]+residual[DF_H+j]+residual[2*DF_H+j]+residual[3*DF_H+j]);
    }
    df_captured++;
}
static void df_append(const float *features,unsigned n) {
    float *projected=malloc((size_t)n*DF_H*4),*k=malloc((size_t)n*1024*4),*v=malloc((size_t)n*1024*4);
    if(!projected||!k||!v)df_error("context allocation");
    df_gemm(df_weight("fc.weight"),features,projected,n);df_norm(projected,projected,df_weight("hidden_norm.weight"),n,DF_H);
    for(unsigned l=0;l<5;l++) {
        df_gemm(df_lw(l,"self_attn.k_proj.weight"),projected,k,n);
        df_gemm(df_lw(l,"self_attn.v_proj.weight"),projected,v,n);
        df_norm(k,k,df_lw(l,"self_attn.k_norm.weight"),n*8,128);df_rope(k,n,8,df_context);
        for(unsigned p=0;p<n;p++) {
            unsigned at=(df_context+p)%DF_WIN;
            memcpy(df_k[l]+(size_t)at*1024,k+(size_t)p*1024,4096);memcpy(df_v[l]+(size_t)at*1024,v+(size_t)p*1024,4096);
        }
    }
    df_context+=n;free(projected);free(k);free(v);
}
static void df_embedding(uint32_t token,float *out) {
    int d=embed_die;CK(cuCtxSetCurrent(ctx[d]));
    CK(cuMemcpyHtoD(df_embed_ids,&token,4));CK(cuMemsetD8(df_head_ctl[d],0,8192));
    CK(cuMemcpyHtoD(df_head_dp[d],df_embed_program.p,sizeof(phase_t)));
    unsigned phases=1,end=1;void *args[]={&df_head_dp[d],&phases,&df_head_ctl[d],&end};
    CK(cuLaunchKernel(fn[d],16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(out,df_embed_out,4*DF_H));
    CK(cuMemcpyHtoD(df_head_dp[d],df_head_program[d].p,(size_t)df_head_program[d].n*sizeof(phase_t)));
}
static void df_candidates(const float *hidden,uint32_t ids[16],float logits[16]) {
    for(unsigned i=0;i<16;i++) { ids[i]=UINT32_MAX;logits[i]=-INFINITY; }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(db[d].xh,hidden,4*DF_H));CK(cuMemsetD8(df_head_ctl[d],0,8192));
    }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));unsigned phases=(unsigned)df_head_program[d].n,end=1;
        void *args[]={&df_head_dp[d],&phases,&df_head_ctl[d],&end};CK(cuLaunchKernel(fn[d],16,1,1,256,1,1,0,0,args,NULL));
    }
    for(int d=0;d<nd;d++) {
        uint32_t list[64];CK(cuCtxSetCurrent(ctx[d]));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(list,db[d].tkout,sizeof list));
        for(unsigned j=0;j<16;j++) {
            uint32_t id=list[2*j]+(unsigned)sp[d][0].o0;float score;memcpy(&score,&list[2*j+1],4);
            for(unsigned rank=0;rank<16;rank++)if(score>logits[rank] || (score==logits[rank] && id<ids[rank])) {
                for(unsigned tail=15;tail>rank;tail--) { ids[tail]=ids[tail-1];logits[tail]=logits[tail-1]; }
                ids[rank]=id;logits[rank]=score;break;
            }
        }
    }
}
static void df_attention(unsigned l,const float *q,const float *k,const float *v,float *out,unsigned n) {
    float scores[DF_WIN+DF_B];
    for(unsigned p=0;p<n;p++)for(unsigned h=0;h<32;h++) {
        unsigned head=h/4,start=df_context+p+1>DF_WIN?df_context+p+1-DF_WIN:0;
        unsigned past=df_context-start,total=past+n;float max=-INFINITY;
        const float *query=q+((size_t)p*32+h)*128;
        for(unsigned t=0;t<total;t++) {
            const float *key=t<past?df_k[l]+((size_t)((start+t)%DF_WIN)*8+head)*128:k+((size_t)(t-past)*8+head)*128;
            float dot=0;for(unsigned j=0;j<128;j++)dot+=query[j]*key[j];dot*=0.08838834765f;scores[t]=dot;if(dot>max)max=dot;
        }
        float sum=0;for(unsigned t=0;t<total;t++) { scores[t]=expf(scores[t]-max);sum+=scores[t]; }
        float *result=out+((size_t)p*32+h)*128;memset(result,0,512);
        for(unsigned t=0;t<total;t++) {
            const float *value=t<past?df_v[l]+((size_t)((start+t)%DF_WIN)*8+head)*128:v+((size_t)(t-past)*8+head)*128;
            float a=scores[t]/sum;for(unsigned j=0;j<128;j++)result[j]+=a*value[j];
        }
    }
}
static void df_propose(uint32_t anchor,uint32_t tokens[DF_B]) {
    double begin=now();unsigned n=DF_B;
    float x[DF_B*DF_H],norm[DF_B*DF_H],input[DF_B*DF_H],coeff[DF_B*640],project[DF_B*DF_H],tmp[DF_B*DF_H];
    float q[DF_B*4096],k[DF_B*1024],v[DF_B*1024],attention[DF_B*4096],gate[DF_B*7680],up[DF_B*7680];
    df_embedding(anchor,x);df_embedding(248077,x+DF_H);
    for(unsigned p=2;p<n;p++)memcpy(x+p*DF_H,x+DF_H,4*DF_H);
    for(unsigned l=0;l<5;l++) {
        df_norm(x,norm,df_lw(l,"input_layernorm.weight"),n,DF_H);
        df_gemm(df_lw(l,"attention_conv.kernel_projection.weight"),norm,coeff,n);
        df_conv(norm,input,coeff,df_lw(l,"attention_conv.base_kernel"),0,n);
        df_gemm(df_lw(l,"self_attn.q_proj.weight"),input,q,n);
        df_gemm(df_lw(l,"self_attn.k_proj.weight"),input,k,n);
        df_gemm(df_lw(l,"self_attn.v_proj.weight"),input,v,n);
        df_norm(q,q,df_lw(l,"self_attn.q_norm.weight"),n*32,128);df_norm(k,k,df_lw(l,"self_attn.k_norm.weight"),n*8,128);
        df_rope(q,n,32,df_context);df_rope(k,n,8,df_context);df_attention(l,q,k,v,attention,n);
        df_gemm(df_lw(l,"self_attn.o_proj.weight"),attention,project,n);
        df_conv(project,tmp,coeff,df_lw(l,"attention_conv.base_kernel"),1,n);
        for(unsigned i=0;i<n*DF_H;i++)x[i]+=tmp[i];
        df_norm(x,norm,df_lw(l,"post_attention_layernorm.weight"),n,DF_H);
        df_gemm(df_lw(l,"mlp_conv.kernel_projection.weight"),norm,coeff,n);
        df_conv(norm,input,coeff,df_lw(l,"mlp_conv.base_kernel"),0,n);
        df_gemm(df_lw(l,"mlp.gate_proj.weight"),input,gate,n);df_gemm(df_lw(l,"mlp.up_proj.weight"),input,up,n);
        for(unsigned i=0;i<n*7680;i++)gate[i]=gate[i]/(1.0f+expf(-gate[i]))*up[i];
        df_gemm(df_lw(l,"mlp.down_proj.weight"),gate,project,n);
        df_conv(project,tmp,coeff,df_lw(l,"mlp_conv.base_kernel"),1,n);
        for(unsigned i=0;i<n*DF_H;i++)x[i]+=tmp[i];
    }
    df_norm(x,x,df_weight("norm.weight"),n,DF_H);
    float projected[DF_B*256];df_gemm(df_weight("candidate_selector.hidden_projection.weight"),x,projected,n);
    const uint16_t *pred=df_weight("candidate_selector.predecessor_codebook")->data,*succ=df_weight("candidate_selector.successor_codebook")->data;
    tokens[0]=anchor;
    for(unsigned p=1;p<n;p++) {
        uint32_t ids[16];float logits[16];df_candidates(x+p*DF_H,ids,logits);
        float best=-INFINITY;unsigned selected=0;
        for(unsigned c=0;c<16;c++) {
            float score=logits[c];
            for(unsigned r=0;r<256;r++)score+=df_bf(pred[(size_t)tokens[p-1]*256+r])*projected[p*256+r]*df_bf(succ[(size_t)ids[c]*256+r]);
            if(score>best) { best=score;selected=c; }
        }
        tokens[p]=ids[selected];
    }
    df_blocks++;df_ms+=(now()-begin)*1e3;
}
static void dflash_run(runctx *r,const prog_t *full,const prog_t *prefix,const uint32_t stop[2]) {
    df_context=df_blocks=df_outputs=df_captured=0;df_ms=0;unsigned phases[MAXD];
    prog_t *paired=new_request_programs();if(!paired)df_error("request allocation");
    const uint32_t no_stop[2]={UINT32_MAX,UINT32_MAX};
    for(int d=0;d<nd;d++) {
        prog_t a,b;model_program(&a,d,0,UINT32_MAX,no_stop,0,1,1,1,0);model_program(&b,d,1,UINT32_MAX,no_stop,0,1,1,1,0);
        prog_init(&paired[d]);int rc=prog_pair(&paired[d],&a,&b);prog_free(&a);prog_free(&b);if(rc)df_error("target pair build");
        prog_pair_fuse_bf16(&paired[d],1);
    }
    upload_programs(prefix,phases,1);unsigned buffered=0;
    while(!r->aborted && r->count<r->nprompt-1) {
        unsigned p=(unsigned)r->count;ple_pump(r,8);mtp_target_step(r,phases,p+1,prefix);
        df_capture(0,df_feature_batch+(size_t)buffered++*DF_F);
        if(buffered==128) { df_append(df_feature_batch,buffered);buffered=0; }
    }
    if(buffered)df_append(df_feature_batch,buffered);
    while(!r->aborted && !r->ended && r->count<r->T) {
        unsigned start=(unsigned)r->count;uint32_t inputs[DF_B];
        df_propose(start<(unsigned)r->nprompt?h_ids[start]:h_toks[start-1],inputs);
        unsigned width=(unsigned)(r->T-r->count);if(width>DF_B)width=DF_B;
        for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(db[d].ids+4ull*start,inputs,4u*width)); }
        for(unsigned p=0;p<width;p++)ple_seq[start+p]=inputs[p];
        runctx staging=*r;staging.nprompt=(int)(start+width);ple_pump(&staging,(int)width);
        for(unsigned offset=0;offset<width;offset+=2) {
            unsigned count=width-offset>=2?2:1;unsigned end=start+offset+count;
            upload_programs(count==2?paired:full,phases,count);
            runctx pending=*r;pending.sent=(int)end;pending.cb=NULL;
            mtp_target_step(&pending,phases,end,count==2?paired:full);
            for(unsigned slot=0;slot<count;slot++)df_capture(slot,df_feature_batch+(size_t)(offset+slot)*DF_F);
        }
        df_append(df_feature_batch,width);
        CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(h_toks+start,db[0].toks+4ull*start,4u*width));
        df_outputs+=width;r->count=(int)(start+width);trace_completed=(unsigned)r->count;last_timing.completed=r->count;
        last_timing.counts_known=1;last_timing.run_ms=(now()-timing_run_start)*1e3;
        if(!r->tfirst)r->tfirst=now();
        last_timing.first_valid=1;last_timing.first_ms=(r->tfirst-timing_run_start)*1e3;
        for(;r->sent<r->count && !r->aborted && !r->ended;r->sent++) { uint32_t token=h_toks[r->sent];r->ended=token==stop[0]||token==stop[1];deliver_token(r,r->sent,token); }
        if(!r->aborted && r->cb && r->cb->alive && r->cb->alive(r->cb->arg))r->aborted=1;
    }
}
static void dflash_stats(FILE *out) {
    fprintf(out,"mtpstats mode=dflash2_full_model_unchecked block_width=4 draft_blocks=%u target_generated_positions=%u captured_positions=%u target_layers=48 feature_layers=5,15,25,35,45 hc_mean=4 selector_top_k=16 draft_ms=%.6f comparisons=0 acceptance=0 recovery=0 fallback=0\n",df_blocks,df_outputs,df_captured,df_ms);fflush(out);
}
