#ifndef QSA_DIRECT_PROJECTION_H
#define QSA_DIRECT_PROJECTION_H
#include "prog.h"
#include "block.h"
#include "quant.h"
#include "gguf.h"
typedef struct{int enabled,real;uint32_t type[3];unsigned char*w[4];size_t bytes[4];CUdeviceptr dw[4],z,zq,s32,s16,as;float input[2560];FILE*inputs;}direct_projection;
/* Cached weights use the holder's repacked row layout. The block-32 kernels
 * implement K=640 and cannot be dispatched for these K=2560 projections. */
static size_t direct_projection_stride(uint32_t type) {
 switch(type) {
 case GG_BF16:return 2560u*2u;
 case GG_Q2_0:return (size_t)q2_0_tile_stride(2560);
 case GG_Q3_K:return (size_t)q3_k_tile_stride(2560);
 case GG_Q4_K:return (size_t)q4_k_tile_stride(2560);
 case GG_Q5_K:return (size_t)q5_k_tile_stride(2560);
 case GG_Q6_K:return (size_t)q6_k_tile_stride(2560);
 case GG_IQ4_XS:return (size_t)iq4_xs_tile_stride(2560);
 default:return 0;
 }
}
static unsigned char*direct_load_weight(const char*dir,int block,const char*name,size_t*bytes){
 char path[1024];int pn=snprintf(path,sizeof path,"%s/blk.%d.%s",dir,block,name);if(pn<0||(size_t)pn>=sizeof path)exit(2);
 FILE*f=fopen(path,"rb");if(!f){perror(path);exit(2);}if(fseek(f,0,SEEK_END))exit(2);long n=ftell(f);if(n<=0||fseek(f,0,SEEK_SET))exit(2);
 unsigned char*p=calloc((size_t)n+16,1);if(!p||fread(p,1,(size_t)n,f)!=(size_t)n)exit(2);fclose(f);*bytes=(size_t)n;
 printf("projection_weight=%s bytes=%ld fnv1a64=%016llx\n",path,n,(unsigned long long)direct_file_hash(path));return p;
}
static void direct_load_norm(const char*dir,int block,const char*name,float*out,size_t bytes){size_t n;unsigned char*p=direct_load_weight(dir,block,name,&n);if(n!=bytes){fprintf(stderr,"wrong norm size %s\n",name);exit(2);}memcpy(out,p,n);free(p);}
static void direct_projection_cpu(direct_projection*p,direct_frame*f,unsigned t,unsigned seed){
 if(!p->enabled)return;
 if(p->inputs){if(fread(p->input,sizeof p->input,1,p->inputs)!=1){fprintf(stderr,"short QSA_INPUTS\n");exit(2);}}
 else for(unsigned d=0;d<2560;d++)p->input[d]=.25f*direct_raw(t,d,seed+193u);
 for(unsigned d=0;d<2560;d++)if(!isfinite(p->input[d])){fprintf(stderr,"nonfinite QSA_INPUTS\n");exit(2);}
 int16_t q16[2560];int8_t q8[2560];int32_t s32[80],s16[160];float as[80];quant_act_gpu32(p->input,2560,q16,s32,s16,as);for(int d=0;d<2560;d++)q8[d]=(int8_t)q16[d];
 float*out[3]={f->qg,f->k,f->v};int rows[3]={12288,512,512};for(int i=0;i<3;i++){if(p->type[i]==GG_BF16)gemv_bf16_ref((const uint16_t*)p->w[i],rows[i],2560,p->input,out[i]);else gemv_ref_any(p->type[i],p->w[i],rows[i],2560,q8,s32,s16,as,out[i]);}
 gemv_bf16_ref((const uint16_t*)p->w[3],640,2560,p->input,f->iqk);
}
static void direct_projection_init(direct_projection*p,direct_frame*f,unsigned steps,unsigned seed,int postprojected){
 memset(p,0,sizeof*p);p->enabled=!postprojected;
 if(!p->enabled){if(getenv("QSA_INPUTS")||getenv("QSA_MODEL_CACHE")){fprintf(stderr,"postprojection and original-input modes are exclusive\n");exit(2);}printf("projection_mode=POSTPROJECTION_DIAGNOSTIC checkpoint_provenance=not_supplied\n");return;}
 const char*dir=getenv("QSA_MODEL_CACHE");p->real=dir!=NULL;
 if(dir){int block=3;const char*bs=getenv("QSA_BLOCK");if(bs){char*end;long b=strtol(bs,&end,10);if(!*bs||*end||b<0||b>=48){fprintf(stderr,"QSA_BLOCK must be an integer 0..47\n");exit(2);}block=(int)b;}char extra;const char*types=getenv("QSA_TYPES");if(block<0||block>=48||!types||sscanf(types,"%u,%u,%u%c",p->type,p->type+1,p->type+2,&extra)!=3){fprintf(stderr,"real weights require valid QSA_TYPES and QSA_BLOCK\n");exit(2);}
  const char*names[4]={"attn_q.weight","attn_k.weight","attn_v.weight","indexer_x.weight"};for(int i=0;i<4;i++)p->w[i]=direct_load_weight(dir,block,names[i],p->bytes+i);
  if(p->bytes[3]!=(size_t)640*2560*2){fprintf(stderr,"wrong indexer_x size\n");exit(2);}
  const unsigned rows[3]={12288,512,512};
  for(int i=0;i<3;i++){size_t stride=direct_projection_stride(p->type[i]);if(!stride||p->bytes[i]!=rows[i]*stride){fprintf(stderr,"unsupported type or wrong repacked projection size: %s type=%u bytes=%zu expected=%zu\n",names[i],p->type[i],p->bytes[i],rows[i]*stride);exit(2);}}
  direct_load_norm(dir,block,"attn_q_norm.weight",f->qn,sizeof f->qn);direct_load_norm(dir,block,"attn_k_norm.weight",f->kn,sizeof f->kn);direct_load_norm(dir,block,"indexer.q_norm.weight",f->iqn,sizeof f->iqn);direct_load_norm(dir,block,"indexer.k_norm.weight",f->ikn,sizeof f->ikn);
  for(int i=0;i<3;i++)if(p->type[i]!=GG_BF16&&!gemv_op(p->type[i],2560,0)){fprintf(stderr,"unsupported projection type %u\n",p->type[i]);exit(2);}
 }else{int rows[4]={12288,512,512,640};for(int i=0;i<4;i++){p->bytes[i]=(size_t)rows[i]*2560*2;p->w[i]=malloc(p->bytes[i]);if(!p->w[i])exit(1);uint16_t*w=(uint16_t*)p->w[i];for(size_t j=0;j<p->bytes[i]/2;j++){float v=(float)((int)(direct_hash((uint32_t)j+seed+997u*(unsigned)i)%33)-16)/256.f;uint32_t u;memcpy(&u,&v,4);w[j]=(uint16_t)(u>>16);}if(i<3)p->type[i]=GG_BF16;}}
 const char*path=getenv("QSA_INPUTS");if(path){p->inputs=fopen(path,"rb");if(!p->inputs||fseek(p->inputs,0,SEEK_END)||ftell(p->inputs)!=(long)(steps*sizeof p->input)||fseek(p->inputs,0,SEEK_SET)){fprintf(stderr,"QSA_INPUTS needs %u FP32[2560] inputs\n",steps);exit(2);}printf("original_inputs=%s fnv1a64=%016llx\n",path,(unsigned long long)direct_file_hash(path));}
 printf("projection_mode=%s types=%u,%u,%u indexer=BF16 original_input=%s checkpoint_provenance=not_supplied_cache_fingerprints_only\n",dir?"real_cached_weights":"synthetic_fixed_BF16_weights",p->type[0],p->type[1],p->type[2],path?path:"deterministic_FP32_Z");
}
static unsigned direct_projection_program(direct_projection*p,desc_t*out,CUdeviceptr qg,CUdeviceptr k,CUdeviceptr v,CUdeviceptr iqk){
 if(!p->enabled)return 0;p->z=dalloc(4*2560);p->zq=dalloc(2*2560);p->s32=dalloc(4*80);p->s16=dalloc(4*160);p->as=dalloc(4*80);
 for(int i=0;i<4;i++){p->dw[i]=dalloc(p->bytes[i]);CK(cuMemcpyHtoD(p->dw[i],p->w[i],p->bytes[i]));}
 prog_t P;prog_init(&P);ph_actq(&P,OP_ACTQ256,p->z,p->zq,p->s32,p->s16,p->as,256,10);CUdeviceptr dst[3]={qg,k,v};unsigned rows[3]={12288,512,512};
 for(int i=0;i<3;i++){if(p->type[i]==GG_BF16)ph_bf16(&P,OP_BF16K2560,p->dw[i],p->z,dst[i],rows[i],0);else ph_gemv_any(&P,p->type[i],2560,p->dw[i],p->zq,p->s32,p->s16,dst[i],rows[i],p->as,0,0);}
 ph_bf16(&P,OP_BF16K2560,p->dw[3],p->z,iqk,640,0);if(P.n>32||sizeof(phase_t)!=sizeof(desc_t))exit(3);unsigned n=(unsigned)P.n;memcpy(out,P.p,n*sizeof*out);
 const phase_t*d=P.p+P.n-1;int wi=2+(0x148-0x140)/4,wo=2+(0x150-0x140)/4;uint64_t wp=(uint64_t)d->w[2]|((uint64_t)d->w[3]<<32),src=(uint64_t)d->w[wi]|((uint64_t)d->w[wi+1]<<32),dest=(uint64_t)d->w[wo]|((uint64_t)d->w[wo+1]<<32);
 if(d->w[0]!=OP_BF16K2560||wp!=p->dw[3]||src!=p->z||dest!=iqk||d->w[2+(0x158-0x140)/4]!=640){fprintf(stderr,"index projection descriptor assertion\n");exit(3);}prog_free(&P);return n;
}
static void direct_projection_fini(direct_projection*p){for(int i=0;i<4;i++)free(p->w[i]);if(p->inputs)fclose(p->inputs);}
#endif
