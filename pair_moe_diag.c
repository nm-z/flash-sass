/* Temporary single-GPU diagnostic: scalar ordering vs generic paired MoE. */
#include "prog.h"
#include "quant.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult rr=(x);if(rr){const char *ss="?";cuGetErrorString(rr,&ss);fprintf(stderr,"line%d %s: %s\n",__LINE__,#x,ss);exit(2);}}while(0)
enum {K=2560,H=640,E=16,V=10};
static uint32_t state=123;
static uint32_t random32(void){state=state*1664525u+1013904223u;return state;}
static CUdeviceptr alloc(size_t n,const void *src){CUdeviceptr p;CK(cuMemAlloc(&p,n));if(src)CK(cuMemcpyHtoD(p,src,n));else CK(cuMemsetD8(p,0,n));return p;}
static CUdeviceptr weights(unsigned rows,unsigned k){
 size_t blocks=(size_t)rows*E*(k/64),bytes=blocks*18,sz=(size_t)rows*E*q2_0_tile_stride(k);
 unsigned char *raw=malloc(bytes),*tile=malloc(sz);if(!raw||!tile)exit(2);
 for(size_t i=0;i<bytes;i++)raw[i]=(unsigned char)(random32()>>24);
 for(size_t i=0;i<blocks;i++){uint16_t h=0x3400;memcpy(raw+18*i,&h,2);}
 repack_q2_0(raw,rows*E,k,tile);CUdeviceptr p=alloc(sz,tile);free(raw);free(tile);return p;
}
typedef struct {CUdeviceptr xq,xs32,xs16,xas,idx,we,ne,g,u,hq,hs32,hs16,has,part;} pos;
static pos create(unsigned p){
 float x[K],as[K/32],we[V];int16_t q[K];int32_t s32[K/32],s16[K/16];uint32_t ids[V];
 for(unsigned i=0;i<K;i++)x[i]=((int)(random32()>>16)-32768)/131072.f;
 for(unsigned i=0;i<V;i++){ids[i]=(3*i+p)%E;we[i]=(float)(p?V-i:i+1)/55.f;}
 quant_act_gpu32(x,K,q,s32,s16,as);
 pos a={.xq=alloc(sizeof q,q),.xs32=alloc(sizeof s32,s32),.xs16=alloc(sizeof s16,s16),.xas=alloc(sizeof as,as),
 .idx=alloc(sizeof ids,ids),.we=alloc(sizeof we,we),.ne=alloc(4,NULL),.g=alloc(4*H*V,NULL),.u=alloc(4*H*V,NULL),
 .hq=alloc(2*H*V,NULL),.hs32=alloc(4*H/32*V,NULL),.hs16=alloc(4*H/16*V,NULL),.has=alloc(4*H/32*V,NULL),.part=alloc(4*K,NULL)};return a;
}
static void init(pos *a,unsigned p,unsigned ne,int poison){
 uint32_t word=poison?(p?0xc0100000:0x3fa00000):0;CK(cuMemcpyHtoD(a->ne,&ne,4));
 CK(cuMemsetD32(a->g,word,H*V));CK(cuMemsetD32(a->u,word,H*V));CK(cuMemsetD16(a->hq,0x5a5a,H*V));
 CK(cuMemsetD32(a->hs32,word,H/32*V));CK(cuMemsetD32(a->hs16,word,H/16*V));CK(cuMemsetD32(a->has,word,H/32*V));
 CK(cuMemsetD32(a->part,ne?word:0,K));
}
static void program(prog_t *p,pos *a,unsigned off,CUdeviceptr wg,CUdeviceptr wu,CUdeviceptr wd){
 p->loop=1;p->token_offset=off;
 ph_q2(p,OP_Q2MOEGN,wg,a->xq,a->xs32,a->g,H,a->idx,a->xas,0,a->ne);prog_nobar(p);
 ph_q2(p,OP_Q2MOEGN,wu,a->xq,a->xs32,a->u,H,a->idx,a->xas,0,a->ne);
 ph_ffnact(p,a->g,a->u,a->hq,a->hs32,a->hs16,a->has,H,V);
 ph_q2(p,OP_Q2MOEDN,wd,a->hq,a->hs32,a->part,K,a->idx,a->has,a->we,a->ne);
}
static int run(CUfunction fn,CUdeviceptr ctl,const prog_t *p,CUdeviceptr dp,unsigned repeat,pos a[2],unsigned n0,unsigned n1,int fill,int pair,float y[2][K]){
 uint32_t hc[2048]={0};hc[9]=2;unsigned nph=p->n,ntok=repeat*2;
 init(&a[0],0,n0,fill);init(&a[1],1,n1,fill);CK(cuMemcpyHtoD(ctl,hc,sizeof hc));
 fprintf(stderr,"RUN pair=%d fill=%d NE=%u,%u repeat=%u\n",pair,fill,n0,n1,repeat);
 void *args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
 CK(cuMemcpyDtoH(hc,ctl,sizeof hc));if(hc[2]!=ntok){fprintf(stderr,"count=%u expected=%u\n",hc[2],ntok);return 1;}
 for(unsigned j=0;j<2;j++)CK(cuMemcpyDtoH(y[j],a[j].part,4*K));
 return 0;
}
int main(int argc,char **argv){
 if(argc!=2&&argc!=3&&argc!=5){fprintf(stderr,"usage: %s cubin [repeat [ne0 ne1]]\n",argv[0]);return 2;}
 unsigned repeat=argc>=3?strtoul(argv[2],0,10):1,lo0=argc==5?strtoul(argv[3],0,10):0,hi0=argc==5?lo0:V,lo1=argc==5?strtoul(argv[4],0,10):0,hi1=argc==5?lo1:V;
 if(!repeat||repeat>1000||hi0>V||hi1>V)return 2;
 CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
 CK(cuModuleLoad(&mod,argv[1]));CK(cuModuleGetFunction(&fn,mod,"die"));
 CUdeviceptr wg=weights(H,K),wu=weights(H,K),wd=weights(K,H),ctl=alloc(8192,NULL);pos a[2]={create(0),create(1)};
 prog_t p0,p1,serial,pair;prog_init(&p0);prog_init(&p1);prog_init(&serial);prog_init(&pair);
 program(&p0,&a[0],0,wg,wu,wd);program(&p1,&a[1],1,wg,wu,wd);if(prog_pair(&pair,&p0,&p1))return 2;
 for(int j=0;j<2;j++){prog_t *p=j?&p1:&p0;for(int i=0;i<p->n;i++)*prog_add(&serial,p->p[i].w[0],p->p[i].w[1])=p->p[i];}
 CUdeviceptr ds=alloc((size_t)serial.n*sizeof(phase_t),serial.p),dp=alloc((size_t)pair.n*sizeof(phase_t),pair.p);
 float reference[2][K],got[2][K];unsigned cases=0;
 for(int fill=0;fill<2;fill++)for(unsigned n0=lo0;n0<=hi0;n0++)for(unsigned n1=lo1;n1<=hi1;n1++){
  if(run(fn,ctl,&serial,ds,repeat,a,n0,n1,fill,0,reference)||run(fn,ctl,&pair,dp,repeat,a,n0,n1,fill,1,got))return 1;
  for(unsigned j=0;j<2;j++)for(unsigned i=0;i<K;i++)if(!isfinite(reference[j][i])||!isfinite(got[j][i])||memcmp(&reference[j][i],&got[j][i],4)){
   fprintf(stderr,"MISMATCH fill=%d ne=%u,%u pos=%u row=%u scalar=%a pair=%a\n",fill,n0,n1,j,i,reference[j][i],got[j][i]);return 1;
  }cases++;
 }
 printf("PASS %u cases %u launches repeats=%u bit-exact finite outputs\n",cases,cases*2,repeat);
 prog_free(&p0);prog_free(&p1);prog_free(&serial);prog_free(&pair);CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));return 0;
}
