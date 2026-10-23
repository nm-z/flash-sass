/* Compare scalar-position order with paired-phase order on real GPU kernels. */
#include "prog.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult check_result=(x); if(check_result!=CUDA_SUCCESS){const char*s="?";cuGetErrorString(check_result,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2);} }while(0)
#define N 10240u
#define H 48u
#define O (H*128u)
static unsigned seed=173;
static int mixed_warp;
static float rnd(void){seed=seed*1664525u+1013904223u;return (float)((int)(seed>>8)-8388608)/8388608.f;}
static CUdeviceptr alloc(size_t n){CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemsetD8(p,0,n));return p;}
static CUdeviceptr upload(const void*x,size_t n){CUdeviceptr p=alloc(n);CK(cuMemcpyHtoD(p,x,n));return p;}
typedef struct {CUdeviceptr x,u,a,b,o,q,s32,s16,as,am,part,cnt;} buffers;
typedef struct {const char*name;CUdeviceptr p;size_t n;void*ref;} region;
static void program(prog_t*p,const buffers*b,unsigned off,CUdeviceptr cs,CUdeviceptr ds,CUdeviceptr cw,CUdeviceptr sa){
    p->loop=1;p->token_offset=off;
    ph_conv1d(p,b->x,cs,cw,b->u,0,N);
    ph_dnstep(p,ds,b->u,b->u+2048u*4,b->u+4096u*4,b->a,b->b,sa,b->o,1e-6f,0,0,H);
    ph_actq(p,mixed_warp?OP_ACTQ2560:OP_ACTQ256,b->o,b->q,b->s32,b->s16,b->as,mixed_warp?2560:256,mixed_warp?1:O/256);
    phase_t*d=prog_add(p,OP_ARGMAX,16);
    ph_ptr(d,0x140,b->o);ph_ptr(d,0x148,b->am);ph_ptr(d,0x150,b->part);ph_ptr(d,0x158,b->cnt);
    ph_u32(d,0x160,O);ph_u32(d,0x164,16);
}
static int launch(CUfunction fn,CUdeviceptr ctl,const prog_t*p,unsigned start,unsigned expect_gen){
    unsigned hc[2048]={0};hc[2]=start;hc[9]=2;CK(cuMemcpyHtoD(ctl,hc,sizeof hc));
    CUdeviceptr dp=upload(p->p,(size_t)p->n*sizeof(phase_t));unsigned nph=(unsigned)p->n,ntok=start+2;
    void*args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(hc,ctl,sizeof hc));CK(cuMemFree(dp));
    if(hc[2]!=ntok||hc[1]!=expect_gen){fprintf(stderr,"control at %u: TOK=%u expected%u GEN=%u expected%u\n",start,hc[2],ntok,hc[1],expect_gen);return 1;}return 0;
}
int main(void){
    mixed_warp=getenv("PAIR_MIXED_WARP")!=NULL;
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;
    CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char*cubin=getenv("FLASH_CUBIN");CK(cuModuleLoad(&mod,cubin?cubin:"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
    int resident;CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,256,0));if(resident<1){fputs("no resident CTA\n",stderr);return 2;}
    size_t cn=4u*4u*N,dn=4u*H*128u*128u;
    float*cinit=malloc(cn),*dinit=malloc(dn),*cw=malloc(4u*4u*N);float sa[H];
    for(unsigned i=0;i<4*N;i++){cinit[i]=0.1f*rnd();cw[i]=0.2f*rnd();}
    for(unsigned i=0;i<H*128u*128u;i++)dinit[i]=0.01f*rnd();
    for(unsigned i=0;i<H;i++)sa[i]=-0.5f-0.1f*(float)i/H;
    CUdeviceptr cs=upload(cinit,cn),ds=upload(dinit,dn),dCW=upload(cw,4u*4u*N),dSA=upload(sa,sizeof sa),ctl=alloc(8192);
    buffers b[2];region r[24];int nr=0;
#define REGION(nm,ptr,len) do{r[nr++]=(region){nm,ptr,len,malloc(len)};}while(0)
    REGION("conv state",cs,cn);REGION("DeltaNet state",ds,dn);
    for(unsigned k=0;k<2;k++){
        float x[N],a[H],bb[H];for(unsigned i=0;i<N;i++)x[i]=rnd()*(float)(k+1);
        for(unsigned i=0;i<H;i++){a[i]=rnd();bb[i]=rnd();}
        b[k]=(buffers){.x=upload(x,sizeof x),.u=alloc(4*N),.a=upload(a,sizeof a),.b=upload(bb,sizeof bb),
            .o=alloc(4*O),.q=alloc(2*O),.s32=alloc(4*O/32),.s16=alloc(4*O/16),.as=alloc(4*O/32),
            .am=alloc(8),.part=alloc(16*8),.cnt=alloc(4)};
        REGION("conv output",b[k].u,4*N);REGION("DeltaNet output",b[k].o,4*O);REGION("quant codes",b[k].q,2*O);
        REGION("sum32",b[k].s32,4*O/32);REGION("sum16",b[k].s16,4*O/16);REGION("quant scales",b[k].as,4*O/32);
        REGION("argmax",b[k].am,8);REGION("argmax counter",b[k].cnt,4);
    }
    prog_t p0,p1,pair,serial;prog_init(&p0);prog_init(&p1);prog_init(&pair);prog_init(&serial);
    program(&p0,&b[0],0,cs,ds,dCW,dSA);program(&p1,&b[1],1,cs,ds,dCW,dSA);
    if(prog_pair(&pair,&p0,&p1)){fputs("prog_pair rejected valid programs\n",stderr);return 2;}
    for(int k=0;k<2;k++){const prog_t*p=k?&p1:&p0;for(int i=0;i<p->n;i++)*prog_add(&serial,p->p[i].w[0],p->p[i].w[1])=p->p[i];}
    const unsigned starts[]={0,1,2,3,4,7,8,9,127,128,2047,2048};int fails=0;unsigned cases=sizeof starts/sizeof starts[0];
    void*got=malloc(dn>cn?dn:cn);
    for(unsigned c=0;c<cases;c++){
        unsigned start=starts[c];int bad=0;
        for(int paired=0;paired<2;paired++){
            CK(cuMemcpyHtoD(cs,cinit,cn));CK(cuMemcpyHtoD(ds,dinit,dn));
            for(int i=2;i<nr;i++)CK(cuMemsetD8(r[i].p,0xa5,r[i].n));
            for(int k=0;k<2;k++){CK(cuMemsetD8(b[k].part,0,16*8));CK(cuMemsetD32(b[k].cnt,0,1));}
            bad+=launch(fn,ctl,paired?&pair:&serial,start,paired?4u:8u);
            for(int i=0;i<nr;i++){
                if(!paired)CK(cuMemcpyDtoH(r[i].ref,r[i].p,r[i].n));
                else{CK(cuMemcpyDtoH(got,r[i].p,r[i].n));if(memcmp(got,r[i].ref,r[i].n)){fprintf(stderr,"start%u region%d %s mismatch\n",start,i,r[i].name);bad++;}}
            }
            for(int k=0;k<2;k++){
                float oo[O];unsigned am[2],cnt;CK(cuMemcpyDtoH(oo,b[k].o,sizeof oo));CK(cuMemcpyDtoH(am,b[k].am,8));CK(cuMemcpyDtoH(&cnt,b[k].cnt,4));
                unsigned best=0;for(unsigned i=1;i<O;i++)if(oo[i]>oo[best])best=i;
                if(cnt||am[0]!=best||memcmp(&am[1],&oo[best],4)){fprintf(stderr,"argmax start%u pos%d invalid\n",start,k);bad++;}
            }
        }
        printf("start=%u positions=%u,%u state/outputs=%s scalar_barriers=8 pair_barriers=4\n",start,start,start+1,bad?"FAIL":"BIT-EXACT");fails+=bad;
    }
    printf("pair_state_test: %u cases, %d failures\n",cases,fails);
    prog_free(&p0);prog_free(&p1);prog_free(&pair);prog_free(&serial);
    for(int i=0;i<nr;i++)free(r[i].ref);
    free(cinit);free(dinit);free(cw);free(got);
    CK(cuCtxDestroy(ctx));return fails?1:0;
}
