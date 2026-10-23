/* Compare scalar-position order with paired-phase order on real GPU kernels. */
#include "prog.h"
#include "deltanet.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult check_result=(x); if(check_result!=CUDA_SUCCESS){const char*s="?";cuGetErrorString(check_result,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2);} }while(0)
#define TEST_GUARD_MAX_ALLOCATIONS 512u
#include "test_guards.h"
#define N 10240u
#define H 48u
#define O (H*128u)
static unsigned seed=173;
static int mixed_warp, dn_batch, cpu_all;
static unsigned positions=2;
static float rnd(void){seed=seed*1664525u+1013904223u;return (float)((int)(seed>>8)-8388608)/8388608.f;}
static CUdeviceptr alloc(size_t n){return test_guard_alloc("pair state buffer",n,0);}
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
    unsigned hc[2048]={0};hc[2]=start;hc[9]=positions;CK(cuMemcpyHtoD(ctl,hc,sizeof hc));
    CUdeviceptr dp;size_t bytes=(size_t)p->n*sizeof(phase_t);CK(cuMemAlloc(&dp,bytes));CK(cuMemcpyHtoD(dp,p->p,bytes));unsigned nph=(unsigned)p->n,ntok=start+positions;
    void*args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(hc,ctl,sizeof hc));CK(cuMemFree(dp));
    if(hc[2]!=ntok||hc[1]!=expect_gen){fprintf(stderr,"control at %u: TOK=%u expected%u GEN=%u expected%u\n",start,hc[2],ntok,hc[1],expect_gen);return 1;}return 0;
}
static int c_reference(const buffers *b,const region *r,const float *cinit,const float *dinit,const float *cw,const float *sa,unsigned start) {
    size_t cn=4u*4u*N,dn=4u*H*128u*128u;
    float *cs=malloc(cn),*ds=malloc(dn),*x=malloc(4u*N),*u=malloc(4u*N),*o=malloc(4u*O);
    if(!cs||!ds||!x||!u||!o){fputs("C reference allocation failed\n",stderr);exit(2);}
    memcpy(cs,cinit,cn);memcpy(ds,dinit,dn);int fails=0;
    for(unsigned k=0;k<positions;k++) {
        float a[H],bb[H];CK(cuMemcpyDtoH(x,b[k].x,4u*N));CK(cuMemcpyDtoH(a,b[k].a,sizeof a));CK(cuMemcpyDtoH(bb,b[k].b,sizeof bb));
        conv1d_silu_gpu(x,cs,cw,start+k,u,N);
        for(unsigned h=0;h<H;h++) {
            unsigned kh=h&15u;
            deltanet_step_gpu(ds+(size_t)h*128u*128u,u+128u*kh,u+2048u+128u*kh,u+4096u+128u*h,a[h],bb[h],sa[h],1e-6f,o+128u*h);
        }
        if(memcmp(u,r[2+8*k].ref,4u*N)){fprintf(stderr,"C conv output mismatch start%u pos%u\n",start,k);fails++;}
        if(memcmp(o,r[3+8*k].ref,4u*O)){fprintf(stderr,"C DeltaNet output mismatch start%u pos%u\n",start,k);fails++;}
    }
    if(memcmp(cs,r[0].ref,cn)){fprintf(stderr,"C conv state mismatch start%u\n",start);fails++;}
    if(memcmp(ds,r[1].ref,dn)){fprintf(stderr,"C DeltaNet state mismatch start%u\n",start);fails++;}
    free(cs);free(ds);free(x);free(u);free(o);
    printf("C reference: start=%u positions=%u state/every output=%s\n",start,positions,fails?"FAIL":"BIT-EXACT");
    return fails;
}
int main(void){
    mixed_warp=getenv("PAIR_MIXED_WARP")!=NULL;
    dn_batch=getenv("PAIR_DN_BATCH")!=NULL;
    cpu_all=getenv("PAIR_C_REF")!=NULL;
    const char *width=getenv("PAIR_POSITIONS");
    if(width && !strcmp(width,"8"))positions=8;
    else if(width && !strcmp(width,"16"))positions=16;
    else if(width && !strcmp(width,"32"))positions=32;
    else if(width && strcmp(width,"2")){fputs("PAIR_POSITIONS must be 2, 8, 16 or 32\n",stderr);return 2;}
    if(dn_batch && positions<8){fputs("PAIR_DN_BATCH requires 8, 16 or 32 positions\n",stderr);return 2;}
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
    buffers b[PROG_MAX_POSITIONS];region r[2+8*PROG_MAX_POSITIONS];int nr=0;
#define REGION(nm,ptr,len) do{r[nr++]=(region){nm,ptr,len,malloc(len)};}while(0)
    REGION("conv state",cs,cn);REGION("DeltaNet state",ds,dn);
    for(unsigned k=0;k<positions;k++){
        float x[N],a[H],bb[H];for(unsigned i=0;i<N;i++)x[i]=rnd()*(float)(k+1);
        for(unsigned i=0;i<H;i++){a[i]=rnd();bb[i]=rnd();}
        b[k]=(buffers){.x=upload(x,sizeof x),.u=alloc(4*N),.a=upload(a,sizeof a),.b=upload(bb,sizeof bb),
            .o=alloc(4*O),.q=alloc(2*O),.s32=alloc(4*O/32),.s16=alloc(4*O/16),.as=alloc(4*O/32),
            .am=alloc(8),.part=alloc(16*8),.cnt=alloc(4)};
        REGION("conv output",b[k].u,4*N);REGION("DeltaNet output",b[k].o,4*O);REGION("quant codes",b[k].q,2*O);
        REGION("sum32",b[k].s32,4*O/32);REGION("sum16",b[k].s16,4*O/16);REGION("quant scales",b[k].as,4*O/32);
        REGION("argmax",b[k].am,8);REGION("argmax counter",b[k].cnt,4);
    }
    prog_t programs[PROG_MAX_POSITIONS],pair,serial;const prog_t *group[PROG_MAX_POSITIONS];
    prog_init(&pair);prog_init(&serial);
    for(unsigned k=0;k<positions;k++) {
        prog_init(&programs[k]);program(&programs[k],&b[k],k,cs,ds,dCW,dSA);group[k]=&programs[k];
        for(int i=0;i<programs[k].n;i++)*prog_add(&serial,programs[k].p[i].w[0],programs[k].p[i].w[1])=programs[k].p[i];
    }
    if(prog_group(&pair,group,positions)){fputs("prog_group rejected valid programs\n",stderr);return 2;}
    if(dn_batch && prog_batch_fuse_dnstep(&pair)!=(int)(positions/8)){
        fputs("DeltaNet fusion rejected valid fixture descriptors\n",stderr);return 2;
    }
    const unsigned starts[]={0,1,2,3,4,7,8,9,127,128,2047,2048};int fails=0;unsigned cases=sizeof starts/sizeof starts[0];
    void*got=malloc(dn>cn?dn:cn);
    for(unsigned c=0;c<cases;c++){
        unsigned start=starts[c];int bad=0;
        for(int paired=0;paired<2;paired++){
            CK(cuMemcpyHtoD(cs,cinit,cn));CK(cuMemcpyHtoD(ds,dinit,dn));
            for(int i=2;i<nr;i++)CK(cuMemsetD8(r[i].p,0xa5,r[i].n));
            for(unsigned k=0;k<positions;k++){CK(cuMemsetD8(b[k].part,0,16*8));CK(cuMemsetD32(b[k].cnt,0,1));}
            bad+=launch(fn,ctl,paired?&pair:&serial,start,paired?4u:4u*positions);
            for(int i=0;i<nr;i++){
                if(!paired)CK(cuMemcpyDtoH(r[i].ref,r[i].p,r[i].n));
                else{CK(cuMemcpyDtoH(got,r[i].p,r[i].n));if(memcmp(got,r[i].ref,r[i].n)){fprintf(stderr,"start%u region%d %s mismatch\n",start,i,r[i].name);bad++;}}
            }
            for(unsigned k=0;k<positions;k++){
                float oo[O];unsigned am[2],cnt;CK(cuMemcpyDtoH(oo,b[k].o,sizeof oo));CK(cuMemcpyDtoH(am,b[k].am,8));CK(cuMemcpyDtoH(&cnt,b[k].cnt,4));
                unsigned best=0;for(unsigned i=1;i<O;i++)if(oo[i]>oo[best])best=i;
                if(cnt||am[0]!=best||memcmp(&am[1],&oo[best],4)){fprintf(stderr,"argmax start%u pos%u invalid\n",start,k);bad++;}
            }
        }
        if(cpu_all || (dn_batch && !c))bad+=c_reference(b,r,cinit,dinit,cw,sa,start);
        printf("start=%u positions=%u width=%u dn_batch=%d state/outputs=%s scalar_barriers=%u group_barriers=4\n",start,start+positions-1,positions,dn_batch,bad?"FAIL":"BIT-EXACT",4u*positions);fails+=bad;
    }
    fails+=test_guard_check();
    printf("pair_state_test: %u cases, %d failures\n",cases,fails);
    for(unsigned k=0;k<positions;k++)prog_free(&programs[k]);
    prog_free(&pair);prog_free(&serial);
    for(int i=0;i<nr;i++)free(r[i].ref);
    free(cinit);free(dinit);free(cw);free(got);
    test_guard_free();
    CK(cuCtxDestroy(ctx));return fails?1:0;
}
