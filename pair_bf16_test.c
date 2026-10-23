/* GPU B1 versus shared-weight B2/B8 parity, including every output guard. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dieops.h"
#define MAX_POSITIONS 8u
static unsigned positions=2;
static unsigned output_begin(unsigned position){return positions==2?(position?12u:4u):4u+position;}
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult check_result=(x); if(check_result!=CUDA_SUCCESS){const char *s="?";cuGetErrorString(check_result,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(1);}}while(0)
static uint32_t rng=19489;
static float sample(void){rng=rng*1664525u+1013904223u;return ((int)((rng>>8)&65535)-32768)/32768.f;}
static void ptr(uint32_t *d,int addr,CUdeviceptr p){int i=2+(addr-0x140)/4;d[i]=(uint32_t)p;d[i+1]=(uint32_t)(p>>32);}
static void word(uint32_t *d,int addr,uint32_t v){d[2+(addr-0x140)/4]=v;}
static void launch(CUfunction fn,CUdeviceptr dp,CUdeviceptr ctl,uint32_t d[MAX_POSITIONS][32],int pair){
    unsigned nph=positions,ntok=pair?positions:1;uint32_t control[16]={0};control[9]=pair?positions:0;
    CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));CK(cuMemcpyHtoD(dp,d,positions*128u));
    void *args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
}
struct variant{const char *name;unsigned scalar,paired,batch;int k,slices,bias,xout;};
static int check_case(CUfunction fn,CUdeviceptr dp,CUdeviceptr ctl,const struct variant *v,unsigned rows,unsigned units){
    const unsigned guard=64,stride=rows+guard,slots=6*positions;size_t nout=(size_t)stride*slots,nx=(size_t)v->k*v->slices,nw=(size_t)rows*nx;
    uint16_t *w=malloc(2*nw);float *x=malloc(4*positions*nx),*bias=malloc(4*positions*rows);uint32_t *ref=malloc(4*nout),*got=malloc(4*nout);
    if(!w||!x||!bias||!ref||!got)exit(2);
    for(size_t i=0;i<nw;i++){float f=sample();uint32_t z;memcpy(&z,&f,4);w[i]=(uint16_t)(z>>16);}
    for(size_t i=0;i<positions*nx;i++)x[i]=sample();
    for(size_t i=0;i<positions*rows;i++)bias[i]=sample();
    CUdeviceptr dw,dx,dy,db,dt;CK(cuMemAlloc(&dw,2*nw));CK(cuMemAlloc(&dx,4*positions*nx));CK(cuMemAlloc(&dy,4*nout));CK(cuMemAlloc(&db,4*positions*rows));CK(cuMemAlloc(&dt,8*slots));
    CK(cuMemcpyHtoD(dw,w,2*nw));CK(cuMemcpyHtoD(dx,x,4*positions*nx));CK(cuMemcpyHtoD(db,bias,4*positions*rows));
    CUdeviceptr tab[6*MAX_POSITIONS];for(unsigned i=0;i<slots;i++)tab[i]=dy+(CUdeviceptr)4*stride*i;CK(cuMemcpyHtoD(dt,tab,8*slots));
    uint32_t scalar[MAX_POSITIONS][32]={{0}},pair[MAX_POSITIONS][32];
    for(unsigned b=0;b<positions;b++){
        scalar[b][0]=v->scalar;scalar[b][1]=units;ptr(scalar[b],0x140,dw);ptr(scalar[b],0x148,dx+(CUdeviceptr)b*4*nx);
        ptr(scalar[b],0x150,v->xout?dt+b*48:tab[b*6]+64);word(scalar[b],0x158,rows);
        if(v->bias)ptr(scalar[b],0x160,db+(CUdeviceptr)b*4*rows);
        if(v->xout)word(scalar[b],0x168,4*output_begin(b));
    }
    memcpy(pair,scalar,sizeof pair);pair[0][0]=positions==8?v->batch:v->paired;pair[0][30]=24|(positions==8?8u<<8:0);
    int failures=0;
    for(int run=0;run<2;run++){
        const uint32_t sentinel=0x7fc01234u;
        CK(cuMemsetD32(dy,sentinel,nout));launch(fn,dp,ctl,scalar,0);CK(cuMemcpyDtoH(ref,dy,4*nout));
        CK(cuMemsetD32(dy,sentinel,nout));launch(fn,dp,ctl,pair,1);CK(cuMemcpyDtoH(got,dy,4*nout));
        unsigned bad=0,guard_bad=0,missing=0;
        for(size_t i=0;i<nout;i++)if(ref[i]!=got[i]){if(bad<3)printf("  %s run%d word%zu scalar=%08x pair=%08x\n",v->name,run,i,ref[i],got[i]);bad++;}
        for(unsigned b=0;b<positions;b++)for(unsigned d=0;d<6;d++)for(unsigned i=0;i<stride;i++){
            unsigned begin=v->xout?output_begin(b):16;int active=(v->xout||d==0)&&i>=begin&&i<begin+rows;size_t q=(b*6+d)*(size_t)stride+i;
            if(active){if(ref[q]==sentinel||got[q]==sentinel)missing++;}
            else if(ref[q]!=sentinel||got[q]!=sentinel)guard_bad++;
        }
        printf("%s width=%u rows=%u units=%u run=%d: mismatches=%u guard_errors=%u missing=%u %s\n",v->name,positions,rows,units,run,bad,guard_bad,missing,(bad||guard_bad||missing)?"FAIL":"BIT-EXACT");
        failures+=bad||guard_bad||missing;
    }
    CK(cuMemFree(dt));CK(cuMemFree(db));CK(cuMemFree(dy));CK(cuMemFree(dx));CK(cuMemFree(dw));free(got);free(ref);free(bias);free(x);free(w);return failures;
}
int main(void){
    const char *width=getenv("BF16_POSITIONS");
    if(width && !strcmp(width,"8"))positions=8;
    else if(width && strcmp(width,"2")){fputs("BF16_POSITIONS must be 2 or 8\n",stderr);return 2;}
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    char name[128];CK(cuDeviceGetName(name,sizeof name,dev));printf("GPU: %s\n",name);const char *cubin=getenv("FLASH_CUBIN");CK(cuModuleLoad(&mod,cubin?cubin:"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
    CUdeviceptr dp,ctl;CK(cuMemAlloc(&dp,MAX_POSITIONS*128u));CK(cuMemAlloc(&ctl,8192));
    const struct variant v[]={
        {"bf16bias",OP_BF16BIAS,OP_BF16BIASP,OP_BF16BIAS8,2560,1,1,0},
        {"bf16k2560",OP_BF16K2560,OP_BF16K2560P,OP_BF16K25608,2560,1,0,0},
        {"bf16k320",OP_BF16K320,OP_BF16K320P,OP_BF16K3208,320,1,0,0},
        {"bf16hc",OP_BF16HC,OP_BF16HCP,OP_BF16HC8,2560,4,0,0},
        {"bf16hcx",OP_BF16HCX,OP_BF16HCXP,OP_BF16HCX8,2560,4,0,1}};
    const unsigned rows[]={1,137,513},units[]={1,16,33};int fails=0;
    for(unsigned i=0;i<sizeof v/sizeof v[0];i++)for(unsigned j=0;j<3;j++)fails+=check_case(fn,dp,ctl,&v[i],rows[j],units[j]);
    CK(cuMemFree(ctl));CK(cuMemFree(dp));CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));printf("paired BF16: %d failed cases\n",fails);return fails?3:0;
}
