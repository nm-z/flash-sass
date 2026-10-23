/* GPU B1 versus shared-weight B2 parity, including output guards. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dieops.h"
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult check_result=(x); if(check_result!=CUDA_SUCCESS){const char *s="?";cuGetErrorString(check_result,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(1);}}while(0)
static uint32_t rng=19489;
static float sample(void){rng=rng*1664525u+1013904223u;return ((int)((rng>>8)&65535)-32768)/32768.f;}
static void ptr(uint32_t *d,int addr,CUdeviceptr p){int i=2+(addr-0x140)/4;d[i]=(uint32_t)p;d[i+1]=(uint32_t)(p>>32);}
static void word(uint32_t *d,int addr,uint32_t v){d[2+(addr-0x140)/4]=v;}
static void launch(CUfunction fn,CUdeviceptr dp,CUdeviceptr ctl,uint32_t d[2][32],int pair){
    unsigned nph=2,ntok=pair?2:1;uint32_t control[16]={0};control[9]=pair?2:0;
    CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));CK(cuMemcpyHtoD(dp,d,256));
    void *args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
}
struct variant{const char *name;unsigned scalar,paired;int k,slices,bias,xout;};
static int check_case(CUfunction fn,CUdeviceptr dp,CUdeviceptr ctl,const struct variant *v,unsigned rows,unsigned units){
    const unsigned guard=64,stride=rows+guard,slots=12;size_t nout=(size_t)stride*slots,nx=(size_t)v->k*v->slices,nw=(size_t)rows*nx;
    uint16_t *w=malloc(2*nw);float *x=malloc(8*nx),*bias=malloc(8*rows);uint32_t *ref=malloc(4*nout),*got=malloc(4*nout);
    if(!w||!x||!bias||!ref||!got)exit(2);
    for(size_t i=0;i<nw;i++){float f=sample();uint32_t z;memcpy(&z,&f,4);w[i]=(uint16_t)(z>>16);}
    for(size_t i=0;i<2*nx;i++)x[i]=sample();
    for(size_t i=0;i<2*rows;i++)bias[i]=sample();
    CUdeviceptr dw,dx,dy,db,dt;CK(cuMemAlloc(&dw,2*nw));CK(cuMemAlloc(&dx,8*nx));CK(cuMemAlloc(&dy,4*nout));CK(cuMemAlloc(&db,8*rows));CK(cuMemAlloc(&dt,96));
    CK(cuMemcpyHtoD(dw,w,2*nw));CK(cuMemcpyHtoD(dx,x,8*nx));CK(cuMemcpyHtoD(db,bias,8*rows));
    CUdeviceptr tab[12];for(unsigned i=0;i<slots;i++)tab[i]=dy+(CUdeviceptr)4*stride*i;CK(cuMemcpyHtoD(dt,tab,sizeof tab));
    uint32_t scalar[2][32]={{0}},pair[2][32];
    for(unsigned b=0;b<2;b++){
        scalar[b][0]=v->scalar;scalar[b][1]=units;ptr(scalar[b],0x140,dw);ptr(scalar[b],0x148,dx+(CUdeviceptr)b*4*nx);
        ptr(scalar[b],0x150,v->xout?dt+b*48:tab[b*6]+64);word(scalar[b],0x158,rows);
        if(v->bias)ptr(scalar[b],0x160,db+(CUdeviceptr)b*4*rows);
        if(v->xout)word(scalar[b],0x168,b?48:16);
    }
    memcpy(pair,scalar,sizeof pair);pair[0][0]=v->paired;pair[0][30]=24;
    int failures=0;
    for(int run=0;run<2;run++){
        const uint32_t sentinel=0x7fc01234u;
        CK(cuMemsetD32(dy,sentinel,nout));launch(fn,dp,ctl,scalar,0);CK(cuMemcpyDtoH(ref,dy,4*nout));
        CK(cuMemsetD32(dy,sentinel,nout));launch(fn,dp,ctl,pair,1);CK(cuMemcpyDtoH(got,dy,4*nout));
        unsigned bad=0,guard_bad=0,missing=0;
        for(size_t i=0;i<nout;i++)if(ref[i]!=got[i]){if(bad<3)printf("  %s run%d word%zu scalar=%08x pair=%08x\n",v->name,run,i,ref[i],got[i]);bad++;}
        for(unsigned b=0;b<2;b++)for(unsigned d=0;d<6;d++)for(unsigned i=0;i<stride;i++){
            unsigned begin=v->xout?(b?12:4):16;int active=(v->xout||d==0)&&i>=begin&&i<begin+rows;size_t q=(b*6+d)*(size_t)stride+i;
            if(active){if(ref[q]==sentinel||got[q]==sentinel)missing++;}
            else if(ref[q]!=sentinel||got[q]!=sentinel)guard_bad++;
        }
        printf("%s rows=%u units=%u run=%d: mismatches=%u guard_errors=%u missing=%u %s\n",v->name,rows,units,run,bad,guard_bad,missing,(bad||guard_bad||missing)?"FAIL":"BIT-EXACT");
        failures+=bad||guard_bad||missing;
    }
    CK(cuMemFree(dt));CK(cuMemFree(db));CK(cuMemFree(dy));CK(cuMemFree(dx));CK(cuMemFree(dw));free(got);free(ref);free(bias);free(x);free(w);return failures;
}
int main(void){
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    char name[128];CK(cuDeviceGetName(name,sizeof name,dev));printf("GPU: %s\n",name);CK(cuModuleLoad(&mod,"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
    CUdeviceptr dp,ctl;CK(cuMemAlloc(&dp,256));CK(cuMemAlloc(&ctl,8192));
    const struct variant v[]={
        {"bf16bias",OP_BF16BIAS,OP_BF16BIASP,2560,1,1,0},
        {"bf16k2560",OP_BF16K2560,OP_BF16K2560P,2560,1,0,0},
        {"bf16k320",OP_BF16K320,OP_BF16K320P,320,1,0,0},
        {"bf16hc",OP_BF16HC,OP_BF16HCP,2560,4,0,0},
        {"bf16hcx",OP_BF16HCX,OP_BF16HCXP,2560,4,0,1}};
    const unsigned rows[]={1,137,513},units[]={1,16,33};int fails=0;
    for(unsigned i=0;i<sizeof v/sizeof v[0];i++)for(unsigned j=0;j<3;j++)fails+=check_case(fn,dp,ctl,&v[i],rows[j],units[j]);
    CK(cuMemFree(ctl));CK(cuMemFree(dp));CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));printf("paired BF16: %d failed cases\n",fails);return fails?3:0;
}
