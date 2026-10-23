/* Compare the fused phase with the existing GPU phases and the CPU activation
 * twin. Synthetic resident weights avoid model loading or filesystem effects. */
#define _POSIX_C_SOURCE 200809L
#include "prog.h"
#include "quant.h"
#include "fmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *, unsigned int, CUdevice);
/* Keep the timing ABI supported by archy's CUDA 11 driver when the host has
 * CUDA 13 headers, which redirect this name to a newer exported symbol. */
#undef cuEventElapsedTime
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
#define CK(x) do { CUresult e=(x); if(e) { const char *s="?"; cuGetErrorString(e,&s); fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s); exit(2); } } while(0)
enum { K=2560, ROWS=640, EXPERTS=10, GUARD=64 };
static unsigned state=731;
static unsigned rng(void) { state=state*1664525u+1013904223u; return state; }
static CUdeviceptr alloc_copy(const void *p,size_t n) {
    CUdeviceptr d; CK(cuMemAlloc(&d,n));
    if(p) CK(cuMemcpyHtoD(d,p,n)); else CK(cuMemsetD8(d,0,n)); return d;
}
typedef struct { CUdeviceptr d; size_t n; unsigned char *a,*b; } output;
static output output_new(size_t n) {
    output o={.d=alloc_copy(NULL,n+2*GUARD),.n=n};
    o.a=malloc(n+2*GUARD); o.b=malloc(n+2*GUARD);
    if(!o.a||!o.b) exit(2);
    return o;
}
static void clear(output *o) { CK(cuMemsetD8(o->d,0xa5,o->n+2*GUARD)); CK(cuMemsetD8(o->d+GUARD,0,o->n)); }
static int compare(output *o,const char *name,int ne,size_t per_expert) {
    CK(cuMemcpyDtoH(o->b,o->d,o->n+2*GUARD));
    size_t n=(size_t)ne*per_expert; int bad=0;
    for(size_t i=0;i<n;i++) if(o->a[GUARD+i]!=o->b[GUARD+i]) {
        if(bad<3) printf("%s byte %zu: %02x != %02x\n",name,i,o->a[GUARD+i],o->b[GUARD+i]);
        bad++;
    }
    for(size_t i=0;i<GUARD;i++) if(o->b[i]!=0xa5 || o->b[GUARD+o->n+i]!=0xa5) bad++;
    for(size_t i=n;i<o->n;i++) if(o->b[GUARD+i]) bad++;
    printf("ne=%d %s %s mismatches=%d\n",ne,name,bad?"FAIL":"PASS",bad);
    return bad!=0;
}
static float run(CUfunction fn,prog_t *p,CUdeviceptr dp,CUdeviceptr ctl,int repeats) {
    CK(cuMemcpyHtoD(dp,p->p,(size_t)p->n*sizeof *p->p));
    unsigned ntok=1,nph=(unsigned)p->n;
    void *args[]={&dp,&nph,&ctl,&ntok};
    CUevent a,b; CK(cuEventCreate(&a,0)); CK(cuEventCreate(&b,0));
    float total=0;
    for(int i=0;i<repeats;i++) {
        CK(cuMemsetD8(ctl,0,8192)); CK(cuEventRecord(a,0));
        CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));
        CK(cuEventRecord(b,0)); CK(cuEventSynchronize(b));
        float t; CK(cuEventElapsedTime(&t,a,b)); total+=t;
    }
    CK(cuEventDestroy(a)); CK(cuEventDestroy(b)); return total/repeats;
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if(argc!=3) { fprintf(stderr,"usage: moe_fused_test CUBIN GPU\n"); return 2; }
    puts("fused test: initializing CUDA");
    CK(cuInit(0)); CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
    CK(cuDeviceGet(&dev,atoi(argv[2]))); CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_BLOCKING_SYNC,dev));
    printf("fused test: loading %s\n",argv[1]);
    CK(cuModuleLoad(&mod,argv[1])); CK(cuModuleGetFunction(&fn,mod,"die"));
    int regs; CK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,fn));
    printf("fused test: GPU=%d registers=%d\n",atoi(argv[2]),regs);
    size_t stride=720, bytes=EXPERTS*ROWS*stride;
    unsigned char *w=malloc(bytes); if(!w) return 2;
    CUdeviceptr wg,wu;
    for(int which=0;which<2;which++) {
        for(size_t i=0;i<bytes;i++) w[i]=(unsigned char)(rng()>>24);
        for(size_t r=0;r<EXPERTS*ROWS;r++) for(int b=0;b<40;b++) {
            unsigned short h=(unsigned short)(0x2400+(rng()&0x7ff)); memcpy(w+r*stride+2*b,&h,2);
        }
        CUdeviceptr d=alloc_copy(w,bytes); if(which) wu=d; else wg=d;
    }
    free(w);
    float x[K],as[K/32]; short xq[K]; int s32[K/32],s16[K/16];
    for(int i=0;i<K;i++) x[i]=((int)(rng()>>16)-32768)/16384.f;
    quant_act_gpu32(x,K,xq,s32,s16,as);
    CUdeviceptr dx=alloc_copy(xq,sizeof xq),ds=alloc_copy(s32,sizeof s32),da=alloc_copy(as,sizeof as);
    unsigned indices[EXPERTS]; for(int i=0;i<EXPERTS;i++) indices[i]=(unsigned)(EXPERTS-1-i);
    CUdeviceptr di=alloc_copy(indices,sizeof indices),dn=alloc_copy(NULL,4);
    output o[]={output_new(EXPERTS*ROWS*4),output_new(EXPERTS*ROWS*4),output_new(EXPERTS*ROWS*2),
                output_new(EXPERTS*ROWS/32*4),output_new(EXPERTS*ROWS/16*4),output_new(EXPERTS*ROWS/32*4)};
    const char *names[]={"gate","up","codes","s32","s16","scales"};
    size_t sizes[]={ROWS*4,ROWS*4,ROWS*2,ROWS/32*4,ROWS/16*4,ROWS/32*4};
    CUdeviceptr dp=alloc_copy(NULL,4096),ctl=alloc_copy(NULL,8192);
    int counts[]={0,1,2,3,6,10},fails=0;
    for(size_t c=0;c<sizeof counts/sizeof *counts;c++) {
        int ne=counts[c]; CK(cuMemcpyHtoD(dn,&ne,4));
        prog_t base,fused; prog_init(&base); prog_init(&fused);
        ph_q2(&base,OP_Q2MOEGN,wg,dx,ds,o[0].d+GUARD,ROWS,di,da,0,dn); prog_nobar(&base);
        ph_q2(&base,OP_Q2MOEGN,wu,dx,ds,o[1].d+GUARD,ROWS,di,da,0,dn);
        if(ne) ph_ffnact(&base,o[0].d+GUARD,o[1].d+GUARD,o[2].d+GUARD,o[3].d+GUARD,o[4].d+GUARD,o[5].d+GUARD,ROWS,(unsigned)ne);
        ph_q2(&fused,OP_Q2MOEGN,wg,dx,ds,o[0].d+GUARD,ROWS,di,da,o[1].d+GUARD,dn);
        phase_t *p=&fused.p[0]; p->w[30]|=32;
        ph_ptr(p,0x188,wu); ph_ptr(p,0x190,o[2].d+GUARD); ph_ptr(p,0x198,o[3].d+GUARD);
        ph_ptr(p,0x1a0,o[4].d+GUARD); ph_ptr(p,0x1a8,o[5].d+GUARD);
        for(int i=0;i<6;i++) clear(&o[i]);
        printf("ne=%d baseline launch\n",ne);
        float bt=run(fn,&base,dp,ctl,3);
        for(int i=0;i<6;i++) CK(cuMemcpyDtoH(o[i].a,o[i].d,o[i].n+2*GUARD));
        for(int i=0;i<6;i++) clear(&o[i]);
        printf("ne=%d fused launch\n",ne);
        float ft=run(fn,&fused,dp,ctl,3);
        for(int i=0;i<6;i++) fails+=compare(&o[i],names[i],ne,sizes[i]);
        /* Independent C SiLU/quantization check of the GPU dot-product outputs. */
        for(int e=0;e<ne;e++) {
            float v[ROWS],scale[ROWS/32]; short q[ROWS]; int a32[ROWS/32],a16[ROWS/16];
            const float *g=(const float *)(o[0].b+GUARD)+e*ROWS;
            const float *u=(const float *)(o[1].b+GUARD)+e*ROWS;
            for(int i=0;i<ROWS;i++) v[i]=silu_gpu(g[i])*u[i];
            quant_act_gpu32(v,ROWS,q,a32,a16,scale);
            int bad=memcmp(q,o[2].b+GUARD+e*sizes[2],sizes[2]) || memcmp(a32,o[3].b+GUARD+e*sizes[3],sizes[3]) ||
                    memcmp(a16,o[4].b+GUARD+e*sizes[4],sizes[4]) || memcmp(scale,o[5].b+GUARD+e*sizes[5],sizes[5]);
            fails+=bad; printf("ne=%d expert=%d CPU activation %s\n",ne,e,bad?"FAIL":"PASS");
        }
        printf("{\"ne\":%d,\"baseline_ms\":%.6f,\"fused_ms\":%.6f,\"speedup\":%.4f}\n",ne,bt,ft,bt/ft);
        prog_free(&base); prog_free(&fused);
    }
    printf("fused test failures=%d\n",fails); CK(cuCtxDestroy(ctx)); return fails?1:0;
}
