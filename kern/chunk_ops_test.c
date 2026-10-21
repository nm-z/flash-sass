/* Native chunk operator check, using the same Q2 packing and CPU dequantizer
 * as the existing GEMM fixture. No model reload or external model files. */
#include "chunk/prog_chunk.h"
#include "quant.h"
#include "gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#undef cuEventElapsedTime
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
#define CK(x) do { CUresult e=(x); if(e) { const char *s="?"; cuGetErrorString(e,&s); fprintf(stderr,"line %d %s: %s\n",__LINE__,#x,s); exit(2); } } while(0)
static unsigned rng=947;
static unsigned next(void) { rng=rng*1664525u+1013904223u; return rng; }
static CUfunction fn;
static CUdeviceptr dp,ctl;
static CUdeviceptr alloc(const void *p,size_t n) {
    CUdeviceptr d; CK(cuMemAlloc(&d,n));
    if(p) CK(cuMemcpyHtoD(d,p,n)); else CK(cuMemsetD8(d,0,n)); return d;
}
static float run(prog_t *p) {
    CK(cuMemcpyHtoD(dp,p->p,p->n*sizeof *p->p)); CK(cuMemsetD8(ctl,0,8192));
    unsigned ntok=1,nph=p->n; void *args[]={&dp,&nph,&ctl,&ntok};
    CUevent a,b; CK(cuEventCreate(&a,0)); CK(cuEventCreate(&b,0));
    CK(cuEventRecord(a,0)); CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));
    CK(cuEventRecord(b,0)); CK(cuEventSynchronize(b));
    float ms; CK(cuEventElapsedTime(&ms,a,b)); CK(cuEventDestroy(a)); CK(cuEventDestroy(b));
    unsigned done; CK(cuMemcpyDtoH(&done,ctl+8,4));
    if(done!=1) { fprintf(stderr,"incomplete persistent pass: %u\n",done); exit(2); }
    return ms;
}
static unsigned char *q2raw(int rows,int K) {
    size_t n=(size_t)rows*(K/64)*18; unsigned char *p=malloc(n); if(!p) exit(2);
    for(size_t i=0;i<n;i++) p[i]=(unsigned char)(next()>>24);
    for(size_t i=0;i<n;i+=18) { unsigned short h=0x2400+(next()&127); memcpy(p+i,&h,2); }
    return p;
}
static int gemm(int type,int K,int rows,int npos) {
    enum { LDX=128, EXTRA=7 };
    size_t rawrow=type==GG_BF16?2*K:(K/64)*18;
    unsigned char *raw;
    int stride=type==GG_BF16?2*K:q2_0_tile_stride(K);
    if(type==GG_BF16) {
        raw=malloc(rawrow*(rows+EXTRA));
        for(int i=0;i<(rows+EXTRA)*K;i++) { float f=((int)(next()>>24)-128)/128.f; unsigned u; memcpy(&u,&f,4); unsigned short h=u>>16; memcpy(raw+2*i,&h,2); }
    } else raw=q2raw(rows+EXTRA,K);
    unsigned char *tile=malloc((size_t)stride*(rows+EXTRA));
    if(type==GG_BF16) memcpy(tile,raw,(size_t)stride*(rows+EXTRA)); else repack_q2_0(raw,rows+EXTRA,K,tile);
    float *x=calloc((size_t)K*LDX,4),*out=malloc(4ull*(rows+EXTRA)*LDX),*w=malloc(K*4);
    for(int k=0;k<K;k++) for(int p=0;p<npos;p++) x[k*LDX+p]=((int)(next()>>24)-128)/128.f;
    CUdeviceptr W=alloc(tile,(size_t)stride*(rows+EXTRA)), X=alloc(x,(size_t)K*LDX*4), Y=alloc(NULL,4ull*(rows+EXTRA)*LDX);
    CK(cuMemsetD32(Y,0x7fc00000,(rows+EXTRA)*LDX));
    prog_t P; prog_init(&P); ph_gemm(&P,type,K,W,X,Y,rows,npos,LDX);
    run(&P); float ms=0; for(int i=0;i<3;i++) ms+=run(&P); ms/=3;
    CK(cuMemcpyDtoH(out,Y,4ull*(rows+EXTRA)*LDX));
    int bad=0; double worst=0;
    for(int r=0;r<rows;r++) {
        dequant(type,raw+(size_t)r*rawrow,w,K);
        double scale=0,ref[LDX]={0};
        for(int p=0;p<npos;p++) { for(int k=0;k<K;k++) ref[p]+=(double)w[k]*x[k*LDX+p]; if(fabs(ref[p])>scale) scale=fabs(ref[p]); }
        for(int p=0;p<npos;p++) { double err=fabs(out[r*LDX+p]-ref[p])/(scale+1e-12); if(err>worst) worst=err; if(!isfinite(out[r*LDX+p]) || err>2e-4) bad++; }
    }
    for(int r=rows;r<rows+EXTRA;r++) for(int p=0;p<LDX;p++) if(!isnan(out[r*LDX+p])) bad++;
    printf("{\"check\":\"gemm\",\"type\":%d,\"K\":%d,\"rows\":%d,\"npos\":%d,\"bad\":%d,\"max_scaled_error\":%.9g,\"ms\":%.6g}\n",type,K,rows,npos,bad,worst,ms);
    prog_free(&P); CK(cuMemFree(W)); CK(cuMemFree(X)); CK(cuMemFree(Y)); free(raw);free(tile);free(x);free(out);free(w); return bad!=0;
}
/* A read-only-cache regression inside one persistent launch: warm the input,
 * rewrite it through an ordinary store, then consume it again on the same SM.
 * All input and output values are exactly representable. */
static int mutable_activation(void) {
    uint16_t weights[320];float input[320],got[2];
    for(int i=0;i<320;i++) { weights[i]=0x3f80;input[i]=1; }
    CUdeviceptr W=alloc(weights,sizeof weights),X=alloc(input,sizeof input),Y=alloc(NULL,sizeof got);
    prog_t P;prog_init(&P);
    ph_bf16(&P,OP_BF16K320,W,X,Y,1,0);
    phase_t *p=prog_add(&P,OP_SLAB_ZERO,1);ph_ptr(p,0x140,X);ph_u32(p,0x148,80);
    ph_bf16(&P,OP_BF16K320,W,X,Y+4,1,0);
    run(&P);CK(cuMemcpyDtoH(got,Y,sizeof got));
    int bad=got[0]!=320 || got[1]!=0;
    printf("{\"check\":\"mutable_activation\",\"before\":%.9g,\"after\":%.9g,\"expected_after\":0,\"bad\":%d}\n",got[0],got[1],bad);
    prog_free(&P);CK(cuMemFree(W));CK(cuMemFree(X));CK(cuMemFree(Y));return bad;
}
static int qact(int packed) {
    enum { K=192, N=17, LDX=128, GUARD=16 };
    size_t n=packed?K*N:K*LDX;
    float *x=calloc(n,4),*got=malloc((n+2*GUARD)*4),*want=calloc(n,4);
    float v[K],scale[K/32]; int16_t q[K]; int32_t s32[K/32],s16[K/16];
    for(int p=0;p<(packed?N:LDX);p++) {
        for(int k=0;k<K;k++) {
            v[k]=p==0?0:((int)(next()>>16)-32768)/17317.f;
            if(p==1) v[k]*=1e-32f;
            if(p==2) v[k]*=10000;
            x[packed?(size_t)p*K+k:(size_t)k*LDX+p]=v[k];
        }
        quant_act_gpu32(v,K,q,s32,s16,scale);
        for(int k=0;k<K;k++) want[packed?(size_t)p*K+k:(size_t)k*LDX+p]=(float)q[k]*scale[k/32];
    }
    CUdeviceptr X=alloc(x,n*4),Y=alloc(NULL,(n+2*GUARD)*4);
    int bad=0;
    for(int in_place=0;in_place<2;in_place++) {
        CK(cuMemsetD32(Y,0x7fc00000,n+2*GUARD));
        if(in_place) CK(cuMemcpyHtoD(Y+4*GUARD,x,n*4));
        prog_t P;prog_init(&P);unsigned blocks=n/32;
        phase_t *p=prog_add(&P,packed?OP_QACT_P:OP_QACT_S,(blocks+255)/256);
        ph_ptr(p,0x140,in_place?Y+4*GUARD:X);ph_ptr(p,0x148,Y+4*GUARD);ph_u32(p,0x150,blocks);ph_u32(p,0x154,7);
        float ms=run(&P);prog_free(&P);CK(cuMemcpyDtoH(got,Y,(n+2*GUARD)*4));
        int errors=0;
        for(size_t i=0;i<n;i++) if(memcmp(&got[GUARD+i],&want[i],4)) {
            if(errors<3) printf("qact packed=%d in_place=%d at %zu: %.9g expected %.9g\n",packed,in_place,i,got[GUARD+i],want[i]);
            errors++;
        }
        for(int i=0;i<GUARD;i++) if(!isnan(got[i]) || !isnan(got[GUARD+n+i])) errors++;
        printf("{\"check\":\"activation_quant\",\"packed\":%d,\"in_place\":%d,\"bad\":%d,\"ms\":%.6g}\n",packed,in_place,errors,ms);bad+=errors!=0;
    }
    CK(cuMemFree(X));CK(cuMemFree(Y));free(x);free(want);free(got);return bad;
}
static int down(int npos) {
    enum { E=5,R=2560,K=640,LDX=128,ITEMS=3 };
    unsigned char *raw=q2raw(E*R,K), *tile=malloc((size_t)E*R*192);
    repack_q2_0(raw,E*R,K,tile);
    float *h=malloc(4ull*ITEMS*npos*192),*semantic=malloc(4ull*ITEMS*npos*192),*weights=malloc(4ull*ITEMS*npos),*w=malloc(4*K);
    unsigned *positions=malloc(4ull*ITEMS*npos);
    int32_t *ref=calloc((size_t)R*LDX,4),*out=malloc(4ull*R*LDX),*first=malloc(4ull*R*LDX);
    for(int i=0;i<ITEMS*npos;i++) { weights[i]=(i/npos+1)/8.f; positions[i]=(unsigned)(i%npos); }
    CUdeviceptr W=alloc(tile,(size_t)E*R*192),H=alloc(NULL,4ull*ITEMS*npos*192),Y=alloc(NULL,4ull*R*LDX);
    CUdeviceptr LP=alloc(positions,4ull*ITEMS*npos),LW=alloc(weights,4ull*ITEMS*npos),IT=alloc(NULL,16*ITEMS),NI=alloc(NULL,16);
    unsigned ni[4]={ITEMS,0,0,0}; CK(cuMemcpyHtoD(NI,ni,sizeof ni));
    double total_ms=0;
    unsigned seed=rng;
    for(int trial=0;trial<2;trial++) {
    rng=seed;
    CK(cuMemsetD32(Y,0,R*LDX));
    for(int i=0;i<ITEMS*npos;i++) positions[i]=(unsigned)((i%npos+trial)%npos);
    CK(cuMemcpyHtoD(LP,positions,4ull*ITEMS*npos));
    for(int g=0;g<4;g++) {
        int count=g==3?1:3,items[ITEMS][4];
        for(int i=0;i<ITEMS;i++) { items[i][0]=((20*i+3*g)<<8)|count; items[i][1]=i*npos; items[i][2]=npos; items[i][3]=2*i; }
        for(int i=0;i<ITEMS*npos*192;i++) semantic[i]=((int)(next()>>16)-32768)/17317.f;
        for(int i=0;i<ITEMS;i++) for(int p=0;p<npos;p++)
            memcpy(h+((size_t)i*npos+p)*192,semantic+((size_t)i*npos+positions[i*npos+p])*192,4*192);
        CK(cuMemcpyHtoD(H,h,4ull*ITEMS*npos*192)); CK(cuMemcpyHtoD(IT,items,sizeof items)); CK(cuMemcpyHtoD(NI,ni,sizeof ni));
        prog_t P; prog_init(&P); phase_t *p=prog_add(&P,OP_EBATCH_DN,16);
        ph_ptr(p,0x140,W);ph_ptr(p,0x148,H);ph_ptr(p,0x150,Y);ph_ptr(p,0x158,LP);ph_ptr(p,0x160,LW);
        ph_ptr(p,0x168,IT);ph_ptr(p,0x170,NI);ph_u32(p,0x178,LDX);
        total_ms+=run(&P);prog_free(&P);
        if(!trial) for(int i=0;i<ITEMS;i++) for(int r=0;r<R;r++) {
            dequant(GG_Q2_0,raw+((size_t)(2*i)*R+r)*180,w,K);
            for(int p=0;p<npos;p++) { double sum=0; for(int k=0;k<count*64;k++) sum+=(double)w[192*g+k]*h[((size_t)i*npos+p)*192+k]*weights[i*npos+p]; ref[r*LDX+p]+=(int32_t)llrint(sum*1048576.0); }
        }
    }
    CK(cuMemcpyDtoH(out,Y,4ull*R*LDX));
    if(!trial) memcpy(first,out,4ull*R*LDX);
    }
    int bad=0,permutation=0; long maxerr=0;
    for(int r=0;r<R;r++) for(int p=0;p<LDX;p++) { long err=labs((long)out[r*LDX+p]-ref[r*LDX+p]); if(err>maxerr) maxerr=err; if(err>8 || (p>=npos && out[r*LDX+p])) { if(bad<3) printf("down row=%d pos=%d got=%d expected=%d\n",r,p,out[r*LDX+p],ref[r*LDX+p]);bad++; } }
    for(int i=0;i<R*LDX;i++) if(first[i]!=out[i]) permutation++;
    printf("{\"check\":\"whole_expert_down\",\"npos\":%d,\"groups\":4,\"experts\":3,\"bad\":%d,\"permutation_mismatches\":%d,\"max_fixed_error\":%ld,\"ms\":%.6g}\n",npos,bad,permutation,maxerr,total_ms);
    CK(cuMemFree(W));CK(cuMemFree(H));CK(cuMemFree(Y));CK(cuMemFree(LP));CK(cuMemFree(LW));CK(cuMemFree(IT));CK(cuMemFree(NI));
    free(raw);free(tile);free(h);free(semantic);free(weights);free(positions);free(ref);free(out);free(first);free(w);return bad!=0 || permutation!=0;
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    if((argc!=3 && argc!=4) || !getenv("GPUQ_JOB_ID")) { fprintf(stderr,"usage inside gpuq: chunk_ops_test CUBIN GPU [cache|down]\n");return 2; }
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;
    CK(cuDeviceGet(&dev,atoi(argv[2])));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_BLOCKING_SYNC,dev));
    CK(cuModuleLoad(&mod,argv[1]));CK(cuModuleGetFunction(&fn,mod,"die"));
    dp=alloc(NULL,8192);ctl=alloc(NULL,8192);
    if(argc==4) {
        int bad;
        if(!strcmp(argv[3],"cache")) bad=mutable_activation();
        else if(!strcmp(argv[3],"down")) { bad=down(13);bad+=down(17);bad+=down(21); }
        else return 2;
        CK(cuCtxDestroy(ctx));return bad;
    }
    int bad=mutable_activation();bad+=gemm(GG_BF16,320,193,117);bad+=gemm(GG_Q2_0,2560,129,63);bad+=gemm(GG_Q2_0,640,257,128);
    bad+=down(13);bad+=down(17);bad+=down(21);
    bad+=qact(0);bad+=qact(1);
    CK(cuCtxDestroy(ctx));printf("chunk_ops_test: %s\n",bad?"FAIL":"PASS");return bad?1:0;
}
