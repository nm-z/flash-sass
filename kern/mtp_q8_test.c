/* Validate the drafter's FP32-input Q8 SASS against raw GGUF block decoding.
 * Covers every supported K, selected experts, shared/strided activations,
 * partial CTAs and output guards. No target/reference tokens enter this test. */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include "gguf.h"
#ifdef MTP_PHASE_TEST
#include "mtp/prog.h"
#include "fmath.h"
#endif
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult r=(x); if(r!=CUDA_SUCCESS) { const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2); } } while(0)
static uint32_t rng=731;
#ifdef MTP_PHASE_TEST
static unsigned phase_limit=3;
#endif
static uint32_t next(void) { rng=rng*1664525u+1013904223u;return rng; }
static float random_float(void) { return ((int)(next()>>8)-8388608)*0x1p-23f; }
static CUdeviceptr upload(const void *src,size_t n) {
    CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemcpyHtoD(p,src,n));return p;
}
#ifdef MTP_PHASE_TEST
static void launch_program(CUfunction fn,const prog_t *p) {
    uint32_t control[2048]={0};control[9]=1;
    CUdeviceptr ctl=upload(control,sizeof control),dp=upload(p->p,(size_t)p->n*sizeof(phase_t));
    unsigned phases=(unsigned)p->n,tokens=1;void *args[]={&dp,&phases,&ctl,&tokens};
    CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(control,ctl,sizeof control));
    if(control[2]!=1 || control[1]!=phases) { fprintf(stderr,"MTP control token=%u phases=%u expected=%u\n",control[2],control[1],phases);exit(1); }
    CK(cuMemFree(ctl));CK(cuMemFree(dp));
}
#endif
static float scalar_tree(const float *w,const float *x,int k) {
    float lane[32];
    for(int l=0;l<32;l++) {
        float a[4]={0};
        for(int base=0;base<k;base+=512) for(int j=0;j<4;j++) {
            int at=base+128*j+4*l;
            if(at>=k) continue;
            for(int t=0;t<4;t++) a[t]=fmaf(w[at+t],x[at+t],a[t]);
        }
        lane[l]=(a[0]+a[1])+(a[2]+a[3]);
    }
    for(int s=16;s;s/=2) {
        float old[32];memcpy(old,lane,sizeof old);
        for(int l=0;l<32;l++) lane[l]=old[l]+old[l^s];
    }
    return lane[0];
}
#ifdef MTP_PHASE_TEST
static int check_helpers(const char *cubin) {
    enum { SLOT=3, FF=640, D=2560, HC=4, GUARD=16 };
    float *g=malloc(SLOT*FF*4),*u=malloc(SLOT*FF*4),*parts=malloc(SLOT*D*4);
    float *shared=malloc(D*4),*hidden=malloc(HC*D*4),*embed=malloc(D*4);
    float weights[SLOT]={0.125f,0.375f,0.5f},shared_weight=0.4375f;
    uint32_t *activation=malloc((SLOT*FF+GUARD)*4),*sum=malloc((D+GUARD)*4),*concat=malloc((HC*2*D+GUARD)*4);
    if(!g||!u||!parts||!shared||!hidden||!embed||!activation||!sum||!concat) exit(2);
    for(int i=0;i<SLOT*FF;i++) { g[i]=random_float()*96;u[i]=random_float(); }
    for(int i=0;i<SLOT*D;i++) parts[i]=random_float();
    for(int i=0;i<HC*D;i++) hidden[i]=random_float();
    for(int i=0;i<D;i++) { shared[i]=random_float();embed[i]=random_float(); }
    unsigned count=0;
    CUdeviceptr dg=upload(g,SLOT*FF*4),du=upload(u,SLOT*FF*4),dp=upload(parts,SLOT*D*4);
    CUdeviceptr ds=upload(shared,D*4),dh=upload(hidden,HC*D*4),de=upload(embed,D*4);
    CUdeviceptr dw=upload(weights,sizeof weights),dsw=upload(&shared_weight,4),dn=upload(&count,4);
    CUdeviceptr da,dout,dc;CK(cuMemAlloc(&da,(SLOT*FF+GUARD)*4));CK(cuMemAlloc(&dout,(D+GUARD)*4));CK(cuMemAlloc(&dc,(HC*2*D+GUARD)*4));
    CUmodule module;CUfunction fn;CK(cuModuleLoad(&module,cubin));CK(cuModuleGetFunction(&fn,module,"die"));
    int bad=0;
    for(unsigned ne=0;ne<=SLOT;ne++) for(int has_shared=0;has_shared<2;has_shared++) {
        CK(cuMemcpyHtoD(dn,&ne,4));CK(cuMemsetD32(da,0x7fa12345u,SLOT*FF+GUARD));
        CK(cuMemsetD32(dout,0x7fa12345u,D+GUARD));CK(cuMemsetD32(dc,0x7fa12345u,HC*2*D+GUARD));
        prog_t program;prog_init(&program);mtp_ph_eh_concat(&program,de,dh,dc);
        for(unsigned p=0;p<SLOT;p++) mtp_ph_swiglu(&program,dg,du,da,p,dn);
        mtp_ph_weighted_sum(&program,dp,dw,dn,has_shared?ds:0,has_shared?dsw:0,dout);
        launch_program(fn,&program);prog_free(&program);
        CK(cuMemcpyDtoH(activation,da,(SLOT*FF+GUARD)*4));CK(cuMemcpyDtoH(sum,dout,(D+GUARD)*4));CK(cuMemcpyDtoH(concat,dc,(HC*2*D+GUARD)*4));
        unsigned ac_bad=0,sum_bad=0,concat_bad=0,guards=0;
        for(int p=0;p<SLOT;p++) for(int i=0;i<FF;i++) {
            uint32_t bits=0x7fa12345u;
            if((unsigned)p<ne) { float x=silu_gpu(g[p*FF+i])*u[p*FF+i];memcpy(&bits,&x,4); }
            if(bits!=activation[p*FF+i]) {
                if(ne==1 && !has_shared && ac_bad<3) {
                    float actual;memcpy(&actual,&activation[p*FF+i],4);
                    fprintf(stderr,"swiglu pos=%d i=%d g=%a u=%a got=%a bits=%08x expected=%08x\n",p,i,(double)g[p*FF+i],(double)u[p*FF+i],(double)actual,activation[p*FF+i],bits);
                }
                ac_bad++;
            }
        }
        for(int i=0;i<D;i++) {
            float expected=0;
            for(unsigned p=0;p<ne;p++) expected=fmaf(weights[p],parts[p*D+i],expected);
            if(has_shared) expected=fmaf(shared_weight,shared[i],expected);
            uint32_t bits;memcpy(&bits,&expected,4);sum_bad+=bits!=sum[i];
        }
        for(int p=0;p<HC;p++) for(int i=0;i<D;i++) {
            uint32_t e,h;memcpy(&e,&embed[i],4);memcpy(&h,&hidden[p*D+i],4);
            concat_bad+=concat[p*2*D+i]!=e;concat_bad+=concat[p*2*D+D+i]!=h;
        }
        for(int i=0;i<GUARD;i++) guards+=(activation[SLOT*FF+i]!=0x7fa12345u)+(sum[D+i]!=0x7fa12345u)+(concat[HC*2*D+i]!=0x7fa12345u);
        printf("mtp_helpers selected=%u shared=%d activation_mismatches=%u sum_mismatches=%u concat_mismatches=%u guard_errors=%u\n",ne,has_shared,ac_bad,sum_bad,concat_bad,guards);
        bad+=(ac_bad||sum_bad||concat_bad||guards);
    }
    CUdeviceptr allocations[]={dg,du,dp,ds,dh,de,dw,dsw,dn,da,dout,dc};
    for(unsigned i=0;i<sizeof allocations/sizeof allocations[0];i++) CK(cuMemFree(allocations[i]));
    CK(cuModuleUnload(module));free(g);free(u);free(parts);free(shared);free(hidden);free(embed);free(activation);free(sum);free(concat);
    return bad;
}
#endif
static int one_case(int k,int rows,int strided,int indexed,unsigned count,int cpu_only,const char *dir) {
    enum { EXPERTS=4, SELECTED=3, GUARD=32 };
    unsigned selected[SELECTED]={3,0,2};
    if(!indexed) memset(selected,0,sizeof selected);
    size_t raw_stride=(size_t)(k/32)*34, packed_stride=(size_t)b32_tile_stride(GG_Q8_0,k);
    size_t raw_bytes=EXPERTS*(size_t)rows*raw_stride, packed_bytes=EXPERTS*(size_t)rows*packed_stride;
    uint8_t *raw=malloc(raw_bytes),*packed=malloc(packed_bytes);
    float *x=malloc(SELECTED*(size_t)k*sizeof(float)),*w=malloc((size_t)k*sizeof(float));
    uint32_t *output=malloc((SELECTED*(size_t)rows+GUARD)*sizeof(uint32_t));
    if(!raw||!packed||!x||!w||!output) exit(2);
    for(size_t b=0;b<raw_bytes/34;b++) {
        /* Finite normal scales, both signs and exact zero; arbitrary byte codes. */
        uint16_t h=(uint16_t)(0x2000u+(next()&0x1fffu));
        if(b%7==0) h|=0x8000u;
        if(b%31==0) h=0;
        memcpy(raw+b*34,&h,2);
        for(int j=0;j<32;j++) raw[b*34+2+j]=(uint8_t)(next()>>24);
    }
    if(repack_b32(GG_Q8_0,raw,EXPERTS*rows,k,packed)<0) exit(2);
    /* Check the SASS layout before submitting any GPU read. The legacy B32
     * packer reserved 48 scale bytes even when K needed a larger prefix. */
    size_t prefix=((size_t)k/16+15)&~(size_t)15;
    if(packed_stride!=prefix+(size_t)k) { fprintf(stderr,"Q8 packed stride violates kernel ABI\n");exit(1); }
    for(int row=0;row<EXPERTS*rows;row++) for(int b=0;b<k/32;b++) {
        const uint8_t *source=raw+(size_t)row*raw_stride+34u*(unsigned)b;
        const uint8_t *tile=packed+(size_t)row*packed_stride;
        if(memcmp(tile+2u*(unsigned)b,source,2) || memcmp(tile+prefix+32u*(unsigned)b,source+2,32)) {
            fprintf(stderr,"Q8 packed scales/codes differ at K=%d row=%d block=%d\n",k,row,b);exit(1);
        }
    }
    for(int p=0;p<SELECTED;p++) for(int j=0;j<k;j++) x[p*k+j]=random_float();
    for(size_t j=0;j<SELECTED*(size_t)rows+GUARD;j++) output[j]=0x7fa12345u;
    CUmodule module=0;CUdeviceptr dw=0,dx=0,dy=0,di=0;
    if(!cpu_only) {
        printf("mtp_q8 begin K=%d rows=%d strided=%d indexed=%d count=%u\n",k,rows,strided,indexed,count);
#ifdef MTP_PHASE_TEST
        CK(cuModuleLoad(&module,dir));CUfunction fn,marker;
        CK(cuModuleGetFunction(&marker,module,"flash_mtp_abi_v1"));CK(cuModuleGetFunction(&fn,module,"die"));
#else
        char path[1024];snprintf(path,sizeof path,"%s/mtp_q8_%d.cubin",dir,k);
        CK(cuModuleLoad(&module,path));CUfunction fn;CK(cuModuleGetFunction(&fn,module,"mtp_q8"));
#endif
        dw=upload(packed,packed_bytes);dx=upload(x,SELECTED*(size_t)k*sizeof(float));
        dy=upload(output,(SELECTED*(size_t)rows+GUARD)*sizeof(uint32_t));
        if(indexed) di=upload(selected,sizeof selected);
        unsigned nr=(unsigned)rows,xstride=strided?(unsigned)k*4:0;
#ifdef MTP_PHASE_TEST
        CUdeviceptr dn=indexed?upload(&count,sizeof count):0;
        prog_t p;prog_init(&p);
        for(unsigned slot=0;slot<phase_limit;slot++)
            if(mtp_ph_q8(&p,(unsigned)k,dw,dx,dy,nr,xstride,di,slot,dn)) exit(2);
        launch_program(fn,&p);prog_free(&p);if(dn) CK(cuMemFree(dn));
#else
        void *args[]={&dw,&dx,&dy,&nr,&xstride,&di};
        CK(cuLaunchKernel(fn,(nr+7)/8,SELECTED,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
#endif
        CK(cuMemcpyDtoH(output,dy,(SELECTED*(size_t)rows+GUARD)*sizeof(uint32_t)));
    }
    unsigned bad=0,math_bad=0;double max_error=0;
    for(int p=0;p<SELECTED;p++) for(int r=0;r<rows;r++) {
        const uint8_t *source=raw+((size_t)selected[p]*rows+r)*raw_stride;
        if(dequant(GG_Q8_0,source,w,(uint64_t)k)) exit(2);
        const float *act=x+(strided?p*k:0);
        float expected=scalar_tree(w,act,k);uint32_t bits;memcpy(&bits,&expected,4);
        if((unsigned)p>=count) bits=0x7fa12345u;
#ifdef MTP_PHASE_TEST
        if((unsigned)p>=phase_limit)bits=0x7fa12345u;
#endif
        if(!cpu_only && output[p*rows+r]!=bits) {
            if(bad<3) fprintf(stderr,"K=%d row=%d selected=%u got=%08x expected=%08x\n",k,r,selected[p],output[p*rows+r],bits);
            bad++;
        }
        double exact=0,absolute=0;
        for(int j=0;j<k;j++) { double term=(double)w[j]*act[j];exact+=term;absolute+=fabs(term); }
        double error=fabs((double)expected-exact)/(absolute+1e-30);
        if(error>max_error) max_error=error;
        if(!isfinite(expected) || error>2e-6) math_bad++;
    }
    unsigned guards=0;
    for(int j=0;j<GUARD;j++) guards+=output[SELECTED*rows+j]!=0x7fa12345u;
    printf("mtp_q8 K=%d rows=%d strided=%d active=%u selected=%u,%u,%u gpu_mismatches=%s%u math_failures=%u max_scaled_error=%.9g guard_errors=%u\n",
           k,rows,strided,count,selected[0],selected[1],selected[2],cpu_only?"not_run:":"",bad,math_bad,max_error,guards);
    if(!cpu_only) { CK(cuMemFree(dw));CK(cuMemFree(dx));CK(cuMemFree(dy));if(di) CK(cuMemFree(di));CK(cuModuleUnload(module)); }
    free(raw);free(packed);free(x);free(w);free(output);
    return (bad||math_bad||guards)?1:0;
}
int main(int argc,char **argv) {
    setvbuf(stdout,NULL,_IONBF,0);
    int cpu_only=argc==2&&!strcmp(argv[1],"--cpu-only");
#ifdef MTP_PHASE_TEST
    if(getenv("MTP_SINGLE_PHASE") && atoi(getenv("MTP_SINGLE_PHASE")))phase_limit=1;
    printf("MTP phase diagnostic slots=%u\n",phase_limit);
#endif
    if(argc!=2) { fprintf(stderr,"usage: mtp_q8_test CUBIN_DIRECTORY|--cpu-only\n");return 2; }
    CUcontext context=0;
    if(!cpu_only) { CUdevice device;CK(cuInit(0));CK(cuDeviceGet(&device,0));CK(cuCtxCreate_v2(&context,CU_CTX_SCHED_SPIN,device)); }
    const int widths[]={320,640,2560,5120,6144,10240};int bad=0;unsigned cases=0;
    for(unsigned i=0;i<sizeof widths/sizeof widths[0];i++) for(int stride=0;stride<2;stride++) for(int indexed=0;indexed<2;indexed++) {
        bad+=one_case(widths[i],stride?19:1,stride,indexed,3,cpu_only,argv[1]);cases++;
#ifdef MTP_PHASE_TEST
        if(indexed) for(unsigned count=0;count<2;count++) {
            bad+=one_case(widths[i],stride?19:1,stride,indexed,count,cpu_only,argv[1]);cases++;
        }
#endif
    }
#ifdef MTP_PHASE_TEST
    if(!cpu_only) bad+=check_helpers(argv[1]);
#endif
    if(!cpu_only) CK(cuCtxDestroy(context));
    printf("mtp_q8_test: %s cases=%u failed=%d\n",cpu_only?"CPU reference only":"GPU bit-exact",cases,bad);
    return bad?1:0;
}
