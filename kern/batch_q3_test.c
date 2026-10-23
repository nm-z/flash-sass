/* Q3 B8: bit parity with scalar GPU and the independent C twin, full output
 * canaries, production quant_act_gpu32 inputs and virtual-unit/tail geometries.
 * --cpu-only exercises descriptor eligibility without initializing CUDA.
 * GPU execution is explicitly launched by the caller through gpuq. */
#include "prog.h"
#include "quant.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult r_=(x);if(r_){const char *s_="?";cuGetErrorString(r_,&s_);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s_);exit(2);} } while(0)
static uint32_t seed=793;
static unsigned repeats=1;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static void plan_checks(void) {
    for(unsigned k=2560;k<=6144;k+=3584)for(unsigned width=8;width<=32;width*=2)for(unsigned flags=0;flags<4;flags++) {
        prog_t p[32],g;const prog_t *inputs[32];prog_init(&g);
        unsigned scalar=k==2560?OP_Q3KK2560:OP_Q3KK6144,batch=k==2560?OP_Q3KK25608:OP_Q3KK61448;
        for(unsigned j=0;j<width;j++) {
            prog_init(p+j);p[j].loop=1;p[j].token_offset=j;inputs[j]=p+j;
            CUdeviceptr base=0x200000000ull+0x100000ull*j;
            ph_gemvk(p+j,scalar,0x100000000ull,base,0,base+0x30000,137,base+0x40000);
            p[j].p[0].w[30]=flags;
        }
        assert(prog_group(&g,inputs,width)==0);
        assert(prog_batch_fuse_q3(&g)==(int)(width/8));
        for(unsigned j=0;j<width;j+=8) {
            assert(g.p[j].w[0]==batch && phase_positions(g.p+j)==8);
            assert(!!(g.p[j].w[30]&2u)==((flags&2u)&&j==0));
            assert(!!(g.p[j].w[30]&1u)==((flags&1u)&&j+8==width));
            assert(!!(g.p[j].w[30]&4u)==(j+8<width));
            for(unsigned q=1;q<8;q++)assert(!memcmp(g.p+j+q,p[j+q].p,sizeof(phase_t)));
        }
        assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        ph_ptr(g.p+1,0x158,0x200030000ull); /* overlapping position outputs */
        assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        ph_ptr(g.p+1,0x158,0x200000000ull); /* output aliases another input */
        assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        ph_ptr(g.p+1,0x158,0x200040000ull); /* output aliases another scale */
        assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        ph_ptr(g.p+1,0x140,0x100100000ull);assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        g.p[2].w[31]=1;assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        g.p[2].w[1]++;assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        g.p[2].w[10]++;assert(prog_batch_fuse_q3(&g)==0);
        assert(prog_group(&g,inputs,width)==0);
        ph_ptr(g.p+2,0x170,0);assert(prog_batch_fuse_q3(&g)==0);
        prog_free(&g);for(unsigned j=0;j<width;j++)prog_free(p+j);
    }
    puts("Q3 B8/B16/B32 eligibility, alias rejection and publication flags PASS");
}
static CUdeviceptr upload(const void *p,size_t bytes) {
    CUdeviceptr d;CK(cuMemAlloc(&d,bytes));CK(cuMemcpyHtoD(d,p,bytes));return d;
}
static float launch(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,const phase_t *p,unsigned count,int batch) {
    uint32_t control[16]={0};control[9]=batch?count:0;unsigned end=batch?count:1;
    CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));
    CK(cuMemcpyHtoD(program,p,count*sizeof *p));
    CUevent begin,finish;float ms=0;CK(cuEventCreate(&begin,0));CK(cuEventCreate(&finish,0));
    void *args[]={&program,&count,&ctl,&end};CK(cuEventRecord(begin,0));
    CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(finish,0));
    CK(cuEventSynchronize(finish));CK(cuEventElapsedTime(&ms,begin,finish));
    CK(cuEventDestroy(finish));CK(cuEventDestroy(begin));return ms;
}
static unsigned mismatches(const char *kind,const void *a,const void *b,size_t bytes) {
    const uint32_t *x=a,*y=b;unsigned bad=0;
    for(size_t i=0;i<bytes/4;i++)if(x[i]!=y[i]) {
        if(bad<3)printf("Q3 %s word=%zu expected=%08x actual=%08x\n",kind,i,x[i],y[i]);
        bad++;
    }
    return bad;
}
static int check(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,unsigned K,unsigned rows,unsigned units,unsigned width,unsigned pattern) {
    unsigned L=q4_k_lanes_per_row((int)K),guard=32,matrix_rows=rows+units*(256/L);
    unsigned xstep=K+2*guard,astep=K/32+2*guard,ystep=rows+units*(256/L)+2*guard;
    size_t stride=q3_k_tile_stride((int)K),wbytes=(size_t)matrix_rows*stride;
    size_t xcount=(size_t)width*xstep,acount=(size_t)width*astep,ycount=(size_t)width*ystep;
    uint8_t *raw=malloc((size_t)matrix_rows*(K/256)*110),*w=malloc(wbytes);
    int16_t *x=malloc(2*xcount);int8_t *x8=malloc((size_t)width*K);
    float *as=malloc(4*acount),*input=malloc(4*K),*initial=malloc(4*ycount);
    float *oracle=malloc(4*ycount),*reference=malloc(4*ycount),*actual=malloc(4*ycount);
    int32_t *sum32=malloc(4*(K/32)),*sum16=malloc(4*(K/16));
    assert(raw&&w&&x&&x8&&as&&input&&initial&&oracle&&reference&&actual&&sum32&&sum16);
    for(size_t i=0;i<(size_t)matrix_rows*(K/256)*110;i++)raw[i]=(uint8_t)(random_word()>>24);
    for(unsigned r=0;r<matrix_rows;r++)for(unsigned b=0;b<K/256;b++) {
        static const uint16_t range[]={0x0001,0x03ff,0x0400,0x1400,0x2800,0x3c00,0x8001,0x9400,0xac00};
        uint16_t h=range[random_word()%9];memcpy(raw+((size_t)r*(K/256)+b)*110+108,&h,2);
    }
    repack_q3_k(raw,(int)matrix_rows,(int)K,w);free(raw);
    for(size_t i=0;i<xcount;i++)x[i]=(int16_t)0x5a5a;
    for(size_t i=0;i<acount;i++){uint32_t canary=0x7fc01357u;memcpy(as+i,&canary,4);}
    for(size_t i=0;i<ycount;i++){uint32_t canary=0x7fc01234u;memcpy(initial+i,&canary,4);}
    memcpy(oracle,initial,4*ycount);
    for(unsigned p=0;p<width;p++) {
        static const float magnitudes[]={0x1p-40f,0x1p-20f,0x1p-8f,1.f,0x1p8f,0x1p20f};
        for(unsigned j=0;j<K;j++) {
            unsigned block=j/32,lane=j%32,kind=(block+p+pattern)%6;
            float m=magnitudes[(block/6+p+pattern)%6];
            input[j]=kind==0?0.f:kind==1?m:kind==2?-m:kind==3?(lane&1?-m:m):
                kind==4?(lane==0?m:-m/128.f):m*((int)(random_word()&65535)-32768)/32768.f;
        }
        quant_act_gpu32(input,(int)K,x+(size_t)p*xstep+guard,sum32,sum16,as+(size_t)p*astep+guard);
        for(unsigned j=0;j<K;j++) {
            int16_t q=x[(size_t)p*xstep+guard+j];assert(q>=-127 && q<=127);
            x8[(size_t)p*K+j]=(int8_t)q;
        }
        gemv_q3_k_ref(w,(int)rows,(int)K,x8+(size_t)p*K,as+(size_t)p*astep+guard,oracle+(size_t)p*ystep+guard);
    }
    CUdeviceptr dw=upload(w,wbytes),dx=upload(x,2*xcount),da=upload(as,4*acount),dy=upload(initial,4*ycount);
    prog_t scalar,batch;prog_init(&scalar);prog_init(&batch);
    for(unsigned p=0;p<width;p++) {
        ph_gemvk(&scalar,K==2560?OP_Q3KK2560:OP_Q3KK6144,dw,dx+2*((size_t)p*xstep+guard),0,
            dy+4*((size_t)p*ystep+guard),rows,da+4*((size_t)p*astep+guard));
        scalar.p[p].w[1]=units;
    }
    batch.n=batch.cap=scalar.n;free(batch.p);batch.p=malloc(width*sizeof(phase_t));memcpy(batch.p,scalar.p,width*sizeof(phase_t));
    batch.p[0].w[30]=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
    assert(prog_batch_fuse_q3(&batch)==(int)(width/8));
    (void)launch(fn,program,ctl,scalar.p,width,0);CK(cuMemcpyHtoD(dy,initial,4*ycount));
    (void)launch(fn,program,ctl,batch.p,width,1);CK(cuMemcpyHtoD(dy,initial,4*ycount));
    double scalar_ms=launch(fn,program,ctl,scalar.p,width,0);CK(cuMemcpyDtoH(reference,dy,4*ycount));CK(cuMemcpyHtoD(dy,initial,4*ycount));
    double batch_ms=launch(fn,program,ctl,batch.p,width,1);CK(cuMemcpyDtoH(actual,dy,4*ycount));
    for(unsigned repeat=1;repeat<repeats;repeat++) {
        CK(cuMemcpyHtoD(dy,initial,4*ycount));scalar_ms+=launch(fn,program,ctl,scalar.p,width,0);
        CK(cuMemcpyHtoD(dy,initial,4*ycount));batch_ms+=launch(fn,program,ctl,batch.p,width,1);
    }
    scalar_ms/=repeats;batch_ms/=repeats;
    unsigned gpu_bad=mismatches("scalar GPU",reference,actual,4*ycount),cpu_bad=mismatches("C reference",oracle,actual,4*ycount);
    void *scratch=malloc(wbytes>2*xcount?wbytes:2*xcount);unsigned immutable_bad=0;
    CK(cuMemcpyDtoH(scratch,dw,wbytes));immutable_bad+=mismatches("weight mutation",w,scratch,wbytes);
    CK(cuMemcpyDtoH(scratch,dx,2*xcount));immutable_bad+=mismatches("input mutation",x,scratch,2*xcount);
    free(scratch);scratch=malloc(4*acount);CK(cuMemcpyDtoH(scratch,da,4*acount));
    immutable_bad+=mismatches("scale mutation",as,scratch,4*acount);free(scratch);
    printf("Q3 K=%u rows=%u units=%u positions=%u pattern=%u gpu_mismatches=%u oracle_mismatches=%u immutable_failures=%u scalar_ms=%.6f batch_ms=%.6f speedup=%.6f repeats=%u %s\n",K,rows,units,width,pattern,gpu_bad,cpu_bad,immutable_bad,scalar_ms,batch_ms,scalar_ms/batch_ms,repeats,gpu_bad||cpu_bad||immutable_bad?"FAIL":"BIT-EXACT");
    CK(cuMemFree(dw));CK(cuMemFree(dx));CK(cuMemFree(da));CK(cuMemFree(dy));prog_free(&scalar);prog_free(&batch);
    free(w);free(x);free(x8);free(as);free(input);free(initial);free(oracle);free(reference);free(actual);free(sum32);free(sum16);
    return !!(gpu_bad||cpu_bad||immutable_bad);
}
int main(int argc,char **argv) {
    plan_checks();if(argc==2&&!strcmp(argv[1],"--cpu-only"))return 0;
    if(argc>2 || (argc==2&&strcmp(argv[1],"--production")))return 2;
    const char *repeat_option=getenv("Q3_TIMING_REPEATS");
    if(repeat_option){char *end;unsigned long n=strtoul(repeat_option,&end,10);if(!*repeat_option||*end||n<1||n>20)return 2;repeats=(unsigned)n;}
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule module;CUfunction fn;
    CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char *image=getenv("FLASH_CUBIN");CK(cuModuleLoad(&module,image?image:"die.cubin"));CK(cuModuleGetFunction(&fn,module,"die"));
    CUdeviceptr program,ctl;CK(cuMemAlloc(&program,32*sizeof(phase_t)));CK(cuMemAlloc(&ctl,8192));int failed=0;
    const unsigned rows[]={1,137,513},units[]={1,16,33};
    for(unsigned K=2560;K<=6144;K+=3584)for(unsigned i=0;i<3;i++)failed+=check(fn,program,ctl,K,rows[i],units[i],8,i);
    if(argc==2)for(unsigned K=2560;K<=6144;K+=3584) {
        for(unsigned width=8;width<=32;width*=2)failed+=check(fn,program,ctl,K,4096,16,width,width/8);
        failed+=check(fn,program,ctl,K,4096,33,8,4);
    }
    CK(cuMemFree(ctl));CK(cuMemFree(program));CK(cuModuleUnload(module));CK(cuCtxDestroy(ctx));return failed?3:0;
}
