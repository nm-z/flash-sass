/* Ordinary RMS across positions: scalar/grouped/batch GPU and independent C.
 * --cpu-only validates eligibility and serialized PLE exclusion without CUDA. */
#include "prog.h"
#include "quant.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult r_=(x);if(r_){const char *s_="?";cuGetErrorString(r_,&s_);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s_);exit(2);} } while(0)
#include "test_guards.h"
static uint64_t address(const phase_t *p,unsigned w){return p->w[w]|((uint64_t)p->w[w+1]<<32);}
static void plan_checks(void) {
    const unsigned widths[]={2,8,16,32};unsigned cases=0;
    for(unsigned w=0;w<4;w++)for(unsigned change=0;change<24;change++) {
        unsigned width=widths[w];prog_t p;prog_init(&p);p.xj=7;p.pending_wait=1;p.loop=1;
        for(unsigned j=0;j<width;j++) {
            CUdeviceptr b=0x200000000ull+0x100000ull*j;
            ph_rms(&p,OP_RMS2560,b+0x1000,b+0x20000,b+0x40000,1e-6f,4);
            p.p[j].w[30]=change==18?3:change==19?6:2;
            if(change==1)ph_ptr(p.p+j,0x148,0x100000000ull);
            if(change==2)ph_f32(p.p+j,0x158,(1+j)*1e-6f);
        }
        p.pending_wait=1;p.p[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
        phase_t *a=p.p,*z=a+width-1;
        if(change==3)z->w[1]++;
        if(change==4)a->w[31]=1;
        if(change==5)z->w[31]=1;
        if(change==6)z->w[30]^=2;
        if(change==7)ph_ptr(z,0x140,0);
        if(change==8)ph_ptr(z,0x148,0);
        if(change==9)ph_ptr(z,0x150,0);
        if(change==10)ph_ptr(z,0x150,address(z,2));
        if(change==11)ph_ptr(z,0x150,address(a,2));
        if(change==12)ph_ptr(z,0x150,address(z,4));
        if(change==13)ph_ptr(z,0x150,address(a,6));
        if(change==14)a->w[1]=0;
        if(change==15)a->w[1]=65536;
        if(change==16)a->w[30]|=32;
        if(change==17)a->w[0]=OP_RMS10240;
        if(change==20)a->w[30]|=PHASE_FUSED_FLAG;
        if(change==21)a->w[30]&=~PHASE_GROUP_FLAG;
        if(change==22)ph_ptr(z,0x150,address(a,4));
        if(change==23)for(unsigned j=0;j<width;j++)ph_ptr(p.p+j,0x150,0x300000000ull);
        phase_t saved[32];memcpy(saved,p.p,width*sizeof *saved);
        int eligible=width>=8&&(change<=2||change==18||change==19);
        assert(prog_batch_fuse_rms(&p)==(eligible?(int)(width/8):0));
        assert(p.n==(int)width && p.xj==7 && p.pending_wait==1 && p.loop==1);
        if(!eligible)assert(!memcmp(saved,p.p,width*sizeof *saved));
        else for(unsigned j=0;j<width;j+=8) {
            uint32_t f=p.p[j].w[30];assert(p.p[j].w[0]==OP_RMS25608&&phase_positions(p.p+j)==8);
            assert(!!(f&2)==(j==0));assert(!!(f&1)==(change==18&&j+8==width));
            assert(!!(f&4)==(change==19||j+8<width));
        }
        memcpy(saved,p.p,width*sizeof *saved);assert(!prog_batch_fuse_rms(&p)&&!memcmp(saved,p.p,width*sizeof *saved));prog_free(&p);cases++;
    }
    for(unsigned width=8;width<=32;width*=2)for(unsigned private_output=0;private_output<2;private_output++) {
        prog_t p[32],out;const prog_t *views[32];prog_init(&out);
        for(unsigned pos=0;pos<width;pos++) {
            prog_init(p+pos);p[pos].loop=1;p[pos].token_offset=pos;views[pos]=p+pos;
            prog_add(p+pos,OP_PLEGATE,4);
            ph_rms(p+pos,OP_RMS2560,0x100000000ull,0x110000000ull,0x200000000ull+(private_output?pos*0x100000ull:0),1e-6f,4);
            prog_add(p+pos,OP_PLECONV,16);
        }
        assert(!prog_group_aliased(&out,views,width));
        phase_t *saved=malloc(out.n*sizeof *saved);memcpy(saved,out.p,out.n*sizeof *saved);
        assert(!prog_batch_fuse_rms(&out)&&!memcmp(saved,out.p,out.n*sizeof *saved));
        for(unsigned pos=0;pos<width;pos++)assert(out.p[3*pos+1].w[0]==OP_RMS2560&&phase_positions(out.p+3*pos+1)==1);
        free(saved);prog_free(&out);for(unsigned pos=0;pos<width;pos++)prog_free(p+pos);
    }
    for(unsigned units=1;units<=33;units++) {
        unsigned char seen[8][33]={{0}},elements[2560]={0};
        for(unsigned cta=0;cta<16;cta++)for(unsigned unit=cta;unit<units;unit+=16)for(unsigned warp=0;warp<8;warp++)seen[warp][unit]++;
        for(unsigned p=0;p<8;p++)for(unsigned u=0;u<units;u++)assert(seen[p][u]==1);
        for(unsigned lane=0;lane<32;lane++)for(unsigned chunk=0;chunk<20;chunk++)for(unsigned e=0;e<4;e++)elements[128*chunk+4*lane+e]++;
        for(unsigned e=0;e<2560;e++)assert(elements[e]==1);
    }
    printf("RMS B2/B8/B16/B32 plan: %u alias/ABI/publication cases,6 PLE exclusions, unit1..33 mapping PASS\n",cases);
}
static CUdeviceptr upload(const char *name,const void *source,size_t bytes) {
    CUdeviceptr p=test_guard_alloc(name,bytes,0);if(source)CK(cuMemcpyHtoD(p,source,bytes));return p;
}
static int compare(const char *name,const void *expected,const void *got,size_t bytes) {
    if(!memcmp(expected,got,bytes))return 0;
    const unsigned char *a=expected,*b=got;size_t first=0;while(first<bytes&&a[first]==b[first])first++;
    fprintf(stderr,"RMS %s mismatch at byte%zu of%zu\n",name,first,bytes);return 1;
}
static float launch(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,CUdeviceptr flags,
    const phase_t *p,unsigned width,unsigned mode) {
    uint32_t control[64]={0},f[96];for(unsigned i=0;i<96;i++)f[i]=i<16?17u:UINT32_MAX;
    for(unsigned e=0;e<6;e++){uint64_t row=flags;memcpy(control+16+2*e,&row,8);}
    uint64_t own=flags;memcpy(control+28,&own,8);control[4]=6;control[8]=17;control[9]=mode?width:0;
    CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));CK(cuMemcpyHtoD(flags,f,sizeof f));
    unsigned count=width+1,end=mode?width:1;CK(cuMemcpyHtoD(program,p,count*sizeof *p));
    void *args[]={&program,&count,&ctl,&end};CUevent begin,finish;float ms;
    CK(cuEventCreate(&begin,0));CK(cuEventCreate(&finish,0));CK(cuEventRecord(begin,0));
    CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(finish,0));CK(cuEventSynchronize(finish));
    CK(cuEventElapsedTime(&ms,begin,finish));CK(cuEventDestroy(begin));CK(cuEventDestroy(finish));return ms;
}
static uint32_t seed=5581;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static int check(CUfunction fn,unsigned width,unsigned units,unsigned pattern) {
    unsigned n=2560*units;size_t bytes=(size_t)n*4,total=(size_t)width*bytes;
    float *inputs=malloc(total),*weights=malloc(total),*oracle=malloc(total),*reference=malloc(total),*got=malloc(total),eps[32];
    if(!inputs||!weights||!oracle||!reference||!got)exit(2);
    const float magnitude[]={0,1e-12f,1e-3f,1,1e4f,1e10f};
    for(unsigned p=0;p<width;p++) {
        float *x=inputs+(size_t)p*n,*w=weights+(size_t)p*n;eps[p]=(1+p%3)*1e-6f;
        for(unsigned i=0;i<n;i++) {
            float scale=magnitude[(i/2560+p)%6];
            x[i]=pattern==0?scale*((int)(random_word()&65535)-32768)/32768.f:
                pattern==1?((i&1)?-scale:scale):(i%2560==((13*p+7)%2560)?1e4f:(i&1?-0.f:0.f));
            w[i]=((int)(random_word()&65535)-32768)/16384.f;
        }
        if(!pattern&&p)memcpy(w,weights,bytes);
        for(unsigned u=0;u<units;u++)rmsnorm_gpu(x+2560*u,w+2560*u,2560,eps[p],oracle+(size_t)p*n+2560*u);
    }
    CUdeviceptr dX[32],dW[32],dY[32];phase_t scalar[33]={0},group[33],batch[33];
    for(unsigned p=0;p<width;p++) {
        dX[p]=upload("RMS input",inputs+(size_t)p*n,bytes);
        dW[p]=!pattern&&p?dW[0]:upload("RMS weights",weights+(size_t)p*n,bytes);
        dY[p]=upload("RMS output",NULL,bytes);
        phase_t *a=scalar+p;a->w[0]=OP_RMS2560;a->w[1]=units;a->w[30]=2;
        ph_ptr(a,0x140,dX[p]);ph_ptr(a,0x148,dW[p]);ph_ptr(a,0x150,dY[p]);ph_f32(a,0x158,eps[p]);
    }
    scalar[width].w[0]=OP_EMBROW;scalar[width].w[30]=1;
    memcpy(group,scalar,(width+1)*sizeof *group);group[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
    memcpy(batch,group,(width+1)*sizeof *batch);prog_t p={.p=batch,.n=(int)width+1,.xj=1};assert(prog_batch_fuse_rms(&p)==(int)(width/8)&&p.xj==1);
    CUdeviceptr program=upload("RMS program",NULL,(width+1)*sizeof *group),ctl=upload("RMS control",NULL,8192),flags=upload("RMS flags",NULL,96*4);
    float times[3];int cpu_bad=0,gpu_bad=0,other_bad=0;
    for(unsigned mode=0;mode<3;mode++) {
        for(unsigned pos=0;pos<width;pos++)CK(cuMemsetD8(dY[pos],0xa7,bytes));
        const phase_t *program_host=mode==2?batch:mode?group:scalar;times[mode]=launch(fn,program,ctl,flags,program_host,width,mode);
        for(unsigned pos=0;pos<width;pos++) {
            CK(cuMemcpyDtoH(got+(size_t)pos*n,dY[pos],bytes));
            float *saved=malloc(bytes);CK(cuMemcpyDtoH(saved,dX[pos],bytes));other_bad+=compare("read-only input",inputs+(size_t)pos*n,saved,bytes);
            CK(cuMemcpyDtoH(saved,dW[pos],bytes));other_bad+=compare("read-only weights",weights+(size_t)pos*n,saved,bytes);free(saved);
        }
        cpu_bad+=compare("C normalized values",oracle,got,total);
        if(mode)gpu_bad+=compare("scalar normalized values",reference,got,total);else memcpy(reference,got,total);
        phase_t saved[33];CK(cuMemcpyDtoH(saved,program,(width+1)*sizeof *saved));other_bad+=compare("read-only descriptors",program_host,saved,(width+1)*sizeof *saved);
        uint32_t control[64],f[96];CK(cuMemcpyDtoH(control,ctl,sizeof control));CK(cuMemcpyDtoH(f,flags,sizeof f));
        other_bad+=control[2]!=(mode?width:1)||control[1]!=(mode?2:width+1)||control[8]!=18||control[11]!=0;
        for(unsigned i=0;i<96;i++)other_bad+=f[i]!=(i<16?18u:UINT32_MAX);
        other_bad+=test_guard_check();
    }
    printf("RMS width=%u vectors=%u pattern=%u shared_weights=%u cpu_failures=%d gpu_failures=%d guard_exchange_failures=%d scalar_ms=%.6f grouped_ms=%.6f parallel_ms=%.6f speedup=%.6f\n",width,units,pattern,!pattern,cpu_bad,gpu_bad,other_bad,times[0],times[1],times[2],times[1]/times[2]);
    test_guard_free();free(inputs);free(weights);free(oracle);free(reference);free(got);return !!(cpu_bad||gpu_bad||other_bad);
}
int main(int argc,char **argv) {
    plan_checks();if(argc==2&&!strcmp(argv[1],"--cpu-only"))return 0;if(argc!=1)return 2;
    unsigned width=8;const char *option=getenv("RMS_POSITIONS");if(option){char *end;width=(unsigned)strtoul(option,&end,10);if(*end||(width!=8&&width!=16&&width!=32))return 2;}
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char *image=getenv("FLASH_CUBIN");CK(cuModuleLoad(&mod,image?image:"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
    const unsigned units[]={1,2,3,4,7,16,17,24,33};unsigned cases=0,failed=0;
    for(unsigned j=0;j<sizeof units/sizeof *units;j++)for(unsigned pattern=0;pattern<3;pattern++){failed+=check(fn,width,units[j],pattern);cases++;}
    CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));printf("RMS width=%u cases=%u failures=%u\n",width,cases,failed);return failed?3:0;
}
