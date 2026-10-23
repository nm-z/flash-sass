/* Position-parallel ACTQ256: scalar/grouped GPU and independent C oracle.
 * --cpu-only checks eligibility and the CTA/warp/vector mapping without CUDA. */
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
static uint64_t address(const phase_t *p,unsigned word){return p->w[word]|((uint64_t)p->w[word+1]<<32);}
static void plan_checks(void) {
    const unsigned widths[]={2,8,16,32};unsigned cases=0;
    for(unsigned w=0;w<4;w++)for(unsigned change=0;change<24;change++) {
        unsigned width=widths[w];prog_t p;prog_init(&p);p.xj=7;p.pending_wait=1;p.loop=1;
        for(unsigned j=0;j<width;j++) {
            CUdeviceptr b=0x200000000ull+0x100000ull*j;
            ph_actq(&p,OP_ACTQ256,b+0x1000,b+0x20000,b+0x30000,b+0x40000,b+0x50000,256,24);
            p.p[j].w[30]=change==18?3:change==19?6:2;
        }
        p.pending_wait=1;p.p[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
        phase_t *a=p.p,*z=a+width-1;
        if(change==1)z->w[1]++;
        if(change==2)z->w[12]=640;
        if(change==3)z->w[0]=OP_ACTQ2560;
        if(change==4)a->w[31]=1;
        if(change==5)z->w[31]=1;
        if(change==6)z->w[30]^=2;
        if(change==7)ph_ptr(z,0x140,0);
        if(change==8)ph_ptr(z,0x148,0);
        if(change==9)ph_ptr(z,0x148,address(z,2));
        if(change==10)ph_ptr(z,0x148,address(a,2));
        if(change==11)ph_ptr(z,0x148,address(z,6));
        if(change==12)ph_ptr(z,0x148,address(a,4));
        if(change==13)ph_ptr(z,0x160,address(z,8));
        if(change==14)a->w[1]=0;
        if(change==15)a->w[1]=65536;
        if(change==16)a->w[30]|=32;
        if(change==17)a->w[31]=1u<<30;
        if(change==20)ph_ptr(z,0x158,address(a,2));
        if(change==21)ph_ptr(z,0x150,address(a,6));
        if(change==22)ph_ptr(z,0x160,0);
        if(change==23)a->w[0]=OP_FFNACT;
        phase_t saved[32];memcpy(saved,p.p,width*sizeof *saved);
        int eligible=width>=8 && (change==0||change==18||change==19);
        assert(prog_batch_fuse_actq(&p)==(eligible?(int)(width/8):0));
        assert(p.n==(int)width && p.xj==7 && p.pending_wait==1 && p.loop==1);
        if(!eligible)assert(!memcmp(saved,p.p,width*sizeof *saved));
        else for(unsigned j=0;j<width;j+=8) {
            uint32_t f=p.p[j].w[30];assert(p.p[j].w[0]==OP_ACTQ2568 && phase_positions(p.p+j)==8);
            assert(!!(f&2)==(j==0));assert(!!(f&1)==(change==18 && j+8==width));
            assert(!!(f&4)==(change==19 || j+8<width));
        }
        memcpy(saved,p.p,width*sizeof *saved);assert(!prog_batch_fuse_actq(&p)&&!memcmp(saved,p.p,width*sizeof *saved));prog_free(&p);cases++;
    }
    for(unsigned units=1;units<=33;units++) {
        unsigned char seen[8][33][256]={{{0}}};
        for(unsigned cta=0;cta<16;cta++)for(unsigned unit=cta;unit<units;unit+=16)
            for(unsigned warp=0;warp<8;warp++)for(unsigned lane=0;lane<32;lane++)for(unsigned j=0;j<2;j++)for(unsigned e=0;e<4;e++)seen[warp][unit][128*j+4*lane+e]++;
        for(unsigned p=0;p<8;p++)for(unsigned u=0;u<units;u++)for(unsigned e=0;e<256;e++)assert(seen[p][u][e]==1);
        for(unsigned p=0;p<8;p++)assert((p?0xbf00-128*p:0xbf80)==(p?0xbe80-128*(p-1):0xbf80));
    }
    printf("ACTQ B2/B8/B16/B32 plan: %u alias/ABI/publication cases and unit1..33 mapping PASS\n",cases);
}
static CUdeviceptr upload(const char *name,const void *source,size_t bytes) {
    CUdeviceptr p=test_guard_alloc(name,bytes,0);if(source)CK(cuMemcpyHtoD(p,source,bytes));return p;
}
static int compare(const char *name,const void *expected,const void *got,size_t bytes) {
    if(!memcmp(expected,got,bytes))return 0;
    const unsigned char *a=expected,*b=got;size_t first=0;while(first<bytes&&a[first]==b[first])first++;
    fprintf(stderr,"ACTQ %s mismatch at byte%zu of%zu\n",name,first,bytes);return 1;
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
static uint32_t seed=1771;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static int check(CUfunction fn,unsigned width,unsigned units,unsigned pattern) {
    unsigned n=256*units;size_t bytes[]={2ull*n,4ull*n/32,4ull*n/16,4ull*n/32};
    size_t total=0,offset[4];for(unsigned j=0;j<4;j++){offset[j]=total;total+=width*bytes[j];}
    float *inputs=malloc((size_t)width*n*4);unsigned char *oracle=malloc(total),*reference=malloc(total),*got=malloc(total);
    if(!inputs||!oracle||!reference||!got)exit(2);
    const float magnitude[]={0,0x1p-110f,0x1p-40f,0x1p-8f,1,0x1p8f,0x1p30f};
    for(unsigned p=0;p<width;p++) {
        float *x=inputs+(size_t)p*n;
        for(unsigned i=0;i<n;i++) {
            unsigned block=i/32,lane=i%32;float scale=magnitude[(block+p+pattern)%7];
            x[i]=pattern==0?scale*((int)(random_word()&65535)-32768)/32768.f:
                pattern==1?(lane&1?-scale:scale):(lane==0?scale:scale*((int)(random_word()%127)-63)/16384.f);
        }
        quant_act_gpu32(x,n,(int16_t *)(oracle+offset[0]+p*bytes[0]),(int32_t *)(oracle+offset[1]+p*bytes[1]),(int32_t *)(oracle+offset[2]+p*bytes[2]),(float *)(oracle+offset[3]+p*bytes[3]));
    }
    CUdeviceptr dX[32],output[32][4];phase_t scalar[33]={0},group[33],batch[33];
    for(unsigned p=0;p<width;p++) {
        dX[p]=upload("ACTQ input",inputs+(size_t)p*n,n*4);
        for(unsigned j=0;j<4;j++)output[p][j]=upload("ACTQ output",NULL,bytes[j]);
        phase_t *a=scalar+p;a->w[0]=OP_ACTQ256;a->w[1]=units;a->w[30]=2;
        ph_ptr(a,0x140,dX[p]);for(unsigned j=0;j<4;j++)ph_ptr(a,0x148+8*j,output[p][j]);a->w[12]=256;
    }
    scalar[width].w[0]=OP_EMBROW;scalar[width].w[30]=1; /* one terminal publication */
    memcpy(group,scalar,(width+1)*sizeof *group);group[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
    memcpy(batch,group,(width+1)*sizeof *batch);prog_t p={.p=batch,.n=(int)width+1,.xj=1};assert(prog_batch_fuse_actq(&p)==(int)(width/8)&&p.xj==1);
    CUdeviceptr program=upload("ACTQ program",NULL,(width+1)*sizeof *group),ctl=upload("ACTQ control",NULL,8192),flags=upload("ACTQ flags",NULL,96*4);
    float times[3];int cpu_bad=0,gpu_bad=0,other_bad=0;
    for(unsigned mode=0;mode<3;mode++) {
        for(unsigned pos=0;pos<width;pos++)for(unsigned j=0;j<4;j++)CK(cuMemsetD8(output[pos][j],0xa7,bytes[j]));
        const phase_t *program_host=mode==2?batch:mode?group:scalar;times[mode]=launch(fn,program,ctl,flags,program_host,width,mode);
        for(unsigned pos=0;pos<width;pos++) {
            for(unsigned j=0;j<4;j++)CK(cuMemcpyDtoH(got+offset[j]+pos*bytes[j],output[pos][j],bytes[j]));
            float *saved=malloc(n*4);CK(cuMemcpyDtoH(saved,dX[pos],n*4));other_bad+=compare("read-only input",inputs+(size_t)pos*n,saved,n*4);free(saved);
        }
        cpu_bad+=compare("C codes/sums/scales",oracle,got,total);
        if(mode)gpu_bad+=compare("scalar codes/sums/scales",reference,got,total);else memcpy(reference,got,total);
        phase_t saved[33];CK(cuMemcpyDtoH(saved,program,(width+1)*sizeof *saved));other_bad+=compare("read-only descriptors",program_host,saved,(width+1)*sizeof *saved);
        uint32_t control[64],f[96];CK(cuMemcpyDtoH(control,ctl,sizeof control));CK(cuMemcpyDtoH(f,flags,sizeof f));
        other_bad+=control[2]!=(mode?width:1)||control[1]!=(mode?2:width+1)||control[8]!=18||control[11]!=0;
        for(unsigned i=0;i<96;i++)other_bad+=f[i]!=(i<16?18u:UINT32_MAX);
        other_bad+=test_guard_check();
    }
    printf("ACTQ width=%u vectors=%u pattern=%u cpu_failures=%d gpu_failures=%d guard_exchange_failures=%d scalar_ms=%.6f grouped_ms=%.6f parallel_ms=%.6f speedup=%.6f\n",width,units,pattern,cpu_bad,gpu_bad,other_bad,times[0],times[1],times[2],times[1]/times[2]);
    test_guard_free();free(inputs);free(oracle);free(reference);free(got);return !!(cpu_bad||gpu_bad||other_bad);
}
int main(int argc,char **argv) {
    plan_checks();if(argc==2&&!strcmp(argv[1],"--cpu-only"))return 0;if(argc!=1)return 2;
    unsigned width=8;const char *option=getenv("ACTQ_POSITIONS");if(option){char *end;width=(unsigned)strtoul(option,&end,10);if(*end||(width!=8&&width!=16&&width!=32))return 2;}
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char *image=getenv("FLASH_CUBIN");CK(cuModuleLoad(&mod,image?image:"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
    const unsigned units[]={1,2,3,7,10,16,17,24,33};unsigned cases=0,failed=0;
    for(unsigned j=0;j<sizeof units/sizeof *units;j++)for(unsigned pattern=0;pattern<3;pattern++){failed+=check(fn,width,units[j],pattern);cases++;}
    CK(cuModuleUnload(mod));CK(cuCtxDestroy(ctx));printf("ACTQ width=%u cases=%u failures=%u\n",width,cases,failed);return failed?3:0;
}
