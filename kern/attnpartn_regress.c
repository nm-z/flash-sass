/* Fixed-input old/new ATTNPARTN comparison through the real persistent dispatcher.
 * Usage: attnpartn_regress old.cubin new.cubin keys [seed] [repeats]
 * All input and padding values are finite; empty chunks retain the documented -1e30 sentinel.
 * GPU events cover 64 positions per launch: a partial-only program and a partial+merge program.
 * Kernel entry is amortized over 64 steps; every step includes the real dispatcher and barriers.
 */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "dieops.h"
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult r=(x); if(r) {const char *e="?";cuGetErrorString(r,&e);fprintf(stderr,"%s: %s\n",#x,e);exit(2);} } while(0)
typedef struct {uint32_t w[32];} desc;
static void ptr(desc *p,int a,CUdeviceptr v){int w=2+(a-0x140)/4;p->w[w]=(uint32_t)v;p->w[w+1]=(uint32_t)(v>>32);}
static CUdeviceptr alloc(size_t n){CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemsetD8(p,0,n));return p;}
static uint32_t rng;
static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)&65535)/32768.f-1.f;}
static int cmp(const char *what,const float *a,const float *b,size_t n){
    size_t bad=0;for(size_t i=0;i<n;i++)if(memcmp(a+i,b+i,4)){if(!bad)fprintf(stderr,"%s[%zu] %.9g / %.9g\n",what,i,a[i],b[i]);bad++;}
    return bad!=0;
}
static int order(const void *a,const void *b){float x=*(const float*)a,y=*(const float*)b;return(x>y)-(x<y);}
static void stat(const char *name,float *s,int n){float *c=malloc(n*sizeof *c);memcpy(c,s,n*sizeof *c);qsort(c,n,sizeof *c,order);printf("\"%s\":{\"median_us\":%.3f,\"p95_us\":%.3f,\"samples_us\":[",name,(n&1)?c[n/2]:.5f*(c[n/2-1]+c[n/2]),c[(int)ceil(.95*n)-1]);for(int i=0;i<n;i++)printf("%s%.3f",i?",":"",s[i]);printf("]}");free(c);}
int main(int argc,char **argv){
    if(argc<4){fputs("usage: attnpartn_regress old.cubin new.cubin keys [seed] [repeats]\n",stderr);return 2;}
    char *e;unsigned n=(unsigned)strtoul(argv[3],&e,10);if(*e||!n||n>16384)return 2;
    unsigned seed=argc>4?(unsigned)strtoul(argv[4],0,10):77;int rep=argc>5?atoi(argv[5]):32;if(rep<4||rep>10000||(rep&1))return 2;rng=seed;
    unsigned npos=(n+8192+64+31)/32*32;
    size_t kc_n=ATTN_KC_FLOATS(npos),vc_n=(size_t)npos*512;
    float *q=malloc(6144*4),*g=malloc(12288*4),*k=calloc(kc_n,4),*v=calloc(vc_n,4);
    for(int i=0;i<6144;i++)q[i]=seed?2*rnd():0.f;
    for(int i=0;i<12288;i++)g[i]=4*rnd();
    for(unsigned t=0;t<n;t++)for(int h=0;h<2;h++)for(int j=0;j<256;j++){k[attn_kidx(t,h,j)]=2*rnd();v[(size_t)t*512+256*h+j]=rnd();}
    const size_t sz[4]={192,192,192*256,6144};
    float *ref[4],*got[2][4];
    for(int a=0;a<4;a++){ref[a]=malloc(sz[a]*4);for(int j=0;j<2;j++)got[j][a]=malloc(sz[a]*4);}
    attn_partials_gpu(q,k,v,n,ref[0],ref[1],ref[2]);attn_merge_gpu(ref[0],ref[1],ref[2],g,ref[3]);
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    char device[256];CK(cuDeviceGetName(device,sizeof device,dev));
    CUmodule mod[2];CUfunction fn[2];for(int j=0;j<2;j++){CK(cuModuleLoad(&mod[j],argv[j+1]));CK(cuModuleGetFunction(&fn[j],mod[j],"die"));}
    CUdeviceptr dq=alloc(6144*4),dg=alloc(12288*4),dk=alloc(kc_n*4),dv=alloc(vc_n*4),dout[4],dp=alloc(256),dc=alloc(8192);
    for(int a=0;a<4;a++)dout[a]=alloc(sz[a]*4);
    CK(cuMemcpyHtoD(dq,q,6144*4));CK(cuMemcpyHtoD(dg,g,12288*4));CK(cuMemcpyHtoD(dk,k,kc_n*4));CK(cuMemcpyHtoD(dv,v,vc_n*4));
    desc p[2]={0};p[0].w[0]=OP_ATTNPARTN;p[0].w[1]=32;
    ptr(&p[0],0x140,dq);ptr(&p[0],0x148,dk);ptr(&p[0],0x150,dv);ptr(&p[0],0x158,dout[0]);ptr(&p[0],0x160,dout[1]);ptr(&p[0],0x168,dout[2]);p[0].w[14]=n;p[0].w[15]=(n+3)/4;
    p[1].w[0]=OP_ATTNMERGE;p[1].w[1]=24;ptr(&p[1],0x140,dout[0]);ptr(&p[1],0x148,dout[1]);ptr(&p[1],0x150,dout[2]);ptr(&p[1],0x158,dg);ptr(&p[1],0x160,dout[3]);
    unsigned trace_ctas = getenv("ATTN_TRACE_CTAS") ? (unsigned)atoi(getenv("ATTN_TRACE_CTAS")) : 0;
    if (trace_ctas && trace_ctas != 1 && trace_ctas != 16) return 2;
    CUdeviceptr dt = trace_ctas ? alloc(2 * trace_ctas * 64) : 0;
    uint64_t *trace = trace_ctas ? malloc(2 * trace_ctas * 64) : NULL;
    if (trace_ctas) {uint32_t meta[5]={(uint32_t)dt,(uint32_t)(dt>>32),0,1,trace_ctas};CK(cuMemcpyHtoD(dc+0xd0,meta,sizeof meta));}
    CK(cuMemcpyHtoD(dp,p,sizeof p));unsigned nph=2,ntok=1;void *args[]={&dp,&nph,&dc,&ntok};int bad=0;
    const char *names[4]={"maxima","denominators","weighted_values","outputs"};
    for(int j=0;j<2;j++){
        CK(cuMemsetD32(dc+8,0,1));for(int a=0;a<4;a++)CK(cuMemsetD32(dout[a],0x7fc00000,sz[a]));
        CK(cuLaunchKernel(fn[j],16,1,1,256,1,1,0,0,args,0));CK(cuCtxSynchronize());
        uint32_t count;CK(cuMemcpyDtoH(&count,dc+8,4));if(count!=1){fprintf(stderr,"token count %u\n",count);return 1;}
        for(int a=0;a<4;a++){CK(cuMemcpyDtoH(got[j][a],dout[a],sz[a]*4));bad|=cmp(names[a],ref[a],got[j][a],sz[a]);}
        if (trace_ctas) {
            CK(cuMemcpyDtoH(trace,dt,2 * trace_ctas * 64));
            for(unsigned c=0;c<2*trace_ctas;c++) {
                uint64_t *z=trace+8*c;
                if(!z[0]||z[0]>z[1]||z[1]>z[2]||z[2]>z[3]||z[3]>z[4]) {fprintf(stderr,"bad trace variant=%d record=%u\n",j,c);return 1;}
                fprintf(stderr,"TRACE variant=%d phase=%u cta=%u start=%llu wait_start=%llu wait_end=%llu body_end=%llu end=%llu\n",j,c/trace_ctas,c%trace_ctas,(unsigned long long)z[0],(unsigned long long)z[1],(unsigned long long)z[2],(unsigned long long)z[3],(unsigned long long)z[4]);
            }
        }
    }
    for(int a=0;a<4;a++)bad|=cmp(names[a],got[0][a],got[1][a],sz[a]);
    if(bad){fprintf(stderr,"FAIL keys=%u seed=%u\n",n,seed);return 1;}
    CUevent start,end;CK(cuEventCreate(&start,0));CK(cuEventCreate(&end,0));
    float *total[2],*part[2],*merge[2];for(int j=0;j<2;j++){total[j]=malloc(rep*4);part[j]=malloc(rep*4);merge[j]=malloc(rep*4);}
    /* Repeated positions in one launch keep the dispatcher and instruction cache warm.
     * Separate one-phase and two-phase programs isolate partial cost without misreading
     * last-token timestamps as first-token timestamps. Alternate balanced variant order. */
    const unsigned steps=64;
    for(int it=-4;it<rep;it++)for(int jj=0;jj<2;jj++){
        int j=(it&1)?jj:1-jj;
        for(unsigned phases=1;phases<=2;phases++){
            nph=phases;ntok=steps;CK(cuMemsetD32(dc+8,0,1));CK(cuEventRecord(start,0));
            CK(cuLaunchKernel(fn[j],16,1,1,256,1,1,0,0,args,0));CK(cuEventRecord(end,0));CK(cuEventSynchronize(end));
            float ms;CK(cuEventElapsedTime(&ms,start,end));
            if(!isfinite(ms)||ms>=1000.f){fputs("timing window too long for 32-bit phase deltas\n",stderr);return 1;}
            uint32_t count;CK(cuMemcpyDtoH(&count,dc+8,4));if(count!=steps)return 1;
            if(it>=0){
                if(phases==1)part[j][it]=1000*ms/steps;
                else {uint32_t ts[2];CK(cuMemcpyDtoH(ts,dc+0x100,8));total[j][it]=1000*ms/steps;merge[j][it]=(uint32_t)(ts[1]-ts[0])*.001f;}
            }
        }
    }
    printf("{\"keys\":%u,\"seed\":%u,\"device\":\"%s\",\"bit_exact\":true,\"steps_per_sample\":64,\"partial_values\":%zu,\"final_values\":6144,\"old\":{",n,seed,device,sz[0]+sz[1]+sz[2]);
    stat("event",total[0],rep);printf(",");stat("partial_program",part[0],rep);printf(",");stat("merge",merge[0],rep);printf("},\"new\":{");stat("event",total[1],rep);printf(",");stat("partial_program",part[1],rep);printf(",");stat("merge",merge[1],rep);puts("}}");
    CK(cuCtxDestroy(ctx));return 0;
}
