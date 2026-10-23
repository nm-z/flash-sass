/* Cross-die publication ordering with fresh, distinguishable payloads, two
 * alternating inbox slots, persistent generations, and alternating launch order.
 * Run on two reserved devices through gpuq; does not map model weight arenas. */
#define _POSIX_C_SOURCE 200809L
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "dieops.h"
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do {CUresult cu_status_=(x);if(cu_status_){const char *s="?";cuGetErrorString(cu_status_,&s);fprintf(stderr,"%s: %s\n",#x,s);exit(2);}}while(0)
enum {D=2,WORDS=10240,HALF=5120,SLOT=65536,NPH=4};
typedef struct {uint32_t w[32];} phase;
static void ptr(phase *p,unsigned a,CUdeviceptr v){unsigned w=2+(a-0x140)/4;p->w[w]=(uint32_t)v;p->w[w+1]=(uint32_t)(v>>32);}
static void val(phase *p,unsigned a,uint32_t v){p->w[2+(a-0x140)/4]=v;}
static CUdeviceptr alloc(size_t n){CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemsetD8(p,0,n));return p;}
static uint32_t payload(unsigned epoch,unsigned die,unsigned arm,unsigned word){
    return 0x13579bdfu ^ (epoch+1)*0x9e3779b9u ^ (die+1)*0x85ebca6bu ^ arm*0xc2b2ae35u ^ word*0x27d4eb2du;
}
int main(int argc,char **argv){
    int mapped=argc==3&&!strcmp(argv[1],"--host");
    if(argc!=2&&!mapped){fprintf(stderr,"usage: exchange_test [--host] die.cubin\n");return 2;}
    const char *image=argv[mapped?2:1];
    CUcontext ctx[D];CUmodule mod[D];CUfunction fn[D];
    CUdeviceptr inbox[D],flags[D],ctl[D],table[D],program[D],input[D][2],output[D][2];
    void *host_inbox[D]={0},*host_flags[D]={0};
    CK(cuInit(0));int count=0;CK(cuDeviceGetCount(&count));if(count<D)return 2;
    for(unsigned d=0;d<D;d++){
        CUdevice dev;CK(cuDeviceGet(&dev,(int)d));CK(cuCtxCreate_v2(&ctx[d],CU_CTX_MAP_HOST,dev));
        CK(cuModuleLoad(&mod[d],image));CK(cuModuleGetFunction(&fn[d],mod[d],"die"));
        if(mapped){
            CK(cuMemHostAlloc(&host_inbox[d],2*SLOT,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
            CK(cuMemHostAlloc(&host_flags[d],512,CU_MEMHOSTALLOC_PORTABLE|CU_MEMHOSTALLOC_DEVICEMAP));
            memset(host_inbox[d],0,2*SLOT);memset(host_flags[d],0,512);
        }else{inbox[d]=alloc(2*SLOT);flags[d]=alloc(512);}
        ctl[d]=alloc(16384);table[d]=alloc(128);program[d]=alloc(sizeof(phase)*NPH);
        for(unsigned a=0;a<2;a++){input[d][a]=alloc(HALF*4);output[d][a]=alloc(WORDS*4);}
    }
    for(unsigned d=0;d<D;d++){
        CK(cuCtxSetCurrent(ctx[d]));
        CUdeviceptr in[D],fl[D];
        for(unsigned e=0;e<D;e++){
            if(mapped){CK(cuMemHostGetDevicePointer(&in[e],host_inbox[e],0));CK(cuMemHostGetDevicePointer(&fl[e],host_flags[e],0));}
            else{in[e]=inbox[e];fl[e]=flags[e];if(e!=d)CK(cuCtxEnablePeerAccess(ctx[e],0));}
        }
        uint64_t tab[12]={0};for(unsigned e=0;e<6;e++){tab[e]=in[e<D?e:0];tab[6+e]=fl[e<D?e:0];}
        CK(cuMemcpyHtoD(table[d],tab,sizeof tab));
        uint32_t f[96];for(unsigned i=0;i<96;i++)f[i]=i/16<D?0:UINT32_MAX;
        if(mapped)memcpy(host_flags[d],f,sizeof f);else CK(cuMemcpyHtoD(flags[d],f,sizeof f));
        uint32_t hdr[32]={0};hdr[3]=d;hdr[4]=D;
        for(unsigned e=0;e<6;e++){uint64_t p=e<D?fl[e]+64*d:ctl[d]+0xc0;memcpy(hdr+16+2*e,&p,8);}
        memcpy(hdr+28,&fl[d],8);CK(cuMemcpyHtoD(ctl[d],hdr,sizeof hdr));
        phase p[NPH]={0};
        for(unsigned a=0;a<2;a++){
            phase *send=&p[2*a],*copy=&p[2*a+1];
            send->w[0]=OP_XSEND;send->w[1]=16;send->w[30]=1;
            ptr(send,0x140,table[d]);ptr(send,0x148,input[d][a]);val(send,0x150,HALF*4);val(send,0x154,d*HALF*4);val(send,0x158,a*SLOT);val(send,0x15c,D);
            ptr(send,0x160,ctl[d]+0x20);val(send,0x168,d);
            copy->w[0]=OP_XCOPY;copy->w[1]=10;copy->w[30]=2;
            ptr(copy,0x140,in[d]+a*SLOT);ptr(copy,0x148,output[d][a]);
        }
        CK(cuMemcpyHtoD(program[d],p,sizeof p));
    }
    uint32_t *host=malloc(WORDS*4);if(!host)return 2;
    unsigned errors=0;
    for(unsigned epoch=0;epoch<128;epoch++){
        for(unsigned d=0;d<D;d++){
            CK(cuCtxSetCurrent(ctx[d]));
            CK(cuMemsetD32(ctl[d]+8,0,1));CK(cuMemsetD32(ctl[d]+0x14,0,2));
            for(unsigned a=0;a<2;a++){
                for(unsigned i=0;i<HALF;i++)host[i]=payload(epoch,d,a,i);
                CK(cuMemcpyHtoD(input[d][a],host,HALF*4));CK(cuMemsetD32(output[d][a],0xdeadbeef,WORDS));
            }
        }
        for(unsigned order=0;order<D;order++){
            unsigned d=(epoch+order)%D;CK(cuCtxSetCurrent(ctx[d]));
            unsigned nph=NPH,ntok=1;void *params[]={&program[d],&nph,&ctl[d],&ntok};
            CK(cuLaunchKernel(fn[d],16,1,1,256,1,1,0,0,params,NULL));
            if(!order && epoch%7==0){struct timespec delay={0,100000};nanosleep(&delay,NULL);}
        }
        for(unsigned d=0;d<D;d++){
            CK(cuCtxSetCurrent(ctx[d]));CK(cuCtxSynchronize());
            uint32_t generation=0,tokens=0;CK(cuMemcpyDtoH(&generation,ctl[d]+0x20,4));CK(cuMemcpyDtoH(&tokens,ctl[d]+8,4));
            if(generation!=2*(epoch+1)||tokens!=1){fprintf(stderr,"bad completion epoch=%u die=%u gen=%u tokens=%u\n",epoch,d,generation,tokens);errors++;}
            for(unsigned a=0;a<2;a++){
                CK(cuMemcpyDtoH(host,output[d][a],WORDS*4));
                for(unsigned i=0;i<WORDS;i++)if(host[i]!=payload(epoch,i/HALF,a,i%HALF)){
                    fprintf(stderr,"stale/corrupt payload epoch=%u die=%u arm=%u word=%u got=%08x expected=%08x\n",epoch,d,a,i,host[i],payload(epoch,i/HALF,a,i%HALF));errors++;break;
                }
            }
        }
        if(errors)break;
    }
    if(mapped)for(unsigned d=0;d<D;d++){CK(cuMemFreeHost(host_inbox[d]));CK(cuMemFreeHost(host_flags[d]));}
    for(unsigned d=0;d<D;d++){CK(cuCtxSetCurrent(ctx[d]));CK(cuCtxDestroy(ctx[d]));}
    free(host);printf("cross-die %s exchange: 128 epochs, two slots, alternating launch order: %s\n",mapped?"host-mapped":"peer-memory",errors?"FAIL":"PASS");return errors?1:0;
}
