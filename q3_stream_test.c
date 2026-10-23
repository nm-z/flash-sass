/* Isolated Q3 occupancy experiment. Every output must equal both the original
 * scalar GPU image and gemv_q3_k_ref. GPU execution belongs in gpuq.
 * Timings use CUDA events over repeated launches and include launch gaps. */
#define _POSIX_C_SOURCE 200809L
#include "quant.h"
#include <cuda.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do{CUresult r_=(x);if(r_){const char*s_="?";cuGetErrorString(r_,&s_);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s_);exit(2);}}while(0)
#include "test_guards.h"
static uint32_t seed=7321;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static int compare(const char *name,const float *want,const float *got,unsigned rows){
    for(unsigned i=0;i<rows;i++)if(memcmp(want+i,got+i,4)){
        fprintf(stderr,"%s row=%u want=%a got=%a\n",name,i,want[i],got[i]);return 1;
    }return 0;
}
static int run_case(unsigned K,unsigned rows,unsigned pattern,unsigned iters,CUfunction functions[3]){
    unsigned L=q4_k_lanes_per_row((int)K),Q=L/8,B=K/256,stride=q3_k_tile_stride((int)K);
    size_t bytes=(size_t)rows*stride;
    /* The original standalone scalar image preloads every initial warp row.
     * Preserve that existing readable extent for small logical-row fixtures;
     * output allocation remains exact, so its tail stores are still guarded. */
    unsigned matrix_rows=rows<32*256/L?32*256/L:rows;
    size_t matrix_bytes=(size_t)matrix_rows*stride;
    unsigned char *weights=calloc(1,matrix_bytes);float *x=malloc(K*4),*as=malloc(K/32*4);
    int16_t *x16=malloc(K*2);int8_t *x8=malloc(K);int32_t *s32=malloc(K/32*4),*s16=malloc(K/16*4);
    float *ref=malloc(rows*4),*oracle=malloc(rows*4),*got=malloc(rows*4);
    if(!weights||!x||!as||!x16||!x8||!s32||!s16||!ref||!oracle||!got)exit(2);
    for(unsigned r=0;r<matrix_rows;r++)for(unsigned b=0;b<B;b++){
        unsigned char *row=weights+(size_t)r*stride,*rec=row+16*Q+112*b;
        uint16_t d=(uint16_t)(0x2400+(random_word()&0x1fff));
        memcpy(row+16*(b%Q)+2*(b/Q),&d,2);
        for(unsigned g=0;g<16;g++){
            uint32_t code=pattern==1?0:pattern==2?UINT32_MAX:random_word();
            uint16_t high=pattern==1?UINT16_MAX:pattern==2?0:(uint16_t)random_word();
            int8_t scale=pattern==1?-32:pattern==2?31:(int8_t)((int)(random_word()%64)-32);
            memcpy(rec+4*g,&code,4);memcpy(rec+64+2*g,&high,2);rec[96+g]=(unsigned char)scale;
        }
    }
    for(unsigned j=0;j<K;j++){
        x[j]=pattern==1?-1.f:pattern==2?1.f:pattern==3?(j%71==0?3.f:0.f):
            ((int)(random_word()&65535)-32768)/8192.f;
    }
    quant_act_gpu32(x,(int)K,x16,s32,s16,as);for(unsigned j=0;j<K;j++)x8[j]=(int8_t)x16[j];
    gemv_q3_k_ref(weights,(int)rows,(int)K,x8,as,ref);
    CUdeviceptr dw=test_guard_alloc("weights",matrix_bytes,0),dx=test_guard_alloc("X16",K*2,0);
    CUdeviceptr ds=test_guard_alloc("sums",K/32*4,0),dy=test_guard_alloc("output",rows*4,0);
    CUdeviceptr dd=test_guard_alloc("debug",4096,0),da=test_guard_alloc("AS",K/32*4,0);
    CK(cuMemcpyHtoD(dw,weights,matrix_bytes));CK(cuMemcpyHtoD(dx,x16,K*2));
    CK(cuMemcpyHtoD(ds,s32,K/32*4));CK(cuMemcpyHtoD(da,as,K/32*4));
    float xs=1.f;void *args[]={&dw,&dx,&ds,&dy,&rows,&xs,&dd,&da};int bad=0;
    const char *names[]={"original","stream","stream_pinned249"};
    for(unsigned f=0;f<3;f++)for(unsigned variant=0;variant<2;variant++){
        unsigned grid=variant?32:16;
        CK(cuMemsetD32(dy,0x7fc12345u,rows));
        CK(cuLaunchKernel(functions[f],grid,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
        CK(cuMemcpyDtoH(got,dy,rows*4));
        int b=compare("C reference",ref,got,rows);
        if(!f&&!variant)memcpy(oracle,got,rows*4);else b|=compare("original scalar GPU",oracle,got,rows);
        bad+=b;
        if(rows==16384&&!pattern&&iters){
            CUevent begin,end;float ms;
            CK(cuEventCreate(&begin,0));CK(cuEventCreate(&end,0));CK(cuEventRecord(begin,0));
            for(unsigned i=0;i<iters;i++)CK(cuLaunchKernel(functions[f],grid,1,1,256,1,1,0,0,args,NULL));
            CK(cuEventRecord(end,0));CK(cuEventSynchronize(end));CK(cuEventElapsedTime(&ms,begin,end));
            printf("{\"kind\":\"q3_occupancy_timing\",\"kernel\":\"%s\",\"K\":%u,\"rows\":%u,\"grid\":%u,\"launches\":%u,\"event_ms\":%.6f,\"event_us_per_launch\":%.6f,\"tile_GB_s\":%.6f,\"exact\":%s,\"clock\":\"CUDA_events_including_launch_gaps\"}\n",names[f],K,rows,grid,iters,ms,ms*1000/iters,(double)bytes*iters/(ms*1e6),b?"false":"true");
            CK(cuEventDestroy(begin));CK(cuEventDestroy(end));
            CK(cuMemcpyDtoH(got,dy,rows*4));bad+=compare("timed output",ref,got,rows);
        }
    }
    bad+=test_guard_check();test_guard_free();
    printf("q3_stream K=%u rows=%u pattern=%u C_and_original_GPU=%s\n",K,rows,pattern,bad?"FAIL":"BIT-EXACT");
    free(weights);free(x);free(as);free(x16);free(x8);free(s32);free(s16);free(ref);free(oracle);free(got);
    return bad;
}
int main(int argc,char **argv){
    unsigned iters=32;if(argc>2){fprintf(stderr,"usage: %s [timing_iterations=32]\n",argv[0]);return 2;}
    if(argc==2){char *end;unsigned long n=strtoul(argv[1],&end,10);if(!*argv[1]||*end||n>256)return 2;iters=(unsigned)n;}
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    unsigned sizes[]={1,7,15,16,17,31,32,33,255,257,16384};int bad=0;unsigned cases=0;
    const char *prefixes[]={"q3_stream_original","q3_stream_lean","q3_stream_pinned"};
    for(unsigned K=2560;K<=6144;K+=3584){
        CUmodule modules[3];CUfunction functions[3];
        for(unsigned f=0;f<3;f++){
            char path[96];snprintf(path,sizeof path,"%s_k%u.cubin",prefixes[f],K);
            CK(cuModuleLoad(modules+f,path));CK(cuModuleGetFunction(functions+f,modules[f],"gemv_q3_k"));
            int regs,shared,resident;CK(cuFuncGetAttribute(&regs,CU_FUNC_ATTRIBUTE_NUM_REGS,functions[f]));
            CK(cuFuncGetAttribute(&shared,CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,functions[f]));
            CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,functions[f],256,0));
            printf("kernel=%s K=%u registers=%d shared=%d resident_CTAs_per_SM=%d\n",prefixes[f],K,regs,shared,resident);
            if(f==1&&(regs>128||resident<2)){fputs("stream kernel did not reach the intended occupancy\n",stderr);return 4;}
        }
        for(unsigned s=0;s<sizeof sizes/sizeof sizes[0];s++)for(unsigned pattern=0;pattern<(sizes[s]==16384?1u:4u);pattern++){
            bad+=run_case(K,sizes[s],pattern,iters,functions);cases++;if(bad)goto done;
        }
        for(unsigned f=0;f<3;f++)CK(cuModuleUnload(modules[f]));
    }
done:
    printf("q3_stream_test cases=%u failures=%d\n",cases,bad);CK(cuCtxDestroy(ctx));return bad?3:0;
}
