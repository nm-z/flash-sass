/* CPU fault-injection oracle for the borrowed entry's ownership and protocol.
 * Every CUDA entry used here is mocked; no device/context/module is created. */
#define _GNU_SOURCE
#include "hcup_offload_probe.h"
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
static CUcontext current;
static int low_die,fail_alloc,fail_launch,fail_read,alloc_calls,launch_calls,read_calls;
static unsigned device_live,pinned_live,streams_live,recovery_copies,sync_calls;
CUresult cuCtxGetCurrent(CUcontext *c){*c=current;return CUDA_SUCCESS;}
CUresult cuCtxSetCurrent(CUcontext c){current=c;return CUDA_SUCCESS;}
CUresult cuCtxSynchronize(void){sync_calls++;return CUDA_SUCCESS;}
CUresult cuMemGetInfo(size_t *free_bytes,size_t *total){*total=64u<<20;*free_bytes=current==(CUcontext)(uintptr_t)(low_die+1)?0:*total;return CUDA_SUCCESS;}
CUresult cuGetErrorString(CUresult r,const char **s){(void)r;*s="mock CUDA error";return CUDA_SUCCESS;}
CUresult cuMemAlloc(CUdeviceptr *p,size_t n) {
    if(++alloc_calls==fail_alloc)return CUDA_ERROR_OUT_OF_MEMORY;
    void *a=malloc(n);assert(a);*p=(uintptr_t)a;device_live++;return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr p){free((void *)(uintptr_t)p);assert(device_live);device_live--;return CUDA_SUCCESS;}
CUresult cuMemsetD8(CUdeviceptr p,unsigned char value,size_t n){memset((void *)(uintptr_t)p,value,n);return CUDA_SUCCESS;}
CUresult cuMemsetD32(CUdeviceptr p,unsigned value,size_t n){unsigned *q=(unsigned *)(uintptr_t)p;for(size_t i=0;i<n;i++)q[i]=value;return CUDA_SUCCESS;}
CUresult cuMemcpyHtoD(CUdeviceptr p,const void *s,size_t n){memcpy((void *)(uintptr_t)p,s,n);return CUDA_SUCCESS;}
CUresult cuMemcpyDtoH(void *p,CUdeviceptr s,size_t n){if(++read_calls==fail_read)return CUDA_ERROR_OUT_OF_MEMORY;memcpy(p,(void *)(uintptr_t)s,n);return CUDA_SUCCESS;}
CUresult cuMemHostAlloc(void **p,size_t n,unsigned flags){(void)flags;*p=malloc(n);assert(*p);pinned_live++;return CUDA_SUCCESS;}
CUresult cuMemFreeHost(void *p){free(p);assert(pinned_live);pinned_live--;return CUDA_SUCCESS;}
CUresult cuStreamCreate(CUstream *s,unsigned flags){(void)flags;*s=(CUstream)(uintptr_t)++streams_live;return CUDA_SUCCESS;}
CUresult cuStreamDestroy(CUstream s){(void)s;assert(streams_live);streams_live--;return CUDA_SUCCESS;}
CUresult cuStreamSynchronize(CUstream s){(void)s;return CUDA_SUCCESS;}
CUresult cuMemcpyHtoDAsync(CUdeviceptr p,const void *s,size_t n,CUstream stream){(void)stream;recovery_copies++;return cuMemcpyHtoD(p,s,n);}
CUresult cuLaunchKernel(CUfunction f,unsigned gx,unsigned gy,unsigned gz,unsigned bx,unsigned by,unsigned bz,
    unsigned shared,CUstream stream,void **args,void **extra) {
    (void)f;(void)gx;(void)gy;(void)gz;(void)bx;(void)by;(void)bz;(void)shared;(void)stream;(void)args;(void)extra;
    return ++launch_calls==fail_launch?CUDA_ERROR_OUT_OF_MEMORY:CUDA_SUCCESS;
}
static void check(int low,int allocation,int launch,int read) {
    low_die=low;fail_alloc=allocation;fail_launch=launch;fail_read=read;
    alloc_calls=launch_calls=read_calls=0;device_live=pinned_live=streams_live=recovery_copies=sync_calls=0;
    CUcontext saved=(CUcontext)(uintptr_t)99;current=saved;
    CUcontext contexts[6];CUfunction functions[6];for(unsigned d=0;d<6;d++){contexts[d]=(CUcontext)(uintptr_t)(d+1);functions[d]=(CUfunction)(uintptr_t)(d+1);}
    uint16_t *weights[2]={calloc(4u*17*320,2),calloc(4u*33*320,2)};assert(weights[0] && weights[1]);
    char *log=NULL;size_t bytes=0;FILE *out=open_memstream(&log,&bytes);assert(out);
    int result=hcup_offload_probe(out,contexts,functions,6,1,(uintptr_t)weights[0],(uintptr_t)weights[1]);
    assert(result==-CUDA_ERROR_OUT_OF_MEMORY);assert(current==saved);
    assert(!device_live && !pinned_live && !streams_live);assert(sync_calls>=6);
    assert(!fclose(out));assert(log && strstr(log,"hcup_offload_probe done status=FAIL result=-2 cleanup_failures=0 allocations_live=0 host_live=0 contexts=preserved modules=preserved"));
    if(low>=0)assert(!alloc_calls && !launch_calls);
    if(launch==2)assert(recovery_copies==2);else assert(!recovery_copies);
    free(log);free(weights[0]);free(weights[1]);
}
int main(void) {
    check(4,0,0,0);check(5,0,0,0);
    for(int allocation=1;allocation<=24;allocation++)check(-1,allocation,0,0);
    check(-1,0,1,0);check(-1,0,2,0);
    /* Third read follows both fake launches; fifth is inside compare(), so
     * cleanup must release host comparison scratch and all four plans too. */
    check(-1,0,0,3);check(-1,0,0,5);
    puts("HCUP borrowed probe: 30 preflight/partial-allocation/launch/read OOM cases; no live allocations, streams or pinned memory; context restored; protocol terminator PASS");
    return 0;
}
