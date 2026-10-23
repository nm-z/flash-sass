/* Diagnose the owned-core startup failure without loading model state.
 * Uses flashcore's context API/flags; destroys only the context it creates. */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
int main(int argc, char **argv) {
    if(argc!=2)return 2;
    int ordinal=atoi(argv[1]);CUdevice device;CUcontext context=NULL;
    CUresult rc=cuInit(0);
    if(rc==CUDA_SUCCESS)rc=cuDeviceGet(&device,ordinal);
    if(rc==CUDA_SUCCESS)rc=cuCtxCreate_v2(&context,CU_CTX_SCHED_SPIN,device);
    const char *name="unknown",*message="unknown";
    cuGetErrorName(rc,&name);cuGetErrorString(rc,&message);
    printf("ordinal=%d cuCtxCreate_v2 result=%d name=%s message=%s\n",ordinal,(int)rc,name,message);
    if(rc==CUDA_SUCCESS) {
        size_t free_bytes=0,total_bytes=0;
        CUresult memory=cuMemGetInfo(&free_bytes,&total_bytes);
        printf("cuMemGetInfo result=%d free_bytes=%zu total_bytes=%zu\n",(int)memory,free_bytes,total_bytes);
        cuCtxDestroy(context);
    }
    return rc==CUDA_SUCCESS?0:1;
}
