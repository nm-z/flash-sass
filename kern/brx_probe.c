#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(call) do { CUresult e=(call); if(e) {const char *s="?";cuGetErrorString(e,&s);fprintf(stderr,"%s: %s\n",#call,s);return 2;} } while(0)
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    CUcontext context;CUdevice device;CUmodule module;CUfunction fn;CUdeviceptr output;
    CK(cuInit(0));CK(cuDeviceGet(&device,0));CK(cuCtxCreate_v2(&context,0,device));
    CK(cuModuleLoad(&module,argv[1]));CK(cuModuleGetFunction(&fn,module,"brx_probe"));CK(cuMemAlloc(&output,4));
    const uint32_t indexes[]={0,1,2,3,4,77,UINT32_MAX};
    const uint32_t values[]={111,29,721,503};
    for(unsigned k=0;k<sizeof indexes/sizeof *indexes;k++) {
        uint32_t index=indexes[k],got=0;void *params[]={&output,&index};
        CK(cuLaunchKernel(fn,1,1,1,1,1,1,0,0,params,0));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(&got,output,4));
        uint32_t want=index<4?values[index]:UINT32_MAX;
        printf("index=%u got=%u expected=%u\n",index,got,want);if(got!=want)return 1;
    }
    CK(cuMemFree(output));CK(cuModuleUnload(module));CK(cuCtxDestroy(context));puts("PASS");return 0;
}
