#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
extern CUresult add_kernel(CUgraphNode *,CUgraph,const CUgraphNode *,size_t,const CUDA_KERNEL_NODE_PARAMS_v1 *) __asm__("cuGraphAddKernelNode");
extern CUresult instantiate(CUgraphExec *,CUgraph,CUgraphNode *,char *,size_t) __asm__("cuGraphInstantiate");
#define CK(x) do { CUresult r=(x); if(r){const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"%s: %s\n",#x,s);return 1;} } while(0)
int main(int argc,char **argv) {
    if(argc!=3 && argc!=4)return 2;
    CUdevice device;CUcontext context;CUmodule module;CUfunction function;
    CK(cuInit(0));CK(cuDeviceGet(&device,0));CK(cuCtxCreate_v2(&context,0,device));
    CK(cuModuleLoad(&module,argv[1]));CK(cuModuleGetFunction(&function,module,argv[2]));
    if(argc==4) {
        size_t stack,before,after,total;CK(cuCtxGetLimit(&stack,CU_LIMIT_STACK_SIZE));CK(cuMemGetInfo(&before,&total));
        CK(cuCtxSetLimit(CU_LIMIT_STACK_SIZE,strtoul(argv[3],NULL,10)));CK(cuCtxGetLimit(&stack,CU_LIMIT_STACK_SIZE));
        CK(cuMemGetInfo(&after,&total));printf("stack=%zu free_before=%zu free_after=%zu\n",stack,before,after);
    }
    const unsigned counts[]={1,16,128,512,2048,8192};
    for(unsigned j=0;j<sizeof counts/sizeof counts[0];j++) {
        CUgraph graph;CUgraphExec executable=NULL;CUgraphNode prior=NULL;
        CK(cuGraphCreate(&graph,0));
        uint64_t args[16]={0};size_t size=sizeof args;
        void *extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,args,CU_LAUNCH_PARAM_BUFFER_SIZE,&size,CU_LAUNCH_PARAM_END};
        CUDA_KERNEL_NODE_PARAMS_v1 k={.func=function,.gridDimX=16,.gridDimY=1,.gridDimZ=1,
            .blockDimX=256,.blockDimY=1,.blockDimZ=1,.extra=extra};
        for(unsigned i=0;i<counts[j];i++){CUgraphNode next;CK(add_kernel(&next,graph,i?&prior:NULL,i?1:0,&k));prior=next;}
        size_t before,after,total;CK(cuMemGetInfo(&before,&total));
        char log[1024]={0};CUgraphNode bad;CUresult result=instantiate(&executable,graph,&bad,log,sizeof log);
        CK(cuMemGetInfo(&after,&total));const char *name="?";cuGetErrorString(result,&name);
        printf("nodes=%u result=%d name=%s free_before=%zu free_after=%zu log=%s\n",counts[j],result,name,before,after,log);fflush(stdout);
        if(executable)CK(cuGraphExecDestroy(executable));CK(cuGraphDestroy(graph));
        if(result)break;
    }
    CK(cuCtxDestroy(context));return 0;
}
