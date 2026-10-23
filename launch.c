/* launch FILE: load a .ptx/.cubin/.fatbin with the driver API, or compile a
 * .nvvm/.ll NVVM IR file with libNVVM first; run kernel k(float*) over 256
 * threads and print a[100] (expect 200). */
#include <cuda.h>
#include <nvvm.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CK(x) do{CUresult r=(x);if(r){const char*s;cuGetErrorString(r,&s);fprintf(stderr,"%s -> %s\n",#x,s);return 1;}}while(0)
int main(int argc,char**argv){
    if(argc<2)return 2;
    const char *path=argv[1],*dot=strrchr(path,'.');
    CK(cuInit(0));CUdevice dev;CK(cuDeviceGet(&dev,0));CUcontext ctx;CK(cuCtxCreate(&ctx,0,dev));
    char name[128];cuDeviceGetName(name,sizeof name,dev);int maj,min;
    cuDeviceGetAttribute(&maj,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,dev);cuDeviceGetAttribute(&min,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,dev);
    CUmodule mod;
    if(dot&&(!strcmp(dot,".nvvm")||!strcmp(dot,".ll"))){
        FILE*f=fopen(path,"rb");if(!f)return 1;fseek(f,0,SEEK_END);long n=ftell(f);rewind(f);char*src=malloc(n+1);fread(src,1,n,f);src[n]=0;fclose(f);
        nvvmProgram p;nvvmCreateProgram(&p);nvvmAddModuleToProgram(p,src,n,path);
        const char*opts[]={"-arch=compute_52"};nvvmResult r=nvvmCompileProgram(p,1,opts);
        if(r){size_t ln;nvvmGetProgramLogSize(p,&ln);char*log=malloc(ln);nvvmGetProgramLog(p,log);fprintf(stderr,"nvvm: %s\n%s\n",nvvmGetErrorString(r),log);return 1;}
        size_t pn;nvvmGetCompiledResultSize(p,&pn);char*ptx=malloc(pn);nvvmGetCompiledResult(p,ptx);
        CK(cuModuleLoadData(&mod,ptx));
    } else CK(cuModuleLoad(&mod,path));
    CUfunction k;CK(cuModuleGetFunction(&k,mod,"k"));CUdeviceptr a;CK(cuMemAlloc(&a,1024));
    void*args[]={&a};CK(cuLaunchKernel(k,1,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
    float h[256];CK(cuMemcpyDtoH(h,a,sizeof h));
    printf("%s on %s (sm_%d%d): a[100]=%g\n",path,name,maj,min,h[100]);return h[100]==200.f?0:1;
}
