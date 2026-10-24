#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "dieops.h"
#define CK(x) do { CUresult r=(x); if(r){const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"%s: %s\n",#x,s);return 2;} } while(0)
static void ptr(uint32_t *p,unsigned a,CUdeviceptr v){unsigned i=2+(a-0x140)/4;p[i]=(uint32_t)v;p[i+1]=(uint32_t)(v>>32);}
static void word(uint32_t *p,unsigned a,uint32_t v){p[2+(a-0x140)/4]=v;}
int main(int argc,char **argv){
 if(argc!=2)return 2;
 CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;
 CK(cuDeviceGet(&dev,0));CK(cuCtxCreate(&ctx,0,dev));CK(cuModuleLoad(&mod,argv[1]));CK(cuModuleGetFunction(&fn,mod,"die"));
 enum{WORDS=16384};uint32_t *source=malloc(WORDS*4),*got=malloc(WORDS*4);
 for(unsigned i=0;i<WORDS;i++)source[i]=0x12345678u^(i*2654435761u);
 CUdeviceptr input,output[6],table,control,program;
 CK(cuMemAlloc(&input,WORDS*4));CK(cuMemcpyHtoD(input,source,WORDS*4));
 for(int d=0;d<6;d++)CK(cuMemAlloc(&output[d],WORDS*4));
 CK(cuMemAlloc(&table,sizeof output));CK(cuMemcpyHtoD(table,output,sizeof output));
 CK(cuMemAlloc(&control,8192));CK(cuMemAlloc(&program,128));
 const unsigned counts[]={0,1,37,3072,12288};unsigned cases=0;
 for(int target=-1;target<6;target++)for(unsigned z=0;z<sizeof counts/sizeof counts[0];z++)for(int rep=0;rep<2;rep++){
  unsigned count=counts[z];uint32_t p[32]={0};p[0]=OP_XSEND;p[1]=16;
  ptr(p,0x140,table);ptr(p,0x148,input);word(p,0x150,count*4);word(p,0x154,256);word(p,0x158,0);
  word(p,0x15c,target<0?6u:(uint32_t)(-target-1));
  CK(cuMemcpyHtoD(program,p,sizeof p));CK(cuMemsetD8(control,0,8192));
  for(int d=0;d<6;d++)CK(cuMemsetD32(output[d],0xdeadbeef,WORDS));
  unsigned phases=1,tokens=1;void *args[]={&program,&phases,&control,&tokens};
  CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,0));CK(cuCtxSynchronize());
  for(int d=0;d<6;d++){
   CK(cuMemcpyDtoH(got,output[d],WORDS*4));
   for(unsigned i=0;i<WORDS;i++){
    uint32_t want=(target<0||target==d)&&i>=64&&i<64+count?source[i-64]:0xdeadbeef;
    if(got[i]!=want){fprintf(stderr,"target=%d words=%u rep=%d die=%d index=%u got=%x want=%x\n",target,count,rep,d,i,got[i],want);return 1;}
   }
  }
  cases++;
 }
 printf("targeted/full xsend: %u cases exact, including untouched destinations\n",cases);
 CK(cuCtxDestroy(ctx));free(source);free(got);return 0;
}
