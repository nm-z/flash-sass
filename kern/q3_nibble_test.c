/* Isolated Q3 nibble arithmetic. --cpu-only makes no CUDA calls. The default
 * GPU run checks integer dot0/dot1; --gemv also compares both scalar GEMV images.
 * GPU execution belongs in gpuq and is launched only by the parent. */
#include "quant.h"
#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do{CUresult r=(x);if(r){const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2);}}while(0)
struct sample {uint32_t code[2],high,pad;int16_t x[32];};
_Static_assert(sizeof(struct sample)==80,"integer fixture ABI");
static struct sample *samples;
static size_t count,capacity;
static uint32_t seed=4271;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static uint32_t word(const void *p){uint32_t v;memcpy(&v,p,4);return v;}
static uint16_t half(const void *p){uint16_t v;memcpy(&v,p,2);return v;}
static void pack(const struct sample *s,uint32_t p[4]){
 for(unsigned g=0;g<2;g++)for(unsigned q=0;q<2;q++){
  uint32_t c=(s->code[g]>>(8*q))&0x00ff00ffu;
  c=(c|(c<<4))&0x0f0f0f0fu;c=(c|(c<<2))&0x33333333u;
  uint32_t hi=(s->high>>(16*g))&65535u;
  uint32_t h=((hi&255u)|((hi>>8)<<16))>>(4*q);h&=0x000f000fu;
  h=(h|(h<<6))&0x03030303u;h=(h|(h<<3))&0x11111111u;
  p[2*g+q]=c|((h<<2)^0x44444444u);
 }
}
static void oracle(const struct sample *s,int32_t out[2]){
 out[0]=out[1]=0;
 for(unsigned g=0;g<2;g++)for(unsigned e=0;e<16;e++){
  int code=(s->code[g]>>(16*(e>>3)+2*(e&7)))&3;
  int hi=(s->high>>(16*g+e))&1;
  out[g]+=(code-4*hi)*s->x[16*g+e];
 }
}
static void packed_dot(const struct sample *s,int32_t out[2]){
 uint32_t p[4];pack(s,p);
 for(unsigned g=0;g<2;g++){
  int32_t bias=0;for(unsigned e=0;e<16;e++)bias-=4*s->x[16*g+e];
  int64_t dot=0;
  for(unsigned q=0;q<2;q++)for(unsigned shift=0;shift<4;shift++){
   int64_t acc=q==0&&shift==0?bias:0;
   for(unsigned h=0;h<2;h++){
    unsigned c=(p[2*g+q]>>(16*h))&(7u<<(4*shift));
    int32_t product=(int32_t)c*s->x[16*g+4*q+8*h+shift];
    assert(product>=-3641344 && product<=3641344);acc+=product;
   }
   assert(acc>=INT32_MIN && acc<=INT32_MAX);
   int32_t divisor=1<<(4*shift);assert(acc%divisor==0);dot+=acc/divisor;
  }
  assert(dot>=-8128 && dot<=8128);out[g]=(int32_t)dot;
 }
}
static void check_sample(const struct sample *s,int keep){
 int32_t expected[2],actual[2];oracle(s,expected);packed_dot(s,actual);
 assert(!memcmp(expected,actual,sizeof expected));
 uint32_t p[4];pack(s,p);
 for(unsigned g=0;g<2;g++)for(unsigned e=0;e<16;e++){
  int actual_q=(int)((p[2*g+((e>>2)&1)]>>(16*(e>>3)+4*(e&3)))&7)-4;
  int expected_q=(int)((s->code[g]>>(16*(e>>3)+2*(e&7)))&3)-4*(int)((s->high>>(16*g+e))&1);
  assert(actual_q==expected_q);
 }
 for(int a=-32;a<=31;a+=63)for(int b=-32;b<=31;b+=63){
  int64_t scaled=(int64_t)a*expected[0]+(int64_t)b*expected[1];
  assert(scaled>=-520192 && scaled<=520192);
 }
 if(keep){
  if(count==capacity){capacity=capacity?2*capacity:4096;samples=realloc(samples,capacity*sizeof *samples);assert(samples);}
  samples[count++]=*s;
 }
}
static uint8_t *tile(unsigned K,unsigned rows){
 unsigned B=K/256;size_t bytes=(size_t)rows*B*110;
 uint8_t *raw=malloc(bytes),*out=malloc((size_t)rows*q3_k_tile_stride(K));assert(raw&&out);
 for(size_t i=0;i<bytes;i++)raw[i]=(uint8_t)(random_word()>>24);
 const uint16_t scales[]={0x0001,0x03ff,0x0400,0x1400,0x2800,0x3c00,0x8001,0x9400,0xac00};
 for(unsigned r=0;r<rows;r++)for(unsigned b=0;b<B;b++){uint16_t h=scales[random_word()%9];memcpy(raw+((size_t)r*B+b)*110+108,&h,2);}
 repack_q3_k(raw,rows,K,out);
 unsigned Q=q4_k_lanes_per_row(K)/8,stride=q3_k_tile_stride(K);
 /* Verify the repacker's actual byte/bit layout against the original block. */
 for(unsigned r=0;r<rows&&r<2;r++)for(unsigned b=0;b<B;b++)for(unsigned g=0;g<16;g++){
  const uint8_t *src=raw+((size_t)r*B+b)*110,*rec=out+(size_t)r*stride+16*Q+112*b;
  uint32_t c=word(rec+4*g);unsigned hi=half(rec+64+2*g),nn=g>>3,j=(g>>1)&3,which=g&1;
  for(unsigned e=0;e<16;e++){
   unsigned raw_c=(src[32+32*nn+16*which+e]>>(2*j))&3,raw_hi=((src[16*which+e]>>(4*nn+j))&1)^1;
   assert(((c>>(16*(e>>3)+2*(e&7)))&3)==raw_c);assert(((hi>>e)&1)==raw_hi);
  }
 }
 free(raw);return out;
}
static void cpu_checks(void){
 struct sample s={0};
 for(unsigned g=0;g<2;g++)for(unsigned q=0;q<4;q++)for(unsigned c=0;c<256;c++)for(unsigned h=0;h<16;h++){
  memset(&s,0,sizeof s);s.code[g]=c<<(8*q);s.high=h<<(16*g+4*q);
  for(unsigned e=0;e<32;e++)s.x[e]=(int16_t)((int)(random_word()%255)-127);
  check_sample(&s,1);
 }
 for(unsigned g=0;g<2;g++)for(unsigned e=0;e<16;e++)for(int value=-4;value<=3;value++)for(int a=-127;a<=127;a++){
  memset(&s,0,sizeof s);s.code[g]=(uint32_t)(value&3)<<(16*(e>>3)+2*(e&7));
  s.high=(uint32_t)(value<0)<<(16*g+e);s.x[16*g+e]=(int16_t)a;check_sample(&s,1);
 }
 for(unsigned i=0;i<65536;i++){
  s.code[0]=random_word();s.code[1]=random_word();s.high=random_word();
  for(unsigned e=0;e<32;e++)s.x[e]=i%3==0?127:i%3==1?-127:(int16_t)((int)(random_word()%255)-127);
  check_sample(&s,1);
 }
 for(unsigned i=0;i<3;i++){
  memset(&s,0,sizeof s);s.code[0]=s.code[1]=i==2?UINT32_MAX:0;s.high=i==2?0:UINT32_MAX;
  for(unsigned e=0;e<32;e++)s.x[e]=i==1?-127:127;
  check_sample(&s,1);
 }
 for(unsigned K=2560;K<=6144;K+=3584){
  unsigned L=q4_k_lanes_per_row(K),Q=L/8,n=(K/256)/Q,stride=q3_k_tile_stride(K);
  uint8_t *w=tile(K,3);int8_t *x=malloc(K);float *as=malloc(4*(K/32)),expected[3],actual[3];assert(x&&as);
  for(unsigned j=0;j<K;j++)x[j]=(int8_t)((int)(random_word()%255)-127);
  for(unsigned j=0;j<K/32;j++)as[j]=ldexpf((float)(j%7+1),(int)(j%25)-20);
  gemv_q3_k_ref(w,3,K,x,as,expected);
  for(unsigned r=0;r<3;r++){
   float v[32],next[32];
   for(unsigned lane=0;lane<L;lane++){
    unsigned q=lane>>3,jj=lane&7;float acc=0;
    for(unsigned i=0;i<n;i++){
     unsigned b=Q*i+q;const uint8_t *rec=w+(size_t)r*stride+16*Q+112*b;
     s.code[0]=word(rec+8*jj);s.code[1]=word(rec+8*jj+4);s.high=word(rec+64+4*jj);
     for(unsigned e=0;e<32;e++)s.x[e]=x[256*b+32*jj+e];
     int32_t dots[2];packed_dot(&s,dots);
     int32_t sumi=(int8_t)rec[96+2*jj]*dots[0]+(int8_t)rec[97+2*jj]*dots[1];
     assert(sumi>=-520192&&sumi<=520192);
     float u=fp16_to_fp32(half(w+(size_t)r*stride+16*q+2*i))*(float)sumi;acc=fmaf(as[8*b+jj],u,acc);
    }
    v[lane]=acc;
   }
   for(unsigned offset=L/2;offset;offset>>=1){for(unsigned lane=0;lane<L;lane++)next[lane]=v[lane]+v[lane^offset];memcpy(v,next,4*L);}
   actual[r]=v[0];
  }
  assert(!memcmp(expected,actual,sizeof expected));free(w);free(x);free(as);
 }
 printf("Q3 nibble CPU PASS: %zu direct cases, exhaustive code/high quartets and all-field basis, signed32 bounds, actual tile layout, K2560/K6144 FP parity\n",count);
}
static CUdeviceptr upload(const void *p,size_t n){CUdeviceptr d;CK(cuMemAlloc(&d,n));CK(cuMemcpyHtoD(d,p,n));return d;}
static int integer_gpu(void){
 const char *path=getenv("Q3_NIBBLE_CUBIN");CUmodule module;CUfunction fn;
 CK(cuModuleLoad(&module,path?path:"q3_nibble_dot.cubin"));CK(cuModuleGetFunction(&fn,module,"q3_nibble_dot"));
 size_t guard=64,bytes=guard+8*count+guard;uint8_t *initial=malloc(bytes),*actual=malloc(bytes);assert(initial&&actual);memset(initial,0xa5,bytes);
 CUdeviceptr input=upload(samples,count*sizeof *samples),output=upload(initial,bytes),start=output+guard;unsigned n=(unsigned)count;
 void *args[]={&input,&start,&n};CK(cuLaunchKernel(fn,(n+255)/256,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());CK(cuMemcpyDtoH(actual,output,bytes));
 unsigned bad=0;for(size_t i=0;i<count;i++){int32_t expected[2];oracle(samples+i,expected);if(memcmp(expected,actual+guard+8*i,8)){if(bad<3)printf("integer case%zu expected=%d,%d actual=%d,%d\n",i,expected[0],expected[1],(int32_t)word(actual+guard+8*i),(int32_t)word(actual+guard+8*i+4));bad++;}}
 for(size_t i=0;i<guard;i++)bad+=actual[i]!=0xa5||actual[guard+8*count+i]!=0xa5;
 void *copy=malloc(count*sizeof *samples);assert(copy);CK(cuMemcpyDtoH(copy,input,count*sizeof *samples));bad+=!!memcmp(copy,samples,count*sizeof *samples);free(copy);
 printf("Q3 nibble integer GPU cases=%zu mismatches_or_guards=%u %s\n",count,bad,bad?"FAIL":"BIT-EXACT");
 CK(cuMemFree(input));CK(cuMemFree(output));CK(cuModuleUnload(module));free(initial);free(actual);return !!bad;
}
static int gemv_gpu(unsigned K,unsigned rows,unsigned grid,unsigned pattern){
 unsigned L=q4_k_lanes_per_row(K),matrix_rows=rows+grid*(256/L),guard=32;
 uint8_t *w=tile(K,matrix_rows);int16_t *x=malloc(2*K);int8_t *x8=malloc(K);float *as=malloc(4*K/32),*input=malloc(4*K);
 int32_t *s32=malloc(4*K/32),*s16=malloc(4*K/16);size_t yc=rows+grid*(256/L)+2*guard;
 float *initial=malloc(4*yc),*expected=malloc(4*yc),*actual=malloc(4*yc),*baseline=malloc(4*yc);assert(x&&x8&&as&&input&&s32&&s16&&initial&&expected&&actual&&baseline);
 for(unsigned j=0;j<K;j++)input[j]=pattern==0?127.f:pattern==1?-127.f:ldexpf((float)((int)(random_word()%65535)-32767),(int)(j/32%21)-10);
 quant_act_gpu32(input,K,x,s32,s16,as);for(unsigned j=0;j<K;j++){assert(x[j]>=-127&&x[j]<=127);x8[j]=(int8_t)x[j];}
 for(size_t j=0;j<yc;j++){uint32_t bits=0x7fc01234;memcpy(initial+j,&bits,4);}memcpy(expected,initial,4*yc);gemv_q3_k_ref(w,rows,K,x8,as,expected+guard);
 CUdeviceptr dw=upload(w,(size_t)matrix_rows*q3_k_tile_stride(K)),dx=upload(x,2*K),da=upload(as,4*K/32),dy=upload(initial,4*yc),out=dy+4*guard,zero=0;float xs=1.f;
 const char *formats[]={"gemv_q3_k_k%u.cubin","gemv_q3_k_nibble_k%u.cubin"};double ms[2]={0};
 for(unsigned variant=0;variant<2;variant++){
  char path[128];snprintf(path,sizeof path,formats[variant],K);CUmodule module;CUfunction fn;CK(cuModuleLoad(&module,path));CK(cuModuleGetFunction(&fn,module,"gemv_q3_k"));
  void *args[]={&dw,&dx,&zero,&out,&rows,&xs,&zero,&da};CK(cuMemcpyHtoD(dy,initial,4*yc));
  CK(cuLaunchKernel(fn,grid,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());CK(cuMemcpyHtoD(dy,initial,4*yc));
  CUevent begin,end;CK(cuEventCreate(&begin,0));CK(cuEventCreate(&end,0));CK(cuEventRecord(begin,0));CK(cuLaunchKernel(fn,grid,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(end,0));CK(cuEventSynchronize(end));float elapsed;CK(cuEventElapsedTime(&elapsed,begin,end));ms[variant]=elapsed;
  CK(cuMemcpyDtoH(variant?actual:baseline,dy,4*yc));CK(cuEventDestroy(begin));CK(cuEventDestroy(end));CK(cuModuleUnload(module));
 }
 unsigned bad=0;for(size_t j=0;j<yc;j++)if(word(actual+j)!=word(expected+j)||word(actual+j)!=word(baseline+j)){if(bad<3)printf("gemv K%u word%zu cpu=%08x baseline=%08x nibble=%08x\n",K,j,word(expected+j),word(baseline+j),word(actual+j));bad++;}
 printf("Q3 nibble GEMV K=%u rows=%u grid=%u pattern=%u mismatches=%u baseline_ms=%.6f candidate_ms=%.6f %s\n",K,rows,grid,pattern,bad,ms[0],ms[1],bad?"FAIL":"BIT-EXACT");
 CK(cuMemFree(dw));CK(cuMemFree(dx));CK(cuMemFree(da));CK(cuMemFree(dy));free(w);free(x);free(x8);free(as);free(input);free(s32);free(s16);free(initial);free(expected);free(actual);free(baseline);return !!bad;
}
int main(int argc,char **argv){
 cpu_checks();if(argc==2&&!strcmp(argv[1],"--cpu-only")){free(samples);return 0;}
 if(argc>2||(argc==2&&strcmp(argv[1],"--gemv")))return 2;
 CK(cuInit(0));CUdevice dev;CUcontext ctx;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));int failed=integer_gpu();
 if(argc==2)for(unsigned K=2560;K<=6144;K+=3584){failed+=gemv_gpu(K,1,1,0);failed+=gemv_gpu(K,137,16,1);failed+=gemv_gpu(K,4096,16,2);failed+=gemv_gpu(K,4096,33,2);}
 CK(cuCtxDestroy(ctx));free(samples);return failed?3:0;
}
