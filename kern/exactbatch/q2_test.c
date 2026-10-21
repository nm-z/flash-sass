/* Imported B8 Q2 fixture, including independent C reference. */
#include "prog_group.h"
/* Isolated B8 Q2 parity: scalar GPU, batch GPU, and the independent C twin.
 * --cpu-only exercises descriptor eligibility; GPU execution belongs in gpuq. */
#include "prog.h"
#include "quant.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#undef cuEventElapsedTime
CUresult cuEventElapsedTime(float *,CUevent,CUevent);
#define CK(x) do{CUresult r=(x);if(r){const char *s="?";cuGetErrorString(r,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2);}}while(0)
static uint32_t seed=173;
static unsigned timing_repeats=1;
static int production_inputs;
static int packing_probe;
static uint32_t random_word(void){seed=1664525u*seed+1013904223u;return seed;}
static void pointer(phase_t *p,int addr,CUdeviceptr v){ph_ptr(p,addr,v);}
struct variant{const char *name;unsigned scalar,batch;int k,moe,acc;};
static const struct variant variants[]={
 {"dense2560",OP_Q2K2560,OP_Q2K25608,2560,0,0},
 {"dense640",OP_Q2K640,OP_Q2K6408,640,0,0},
 {"acc640",OP_Q2ACC640,OP_Q2ACC6408,640,0,1},
 {"gather",OP_Q2MOEGN,OP_Q2MOEGN8,2560,1,0},
 {"down",OP_Q2MOEDN,OP_Q2MOEDN8,640,2,0}};
static uint32_t prmt(uint32_t a,unsigned selector,uint32_t b){
 uint32_t out=0;
 for(unsigned i=0;i<4;i++){
  unsigned digit=(selector>>(4*i))&15,source=digit&7;
  unsigned byte=((source<4?a:b)>>(8*(source&3)))&255;
  if(digit&8)byte=(byte&128)?255:0;
  out|=byte<<(8*i);
 }
 return out;
}
static void packing_checks(void){
 /* Every pair of signed bytes appears in low/high positions. The GPU basis
  * cases below independently observe each activation after the same packing. */
 for(unsigned pair=0;pair<65536;pair++){
  int16_t input[4]={(int8_t)pair,(int8_t)(pair>>8),(int8_t)~pair,(int8_t)~(pair>>8)};
  uint32_t a,b;memcpy(&a,input,4);memcpy(&b,input+2,4);
  uint32_t packed=prmt(a,0x6420,b),lo=prmt(packed,0x9180,0),hi=prmt(packed,0xb3a2,0);
  assert(lo==a && hi==b);
 }
 float input[2560],as[80];int16_t q[2560];int32_t sum32[80],sum16[160];
 for(int power=-120;power<=120;power+=8){
  float magnitude=ldexpf(1.f,power);
  for(unsigned i=0;i<2560;i++)input[i]=magnitude*((int)(i%255)-127)/127.f;
  quant_act_gpu32(input,2560,q,sum32,sum16,as);
  for(unsigned i=0;i<2560;i++)assert(q[i]>=-127 && q[i]<=127);
 }
 puts("Q2 signed-byte PRMT roundtrip and quantized producer range PASS");
}
static void plan_checks(void){
 packing_checks();
 for(unsigned t=0;t<256;t++)assert(((t*0xcccdu)>>19)==t/10);
 for(unsigned v=0;v<5;v++){
  prog_t p[8],g;const prog_t *inputs[8];prog_init(&g);
  for(unsigned j=0;j<8;j++){
   prog_init(p+j);p[j].loop=1;p[j].token_offset=j;inputs[j]=p+j;
   CUdeviceptr base=0x200000000ull+0x100000ull*j;
   ph_q2(p+j,variants[v].scalar,0x100000000ull,base,base+0x10000,base+0x30000,
     variants[v].moe==1?640:variants[v].moe==2?2560:137,
     variants[v].moe?base+0x20000:0,base+0x40000,
     (variants[v].moe==2||variants[v].acc)?base+0x50000:0,
     variants[v].moe?base+0x60000:0);
  }
  assert(prog_group(&g,inputs,8)==0);assert(prog_batch_fuse_q2(&g,1)==1);
  assert(g.p[0].w[0]==variants[v].batch && (g.p[0].w[30]&16u));
  assert(prog_group(&g,inputs,8)==0);pointer(g.p+1,0x158,((uint64_t)g.p[0].w[9]<<32)|g.p[0].w[8]);
  assert(prog_batch_fuse_q2(&g,1)==0);
  assert(prog_group(&g,inputs,8)==0);g.p[2].w[31]=1;assert(prog_batch_fuse_q2(&g,1)==0);
  if(variants[v].moe){assert(prog_group(&g,inputs,8)==0);assert(prog_batch_fuse_q2(&g,0)==0);}
  prog_free(&g);for(unsigned j=0;j<8;j++)prog_free(p+j);
 }
 for(unsigned width=16;width<=32;width*=2){
  prog_t g;prog_init(&g);
  for(unsigned p=0;p<width;p++){
   CUdeviceptr base=0x200000000ull+0x100000ull*p;
   ph_q2(&g,OP_Q2K2560,0x100000000ull,base,base+0x10000,base+0x30000,137,0,base+0x40000,0,0);
   g.p[p].w[30]=3;
  }
  g.p[0].w[30]|=PHASE_GROUP_FLAG|(width<<PHASE_WIDTH_SHIFT);
  assert(prog_batch_fuse_q2(&g,1)==(int)(width/8));
  for(unsigned p=0;p<width;p+=8){
   assert(g.p[p].w[0]==OP_Q2K25608 && phase_positions(g.p+p)==8);
   assert(!!(g.p[p].w[30]&2u)==(p==0));
   assert(!!(g.p[p].w[30]&1u)==(p+8==width));
   assert(!!(g.p[p].w[30]&4u)==(p+8<width));
  }
  prog_free(&g);
 }
 puts("Q2 B8/B16/B32 descriptor eligibility and publication flags PASS");
}
static CUdeviceptr upload(const void *host,size_t bytes){CUdeviceptr p;CK(cuMemAlloc(&p,bytes));CK(cuMemcpyHtoD(p,host,bytes));return p;}
static float launch(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,phase_t d[8],int batch){
 uint32_t control[16]={0};control[9]=batch?8:0;unsigned count=8,end=batch?8:1;
 CK(cuMemsetD8(ctl,0,8192));CK(cuMemcpyHtoD(ctl,control,sizeof control));CK(cuMemcpyHtoD(program,d,8*sizeof *d));
 CUevent begin,finish;float ms=0;CK(cuEventCreate(&begin,0));CK(cuEventCreate(&finish,0));
 void *args[]={&program,&count,&ctl,&end};CK(cuEventRecord(begin,0));
 CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuEventRecord(finish,0));
 CK(cuEventSynchronize(finish));CK(cuEventElapsedTime(&ms,begin,finish));
 CK(cuEventDestroy(finish));CK(cuEventDestroy(begin));return ms;
}
static int check(CUfunction fn,CUdeviceptr program,CUdeviceptr ctl,const struct variant *v,unsigned rows,unsigned units,unsigned incidence){
 unsigned K=v->k,half=K/32,nvec=v->moe==2?10:1,experts=v->moe?(production_inputs?128:17):1,guard=32;
 unsigned matrix_rows=v->moe?rows:rows+units*(K==2560?16:64);
 unsigned output_rows=rows*(v->moe==1?10:1),ystep=output_rows+units*64+2*guard;
 size_t stride=q2_0_tile_stride(K),wbytes=(size_t)experts*matrix_rows*stride,xcount=(size_t)8*nvec*K,scount=(size_t)8*nvec*half,ycount=(size_t)8*ystep;
 uint8_t *w=malloc(wbytes);int8_t *x8=malloc(xcount);int16_t *x=malloc(2*xcount);int32_t *s=calloc(scount,4);float *as=malloc(4*scount),*initial=malloc(4*ycount),*oracle=malloc(4*ycount),*reference=malloc(4*ycount),*actual=malloc(4*ycount),*dot=malloc(4*rows);
 uint32_t ids[8][10],ne[8];float weight[8][10];
 for(size_t i=0;i<wbytes;i++)w[i]=(uint8_t)(random_word()>>24);
 unsigned pad=(unsigned)stride-16*(K/64);
 for(unsigned e=0;e<experts;e++)for(unsigned r=0;r<matrix_rows;r++)for(unsigned b=0;b<K/64;b++){
  static const uint16_t range[]={0x0001,0x03ff,0x0400,0x1400,0x2800,0x3c00,0x8001,0x9400,0xac00};
  uint32_t z=random_word();uint16_t h=production_inputs?range[z%9]:(uint16_t)(0x2800u+(z&1023u));memcpy(w+((size_t)e*matrix_rows+r)*stride+2*b,&h,2);
 }
 (void)pad;
 for(size_t i=0;i<xcount;i++){x8[i]=(int8_t)((int)(random_word()%255)-127);x[i]=x8[i];s[i/32]+=x8[i];}
 for(size_t i=0;i<scount;i++)as[i]=.001f*(1+(random_word()%23));
 if(production_inputs){
  static const float magnitude[]={0x1p-40f,0x1p-20f,0x1p-8f,1.f,0x1p8f,0x1p20f};
  float input[2560];int32_t sum16[160];
  for(unsigned vector=0;vector<8*nvec;vector++){
   for(unsigned j=0;j<K;j++){
    unsigned block=j/32,lane=j%32,pattern=(block+vector)%6;
    float m=magnitude[(block/6+vector)%6];
    input[j]=pattern==0?0.f:pattern==1?m:pattern==2?-m:
     pattern==3?(lane&1?-m:m):pattern==4?(lane==0?m:-m/128.f):
     m*((int)(random_word()&65535)-32768)/32768.f;
   }
   quant_act_gpu32(input,K,x+(size_t)vector*K,s+(size_t)vector*half,sum16,as+(size_t)vector*half);
  }
  for(size_t i=0;i<xcount;i++)x8[i]=(int8_t)x[i];
 }
 if(packing_probe){
  assert(!v->moe && !v->acc);
  memset(s,0,4*scount);
  for(size_t i=0;i<xcount;i++){x8[i]=(int8_t)((73*i+37*(i/K))&255);x[i]=x8[i];s[i/32]+=x8[i];}
  for(size_t i=0;i<scount;i++)as[i]=1.f;
  /* All codes are one (zero effective weight), except code two selects one
   * input per output row. Unit scales make the independent oracle x[row]. */
  for(unsigned r=0;r<matrix_rows;r++){
   uint8_t *row=w+(size_t)r*stride;uint16_t one=0x3c00;
   memset(row,0,stride);memset(row+pad,0x55,16*(K/64));
   for(unsigned b=0;b<K/64;b++)memcpy(row+2*b,&one,2);
   unsigned column=r%K,at=pad+column/4,shift=2*(column&3);
   row[at]=(uint8_t)((row[at]&~(3u<<shift))|(2u<<shift));
  }
 }else{
  for(size_t i=0;i<xcount;i++)assert(x[i]>=-127 && x[i]<=127);
 }
 for(unsigned p=0;p<8;p++){
  const unsigned counts[8]={0,1,10,3,0,7,10,2};ne[p]=incidence==1?0:incidence==2?10:incidence==3?1:counts[p];
  for(unsigned r=0;r<10;r++){ids[p][r]=((production_inputs?83:0)+(incidence==3?p:3*p)+5*r)%experts;weight[p][r]=(r&1?-.125f:.125f)*(1+(random_word()%11));}
  if(incidence>=4 && incidence<12){
   const unsigned masks[]={0x40,0x82,0xa1,0xa5,0xb5,0xbd,0xfd,0xff};
   ne[p]=(masks[incidence-4]>>p)&1;ids[p][0]=127;
  }else if(incidence==12){
   ne[p]=10;for(unsigned r=0;r<10;r++)ids[p][r]=118+(r+3*p)%10;
  }
 }
 for(size_t i=0;i<ycount;i++){uint32_t z=0x7fc01234u;memcpy(initial+i,&z,4);}
 if(v->acc||v->moe==2)for(unsigned p=0;p<8;p++)for(unsigned r=0;r<output_rows;r++)initial[p*ystep+guard+r]=.25f;
 memcpy(oracle,initial,4*ycount);
 for(unsigned p=0;p<8;p++){
  unsigned count=v->moe?ne[p]:1;
  for(unsigned rank=0;rank<count;rank++){
   unsigned e=v->moe?ids[p][rank]:0,vector=p*nvec+(v->moe==2?rank:0);
   if(packing_probe)for(unsigned r=0;r<rows;r++)dot[r]=(float)x8[(size_t)vector*K+r%K];
   else gemv_q2_0_ref(w+(size_t)e*matrix_rows*stride,rows,K,x8+(size_t)vector*K,s+(size_t)vector*half,as+(size_t)vector*half,dot);
   for(unsigned r=0;r<rows;r++){
    float *out=oracle+p*ystep+guard+(v->moe==1?rank*rows:0)+r;
    if(v->moe==2)*out=rank?fmaf(weight[p][rank],dot[r],*out):dot[r]*weight[p][rank];
    else if(v->acc)*out=fmaf(weight[p][0],dot[r],*out);else *out=dot[r];
   }
  }
 }
 CUdeviceptr dw=upload(w,wbytes),dx=upload(x,2*xcount),ds=upload(s,4*scount),da=upload(as,4*scount),dy=upload(initial,4*ycount),di=upload(ids,sizeof ids),dn=upload(ne,sizeof ne),de=upload(weight,sizeof weight);
 phase_t scalar[8],batch[8];memset(scalar,0,sizeof scalar);
 for(unsigned p=0;p<8;p++){
  phase_t *d=scalar+p;d->w[0]=v->scalar;d->w[1]=units;
  pointer(d,0x140,dw);pointer(d,0x148,dx+(size_t)p*nvec*K*2);pointer(d,0x150,ds+(size_t)p*nvec*half*4);pointer(d,0x158,dy+4*((size_t)p*ystep+guard));ph_u32(d,0x160,rows);
  ph_f32(d,0x164,1);pointer(d,0x168,di+p*40);pointer(d,0x170,da+(size_t)p*nvec*half*4);pointer(d,0x178,de+p*40);pointer(d,0x180,dn+p*4);
 }
 memcpy(batch,scalar,sizeof batch);batch[0].w[0]=v->batch;batch[0].w[30]=24u|(8u<<8);
 /* Warm both bodies once; all timed launches restore the same output state. */
 (void)launch(fn,program,ctl,scalar,0);CK(cuMemcpyHtoD(dy,initial,4*ycount));
 (void)launch(fn,program,ctl,batch,1);CK(cuMemcpyHtoD(dy,initial,4*ycount));
 double scalar_ms=launch(fn,program,ctl,scalar,0);CK(cuMemcpyDtoH(reference,dy,4*ycount));CK(cuMemcpyHtoD(dy,initial,4*ycount));
 double batch_ms=launch(fn,program,ctl,batch,1);CK(cuMemcpyDtoH(actual,dy,4*ycount));
 for(unsigned repeat=1;repeat<timing_repeats;repeat++){
  CK(cuMemcpyHtoD(dy,initial,4*ycount));scalar_ms+=launch(fn,program,ctl,scalar,0);
  CK(cuMemcpyHtoD(dy,initial,4*ycount));batch_ms+=launch(fn,program,ctl,batch,1);
 }
 scalar_ms/=timing_repeats;batch_ms/=timing_repeats;
 unsigned gpu_bad=0,cpu_bad=0;
 for(size_t i=0;i<ycount;i++){
  uint32_t a,b,c;memcpy(&a,actual+i,4);memcpy(&b,reference+i,4);memcpy(&c,oracle+i,4);
  if(a!=b){if(gpu_bad<3)printf("%s word%zu scalar=%08x batch=%08x\n",v->name,i,b,a);gpu_bad++;}
  if(a!=c){if(cpu_bad<3)printf("%s word%zu oracle=%08x batch=%08x\n",v->name,i,c,a);cpu_bad++;}
 }
 printf("%s rows=%u units=%u incidence=%u experts=%u quantized_inputs=%d packing_probe=%d gpu_mismatches=%u oracle_mismatches=%u scalar_ms=%.6f batch_ms=%.6f speedup=%.6f repeats=%u %s\n",v->name,rows,units,incidence,experts,production_inputs,packing_probe,gpu_bad,cpu_bad,scalar_ms,batch_ms,scalar_ms/batch_ms,timing_repeats,gpu_bad||cpu_bad?"FAIL":"BIT-EXACT");
 CUdeviceptr allocated[]={dw,dx,ds,da,dy,di,dn,de};for(unsigned i=0;i<8;i++)CK(cuMemFree(allocated[i]));
 free(w);free(x8);free(x);free(s);free(as);free(initial);free(oracle);free(reference);free(actual);free(dot);
 return !!(gpu_bad||cpu_bad);
}
int main(int argc,char **argv){
 setvbuf(stdout,NULL,_IONBF,0);
 plan_checks();if(argc==2&&!strcmp(argv[1],"--cpu-only"))return 0;
 if(argc>2 || (argc==2&&strcmp(argv[1],"--production")&&strcmp(argv[1],"--packing")))return 2;
 int packing_only=argc==2&&!strcmp(argv[1],"--packing");
 production_inputs=argc==2&&!packing_only;
 const char *repeats=getenv("Q2_TIMING_REPEATS");
 if(repeats){char *end;unsigned long n=strtoul(repeats,&end,10);if(!*repeats||*end||n<1||n>20)return 2;timing_repeats=(unsigned)n;}
 CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule module;CUfunction fn;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
 const char *image=getenv("FLASH_CUBIN");CK(cuModuleLoad(&module,image?image:"die.cubin"));CK(cuModuleGetFunction(&fn,module,"die"));
 CUdeviceptr program,ctl;CK(cuMemAlloc(&program,8*sizeof(phase_t)));CK(cuMemAlloc(&ctl,8192));int failed=0;
 const unsigned rows[]={1,137,513},units[]={1,16,33};
 if(!packing_only)for(unsigned v=0;v<3;v++)for(unsigned i=0;i<3;i++)failed+=check(fn,program,ctl,variants+v,rows[i],units[i],0);
 if(production_inputs)for(unsigned v=0;v<3;v++){
  failed+=check(fn,program,ctl,variants+v,4096,16,0);
  failed+=check(fn,program,ctl,variants+v,4096,33,0);
 }
 if(!packing_only)for(unsigned v=3;v<5;v++)for(unsigned i=0;i<4;i++)failed+=check(fn,program,ctl,variants+v,v==3?640:2560,16,i);
 if(production_inputs||packing_only){packing_probe=1;for(unsigned v=0;v<2;v++)failed+=check(fn,program,ctl,variants+v,variants[v].k,33,0);}
 if(production_inputs){packing_probe=0;for(unsigned v=3;v<5;v++)for(unsigned i=4;i<13;i++)failed+=check(fn,program,ctl,variants+v,v==3?640:2560,16,i);}
 CK(cuMemFree(ctl));CK(cuMemFree(program));CK(cuModuleUnload(module));CK(cuCtxDestroy(ctx));return failed?3:0;
}
