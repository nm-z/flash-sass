#include "qsa.h"
#include "fmath.h"
#include "quant_range_cases.h"
#include <assert.h>
#include <float.h>
#include <fenv.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
static void old(const float*x,int8_t*q,uint16_t*s){
 float g=0;for(int i=0;i<32;i++)if(fabsf(x[i])>g)g=fabsf(x[i]);
 if(g<7.8886090522101180541e-31f)g=7.8886090522101180541e-31f;
 float inv=rcp_gpu(g)*127.f;for(int i=0;i<32;i++)q[i]=(int8_t)lrintf(x[i]*inv);*s=qsa_f2h(g*(1.f/127.f));
}
static void group(float a){
 float x[32],clip[32];for(int i=0;i<32;i++){x[i]=i%3==0?0:(i&1)?-a:a;clip[i]=fmaxf(-QSA_KV_MAX_ABS,fminf(QSA_KV_MAX_ABS,x[i]));}
 int8_t q[32],eq[32];uint16_t s,es;assert(!kv_quant32_checked(x,q,&s));old(clip,eq,&es);assert(!memcmp(q,eq,32)&&s==es);
 for(int i=0;i<32;i++){assert(q[i]!=-128);assert(isfinite(kv_deq_gpu(q[i],s)));}
 if(fabsf(a)>=QSA_KV_MAX_ABS)assert(s==0x7bff&&fabsf(kv_deq_gpu(q[1],s))==QSA_KV_MAX_ABS);
 uint32_t bits;memcpy(&bits,&a,4);printf("input=%08x scale=%04x q1=%d q2=%d\n",bits,s,q[1],q[2]);
}
static void explicit_boundaries(void) {
 const float reciprocal = 1.f / 127.f;
 assert(quant_float_bits(reciprocal) == 0x3c010204u);
 volatile float lower_product = quant_float_from_bits(0x367e0001u) * reciprocal;
 volatile float upper_product = quant_float_from_bits(0x367e0002u) * reciprocal;
 assert(quant_float_bits(lower_product) == 0x33000000u);
 assert(quant_float_bits(upper_product) == 0x33000001u);
 for (unsigned i = 8; i < QUANT_RANGE_CASE_COUNT; i += 2) {
  uint32_t lo = quant_range_cases[i].input_bits, hi = quant_range_cases[i + 1].input_bits;
  assert(hi == lo + 1);
  assert(quant_float_bits(nextafterf(quant_float_from_bits(lo), INFINITY)) == hi);
 }
 for (unsigned fixture = 0; fixture < QUANT_RANGE_SIGNED_COUNT; ++fixture) {
  const quant_range_case *test = &quant_range_cases[fixture / 2];
  float a = quant_range_amplitude(fixture), x[32];
  int8_t q[32]; uint16_t scale;
  quant_range_group(x, a);
  assert(!kv_quant32_checked(x, q, &scale));
  assert(scale == test->scale_bits);
  for (unsigned i = 0; i < 32; ++i) {
   int expected = x[i] > 0.f ? 127 : x[i] < 0.f ? -127 : 0;
   assert(q[i] == expected);
   assert(isfinite(kv_deq_gpu(q[i], scale)));
  }
  printf("quant-range cpu fixture=%u case=%s input=%08x expected_scale=%04x scale=%04x q1=%d q2=%d deq1=%08x deq2=%08x\n",
         fixture, test->name, quant_float_bits(a), test->scale_bits, scale, q[1], q[2],
         quant_float_bits(kv_deq_gpu(q[1], scale)), quant_float_bits(kv_deq_gpu(q[2], scale)));
 }
 printf("PASS: %u explicit signed KV range fixtures; adjacent scale boundaries verified\n",
        (unsigned)QUANT_RANGE_SIGNED_COUNT);
}
int main(void){
 if(fegetround()!=FE_TONEAREST){fputs("requires FE_TONEAREST\n",stderr);return 2;}
 explicit_boundaries();
 {float levels[]={0,127.f*0x1p-24f,127.f*0x1p-14f,QSA_KV_MAX_ABS};uint16_t expected[]={0,1,0x400,0x7bff};for(unsigned j=0;j<4;j++){float x[32];for(int i=0;i<32;i++)x[i]=levels[j];int8_t q[32];uint16_t h;assert(!kv_quant32_checked(x,q,&h)&&h==expected[j]);}}
 float ordinary[]={0,-0.f,1e-10f,1e-6f,-1e-6f,.1f,1,-1,3000,QSA_KV_MAX_ABS,-QSA_KV_MAX_ABS,1e7f,-1e7f,FLT_MAX,-FLT_MAX,nextafterf(QSA_KV_MAX_ABS,0),nextafterf(QSA_KV_MAX_ABS,INFINITY)};
 for(unsigned i=0;i<sizeof ordinary/sizeof*ordinary;i++)group(ordinary[i]);
 unsigned lo[]={0,1,0x3ff,0x400,0x3c00,0x7bfe};
 for(unsigned i=0;i<sizeof lo/sizeof*lo;i++){float a=(qsa_h2f(lo[i])+qsa_h2f(lo[i]+1))*.5f*127.f;group(nextafterf(a,0));group(a);group(nextafterf(a,INFINITY));group(-a);}
 group(65520.f*127.f);group(-65520.f*127.f);
 float bad[]={NAN,INFINITY,-INFINITY};for(int j=0;j<3;j++)for(int pos=0;pos<32;pos++){float x[32]={0};x[pos]=bad[j];int8_t q[32],before[32];memset(q,0x5a,32);memcpy(before,q,32);uint16_t s=0x1234;assert(kv_quant32_checked(x,q,&s)&&s==0x1234&&!memcmp(q,before,32));}
 float x[32]={0};int8_t q[32];uint16_t s;assert(kv_quant32_checked(NULL,q,&s));assert(kv_quant32_checked(x,NULL,&s));assert(kv_quant32_checked(x,q,NULL));
 {int modes[]={FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};for(unsigned i=0;i<3;i++){assert(!fesetround(modes[i]));memset(q,0x5a,32);s=0x1234;assert(kv_quant32_checked(x,q,&s)&&s==0x1234);for(int j=0;j<32;j++)assert(q[j]==0x5a);}assert(!fesetround(FE_TONEAREST));}
 puts("PASS: bounded finite quantizer, in-range compatibility, half boundaries, underflow, overflow, unchanged nonfinite rejection");return 0;
}
