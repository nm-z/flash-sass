/* CPU-only contract checks. No CUDA header, library, context or launch. */
#include "checked.h"
#include "attn.h"
#include "qsa.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(void){
 const char*bad[]={"","-1","+1"," 1","1 ","1x","1.0","0x10","99999999999999999999999"};
 unsigned n;for(unsigned i=0;i<sizeof bad/sizeof*bad;i++){n=77;assert(checked_parse_unsigned(bad[i],0,UINT_MAX,&n)&&n==77);}
 n=77;assert(checked_parse_unsigned("0",1,UINT_MAX,&n)&&n==77);assert(!checked_parse_unsigned("0",0,UINT_MAX,&n)&&n==0);assert(!checked_parse_unsigned("4096",1,4096,&n)&&n==4096);n=77;assert(checked_parse_unsigned("4097",1,4096,&n)&&n==77);
 char max[32];snprintf(max,sizeof max,"%u",UINT_MAX);assert(!checked_parse_unsigned(max,0,UINT_MAX,&n)&&n==UINT_MAX);assert(checked_parse_unsigned(NULL,0,9,&n));assert(checked_parse_unsigned("1",0,9,NULL));
 unsigned cases[][3]={{0,0,512},{1,1,512},{UINT_MAX,UINT_MAX,512},{0,4,0},{0,4,4097}};
 for(unsigned i=0;i<sizeof cases/sizeof*cases;i++){n=77;assert(qsa_nkeys_checked(cases[i][0],cases[i][1],cases[i][2],&n)&&n==77);}
 assert(!qsa_nkeys_checked(0,1,1,&n)&&n==1);assert(!qsa_nkeys_checked(2050,262144,512,&n)&&n==2051);assert(!qsa_nkeys_checked(2051,262144,512,&n)&&n==2048);assert(!qsa_nkeys_checked(262143,262144,4096,&n)&&n==16384);
 size_t kf=77,vf=88;assert(attn_cache_sizes_checked(0,&kf,&vf)&&kf==77&&vf==88);assert(attn_cache_sizes_checked(UINT_MAX,&kf,&vf)&&kf==77&&vf==88);
 assert(!attn_cache_sizes_checked(262144,&kf,&vf)&&kf==(size_t)262144*512&&vf==kf);
 enum{CAP=16,KF=32*512,VF=CAP*512};
 float*q=calloc(6144,4),*k=calloc(KF,4),*v=calloc(VF,4),*gate=calloc(12288,4),*out=malloc(6144*4),*before=malloc(6144*4);
 assert(q&&k&&v&&gate&&out&&before);for(int i=0;i<6144;i++)out[i]=123.25f;memcpy(before,out,6144*4);
 assert(attn_decode_checked(q,k,KF,v,VF,0,gate,out));assert(!memcmp(out,before,6144*4));
 assert(attn_decode_checked(q,k,KF-1,v,VF,1,gate,out));assert(!memcmp(out,before,6144*4));
 assert(attn_decode_checked(q,k,KF,v,511,1,gate,out));assert(!memcmp(out,before,6144*4));
 assert(attn_decode_checked(NULL,k,KF,v,VF,1,gate,out));assert(!memcmp(out,before,6144*4));
 for(int i=0;i<512;i++)v[i]=1;assert(!attn_decode_checked(q,k,KF,v,VF,1,gate,out));for(int i=0;i<6144;i++)assert(out[i]==.5f);
 int8_t*kq=calloc(CAP*512,1),*vq=calloc(CAP*512,1);uint16_t*ks=malloc(CAP*32),*vs=malloc(CAP*32);assert(kq&&vq&&ks&&vs);
 for(int i=0;i<CAP*16;i++)ks[i]=vs[i]=0x3c00;for(int i=0;i<512;i++){kq[i]=1;vq[i]=2;}
 float*kb=malloc(KF*4),*vb=malloc(VF*4);assert(kb&&vb);memcpy(kb,k,KF*4);memcpy(vb,v,VF*4);
 uint32_t ids[4][2]={{0,2},{1,1},{2,1},{0,4}};
 for(int i=0;i<4;i++){n=77;assert(qsa_gather_checked(kq,ks,vq,vs,CAP,ids[i],i?2:1,15,2,k,KF,v,VF,&n));assert(n==77&&!memcmp(k,kb,KF*4)&&!memcmp(v,vb,VF*4));}
 n=77;assert(qsa_gather_checked(kq,ks,vq,vs,1,NULL,0,0,1,k,KF,v,VF,&n)&&n==77);
 assert(!qsa_gather_checked(kq,ks,vq,vs,CAP,NULL,0,0,1,k,KF,v,VF,&n)&&n==1);assert(k[attn_kidx(0,0,0)]==1&&v[0]==2);
 assert(!qsa_gather_checked(kq,ks,vq,vs,CAP,ids[0],2,15,2,k,KF,v,VF,&n)&&n==8);
 free(q);free(k);free(v);free(gate);free(out);free(before);free(kq);free(vq);free(ks);free(vs);free(kb);free(vb);
 puts("PASS: strict parser, empty/overflow/capacity/budget rejection, unchanged outputs, one-key attention, selected gather");return 0;
}
