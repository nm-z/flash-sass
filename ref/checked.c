#include "checked.h"
#include "attn.h"
#include "qsa.h"
static int float_count(size_t n,size_t*out){if(n>SIZE_MAX/512u)return 1;n*=512u;if(n>SIZE_MAX/sizeof(float))return 1;*out=n;return 0;}
int attn_cache_sizes_checked(unsigned n,size_t*k,size_t*v){
 if(!k||!v||!n||n>UINT_MAX-8191u||(size_t)n>SIZE_MAX-31u)return 1;
 size_t kk,vv;if(float_count(((size_t)n+31u)/32u*32u,&kk)||float_count(n,&vv))return 1;*k=kk;*v=vv;return 0;
}
int attn_decode_checked(const float*q,const float*k,size_t kf,const float*v,size_t vf,unsigned n,const float*g,float*out){
 if(!q||!k||!v||!g||!out)return 1;size_t nk,nv;if(attn_cache_sizes_checked(n,&nk,&nv)||kf<nk||vf<nv)return 1;attn_decode_gpu(q,k,v,n,g,out);return 0;
}
int qsa_nkeys_checked(unsigned p,unsigned cap,unsigned b,unsigned*n){
 if(!n||!cap||p>=cap||b<1||b>4096)return 1;size_t k,v;if(attn_cache_sizes_checked(cap,&k,&v))return 1;
 unsigned visible=p+1,blocks=visible/4,tail=visible%4;*n=blocks<=b?visible:4*b+tail;return 0;
}
int qsa_gather_checked(const int8_t*kq,const uint16_t*ks,const int8_t*vq,const uint16_t*vs,unsigned cap,const uint32_t*sel,size_t sc,unsigned p,unsigned b,float*k,size_t kf,float*v,size_t vf,unsigned*n){
 if(!kq||!ks||!vq||!vs||!k||!v||!n)return 1;unsigned keys;if(qsa_nkeys_checked(p,cap,b,&keys))return 1;
 unsigned visible=p+1,blocks=visible/4,tail=visible%4,selected=blocks<b?blocks:b,slots=4*(selected+(tail!=0));
 if(tail&&((uint64_t)blocks+1)*4>cap)return 1;
 size_t nk,nv;if(attn_cache_sizes_checked(slots,&nk,&nv)||kf<nk||vf<nv)return 1;
 if(blocks>b){if(!sel||sc<selected)return 1;for(unsigned i=0;i<selected;i++)if(sel[i]>=blocks||(i&&sel[i-1]>=sel[i]))return 1;}
 *n=qsa_gather_gpu(kq,ks,vq,vs,sel,p,b,k,v);return 0;
}
