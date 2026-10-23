/* Quantized B4 draft operations. The host connects these in resident streams
 * or graphs; tensors and caches stay on the device throughout a draft round. */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math_constants.h>
#include "device.h"

static __device__ __forceinline__ float half_value(const unsigned char *p) {
    return __half2float(__ushort_as_half(*(const unsigned short *)p));
}
static __device__ __forceinline__ void scale_min(const unsigned char *s,unsigned j,unsigned *scale,unsigned *minimum) {
    if(j<4){*scale=s[j]&63;*minimum=s[j+4]&63;}
    else {*scale=(s[j+4]&15)|((s[j-4]>>6)<<4);*minimum=(s[j+4]>>4)|((s[j]>>6)<<4);}
}
static __device__ __forceinline__ float matrix_weight(const unsigned char *row,unsigned kind,unsigned k,unsigned column) {
    unsigned block=column>>8,within=column&255,group=within>>5,lane=within&31;
    if(kind==DFLASH_Q6_K) {
        const unsigned char *p=row+210u*block;
        unsigned half=group>>2,part=group&3;
        unsigned low=p[half*64+(part&1)*32+lane];
        low=(low>>(part>=2?4:0))&15;
        unsigned high=(p[128+half*32+lane]>>(2*part))&3;
        int scale=(signed char)p[192+half*8+part*2+(lane>>4)];
        return half_value(p+208)*(float)(scale*((int)(low|(high<<4))-32));
    }
    const unsigned char *header;
    unsigned code;
    if(kind==DFLASH_TARGET_Q5_TILE) {
        unsigned blocks=k>>8;header=row+16u*block;
        unsigned byte=row[16u*blocks+128u*block+16u*group+(lane>>1)];
        code=(byte>>(4*(lane&1)))&15;
        unsigned high=*(const unsigned *)(row+144u*blocks+32u*block+4u*group);
        code|=((high>>lane)&1)<<4;
    } else {
        header=row+144u*block;
        unsigned byte=header[16+32*(group>>1)+lane];
        code=(byte>>(4*(group&1)))&15;
    }
    unsigned scale,minimum;scale_min(header+4,group,&scale,&minimum);
    return half_value(header)*(float)(scale*code)-half_value(header+2)*(float)minimum;
}

/* Four independent accumulators share each unpacked weight. Each warp owns
 * one output row; every lane retains its own within-result addition order. */
extern "C" __global__ void df4_matrix(const unsigned char *weights,const float *x,float *y,
                                      unsigned rows,unsigned k,unsigned n,unsigned kind,unsigned stride) {
    unsigned lane=threadIdx.x&31,row=blockIdx.x*(blockDim.x/32)+(threadIdx.x>>5);
    if(row>=rows)return;
    const unsigned char *w=weights+(size_t)row*stride;
    float sum[4]={0,0,0,0};
    for(unsigned block=0;block<k;block+=256) {
#pragma unroll
        for(unsigned j=0;j<8;j++) {
            unsigned column=block+j*32+lane;
            float value=matrix_weight(w,kind,k,column);
#pragma unroll
            for(unsigned p=0;p<4;p++)if(p<n)sum[p]=fmaf(value,x[(size_t)p*k+column],sum[p]);
        }
    }
#pragma unroll
    for(unsigned p=0;p<4;p++)if(p<n) {
        for(unsigned delta=16;delta;delta>>=1)sum[p]+=__shfl_down_sync(0xffffffffu,sum[p],delta);
        if(!lane)y[(size_t)p*rows+row]=sum[p];
    }
}

extern "C" __global__ void df4_norm(const float *x,float *y,const float *weight,unsigned width,unsigned vectors,float epsilon) {
    unsigned vector=blockIdx.x,lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(vector>=vectors)return;
    __shared__ float sums[8];float sum=0;
    for(unsigned j=threadIdx.x;j<width;j+=blockDim.x){float v=x[(size_t)vector*width+j];sum=fmaf(v,v,sum);}
    for(unsigned delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,delta);
    if(!lane)sums[warp]=sum;
    __syncthreads();
    if(!warp) {
        sum=lane<blockDim.x/32?sums[lane]:0;
        for(unsigned delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffffu,sum,delta);
        if(!lane)sums[0]=rsqrtf(sum/width+epsilon);
    }
    __syncthreads();float inverse=sums[0];
    for(unsigned j=threadIdx.x;j<width;j+=blockDim.x)y[(size_t)vector*width+j]=x[(size_t)vector*width+j]*inverse*weight[j];
}
extern "C" __global__ void df4_conv(const float *x,float *y,const float *projection,const float *base,unsigned side,unsigned n) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*2560)return;
    unsigned p=i/2560,c=i%2560,group=c/16,offset=p*640+side*320+group;
    float a=base[side*5120+c]+projection[offset];
    float b=base[side*5120+2560+c]+projection[offset+160];
    y[i]=a*x[i]+(p?b*x[i-2560]:0.f);
}
extern "C" __global__ void df4_add(float *x,const float *residual,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)x[i]+=residual[i];
}
extern "C" __global__ void df4_swiglu(float *gate,const float *up,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count){float v=gate[i];gate[i]=v/(1.f+__expf(-v))*up[i];}
}
extern "C" __global__ void df4_rope(float *x,const float *frequencies,const dflash_device_control *control,unsigned heads,unsigned n,unsigned append) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n*heads*64)return;
    unsigned j=i&63,head=(i/64)%heads,p=i/(heads*64);
    unsigned start=append?control->append_start:control->context;
    float sine,cosine;sincosf((float)(start+p)*frequencies[j],&sine,&cosine);
    float *v=x+((size_t)p*heads+head)*128;
    float a=v[j],b=v[j+64];v[j]=a*cosine-b*sine;v[j+64]=a*sine+b*cosine;
}
extern "C" __global__ void df4_cache_put(const float *k,const float *v,unsigned short *kc,unsigned short *vc,const dflash_device_control *control) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=control->append_count*1024)return;
    unsigned p=i/1024,element=i%1024;
    size_t dst=(size_t)((control->append_start+p)&2047)*1024+element;
    kc[dst]=__half_as_ushort(__float2half_rn(k[i]));vc[dst]=__half_as_ushort(__float2half_rn(v[i]));
}

/* A CTA shares a32-key K/V tile across four query heads for each of four
 * draft positions. Four independent context partitions fill the16-SM device.
 * Noncausal draft positions see the whole current block, as in the checkpoint. */
extern "C" __global__ void df4_attention_parts(const float *q,const float *self_k,const float *self_v,
                                              const unsigned short *cache_k,const unsigned short *cache_v,
                                              float *partial,float *statistics,const dflash_device_control *control,unsigned n) {
    __shared__ unsigned short keys[32*128],values[32*128];
    unsigned lane=threadIdx.x&31,p=threadIdx.x>>5,kv=blockIdx.x%8,part=blockIdx.x/8;
    unsigned context=control->context,first=context+1>2048?context+1-2048:0;
    unsigned total=context-first+n,begin=total*part/4,end=total*(part+1)/4;
    float query[4][4],acc[4][4],maximum[4],denominator[4];
#pragma unroll
    for(unsigned h=0;h<4;h++) {
        maximum[h]=-CUDART_INF_F;denominator[h]=0;
#pragma unroll
        for(unsigned j=0;j<4;j++) {
            query[h][j]=p<n?q[((size_t)p*32+kv*4+h)*128+lane+32*j]:0;
            acc[h][j]=0;
        }
    }
    for(unsigned at=begin;at<end;at+=32) {
        unsigned count=end-at<32?end-at:32;
        for(unsigned i=threadIdx.x;i<count*128;i+=blockDim.x) {
            unsigned position=first+at+i/128,dimension=i%128;
            size_t index=((size_t)(position<context?(position&2047):position-context)*8+kv)*128+dimension;
            keys[i]=position<context?cache_k[index]:__half_as_ushort(__float2half_rn(self_k[index]));
            values[i]=position<context?cache_v[index]:__half_as_ushort(__float2half_rn(self_v[index]));
        }
        __syncthreads();
        if(p<n)for(unsigned t=0;t<count;t++) {
            unsigned position=first+at+t,lower=context+p+1>2048?context+p+1-2048:0;
            if(position<lower)continue;
            float key[4],value[4];
#pragma unroll
            for(unsigned j=0;j<4;j++) {
                key[j]=__half2float(__ushort_as_half(keys[t*128+lane+32*j]));
                value[j]=__half2float(__ushort_as_half(values[t*128+lane+32*j]));
            }
#pragma unroll
            for(unsigned h=0;h<4;h++) {
                float score=0;
#pragma unroll
                for(unsigned j=0;j<4;j++)score=fmaf(query[h][j],key[j],score);
                for(unsigned delta=16;delta;delta>>=1)score+=__shfl_xor_sync(0xffffffffu,score,delta);
                score*=0.08838834764831845f;
                float next=fmaxf(maximum[h],score);
                float old_scale=__expf(maximum[h]-next),new_scale=__expf(score-next);
                denominator[h]=fmaf(denominator[h],old_scale,new_scale);maximum[h]=next;
#pragma unroll
                for(unsigned j=0;j<4;j++)acc[h][j]=fmaf(acc[h][j],old_scale,value[j]*new_scale);
            }
        }
        __syncthreads();
    }
    if(p<n) {
#pragma unroll
        for(unsigned h=0;h<4;h++) {
            size_t query_index=(size_t)part*n*32+p*32+kv*4+h;
            if(!lane){statistics[2*query_index]=maximum[h];statistics[2*query_index+1]=denominator[h];}
#pragma unroll
            for(unsigned j=0;j<4;j++)partial[query_index*128+lane+32*j]=acc[h][j];
        }
    }
}
extern "C" __global__ void df4_attention_merge(const float *partial,const float *statistics,float *out,unsigned n) {
    unsigned query=blockIdx.x,lane=threadIdx.x;if(query>=n*32 || lane>=32)return;
    float maximum=-CUDART_INF_F,scale[4],denominator=0;
    for(unsigned part=0;part<4;part++)maximum=fmaxf(maximum,statistics[2*((size_t)part*n*32+query)]);
    for(unsigned part=0;part<4;part++) {
        size_t at=(size_t)part*n*32+query;float d=statistics[2*at+1];
        scale[part]=d>0?__expf(statistics[2*at]-maximum):0;denominator=fmaf(scale[part],d,denominator);
    }
#pragma unroll
    for(unsigned j=0;j<4;j++) {
        float sum=0;unsigned dimension=lane+32*j;
        for(unsigned part=0;part<4;part++)sum=fmaf(scale[part],partial[((size_t)part*n*32+query)*128+dimension],sum);
        out[(size_t)query*128+dimension]=sum/denominator;
    }
}

extern "C" __global__ void df4_features(const float *ring,float *out,const dflash_device_control *control) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*12800)return;
    unsigned p=i/12800,element=i%12800;
    out[i]=p<control->append_count?ring[(size_t)((control->append_start+p)&2047)*12800+element]:0;
}
extern "C" __global__ void df4_embedding(const dflash_embedding_shard *shards,float *out,
                                         const dflash_device_control *control,unsigned mask_token) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=4*2560)return;
    unsigned p=i/2560,column=i%2560,group=column/32,lane=column%32;
    unsigned token=p?mask_token:control->anchor;
    for(unsigned owner=0;owner<6;owner++) {
        dflash_embedding_shard s=shards[owner];
        if(!s.data || group<s.first_group || group>=s.first_group+s.groups)continue;
        const unsigned char *row=(const unsigned char *)(uintptr_t)s.data+(size_t)token*s.stride;
        unsigned code,high,scale;float d;
        if(!s.raw_q3) {
            const unsigned char *g=row+16*(group-s.first_group);
            unsigned long long bits=*(const unsigned long long *)g;
            code=(bits>>(2*lane))&3;high=(*(const unsigned *)(g+8)>>lane)&1;
            scale=g[12+(lane>>4)];d=half_value(g+14);
        } else {
            const unsigned char *g=row+110*(column/256);unsigned v=column%256;
            unsigned half=v>>7,j=(v>>5)&3,which=(v>>4)&1,l=v&15,sg=v>>4;
            code=(g[32+32*half+16*which+l]>>(2*j))&3;
            high=(g[16*which+l]>>(4*half+j))&1;
            scale=((g[96+(sg&7)]>>(sg<8?0:4))&15)|(((g[104+(sg&3)]>>(2*(sg>>2)))&3)<<4);
            d=half_value(g+108);
        }
        out[i]=(d*(float)((int)scale-32))*(float)((int)code-(high?0:4));return;
    }
}

static __device__ __forceinline__ void sort256(float *scores,unsigned *ids) {
    unsigned t=threadIdx.x;
    for(unsigned size=2;size<=256;size<<=1)for(unsigned stride=size>>1;stride;stride>>=1) {
        unsigned peer=t^stride;
        if(t<peer) {
            bool worse=scores[t]<scores[peer] || (scores[t]==scores[peer] && ids[t]>ids[peer]);
            if((!(t&size) && worse) || ((t&size) && !worse)) {
                float value=scores[t];scores[t]=scores[peer];scores[peer]=value;
                unsigned id=ids[t];ids[t]=ids[peer];ids[peer]=id;
            }
        }
        __syncthreads();
    }
}
extern "C" __global__ void df4_topk_logits(const float *logits,dflash_candidate *out,unsigned rows,unsigned first) {
    __shared__ float values[256];__shared__ unsigned ids[256];
    unsigned t=threadIdx.x,row=blockIdx.x*256+t,p=blockIdx.y,blocks=(rows+255)/256;
    values[t]=row<rows?logits[(size_t)p*rows+row]:-CUDART_INF_F;ids[t]=row<rows?first+row:0xffffffffu;
    __syncthreads();sort256(values,ids);
    if(t<16)out[((size_t)p*blocks+blockIdx.x)*16+t]={values[t],ids[t]};
}
extern "C" __global__ void df4_topk_pairs(const dflash_candidate *input,dflash_candidate *out,unsigned count) {
    __shared__ float values[256];__shared__ unsigned ids[256];
    unsigned t=threadIdx.x,i=blockIdx.x*256+t,p=blockIdx.y,blocks=(count+255)/256;
    dflash_candidate v=i<count?input[(size_t)p*count+i]:dflash_candidate{-CUDART_INF_F,0xffffffffu};
    values[t]=v.value;ids[t]=v.token;__syncthreads();sort256(values,ids);
    if(t<16)out[((size_t)p*blocks+blockIdx.x)*16+t]={values[t],ids[t]};
}
extern "C" __global__ void df4_head_merge(const uint64_t *sources,dflash_candidate *out) {
    __shared__ float values[256];__shared__ unsigned ids[256];
    unsigned t=threadIdx.x,p=blockIdx.x;dflash_candidate v={-CUDART_INF_F,0xffffffffu};
    if(t<96)v=((const dflash_candidate *)(uintptr_t)sources[t/16])[p*16+t%16];
    values[t]=v.value;ids[t]=v.token;__syncthreads();sort256(values,ids);
    if(t<16)out[p*16+t]={values[t],ids[t]};
}
extern "C" __global__ void df4_select(const unsigned char *predecessor,const unsigned char *successor,
                                      unsigned predecessor_type,unsigned successor_type,unsigned predecessor_stride,unsigned successor_stride,
                                      const float *projection,const dflash_candidate *candidates,
                                      const dflash_device_control *control,unsigned *tokens) {
    __shared__ float scores[16];__shared__ unsigned chosen[4];
    unsigned t=threadIdx.x,c=t>>4,lane=t&15;
    if(!t){chosen[0]=control->anchor;tokens[0]=control->anchor;}__syncthreads();
    for(unsigned p=1;p<4;p++) {
        dflash_candidate candidate=candidates[(p-1)*16+c];
        const unsigned char *a=predecessor+(size_t)chosen[p-1]*predecessor_stride;
        const unsigned char *b=successor+(size_t)candidate.token*successor_stride;
        float sum=0;
        for(unsigned r=lane;r<256;r+=16) {
            float before=matrix_weight(a,predecessor_type,256,r);
            float after=matrix_weight(b,successor_type,256,r);
            sum=fmaf(before*projection[p*256+r],after,sum);
        }
        for(unsigned delta=8;delta;delta>>=1)sum+=__shfl_xor_sync(0xffffffffu,sum,delta,16);
        if(!lane)scores[c]=candidate.value+sum;__syncthreads();
        if(!t) {
            unsigned best=0;for(unsigned j=1;j<16;j++)if(scores[j]>scores[best])best=j;
            chosen[p]=candidates[(p-1)*16+best].token;tokens[p]=chosen[p];
        }
        __syncthreads();
    }
}
