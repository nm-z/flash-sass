/* CPU upload substitution: validate actual checkpoint slicing and packed values
 * without touching a CUDA context. Raw GGUF dequantization is the value oracle. */
#define _POSIX_C_SOURCE 200809L
#include "mtp/load.h"
#include "quant.h"
#include "split.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
typedef struct { uint8_t *p;size_t n; } allocation;
static allocation allocated[128];static unsigned count;static unsigned failures;
static CUdeviceptr allocate_cpu(size_t n) {
    if(count==128)exit(2);
    uint8_t *p=malloc(n+64);if(!p)exit(2);memset(p,0xa5,n+64);
    allocated[count++]=(allocation){p,n};return (CUdeviceptr)(uintptr_t)p;
}
CUresult CUDAAPI cuMemcpyHtoD_v2(CUdeviceptr destination,const void *source,size_t bytes) {
    for(unsigned i=0;i<count;i++) {
        uintptr_t begin=(uintptr_t)allocated[i].p;
        if(destination>=begin && destination-begin<=allocated[i].n && bytes<=allocated[i].n-(destination-begin)) {
            memcpy((void *)(uintptr_t)destination,source,bytes);return CUDA_SUCCESS;
        }
    }
    fprintf(stderr,"CPU upload outside allocation: %zu bytes\n",bytes);exit(2);
}
static const gguf_tensor *find(const gguf_file *g,const char *suffix) {
    char name[128];snprintf(name,sizeof name,"blk.48.%s",suffix);
    const gguf_tensor *t=gguf_find(g,name);if(!t)exit(2);return t;
}
static void q8_row(const gguf_file *g,const char *name,CUdeviceptr packed,unsigned local,unsigned rawrow,unsigned k) {
    const gguf_tensor *t=find(g,name);float values[10240];
    if(dequant(GG_Q8_0,t->data+(size_t)rawrow*(k/32)*34,values,k))exit(2);
    unsigned prefix=(k/16+15)&~15u;
    const uint8_t *row=(const uint8_t *)(uintptr_t)packed+(size_t)local*(prefix+k);
    for(unsigned j=0;j<k;j++) {
        uint16_t scale;memcpy(&scale,row+2*(j/32),2);
        float actual=fp16_to_fp32(scale)*(float)(int8_t)row[prefix+j];
        if(memcmp(&actual,&values[j],4)) { if(failures<8)fprintf(stderr,"%s local=%u raw=%u col=%u differs\n",name,local,rawrow,j);failures++;break; }
    }
}
static void sample_q8(const gguf_file *g,const char *name,CUdeviceptr ptr,unsigned start,unsigned n,unsigned k) {
    unsigned sample[]={0,n/2,n-1};for(unsigned i=0;i<3;i++)q8_row(g,name,ptr,sample[i],start+sample[i],k);
}
static void raw(const gguf_file *g,const char *name,CUdeviceptr ptr,size_t bytes) {
    if(memcmp((void *)(uintptr_t)ptr,find(g,name)->data,bytes)) { fprintf(stderr,"%s raw bytes differ\n",name);failures++; }
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    setvbuf(stdout,NULL,_IONBF,0);
    setenv("SPLIT_SHARES","112,97,81,116,97,97",1);
    gguf_file g;if(gguf_open(&g,argv[1]))return 2;
    dn_weights_t target={.output=0x1234,.obases=0x5678};
    for(int die=0;die<6;die++) {
        split_t s;split_of(&s,die,6,48);mtp_weights_t mw,plan;size_t bytes,planned;
        if(mtp_load_weights(&plan,&g,&s,&target,5,NULL,&planned)!=CUDA_SUCCESS) return 1;
        CUresult result=mtp_load_weights(&mw,&g,&s,&target,5,allocate_cpu,&bytes);
        if(result!=CUDA_SUCCESS) { fprintf(stderr,"MTP load die%d failed %d\n",die,result);return 1; }
        if(bytes!=planned)failures++;
        const dn_weights_t *w=&mw.block;
        if(w->output!=target.output||w->obases!=target.obases)failures++;
        raw(&g,"nextn.hnorm.weight",mw.hnorm,4*10240);raw(&g,"nextn.enorm.weight",mw.enorm,4*2560);
        raw(&g,"nextn.hc_head_norm.weight",mw.head_norm,4*10240);
        raw(&g,"hc_attn_norm.weight",w->hc_attn_norm,4*10240);raw(&g,"hc_ffn_norm.weight",w->hc_ffn_norm,4*10240);
        sample_q8(&g,"nextn.eh_proj.weight",mw.eh_proj,0,2560,5120);
        sample_q8(&g,"nextn.hc_head_down.weight",mw.head_down,(unsigned)s.hd0,(unsigned)s.hdn,10240);
        for(unsigned lane=0;lane<4;lane++) {
            unsigned n=32u*(unsigned)s.cbn,start=lane*2560+32u*(unsigned)s.cb0;
            const char *names[]={"hc_attn_up.weight","hc_ffn_up.weight","nextn.hc_head_up.weight"};
            CUdeviceptr matrices[]={w->hc_attn_up,w->hc_ffn_up,mw.head_up};
            for(int m=0;m<3;m++) { q8_row(&g,names[m],matrices[m],lane*n,start,320);q8_row(&g,names[m],matrices[m],(lane+1)*n-1,start+n-1,320); }
        }
        for(unsigned row=0;row<(unsigned)s.djn;row++) {
            unsigned global=(unsigned)s.dj0+row,source=global<320?global:global-320;
            q8_row(&g,global<320?"hc_attn_down.weight":"hc_attn_inject.weight",w->hc_attn_dj,row,source,10240);
            q8_row(&g,global<320?"hc_ffn_down.weight":"hc_ffn_inject.weight",w->hc_ffn_dj,row,source,10240);
        }
        sample_q8(&g,"ffn_gate_exps.weight",w->ffn_gate_exps,(unsigned)s.e0*640,(unsigned)s.en*640,2560);
        sample_q8(&g,"ffn_up_exps.weight",w->ffn_up_exps,(unsigned)s.e0*640,(unsigned)s.en*640,2560);
        sample_q8(&g,"ffn_down_exps.weight",w->ffn_down_exps,(unsigned)s.e0*2560,(unsigned)s.en*2560,640);
        sample_q8(&g,"attn_output.weight",w->attn_output,(unsigned)s.y0,(unsigned)s.yn,6144);
        if(s.shexp) {
            sample_q8(&g,"ffn_gate_shexp.weight",w->ffn_gate_shexp,0,640,2560);
            sample_q8(&g,"ffn_up_shexp.weight",w->ffn_up_shexp,0,640,2560);
            sample_q8(&g,"ffn_down_shexp.weight",w->ffn_down_shexp,0,2560,640);
        }
        const uint16_t *router=(const uint16_t *)(uintptr_t)w->ffn_gate_inp_x;
        for(unsigned row=0;row<513;row+=512) {
            const float *source=(const float *)find(&g,row?"ffn_gate_inp_shexp.weight":"ffn_gate_inp.weight")->data;
            for(unsigned j=0;j<2560;j++) {
                float actual=bf16_to_fp32(router[(size_t)row*2560+j]);
                /* Round-trip relative bound for conversion, with exact zero. */
                float delta=actual-source[j];if(delta<0)delta=-delta;
                float limit=source[j]<0?-source[j]:source[j];
                if(delta>limit/256.0f+0x1p-126f)failures++;
            }
        }
        if(die==5) {
            sample_q8(&g,"attn_q.weight",w->attn_q,0,12288,2560);
            sample_q8(&g,"attn_k.weight",w->attn_k,0,512,2560);sample_q8(&g,"attn_v.weight",w->attn_v,0,512,2560);
            raw(&g,"indexer.q_proj.weight",w->indexer_x,2*512*2560);
            raw(&g,"indexer.k_proj.weight",w->indexer_x+2*512*2560,2*128*2560);
        } else if(w->attn_q||w->indexer_x)failures++;
        for(unsigned e=0;e<512;e++) {
            uint32_t actual=((uint32_t *)(uintptr_t)w->emap)[e];
            uint32_t expected=e>=(unsigned)s.e0&&e<(unsigned)(s.e0+s.en)?e-(unsigned)s.e0:0xffffffffu;
            failures+=actual!=expected;
        }
        for(unsigned a=0;a<count;a++) { for(unsigned j=0;j<64;j++)failures+=allocated[a].p[allocated[a].n+j]!=0xa5;free(allocated[a].p); }
        printf("mtp_load_test die=%d bytes=%zu allocations=%u cumulative_failures=%u\n",die,bytes,count,failures);count=0;
    }
    float embedding[2560];const gguf_tensor *e=mtp_embedding_tensor(&g);
    if(!e||mtp_embedding_row(e,0,embedding)||mtp_embedding_row(e,248319,embedding)||!mtp_embedding_row(e,248320,embedding))failures++;
    gguf_close(&g);printf("mtp_load_test: six-die checkpoint values/slices/guards %s\n",failures?"FAIL":"PASS");return failures?1:0;
}
