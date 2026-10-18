/* CPU-only readback mock: verifies exporter byte layout and failure handling.
 * This does not execute CUDA or validate kernel arithmetic. */
#define _POSIX_C_SOURCE 200809L
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define MAXD 1
#define MAXB 48
#define FLASH_REV "cpu-readback-mock"
#define BLK_IS_ATTN(b) (((b) & 3) == 3)
typedef uintptr_t CUdeviceptr;
typedef struct {
    CUdeviceptr qg,k,v,iqk,iq,qo,aout,kq,vq,ks,vs,pkt,sc,sel,rope_cs,rope_sn;
} dn_bufs_t;
typedef struct {
    CUdeviceptr attn_q_norm,attn_k_norm,indexer_qn,indexer_kn;
} dn_weights_t;
static int model_qsa=1,nd=1,model_host=1;
static unsigned model_budget=1;
static dn_bufs_t model_qdev[MAXD][MAXB];
static CUdeviceptr model_trace[MAXD][MAXB];
static int ctx[MAXD];
static int attn_die_of(int b) { (void)b; return 0; }
static int cuCtxSetCurrent(int c) { (void)c; return 0; }
static int cuMemcpyDtoH(void *p, CUdeviceptr src, size_t n) {
    memcpy(p,(const void *)src,n); return 0;
}
#define CK(x) do { if ((x)) abort(); } while(0)
#include "model_layer_capture.h"

int main(int argc,char **argv) {
    if(argc!=3) return 2;
    (void)model_host;
    setenv("MODEL_LAYER_CAPTURE",argv[1],1);
    setenv("MODEL_LAYER_INDEX","3",1);
    setenv("MODEL_LAYER_LENGTHS",argv[2],1);
    setenv("MODEL_IDS","cpu-mock-only",1);
    setenv("MODEL_PLE","cpu-mock-only",1);
    model_layer_capture_check(0,48,1,1,0,9);
    float qg[12288]={0},k[512],v[512],iqk[640]={0},iq[512]={0};
    float qo[6144]={0},aout[6144]={0},out[10240]={0},norm[256],rope[9*32];
    float sc[2]={0.5f,1.5f}; uint32_t sel[1]={1},ids[9];
    int8_t kq[9*512],vq[9*512]; uint16_t ks[9*16],vs[9*16],pkt[4096]={0};
    for(unsigned i=0;i<256;i++) norm[i]=1.0f+(float)i/256;
    for(unsigned i=0;i<9*32;i++) rope[i]=(float)i/32;
    memset(kq,7,sizeof kq);memset(vq,-7,sizeof vq);
    for(unsigned i=0;i<9*16;i++) {ks[i]=0x3c00;vs[i]=0x4000;}
    for(unsigned b=0;b<2;b++) for(unsigned d=0;d<128;d++)
        pkt[(d>>2)*128+b*4+(d&3)]=(uint16_t)(0x3000+b*128+d);
    for(unsigned i=0;i<9;i++)ids[i]=10+i;
    dn_bufs_t *q=&model_qdev[0][3],db[1]={{0}};
#define PTR(field,value) q->field=(CUdeviceptr)(value)
    PTR(qg,qg);PTR(k,k);PTR(v,v);PTR(iqk,iqk);PTR(iq,iq);PTR(qo,qo);
    PTR(aout,aout);PTR(kq,kq);PTR(vq,vq);PTR(ks,ks);PTR(vs,vs);
    PTR(pkt,pkt);PTR(sc,sc);PTR(sel,sel);
#undef PTR
    db[0].rope_cs=(CUdeviceptr)rope;db[0].rope_sn=(CUdeviceptr)rope;
    model_trace[0][3]=(CUdeviceptr)out;
    dn_weights_t dw[1][48]={0};
    dw[0][3].attn_q_norm=dw[0][3].attn_k_norm=(CUdeviceptr)norm;
    dw[0][3].indexer_qn=dw[0][3].indexer_kn=(CUdeviceptr)norm;
    for(unsigned t=0;t<9;t++){
        for(unsigned i=0;i<512;i++){k[i]=(float)(100*t+i);v[i]=-k[i];}
        for(unsigned i=0;i<128;i++)iqk[512+i]=(float)(t*10+i);
        qg[0]=(float)t; aout[0]=(float)(t+1000);
        model_layer_capture_record(t,9,ids,db,dw);
    }
    return 0;
}
