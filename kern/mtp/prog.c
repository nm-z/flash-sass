#include "mtp/prog.h"
int mtp_ph_q8(prog_t *p,unsigned k,CUdeviceptr w,CUdeviceptr x,CUdeviceptr y,
              unsigned rows,unsigned stride,CUdeviceptr indices,unsigned slot,CUdeviceptr count) {
    int op;
    switch(k) {
    case 320:op=OP_MTP_Q8_320;break;
    case 640:op=OP_MTP_Q8_640;break;
    case 2560:op=OP_MTP_Q8_2560;break;
    case 5120:op=OP_MTP_Q8_5120;break;
    case 6144:op=OP_MTP_Q8_6144;break;
    case 10240:op=OP_MTP_Q8_10240;break;
    default:return -1;
    }
    if(!rows || rows>248320 || slot>10) return -1;
    phase_t *a=prog_add(p,op,(rows+7)/8);
    ph_ptr(a,0x140,w);ph_ptr(a,0x148,x);ph_ptr(a,0x150,y);
    ph_u32(a,0x158,rows);ph_u32(a,0x15c,stride);ph_ptr(a,0x160,indices);
    ph_u32(a,0x168,slot);ph_ptr(a,0x170,count);return 0;
}
void mtp_ph_swiglu(prog_t *p,CUdeviceptr g,CUdeviceptr u,CUdeviceptr y,unsigned slot,CUdeviceptr count) {
    phase_t *a=prog_add(p,OP_MTP_SWIGLU,3);
    ph_ptr(a,0x140,g);ph_ptr(a,0x148,u);ph_ptr(a,0x150,y);
    ph_u32(a,0x158,slot);ph_ptr(a,0x160,count);
}
void mtp_ph_weighted_sum(prog_t *p,CUdeviceptr x,CUdeviceptr w,CUdeviceptr count,
                         CUdeviceptr shared,CUdeviceptr shared_weight,CUdeviceptr y) {
    phase_t *a=prog_add(p,OP_MTP_WEIGHTED_SUM,10);
    ph_ptr(a,0x140,x);ph_ptr(a,0x148,w);ph_ptr(a,0x150,count);
    ph_ptr(a,0x158,shared);ph_ptr(a,0x160,shared_weight);ph_ptr(a,0x168,y);
}
void mtp_ph_eh_concat(prog_t *p,CUdeviceptr embedding,CUdeviceptr hidden,CUdeviceptr y) {
    phase_t *a=prog_add(p,OP_MTP_EH_CONCAT,4);
    ph_ptr(a,0x140,embedding);ph_ptr(a,0x148,hidden);ph_ptr(a,0x150,y);
}
