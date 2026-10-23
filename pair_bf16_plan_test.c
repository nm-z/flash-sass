#include "prog.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if(!(c)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#c);exit(1);} } while(0)
enum change { SAME, OTHER_W, OTHER_ROWS, MASK_A, MASK_B, OWN_X, CROSS_X, SAME_Y, OWN_BIAS, WRITE_W, OTHER_UNITS, OTHER_OP, OTHER_FLAGS };
static unsigned tests,width=2;
static void test(const char *name,int op,int fused,int trusted,enum change change,int expected){
    prog_t positions[PROG_MAX_POSITIONS],p;const prog_t *group[PROG_MAX_POSITIONS];prog_init(&p);
    for(unsigned i=0;i<width;i++){
        prog_init(&positions[i]);positions[i].loop=1;positions[i].token_offset=i;group[i]=&positions[i];
        phase_t *d=prog_add(&positions[i],op,16);
        ph_ptr(d,0x140,0x100000);ph_ptr(d,0x148,0x1000000+(CUdeviceptr)i*0x100000);
        ph_ptr(d,0x150,0x3000000+(CUdeviceptr)i*0x100000);ph_u32(d,0x158,17);
        ph_ptr(d,0x160,0x5000000+(CUdeviceptr)i*0x100000);ph_u32(d,0x168,16+4*i);
    }
    CHECK(!prog_group(&p,group,width));CHECK(p.n==(int)width);CHECK(p.p[0].w[30]&8);
    phase_t *da=&p.p[0],*db=&p.p[width-1];
    switch(change){
    case SAME:break;
    case OTHER_W:ph_ptr(db,0x140,0x200000);break;
    case OTHER_ROWS:ph_u32(db,0x158,18);break;
    case MASK_A:da->w[31]=1;break;
    case MASK_B:db->w[31]=1;break;
    case OWN_X:ph_ptr(da,0x150,0x1000000);break;
    case CROSS_X:ph_ptr(da,0x150,0x1000000+(CUdeviceptr)(width-1)*0x100000);break;
    case SAME_Y:ph_ptr(db,0x150,0x3000000);break;
    case OWN_BIAS:ph_ptr(da,0x150,0x5000000);break;
    case WRITE_W:ph_ptr(da,0x150,0x100000);break;
    case OTHER_UNITS:db->w[1]=32;break;
    case OTHER_OP:db->w[0]=OP_BF16K320;break;
    case OTHER_FLAGS:db->w[30]^=1;break;
    }
    if(width==8){
        switch(op){
        case OP_BF16BIAS:fused=OP_BF16BIAS8;break;
        case OP_BF16K2560:fused=OP_BF16K25608;break;
        case OP_BF16K320:fused=OP_BF16K3208;break;
        case OP_BF16HC:fused=OP_BF16HC8;break;
        case OP_BF16HCX:fused=OP_BF16HCX8;break;
        }
        prog_batch_fuse_bf16(&p,trusted);
    }else prog_pair_fuse_bf16(&p,trusted);
    CHECK(da->w[0]==(uint32_t)(expected?fused:op));CHECK(!!(da->w[30]&16)==expected);CHECK(p.n==(int)width);
    uint32_t first[32];memcpy(first,da->w,sizeof first);
    if(width==8)prog_batch_fuse_bf16(&p,trusted);else prog_pair_fuse_bf16(&p,trusted);
    CHECK(!memcmp(first,da->w,sizeof first));
    printf("B%u %s: %s\n",width,name,expected?"fused":"generic fallback");tests++;prog_free(&p);
    for(unsigned i=0;i<width;i++)prog_free(&positions[i]);
}
int main(void){
#define T(n,c,e) test(n,OP_BF16K2560,OP_BF16K2560P,0,c,e)
    for(unsigned mode=0;mode<2;mode++){
    width=mode?8:2;
    T("matching weights and independent buffers",SAME,1);
    T("different weights",OTHER_W,0);T("different rows",OTHER_ROWS,0);
    T("primary token mask",MASK_A,0);T("secondary token mask",MASK_B,0);
    T("own output/input alias",OWN_X,0);T("cross output/input alias",CROSS_X,0);
    T("cross output/output alias",SAME_Y,0);T("output/weight alias",WRITE_W,0);
    T("different virtual units",OTHER_UNITS,0);T("different scalar opcode",OTHER_OP,0);
    T("different exchange flags",OTHER_FLAGS,0);
    test("separate biases",OP_BF16BIAS,OP_BF16BIASP,0,SAME,1);
    test("output/bias alias",OP_BF16BIAS,OP_BF16BIASP,0,OWN_BIAS,0);
    test("K320 independent buffers",OP_BF16K320,OP_BF16K320P,0,SAME,1);
    test("four-slice HC independent buffers",OP_BF16HC,OP_BF16HCP,0,SAME,1);
    test("XOUT no allocation guarantee",OP_BF16HCX,OP_BF16HCXP,0,SAME,0);
    test("XOUT disjoint allocation guarantee",OP_BF16HCX,OP_BF16HCXP,1,SAME,1);
    test("RMS remains generic",OP_BF16HCRX,OP_BF16HCRX,1,SAME,0);
    test("HCUP remains generic",OP_BF16HCUP,OP_BF16HCUP,1,SAME,0);
    }
    printf("BF16 fusion eligibility: %u cases passed\n",tests);return 0;
}
