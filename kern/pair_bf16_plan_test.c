#include "prog.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(c) do { if(!(c)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#c);exit(1);} } while(0)
enum change { SAME, OTHER_W, OTHER_ROWS, MASK_A, MASK_B, OWN_X, CROSS_X, SAME_Y, OWN_BIAS, WRITE_W, OTHER_UNITS, OTHER_OP };
static unsigned tests;
static void test(const char *name,int op,int fused,int trusted,enum change change,int expected){
    prog_t a,b,p;prog_init(&a);prog_init(&b);prog_init(&p);a.loop=b.loop=1;a.token_offset=0;b.token_offset=1;
    phase_t *da=prog_add(&a,op,16),*db=prog_add(&b,op,16);
    ph_ptr(da,0x140,0x100000);ph_ptr(db,0x140,0x100000);
    ph_ptr(da,0x148,0x800000);ph_ptr(db,0x148,0xa00000);
    ph_ptr(da,0x150,0xc00000);ph_ptr(db,0x150,0xd00000);
    ph_u32(da,0x158,17);ph_u32(db,0x158,17);
    ph_ptr(da,0x160,0xe00000);ph_ptr(db,0x160,0xf00000);
    ph_u32(da,0x168,16);ph_u32(db,0x168,48);
    CHECK(!prog_pair(&p,&a,&b));CHECK(p.n==2);CHECK(p.p[0].w[30]&8);
    da=&p.p[0];db=&p.p[1];
    switch(change){
    case SAME:break;
    case OTHER_W:ph_ptr(db,0x140,0x200000);break;
    case OTHER_ROWS:ph_u32(db,0x158,18);break;
    case MASK_A:da->w[31]=1;break;
    case MASK_B:db->w[31]=1;break;
    case OWN_X:ph_ptr(da,0x150,0x800000);break;
    case CROSS_X:ph_ptr(da,0x150,0xa00000);break;
    case SAME_Y:ph_ptr(db,0x150,0xc00000);break;
    case OWN_BIAS:ph_ptr(da,0x150,0xe00000);break;
    case WRITE_W:ph_ptr(da,0x150,0x100000);break;
    case OTHER_UNITS:db->w[1]=32;break;
    case OTHER_OP:db->w[0]=OP_BF16K320;break;
    }
    prog_pair_fuse_bf16(&p,trusted);
    CHECK(da->w[0]==(uint32_t)(expected?fused:op));CHECK(!!(da->w[30]&16)==expected);CHECK(p.n==2);
    uint32_t first[32];memcpy(first,da->w,sizeof first);prog_pair_fuse_bf16(&p,trusted);CHECK(!memcmp(first,da->w,sizeof first));
    printf("%s: %s\n",name,expected?"fused":"generic fallback");tests++;prog_free(&p);prog_free(&b);prog_free(&a);
}
int main(void){
#define T(n,c,e) test(n,OP_BF16K2560,OP_BF16K2560P,0,c,e)
    T("matching weights and independent buffers",SAME,1);
    T("different weights",OTHER_W,0);T("different rows",OTHER_ROWS,0);
    T("primary token mask",MASK_A,0);T("secondary token mask",MASK_B,0);
    T("own output/input alias",OWN_X,0);T("cross output/input alias",CROSS_X,0);
    T("cross output/output alias",SAME_Y,0);T("output/weight alias",WRITE_W,0);
    T("different virtual units",OTHER_UNITS,0);T("different scalar opcode",OTHER_OP,0);
    test("separate biases",OP_BF16BIAS,OP_BF16BIASP,0,SAME,1);
    test("output/bias alias",OP_BF16BIAS,OP_BF16BIASP,0,OWN_BIAS,0);
    test("K320 independent buffers",OP_BF16K320,OP_BF16K320P,0,SAME,1);
    test("four-slice HC independent buffers",OP_BF16HC,OP_BF16HCP,0,SAME,1);
    test("XOUT no allocation guarantee",OP_BF16HCX,OP_BF16HCXP,0,SAME,0);
    test("XOUT disjoint allocation guarantee",OP_BF16HCX,OP_BF16HCXP,1,SAME,1);
    printf("BF16 fusion eligibility: %u cases passed\n",tests);return 0;
}
