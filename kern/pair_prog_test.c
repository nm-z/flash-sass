#include "prog.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s failed\n",__FILE__,__LINE__,#x); return 1; } } while (0)
static unsigned long long ptr_at(const phase_t *p, int addr) {
    int w = 2 + (addr - 0x140) / 4;
    return p->w[w] | ((unsigned long long)p->w[w + 1] << 32);
}
static void build(prog_t *p,unsigned off) {
    p->loop=1; p->token_offset=off;
    ph_conv1d(p,0x1000+off*0x100,0x2000,0x3000,0x4000+off*0x100,99,1024);
    ph_actq(p,OP_ACTQ256,0x4000+off*0x100,0x5000+off*0x100,0x6000+off*0x100,0x7000+off*0x100,0x8000+off*0x100,256,4);
    p->xj=2;
}
static int test_ple_rms_barriers(void) {
    prog_t p; split_t s; xchg_t x={0}; dn_weights_t w={0}; dn_bufs_t b={0};
    prog_init(&p); split_of(&s,0,6,1); unsetenv("PROG_NONOBAR");
    prog_ple_prefix(&p,&s,&x,&w,&b,0,1e-5f);
    int i;
    for(i=0;i+2<p.n;i++)
        if(p.p[i].w[0]==OP_RMS2560 && p.p[i+1].w[0]==OP_RMS2560 && p.p[i+2].w[0]==OP_BF16K2560) break;
    CHECK(i+2<p.n);
    CHECK(p.p[i].w[30]&4u); CHECK(p.p[i+1].w[30]&4u);
    CHECK(!(p.p[i+2].w[30]&4u));
    prog_free(&p);
    return 0;
}
static int test_moe_actq_projection_overlap(void) {
    prog_t p; split_t s; xchg_t x={0}; dn_weights_t w={0}; dn_bufs_t b={0};
    prog_init(&p); split_of(&s,0,6,1); unsetenv("PROG_NONOBAR");
    unsetenv("PROG_NOHCUP"); unsetenv("PROG_NORMSFUSE"); unsetenv("PROG_NORMSW");
    x.inbox=0x1000000; b.zq=0x2000000; b.r=0x3000000;
    w.attn_qkv_type=GG_Q4_K; w.attn_gate_type=GG_Q4_K; w.ssm_out_type=GG_Q4_K;
    prog_deltanet_block(&p,&s,&x,&w,&b,0,1e-5f);
    int i;
    for(i=0;i+1<p.n;i++)
        if(p.p[i].w[0]==OP_ACTQ256 && p.p[i+1].w[0]==OP_BF16K2560) break;
    CHECK(i+1<p.n);
    CHECK(p.p[i].w[30]&4u);
    CHECK(!(p.p[i+1].w[30]&4u));
    CHECK(ptr_at(&p.p[i],0x140)==ptr_at(&p.p[i+1],0x148));
    CHECK(ptr_at(&p.p[i],0x148)!=ptr_at(&p.p[i+1],0x150));
    prog_free(&p);
    return 0;
}
static int test_explicit_exchange_wait_clear(void) {
    prog_t p; xchg_t x={0};
    prog_init(&p);
    ph_xchg(&p,&x,0,6,0x1000,1024,0,6144);
    CHECK(p.pending_wait && (p.p[0].w[30]&1u));
    phase_t *consumer=prog_add(&p,OP_ACTQ256,1);
    CHECK(consumer->w[30]&2u);
    prog_free(&p);

    prog_init(&p);
    ph_xchg(&p,&x,0,6,0x1000,1024,0,6144);
    prog_clear_wait(&p);
    phase_t *independent=prog_add(&p,OP_XSEND,1);
    CHECK(!(independent->w[30]&2u) && !p.pending_wait);
    prog_free(&p);
    return 0;
}
int main(void) {
    CHECK(!test_ple_rms_barriers()); CHECK(!test_moe_actq_projection_overlap());
    CHECK(!test_explicit_exchange_wait_clear());
    prog_t a,b,out; prog_init(&a);prog_init(&b);prog_init(&out);
    CHECK(!a.token_offset); build(&a,0);build(&b,1);
    const int tw=2+(0x160-0x140)/4;
    CHECK(a.p[0].w[tw]==0 && b.p[0].w[tw]==1);
    CHECK(a.p[0].w[31] == (1u<<tw));
    CHECK(prog_pair(&out,&a,&b)==0); CHECK(out.n==4 && out.xj==2 && out.loop==1);
    for(int i=0;i<a.n;i++) {
        phase_t p=out.p[2*i]; CHECK(p.w[30] & 8u); p.w[30]&=~8u;
        CHECK(!memcmp(&p,&a.p[i],sizeof p));
        CHECK(!memcmp(&out.p[2*i+1],&b.p[i],sizeof p));
    }
    phase_t saved[4]; memcpy(saved,out.p,sizeof saved); phase_t *savedp=out.p;
    const int savedn=out.n,savedxj=out.xj;
#define BAD(expr,undo) do { expr; CHECK(prog_pair(&out,&a,&b)<0); undo; CHECK(out.p==savedp && out.n==savedn && out.xj==savedxj && !memcmp(saved,out.p,sizeof saved)); } while(0)
    BAD(b.n--,b.n++); BAD(b.xj++,b.xj--); BAD(b.loop=0,b.loop=1);
    BAD(b.token_offset=2,b.token_offset=1);
    BAD(b.p[0].w[0]++,b.p[0].w[0]--);
    BAD(b.p[0].w[1]++,b.p[0].w[1]--);
    BAD(b.p[0].w[30]^=4,b.p[0].w[30]^=4);
    BAD(a.p[0].w[30]|=8,a.p[0].w[30]&=~8u);
    BAD(b.pending_wait=1,b.pending_wait=0);
    CHECK(prog_pair(&a,&a,&b)<0);
    prog_t absolute;prog_init(&absolute);absolute.token_offset=7;
    ph_conv1d(&absolute,1,2,3,4,99,1024); CHECK(absolute.p[0].w[tw]==99 && !absolute.p[0].w[31]);
    prog_free(&absolute);prog_free(&a);prog_free(&b);prog_free(&out);
    puts("pair_prog_test: safe MoE ACTQ overlap, PLE barriers, explicit exchange-wait clear, token offsets, descriptor preservation, shape rejection, and unchanged-on-error PASS");return 0;
}
