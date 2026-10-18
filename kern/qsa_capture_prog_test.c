/* CPU-only check: isolating the capture scratch changes descriptor pointers,
 * not operations, masks, barriers, dimensions, or exchange order. This does not
 * substitute for capture-enabled/disabled output parity on a GPU. */
#include "prog.h"
#include "qsa_capture.h"
#include "gguf.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s failed\n",__LINE__,#x); exit(1); } } while (0)
int main(void) {
    unsigned cases=0, descriptors=0, pointer_rewrites=0;
    const unsigned positions[]={0,3,2046,2047,2048,2051,3099,7999,64999,261999};
    const int ndies[]={1,2,6};
    unsetenv("SPLIT_SHARES"); unsetenv("ATTN_DIE");
    for (unsigned n=0;n<sizeof ndies/sizeof *ndies;n++)
    for (int d=0;d<ndies[n];d++)
    for (unsigned it=0;it<sizeof positions/sizeof *positions;it++) {
        dn_bufs_t a={0},b={0}; uint64_t old[32],fresh[32]; unsigned np=0;
#define INIT(field) a.field=UINT64_C(0x1100000000)+(uint64_t)(np+1)*0x1000000; old[np]=a.field; fresh[np]=a.field+UINT64_C(0x8000000000); ++np;
        QSA_CAPTURE_SCRATCH_FIELDS(INIT)
#undef INIT
        a.qsa=1; a.nblk=512; a.kq=d==0?0x800000:0; a.vq=0x900000; a.ks=0xa00000; a.vs=0xb00000;
        b=a;
        if (d==0) { unsigned j=0;
#define BIND(field) b.field=fresh[j++];
            QSA_CAPTURE_SCRATCH_FIELDS(BIND)
#undef BIND
            CHECK(j==np);
        }
        dn_weights_t w={.attn_q_type=GG_Q5_K,.attn_k_type=GG_Q5_K,.attn_v_type=GG_Q5_K,
            .attn_output_type=GG_Q5_K,.ffn_gate_shexp_type=GG_Q2_0,
            .ffn_up_shexp_type=GG_Q2_0,.ffn_down_shexp_type=GG_Q2_0};
        split_t split; split_of(&split,d,ndies[n],3);
        xchg_t x={.inbox=0x100000,.flags=0x200000,.xgen=0x300000,.xtab=0x400000};
        prog_t plain,capture; prog_init(&plain); prog_init(&capture);
        plain.loop=capture.loop=(int)(it%2);
        prog_attn_block(&plain,&split,&x,&w,&a,positions[it],1e-6f);
        prog_attn_block(&capture,&split,&x,&w,&b,positions[it],1e-6f);
        CHECK(plain.n==capture.n && plain.xj==capture.xj && plain.pending_wait==capture.pending_wait);
        unsigned seen[32]={0};
        for (int i=0;i<plain.n;i++) {
            phase_t normalized=capture.p[i];
            for (int j=2;j<29;j++) {
                uint64_t addr=(uint64_t)normalized.w[j]|((uint64_t)normalized.w[j+1]<<32);
                for (unsigned p=0;p<np;p++) if (addr==fresh[p]) {
                    CHECK(d==0); normalized.w[j]=(uint32_t)old[p]; normalized.w[j+1]=(uint32_t)(old[p]>>32);
                    seen[p]++; pointer_rewrites++; j++; break;
                }
            }
            CHECK(!memcmp(&plain.p[i],&normalized,sizeof normalized)); descriptors++;
        }
        if (d==0) for (unsigned p=0;p<np;p++) CHECK(seen[p]);
        else for (unsigned p=0;p<np;p++) CHECK(!seen[p]);
        prog_free(&plain); prog_free(&capture); cases++;
    }
    printf("PASS_SYNTHETIC_CAPTURE_PLAN: %u cases, %u descriptors, %u expected pointer rewrites; operations/barriers/masks/exchange order unchanged\n",cases,descriptors,pointer_rewrites);
    return 0;
}
