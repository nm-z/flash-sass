#include "prog.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"%s:%d: %s failed\n",__FILE__,__LINE__,#x); return 1; } } while (0)
static void build(prog_t *p,unsigned off) {
    p->loop=1; p->token_offset=off;
    ph_conv1d(p,0x1000+off*0x100,0x2000,0x3000,0x4000+off*0x100,99,1024);
    ph_actq(p,OP_ACTQ256,0x4000+off*0x100,0x5000+off*0x100,0x6000+off*0x100,0x7000+off*0x100,0x8000+off*0x100,256,4);
    p->xj=2;
}
static int group_checks(void) {
    prog_t p[PROG_MAX_POSITIONS], out;
    const prog_t *positions[PROG_MAX_POSITIONS];
    for (unsigned k=0;k<8;k++) {
        prog_init(&p[k]); build(&p[k],k); positions[k]=&p[k];
    }
    prog_init(&out);
    CHECK(prog_group(&out,positions,8)==0 && out.n==16 && out.xj==2);
    for(int i=0;i<p[0].n;i++) {
        CHECK(phase_positions(&out.p[8*i])==8);
        for(unsigned k=0;k<8;k++) {
            phase_t desc=out.p[8*i+k];
            if(!k)desc.w[30]&=~(PHASE_GROUP_FLAG|PHASE_WIDTH_MASK);
            CHECK(!memcmp(&desc,&p[k].p[i],sizeof desc));
            if(k)CHECK(phase_positions(&desc)==1);
        }
    }
    CHECK(!prog_pair_fuse_bf16(&out,1)); /* wider groups cannot use B2 bodies */
    phase_t saved[16];memcpy(saved,out.p,sizeof saved);phase_t *savedp=out.p;
#define GROUP_BAD(expr,undo) do { expr; CHECK(prog_group(&out,positions,8)<0); undo; CHECK(out.p==savedp && out.n==16 && !memcmp(saved,out.p,sizeof saved)); } while(0)
    GROUP_BAD(p[7].token_offset++,p[7].token_offset--);
    GROUP_BAD(p[7].xj++,p[7].xj--);
    GROUP_BAD(p[7].p[1].w[30]^=4,p[7].p[1].w[30]^=4);
    GROUP_BAD(p[0].p[0].w[30]|=8,p[0].p[0].w[30]&=~8u);
    CHECK(prog_group(&p[7],positions,8)<0);
    CHECK(prog_group(&out,positions,0)<0 && prog_group(&out,positions,PROG_MAX_POSITIONS+1)<0);
    CHECK(prog_group(&out,NULL,8)<0);
    for(unsigned k=0;k<8;k++)p[k].token_offset=UINT32_MAX-3u+k;
    CHECK(prog_group(&out,positions,8)<0); /* offset range cannot wrap */
    for(unsigned k=0;k<8;k++) {
        p[k].token_offset=k;
        const int ops[]={OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE};
        for(int j=0;j<3;j++) {
            phase_t *desc=prog_add(&p[k],ops[j],1);
            desc->w[20]=100*k+(unsigned)j;
            ph_tokv(&p[k],desc,0x178,0,0);
        }
    }
    CHECK(!prog_group(&out,positions,8) && out.n==40);
    for(unsigned k=0;k<8;k++)for(int j=0;j<3;j++) {
        CHECK(phase_positions(&out.p[16+3*k+j])==1);
        CHECK(!memcmp(&out.p[16+3*k+j],&p[k].p[2+j],sizeof(phase_t)));
    }
    unsigned phase_count=0;
    for(unsigned i=0;i<(unsigned)out.n;i+=phase_positions(&out.p[i]))phase_count++;
    CHECK(phase_count==26); /* two B8 phases, then eight scalar triples */
    savedp=out.p;
    for(unsigned k=0;k<8;k++)p[k].p[2].w[30]=2;
    CHECK(prog_group(&out,positions,8)<0 && out.p==savedp && out.n==40);
    for(unsigned k=0;k<8;k++){p[k].p[2].w[30]=0;p[k].p[4].w[0]=OP_ACTQ256;}
    CHECK(prog_group(&out,positions,8)<0 && out.p==savedp && out.n==40);
    for(unsigned k=0;k<8;k++)p[k].p[4].w[0]=OP_ATTNMERGE;
    CHECK(!prog_group(&out,positions,2) && out.n==10);
    CHECK(phase_positions(&out.p[4])==2); /* B2 retains phase-major gathers */
    for(unsigned k=0;k<8;k++) {
        prog_free(&p[k]);prog_init(&p[k]);p[k].loop=1;p[k].token_offset=k;
        for(unsigned i=0;i<PROG_MAX_PHASES;i++)prog_add(&p[k],OP_CONV1D,1);
    }
    CHECK(!prog_group(&out,positions,8));
    CHECK(out.n==(int)(PROG_MAX_PHASES*8u));
    unsigned logical=0;
    for(unsigned i=0;i<(unsigned)out.n;i+=phase_positions(&out.p[i])) {
        CHECK(phase_positions(&out.p[i])==8);logical++;
    }
    CHECK(logical==PROG_MAX_PHASES);
    for(unsigned k=0;k<8;k++)prog_free(&p[k]);
    prog_free(&out);
    return 0;
}
static void wide_span(prog_t *p,unsigned position) {
    static const int ops[]={OP_IDXQK,OP_IDXSCORE,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_IDXSEL,OP_KVGATHER,OP_ATTNPARTN,OP_ATTNMERGE,OP_PLEGATE,OP_RMS2560,OP_PLECONV};
    p->loop=1;p->token_offset=position;
    for(unsigned i=0;i<13;i++) {
        phase_t *d=prog_add(p,ops[i],1);d->w[20]=position*100+i;
        if(i>=2 && i<7)d->w[10]=i;
        if(i==10)d->w[30]=2; /* one preceding V publication; no send inside */
    }
}
static int wide_checks(void) {
    const unsigned widths[]={2,16,32};
    for(unsigned mode=0;mode<3;mode++) {
        unsigned n=widths[mode];prog_t p[PROG_MAX_POSITIONS],out;const prog_t *group[PROG_MAX_POSITIONS];
        for(unsigned k=0;k<n;k++){prog_init(&p[k]);wide_span(&p[k],k);group[k]=&p[k];}
        prog_init(&out);CHECK(!prog_group_aliased(&out,group,n));CHECK(out.n==(int)(13*n));
        for(unsigned k=0;k<n;k++) {
            for(unsigned j=0;j<10;j++)CHECK(!memcmp(&out.p[10*k+j],&p[k].p[j],sizeof(phase_t)));
            for(unsigned j=0;j<3;j++)CHECK(!memcmp(&out.p[10*n+3*k+j],&p[k].p[10+j],sizeof(phase_t)));
        }
        phase_t *saved=out.p;
        p[n-1].p[2].w[10]=99;CHECK(prog_group_aliased(&out,group,n)<0 && out.p==saved);p[n-1].p[2].w[10]=2;
        for(unsigned k=0;k<n;k++)p[k].p[11].w[30]=1;
        CHECK(prog_group_aliased(&out,group,n)<0 && out.p==saved);
        for(unsigned k=0;k<n;k++)prog_free(&p[k]);
        prog_free(&out);
    }
    /* Many serialized spans must fit profiling even above 4096 phases. */
    prog_t p[PROG_MAX_POSITIONS],out;const prog_t *group[PROG_MAX_POSITIONS];
    for(unsigned k=0;k<PROG_MAX_POSITIONS;k++) {
        prog_init(&p[k]);p[k].loop=1;p[k].token_offset=k;group[k]=&p[k];
        for(unsigned repeat=0;repeat<100;repeat++) {
            prog_t span;prog_init(&span);wide_span(&span,k);
            for(int j=0;j<span.n;j++)*prog_add(&p[k],span.p[j].w[0],1)=span.p[j];
            prog_free(&span);
        }
    }
    prog_init(&out);CHECK(!prog_group_aliased(&out,group,32));
    unsigned logical=0;for(unsigned i=0;i<(unsigned)out.n;i+=phase_positions(&out.p[i]))logical++;
    CHECK(logical==41600 && logical>PROG_MAX_PHASES && logical<=PROG_MAX_PROFILE_PHASES);
    for(unsigned k=0;k<PROG_MAX_POSITIONS;k++)prog_free(&p[k]);
    prog_free(&out);
    return 0;
}
static int dnstep_checks(void) {
    const unsigned widths[]={2,8,16,32};
    for(unsigned m=0;m<4;m++)for(unsigned flags=0;flags<8;flags++) {
        unsigned n=widths[m];prog_t p[PROG_MAX_POSITIONS],out;const prog_t *group[PROG_MAX_POSITIONS];
        for(unsigned k=0;k<n;k++) {
            CUdeviceptr base=0x200000000ull+0x100000ull*k;
            prog_init(&p[k]);p[k].loop=1;p[k].token_offset=k;p[k].xj=5;group[k]=&p[k];
            ph_dnstep(&p[k],0x100000000ull,base,base+0x10000,base+0x20000,
                base+0x30000,base+0x40000,0x110000000ull,base+0x50000,1e-6f,3,1,48);
            p[k].p[0].w[30]=flags;
        }
        prog_init(&out);CHECK(!prog_group(&out,group,n));
        phase_t saved[PROG_MAX_POSITIONS];memcpy(saved,out.p,n*sizeof *saved);
        CHECK(prog_batch_fuse_dnstep(&out)==(n>=8?(int)n/8:0));
        CHECK(out.n==(int)n && out.xj==5 && out.loop==1);
        if(n<8)CHECK(!memcmp(saved,out.p,n*sizeof *saved));
        else for(unsigned k=0;k<n;k++) {
            phase_t expected=saved[k];
            if(!(k%8)) {
                unsigned part=flags;
                if(k)part&=~2u;
                if(k+8<n)part=(part&~1u)|4u;
                expected.w[0]=OP_DNSTEP8;
                expected.w[30]=part|PHASE_GROUP_FLAG|PHASE_FUSED_FLAG|(8u<<PHASE_WIDTH_SHIFT);
            }
            CHECK(!memcmp(&expected,&out.p[k],sizeof expected));
        }
        if(n>=8) {
            memcpy(out.p,saved,n*sizeof *saved);
#define DN_REJECT(change) do { \
    change; phase_t before[PROG_MAX_POSITIONS];memcpy(before,out.p,n*sizeof *before); \
    CHECK(!prog_batch_fuse_dnstep(&out));CHECK(!memcmp(before,out.p,n*sizeof *before)); \
    memcpy(out.p,saved,n*sizeof *saved); \
} while(0)
            DN_REJECT(out.p[n-1].w[2]+=16); /* different recurrent state */
            DN_REJECT(out.p[n-1].w[14]+=4); /* different shared gates */
            DN_REJECT(out.p[n-1].w[18]^=1); /* epsilon differs */
            DN_REJECT(out.p[n-1].w[19]++);
            DN_REJECT(out.p[n-1].w[20]++);
            DN_REJECT(out.p[n-1].w[31]=1u<<4);
            DN_REJECT(out.p[n-1].w[4]++); /* unaligned vector load */
            DN_REJECT(out.p[n-1].w[1]--);
            DN_REJECT(ph_ptr(&out.p[n-1],0x178,0x200000000ull)); /* output aliases another input */
            DN_REJECT(ph_ptr(&out.p[n-1],0x178,0x200050000ull)); /* outputs overlap */
            DN_REJECT(ph_ptr(&out.p[n-1],0x178,0x100000000ull)); /* output aliases state */
            DN_REJECT(ph_ptr(&out.p[n-1],0x160,0x100000000ull)); /* gate aliases state */
            DN_REJECT(out.p[0].w[30]|=1u<<24); /* unknown flag */
#undef DN_REJECT
        }
        for(unsigned k=0;k<n;k++)prog_free(&p[k]);
        prog_free(&out);
    }
    return 0;
}
int main(void) {
    CHECK(!group_checks());
    CHECK(!wide_checks());
    CHECK(!dnstep_checks());
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
    puts("pair_prog_test: B2/B8/B16/B32 offsets, serialized lifetimes, 41600 profile phases, DeltaNet subgroup flags/aliases, and shape rejection PASS");return 0;
}
