/* Host rewrite oracle and guarded two-context HCUP/HCWSUM oracle.
 * No CUDA calls in --cpu-only. GPU entry requires gpuq and explicit device
 * selection; physical 7 is forbidden because it belongs to another workload. */
#include "prog.h"
#include "quant.h"
#include "hc.h"
#include "fmath.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <assert.h>
#include <setjmp.h>
#ifdef HCUP_OFFLOAD_BORROWED
#include "hcup_offload_probe.h"
#endif

CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
static FILE *fixture_output;
#define FIXTURE_OUT (fixture_output?fixture_output:stdout)
#define FIXTURE_ERR (fixture_output?fixture_output:stderr)
#ifdef HCUP_OFFLOAD_BORROWED
static jmp_buf fixture_jb;
static CUresult fixture_error;
static void fixture_fail(CUresult error,const char *expression,unsigned line) {
    const char *message="fixture assertion";if(error)cuGetErrorString(error,&message);
    fprintf(FIXTURE_ERR,"hcup_offload_probe error line=%u expression=%s cuda=%u message=%s\n",line,expression,(unsigned)error,message);
    fixture_error=error?error:CUDA_ERROR_INVALID_VALUE;longjmp(fixture_jb,1);
}
#define CK(x) do { CUresult r_=(x);if(r_)fixture_fail(r_,#x,__LINE__); } while(0)
#define REQUIRE(x) do { if(!(x))fixture_fail(CUDA_ERROR_INVALID_VALUE,#x,__LINE__); } while(0)
#else
#define CK(x) do { CUresult r_=(x);if(r_){const char *s_="?";cuGetErrorString(r_,&s_);fprintf(FIXTURE_ERR,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s_);exit(2);} } while(0)
#define REQUIRE(x) do { if(!(x)){fprintf(FIXTURE_ERR,"%s:%d %s\n",__FILE__,__LINE__,#x);exit(3);} } while(0)
#endif
/* These plans survive a caught CUDA error so the borrowed entry can free every
 * host descriptor allocation, including partially constructed programs. */
static prog_t build_temp[32],live_plans[4];
#ifndef HCUP_OFFLOAD_BORROWED
static uint64_t address(const phase_t *p,unsigned word){return p->w[word]|((uint64_t)p->w[word+1]<<32);}
static unsigned publications(const prog_t *p) {
    unsigned n=0;for(int i=0;i<p->n;i+=(int)phase_positions(p->p+i))n+=!!(p->p[i].w[30]&1);return n;
}
#endif
static prog_t clone(const prog_t *p) {
    prog_t q=*p;q.p=malloc((size_t)p->n*sizeof *q.p);REQUIRE(q.p);
    memcpy(q.p,p->p,(size_t)p->n*sizeof *q.p);q.cap=q.n;return q;
}
#ifndef HCUP_OFFLOAD_BORROWED
static void same(const prog_t *a,const prog_t *b) {
    REQUIRE(a->n==b->n && a->xj==b->xj && a->pending_wait==b->pending_wait);
    REQUIRE(!memcmp(a->p,b->p,(size_t)a->n*sizeof *a->p));
}
#endif
/* Each site has its own incoming publication and an immediate consuming
 * HCWSUM (when stream/output are supplied), otherwise an inert WAIT phase.
 * Reusing the input slot is intentional: the preceding stand-in producer
 * publishes the identical immutable canonical DOWN on both contexts. */
static void make_plan(prog_t *out,const hcup_offload_layout *l,unsigned width,
    CUdeviceptr weight,unsigned sites,unsigned slot,unsigned tn,int fused,int extra,
    const CUdeviceptr *stream,const CUdeviceptr *reduced) {
    prog_t *p=build_temp;const prog_t *input[32];
    for(unsigned k=0;k<width;k++) {
        prog_init(p+k);p[k].loop=1;p[k].token_offset=k;input[k]=p+k;
        if(!slot){prog_add(p+k,OP_EMBROW,0)->w[30]=1;p[k].xj++;p[k].pending_wait=1;}
        for(unsigned j=0;j<sites;j++) {
            prog_add(p+k,OP_EMBROW,0)->w[30]|=1;p[k].xj++;p[k].pending_wait=1;
            xchg_t x={.inbox=l[k].inbox,.xtab=l[k].table};
            CUdeviceptr down=l[k].inbox+(1u-slot)*XSLOT;
            ph_hcup(p+k,&x,weight,down,l[k].up,l[k].columns,
                l[k].xh+4u*l[k].first_column,l[k].t,tn,4u*l[k].first_column);
            if(stream) {
                phase_t *c=prog_add(p+k,OP_HCWSUM,16);
                ph_ptr(c,0x140,l[k].t+320*4);ph_ptr(c,0x148,l[k].inbox+slot*XSLOT);
                ph_ptr(c,0x150,stream[k]);ph_u32(c,0x158,2560);ph_u32(c,0x15c,10240);
                ph_u32(c,0x164,1);ph_ptr(c,0x168,reduced[k]);ph_ptr(c,0x170,l[k].xh);
            } else prog_add(p+k,OP_EMBROW,0);
            if(extra)prog_add(p+k,OP_EMBROW,0);
        }
    }
    if(width==1)*out=clone(p);
    else {prog_init(out);REQUIRE(!prog_group(out,input,width));}
    if(fused && width>=8)REQUIRE(prog_batch_fuse_hcup(out,1)==(int)(sites*width/8));
    for(unsigned k=0;k<width;k++)prog_free(p+k);
}
#ifndef HCUP_OFFLOAD_BORROWED
static void fake_layout(hcup_offload_layout l[2][32],unsigned width) {
    for(unsigned d=0;d<2;d++)for(unsigned p=0;p<width;p++) {
        uint64_t base=0x200000000ull+d*0x10000000ull+p*0x100000ull;
        l[d][p]=(hcup_offload_layout){.inbox=base,.table=base+0x30000,.xh=base+0x40000,
            .up=base+0x50000,.t=base+0x60000,.first_column=d?2144:1728,.columns=416};
    }
}
static void plan_checks(void) {
    unsigned cases=0;const unsigned widths[]={1,2,8,16,32},periods[]={0,1,2,3,4,9};
    for(unsigned w=0;w<5;w++)for(unsigned f=0;f<2;f++)for(unsigned slot=0;slot<2;slot++)
    for(unsigned tn=320;tn<=324;tn+=4)for(unsigned q=0;q<6;q++) {
        unsigned n=widths[w],period=periods[q];hcup_offload_layout l[2][32];fake_layout(l,n);
        prog_t a,b;make_plan(&a,l[0],n,0x100000000ull,8,slot,tn,f,0,NULL,NULL);
        make_plan(&b,l[1],n,0x110000000ull,8,slot,tn,f,1,NULL,NULL);
        prog_t olda=clone(&a),oldb=clone(&b);unsigned origin[4096];memset(origin,0xa5,sizeof origin);
        unsigned selected=period?8/period:0;
        REQUIRE(prog_offload_hcup(&a,&b,l[0],l[1],n,period,4096,origin)==(int)selected);
        REQUIRE(a.n==olda.n && b.n==oldb.n+(int)(selected*n));
        REQUIRE(publications(&a)==publications(&olda) && publications(&b)==publications(&oldb));
        REQUIRE(a.xj==olda.xj && b.xj==oldb.xj && b.pending_wait==oldb.pending_wait);
        unsigned source_sites=0,own_sites=0,remote=0;
        for(int i=0;i<a.n;i+=(int)phase_positions(a.p+i)) {
            phase_t *z=a.p+i;int hcup=z->w[0]==OP_BF16HCUP || z->w[0]==OP_BF16HCUP8;
            if(hcup) {
                unsigned position=0;
                do {
                    unsigned width=phase_positions(a.p+i+position);
                    for(unsigned k=0;k<width;k++) {
                        phase_t expected=olda.p[i+position+k];
                        if(period && (source_sites+1)%period==0)expected.w[1]=0;
                        REQUIRE(!memcmp(&expected,a.p+i+position+k,sizeof expected));
                    }
                    position+=width;
                }while(position<n);
                i+=(int)n-(int)phase_positions(a.p+i);source_sites++;
            } else REQUIRE(!memcmp(z,olda.p+i,sizeof *z));
        }
        if(!selected)same(&oldb,&b);
        else for(int i=0;i<b.n;) {
            unsigned oi=origin[i];REQUIRE(oi<(unsigned)oldb.n);
            phase_t *z=b.p+i;int hcup=z->w[0]==OP_BF16HCUP || z->w[0]==OP_BF16HCUP8;
            if(!hcup) {REQUIRE(!memcmp(z,oldb.p+oi,sizeof *z));i+=(int)phase_positions(z);continue;}
            int moved=address(z,2)==0x100000000ull;
            if(moved)remote++;else own_sites++;
            unsigned position=0;
            while(position<n) {
                phase_t *primary=z+position;unsigned width=phase_positions(primary);
                if(moved) {
                    REQUIRE(!(primary->w[30]&2));
                    REQUIRE(!!(primary->w[30]&1)==(position+width==n));
                } else {
                    int chosen=own_sites%period==0;
                    REQUIRE(!!(primary->w[30]&2)==(position==0));
                    REQUIRE(!!(primary->w[30]&1)==(!chosen && position+width==n));
                    REQUIRE(!!(primary->w[30]&4)==(chosen || position+width<n));
                }
                for(unsigned k=0;k<width;k++) {
                    phase_t expected=oldb.p[origin[i+position+k]];
                    if(moved) {
                        ph_ptr(&expected,0x140,0x100000000ull);ph_ptr(&expected,0x150,l[0][position+k].up);
                        ph_ptr(&expected,0x170,l[0][position+k].table);ph_ptr(&expected,0x180,l[0][position+k].t);
                        ph_ptr(&expected,0x178,l[1][position+k].xh+4u*l[0][position+k].first_column);
                        expected.w[12]=slot*XSLOT+4u*l[0][position+k].first_column;expected.w[30]&=~2u;
                    } else if(!k && own_sites%period==0 && position+width==n)expected.w[30]=(expected.w[30]&~1u)|4u;
                    REQUIRE(!memcmp(&expected,z+position+k,sizeof expected));
                }
                position+=width;
            }
            i+=(int)n;
        }
        REQUIRE(source_sites==8 && (!selected || (own_sites==8 && remote==selected)));
        if(selected)REQUIRE(prog_offload_hcup(&a,&b,l[0],l[1],n,period,4096,NULL)<0);
        prog_free(&a);prog_free(&b);prog_free(&olda);prog_free(&oldb);cases++;
    }
    for(unsigned w=0;w<5;w++)for(unsigned change=0;change<27;change++) {
        unsigned n=widths[w];hcup_offload_layout l[2][32];fake_layout(l,n);prog_t a,b;
        make_plan(&a,l[0],n,0x100000000ull,3,1,324,1,0,NULL,NULL);
        make_plan(&b,l[1],n,0x110000000ull,3,1,324,1,0,NULL,NULL);
        phase_t *z=b.p+n;unsigned capacity=4096;
        switch(change) {
        case 0:b.xj++;break;case 1:b.loop=0;break;case 2:b.token_offset++;break;
        case 3:b.pending_wait^=1;break;case 4:b.stop[0]--;break;case 5:b.samp_k=2;break;
        case 6:z->w[31]=1;break;case 7:z->w[1]=0;break;case 8:z->w[8]+=4;break;
        case 9:z->w[12]+=4;break;case 10:z->w[20]=320;break;case 11:z->w[30]^=2;break;
        case 12:ph_ptr(z,0x148,l[1][0].inbox+XSLOT);break;
        case 13:ph_ptr(z,0x178,l[1][0].xh+4u*l[0][0].first_column);break;
        case 14:ph_ptr(z,0x150,l[0][0].up);break;case 15:ph_ptr(z,0x170,l[0][0].table);break;
        case 16:ph_ptr(z,0x180,l[0][0].t);break;case 17:ph_ptr(z,0x140,l[0][0].xh);break;
        case 18:z->w[30]|=32;break;case 19:l[1][0].columns=2561;break;
        case 20:l[1][0].first_column=l[0][0].first_column;break;
        case 21:l[1][n-1].t=l[0][0].xh+4;break;case 22:l[1][n-1].xh=l[0][0].xh;break;
        case 23:capacity=(unsigned)b.n;break;case 24:ph_ptr(z,0x140,0);break;
        case 25:ph_ptr(z,0x140,UINT64_MAX-15);break;
        case 26: /* Same total sends, but HCUP publication ordinal differs. */
            b.p[0].w[30]&=~1u;b.p[2*n].w[30]|=1u;break;
        }
        prog_t olda=clone(&a),oldb=clone(&b);phase_t *ap=a.p,*bp=b.p;unsigned origin[4096];memset(origin,0xa5,sizeof origin);
        REQUIRE(prog_offload_hcup(&a,&b,l[0],l[1],n,1,capacity,origin)<0);
        REQUIRE(a.p==ap && b.p==bp && origin[0]==0xa5a5a5a5u);same(&a,&olda);same(&b,&oldb);
        prog_free(&a);prog_free(&b);prog_free(&olda);prog_free(&oldb);cases++;
    }
    REQUIRE(!prog_offload_hcup(NULL,NULL,NULL,NULL,0,0,0,NULL));
    fprintf(FIXTURE_OUT,"HCUP offload: %u B1/B2/B8/B16/B32 full-byte remap/selection/fusion/origin/atomic-rejection cases PASS\n",cases);
}
#endif

enum { GUARD=256,MAX_ALLOC=2048,CTA=16,DIES=6 };
typedef struct { CUcontext ctx;CUdeviceptr base,p;size_t bytes;const char *name; } allocation;
static allocation allocations[MAX_ALLOC];static unsigned nalloc;
static CUcontext contexts[2];static CUfunction functions[2];
static CUdeviceptr borrowed_weights[2],live_flags[2];
static unsigned launched[2];
static void *host_scratch[64];static unsigned nhost;
static void *host_alloc(size_t bytes,int zero) {
    REQUIRE(nhost<64);void *p=zero?calloc(1,bytes):malloc(bytes);
#ifdef HCUP_OFFLOAD_BORROWED
    if(!p)fixture_fail(CUDA_ERROR_OUT_OF_MEMORY,"host scratch",__LINE__);
#else
    REQUIRE(p);
#endif
    host_scratch[nhost++]=p;return p;
}
static void host_free(void *p) {
    if(!p)return;
    for(unsigned i=0;i<nhost;i++)if(host_scratch[i]==p){free(p);host_scratch[i]=host_scratch[--nhost];return;}
    REQUIRE(0 && "unregistered fixture host scratch");
}
static size_t memory_needed(unsigned width) {
    return (size_t)width*(2*XSLOT+2*40960+10240+2*10240+1296+16*33+96+9*GUARD*2)+512*1024;
}
static CUdeviceptr guarded(unsigned d,const char *name,size_t bytes) {
    REQUIRE(nalloc<MAX_ALLOC);CK(cuCtxSetCurrent(contexts[d]));
    allocation *a=allocations+nalloc++;*a=(allocation){.ctx=contexts[d],.bytes=bytes,.name=name};
    CK(cuMemAlloc(&a->base,bytes+2*GUARD));a->p=a->base+GUARD;
    CK(cuMemsetD8(a->base,0xa5,bytes+2*GUARD));return a->p;
}
static int compare(const char *name,CUdeviceptr p,const void *expected,size_t bytes) {
    unsigned char *got=host_alloc(bytes,0);CK(cuMemcpyDtoH(got,p,bytes));int bad=memcmp(got,expected,bytes)!=0;
    if(bad){size_t i=0;while(i<bytes && got[i]==((const unsigned char *)expected)[i])i++;fprintf(FIXTURE_ERR,"%s mismatch byte %zu/%zu\n",name,i,bytes);}
    host_free(got);return bad;
}
static int guards(void) {
    int bad=0;unsigned char expected[GUARD];memset(expected,0xa5,sizeof expected);
    for(unsigned i=0;i<nalloc;i++) {
        allocation *a=allocations+i;CK(cuCtxSetCurrent(a->ctx));
        bad+=compare(a->name,a->base,expected,GUARD)+compare(a->name,a->p+a->bytes,expected,GUARD);
    }
    return bad;
}
static void free_allocations(void) {
    while(nalloc){allocation *a=allocations+nalloc-1;CK(cuCtxSetCurrent(a->ctx));if(a->base)CK(cuMemFree(a->base));memset(a,0,sizeof *a);nalloc--;}
}
static uint32_t seed=8819;
static float sample(void){seed=1664525u*seed+1013904223u;return ((int)((seed>>8)&65535)-32768)/32768.f;}

static int gpu_case(unsigned width,unsigned slot,unsigned tn,unsigned period) {
    enum { SITES=2 };hcup_offload_layout l[2][32];CUdeviceptr stream[2][32],reduced[2][32];
    CUdeviceptr weights[2],program[2],ctl[2],flags[2],extra[32][4];uint64_t tables[32][12];
    uint16_t *w[2];float *up[2],*mixed=host_alloc((size_t)width*2560*4,1),*xh=host_alloc((size_t)width*40960,0);
    float down[32][324],t[32][324];unsigned first[2]={17,123},cols[2]={17,33};
    REQUIRE(mixed && xh);
    for(unsigned p=0;p<width;p++) {
        for(unsigned i=0;i<324;i++)down[p][i]=4.f*sample();
        /* Nontrivial inject bits must survive the remote T write unchanged. */
        uint32_t inject[4]={0x80000000u,0x3f800001u+p,0xc0800000u,0x00000000u};
        memcpy(down[p]+320,inject,sizeof inject);hc_silu_gpu(down[p],t[p]);memcpy(t[p]+320,inject,sizeof inject);
        for(unsigned i=0;i<10240;i++)xh[(size_t)p*10240+i]=2.f*sample();
    }
    for(unsigned d=0;d<2;d++) {
        size_t wb=(size_t)4*cols[d]*320*2;w[d]=host_alloc(wb,0);up[d]=host_alloc((size_t)width*16*cols[d],0);
        CK(cuCtxSetCurrent(contexts[d]));
        if(borrowed_weights[d]) {
            /* Use a prefix of the core's existing IPC weight mapping. The
             * small fixture interprets that immutable prefix as a packed
             * matrix, proving helper access to the actual mapped pointer. */
            weights[d]=borrowed_weights[d];CK(cuMemcpyDtoH(w[d],weights[d],wb));
        } else {
            for(size_t i=0;i<wb/2;i++){float v=sample()*.125f+(d?.03125f:0);uint32_t bits;memcpy(&bits,&v,4);w[d][i]=(uint16_t)(bits>>16);}
            weights[d]=guarded(d,"weights",wb);CK(cuMemcpyHtoD(weights[d],w[d],wb));
        }
        flags[d]=guarded(d,"exchange flags",96*4);ctl[d]=guarded(d,"control",8192);
        live_flags[d]=flags[d];
        program[d]=guarded(d,"program",32u*width*sizeof(phase_t));
        for(unsigned p=0;p<width;p++) {
            l[d][p]=(hcup_offload_layout){.inbox=guarded(d,"inbox",2*XSLOT),.table=guarded(d,"table",96),
                .xh=guarded(d,"canonical XH",40960),.up=guarded(d,"UP",16u*cols[d]),
                .t=guarded(d,"T",1296),.first_column=first[d],.columns=cols[d]};
            CK(cuMemcpyHtoD(l[d][p].xh,xh+(size_t)p*10240,40960));
            stream[d][p]=guarded(d,"HCWSUM stream",40960);reduced[d][p]=guarded(d,"HCWSUM Y",10240);
            gemv_bf16_ref(w[d],4*(int)cols[d],320,t[p],up[d]+(size_t)p*4*cols[d]);
            for(unsigned c=0;c<cols[d];c++) {
                float v=sigmoid_gpu(up[d][(size_t)p*4*cols[d]+c])*xh[(size_t)p*10240+first[d]+c];
                for(unsigned row=1;row<4;row++)v=fmaf(sigmoid_gpu(up[d][(size_t)p*4*cols[d]+row*cols[d]+c]),xh[(size_t)p*10240+row*2560+first[d]+c],v);
                mixed[(size_t)p*2560+first[d]+c]=.25f*v;
            }
        }
    }
    for(unsigned p=0;p<width;p++) {
        for(unsigned e=0;e<4;e++)extra[p][e]=guarded(e&1,"peer destination",10240);
        tables[p][0]=l[0][p].inbox;tables[p][1]=l[1][p].inbox;
        /* The other four peers need only the output span; bias their table
         * base so the actual indexed store still lands in guarded storage. */
        for(unsigned e=0;e<4;e++)tables[p][e+2]=extra[p][e]-slot*XSLOT;
        for(unsigned e=0;e<6;e++)tables[p][e+6]=flags[e&1];
        for(unsigned d=0;d<2;d++){CK(cuCtxSetCurrent(contexts[d]));CK(cuMemcpyHtoD(l[d][p].table,tables[p],96));}
    }
    prog_t *original=live_plans,*moved=live_plans+2;
    for(unsigned d=0;d<2;d++) {
        make_plan(original+d,l[d],width,weights[d],SITES,slot,tn,1,0,stream[d],reduced[d]);
        moved[d]=clone(original+d);
    }
    REQUIRE(prog_offload_hcup(moved,moved+1,l[0],l[1],width,period,32u*width,NULL)==(int)(SITES/period));
    int bad=0;size_t output_bytes=0;
    for(unsigned d=0;d<2;d++)output_bytes+=(size_t)width*(16*cols[d]+1296+2*XSLOT+40960+10240);
    output_bytes+=(size_t)width*4*10240;
    unsigned char *baseline=host_alloc(output_bytes,0),*cursor;
    for(unsigned mode=0;mode<2;mode++) {
        for(unsigned d=0;d<2;d++) {
            CK(cuCtxSetCurrent(contexts[d]));uint32_t control[64]={0},f[96];
            for(unsigned i=0;i<96;i++)f[i]=(i/16==4 || i/16==5)?0:UINT32_MAX;
            for(unsigned e=0;e<6;e++){uint64_t row=flags[e==5?1:0]+64u*(4+d);memcpy(control+16+2*e,&row,8);}
            uint64_t own=flags[d];memcpy(control+28,&own,8);control[3]=4+d;control[4]=6;control[9]=width;
            CK(cuMemsetD8(ctl[d],0,8192));CK(cuMemcpyHtoD(ctl[d],control,sizeof control));CK(cuMemcpyHtoD(flags[d],f,sizeof f));
            const prog_t *plan=(mode?moved:original)+d;CK(cuMemcpyHtoD(program[d],plan->p,(size_t)plan->n*sizeof(phase_t)));
            for(unsigned p=0;p<width;p++) {
                CK(cuMemsetD8(l[d][p].inbox,0,2*XSLOT));CK(cuMemcpyHtoD(l[d][p].inbox+(1u-slot)*XSLOT,down[p],1296));
                CK(cuMemsetD32(l[d][p].up,0x7fc5a5a5u,4*cols[d]));CK(cuMemsetD32(l[d][p].t,0x7fc5a5a5u,324));
                if(tn==320)CK(cuMemcpyHtoD(l[d][p].t+1280,down[p]+320,16));
                CK(cuMemsetD32(stream[d][p],0x7fc5a5a5u,10240));CK(cuMemsetD32(reduced[d][p],0x7fc5a5a5u,2560));
            }
        }
        for(unsigned p=0;p<width;p++)for(unsigned e=0;e<4;e++){CK(cuCtxSetCurrent(contexts[e&1]));CK(cuMemsetD8(extra[p][e],0,10240));}
        /* Both launches are asynchronous. Synchronizing either one before
         * launching its peer would deadlock the incoming exchange waits. */
        for(unsigned d=0;d<2;d++) {
            CK(cuCtxSetCurrent(contexts[d]));unsigned count=(unsigned)(mode?moved[d].n:original[d].n),end=width;
            void *args[]={program+d,&count,ctl+d,&end};CK(cuLaunchKernel(functions[d],16,1,1,256,1,1,0,0,args,NULL));launched[d]=1;
        }
        for(unsigned d=0;d<2;d++){CK(cuCtxSetCurrent(contexts[d]));CK(cuCtxSynchronize());launched[d]=0;}
        cursor=baseline;
        for(unsigned d=0;d<2;d++) {
            CK(cuCtxSetCurrent(contexts[d]));uint32_t control[64],f[96];CK(cuMemcpyDtoH(control,ctl[d],sizeof control));CK(cuMemcpyDtoH(f,flags[d],sizeof f));
            unsigned exchanges=4+!slot;bad+=control[2]!=width || control[8]!=exchanges || control[1]!=6u+!slot || control[11]!=0;
            for(unsigned i=0;i<96;i++)bad+=f[i]!=((i/16==4 || i/16==5)?exchanges:UINT32_MAX);
            const prog_t *plan=(mode?moved:original)+d;bad+=compare("immutable program",program[d],plan->p,(size_t)plan->n*sizeof(phase_t));
            bad+=compare("immutable weights",weights[d],w[d],(size_t)4*cols[d]*320*2);
            for(unsigned p=0;p<width;p++) {
                float inbox[2*XSLOT/4]={0},result[10240];memcpy((char *)inbox+(1u-slot)*XSLOT,down[p],1296);
                memcpy((char *)inbox+slot*XSLOT,mixed+(size_t)p*2560,10240);
                memcpy(result,xh+(size_t)p*10240,40960);hc_write_gpu(down[p]+320,mixed+(size_t)p*2560,result);
                bad+=compare("T CPU",l[d][p].t,t[p],1296)+compare("UP CPU",l[d][p].up,up[d]+(size_t)p*4*cols[d],16*cols[d]);
                bad+=compare("inbox CPU",l[d][p].inbox,inbox,2*XSLOT)+compare("HCWSUM stream CPU",stream[d][p],result,40960);
                bad+=compare("HCWSUM Y CPU",reduced[d][p],mixed+(size_t)p*2560,10240);
                bad+=compare("immutable XH",l[d][p].xh,xh+(size_t)p*10240,40960)+compare("immutable table",l[d][p].table,tables[p],96);
                CUdeviceptr ptr[]={l[d][p].up,l[d][p].t,l[d][p].inbox,stream[d][p],reduced[d][p]};
                size_t bytes[]={16*cols[d],1296,2*XSLOT,40960,10240};
                for(unsigned k=0;k<5;k++){if(mode)bad+=compare("offload vs original GPU",ptr[k],cursor,bytes[k]);else CK(cuMemcpyDtoH(cursor,ptr[k],bytes[k]));cursor+=bytes[k];}
            }
        }
        for(unsigned p=0;p<width;p++)for(unsigned e=0;e<4;e++) {
            CK(cuCtxSetCurrent(contexts[e&1]));bad+=compare("peer CPU",extra[p][e],mixed+(size_t)p*2560,10240);
            if(mode)bad+=compare("peer offload vs original GPU",extra[p][e],cursor,10240);else CK(cuMemcpyDtoH(cursor,extra[p][e],10240));cursor+=10240;
        }
        REQUIRE(cursor==baseline+output_bytes);bad+=guards();
    }
    fprintf(FIXTURE_OUT,"HCUP offload width=%u tn=%u slot=%u period=%u sites=2 selected=%u six_destinations=distinct contexts=2 cpu_and_GPU_bit_errors=%d guards=checked publications=checked HCWSUM_WAIT=checked\n",width,tn,slot,period,2/period,bad);
    for(unsigned d=0;d<2;d++){host_free(w[d]);host_free(up[d]);prog_free(original+d);prog_free(moved+d);}
    host_free(mixed);host_free(xh);host_free(baseline);free_allocations();memset(live_flags,0,sizeof live_flags);return bad;
}
#ifndef HCUP_OFFLOAD_BORROWED
int main(int argc,char **argv) {
    plan_checks();if(argc==2 && !strcmp(argv[1],"--cpu-only"))return 0;
    unsigned width=8;int device[2]={-1,-1};
    for(int i=1;i<argc;i++) {
        if(!strcmp(argv[i],"--devices") && i+2<argc){device[0]=atoi(argv[++i]);device[1]=atoi(argv[++i]);}
        else if(!strcmp(argv[i],"--width") && i+1<argc)width=(unsigned)atoi(argv[++i]);
        else {fprintf(FIXTURE_ERR,"usage: %s --cpu-only | --devices 0 1 [--width 1|2|8|16|32]\n",argv[0]);return 2;}
    }
    REQUIRE(device[0]>=0 && device[0]<=6 && device[1]>=0 && device[1]<=6 && device[0]!=device[1]);
    REQUIRE(width==1 || width==2 || width==8 || width==16 || width==32);
    CK(cuInit(0));CUdevice dev[2];CUmodule module[2];
    for(unsigned d=0;d<2;d++){CK(cuDeviceGet(dev+d,device[d]));CK(cuCtxCreate_v2(contexts+d,CU_CTX_SCHED_SPIN,dev[d]));}
    const char *cubin=getenv("FLASH_CUBIN");if(!cubin)cubin="die.cubin";
    for(unsigned d=0;d<2;d++) {
        int access=0;CK(cuDeviceCanAccessPeer(&access,dev[d],dev[1-d]));REQUIRE(access);
        CK(cuCtxSetCurrent(contexts[d]));CK(cuCtxEnablePeerAccess(contexts[1-d],0));
        CK(cuModuleLoad(module+d,cubin));CK(cuModuleGetFunction(functions+d,module[d],"die"));
        size_t free_bytes,total;CK(cuMemGetInfo(&free_bytes,&total));
        size_t need=memory_needed(width);
        fprintf(FIXTURE_OUT,"HCUP fixture physical=%d free=%zu estimated_data_need=%zu cubin=%s\n",device[d],free_bytes,need,cubin);REQUIRE(free_bytes>=need);
    }
    int bad=0;for(unsigned slot=0;slot<2;slot++)for(unsigned tn=320;tn<=324;tn+=4)for(unsigned period=1;period<=2;period++)bad+=gpu_case(width,slot,tn,period);
    for(unsigned d=0;d<2;d++){CK(cuCtxSetCurrent(contexts[d]));CK(cuModuleUnload(module[d]));CK(cuCtxDestroy(contexts[d]));}
    return bad?3:0;
}
#endif

#ifdef HCUP_OFFLOAD_BORROWED
/* Error recovery has an independent DMA stream and a pinned all-ready flag
 * row. If just one peer launch succeeds, release only the fixture's waits so
 * it can retire before scratch is freed. No serving flags/CTL are touched. */
static CUstream recovery_stream[2];
static uint32_t *recovery_flags;
static unsigned cleanup_error(CUresult result,const char *operation) {
    if(!result)return 0;
    const char *message="?";cuGetErrorString(result,&message);
    fprintf(FIXTURE_ERR,"hcup_offload_probe cleanup_error operation=%s cuda=%u message=%s\n",operation,(unsigned)result,message);
    return 1;
}
int hcup_offload_probe(FILE *out,const CUcontext *borrow_context,const CUfunction *borrow_function,
    unsigned dies,unsigned width,CUdeviceptr source_weights,CUdeviceptr helper_weights) {
    if(!out || !borrow_context || !borrow_function || dies!=6 ||
       (width!=1 && width!=2 && width!=8 && width!=16 && width!=32) ||
       !source_weights || !helper_weights || fixture_output) {
        if(out)fprintf(out,"hcup_offload_probe done status=FAIL result=-1 stage=arguments\n");
        return -1;
    }
    fixture_output=out;fixture_error=CUDA_SUCCESS;
    CUcontext previous=NULL;CUresult initial=cuCtxGetCurrent(&previous);
    if(initial) {
        fprintf(out,"hcup_offload_probe done status=FAIL result=-%u stage=get_current\n",(unsigned)initial);
        fixture_output=NULL;return -(int)initial;
    }
    int result=0;
    fprintf(out,"hcup_offload_probe begin width=%u source=4 helper=5 contexts=borrowed functions=borrowed weights=core_IPC_prefixes\n",width);fflush(out);
    if(setjmp(fixture_jb)) {result=-(int)fixture_error;goto cleanup;}
    REQUIRE(!nalloc && !nhost && !launched[0] && !launched[1]);
    for(unsigned d=0;d<dies;d++) {
        REQUIRE(borrow_context[d] && borrow_function[d]);CK(cuCtxSetCurrent(borrow_context[d]));CK(cuCtxSynchronize());
    }
    for(unsigned d=0;d<2;d++) {
        contexts[d]=borrow_context[d+4];functions[d]=borrow_function[d+4];borrowed_weights[d]=d?helper_weights:source_weights;
        CK(cuCtxSetCurrent(contexts[d]));size_t free_bytes,total;CK(cuMemGetInfo(&free_bytes,&total));
        size_t need=memory_needed(width);
        fprintf(out,"hcup_offload_probe die=%u free=%zu estimated_data_need=%zu capacities=unchanged\n",d+4,free_bytes,need);
        if(free_bytes<need)fixture_fail(CUDA_ERROR_OUT_OF_MEMORY,"guarded scratch preflight",__LINE__);
    }
    CK(cuMemHostAlloc((void **)&recovery_flags,96*4,CU_MEMHOSTALLOC_PORTABLE));
    for(unsigned i=0;i<96;i++)recovery_flags[i]=UINT32_MAX;
    for(unsigned d=0;d<2;d++){CK(cuCtxSetCurrent(contexts[d]));CK(cuStreamCreate(recovery_stream+d,CU_STREAM_NON_BLOCKING));}
    for(unsigned slot=0;slot<2;slot++)for(unsigned tn=320;tn<=324;tn+=4)for(unsigned period=1;period<=2;period++)
        result+=gpu_case(width,slot,tn,period);
cleanup:;
    unsigned cleanup_failures=0;
    if(launched[0] || launched[1]) {
        for(unsigned d=0;d<2;d++)if(live_flags[d] && recovery_flags && recovery_stream[d]) {
            cleanup_failures+=cleanup_error(cuCtxSetCurrent(contexts[d]),"recovery_context");
            cleanup_failures+=cleanup_error(cuMemcpyHtoDAsync(live_flags[d],recovery_flags,96*4,recovery_stream[d]),"release_fixture_waits");
        }
        for(unsigned d=0;d<2;d++)if(contexts[d]) {
            cleanup_failures+=cleanup_error(cuCtxSetCurrent(contexts[d]),"recovery_context");
            if(recovery_stream[d])cleanup_failures+=cleanup_error(cuStreamSynchronize(recovery_stream[d]),"recovery_DMA");
            cleanup_failures+=cleanup_error(cuCtxSynchronize(),"retire_fixture_kernel");launched[d]=0;
        }
    }
    while(nalloc) {
        allocation *a=allocations+--nalloc;
        cleanup_failures+=cleanup_error(cuCtxSetCurrent(a->ctx),"free_context");
        if(a->base)cleanup_failures+=cleanup_error(cuMemFree(a->base),"free_fixture_allocation");
        memset(a,0,sizeof *a);
    }
    while(nhost)free(host_scratch[--nhost]);
    for(unsigned i=0;i<4;i++)prog_free(live_plans+i);
    for(unsigned i=0;i<32;i++)prog_free(build_temp+i);
    for(unsigned d=0;d<2;d++)if(recovery_stream[d]) {
        cleanup_failures+=cleanup_error(cuCtxSetCurrent(contexts[d]),"stream_context");
        cleanup_failures+=cleanup_error(cuStreamDestroy(recovery_stream[d]),"free_fixture_stream");recovery_stream[d]=NULL;
    }
    if(recovery_flags){cleanup_failures+=cleanup_error(cuMemFreeHost(recovery_flags),"free_recovery_flags");recovery_flags=NULL;}
    cleanup_failures+=cleanup_error(cuCtxSetCurrent(previous),"restore_current_context");
    memset(contexts,0,sizeof contexts);memset(functions,0,sizeof functions);memset(borrowed_weights,0,sizeof borrowed_weights);memset(live_flags,0,sizeof live_flags);
    if(cleanup_failures && !result)result=-CUDA_ERROR_UNKNOWN;
    fprintf(out,"hcup_offload_probe done status=%s result=%d cleanup_failures=%u allocations_live=%u host_live=%u contexts=preserved modules=preserved\n",
            result?"FAIL":"PASS",result,cleanup_failures,nalloc,nhost);fflush(out);fixture_output=NULL;return result;
}
#endif
