/* Fixed-input old/new ATTNPARTN comparison through the real persistent dispatcher.
 * Usage: attnpartn_regress old.cubin new.cubin keys [seed] [repeats]
 * ATTN_TRACE_CHECK=1 requires both cubins built with DIE_CTA_TRACE=1.
 * All input and padding values are finite; empty chunks use the -1e30 sentinel.
 * GPU events cover 64 positions per launch. Guards and copies are outside events.
 */
#define _POSIX_C_SOURCE 200809L
#include "attn.h"
#include "dieops.h"
#include "cta_trace.h"
#include <cuda.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult r=(x); if(r) {const char *e="?";cuGetErrorString(r,&e);fprintf(stderr,"%s: %s\n",#x,e);exit(2);} } while(0)
enum { GUARD_WORDS=64, TRACE_TOKENS=2, PHASES=2, CTL_BYTES=8192 };
typedef struct {uint32_t w[32];} desc;
typedef struct {CUdeviceptr base, data; size_t words; unsigned id;} guarded;
typedef struct {
    CUdeviceptr device;
    cta_trace_record *host;
    size_t bytes;
    unsigned enabled, nonce, first, count, launches;
    uint64_t previous_end, records;
} trace_state;
static void *xmalloc(size_t n){void *p=malloc(n);if(!p){perror("malloc");exit(2);}return p;}
static void *xcalloc(size_t n,size_t s){void *p=calloc(n,s);if(!p){perror("calloc");exit(2);}return p;}
static void ptr(desc *p,int a,CUdeviceptr v){int w=2+(a-0x140)/4;p->w[w]=(uint32_t)v;p->w[w+1]=(uint32_t)(v>>32);}
static CUdeviceptr alloc(size_t n){CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemsetD8(p,0,n));return p;}
static uint32_t guard_word(unsigned id,unsigned side,unsigned i){return 0x13579bdfu^(0x9e3779b9u*(id+1))^(0x6a09e667u*(side+1))^(0x85ebca6bu*(i+1));}
static void guard_pattern(uint32_t *p,unsigned id,unsigned side){for(unsigned i=0;i<GUARD_WORDS;i++)p[i]=guard_word(id,side,i);}
static int guard_host_ok(const uint32_t *p,unsigned id,unsigned side){for(unsigned i=0;i<GUARD_WORDS;i++)if(p[i]!=guard_word(id,side,i))return 0;return 1;}
static guarded guard_alloc(size_t words,unsigned id){guarded g={0};g.words=words;g.id=id;g.base=alloc((words+2*GUARD_WORDS)*sizeof(uint32_t));g.data=g.base+GUARD_WORDS*sizeof(uint32_t);return g;}
static void guard_reset(const guarded *g){
    uint32_t p[GUARD_WORDS];guard_pattern(p,g->id,0);CK(cuMemcpyHtoD(g->base,p,sizeof p));
    guard_pattern(p,g->id,1);CK(cuMemcpyHtoD(g->data+g->words*sizeof(uint32_t),p,sizeof p));
    CK(cuMemsetD32(g->data,0x7fc00000u,g->words));
}
static int guard_check(const guarded *g,const char *name,int variant,unsigned launch){
    uint32_t p[GUARD_WORDS];int bad=0;
    for(unsigned side=0;side<2;side++){
        CUdeviceptr at=side?g->data+g->words*sizeof(uint32_t):g->base;
        CK(cuMemcpyDtoH(p,at,sizeof p));
        if(!guard_host_ok(p,g->id,side)){
            fprintf(stderr,"boundary canary overwritten: %s side=%s variant=%d launch=%u\n",name,side?"trailing":"leading",variant,launch);bad=1;
        }
    }
    return bad;
}
static uint32_t rng;
static float rnd(void){rng=rng*1664525u+1013904223u;return ((rng>>8)&65535)/32768.f-1.f;}
static int cmp(const char *what,const float *a,const float *b,size_t n){
    size_t bad=0;for(size_t i=0;i<n;i++)if(memcmp(a+i,b+i,sizeof(float))){if(!bad)fprintf(stderr,"%s[%zu] %.9g / %.9g\n",what,i,a[i],b[i]);bad++;}
    return bad!=0;
}
static int order(const void *a,const void *b){float x=*(const float*)a,y=*(const float*)b;return(x>y)-(x<y);}
static void stat(const char *name,float *s,int n){float *c=xmalloc((size_t)n*sizeof *c);memcpy(c,s,(size_t)n*sizeof *c);qsort(c,(size_t)n,sizeof *c,order);printf("\"%s\":{\"median_us\":%.3f,\"p95_us\":%.3f,\"samples_us\":[",name,(n&1)?c[n/2]:.5f*(c[n/2-1]+c[n/2]),c[(int)ceil(.95*n)-1]);for(int i=0;i<n;i++)printf("%s%.3f",i?",":"",s[i]);printf("]}");free(c);}
static unsigned number(const char *s,unsigned max,const char *name){
    char *end;errno=0;unsigned long v=strtoul(s,&end,10);
    if(errno||!*s||*end||*s=='-'||v>max){fprintf(stderr,"invalid %s: %s\n",name,s);exit(2);}return (unsigned)v;
}
static int trace_requested(void){
    const char *check=getenv("ATTN_TRACE_CHECK"),*ctas=getenv("ATTN_TRACE_CTAS");int on=0;
    if(check){if(strcmp(check,"0")&&strcmp(check,"1")){fputs("ATTN_TRACE_CHECK must be 0 or 1\n",stderr);exit(2);}on=!strcmp(check,"1");}
    if(ctas){if(strcmp(ctas,"0")&&strcmp(ctas,"16")){fputs("ATTN_TRACE_CTAS must be 0 or 16\n",stderr);exit(2);}on|=!strcmp(ctas,"16");}
    return on;
}
static void launch_reset(CUdeviceptr control,guarded output[4],trace_state *tr,unsigned tokens){
    CK(cuMemsetD8(control,0,CTL_BYTES));
    for(unsigned a=0;a<4;a++)guard_reset(output+a);
    if(tr->enabled){
        /* Clearing rejects absent instrumentation even if the previous variant wrote
         * a valid-looking record. Nonce and absolute time independently reject replay. */
        CK(cuMemsetD8(tr->device,0,tr->bytes));
        if(++tr->nonce==0){fputs("trace launch nonce overflow\n",stderr);exit(2);}
        tr->count=tokens<TRACE_TOKENS?tokens:TRACE_TOKENS;tr->first=tokens-tr->count;
        uint32_t meta[6]={(uint32_t)tr->device,(uint32_t)(tr->device>>32),tr->first,tr->count,CTA_TRACE_CTAS,tr->nonce};
        CK(cuMemcpyHtoD(control+0xd0,meta,sizeof meta));
    }
}
static int trace_check(trace_state *tr,unsigned phases,unsigned keys,int variant,int emit){
    if(!tr->enabled)return 0;
    CK(cuMemcpyDtoH(tr->host,tr->device,tr->bytes));
    uint64_t greatest=tr->previous_end;size_t index=0;
    for(unsigned t=0;t<tr->count;t++)for(unsigned phase=0;phase<phases;phase++)for(unsigned cta=0;cta<CTA_TRACE_CTAS;cta++,index++){
        const cta_trace_record *r=tr->host+index;
        unsigned op=phase?OP_ATTNMERGE:OP_ATTNPARTN;
        unsigned units=phase?0:cta_attnpartn_units(keys,cta,variant);
        const char *error=cta_trace_error(r,tr->first+t,phase,cta,op,tr->nonce,tr->previous_end,units);
        if(error){fprintf(stderr,"trace rejected variant=%d launch=%u record=%zu: %s (token=%u phase=%u cta=%u op=%u units=%u nonce=%u)\n",variant,tr->nonce,index,error,r->token,r->phase,r->cta,r->opcode,r->units,r->nonce);return 1;}
        if(index>=CTA_TRACE_CTAS && tr->host[index-CTA_TRACE_CTAS].end>r->wait_start){
            fputs("trace rejected: phase/token order within a physical CTA\n",stderr);return 1;
        }
        if(r->end>greatest)greatest=r->end;
        if(emit)fprintf(stderr,"TRACE variant=%d launch=%u token=%u phase=%u cta=%u opcode=%u nonempty_units=%u nonce=%u wait_start=%llu wait_end=%llu body_start=%llu body_end=%llu end=%llu exchange_wait_ns=%llu dispatch_ns=%llu body_ns=%llu grid_wait_ns=%llu\n",variant,tr->nonce,r->token,r->phase,r->cta,r->opcode,r->units,r->nonce,(unsigned long long)r->wait_start,(unsigned long long)r->wait_end,(unsigned long long)r->body_start,(unsigned long long)r->body_end,(unsigned long long)r->end,(unsigned long long)(r->wait_end-r->wait_start),(unsigned long long)(r->body_start-r->wait_end),(unsigned long long)(r->body_end-r->body_start),(unsigned long long)(r->end-r->body_end));
    }
    /* A one-phase program must not write the cleared capacity reserved for phase2. */
    const unsigned char *tail=(const unsigned char *)(tr->host+index);
    for(size_t i=index*sizeof(cta_trace_record);i<tr->bytes;i++,tail++)if(*tail){fprintf(stderr,"trace wrote outside selected token/phase range\n");return 1;}
    tr->previous_end=greatest;tr->records+=index;tr->launches++;return 0;
}
static int check_outputs(guarded output[4],const char *const names[4],float *const ref[4],float *const got[4],unsigned phases,int variant,unsigned launch){
    int bad=0;for(unsigned a=0;a<4;a++){
        bad|=guard_check(output+a,names[a],variant,launch);
        if(a<3||phases==2){CK(cuMemcpyDtoH(got[a],output[a].data,output[a].words*sizeof(float)));bad|=cmp(names[a],ref[a],got[a],output[a].words);}
    }return bad;
}
static int self_test(void){
    unsigned canary_negative=0,trace_negative=0;uint32_t p[GUARD_WORDS];
    for(unsigned a=0;a<4;a++)for(unsigned side=0;side<2;side++){
        guard_pattern(p,a,side);if(!guard_host_ok(p,a,side))return 1;
        p[0]^=1;if(guard_host_ok(p,a,side))return 1;canary_negative++;
        guard_pattern(p,a,side);p[GUARD_WORDS-1]^=1;if(guard_host_ok(p,a,side))return 1;canary_negative++;
    }
    cta_trace_record valid={100,110,120,130,140,9,3,6,OP_ATTNPARTN,2,23},r;
#define VALID(z,lower) cta_trace_valid(&(z),9,3,6,OP_ATTNPARTN,23,(lower),2)
#define REJECT(z,lower) do{if(VALID(z,lower)){fputs("negative trace self-test falsely passed\n",stderr);return 1;}trace_negative++;}while(0)
    if(!VALID(valid,99))return 1;
    memset(&r,0,sizeof r);REJECT(r,0); /* cleared, uninstrumented launch */
    r=valid;r.nonce++;REJECT(r,0); /* old launch replay */
    r=valid;r.token++;REJECT(r,0);
    r=valid;r.phase++;REJECT(r,0);
    r=valid;r.cta++;REJECT(r,0);
    r=valid;r.opcode++;REJECT(r,0);
    r=valid;r.units++;REJECT(r,0);
    REJECT(valid,140); /* metadata updated but previous launch timestamps */
    r=valid;r.wait_start=0;REJECT(r,0);
    r=valid;r.wait_end=99;REJECT(r,0);
    r=valid;r.body_start=109;REJECT(r,0);
    r=valid;r.body_end=119;REJECT(r,0);
    r=valid;r.end=129;REJECT(r,0);
    r=valid;r.wait_start=r.wait_end=r.body_start=r.body_end=r.end=100;REJECT(r,0);
#undef REJECT
#undef VALID
    for(unsigned cta=0;cta<16;cta++){
        unsigned old2048=(cta==0||cta==1||cta==8||cta==9)?2:0;
        unsigned old2051=(cta==0||cta==1||cta==2||cta==8||cta==9||cta==10)?2:0;
        if(cta_attnpartn_units(2048,cta,0)!=old2048||cta_attnpartn_units(2048,cta,1)!=(cta<8?1u:0u)||
           cta_attnpartn_units(2051,cta,0)!=old2051||cta_attnpartn_units(2051,cta,1)!=(cta<12?1u:0u))return 1;
    }
    printf("{\"self_test\":\"pass\",\"cpu_only\":true,\"negative_canary_cases\":%u,\"negative_trace_cases\":%u,\"mapping_cases\":64}\n",canary_negative,trace_negative);return 0;
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--self-test"))return self_test();
    if(argc<4||argc>6){fputs("usage: attnpartn_regress old.cubin new.cubin keys [seed] [repeats]\n       attnpartn_regress --self-test\n",stderr);return 2;}
    unsigned n=number(argv[3],16384,"keys");if(!n)return 2;
    unsigned seed=argc>4?number(argv[4],UINT32_MAX,"seed"):77;
    unsigned rep=argc>5?number(argv[5],10000,"repeats"):32;if(rep<4||(rep&1))return 2;rng=seed;
    int tracing=trace_requested();
    unsigned npos=(n+8192+64+31)/32*32;
    size_t kc_n=ATTN_KC_FLOATS(npos),vc_n=(size_t)npos*512;
    float *q=xmalloc(6144*sizeof(float)),*g=xmalloc(12288*sizeof(float)),*k=xcalloc(kc_n,sizeof(float)),*v=xcalloc(vc_n,sizeof(float));
    for(int i=0;i<6144;i++)q[i]=seed?2*rnd():0.f;
    for(int i=0;i<12288;i++)g[i]=4*rnd();
    for(unsigned t=0;t<n;t++)for(int h=0;h<2;h++)for(int j=0;j<256;j++){k[attn_kidx(t,h,j)]=2*rnd();v[(size_t)t*512+256*h+j]=rnd();}
    const size_t sz[4]={192,192,192*256,6144};
    const char *const names[4]={"maxima","denominators","weighted_values","outputs"};
    float *ref[4],*got[2][4];
    for(int a=0;a<4;a++){ref[a]=xmalloc(sz[a]*sizeof(float));for(int j=0;j<2;j++)got[j][a]=xmalloc(sz[a]*sizeof(float));}
    attn_partials_gpu(q,k,v,n,ref[0],ref[1],ref[2]);attn_merge_gpu(ref[0],ref[1],ref[2],g,ref[3]);
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    char device[256];CK(cuDeviceGetName(device,sizeof device,dev));
    CUmodule mod[2];CUfunction fn[2];for(int j=0;j<2;j++){CK(cuModuleLoad(&mod[j],argv[j+1]));CK(cuModuleGetFunction(&fn[j],mod[j],"die"));}
    CUdeviceptr dq=alloc(6144*sizeof(float)),dg=alloc(12288*sizeof(float)),dk=alloc(kc_n*sizeof(float)),dv=alloc(vc_n*sizeof(float)),dp=alloc(256),dc=alloc(CTL_BYTES);
    guarded output[4];for(int a=0;a<4;a++)output[a]=guard_alloc(sz[a],(unsigned)a);
    CK(cuMemcpyHtoD(dq,q,6144*sizeof(float)));CK(cuMemcpyHtoD(dg,g,12288*sizeof(float)));CK(cuMemcpyHtoD(dk,k,kc_n*sizeof(float)));CK(cuMemcpyHtoD(dv,v,vc_n*sizeof(float)));
    desc p[2]={0};p[0].w[0]=OP_ATTNPARTN;p[0].w[1]=32;
    ptr(&p[0],0x140,dq);ptr(&p[0],0x148,dk);ptr(&p[0],0x150,dv);ptr(&p[0],0x158,output[0].data);ptr(&p[0],0x160,output[1].data);ptr(&p[0],0x168,output[2].data);p[0].w[14]=n;p[0].w[15]=(n+3)/4;
    p[1].w[0]=OP_ATTNMERGE;p[1].w[1]=24;ptr(&p[1],0x140,output[0].data);ptr(&p[1],0x148,output[1].data);ptr(&p[1],0x150,output[2].data);ptr(&p[1],0x158,dg);ptr(&p[1],0x160,output[3].data);
    trace_state tr={0};tr.enabled=(unsigned)tracing;tr.bytes=TRACE_TOKENS*PHASES*CTA_TRACE_CTAS*sizeof(cta_trace_record);
    if(tr.enabled){tr.device=alloc(tr.bytes);tr.host=xmalloc(tr.bytes);}
    CK(cuMemcpyHtoD(dp,p,sizeof p));unsigned nph=2,ntok=2;void *args[]={&dp,&nph,&dc,&ntok};unsigned launches=0;int bad=0;
    /* ABBA launches prove each variant writes a fresh record, twice, across tokens. */
    const int variants[4]={0,1,1,0};
    for(unsigned iteration=0;iteration<4;iteration++){
        int j=variants[iteration];launch_reset(dc,output,&tr,ntok);launches++;
        CK(cuLaunchKernel(fn[j],16,1,1,256,1,1,0,0,args,0));CK(cuCtxSynchronize());
        uint32_t count;CK(cuMemcpyDtoH(&count,dc+8,sizeof count));if(count!=ntok){fprintf(stderr,"token count %u expected %u\n",count,ntok);return 1;}
        bad|=check_outputs(output,names,ref,got[j],nph,j,launches);bad|=trace_check(&tr,nph,n,j,1);
        if(bad)return 1;
    }
    for(int a=0;a<4;a++)bad|=cmp(names[a],got[0][a],got[1][a],sz[a]);
    if(bad){fprintf(stderr,"FAIL keys=%u seed=%u\n",n,seed);return 1;}
    CUevent start,end;CK(cuEventCreate(&start,0));CK(cuEventCreate(&end,0));
    float *total[2],*part[2],*merge[2];for(int j=0;j<2;j++){total[j]=xmalloc(rep*sizeof(float));part[j]=xmalloc(rep*sizeof(float));merge[j]=xmalloc(rep*sizeof(float));}
    /* Repeated positions keep the dispatcher warm. Alternate balanced variant order.
     * Reset/check every allocation around every launch, outside measured events. */
    const unsigned steps=64;
    for(int it=-4;it<(int)rep;it++)for(int jj=0;jj<2;jj++){
        int j=(it&1)?jj:1-jj;
        for(unsigned phases=1;phases<=2;phases++){
            nph=phases;ntok=steps;launch_reset(dc,output,&tr,ntok);launches++;
            CK(cuEventRecord(start,0));CK(cuLaunchKernel(fn[j],16,1,1,256,1,1,0,0,args,0));CK(cuEventRecord(end,0));CK(cuEventSynchronize(end));
            float ms;CK(cuEventElapsedTime(&ms,start,end));
            if(!isfinite(ms)||ms<=0.f||ms>=1000.f){fputs("invalid timing window for 32-bit phase deltas\n",stderr);return 1;}
            uint32_t count;CK(cuMemcpyDtoH(&count,dc+8,sizeof count));if(count!=steps){fprintf(stderr,"timed token count %u expected %u\n",count,steps);return 1;}
            bad=check_outputs(output,names,ref,got[j],phases,j,launches)|trace_check(&tr,phases,n,j,0);if(bad)return 1;
            if(it>=0){if(phases==1)part[j][it]=1000*ms/steps;else {uint32_t ts[2];CK(cuMemcpyDtoH(ts,dc+0x100,sizeof ts));total[j][it]=1000*ms/steps;merge[j][it]=(uint32_t)(ts[1]-ts[0])*.001f;}}
        }
    }
    printf("{\"keys\":%u,\"seed\":%u,\"device\":\"%s\",\"bit_exact\":true,\"boundary_canaries\":true,\"boundary_canary_words\":%u,\"guard_checked_launches\":%u,\"trace_verified\":%s,\"trace_launches\":%u,\"trace_records\":%llu,\"timing_trace_enabled\":%s,\"steps_per_sample\":64,\"partial_values\":%zu,\"final_values\":6144,\"old\":{",n,seed,device,GUARD_WORDS,launches,tr.enabled?"true":"false",tr.launches,(unsigned long long)tr.records,tr.enabled?"true":"false",sz[0]+sz[1]+sz[2]);
    stat("event",total[0],(int)rep);printf(",");stat("partial_program",part[0],(int)rep);printf(",");stat("merge",merge[0],(int)rep);printf("},\"new\":{");stat("event",total[1],(int)rep);printf(",");stat("partial_program",part[1],(int)rep);printf(",");stat("merge",merge[1],(int)rep);puts("}}");
    CK(cuEventDestroy(start));CK(cuEventDestroy(end));CK(cuCtxDestroy(ctx));return 0;
}
