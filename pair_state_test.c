/* Compare scalar-position order with paired-phase order on real GPU kernels. */
#include "prog.h"
#ifdef FLASH_EXACT_STATE
#include "exactbatch/prog_group.h"
#define POSITIONS 8u
#else
#define POSITIONS 2u
#endif
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
CUresult cuCtxCreate_v2(CUcontext *,unsigned,CUdevice);
#define CK(x) do { CUresult check_result=(x); if(check_result!=CUDA_SUCCESS){const char*s="?";cuGetErrorString(check_result,&s);fprintf(stderr,"%s:%d %s: %s\n",__FILE__,__LINE__,#x,s);exit(2);} }while(0)
#define N 10240u
#define H 48u
#define O (H*128u)
static unsigned seed=173;
static int mixed_warp;
static float rnd(void){seed=seed*1664525u+1013904223u;return (float)((int)(seed>>8)-8388608)/8388608.f;}
static CUdeviceptr alloc(size_t n){CUdeviceptr p;CK(cuMemAlloc(&p,n));CK(cuMemsetD8(p,0,n));return p;}
static CUdeviceptr upload(const void*x,size_t n){CUdeviceptr p=alloc(n);CK(cuMemcpyHtoD(p,x,n));return p;}
typedef struct {CUdeviceptr x,u,a,b,o,q,s32,s16,as,am,part,cnt;} buffers;
typedef struct {const char*name;CUdeviceptr p;size_t n;void*ref;int fused_only;} region;
static void program(prog_t*p,const buffers*b,unsigned off,CUdeviceptr cs,CUdeviceptr ds,CUdeviceptr cw,CUdeviceptr sa){
    p->loop=1;p->token_offset=off;
    ph_conv1d(p,b->x,cs,cw,b->u,0,N);
    ph_dnstep(p,ds,b->u,b->u+2048u*4,b->u+4096u*4,b->a,b->b,sa,b->o,1e-6f,0,0,H);
    ph_actq(p,mixed_warp?OP_ACTQ2560:OP_ACTQ256,b->o,b->q,b->s32,b->s16,b->as,mixed_warp?2560:256,mixed_warp?1:O/256);
    phase_t*d=prog_add(p,OP_ARGMAX,16);
    ph_ptr(d,0x140,b->o);ph_ptr(d,0x148,b->am);ph_ptr(d,0x150,b->part);ph_ptr(d,0x158,b->cnt);
    ph_u32(d,0x160,O);ph_u32(d,0x164,16);
}
static int launch(CUfunction fn,CUdeviceptr ctl,const prog_t*p,unsigned start,unsigned expect_gen){
    unsigned hc[2048]={0};hc[2]=start;hc[9]=POSITIONS;CK(cuMemcpyHtoD(ctl,hc,sizeof hc));
    CUdeviceptr dp=upload(p->p,(size_t)p->n*sizeof(phase_t));unsigned nph=(unsigned)p->n,ntok=start+POSITIONS;
    void*args[]={&dp,&nph,&ctl,&ntok};CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(hc,ctl,sizeof hc));CK(cuMemFree(dp));
    if(hc[2]!=ntok||hc[1]!=expect_gen){fprintf(stderr,"control at %u: TOK=%u expected%u GEN=%u expected%u\n",start,hc[2],ntok,hc[1],expect_gen);return 1;}return 0;
}
static unsigned logical_phases(const prog_t *p) {
    unsigned phases=0;
    for(int i=0;i<p->n;phases++) {
#ifdef FLASH_EXACT_STATE
        i+=(int)phase_descriptors(p->p+i);
#else
        i+=(p->p[i].w[30]&8u)?2:1;
#endif
    }
    return phases;
}
#ifdef FLASH_VERIFY_STATE
static int copy_bounds(CUfunction fn,CUdeviceptr ctl) {
    const size_t sizes[]={16,2032,2048,4096,4112,65552};
    size_t cap=65552+128;uint8_t *src=malloc(cap),*got=malloc(cap);int bad=0;
    for(size_t i=0;i<cap;i++)src[i]=(uint8_t)(i*71u+19u);
    CUdeviceptr ds=upload(src,cap),dd=alloc(cap);
    for(unsigned i=0;i<sizeof sizes/sizeof sizes[0];i++) {
        CK(cuMemsetD8(dd,0xa5,cap));prog_t p;prog_init(&p);
        if(prog_verify_copy(&p,ds+64,dd+64,sizes[i]))exit(2);
        bad+=launch(fn,ctl,&p,0,1);CK(cuMemcpyDtoH(got,dd,cap));
        unsigned errors=memcmp(got+64,src+64,sizes[i])!=0;
        for(size_t j=0;j<64;j++)errors+=got[j]!=0xa5;
        for(size_t j=64+sizes[i];j<cap;j++)errors+=got[j]!=0xa5;
        printf("verify copy bytes=%zu errors=%u\n",sizes[i],errors);bad+=errors!=0;prog_free(&p);
    }
    CK(cuMemFree(ds));CK(cuMemFree(dd));free(src);free(got);return bad;
}
#endif
int main(void){
    setvbuf(stdout,NULL,_IONBF,0);
    mixed_warp=getenv("PAIR_MIXED_WARP")!=NULL;
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;
    CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char*cubin=getenv("FLASH_CUBIN");CK(cuModuleLoad(&mod,cubin?cubin:"die.cubin"));CK(cuModuleGetFunction(&fn,mod,"die"));
#ifdef FLASH_VERIFY_STATE
    CUfunction marker;CK(cuModuleGetFunction(&marker,mod,"flash_verify_checkpoints_abi_v1"));
#endif
    int resident;CK(cuOccupancyMaxActiveBlocksPerMultiprocessor(&resident,fn,256,0));if(resident<1){fputs("no resident CTA\n",stderr);return 2;}
    size_t cn=4u*4u*N,dn=4u*H*128u*128u;
    float*cinit=malloc(cn),*dinit=malloc(dn),*cw=malloc(4u*4u*N);float sa[H];
    for(unsigned i=0;i<4*N;i++){cinit[i]=0.1f*rnd();cw[i]=0.2f*rnd();}
    for(unsigned i=0;i<H*128u*128u;i++)dinit[i]=0.01f*rnd();
    for(unsigned i=0;i<H;i++)sa[i]=-0.5f-0.1f*(float)i/H;
    CUdeviceptr cs=upload(cinit,cn),ds=upload(dinit,dn),dCW=upload(cw,4u*4u*N),dSA=upload(sa,sizeof sa),ctl=alloc(8192);
    buffers b[POSITIONS];region r[10*POSITIONS+2];int nr=0;
#define REGION(nm,ptr,len) do{r[nr++]=(region){nm,ptr,len,malloc(len),0};}while(0)
    REGION("conv state",cs,cn);REGION("DeltaNet state",ds,dn);
    for(unsigned k=0;k<POSITIONS;k++){
        float x[N],a[H],bb[H];for(unsigned i=0;i<N;i++)x[i]=rnd()*(float)(k+1);
        for(unsigned i=0;i<H;i++){a[i]=rnd();bb[i]=rnd();}
        b[k]=(buffers){.x=upload(x,sizeof x),.u=alloc(4*N),.a=upload(a,sizeof a),.b=upload(bb,sizeof bb),
            .o=alloc(4*O),.q=alloc(2*O),.s32=alloc(4*O/32),.s16=alloc(4*O/16),.as=alloc(4*O/32),
            .am=alloc(8),.part=alloc(16*8),.cnt=alloc(4)};
        REGION("conv output",b[k].u,4*N);REGION("DeltaNet output",b[k].o,4*O);REGION("quant codes",b[k].q,2*O);
        REGION("sum32",b[k].s32,4*O/32);REGION("sum16",b[k].s16,4*O/16);REGION("quant scales",b[k].as,4*O/32);
        REGION("argmax",b[k].am,8);REGION("argmax counter",b[k].cnt,4);
    }
    prog_t pos[POSITIONS],pair,serial;prog_init(&pair);prog_init(&serial);
    for(unsigned k=0;k<POSITIONS;k++) {
        prog_init(&pos[k]);program(&pos[k],&b[k],k,cs,ds,dCW,dSA);
#ifdef FLASH_VERIFY_STATE
        CUdeviceptr conv_checkpoint=alloc(cn+128);
        REGION("conv checkpoint",conv_checkpoint,cn+128);
        if(prog_verify_copy(&pos[k],cs,conv_checkpoint+64,cn))exit(2);
        phase_t copy=pos[k].p[pos[k].n-1];
        memmove(pos[k].p+2,pos[k].p+1,(size_t)(pos[k].n-2)*sizeof(phase_t));pos[k].p[1]=copy;
        CUdeviceptr checkpoint=alloc(dn+128);
        REGION("Delta checkpoint",checkpoint,dn+128);r[nr-1].fused_only=1;
        /* The scalar oracle copies each updated state before the next position.
         * Fused DNSTEP8 captures directly from its live state registers. */
        for(int i=0;i<pos[k].n;i++)if(pos[k].p[i].w[0]==OP_DNSTEP)ph_ptr(&pos[k].p[i],0x198,checkpoint+64);
#endif
        for(int i=0;i<pos[k].n;i++) {
            *prog_add(&serial,pos[k].p[i].w[0],pos[k].p[i].w[1])=pos[k].p[i];
#ifdef FLASH_VERIFY_STATE
            if(pos[k].p[i].w[0]==OP_DNSTEP) {
                phase_t *copy=prog_add(&serial,OP_XCOPY,(unsigned)(dn/4096));
                ph_ptr(copy,0x140,ds);ph_ptr(copy,0x148,checkpoint+64);
            }
#endif
        }
    }
#ifdef FLASH_EXACT_STATE
    const prog_t *group[POSITIONS];for(unsigned k=0;k<POSITIONS;k++)group[k]=&pos[k];
    if(prog_group(&pair,group,POSITIONS)){fputs("prog_group rejected valid programs\n",stderr);return 2;}
    prog_t fused;prog_init(&fused);
    if(prog_group(&fused,group,POSITIONS)){fputs("prog_group rejected fused input\n",stderr);return 2;}
    int dn_fused=prog_batch_fuse_dnstep(&fused),act_fused=prog_batch_fuse_actq(&fused);
    if(dn_fused!=1 || act_fused!=(mixed_warp?0:1)){fprintf(stderr,"wrong fuse counts dn=%d act=%d\n",dn_fused,act_fused);return 2;}
    const int modes=3;
#else
    if(prog_pair(&pair,&pos[0],&pos[1])){fputs("prog_pair rejected valid programs\n",stderr);return 2;}
    const int modes=2;
#endif
    const unsigned starts[]={0,1,2,3,4,7,8,9,127,128,2047,2048};int fails=0;unsigned cases=sizeof starts/sizeof starts[0];
    void*got=malloc((dn>cn?dn:cn)+128);
    for(unsigned c=0;c<cases;c++){
        unsigned start=starts[c];int bad=0;
        for(int paired=0;paired<modes;paired++){
            CK(cuMemcpyHtoD(cs,cinit,cn));CK(cuMemcpyHtoD(ds,dinit,dn));
            for(int i=2;i<nr;i++)CK(cuMemsetD8(r[i].p,0xa5,r[i].n));
            for(unsigned k=0;k<POSITIONS;k++){CK(cuMemsetD8(b[k].part,0,16*8));CK(cuMemsetD32(b[k].cnt,0,1));}
            const prog_t *selected=paired?&pair:&serial;
#ifdef FLASH_EXACT_STATE
            if(paired==2) selected=&fused;
#endif
            bad+=launch(fn,ctl,selected,start,logical_phases(selected));
            for(int i=0;i<nr;i++){
                if(!paired)CK(cuMemcpyDtoH(r[i].ref,r[i].p,r[i].n));
                else if(!(paired==1&&r[i].fused_only)){CK(cuMemcpyDtoH(got,r[i].p,r[i].n));if(memcmp(got,r[i].ref,r[i].n)){fprintf(stderr,"start%u region%d %s mismatch\n",start,i,r[i].name);bad++;}}
            }
            for(unsigned k=0;k<POSITIONS;k++){
                float oo[O];unsigned am[2],cnt;CK(cuMemcpyDtoH(oo,b[k].o,sizeof oo));CK(cuMemcpyDtoH(am,b[k].am,8));CK(cuMemcpyDtoH(&cnt,b[k].cnt,4));
                unsigned best=0;for(unsigned i=1;i<O;i++)if(oo[i]>oo[best])best=i;
                if(cnt||am[0]!=best||memcmp(&am[1],&oo[best],4)){fprintf(stderr,"argmax start%u pos%d invalid\n",start,k);bad++;}
            }
        }
        printf("start=%u width=%u modes=%d state/outputs=%s scalar_barriers=%u group_barriers=%u\n",start,POSITIONS,modes,bad?"FAIL":"BIT-EXACT",logical_phases(&serial),logical_phases(&pair));fails+=bad;
    }
    printf("pair_state_test: %u cases, %d failures\n",cases,fails);
#ifdef FLASH_VERIFY_STATE
    fails+=copy_bounds(fn,ctl);
#endif
    for(unsigned k=0;k<POSITIONS;k++)prog_free(&pos[k]);
    prog_free(&pair);prog_free(&serial);
#ifdef FLASH_EXACT_STATE
    prog_free(&fused);
#endif
    for(int i=0;i<nr;i++)free(r[i].ref);
    free(cinit);free(dinit);free(cw);free(got);
    CK(cuCtxDestroy(ctx));return fails?1:0;
}
