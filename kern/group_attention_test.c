/* B8 QSA schedule parity against scalar-position order on the same kernels.
 * Synthetic postprojection inputs isolate cache/indexer/attention scheduling.
 * Mirrors engine.c: per-position IQ/score/selection, shared ring/pool/KV,
 * and compact/partial scratch shared by positions 0 and 2..7.
 * Run only under gpuq. This fixture does not measure model throughput. */
#define _POSIX_C_SOURCE 200809L
#include "prog.h"
#include "qsa.h"
#include "attn.h"
#include <cuda.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

CUresult cuCtxCreate_v2(CUcontext *, unsigned, CUdevice);
#define CK(x) do { CUresult r_ = (x); if (r_ != CUDA_SUCCESS) { \
    const char *s_ = "?"; cuGetErrorString(r_, &s_); \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, s_); exit(2); \
} } while (0)
#include "test_guards.h"

#define WIDTH 8u
typedef struct {
    const char *name;
    unsigned position;
    CUdeviceptr ptr;
    size_t bytes;
    void *initial, *scalar;
    int compare;
    size_t compared_bytes;
} region;
static region regions[192];
static unsigned nregions;
static int host_kv = 1;
static CUdeviceptr make_region(const char *name, unsigned position, size_t bytes,
                               const void *initial, int compare, int host) {
    if (nregions == sizeof regions / sizeof regions[0]) exit(2);
    region *r = regions + nregions++;
    *r = (region){name, position, test_guard_alloc(name, bytes, host), bytes,
                  calloc(1, bytes), malloc(bytes), compare, compare ? bytes : 0};
    if (!r->initial || !r->scalar) exit(2);
    if (initial) memcpy(r->initial, initial, bytes);
    CK(cuMemcpyHtoD(r->ptr, r->initial, bytes));
    return r->ptr;
}
static void restore_regions(void) {
    for (unsigned i = 0; i < nregions; i++)
        CK(cuMemcpyHtoD(regions[i].ptr, regions[i].initial, regions[i].bytes));
}
static int compare_regions(int grouped) {
    int bad = 0;
    for (unsigned i = 0; i < nregions; i++) {
        region *r = regions + i;
        if (!grouped) CK(cuMemcpyDtoH(r->scalar, r->ptr, r->bytes));
        else {
            if (!r->compare) continue;
            size_t n = r->compared_bytes;
            unsigned char *got = malloc(n ? n : 1);
            if (!got) exit(2);
            CK(cuMemcpyDtoH(got, r->ptr, n));
            if (memcmp(got, r->scalar, n)) {
                size_t j = 0;
                while (got[j] == ((unsigned char *)r->scalar)[j]) j++;
                fprintf(stderr, "position=%u region=%s byte=%zu scalar=%02x group=%02x\n",
                        r->position, r->name, j, ((unsigned char *)r->scalar)[j], got[j]);
                bad++;
            }
            free(got);
        }
    }
    return bad;
}
static int compare_compact(CUdeviceptr ptr, unsigned n, int blocked) {
    region *r = NULL;
    for(unsigned i=0;i<nregions;i++)if(regions[i].ptr==ptr){r=regions+i;break;}
    if(!r)exit(2);
    float *got=malloc(r->bytes);if(!got)exit(2);
    CK(cuMemcpyDtoH(got,ptr,r->bytes));
    for(unsigned t=0;t<n;t++)for(unsigned kv=0;kv<2;kv++)for(unsigned d=0;d<256;d++){
        size_t j=blocked?attn_kidx(t,(int)kv,(int)d):(size_t)t*512+kv*256+d;
        if(memcmp(got+j,(float*)r->scalar+j,4)){
            fprintf(stderr,"position=%u region=%s valid key=%u head=%u dim=%u differs\n",r->position,r->name,t,kv,d);
            free(got);return 1;
        }
    }
    free(got);return 0;
}
static int validate_output(CUdeviceptr ptr, unsigned position, int grouped) {
    float values[6144];int nonzero=0;
    CK(cuMemcpyDtoH(values,ptr,sizeof values));
    for(unsigned i=0;i<6144;i++){
        if(!isfinite(values[i])){fprintf(stderr,"%s position=%u output[%u] nonfinite\n",grouped?"B8":"scalar",position,i);return 1;}
        nonzero|=values[i]!=0.f;
    }
    if(!nonzero){fprintf(stderr,"%s position=%u output is zero\n",grouped?"B8":"scalar",position);return 1;}
    return 0;
}
static uint32_t hash32(uint32_t x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
    x *= 0x846ca68bu; return x ^ (x >> 16);
}
static float raw(unsigned t, unsigned d, unsigned salt) {
    return ((int)(hash32(t * 1664525u + d * 1013904223u + salt) & 65535u) - 32768) / 32768.f;
}
static size_t pool_index(unsigned b, unsigned d) {
    return (size_t)(b >> 5) * 4096 + (d >> 2) * 128 + (b & 31) * 4 + (d & 3);
}
typedef struct {
    CUdeviceptr qg, k, v, iqk, qo, iq, sc, hst, sel, out;
    CUdeviceptr kc, vc, pm, pd, po;
} position_buf;
typedef struct {
    CUdeviceptr kq, vq, ks, vs, pkt, ring, cs, sn, qn, kn, iqn, ikn;
} shared_buf;

static void program(prog_t *p, unsigned position, unsigned budget,
                    const shared_buf *s, const position_buf *b) {
    p->loop = 1; p->token_offset = position;
    phase_t *d = prog_add(p, OP_ATTNQKQ, 26);
    ph_ptr(d,0x140,b->qg); ph_ptr(d,0x148,b->k); ph_ptr(d,0x150,b->v);
    ph_ptr(d,0x158,s->qn); ph_ptr(d,0x160,s->kn); ph_ptr(d,0x168,s->cs);
    ph_ptr(d,0x170,s->sn); ph_ptr(d,0x178,b->qo); ph_ptr(d,0x180,s->kq);
    ph_ptr(d,0x188,s->vq); ph_tokv(p,d,0x190,0,0); ph_f32(d,0x194,1e-6f);
    ph_ptr(d,0x198,s->ks); ph_ptr(d,0x1a0,s->vs);
    prog_nobar(p); /* exactly the production append/indexer independence */
    d = prog_add(p, OP_IDXQK, 6);
    ph_ptr(d,0x140,b->iqk); ph_ptr(d,0x148,s->iqn); ph_ptr(d,0x150,s->ikn);
    ph_ptr(d,0x158,s->cs); ph_ptr(d,0x160,s->sn); ph_ptr(d,0x168,b->iq);
    ph_ptr(d,0x170,s->ring); ph_ptr(d,0x178,s->pkt); ph_ptr(d,0x180,b->hst);
    ph_tokv(p,d,0x188,0,0); ph_f32(d,0x18c,1e-6f);
    d = prog_add(p, OP_IDXSCORE, 16);
    ph_ptr(d,0x140,s->pkt); ph_ptr(d,0x148,b->iq); ph_ptr(d,0x150,b->sc);
    ph_ptr(d,0x158,b->hst); ph_tokv(p,d,0x160,0,0); ph_u32(d,0x164,budget);
    for (unsigned pass = 2; pass <= 6; pass++) {
        d = prog_add(p, OP_IDXSEL, 16);
        ph_ptr(d,0x140,b->sc); ph_ptr(d,0x148,b->hst); ph_ptr(d,0x150,b->sel);
        ph_tokv(p,d,0x158,0,0); ph_u32(d,0x15c,budget); ph_u32(d,0x160,pass);
    }
    d = prog_add(p, OP_KVGATHER, budget + 1);
    ph_ptr(d,0x140,s->kq); ph_ptr(d,0x148,s->ks); ph_ptr(d,0x150,s->vq);
    ph_ptr(d,0x158,s->vs); ph_ptr(d,0x160,b->kc); ph_ptr(d,0x168,b->vc);
    ph_ptr(d,0x170,b->sel); ph_tokv(p,d,0x178,0,0); ph_u32(d,0x17c,budget);
    d = prog_add(p, OP_ATTNPARTN, 32);
    ph_ptr(d,0x140,b->qo); ph_ptr(d,0x148,b->kc); ph_ptr(d,0x150,b->vc);
    ph_ptr(d,0x158,b->pm); ph_ptr(d,0x160,b->pd); ph_ptr(d,0x168,b->po);
    ph_tokv(p,d,0x170,1,0); ph_u32(d,0x174,budget);
    d = prog_add(p, OP_ATTNMERGE, 24);
    ph_ptr(d,0x140,b->pm); ph_ptr(d,0x148,b->pd); ph_ptr(d,0x150,b->po);
    ph_ptr(d,0x158,b->qg); ph_ptr(d,0x160,b->out);
}
static unsigned barriers(const prog_t *p) {
    unsigned count = 0;
    for (int i = 0; i < p->n; ) {
        unsigned width = phase_positions(p->p + i);
        if (!(p->p[i].w[30] & 4u)) count++;
        i += (int)width;
    }
    return count;
}
static int launch(CUfunction fn, CUdeviceptr ctl, CUdeviceptr dp,
                  const prog_t *p, unsigned start, int grouped) {
    uint32_t control[2048] = {0};
    control[2] = start; control[9] = WIDTH;
    CK(cuMemcpyHtoD(ctl, control, sizeof control));
    CK(cuMemcpyHtoD(dp, p->p, (size_t)p->n * sizeof(phase_t)));
    unsigned nph = (unsigned)p->n, end = start + WIDTH;
    void *args[] = { &dp, &nph, &ctl, &end };
    CK(cuLaunchKernel(fn,16,1,1,256,1,1,0,0,args,NULL));
    CK(cuCtxSynchronize());
    CK(cuMemcpyDtoH(control,ctl,sizeof control));
    unsigned expected = barriers(p);
    if (control[2] != end || control[1] != expected) {
        fprintf(stderr,"%s control: TOK=%u expected=%u GEN=%u expected=%u\n",
                grouped?"B8":"scalar",control[2],end,control[1],expected);
        return 1;
    }
    return 0;
}
static int run_case(CUfunction fn, unsigned start, unsigned budget) {
    unsigned npos = (start + WIDTH + 127u) & ~127u;
    unsigned cpos = (4u * (budget + 1u) + 2048u + 31u) & ~31u;
    size_t kcf = ATTN_KC_FLOATS(cpos), vcf = (size_t)cpos * 512;
    int8_t *kq = malloc((size_t)npos*512), *vq = malloc((size_t)npos*512);
    uint16_t *ks = malloc((size_t)npos*32), *vs = malloc((size_t)npos*32);
    uint16_t *pkt = calloc((size_t)npos/128,8192);
    float *cs = calloc((size_t)npos,128), *sn = calloc((size_t)npos,128);
    float qn[256],kn[256],iqn[128],ikn[128],ring[512] = {0};
    if (!kq||!vq||!ks||!vs||!pkt||!cs||!sn) exit(2);
    memset(kq,127,(size_t)npos*512); memset(vq,0x81,(size_t)npos*512);
    for (size_t i=0;i<(size_t)npos*16;i++) ks[i]=vs[i]=qsa_f2h(4.f);
    for (unsigned t=0;t<start;t++) {
        for (unsigned d=0;d<512;d++) {
            kq[(size_t)t*512+d]=(int8_t)((int)(hash32(t*512+d+17)%255)-127);
            vq[(size_t)t*512+d]=(int8_t)((int)(hash32(t*512+d+53)%255)-127);
        }
        for(unsigned g=0;g<16;g++)ks[(size_t)t*16+g]=vs[(size_t)t*16+g]=qsa_f2h(.01f);
    }
    for(unsigned d=0;d<256;d++){qn[d]=.75f+.125f*raw(0,d,1);kn[d]=.75f+.125f*raw(0,d,2);}
    for(unsigned d=0;d<128;d++){iqn[d]=.75f+.125f*raw(0,d,3);ikn[d]=.75f+.125f*raw(0,d,4);}
    for(unsigned block=0;block<start/4;block++) {
        float input[512],bc[32],bs[32]; uint16_t key[128];
        for(unsigned j=0;j<4;j++)for(unsigned d=0;d<128;d++)input[j*128+d]=raw(4*block+j,d,5);
        rope_table_gpu(4*block,1e7f,bc,bs);idx_pool_gpu(input,ikn,1e-6f,bc,bs,key);
        for(unsigned d=0;d<128;d++)pkt[pool_index(block,d)]=key[d];
    }
    for(unsigned j=0;j<4&&j<start;j++)for(unsigned d=0;d<128;d++){
        unsigned t=start-1-j;ring[(t&3)*128+d]=raw(t,d,5);
    }
    for(unsigned t=start&~3u;t<start+WIDTH;t++)rope_table_gpu(t,1e7f,cs+32*t,sn+32*t);
    shared_buf s;
#define SHKV(field,name,bytes,init) s.field=make_region(name,99,bytes,init,1,host_kv)
    SHKV(kq,"K codes",(size_t)npos*512,kq);SHKV(vq,"V codes",(size_t)npos*512,vq);
    SHKV(ks,"K scales",(size_t)npos*32,ks);SHKV(vs,"V scales",(size_t)npos*32,vs);
#undef SHKV
#define SH(field,name,bytes,init,compare) s.field=make_region(name,99,bytes,init,compare,0)
    SH(pkt,"index pool",(size_t)npos/128*8192,pkt,1);SH(ring,"index ring",2048,ring,1);
    SH(cs,"RoPE cos",(size_t)npos*128,cs,1);SH(sn,"RoPE sin",(size_t)npos*128,sn,1);
    SH(qn,"Q norm",sizeof qn,qn,1);SH(kn,"K norm",sizeof kn,kn,1);
    SH(iqn,"IQ norm",sizeof iqn,iqn,1);SH(ikn,"IK norm",sizeof ikn,ikn,1);
#undef SH
    position_buf b[WIDTH];
    float *poison=malloc((kcf>vcf?kcf:vcf)*4);
    if(!poison)exit(2);
    for(size_t i=0;i<(kcf>vcf?kcf:vcf);i++){uint32_t nan=0x7fc12345u;memcpy(poison+i,&nan,4);}
    for(unsigned k=0;k<WIDTH;k++) {
        float qg[12288],key[512],value[512],iqk[640];unsigned t=start+k;
        for(unsigned d=0;d<12288;d++)qg[d]=raw(t,d,7);
        for(unsigned d=0;d<512;d++){key[d]=raw(t,d,11);value[d]=raw(t,d,13);iqk[d]=raw(t,d,19);}
        for(unsigned d=0;d<128;d++)iqk[512+d]=raw(t,d,5);
#define POS(field,name,bytes,init,compare) b[k].field=make_region(name,k,bytes,init,compare,0)
        POS(qg,"Q gate",sizeof qg,qg,1);POS(k,"K input",sizeof key,key,1);
        POS(v,"V input",sizeof value,value,1);POS(iqk,"IQK input",sizeof iqk,iqk,1);
        POS(qo,"Q output",6144*4,NULL,1);POS(iq,"IQ output",512*4,NULL,1);
        POS(sc,"scores",((size_t)npos/4+64)*4,NULL,1);
        /* IDXSCORE stores all 32 lanes of its last pool group, but only
         * blocks < nv enter selection. Future pool rows legitimately differ
         * after B8 has appended the whole group. Compare logical scores. */
        regions[nregions-1].compared_bytes=(size_t)((t+1)/4)*4;
        POS(hst,"histogram",1280*4,NULL,1);
        POS(sel,"selection",(budget+8)*4,NULL,1);POS(out,"attention output",6144*4,NULL,1);
        if(k<2) {
            /* Full compact buffers are excluded from the byte comparison:
             * future rows inside a gathered tail block may differ legally.
             * Poisoned padding must still yield identical partials/output. */
            POS(kc,"compact K",kcf*4,poison,0);POS(vc,"compact V",vcf*4,poison,0);
            POS(pm,"partial maxima",192*4,NULL,1);POS(pd,"partial denominator",192*4,NULL,1);
            POS(po,"partial values",49152*4,NULL,1);
        } else {
            b[k].kc=b[0].kc;b[k].vc=b[0].vc;b[k].pm=b[0].pm;b[k].pd=b[0].pd;b[k].po=b[0].po;
        }
#undef POS
    }
    CUdeviceptr ctl=make_region("control",99,8192,NULL,0,0);
    CUdeviceptr dp=make_region("program",99,128*128,NULL,0,0);
    prog_t scalar,group,position[WIDTH];const prog_t *positions[WIDTH];
    prog_init(&scalar);prog_init(&group);
    for(unsigned k=0;k<WIDTH;k++) {
        prog_init(position+k);program(position+k,k,budget,&s,b+k);positions[k]=position+k;
        for(int i=0;i<position[k].n;i++)*prog_add(&scalar,position[k].p[i].w[0],position[k].p[i].w[1])=position[k].p[i];
    }
    if(prog_group(&group,positions,WIDTH)){fputs("B8 grouping rejected fixture\n",stderr);exit(2);}
    int bad=0;
    for(int grouped=0;grouped<2;grouped++) {
        restore_regions();bad+=launch(fn,ctl,dp,grouped?&group:&scalar,start,grouped);
        bad+=compare_regions(grouped);bad+=test_guard_check();
        for(unsigned k=0;k<WIDTH;k++)bad+=validate_output(b[k].out,k,grouped);
        if(grouped)for(unsigned k=0;k<2;k++){
            unsigned t=start+(k?1:WIDTH-1),n=qsa_nkeys(t,budget);
            bad+=compare_compact(b[k].kc,n,1);bad+=compare_compact(b[k].vc,n,0);
        }
    }
    printf("group_attention start=%u end=%u budget=%u width=8 hostKV=%d barriers=%u/%u allocation_guards=%u poisoned_padding=yes parity=%s\n",
           start,start+WIDTH-1,budget,host_kv,barriers(&scalar),barriers(&group),test_guard_count,bad?"FAIL":"BIT-EXACT");
    for(unsigned k=0;k<WIDTH;k++)prog_free(position+k);
    prog_free(&scalar);prog_free(&group);
    for(unsigned i=0;i<nregions;i++){free(regions[i].initial);free(regions[i].scalar);}nregions=0;
    test_guard_free();free(kq);free(vq);free(ks);free(vs);free(pkt);free(cs);free(sn);free(poison);
    return bad;
}
int main(int argc,char **argv) {
    unsigned budget=512;
    if(argc>2){fprintf(stderr,"usage: %s [start_position]\n",argv[0]);return 2;}
    const char *host=getenv("GROUP_ATTN_HOSTKV");
    if(host&&strcmp(host,"0")&&strcmp(host,"1")){fputs("GROUP_ATTN_HOSTKV must be 0 or 1\n",stderr);return 2;}
    if(host)host_kv=!strcmp(host,"1");
    CK(cuInit(0));CUdevice dev;CUcontext ctx;CUmodule mod;CUfunction fn;
    CK(cuDeviceGet(&dev,0));CK(cuCtxCreate_v2(&ctx,CU_CTX_SCHED_SPIN,dev));
    const char *cubin=getenv("FLASH_CUBIN");if(!cubin)cubin=getenv("DIE_CUBIN");if(!cubin)cubin="die.cubin";
    CK(cuModuleLoad(&mod,cubin));CK(cuModuleGetFunction(&fn,mod,"die"));
    const unsigned starts[]={0,1,2,3,124,127,2041,2047,2048,8189,8192};
    int bad=0;unsigned cases=0;
    if(argc==2){char *end;unsigned long t=strtoul(argv[1],&end,10);if(!*argv[1]||*end||t>262136)return 2;bad=run_case(fn,(unsigned)t,budget);cases=1;}
    else for(unsigned i=0;i<sizeof starts/sizeof starts[0];i++){bad+=run_case(fn,starts[i],budget);cases++;}
    printf("group_attention_test cases=%u failures=%d cubin=%s\n",cases,bad,cubin);
    CK(cuCtxDestroy(ctx));return bad?1:0;
}
