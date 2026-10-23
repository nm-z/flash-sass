#define _GNU_SOURCE
#include <cuda.h>
#include <stdlib.h>
static CUresult test_context_entry(CUcontext context);
static CUresult unexpected_cuda_call(void);
static void *test_program_calloc(size_t count, size_t size);
static CUresult test_image_copy(CUdeviceptr destination,const void *source,size_t bytes);
static CUresult test_image_reset(CUdeviceptr destination,unsigned value,size_t count);
#define calloc(count,size) test_program_calloc(count,size)
#define cuCtxSetCurrent(context) test_context_entry(context)
#define cuInit(...) unexpected_cuda_call()
#define cuLaunchKernel(fn,gx,gy,gz,bx,by,bz,shared,stream,params,extra) ((void)(params), unexpected_cuda_call())
#undef cuMemcpyHtoDAsync
#define cuMemcpyHtoDAsync(...) unexpected_cuda_call()
#define cuStreamSynchronize(...) unexpected_cuda_call()
#undef cuMemcpyHtoD
#define cuMemcpyHtoD(destination,source,bytes) test_image_copy(destination,source,bytes)
#undef cuMemsetD32
#define cuMemsetD32(destination,value,count) test_image_reset(destination,value,count)
#include "engine.c"
#undef calloc
#undef cuCtxSetCurrent
#undef cuInit
#undef cuLaunchKernel
#undef cuMemcpyHtoDAsync
#undef cuStreamSynchronize
#undef cuMemcpyHtoD
#undef cuMemsetD32
#undef nd
#undef ctx
#include <assert.h>
#include <sys/socket.h>

static unsigned calloc_fail_at, calloc_calls;
static void *test_program_calloc(size_t count, size_t size) {
    if (++calloc_calls == calloc_fail_at) return NULL;
    return calloc(count, size);
}

static int abort_driver_entries;
static int image_test_mode,image_contexts,image_uploads,map_uploads,map_resets;
static CUresult unexpected_cuda_call(void) {
    fputs("trace_session_test: unexpected CUDA operation\n", stderr);
    abort();
}
static CUresult test_context_entry(CUcontext context) {
    if(image_test_mode){assert(context==core->ctx[embed_die]);image_contexts++;return CUDA_SUCCESS;}
    (void)context;
    /* A callback-return timestamp must exist before abort_run enters CUDA. */
    assert(trace_host[0].callback_start != 0);
    assert(trace_host[0].callback_end >= trace_host[0].callback_start);
    abort_driver_entries++;
    return CUDA_SUCCESS;
}
static CUresult test_image_copy(CUdeviceptr destination,const void *source,size_t bytes) {
    if(!image_test_mode)return unexpected_cuda_call();
    if(destination==db[embed_die].img) {
        assert(bytes==2u*2560);const uint16_t *row=source;
        for(unsigned j=0;j<2560;j++)assert(row[j]==qsa_f2h((float)j/4096.0f));
        image_uploads++;
    } else {
        assert(destination==db[embed_die].imap && bytes==5u*4);
        const uint32_t *map=source;assert(map[0]==UINT32_MAX && map[1]==0 && map[2]==UINT32_MAX);
        map_uploads++;
    }
    return CUDA_SUCCESS;
}
static CUresult test_image_reset(CUdeviceptr destination,unsigned value,size_t count) {
    if(!image_test_mode)return unexpected_cuda_call();
    assert(destination==db[embed_die].imap && value==UINT32_MAX && count==(size_t)MAXT+2);
    map_resets++;return CUDA_SUCCESS;
}
static int request_cancel(void *arg, int position, uint32_t token) {
    assert(position == 4 && token == 123);
    assert(!trace_host[0].callback_end);
    (*(int *)arg)++;
    return 1;
}
static void callback_return_before_cancel(void) {
    core_t fake={.nd=1};fake.ctx[0]=(CUcontext)(uintptr_t)1;
    core_t *saved_core=core;core=&fake;
    int calls = 0;
    engine_cb cb = { .arg = &calls, .token = request_cancel };
    runctx r = { .count = 5, .ntok = 6, .cb = &cb };
    uint32_t abort_count = 0;
    h_abort = &abort_count;
    trace_first = 4; trace_n = 1;
    memset(trace_host, 0, sizeof trace_host);
    abort_driver_entries = 0;
    deliver_token(&r, 4, 123);
    assert(calls == 1 && r.aborted && abort_count == 13);
    assert(abort_driver_entries == 1);

    /* A subsequent cancellation must retain the existing termination state. */
    memset(trace_host, 0, sizeof trace_host);
    deliver_token(&r, 4, 123);
    assert(calls == 2 && abort_driver_entries == 1);
    r.aborted = 0; r.ended = 1;
    memset(trace_host, 0, sizeof trace_host);
    deliver_token(&r, 4, 123);
    assert(calls == 3 && !r.aborted && abort_driver_entries == 1);
    h_abort = NULL; trace_n = 0;core=saved_core;
}

static void session_options(void) {
    char input[]="timings on\nchat nonsense\ntimings off\nchat nonsense\ntimings\ncancel_on_eof 1\ncancel_on_eof 0\nquit\n";
    FILE *in=fmemopen(input,strlen(input),"r");char *output=NULL;size_t n=0;
    FILE *out=open_memstream(&output,&n);assert(in && out);
    assert(engine_session(in,out)==1);assert(!fclose(in) && !fclose(out));
    assert(strstr(output,"ok: timings 1\n") && strstr(output,"ok: timings 0\n"));
    assert(strstr(output,"ok: cancel_on_eof 1\n") && strstr(output,"ok: cancel_on_eof 0\n"));
    int records=0;for(char *p=output;(p=strstr(p,"timings {"));p++)records++;
    assert(records==2); /* first opted-in error, then explicit last-record query */
    free(output);
}

static void half_close_options(void) {
    int fd[2];assert(!socketpair(AF_UNIX,SOCK_STREAM,0,fd));
    FILE *out=fdopen(fd[0],"w");assert(out);
    session_client client={.out=out};
    assert(!shutdown(fd[1],SHUT_WR));
    assert(!client_gone(&client)); /* a normal CLI command pipe may reach EOF */
    client.cancel_on_eof=1;assert(client_gone(&client));
    assert(!put_token(&client,0,123)); /* cancellation still leaves the reply readable */
    char buf[16]={0};assert(read(fd[1],buf,sizeof buf)==8 && !strcmp(buf,"tok 123\n"));
    assert(!fclose(out) && !close(fd[1]));
}

/* Audit headless arena offsets and the deliberately shared attention/head
 * lifetimes without allocating or touching any device memory. */
static void headless_arena_layout(void) {
    assert(!allocs && !nallocs);
    allocrec shapes[]={{.p=0x1000,.bytes=64},{.p=0x2000,.bytes=2052},{.p=0x3000,.bytes=96}};
    allocs=shapes;nallocs=3;
    dn_bufs_t original={.S=0x1000,.r=0x2000,.aout=0x3000,
        .kc=0x4000,.vc=0x5000,.pm=0x6000,.pd=0x7000,.po=0x8000,
        .hlo=0x9000,.ids=0xa000,.dn_state=0xb000,.ple_state=0xc000};
    dn_bufs_t b=original;size_t bytes=0;
    clone_position_fields(&b,2,0,&bytes);
    assert(bytes==256+2304+256);
    b=original;size_t offset=0;
    const CUdeviceptr arena=0x100000;
    clone_position_fields(&b,2,arena,&offset);
    assert(offset==bytes && b.S==arena && b.r==arena+256 && b.aout==arena+256+2304);
    assert(b.kc==original.kc && b.vc==original.vc && b.pm==original.pm);
    assert(b.pd==original.pd && b.po==original.po && b.hlo==original.hlo);
    assert(b.ids==original.ids && b.dn_state==original.dn_state && b.ple_state==original.ple_state);
    allocs=NULL;nallocs=0;
}

static int overlap(CUdeviceptr a,size_t an,CUdeviceptr b,size_t bn) {
    return a<b+bn && b<a+an;
}
static void wide_arena_lifetimes(void) {
    void *memory=NULL;assert(!posix_memalign(&memory,256,WIDE_ARENA_BYTES+256));
    memset(memory,0xa5,WIDE_ARENA_BYTES+256);
    CUdeviceptr base=(CUdeviceptr)(uintptr_t)memory;
    dn_bufs_t original={.sc=1,.iq=2,.hst=3,.sel=4,.kc=5,.vc=6,.pm=7,.pd=8,.po=9,
        .pG=10,.px=11,.pgates=12,.ids=13,.dn_state=14,.ple_state=15,.hlo=16};
    xchg_t x={.flags=17,.xgen=18};dn_bufs_t b=original;
    split_t shape={.hn=8,.nk=8,.cbn=14,.pkn=1707,.yn=427};
    assert(!wide_position_fields(&b,&x,base,&shape));
    assert(WIDE_ARENA_BYTES==301056 && x.inbox+2*XSLOT==base+WIDE_ARENA_BYTES);
    assert(b.sc==original.sc && b.iq==original.iq && b.hst==original.hst && b.sel==original.sel);
    assert(b.kc==original.kc && b.vc==original.vc && b.pm==original.pm && b.pd==original.pd && b.po==original.po);
    assert(b.pG==original.pG && b.px==original.px && b.pgates==original.pgates);
    assert(b.ids==original.ids && b.dn_state==original.dn_state && b.ple_state==original.ple_state);
    assert(b.hlo==original.hlo && x.flags==17 && x.xgen==18);
    assert(!overlap(b.lqkv,12288,b.lu,12288));
    assert(!overlap(b.lu,12288,b.lo,4096) && !overlap(b.lgate,4096,b.lo,4096));
    assert(!overlap(b.g,28160,b.gu,28160));
    assert(!overlap(b.g,56320,b.hq,17920) && !overlap(b.S2,40960,b.hq,17920));
    assert(!overlap(b.hq,17920,b.part,10240) && !overlap(b.S2,40960,b.part,10240));
    assert(!overlap(b.k,2048,b.v,2048) && !overlap(b.v,2048,b.iqk,2560));
    assert(!overlap(b.pg,10240,b.pgq,6912));
    assert(b.part==b.r && b.part==b.out && b.part==b.lup && b.part==b.ly);
    /* A zero-local-expert die relies on MOESEL's zero partial surviving all
     * G/GU/HQ writes until its optional shared-expert down accumulation. */
    memset((void *)(uintptr_t)b.r,0x23,2052); /* previous router result */
    memset((void *)(uintptr_t)b.part,0,10240); /* MOESEL, ne=0 */
    memset((void *)(uintptr_t)b.g,0x31,28160);
    memset((void *)(uintptr_t)b.gu,0x32,28160);
    memset((void *)(uintptr_t)b.hq,0x33,17920);
    memset((void *)(uintptr_t)b.S2,0x34,40960);
    const unsigned char *partial=(const unsigned char *)(uintptr_t)b.part;
    for(unsigned i=0;i<10240;i++)assert(!partial[i]);
    const unsigned char *stream=(const unsigned char *)(uintptr_t)b.S;
    for(unsigned i=0;i<40960;i++)assert(stream[i]==0xa5);
    memset((void *)(uintptr_t)b.pg,0x41,10240);
    memset((void *)(uintptr_t)b.pgq,0x42,6912);
    const unsigned char *pg=(const unsigned char *)(uintptr_t)b.pg;
    for(unsigned i=0;i<10240;i++)assert(pg[i]==0x41);
    const unsigned char *guard=(const unsigned char *)memory+WIDE_ARENA_BYTES;
    for(unsigned i=0;i<256;i++)assert(guard[i]==0xa5);
    assert(sizeof prof_op[0]/sizeof prof_op[0][0]>=PROG_MAX_DESCRIPTORS);
    assert(sizeof prof_layer[0]/sizeof prof_layer[0][0]>=PROG_MAX_DESCRIPTORS);
    free(memory);
}

static void wide_head_case(size_t hlo_bytes) {
    assert(!allocs && !nallocs);
    const size_t sizes[]={hlo_bytes,8,128,4,8,256,4096,4};
    allocrec shapes[8];for(unsigned i=0;i<8;i++)shapes[i]=(allocrec){.p=0x1000u+0x1000u*i,.bytes=sizes[i]};
    allocs=shapes;nallocs=8;
    dn_bufs_t original={.hlo=0x1000,.amo=0x2000,.apart=0x3000,.acnt=0x4000,
        .tokout=0x5000,.tkout=0x6000,.tkpart=0x7000,.tkcnt=0x8000,.S=0xabcdef};
    dn_bufs_t b=original;size_t bytes=0;clone_head_fields(&b,0,&bytes);
    assert(bytes==wide_align(hlo_bytes)+5632);
    b=original;size_t used=0;clone_head_fields(&b,0x100000,&used);
    assert(used==bytes && b.S==original.S && b.hlo!=original.hlo && b.tkcnt!=original.tkcnt);
    assert(!overlap(b.hlo,sizes[0],b.amo,sizes[1]));
    assert(b.tkcnt+256==0x100000+bytes);
    allocs=NULL;nallocs=0;
}

static void wide_live_split(void) {
    /* Main sets the exact live weighted split before split_bound's first use. */
    const unsigned heads[]={8,8,7,9,8,8};
    void *memory=NULL;assert(!posix_memalign(&memory,256,WIDE_ARENA_BYTES+256));
    CUdeviceptr base=(CUdeviceptr)(uintptr_t)memory;
    for(int d=0;d<6;d++) {
        split_t shape;split_of(&shape,d,6,0);assert((unsigned)shape.hn==heads[d]);
        dn_bufs_t b={0};xchg_t x={0};
        assert(!wide_position_fields(&b,&x,base,&shape));
        size_t qkv=4u*SPLIT_NLOC(&shape),gate=512u*(unsigned)shape.hn,ab=8u*(unsigned)shape.hn;
        assert(!overlap(b.lqkv,qkv,b.lu,qkv));
        assert(!overlap(b.lu,qkv,b.lgate,gate));
        assert(!overlap(b.lgate,gate,b.la,ab));
        assert(!overlap(b.lo,gate,b.lu,qkv));
        assert(b.la+ab<=base+WIDE_B);
        assert(b.lup+512u*(unsigned)shape.cbn<=base+WIDE_S);
        assert(b.lk+4u*(unsigned)shape.pkn<=base+WIDE_S && b.ly+4u*(unsigned)shape.yn<=base+WIDE_S);
        assert(b.qg+4u*12288<=base+WIDE_B && b.xh+40960<=base+WIDE_B);
        assert(b.aout+24576<=base+WIDE_C && b.oas+768<=base+WIDE_C);
        assert(b.has+880<=base+WIDE_D && b.pgas+320<=base+WIDE_D);
        assert(b.iqk+2560<=base+WIDE_S && b.part+10240<=base+WIDE_S);
        memset(memory,0xa5,WIDE_ARENA_BYTES+256);
        memset((void *)(uintptr_t)b.lgate,0x51,gate);
        memset((void *)(uintptr_t)b.lqkv,0x52,qkv);
        memset((void *)(uintptr_t)b.lu,0x53,qkv);
        for(size_t i=0;i<qkv;i++)assert(((unsigned char *)(uintptr_t)b.lqkv)[i]==0x52);
        for(size_t i=0;i<gate;i++)assert(((unsigned char *)(uintptr_t)b.lgate)[i]==0x51);
        memset((void *)(uintptr_t)b.la,0x54,ab);
        memset((void *)(uintptr_t)b.lo,0x55,gate);
        for(size_t i=0;i<qkv;i++)assert(((unsigned char *)(uintptr_t)b.lu)[i]==0x53);
        for(size_t i=0;i<gate;i++)assert(((unsigned char *)(uintptr_t)b.lgate)[i]==0x51);
        for(unsigned i=0;i<256;i++)assert(((unsigned char *)memory)[WIDE_ARENA_BYTES+i]==0xa5);
        wide_head_case(4u*(unsigned)shape.on);
        printf("wide split die=%d hn=%d nk=%d qkv=%zu gate=%zu lu_off=%llu gate_off=%llu ab_off=%llu cbn=%d pkn=%d yn=%d PASS\n",
               d,shape.hn,shape.nk,qkv,gate,(unsigned long long)(b.lu-base),(unsigned long long)(b.lgate-base),(unsigned long long)(b.la-base),shape.cbn,shape.pkn,shape.yn);
    }
    split_t too_large={.hn=48,.nk=16,.cbn=16,.pkn=1980,.yn=495};dn_bufs_t b={.S=123},saved=b;xchg_t x={0};
    assert(wide_position_fields(&b,&x,base,&too_large)<0 && !memcmp(&b,&saved,sizeof b));
    too_large=(split_t){.hn=9,.nk=9,.cbn=80,.pkn=1980,.yn=495};
    assert(wide_position_fields(&b,&x,base,&too_large)<0 && !memcmp(&b,&saved,sizeof b));
    too_large.cbn=16;too_large.pkn=10240;
    assert(wide_position_fields(&b,&x,base,&too_large)<0 && !memcmp(&b,&saved,sizeof b));
    free(memory);
}

static uint64_t wide_desc_pointer(const phase_t *p,unsigned word) {
    return p->w[word]|((uint64_t)p->w[word+1]<<32);
}
static void wide_hcr_decomposition(void) {
    enum { WIDTH=32 };void *memory=NULL;
    assert(!posix_memalign(&memory,256,(size_t)WIDTH*WIDE_ARENA_BYTES+256));
    for(int d=0;d<6;d++) {
        split_t shape;split_of(&shape,d,6,0);dn_bufs_t b[WIDTH];xchg_t x[WIDTH];CUdeviceptr scratch[WIDTH];
        memset(memory,0xa5,(size_t)WIDTH*WIDE_ARENA_BYTES+256);
        for(unsigned p=0;p<WIDTH;p++) {
            b[p]=(dn_bufs_t){0};x[p]=(xchg_t){0};
            CUdeviceptr base=(CUdeviceptr)(uintptr_t)memory+(size_t)p*WIDE_ARENA_BYTES;
            assert(!wide_position_fields(b+p,x+p,base,&shape));scratch[p]=b[p].out;
        }
        prog_t program;prog_init(&program);program.xj=96;program.pending_wait=1;
        for(unsigned layer=0;layer<48;layer++)for(unsigned rw=0;rw<2;rw++) {
            for(unsigned p=0;p<WIDTH;p++) {
                phase_t *a=prog_add(&program,rw?OP_BF16HCRWX:OP_BF16HCRX,16);
                ph_ptr(a,0x140,0x100000000ull+0x1000000ull*layer);
                ph_ptr(a,0x148,b[p].S);ph_ptr(a,0x150,x[p].xtab);ph_u32(a,0x158,(unsigned)shape.djn);
                ph_u32(a,0x168,XSLOT+4u*(unsigned)shape.dj0);ph_ptr(a,0x178,b[p].xh);
                ph_ptr(a,0x190,0x100800000ull+0x1000000ull*layer);ph_f32(a,0x198,1e-6f);
                if(rw){ph_ptr(a,0x160,b[p].S2);ph_ptr(a,0x1a0,x[p].inbox);ph_ptr(a,0x1a8,b[p].t+1280);}
                a->w[30]=3|(p?0:PHASE_GROUP_FLAG|(WIDTH<<PHASE_WIDTH_SHIFT));
            }
        }
        program.pending_wait=1;int original=program.n;
        assert(prog_batch_decompose_hcr(&program,scratch,WIDTH,1)==96);
        assert(program.n-original==144*WIDTH && program.xj==96 && program.pending_wait==1);
        assert(program.n<=PROG_MAX_DESCRIPTORS);
        for(unsigned p=0;p<WIDTH;p++) {
            phase_t *w=program.p+2*WIDTH+p,*r=w+WIDTH,*hc=r+WIDTH;
            assert(w->w[0]==OP_HCWSUM && w->w[11]==1 && wide_desc_pointer(w,12)==b[p].out);
            assert(wide_desc_pointer(w,14)==b[p].S && wide_desc_pointer(w,6)==b[p].S2);
            assert(r->w[0]==OP_RMS2560 && wide_desc_pointer(r,2)==b[p].S2 && wide_desc_pointer(r,6)==b[p].xh);
            assert(hc->w[0]==OP_BF16HCX && wide_desc_pointer(hc,4)==b[p].xh);
            /* HCWSUM's D output is dead before HCUP/router/MOESEL reuse it.
             * The updated stream in B and inject values in T remain live. */
            memset((void *)(uintptr_t)b[p].S2,0x34,40960);
            memset((void *)(uintptr_t)b[p].out,0x41,10240);
            memset((void *)(uintptr_t)b[p].xh,0x42,40960);
            memset((void *)(uintptr_t)b[p].lup,0x43,512u*(unsigned)shape.cbn);
            memset((void *)(uintptr_t)b[p].r,0x44,2052);
            memset((void *)(uintptr_t)b[p].part,0,10240);
            for(unsigned j=0;j<40960;j++)assert(((unsigned char *)(uintptr_t)b[p].S2)[j]==0x34);
            for(unsigned j=0;j<40960;j++)assert(((unsigned char *)(uintptr_t)b[p].S)[j]==0xa5);
            for(unsigned j=0;j<16;j++)assert(((unsigned char *)(uintptr_t)(b[p].t+1280))[j]==0xa5);
        }
        assert(prog_batch_fuse_bf16(&program,1)==96*4);
        for(unsigned j=0;j<256;j++)assert(((unsigned char *)memory)[(size_t)WIDTH*WIDE_ARENA_BYTES+j]==0xa5);
        printf("wide HCR die=%d hn=%d djn=%d hcr_groups=96 extra_descriptors=%d extra_bytes=%zu S2/T/ne0_lifetimes PASS\n",d,shape.hn,shape.djn,program.n-original,(size_t)(program.n-original)*sizeof(phase_t));
        prog_free(&program);
    }
    free(memory);
}

static void wide_image_owner(void) {
    core_t fake={0};fake.nd=6;core=&fake;embed_die=0;MAXT=8;img_rows_max=4096;
    for(int d=0;d<6;d++)fake.ctx[d]=(CUcontext)(uintptr_t)(d+1);
    memset(db,0,sizeof db);memset(has_attn,0,sizeof has_attn);
    db[0].img=0x100000;db[0].imap=0x200000;
    for(int d=0;d<6;d++) {
        prog_t p;prog_init(&p);dn_weights_t w={.token_embd=d==embed_die?0x300000:0};
        xchg_t x={.inbox=0x400000,.xtab=0x500000};
        prog_embed_bcast(&p,&x,d,6,embed_die,&w,&db[d],0);
        unsigned embeds=0;
        for(int i=0;i<p.n;i++)if(p.p[i].w[0]==OP_EMBROW) {
            embeds++;assert(d==embed_die);
            assert((p.p[i].w[10]|((uint64_t)p.p[i].w[11]<<32))==db[0].imap);
            assert((p.p[i].w[12]|((uint64_t)p.p[i].w[13]<<32))==db[0].img);
        }
        assert(embeds==(unsigned)(d==embed_die));prog_free(&p);
    }
    char path[]="/tmp/flash-image-owner-XXXXXX";int fd=mkstemp(path);assert(fd>=0);
    FILE *f=fdopen(fd,"wb");assert(f);
    uint32_t header[]={1,1,1,1,2560},positions[]={0,0,0};float row[2560];
    for(unsigned j=0;j<2560;j++)row[j]=(float)j/4096.0f;
    assert(fwrite(header,sizeof header,1,f)==1 && fwrite(row,sizeof row,1,f)==1);
    assert(fwrite(positions,sizeof positions,1,f)==1 && !fclose(f));
    engine_img image={.s=1};snprintf(image.path,sizeof image.path,"%s",path);
    h_imap=calloc((size_t)MAXT+2,4);assert(h_imap);imap_dirty=rope_dirty_T=0;
    image_test_mode=1;image_contexts=image_uploads=map_uploads=map_resets=0;
    assert(!prepare_images(&image,1,3,3));
    assert(image_uploads==1 && map_uploads==1 && image_contexts==1 && imap_dirty);
    assert(!prepare_images(NULL,0,3,3));
    assert(map_resets==1 && image_contexts==2 && !imap_dirty && !rope_dirty_T);
    assert(img_rows_max==4096);
    image_test_mode=0;free(h_imap);h_imap=NULL;assert(!unlink(path));
    memset(db,0,sizeof db);core=NULL;img_rows_max=0;
}

/* Exercise the real session parser/exporters without initializing CUDA. */
static void reject_case(const char *command) {
    core_t fake_core;
    memset(&fake_core, 0, sizeof fake_core);
    fake_core.nd = 1;
    core = &fake_core;
    MAXT = 32;
    trace_request = 41;
    trace_first = 4;
    trace_n = trace_valid = 1;
    trace_completed = trace_requested = 5;
    trace_aborted = trace_ended = 0;
    trace_status = "completed";
    memset(trace_host, 0, sizeof trace_host);
    trace_host[0].observed = 100;
    trace_host[0].callback_start = 101;
    trace_host[0].callback_end = 102;
    /* Even an unfixed trace_dump stays on its CPU validation path. */
    prof_nph[0] = 0;

    char input[512];
    int n = snprintf(input, sizeof input, "timings on\n%s\nhosttrace\ntrace 0\nquit\n", command);
    assert(n > 0 && (size_t)n < sizeof input);
    FILE *in = fmemopen(input, (size_t)n, "r");
    assert(in);
    char *output = NULL;
    size_t output_size = 0;
    FILE *out = open_memstream(&output, &output_size);
    assert(out);
    int rc = engine_session(in, out);
    assert(rc == 1);
    assert(fclose(in) == 0 && fclose(out) == 0);
    assert(output);
    assert(strstr(output, "error: gen|chat <n_new> <id>..."));
    assert(!strstr(output, "hosttrace_begin"));
    assert(!strstr(output, "trace_begin"));
    assert(!strstr(output, "status=completed"));
    assert(strstr(output, "request=42"));
    assert(strstr(output, "status=argument_error aborted=0 ended=0 completed=0 requested=0"));
    assert(trace_request == 42 && !trace_valid && !trace_completed && !trace_requested);
    assert(!trace_aborted && !trace_ended && !strcmp(trace_status, "argument_error"));
#ifdef FLASH_TIMINGS_H
    const char *timing_record = strstr(output, "timings {\"schema_version\":1");
    assert(timing_record && timing_record < strstr(output, "error: gen|chat"));
    assert(last_timing_valid && last_timing.error_code == -1);
    assert(last_timing.request_id==trace_request);
    assert(!last_timing.counts_known && !last_timing.counts_final);
    assert(!last_timing.first_valid && !last_timing.stop_known);
    assert(isnan(last_timing.build_ms) && isnan(last_timing.run_ms));
#endif
    free(output);
    core = NULL;
}

/* Fail each program allocation before any snapshot restore or device launch.
 * The real session must export the failure, never a successful timing record. */
static void program_allocation_failure(void) {
    core_t fake_core = {0};
    uint32_t prompt[] = {17, 23}, ids_buffer[34] = {0};
    uint32_t stop[] = {0xffffffffu, 0xffffffffu};
    fake_core.nd = 1; core = &fake_core;
    MAXT = 32; h_ids = ids_buffer;
    snap_s = ple_disk = imap_dirty = rope_dirty_T = 0;
    two_position = 1; prefill_chunk = 2;
    for (unsigned fail_at = 1; fail_at <= 3; fail_at++) {
        trace_request = 41; trace_n = trace_valid = 1;
        trace_status = "completed";
        calloc_calls = 0; calloc_fail_at = fail_at;
        assert(engine_run(prompt, 2, 2, stop, NULL, 0, NULL, 0, NULL, NULL) == -4);
        assert(calloc_calls == 3 && !in_run);
        assert(last_timing_valid && last_timing.error_code == -4);
        assert(last_timing.request_id == 42 && trace_request == 42);
        assert(!strcmp(trace_status, "allocation_error") && !trace_valid);
        assert(!last_timing.counts_known && !last_timing.counts_final);
        assert(!last_timing.first_valid && !last_timing.stop_known);
        assert(isfinite(last_timing.build_ms) && last_timing.build_ms >= 0);
        assert(isnan(last_timing.run_ms));
    }
    prefill_batch = 8;
    for (unsigned fail_at = 1; fail_at <= 4; fail_at++) {
        calloc_calls = 0; calloc_fail_at = fail_at;
        assert(engine_run(prompt, 2, 2, stop, NULL, 0, NULL, 0, NULL, NULL) == -4);
        assert(calloc_calls == (fail_at == 3 ? 3u : 4u) && !in_run);
        assert(last_timing.error_code == -4 && !last_timing.counts_known);
        assert(!strcmp(trace_status, "allocation_error"));
    }
    prefill_batch = 0;
    char input[] = "timings on\nchat 2 17 23\nhosttrace\ntrace 0\nquit\n";
    FILE *in = fmemopen(input, strlen(input), "r");
    char *output = NULL; size_t output_size = 0;
    FILE *out = open_memstream(&output, &output_size);
    assert(in && out && s_ids && s_toks);
    calloc_calls = 0; calloc_fail_at = 1;
    assert(engine_session(in, out) == 1);
    assert(!fclose(in) && !fclose(out));
    const char *record = strstr(output, "timings {");
    const char *error = strstr(output, "error: host program allocation failed\n");
    assert(record && error && record < error);
    assert(strstr(output, "\"error_code\":-4"));
    assert(strstr(output, "\"positions_completed\":null"));
    assert(strstr(output, "\"run_ms\":null"));
    assert(strstr(output, "status=allocation_error"));
    assert(!strstr(output, "done ") && !strstr(output, "trace_begin"));
    free(output);
    calloc_fail_at = 0; two_position = prefill_chunk = 0;
    h_ids = NULL; core = NULL; trace_n = 0;
}
int main(void) {
    assert(!setenv("SPLIT_SHARES","112,97,81,116,97,97",1));
    reject_case("chat nonsense");
    reject_case("gen -1");
    reject_case("chat");
    reject_case("gen 2147483648");
    reject_case("chat 999999999999999999999999999999999999999999");
    session_options();half_close_options();
    callback_return_before_cancel();
    program_allocation_failure();
    headless_arena_layout();
    wide_arena_lifetimes();wide_head_case(165544);wide_live_split();wide_hcr_decomposition();
    wide_image_owner();
    free(s_ids); free(s_toks); s_ids = s_toks = NULL;
    puts("trace_session_test: session/timing/cancellation/allocation, weighted wide arenas/ne=0, and 4096-row image ownership PASS");
    return 0;
}
