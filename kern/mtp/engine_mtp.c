/* Included by engine.c. The target and draft share read-only target head/RoPE
 * weights; every mutable draft buffer, peer flag and controller is independent. */
#include "mtp/load.h"
static int mtp_enabled,mtp_file_live,mtp_owner;
static gguf_file mtp_file;
static const gguf_tensor *mtp_embedding;
static mtp_weights_t mtp_weights[MAXD];
static split_t mtp_splits[MAXD];
static mtp_buffers_t mtp_buffers[MAXD];
static xchg_t mtp_exchange[MAXD];
static CUmodule mtp_module[MAXD];static CUfunction mtp_function[MAXD];
static CUdeviceptr mtp_control[MAXD],mtp_hidden[MAXD],mtp_device_program[3][MAXD];
static prog_t mtp_programs[3][MAXD];
static unsigned mtp_phases[3][MAXD],mtp_cache_positions;
static char mtp_model_path[1024],mtp_cubin_path[1024];static uint64_t mtp_cubin_fnv;
static unsigned mtp_steps,mtp_proposals,mtp_matches;
static unsigned mtp_full_positions,mtp_full_batches;
static unsigned mtp_index_reuses;
static double mtp_total_ms,mtp_head_ms;
static uint32_t mtp_expected_xgen[MAXD],mtp_target_xgen[MAXD];

static int mtp_check_completed(const char *stage,int d,const uint32_t control[48],unsigned end,
                                uint32_t expected,xchg_t *exchange) {
    uint32_t lo=control[32],hi=lo;
    for(unsigned c=1;c<16;c++) { if(control[32+c]<lo)lo=control[32+c];if(control[32+c]>hi)hi=control[32+c]; }
    int bad=(control[2]&0x7fffffffu)!=end || control[11] || control[8]!=expected || lo!=control[1] || hi!=control[1];
    if(bad) {
        uint32_t flags[16];CK(cuMemcpyDtoH(flags,exchange->flags+64u*(unsigned)d,sizeof flags));
        uint32_t flo=flags[0],fhi=flo;for(unsigned c=1;c<16;c++) { if(flags[c]<flo)flo=flags[c];if(flags[c]>fhi)fhi=flags[c]; }
        fprintf(stderr,"engine: %s completion invariant die=%d pos=%u end=%u expired=%u xgen=%u expected=%u own_flags=%u..%u grid=%u cta_grid=%u..%u\n",
                stage,d,control[2],end,control[11],control[8],expected,flo,fhi,control[1],lo,hi);
    }
    return bad;
}
static void mtp_inspect_program(int d,const prog_t *program,const uint32_t control[48]) {
    size_t words=(size_t)program->n*32;uint32_t *image=malloc(words*4),*stamps=malloc((size_t)program->n*4);
    if(!image||!stamps) { free(image);free(stamps);return; }
    CK(cuMemcpyDtoH(image,dPROG[d],words*4));CK(cuMemcpyDtoH(stamps,dCTL[d]+0x100,(size_t)program->n*4));
    unsigned differences=0,stale=0;
    for(size_t w=0;w<words;w++)if(image[w]!=program->p[w/32].w[w%32]) {
        if(differences<6)fprintf(stderr,"engine: target program changed die=%d phase=%zu word=%zu host=%08x device=%08x\n",d,w/32,w%32,program->p[w/32].w[w%32],image[w]);
        differences++;
    }
    /* Each observed failed launch is shorter than one second. An unsigned
     * timestamp delta beyond that interval identifies an older phase stamp. */
    for(int p=0;p<program->n;p++)if((uint32_t)(stamps[p]-control[15])>1000000000u) {
        if(stale<12)fprintf(stderr,"engine: target stale phase stamp die=%d phase=%d op=%u flags=%x delta=%u\n",d,p,program->p[p].w[0],program->p[p].w[30],(uint32_t)(stamps[p]-control[15]));
        stale++;
    }
    fprintf(stderr,"engine: target completed-program audit die=%d phases=%d changed_words=%u stale_stamps=%u\n",d,program->n,differences,stale);
    free(image);free(stamps);
}

static void mtp_fini(void) {
    if(core) for(int d=0;d<nd;d++) {
        if(mtp_module[d]) { cuCtxSetCurrent(ctx[d]);cuModuleUnload(mtp_module[d]); }
        for(int h=0;h<3;h++)prog_free(&mtp_programs[h][d]);
    }
    if(mtp_file_live)gguf_close(&mtp_file);
    memset(&mtp_file,0,sizeof mtp_file);mtp_file_live=0;mtp_embedding=NULL;mtp_enabled=0;
    memset(mtp_module,0,sizeof mtp_module);memset(mtp_function,0,sizeof mtp_function);
    memset(mtp_buffers,0,sizeof mtp_buffers);memset(mtp_weights,0,sizeof mtp_weights);
    memset(mtp_exchange,0,sizeof mtp_exchange);memset(mtp_control,0,sizeof mtp_control);
    memset(mtp_hidden,0,sizeof mtp_hidden);memset(mtp_device_program,0,sizeof mtp_device_program);
    mtp_model_path[0]=mtp_cubin_path[0]=0;mtp_cubin_fnv=0;
}

static void mtp_init(void) {
    const char *option=getenv("FLASH_MTP_AUDIT");
    if(!option || !strcmp(option,"0"))return;
    if((strcmp(option,"1") && strcmp(option,"2") && strcmp(option,"3")) || nd!=6 || nlayers_run!=48 || nblk_qsa!=512) {
        fprintf(stderr,"engine: MTP requires mode1(shadow) or2(unchecked rollout), six dies, full48/B512\n");longjmp(init_jb,1);
    }
    if((getenv("FLASH_FUSED_MOE") && atoi(getenv("FLASH_FUSED_MOE"))) ||
       (getenv("FLASH_FAST_EXCHANGE") && atoi(getenv("FLASH_FAST_EXCHANGE"))) ||
       (getenv("FLASH_CHUNK_PREFILL") && atoi(getenv("FLASH_CHUNK_PREFILL")))) {
        fprintf(stderr,"engine: MTP audit requires ordinary target phases\n");longjmp(init_jb,1);
    }
    const char *model=getenv("FLASH_MTP_MODEL"),*cubin=getenv("FLASH_MTP_CUBIN");
    if(!model||!*model||strlen(model)>=sizeof mtp_model_path||!cubin||!*cubin||strlen(cubin)>=sizeof mtp_cubin_path) {
        fprintf(stderr,"engine: MTP audit requires FLASH_MTP_MODEL and FLASH_MTP_CUBIN\n");longjmp(init_jb,1);
    }
    strcpy(mtp_model_path,model);strcpy(mtp_cubin_path,cubin);mtp_cubin_fnv=file_fnv(cubin);
    mtp_file_live=1;
    if(gguf_open(&mtp_file,model) || !(mtp_embedding=mtp_embedding_tensor(&mtp_file))) {
        fprintf(stderr,"engine: incompatible MTP checkpoint\n");longjmp(init_jb,1);
    }
    mtp_owner=attn_die_of(48);
    if(mtp_owner<0||mtp_owner>=nd||!db[mtp_owner].rope_cs) { fprintf(stderr,"engine: MTP owner lacks a RoPE table\n");longjmp(init_jb,1); }
    mtp_cache_positions=qsa_cache_positions((unsigned)MAXT);
    /* The target's expert partition need not fit an additional Q8 draft. Plan
     * independent contiguous draft ranges using actual free bytes on all six
     * target dies. Geometry for HC/output rows retains the resident split. */
    size_t free_bytes[MAXD],fixed_bytes[MAXD],reserve[MAXD];unsigned capacity[MAXD],total_capacity=0;
    int fixed_fit=1;
    const size_t expert_bytes=2u*640u*2720u + 2560u*688u;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));size_t total;
        CK(cuMemGetInfo(&free_bytes[d],&total));
        split_of(&mtp_splits[d],d,nd,48);
        split_t empty=mtp_splits[d];empty.e0=empty.en=0;mtp_weights_t unused;
        CK(mtp_load_weights(&unused,&mtp_file,&empty,&dw[d][0],mtp_owner,NULL,&fixed_bytes[d]));
        reserve[d]=(d==mtp_owner?64u:32u)*1024u*1024u;
        size_t fixed=fixed_bytes[d]+reserve[d];
        if(free_bytes[d]<=fixed)fixed_fit=0;
        capacity[d]=free_bytes[d]>fixed?(unsigned)((free_bytes[d]-fixed)/expert_bytes):0;
        if(capacity[d]>512)capacity[d]=512;
        total_capacity+=capacity[d];
        fprintf(stderr,"engine: MTP capacity die=%d free_bytes=%zu fixed_bytes=%zu reserve=%zu experts=%u\n",d,free_bytes[d],fixed_bytes[d],reserve[d],capacity[d]);
    }
    if(!fixed_fit || total_capacity<512) {
        fprintf(stderr,"engine: MTP requires512 expert slots; capacity=%u. Draft disabled before allocation.\n",total_capacity);
        gguf_close(&mtp_file);mtp_file_live=0;mtp_embedding=NULL;mtp_cubin_fnv=0;mtp_model_path[0]=mtp_cubin_path[0]=0;
        if(strcmp(option,"1"))longjmp(init_jb,1); /* Unchecked modes never fall back. */
        return;
    }
    unsigned prefix=0,assigned=0;
    for(int d=0;d<nd;d++) {
        unsigned next=(unsigned)(512ull*(prefix+capacity[d])/total_capacity);
        mtp_splits[d].e0=(int)assigned;mtp_splits[d].en=(int)(next-assigned);
        assigned=next;prefix+=capacity[d];
    }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CUfunction marker;
        CK(cuModuleLoad(&mtp_module[d],cubin));CK(cuModuleGetFunction(&marker,mtp_module[d],"flash_mtp_abi_v1"));
        CK(cuModuleGetFunction(&mtp_function[d],mtp_module[d],"die"));
        split_t split=mtp_splits[d];size_t bytes=0;
        CK(mtp_load_weights(&mtp_weights[d],&mtp_file,&split,&dw[d][0],mtp_owner,dalloc,&bytes));
        fprintf(stderr,"engine: MTP die=%d weights_bytes=%zu experts=%d..%d\n",d,bytes,split.e0,split.e0+split.en);
        mtp_buffers_t *mb=&mtp_buffers[d];dn_bufs_t *b=&mb->base;xchg_t *x=&mtp_exchange[d];
        clone_position(d,b,x,0);
        b->ids=dalloc(4*((size_t)MAXT+2));b->toks=dalloc(4*((size_t)MAXT+2));
        mb->embedding=dalloc(4*2560);mb->eh_concat=dalloc(4*4*5120);mb->expert_rows=dalloc(4*11*2560);
        mtp_control[d]=dalloc(8192);mtp_hidden[d]=dalloc(4*10240);
        x->flags=dalloc(512);x->xgen=mtp_control[d]+0x20;
        CUdeviceptr kq=0,ks=0,vq=0,vs=0,pkt=0,ring=0;
        if(d==mtp_owner) {
            kq=dalloc((size_t)mtp_cache_positions*512);vq=dalloc((size_t)mtp_cache_positions*512);
            ks=dalloc((size_t)mtp_cache_positions*32);vs=dalloc((size_t)mtp_cache_positions*32);
            pkt=dalloc((size_t)mtp_cache_positions/128*8192);ring=dalloc(2048);
        }
        qsa_bind_layer(b,nblk_qsa,kq,ks,vq,vs,pkt,ring);b->kv_rawmax=0;
        for(int head=0;head<3;head++) {
            prog_t *p=&mtp_programs[head][d];prog_init(p);p->loop=1;p->end=mtp_control[d]+0x14;
            if(mtp_program(p,&split,x,&mtp_weights[d],mb,mtp_hidden[d],0,0,eps,head==1) || p->n>512) {
                fprintf(stderr,"engine: cannot build MTP program on die%d\n",d);longjmp(init_jb,1);
            }
            if(head==2) {
                /* Reuse selection across speculative steps. Keep IDXQK to
                 * append pooled keys/ring state; omit scoring and top-k. */
                int n=0;
                for(int i=0;i<p->n;i++)if(p->p[i].w[0]!=OP_IDXSCORE && p->p[i].w[0]!=OP_IDXSEL)p->p[n++]=p->p[i];
                p->n=n;
            }
            mtp_phases[head][d]=(unsigned)p->n;
            mtp_device_program[head][d]=dalloc((size_t)p->n*sizeof(phase_t));
            CK(cuMemcpyHtoD(mtp_device_program[head][d],p->p,(size_t)p->n*sizeof(phase_t)));
        }
    }
    uint64_t table[12];
    for(int d=0;d<6;d++) { table[d]=mtp_exchange[d].inbox;table[6+d]=mtp_exchange[d].flags; }
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(mtp_exchange[d].xtab,table,sizeof table)); }
    CK(cuCtxSetCurrent(ctx[0]));mtp_enabled=atoi(option);
}

static void mtp_reset(void) {
    mtp_steps=mtp_proposals=mtp_matches=0;mtp_total_ms=mtp_head_ms=0;
    mtp_full_positions=mtp_full_batches=0;
    mtp_index_reuses=0;
    memset(mtp_expected_xgen,0,sizeof mtp_expected_xgen);
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));uint32_t header[32]={0};
        header[3]=(unsigned)d;header[4]=(unsigned)nd;header[9]=1;
        for(int peer=0;peer<6;peer++) {
            uint64_t ptr=mtp_exchange[peer].flags+64u*(unsigned)d;memcpy(&header[16+2*peer],&ptr,8);
        }
        uint64_t mine=mtp_exchange[d].flags;memcpy(&header[28],&mine,8);
        CK(cuMemsetD8(mtp_control[d],0,8192));CK(cuMemcpyHtoD(mtp_control[d],header,sizeof header));
        CK(cuMemsetD8(mtp_exchange[d].flags,0,512));CK(cuMemsetD8(mtp_hidden[d],0,4*10240));
        dn_bufs_t *b=&mtp_buffers[d].base;
        CK(cuMemsetD8(b->ids,0,4*((size_t)MAXT+2)));CK(cuMemsetD8(b->toks,0,4*((size_t)MAXT+2)));
        if(b->kq) {
            CK(cuMemsetD8(b->kq,0,(size_t)mtp_cache_positions*512));CK(cuMemsetD8(b->vq,0,(size_t)mtp_cache_positions*512));
            CK(cuMemsetD8(b->ks,0,(size_t)mtp_cache_positions*32));CK(cuMemsetD8(b->vs,0,(size_t)mtp_cache_positions*32));
            CK(cuMemsetD8(b->pkt,0,(size_t)mtp_cache_positions/128*8192));CK(cuMemsetD8(b->ring,0,2048));
        }
    }
}

static uint32_t mtp_step(unsigned position,uint32_t token,int head) {
    double begin=now();float embedding[2560];
    if(mtp_embedding_row(mtp_embedding,token,embedding)) { fprintf(stderr,"engine: invalid MTP input token\n");longjmp(run_jb,1); }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));dn_bufs_t *b=&mtp_buffers[d].base;
        CK(cuMemcpyHtoD(mtp_buffers[d].embedding,embedding,sizeof embedding));
        CK(cuMemcpyHtoD(b->ids+4u*position,&token,4));
        CK(cuMemsetD32(mtp_control[d]+0x8,position,1));
    }
    /* Complete synchronous host uploads on every peer before any persistent
     * kernel can wait on another die. No transfer may block peer admission. */
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        unsigned end=position+1,phases=mtp_phases[head][d];
        void *args[]={&mtp_device_program[head][d],&phases,&mtp_control[d],&end};
        CK(cuLaunchKernel(mtp_function[d],16,1,1,256,1,1,0,0,args,NULL));
    }
    for(;;) {
        int done=1;
        for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CUresult q=cuStreamQuery(0);if(q==CUDA_ERROR_NOT_READY)done=0;else CK(q); }
        if(done)break;
        if(now()-begin>60) { fprintf(stderr,"engine: MTP launch stalled at position%u\n",position);longjmp(run_jb,1); }
        struct timespec nap={0,50000};nanosleep(&nap,NULL);
    }
    uint32_t prediction=0xffffffffu;
    int bad=0;
    for(int d=0;mtp_enabled==1 && d<nd;d++) {
        uint32_t control[48];CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(control,mtp_control[d],sizeof control));
        mtp_expected_xgen[d]+=(unsigned)mtp_programs[head][d].xj;
        bad+=mtp_check_completed("draft",d,control,position+1,mtp_expected_xgen[d],&mtp_exchange[d]);
    }
    if(bad)longjmp(run_jb,1);
    if(head==1) { CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(&prediction,mtp_buffers[0].base.toks+4u*position,4)); }
    double ms=(now()-begin)*1e3;mtp_total_ms+=ms;if(head==1)mtp_head_ms+=ms;mtp_steps++;
    mtp_index_reuses+=head==2;
    return prediction;
}

/* Known one-position target work needs no in-flight counter/token transfer.
 * Wait for every peer launch, then read and publish its completed result. This
 * also keeps the future verifier's unaccepted outputs off the streaming path. */
static void mtp_target_step(runctx *r,const unsigned phases[MAXD],unsigned end,const prog_t *program) {
    double begin=now();r->ntok=(int)end;
    unsigned reported=0;
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemsetD32(dCTL[d]+0x2c,0,1)); }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));void *args[]={&dPROG[d],(void *)&phases[d],&dCTL[d],&end};
        CK(cuLaunchKernel(fn[d],16,1,1,256,1,1,0,0,args,NULL));
    }
    for(;;) {
        int done=1;
        for(int d=0;d<nd;d++) {
            CK(cuCtxSetCurrent(ctx[d]));CUresult q=cuStreamQuery(0);
            if(q==CUDA_ERROR_NOT_READY)done=0;
            else {
                CK(q);
                if(mtp_enabled==1 && now()-begin>0.1 && !(reported&(1u<<d))) {
                    uint32_t control[48];CK(cuMemcpyDtoH(control,dCTL[d],sizeof control));
                    fprintf(stderr,"engine: target finished peer die=%d expected_end=%u actual_end=%u xgen=%u expected_xgen=%u grid=%u expired=%u\n",
                            d,end,control[2],control[8],mtp_target_xgen[d]+(unsigned)program[d].xj,control[1],control[11]);
                    if(mtp_check_completed("early-target",d,control,end,mtp_target_xgen[d]+(unsigned)program[d].xj,&xc[d]))mtp_inspect_program(d,&program[d],control);
                    reported|=1u<<d;
                }
            }
        }
        if(done)break;
        if(now()-begin>60) { fprintf(stderr,"engine: MTP target launch stalled before%u\n",end);longjmp(run_jb,1); }
        struct timespec nap={0,50000};nanosleep(&nap,NULL);
    }
    int bad=0;
    for(int d=0;mtp_enabled==1 && d<nd;d++) {
        uint32_t control[48];CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(control,dCTL[d],sizeof control));
        mtp_target_xgen[d]+=(unsigned)program[d].xj;
        bad+=mtp_check_completed("target",d,control,end,mtp_target_xgen[d],&xc[d]);
        if(!d)r->ended=!!(control[5]&0x80000000u);
    }
    if(bad)longjmp(run_jb,1);
    r->count=(int)end;trace_completed=end;last_timing.completed=end;
    last_timing.run_ms=(now()-timing_run_start)*1e3;
    if(r->count>r->sent) {
        CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(h_toks+r->sent,db[0].toks+4u*(unsigned)r->sent,4u*(unsigned)(r->count-r->sent)));
        if(!r->tfirst)r->tfirst=now();
        last_timing.first_valid=1;last_timing.first_ms=(r->tfirst-timing_run_start)*1e3;
        for(;r->sent<r->count;r->sent++) {
            int ti=trace_slot(r->sent);if(ti>=0)trace_host[ti].observed=host_ns();
            deliver_token(r,r->sent,h_toks[r->sent]);
        }
    }
    if(!r->aborted && r->cb && r->cb->alive && r->cb->alive(r->cb->arg))r->aborted=1;
}

/* MTP supplies the next input to a two-position invocation of the production
 * full model. Both positions run every layer and the target output head.
 * This deliberately has no proposal comparison, acceptance, or state rewind. */
static void mtp_full_run(runctx *r,const prog_t *full,const prog_t *prefix,const uint32_t stop[2]) {
    mtp_reset();unsigned phases[MAXD];
    prog_t *paired=new_request_programs();
    if(!paired) { fprintf(stderr,"engine: MTP paired program allocation failed\n");longjmp(run_jb,1); }
    const uint32_t no_stop[2]={UINT32_MAX,UINT32_MAX};
    for(int d=0;d<nd;d++) {
        prog_t first,second;
        model_program(&first,d,0,UINT32_MAX,no_stop,0,1.0f,1.0f,1,0);
        model_program(&second,d,1,UINT32_MAX,no_stop,0,1.0f,1.0f,1,0);
        prog_init(&paired[d]);
        int rc=prog_pair(&paired[d],&first,&second);
        prog_free(&first);prog_free(&second);
        if(rc) { fprintf(stderr,"engine: MTP full-model pair construction failed\n");longjmp(run_jb,1); }
        prog_pair_fuse_bf16(&paired[d],1);
    }
    upload_programs(prefix,phases,1);
    while(!r->aborted && r->count<r->nprompt-1) {
        unsigned p=(unsigned)r->count;
        ple_pump(r,8);mtp_step(p,h_ids[p],0);
        mtp_target_step(r,phases,p+1,prefix);
        for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoD(mtp_hidden[d],db[d].S,4*10240)); }
    }
    upload_programs(paired,phases,2);
    while(!r->aborted && !r->ended && r->count<r->T) {
        unsigned p=(unsigned)r->count,width=r->T-r->count>=2?2u:1u;
        uint32_t input[2]={p<(unsigned)r->nprompt?h_ids[p]:h_toks[p-1],0};
        input[1]=mtp_step(p,input[0],width==2);
        mtp_proposals+=width==2;
        for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(db[d].ids+4ull*p,input,4u*width)); }
        /* PLE hashes follow the actual proposed inputs consumed by the model.
         * These rows are now known; do not substitute any reference outputs. */
        for(unsigned j=0;j<width;j++)ple_seq[p+j]=input[j];
        runctx staging=*r;staging.nprompt=(int)(p+width);
        ple_pump(&staging,(int)width);
        if(width==1)upload_programs(full,phases,1);
        runctx pending=*r;pending.sent=(int)(p+width);pending.cb=NULL;
        mtp_target_step(&pending,phases,p+width,width==2?paired:full);
        if(width==2) {
            /* Advance draft cache for the second consumed input, using the
             * first position's full-model hidden state. No token comparison. */
            for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoD(mtp_hidden[d],db[d].S,4*10240)); }
            mtp_step(p+1,input[1],2);
        }
        for(int d=0;d<nd;d++) {
            CK(cuCtxSetCurrent(ctx[d]));
            CK(cuMemcpyDtoD(mtp_hidden[d],position_buf(d,width-1)->S,4*10240));
        }
        CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(h_toks+p,db[0].toks+4ull*p,4u*width));
        mtp_full_positions+=width;mtp_full_batches++;
        r->count=(int)(p+width);trace_completed=p+width;last_timing.completed=r->count;
        last_timing.run_ms=(now()-timing_run_start)*1e3;last_timing.counts_known=1;
        if(!r->tfirst)r->tfirst=now();
        last_timing.first_valid=1;last_timing.first_ms=(r->tfirst-timing_run_start)*1e3;
        for(;r->sent<r->count && !r->aborted && !r->ended;r->sent++) {
            uint32_t token=h_toks[r->sent];
            r->ended=token==stop[0] || token==stop[1];
            deliver_token(r,r->sent,token);
        }
        if(!r->aborted && r->cb && r->cb->alive && r->cb->alive(r->cb->arg))r->aborted=1;
    }
}

static void mtp_audit_run(runctx *r,const prog_t *full,const prog_t *prefix,const uint32_t stop[2]) {
    if(mtp_enabled==3) { mtp_full_run(r,full,prefix,stop);return; }
    mtp_reset();unsigned phases[MAXD];int uploaded=-1;
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(&mtp_target_xgen[d],dCTL[d]+0x20,4)); }
    while(!r->aborted && !r->ended && r->count<r->T) {
        unsigned position=(unsigned)r->count;
        int head=position>=(unsigned)(r->nprompt-1);
        uint32_t token=position<(unsigned)r->nprompt?h_ids[position]:h_toks[position-1];
        /* A one-position launch can finish before the in-flight PLE pump runs.
         * Stage the now-known next input before admitting its target kernel. */
        if(!head || mtp_enabled!=2)ple_pump(r,8);
        if(position<2 || (position&15u)==15u)fprintf(stderr,"engine: MTP shadow position=%u draft begin\n",position);
        uint32_t prediction=mtp_step(position,token,head);
        if(head && mtp_enabled==2) {
            /* Unchecked architectural rollout. The MTP block consumes the
             * previous hidden state plus current token embedding. Its hidden
             * output and predicted token become the next step's inputs.
             * No target launch, token comparison, acceptance, or replay. */
            h_toks[position]=prediction;mtp_proposals++;
            for(int d=0;d<nd;d++) {
                CK(cuCtxSetCurrent(ctx[d]));
                CK(cuMemcpyDtoD(mtp_hidden[d],mtp_buffers[d].base.S,4*10240));
            }
            r->count=(int)position+1;trace_completed=position+1;
            last_timing.completed=r->count;last_timing.counts_known=1;
            last_timing.run_ms=(now()-timing_run_start)*1e3;
            if(!r->tfirst)r->tfirst=now();
            last_timing.first_valid=1;last_timing.first_ms=(r->tfirst-timing_run_start)*1e3;
            r->ended=prediction==stop[0] || prediction==stop[1];
            deliver_token(r,(int)position,prediction);r->sent=r->count;
            if(!r->aborted && r->cb && r->cb->alive && r->cb->alive(r->cb->arg))r->aborted=1;
            continue;
        }
        if(position<2 || (position&15u)==15u)fprintf(stderr,"engine: MTP shadow position=%u target begin ple_ready=%u rid=%u\n",position,ple_disk?h_rdy[position]:0,ple_rid);
        int use_full=head||!prefix;
        if(use_full!=uploaded) { upload_programs(use_full?full:prefix,phases,1);uploaded=use_full; }
        mtp_target_step(r,phases,position+1,use_full?full:prefix);
        if(position<2 || (position&15u)==15u)fprintf(stderr,"engine: MTP shadow position=%u target completed=%d\n",position,r->count);
        if(r->aborted || r->count!=(int)position+1)break;
        for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoD(mtp_hidden[d],db[d].S,4*10240)); }
        if(head) { mtp_proposals++;mtp_matches+=prediction==h_toks[position]; }
    }
}

static void mtp_stats(FILE *out) {
    if(mtp_enabled==3) {
        fprintf(out,"mtpstats mode=full_model_unchecked enabled=3 draft_steps=%u proposals=%u target_generated_positions=%u target_batches=%u target_layers=48 batch_width=2 draft_ms=%.6f qsa_index_reuse_steps=%u comparisons=0 acceptance=0 recovery=0 fallback=0\n",
                mtp_steps,mtp_proposals,mtp_full_positions,mtp_full_batches,mtp_total_ms,mtp_index_reuses);
        fflush(out);return;
    }
    fprintf(out,"mtpstats shadow=%d enabled=%d steps=%u one_step_proposals=%u one_step_matches=%u draft_ms=%.6f head_draft_ms=%.6f rollout_tokens=%u target_verified=0\n",
            mtp_enabled!=2,mtp_enabled,mtp_steps,mtp_proposals,mtp_matches,mtp_total_ms,mtp_head_ms,mtp_enabled==2?mtp_proposals:0);
    fflush(out);
}
