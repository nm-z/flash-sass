/* Included by engine.c. Greedy, lossless prompt-lookup speculation on the full
 * balanced target. Proposals use only actual prompt/committed input history.
 * No candidate output is delivered before target acceptance. */
static struct {
    unsigned blocks,drafted,accepted,rejected,recoveries,verified,replayed,committed;
    unsigned long long outer_phases,causal_replays,inner_joins,publications,acquires;
    unsigned long long snapshot_bytes,restore_bytes;
    unsigned long long native_hcup_phases;
    unsigned long long native_dn_phases;
    unsigned long long native_gr_phases;
    unsigned long long native_matrix_phases;
    unsigned long long native_moe_phases;
    unsigned long long local_quant_phases;
    double build_ms,draft_ms,decode_ms;
    unsigned verify_calls[6],replay_calls[6];
    double verify_ms[6],replay_ms[6];
    unsigned profiled[6];
    unsigned committed_histogram[5];
    unsigned long long state_selections;
    double maintenance_ms;
} bspec;
static unsigned balanced_draft_limit=4;
static const char *balanced_proposer_name(void) {
#ifdef FLASH_DFLASH_QUANT
    if(dflash_quant)return "dflash2_q4km_b4";
#endif
    return "prompt_lookup_1_16";
}

static unsigned balanced_lookup(unsigned position,unsigned limit,uint32_t out[5]) {
    out[0]=h_ids[position];unsigned known=position+1,maxn=known<16?known:16;
    for(unsigned n=maxn;n;n--) {
        /* Search most recent matching suffix first, but require continuation
         * tokens to be already present in the committed history. */
        for(unsigned end=known-1;end>=n;end--) {
            unsigned start=end-n;
            if(!memcmp(h_ids+start,h_ids+known-n,4u*n)) {
                unsigned count=known-end;if(count>limit)count=limit;
                memcpy(out+1,h_ids+end,4u*count);return count;
            }
            if(end==n)break;
        }
    }
    return 0;
}
static size_t balanced_snapshot_bytes(void) {
    size_t bytes=0;
    for(int d=0;d<nd;d++) {
        bytes+=4u*10u*10240u;
        for(int l=0;l<nlayers_run;l++)
            bytes+=BLK_IS_ATTN(l)?(dst_ring[d][l]?2048u:0u):dn_bytes[d][l]+conv_bytes[d][l];
    }
    return bytes;
}
/* Optional first-call profile from the existing outer phase timestamps. It
 * preserves the underlying opcode identity inside generic/native wrappers.
 * Readback time remains part of decode_work_ms; these are inclusive device
 * phase intervals, not isolated matrix-body or host-clock measurements. */
static void balanced_spec_profile(const prog_t *program,unsigned start,unsigned width) {
    const char *option=getenv("FLASH_SPEC_PROFILE_WIDTH");
    if(!option || (unsigned)atoi(option)!=width || bspec.profiled[width])return;
    bspec.profiled[width]=1;
    for(int d=0;d<nd;d++) {
        const prog_t *p=program+d;uint32_t *ts=malloc((size_t)p->n*4);
        if(!ts){fprintf(stderr,"engine: speculative profile allocation\n");longjmp(run_jb,1);}
        CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(ts,dCTL[d]+0x100,(size_t)p->n*4));
        double total[128]={0};unsigned count[128]={0},native[128]={0};int previous=-1;
        for(int i=0;i<p->n;) {
            const phase_t *a=p->p+i;unsigned op=a->w[0];
            int fused=op==OP_BALANCED_HCUP5;
            if(op>=OP_DFLASH_DN4 && op<=OP_DFLASH_BF16_4)op=a->w[4];
            if(op==OP_PAIRED_DOWN4)op=a->w[4];
            if(op==OP_DFLASH_ROUTER4)op=a->w[4];
            if(op==OP_BALANCED_BATCH || op==OP_BALANCED_PREQUANT || fused)op=a->w[4];
            if(previous>=0 && op<128) {
                total[op]+=(uint32_t)(ts[i]-ts[previous])/1e3;count[op]++;native[op]+=fused;
            }
            previous=i;i+=(int)phase_descriptors(a);
        }
        free(ts);
        for(unsigned op=0;op<128;op++)if(count[op])
            fprintf(stderr,"specprofile start=%u width=%u die=%d op=%s phases=%u native=%u us=%.3f\n",
                start,width,d,profile_op_name(op),count[op],native[op],total[op]);
        if(native[96] && getenv("FLASH_HCUP5_STAGE_TRACE") && atoi(getenv("FLASH_HCUP5_STAGE_TRACE"))) {
            if(0x100u+4u*(unsigned)p->n>0x6000u){fprintf(stderr,"engine: HCUP trace overlaps phase timestamps\n");longjmp(run_jb,1);}
            unsigned rows=0;
            for(int i=0;i<p->n;i+=(int)phase_descriptors(p->p+i))
                if(p->p[i].w[0]==OP_BALANCED_HCUP5)rows=p->p[i+1].w[8];
            uint32_t stages[16][8];CK(cuMemcpyDtoH(stages,dCTL[d]+0x6000,sizeof stages));
            for(unsigned c=0;c<16;c++)fprintf(stderr,
                "hcupstage start=%u width=%u die=%d cta=%u sm=%u matrix_active=%u input_us=%.3f matrix_us=%.3f gate_quant_us=%.3f publish_us=%.3f timing=instrumented_last_hcup\n",
                start,width,d,c,stages[c][5],c*32<rows/4,
                (uint32_t)(stages[c][1]-stages[c][0])/1e3,
                (uint32_t)(stages[c][2]-stages[c][1])/1e3,
                (uint32_t)(stages[c][3]-stages[c][2])/1e3,
                (uint32_t)(stages[c][4]-stages[c][3])/1e3);
        }
    }
}
static void balanced_stage(runctx *r,unsigned start,unsigned width,const uint32_t *input,int replay) {
    (void)r;
    memcpy(h_ids+start,input,4u*width);
    for(unsigned p=0;p<width;p++)ple_seq[start+p]=input[p];
    /* A previous launch may have queued lookahead DMA. Complete it before
     * invalidating a position whose proposed token is about to change. */
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CK(cuStreamSynchronize(pollst[d]));
        CK(cuMemcpyHtoD(db[d].ids+4ull*start,input,4u*width));
        CK(cuMemsetD32(dCTL[d]+0x8,start,1));
        CK(cuMemsetD32(dCTL[d]+0x14,0,2));
        if(balanced_prequant_ready[d])CK(cuMemsetD32(balanced_prequant_ready[d],0,16));
        if(ple_device_cache)CK(cuMemsetD32(ple_device_ready[d]+4ull*start,0,width));
        /* Replay can reuse exactly the same token/layer epoch. Clear its
         * local completion slots before any producer or consumer launches. */
        for(unsigned p=0;p<width;p++) {
            dn_bufs_t *b=position_buf(d,p);
            CK(cuMemsetD32(b->r+0x804,0,16));
            CK(cuMemsetD32(b->balance_quant_ready,0,16));
            CK(cuMemsetD32(b->balance_attn_ready,0,16));
            if(unique_rows)CK(cuMemsetD32(b->ne+4,0,1));
            if(replay && paired_columns)CK(cuMemsetD32(b->paired_flags,0,PAIR_FLAG_WORDS));
        }
    }
    if(ple_disk) {
        memset(h_rdy+start,0,4u*width);__atomic_thread_fence(__ATOMIC_RELEASE);
        ple_next=(int)start;
        /* Launch first; PLE staging overlaps embedding and the first layer. */
    }
}
static void balanced_execute(runctx *r,const prog_t *program,unsigned start,unsigned width,
                             const balance_batch_work *work,int replay) {
    double begin=now();
    unsigned nph[MAXD];upload_programs(program,nph,width);
    runctx pending=*r;pending.count=(int)start;
    pending.T=pending.nprompt=pending.sent=(int)(start+width);pending.cb=NULL;
    run_segment(&pending,nph,(int)(start+width));
    if(pending.aborted || pending.count!=(int)(start+width)) {
        fprintf(stderr,"engine: incomplete balanced speculative target block\n");longjmp(run_jb,1);
    }
    bspec.outer_phases+=work->outer_phases;
    last_position_slot=width-1;
    bspec.causal_replays+=work->causal_phase_replays;
    bspec.inner_joins+=work->inner_grid_joins;
    bspec.publications+=work->publications;bspec.acquires+=work->peer_acquires;
    bspec.native_hcup_phases+=work->native_hcup_phases;
    bspec.native_dn_phases+=work->native_dn_phases;
    bspec.native_gr_phases+=work->native_gr_phases;
    bspec.native_matrix_phases+=work->native_matrix_phases;
    bspec.native_moe_phases+=work->native_moe_phases;
    bspec.local_quant_phases+=work->local_quant_phases;
    if(replay){bspec.replay_calls[width]++;bspec.replay_ms[width]+=(now()-begin)*1e3;}
    else {bspec.verify_calls[width]++;bspec.verify_ms[width]+=(now()-begin)*1e3;}
    if(!replay)balanced_spec_profile(program,start,width);
}
static void balanced_spec_run(runctx *r,const prog_t *full,const prog_t *prefix,const uint32_t stop[2]) {
    memset(&bspec,0,sizeof bspec);last_extra_executed=0;
    unsigned capacity=BALANCED_BATCH_MAX;
    int retained_prefixes=0;
#ifdef FLASH_DFLASH_QUANT
    retained_prefixes=dflash_quant;
    if(dflash_quant)capacity=4;
#endif
    balanced_draft_limit=capacity-1;
    const char *limit_env=getenv("FLASH_SPEC_MAX_DRAFTS");
    if(limit_env) {
        char *end;unsigned long n=strtoul(limit_env,&end,10);
        if(!*limit_env || *end || n>=capacity){fprintf(stderr,"engine: FLASH_SPEC_MAX_DRAFTS must be0..%u\n",capacity-1);longjmp(run_jb,1);}
        balanced_draft_limit=(unsigned)n;
    }
    prog_t *program[6]={0};balance_batch_work work[6]={{0}};
    const uint32_t no_stop[2]={UINT32_MAX,UINT32_MAX};double build_start=now();
    for(unsigned width=1;width<=capacity;width++) {
        program[width]=new_request_programs();
        if(!program[width]) {fprintf(stderr,"engine: speculative program allocation\n");longjmp(run_jb,1);}
        for(int d=0;d<nd;d++) {
            if(width==1) {
                model_program(&program[width][d],d,0,UINT32_MAX,no_stop,0,1,1,1,retained_prefixes?2:0);
                balance_batch_work w={.positions=1};
                for(int i=0;i<program[width][d].n;) {
                    phase_t *a=program[width][d].p+i;w.outer_phases++;w.causal_phase_replays++;
                    w.publications+=(a->w[30]&1u)!=0;w.peer_acquires+=(a->w[30]&2u)!=0;
                    i+=(int)phase_descriptors(a);
                }
                if(!d)work[width]=w;
                else if(memcmp(&work[width],&w,sizeof w)) {fprintf(stderr,"engine: speculative scalar geometry differs across dies\n");longjmp(run_jb,1);}
            } else {
                prog_t one[5]={{0}};const prog_t *positions[5];balance_batch_work w;
                for(unsigned p=0;p<width;p++) {
                    model_program(&one[p],d,p,UINT32_MAX,no_stop,0,1,1,1,retained_prefixes?2:0);positions[p]=one+p;
                }
                prog_init(&program[width][d]);int rc=prog_balance_batch(&program[width][d],positions,width,&w,balanced_prequant_ready[d]);
                for(unsigned p=0;p<width;p++)prog_free(one+p);
                if(rc || (d && memcmp(&work[width],&w,sizeof w))) {
                    fprintf(stderr,"engine: incompatible balanced speculative program width=%u die=%d rc=%d\n",width,d,rc);longjmp(run_jb,1);
                }
                if(!d)work[width]=w;
            }
        }
    }
    bspec.build_ms=(now()-build_start)*1e3;
    run_prefill_prefix(r,full,prefix,NULL,NULL,r->nprompt-1);
#ifdef FLASH_DFLASH_QUANT
    if(dflash_quant) {
        char error[512]={0};
        dflash_quant_error(dflash_resident_begin(dflash_quant_resident,(unsigned)r->count,error,sizeof error),error);
    }
#endif
    /* Block snapshots now own the existing snapshot allocations. */
    snap_s=0;size_t state_bytes=balanced_snapshot_bytes();double decode_start=now();
    while(!r->aborted && !r->ended && r->count<r->T) {
        unsigned start=(unsigned)r->count,remaining=(unsigned)(r->T-r->count);
        unsigned limit=remaining>capacity?capacity-1:remaining-1;uint32_t input[5],output[5];
        if(limit>balanced_draft_limit)limit=balanced_draft_limit;
        double draft_start=now();unsigned drafts;
#ifdef FLASH_DFLASH_QUANT
        if(dflash_quant) {
            char error[512]={0};
            dflash_quant_error(dflash_resident_propose(dflash_quant_resident,h_ids[start],input,error,sizeof error),error);
            drafts=limit;
        } else
#endif
        drafts=balanced_lookup(start,limit,input);
        bspec.draft_ms+=(now()-draft_start)*1e3;
        unsigned width=drafts+1;bspec.blocks++;bspec.drafted+=drafts;
        if(width>1 && !retained_prefixes){state_copy(1);bspec.snapshot_bytes+=state_bytes;}
        balanced_stage(r,start,width,input,0);
        balanced_execute(r,program[width],start,width,&work[width],0);bspec.verified+=width;
        CK(cuCtxSetCurrent(ctx[0]));CK(cuMemcpyDtoH(output,db[0].toks+4ull*start,4u*width));
        unsigned matched=drafts;
        for(unsigned p=1;p<width;p++)if(input[p]!=output[p-1]){matched=p-1;break;}
        unsigned keep=matched+1;
        for(unsigned p=0;p<keep;p++)if(output[p]==stop[0] || output[p]==stop[1]) {
            keep=p+1;r->ended=1;break;
        }
        if(keep<width && keep>matched)bspec.rejected++;
#ifdef FLASH_DFLASH_QUANT
        if(dflash_quant) {
            double maintenance_start=now();char error[512]={0};
            dflash_prefix_select(keep-1);bspec.state_selections++;
            last_position_slot=keep-1;
            dflash_quant_error(dflash_resident_append(dflash_quant_resident,start,keep,error,sizeof error),error);
            dflash_quant_error(dflash_resident_finish(dflash_quant_resident,error,sizeof error),error);
            bspec.maintenance_ms+=(now()-maintenance_start)*1e3;
        } else
#endif
        if(keep<width) {
            bspec.recoveries++;
            state_copy(0);bspec.restore_bytes+=state_bytes;
            balanced_stage(r,start,keep,input,1);
            balanced_execute(r,program[keep],start,keep,&work[keep],1);bspec.replayed+=keep;
        }
        bspec.accepted+=matched<keep?matched:keep;bspec.committed+=keep;
        bspec.committed_histogram[keep]++;
        last_extra_executed=(int)(bspec.verified+bspec.replayed-bspec.committed);
        memcpy(h_toks+start,output,4u*keep);
        h_ids[start+keep]=output[keep-1];ple_seq[start+keep]=output[keep-1];
        r->count=(int)(start+keep);last_timing.completed=r->count;
        if(!r->tfirst)r->tfirst=now();
        for(;r->sent<r->count && !r->aborted;r->sent++)deliver_token(r,r->sent,h_toks[r->sent]);
        if(!r->aborted && r->cb && r->cb->alive && r->cb->alive(r->cb->arg))r->aborted=1;
#ifdef FLASH_DFLASH_QUANT
        if(dflash_quant)fprintf(stderr,
            "spec_round proposer=dflash2_q4km_b4 start=%u proposed=%u matched=%u retained_inputs=%u committed=%u state_bank=%u replayed=0 complete_ms=%.6f\n",
            start,drafts,matched,keep,keep,dflash_state_bank,(now()-draft_start)*1e3);
#endif
    }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));uint32_t count=(uint32_t)r->count|(r->ended?0x80000000u:0u);
        uint32_t ended=r->ended?0x80000000u:0u;
        /* Preserve the scalar API's final next-input slot even when no later
         * speculative stage will upload that replacement or bonus token. */
        CK(cuMemcpyHtoD(db[d].ids+4ull*(unsigned)r->count,h_ids+r->count,4));
        CK(cuMemcpyHtoD(dCTL[d]+0x8,&count,4));CK(cuMemcpyHtoD(dCTL[d]+0x14,&ended,4));
    }
#ifdef FLASH_DFLASH_QUANT
    if(dflash_quant) {
        char error[512]={0};double maintenance_start=now();
        dflash_quant_error(dflash_resident_finish(dflash_quant_resident,error,sizeof error),error);
        bspec.maintenance_ms+=(now()-maintenance_start)*1e3;
    }
#endif
    bspec.decode_ms=(now()-decode_start)*1e3;
}
static void balanced_spec_stats(FILE *out) {
    fprintf(out,"specstats enabled=%d proposer=%s max_drafts=%u greedy_acceptance=1 blocks=%u proposed=%u accepted_drafts=%u rejected_blocks=%u recovery_blocks=%u verified_positions=%u replayed_positions=%u committed_tokens=%u outer_phases=%llu causal_phase_replays=%llu inner_grid_joins=%llu publications=%llu peer_acquires=%llu snapshot_bytes=%llu restore_bytes=%llu program_build_ms=%.6f draft_ms=%.6f decode_work_ms=%.6f decode_work_tps=%.9f synchronization_phases_per_committed=%.9f",
        balanced_speculation,balanced_proposer_name(),balanced_draft_limit,bspec.blocks,bspec.drafted,bspec.accepted,bspec.rejected,bspec.recoveries,
        bspec.verified,bspec.replayed,bspec.committed,bspec.outer_phases,bspec.causal_replays,
        bspec.inner_joins,bspec.publications,bspec.acquires,bspec.snapshot_bytes,bspec.restore_bytes,
        bspec.build_ms,bspec.draft_ms,bspec.decode_ms,bspec.decode_ms>0?1e3*bspec.committed/bspec.decode_ms:0,
        bspec.committed?(double)(bspec.outer_phases+bspec.inner_joins)/bspec.committed:0);
    for(unsigned w=1;w<=5;w++)fprintf(out," w%u_verify_calls=%u w%u_verify_ms=%.6f w%u_replay_calls=%u w%u_replay_ms=%.6f",
        w,bspec.verify_calls[w],w,bspec.verify_ms[w],w,bspec.replay_calls[w],w,bspec.replay_ms[w]);
    fprintf(out," native_hcup_phases=%llu native_dn4_phases=%llu native_gr4_phases=%llu",bspec.native_hcup_phases,bspec.native_dn_phases,bspec.native_gr_phases);
    fprintf(out," native_matrix4_phases=%llu",bspec.native_matrix_phases);
    fprintf(out," native_moe4_phases=%llu",bspec.native_moe_phases);
    fprintf(out," receiver_quant_phases=%llu",bspec.local_quant_phases);
    fprintf(out," state_bank_selections=%llu maintenance_host_ms=%.6f",bspec.state_selections,bspec.maintenance_ms);
    for(unsigned n=1;n<=4;n++)fprintf(out," committed_%u_blocks=%u",n,bspec.committed_histogram[n]);
    fputc('\n',out);
    fflush(out);
}
