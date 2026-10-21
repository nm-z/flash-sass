/* Included by engine.c. First validate rollback on known prompt positions;
 * no reference output is used as a proposal or fed into generation. */
static unsigned verify_executed,verify_replayed,verify_changed;
static int verify_check_state=1;
static void verify_init(void) {
    if(!verify_accept)return;
    const char *check=getenv("FLASH_VERIFY_STATE_CHECK");
    if(check && strcmp(check,"0") && strcmp(check,"1")) { fprintf(stderr,"engine: FLASH_VERIFY_STATE_CHECK must be0 or1\n");longjmp(init_jb,1); }
    verify_check_state=!check || strcmp(check,"0");
    if(exact_width!=8 || nd!=6 || nlayers_run!=48 || nblk_qsa!=512 || trace_n) {
        fprintf(stderr,"engine: verification audit requires B8, full48/six/B512 and tracing off\n");longjmp(init_jb,1);
    }
    size_t needed[MAXD];int fits=1;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CUfunction marker;
        CK(cuModuleGetFunction(&marker,exact_mod[d],"flash_verify_checkpoints_abi_v1"));
        size_t one=4*10*10240;
        for(int bb=0;bb<NB;bb++)one+=BLK_IS_ATTN(bb)?(dst_ring[d][bb]?2048:0):dn_bytes[d][bb]+conv_bytes[d][bb];
        needed[d]=6*one;size_t available,total;CK(cuMemGetInfo(&available,&total));
        fprintf(stderr,"engine: verifier checkpoint die=%d arena_bytes=%zu free_bytes=%zu\n",d,needed[d],available);
        if(needed[d]+8u*1024u*1024u>available)fits=0;
    }
    if(!fits) { fprintf(stderr,"engine: verifier checkpoints do not fit; audit disabled before allocation\n");verify_accept=0;return; }
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));CUdeviceptr arena=dalloc(needed[d]);size_t offset=0;
        for(unsigned p=0;p<7;p++) {
            for(int bb=0;bb<NB;bb++) {
                if(BLK_IS_ATTN(bb)) {
                    if(dst_ring[d][bb]) { verify_ring[p][d][bb]=p?arena+offset:snap_ring[d][bb];if(p)offset+=2048; }
                } else {
                    verify_dn[p][d][bb]=p?arena+offset:snap_dn[d][bb];if(p)offset+=dn_bytes[d][bb];
                    verify_conv[p][d][bb]=p?arena+offset:snap_conv[d][bb];if(p)offset+=conv_bytes[d][bb];
                }
            }
            verify_ple[p][d]=p?arena+offset:snap_ple[d];if(p)offset+=4*10*10240;
        }
        if(offset!=needed[d]) { fprintf(stderr,"engine: invalid checkpoint arena size\n");longjmp(init_jb,1); }
    }
}
static void verify_fini(void) {
    verify_accept=0;verify_executed=verify_replayed=verify_changed=0;verify_check_state=1;
    memset(verify_dn,0,sizeof verify_dn);memset(verify_conv,0,sizeof verify_conv);
    memset(verify_ring,0,sizeof verify_ring);memset(verify_ple,0,sizeof verify_ple);
}
static int build_verify_program(prog_t *out,int d) {
    prog_t one[8]={{0}};const prog_t *positions[8];
    const uint32_t stop[2]={0xffffffffu,0xffffffffu};
    for(unsigned p=0;p<8;p++) {
        /* UINT_MAX keeps all eight input ids teacher-forced and suppresses
         * stop handling until the eventual acceptance decision. */
        model_program(&one[p],d,p,UINT32_MAX,stop,0,1.0f,1.0f,1,1);positions[p]=&one[p];
    }
    prog_init(out);int rc=prog_group_aliased(out,positions,8);
    for(unsigned p=0;p<8;p++)prog_free(&one[p]);
    if(rc)return rc;
    if(fuse_exact_program(out,d)!=36)return -1;
    return 0;
}
static void verify_restore(unsigned kept) {
    if(kept<1 || kept>8) { fprintf(stderr,"engine: invalid verification commit\n");longjmp(run_jb,1); }
    if(kept==8)return;
    unsigned p=kept-1;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for(int bb=0;bb<NB;bb++) {
            if(BLK_IS_ATTN(bb)) {
                if(dst_ring[d][bb])CK(cuMemcpyDtoD(dst_ring[d][bb],verify_ring[p][d][bb],2048));
            } else {
                CK(cuMemcpyDtoD(dst_dn[d][bb],verify_dn[p][d][bb],dn_bytes[d][bb]));
                CK(cuMemcpyDtoD(dst_conv[d][bb],verify_conv[p][d][bb],conv_bytes[d][bb]));
            }
        }
        CK(cuMemcpyDtoD(dst_ple[d],verify_ple[p][d],4*10*10240));
    }
}
/* Same valid-prefix convention as the existing chunk-state check: ignore
 * uncommitted KV rows and unused columns of transposed pooled-key tiles. */
static void verify_fingerprints(uint64_t hashes[MAXD][NB+2],unsigned positions,unsigned stream_position) {
    unsigned char pool[8192];const uint64_t seed=0xcbf29ce484222325ull;
    for(int d=0;d<nd;d++) {
        CK(cuCtxSetCurrent(ctx[d]));
        for(int bb=0;bb<NB;bb++) {
            uint64_t h=seed;
            if(!BLK_IS_ATTN(bb)) {
                h=device_fnv(dst_dn[d][bb],dn_bytes[d][bb],h);h=device_fnv(dst_conv[d][bb],conv_bytes[d][bb],h);
            } else if(dst_kq[d][bb]) {
                h=device_fnv(dst_ring[d][bb],2048,h);
                h=device_fnv(dst_kq[d][bb],512ull*positions,h);h=device_fnv(dst_vq[d][bb],512ull*positions,h);
                h=device_fnv(dst_ks[d][bb],32ull*positions,h);h=device_fnv(dst_vs[d][bb],32ull*positions,h);
                unsigned groups=positions/4;
                for(unsigned at=0;at<groups;at+=32) {
                    unsigned n=groups-at<32?groups-at:32;
                    CK(cuMemcpyDtoH(pool,dst_pkt[d][bb]+8192ull*(at/32),sizeof pool));
                    for(unsigned row=0;row<128;row++) for(unsigned j=0;j<2*n;j++) { h^=pool[64*row+j];h*=0x100000001b3ull; }
                }
            }
            hashes[d][bb]=h;
        }
        hashes[d][NB]=device_fnv(dst_ple[d],4*10*10240,seed);
        hashes[d][NB+1]=device_fnv(position_buf(d,stream_position)->S,4*10240,seed);
    }
}
static void verify_position(runctx *r,unsigned position) {
    r->count=(int)position;
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemsetD32(dCTL[d]+0x8,position,1)); }
}
static void verify_audit_run(runctx *r,const prog_t *full,const prog_t *prefix,const prog_t *verification) {
    const unsigned start=16,width=8;unsigned phases[MAXD];
    uint64_t reference[MAXD][NB+2],actual[MAXD][NB+2];
    ple_pump(r,PLE_AHEAD);
    upload_programs(prefix,phases,1);run_segment(r,phases,(int)start);
    if(r->aborted)return;
    if(r->count!=(int)start) { fprintf(stderr,"engine: verifier start not reached\n");longjmp(run_jb,1); }
    if(verify_check_state) {
        /* The ordinary prefix snapshot is also verifier slot0. Restore it
         * before the batch starts overwriting that slot. */
        state_copy(1);run_segment(r,phases,(int)(start+verify_accept));
        if(r->aborted)return;
        verify_fingerprints(reference,start+verify_accept,0);
        state_copy(0);verify_position(r,start);
    }
    uint32_t original[8],poisoned[8];
    for(unsigned p=0;p<width;p++) {
        original[p]=h_ids[start+p];poisoned[p]=p<verify_accept?original[p]:(original[p]+1)%248320u;
    }
    /* Poison only discarded future embeddings, retaining the original PLE
     * rows. No such future-input change may affect the retained causal state. */
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(db[d].ids+4*start,poisoned,sizeof poisoned)); }
    upload_programs(verification,phases,width);run_segment(r,phases,(int)(start+width));
    if(r->aborted)return;
    if(r->count!=(int)(start+width)) { fprintf(stderr,"engine: verifier window incomplete\n");longjmp(run_jb,1); }
    verify_executed=width;verify_restore(verify_accept);verify_replayed=width-verify_accept;
    verify_position(r,start+verify_accept);
    for(int d=0;d<nd;d++) { CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyHtoD(db[d].ids+4*start,original,sizeof original)); }
    if(verify_check_state) {
        verify_fingerprints(actual,start+verify_accept,verify_accept-1);
        for(int d=0;d<nd;d++)for(int bb=0;bb<NB+2;bb++)if(actual[d][bb]!=reference[d][bb]) {
            if(verify_changed<5)fprintf(stderr,"engine: rollback fingerprint die=%d layer=%d expected=%016llx actual=%016llx\n",d,bb,(unsigned long long)reference[d][bb],(unsigned long long)actual[d][bb]);
            verify_changed++;
        }
    }
    /* Peer/grid generations stay monotonic; only logical model position rewinds. */
    upload_programs(prefix,phases,1);run_segment(r,phases,r->nprompt-1);
    if(!r->aborted) { upload_programs(full,phases,1);run_segment(r,phases,r->T); }
}
static void verify_stats(FILE *out) {
    fprintf(out,"verifystats kept=%u window_start=16 width=8 verified_positions=%u replayed_positions=%u state_check=%d changed_buffers=%u reference_positions=%u poisoned_positions=%u\n",
            verify_accept,verify_executed,verify_replayed,verify_check_state,verify_changed,verify_executed&&verify_check_state?verify_accept:0,verify_replayed);fflush(out);
}
