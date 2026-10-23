/* All inputs in a verifier block are already known. Read its n*16 rows in
 * one pool submission while the launched target executes its early layers. */
static int balanced_ple_block(runctx *r,int budget) {
    if(!balanced_speculation || r->nprompt!=r->T || r->sent!=r->T ||
       ple_next<r->count || ple_next>=r->T)return 0;
    unsigned count=(unsigned)(r->T-ple_next);if(count>5 || count>(unsigned)budget)return 0;
    int first=ple_next,cache_fd=ple_cache_open(),read_fd=cache_fd>=0?cache_fd:ple_fd;
    off_t offsets[5*PLE_READ_HEADS];struct ple_read_result results[5*PLE_READ_HEADS];
    uint64_t begin=ple_dma_ns();
    for(unsigned p=0;p<count;p++) {
        unsigned q=(unsigned)first+p;uint64_t rows[16];uint32_t endid=ple_end_id;
        int ti=trace_slot((int)q);if(ti>=0)trace_host[ti].staging_start=host_ns();
        ple_rows_c(ple_seq,q,ple_absent_id,&endid,1,ple_multipliers,ple_head_offsets,ple_head_vocab,rows);
        for(unsigned h=0;h<16;h++) {
            offsets[p*16+h]=ple_off+(off_t)(rows[h]*90);
            posix_fadvise(read_fd,offsets[p*16+h],90,POSIX_FADV_WILLNEED);
        }
    }
    uint8_t *dst=ple_host+(size_t)first*1440;
    ple_read_batch_n(&ple_reads,read_fd,offsets,dst,results,count*16);
    if(cache_fd>=0) {
        int bad=0;for(unsigned i=0;i<count*16;i++)bad|=results[i].error || results[i].got!=90;
        close(cache_fd);
        if(bad){ple_cache_active=0;ple_read_batch_n(&ple_reads,ple_fd,offsets,dst,results,count*16);}
    }
    uint64_t read_end=ple_dma_ns();
    for(unsigned p=0;p<count;p++) {
        unsigned q=(unsigned)first+p;int ti=trace_slot((int)q);
        for(unsigned h=0;h<16;h++) {
            size_t got=results[p*16+h].got;
            if(ti>=0){trace_host[ti].bytes+=(unsigned)got;trace_host[ti].short_reads+=got<90;}
            if(got<90) {
                memset(dst+(p*16+h)*90+got,0,90-got);
                if(!ple_short_reads++)fprintf(stderr,"engine: PLE block read failed at position %u head %u\n",q,h);
            }
        }
        uint64_t queued=ple_dma_ns();
        if(ple_device_cache)CK(ple_dma_prepare(&ple_dma,q,dst+(size_t)p*1440,h_rdy+q,ple_rid));
        __atomic_store_n(h_rdy+q,ple_rid,__ATOMIC_RELEASE);
        if(ple_device_cache) {
            struct ple_dma_span span;CK(ple_dma_finish(&ple_dma,&span));
            uint64_t submitted=ple_dma_ns(),spread=span.last_end-span.first_end;
            ple_device_positions++;ple_device_row_bytes+=(uint64_t)nd*1440;ple_device_flag_bytes+=(uint64_t)nd*4;
            ple_device_queue_ms+=(submitted-queued)*1e-6;
            ple_enqueue_spread_sum+=spread;if(spread>ple_enqueue_spread_max)ple_enqueue_spread_max=spread;
            unsigned slot=q%16;ple_handoff[slot].position=(int)q;
            ple_handoff[slot].read_ns=read_end-begin;ple_handoff[slot].submit_ns=submitted-queued;ple_handoff[slot].span=span;
        }
        if(ti>=0){trace_host[ti].staging_end=host_ns();trace_host[ti].ready=ple_device_cache?0:trace_host[ti].staging_end;}
    }
    ple_next+=count;return 1;
}
