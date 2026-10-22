/* All physical phases and CTAs are emitted, including missing/invalid records.
 * Device-local clocks must not be subtracted across GPUs. The body-minus-poll
 * field retains arithmetic, local barriers, publication and dispatch overhead;
 * it is not a claim that all remaining time is useful arithmetic. */
static void balance_body_dump(FILE *out,int d) {
    if(!body_trace || !row_balance || d<0 || d>=nd || !xtrace_n ||
       !last_timing.counts_final || last_timing.error_code) {
        fprintf(out,"error: bodytrace requires an enabled completed balanced request\n");return;
    }
    size_t bytes=(size_t)xtrace_count*4096*2048;
    uint64_t *data=malloc(bytes);if(!data){fprintf(out,"error: bodytrace allocation\n");return;}
    CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(data,dXTRACE[d],bytes));
    unsigned records=0,bad=0;
    for(unsigned t=0;t<xtrace_count;t++)for(int p=0;p<prof_nph[d];p++) {
        unsigned index=prof_index[d][p],op=prof_op[d][p],flags=prof_flags[d][p];
        const uint64_t *primary=data+((size_t)t*4096+index)*256;
        unsigned events=op==OP_BALANCED_SELECT?5u:
            ((flags&PHASE_BALANCED_PREPARE) || op==100?1u:0u);
        const char *qsa[]={"norm_index_grid","score_producers","selection_producer","gather_grid","partition_grid"};
        for(unsigned c=0;c<16;c++) {
            const uint64_t *a=primary+8*c;
            uint64_t elapsed=a[7]>=a[6]?a[7]-a[6]:0,total=0;
            int valid=a[6]!=0 && a[7]>=a[6];
            fprintf(out,"{\"kind\":\"balanced_body\",\"request\":%llu,\"position\":%u,\"die\":%d,\"physical_gpu\":%d,\"phase\":%d,\"descriptor\":%u,\"layer\":%d,\"op\":\"%s\",\"cta\":%u,\"resident_sm\":%llu,\"start_ns\":%llu,\"end_ns\":%llu,\"polls\":[",
                last_timing.request_id,xtrace_first+t,d,ar[d].die,p,index,prof_layer[d][index],profile_op_name(op),c,
                (unsigned long long)primary[224+2*c],(unsigned long long)a[6],(unsigned long long)a[7]);
            for(unsigned e=0;e<events;e++) {
                const uint64_t *v=primary+256*(1+e/4)+8*c+2*(e%4);
                int expected=!(op==OP_BALANCED_SELECT && e==1 && c!=0) && !(op==100 && c!=0);
                int observed=v[0]!=0 || v[1]!=0;
                int ok=observed?v[0]>=a[6] && v[1]>=v[0] && v[1]<=a[7]:!expected;
                uint64_t ns=ok && observed?v[1]-v[0]:0;total+=ns;valid&=ok;
                const char *name=op==OP_BALANCED_SELECT?qsa[e]:op==100?"router_producers":"moe_prepare_grid";
                fprintf(out,"%s{\"name\":\"%s\",\"expected\":%s,\"observed\":%s,\"valid\":%s,\"start_ns\":%llu,\"end_ns\":%llu}",
                    e?",":"",name,expected?"true":"false",observed?"true":"false",ok?"true":"false",(unsigned long long)v[0],(unsigned long long)v[1]);
            }
            valid&=total<=elapsed;
            fprintf(out,"],\"valid\":%s,\"elapsed_ns\":%llu,\"explicit_poll_ns\":%llu,\"body_minus_poll_ns\":%llu}\n",
                valid?"true":"false",(unsigned long long)elapsed,(unsigned long long)total,(unsigned long long)(valid?elapsed-total:0));
            records++;bad+=!valid;
        }
        if(op==OP_Q2MOEGN && flags&PHASE_BALANCED_ROW8) {
            const uint32_t *ids=(const uint32_t *)(primary+512);
            fprintf(out,"{\"kind\":\"balanced_experts\",\"request\":%llu,\"position\":%u,\"die\":%d,\"phase\":%d,\"layer\":%d,\"selected_count\":%u,\"selected_global_ids\":[",
                last_timing.request_id,xtrace_first+t,d,p,prof_layer[d][index],ids[10]);
            for(unsigned k=0;k<10;k++)fprintf(out,"%s%u",k?",":"",ids[k]);
            fprintf(out,"]}\n");
        }
    }
    fprintf(out,"bodytrace_end die=%d records=%u invalid=%u clock=SR_GLOBALTIMER device_local=1 trace_overhead_included=1\n",d,records,bad);
    free(data);
}
