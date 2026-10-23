/* Per-exchange observations; clocks remain device-local until analysis. */
static void exchange_trace_dump(FILE *out,int d) {
    if(d<0 || d>=nd || !xtrace_n || !last_timing.counts_final || last_timing.error_code) {
        fprintf(out,"error: xtrace requires an enabled completed request\n");return;
    }
    int hash_only=getenv("FLASH_XTRACE_HASH_ONLY") && atoi(getenv("FLASH_XTRACE_HASH_ONLY"));
    size_t bytes=(size_t)xtrace_count*(gr_write_trace?128:4096)*2048;
    uint64_t *data=malloc(bytes);if(!data){fprintf(out,"error: xtrace allocation\n");return;}
    CK(cuCtxSetCurrent(ctx[d]));CK(cuMemcpyDtoH(data,dXTRACE[d],bytes));
    unsigned records=0,bad=0;
    for(unsigned t=0;t<xtrace_count;t++) {
        int sender=-1;unsigned gr_slot=0;
        for(int p=0;p<prof_nph[d];p++) {
            unsigned flags=prof_flags[d][p],index=prof_index[d][p];
            if(gr_write_trace) {
                if(prof_op[d][p]!=94 && prof_op[d][p]!=OP_BF16HCRWX)continue;
                const uint64_t *r=data+((size_t)t*128+gr_slot++)*256;
                uint64_t seen=0,hash=14695981039346656037ull;for(unsigned k=0;k<256;k++)seen|=r[k];
                if(!seen)continue;
                const unsigned char *raw=(const unsigned char *)r;
                for(unsigned k=0;k<2048;k++)hash=(hash^raw[k])*1099511628211ull;
                fprintf(out,"grhash request=%llu position=%u die=%d phase=%d layer=%d op=%s point=updated_residual hash=%016llx\n",last_timing.request_id,xtrace_first+t,d,p,prof_layer[d][index],profile_op_name(prof_op[d][p]),(unsigned long long)hash);
                records++;continue;
            }
            const uint64_t *acq=data+((size_t)t*4096+index)*256;
            if(!body_trace && acq[224]) {
                uint64_t hash=14695981039346656037ull;
                const unsigned char *bytes=(const unsigned char *)(acq+224);
                for(unsigned k=0;k<256;k++)hash=(hash^bytes[k])*1099511628211ull;
                fprintf(out,"grhash request=%llu position=%u die=%d phase=%d layer=%d op=%s hash=%016llx\n",last_timing.request_id,xtrace_first+t,d,p,prof_layer[d][index],profile_op_name(prof_op[d][p]),(unsigned long long)hash);
                if(hash_only)records++;
            }
            if(hash_only)continue;
            if(flags&2u) {
                if(sender<0){bad++;continue;}
                unsigned si=prof_index[d][sender];
                const uint64_t *src=data+((size_t)t*4096+si)*256;
                uint64_t ready=0,pub=0,start_min=UINT64_MAX,start_max=0,end=0,epoch=acq[5],arrivals[6]={0};
                int valid=epoch!=0;
                for(unsigned c=0;c<16;c++) {
                    const uint64_t *a=acq+8*c,*s=src+8*c;
                    if(!s[0] || !s[1] || !a[2] || !a[3] || s[4]!=epoch || a[5]!=epoch || s[1]<s[0] || a[3]<a[2])valid=0;
                    if(s[0]>ready)ready=s[0];
                    if(s[1]>pub)pub=s[1];
                    if(a[2]<start_min)start_min=a[2];
                    if(a[2]>start_max)start_max=a[2];
                    if(a[3]>end)end=a[3];
                }
                for(unsigned k=0;k<96;k++){uint64_t v=acq[128+k];if(!v)valid=0;if(v>arrivals[k/16])arrivals[k/16]=v;}
                if(end<ready)valid=0;
                fprintf(out,"{\"kind\":\"exchange_trace\",\"request\":%llu,\"position\":%u,\"die\":%d,\"epoch\":%llu,\"producer_phase\":%d,\"consumer_phase\":%d,\"producer_op\":\"%s\",\"consumer_op\":\"%s\",\"valid\":%s,\"ready_ns\":%llu,\"published_ns\":%llu,\"wait_start_min_ns\":%llu,\"wait_start_max_ns\":%llu,\"acquire_ns\":%llu,\"arrival_ns\":[",
                    last_timing.request_id,xtrace_first+t,d,(unsigned long long)epoch,sender,p,profile_op_name(prof_op[d][sender]),profile_op_name(prof_op[d][p]),valid?"true":"false",(unsigned long long)ready,(unsigned long long)pub,(unsigned long long)start_min,(unsigned long long)start_max,(unsigned long long)end);
                for(unsigned i=0;i<6;i++)fprintf(out,"%s%llu",i?",":"",(unsigned long long)arrivals[i]);
                const char *labels[]={"cta_ready_ns","cta_published_ns","cta_wait_start_ns","cta_acquire_ns"};
                for(unsigned k=0;k<4;k++) {
                    fprintf(out,"],\"%s\":[",labels[k]);
                    for(unsigned c=0;c<16;c++)fprintf(out,"%s%llu",c?",":"",(unsigned long long)(k<2?src:acq)[8*c+k]);
                }
                fprintf(out,"],\"producer_flag_observations_ns\":[");
                for(unsigned k=0;k<96;k++)fprintf(out,"%s%llu",k?",":"",(unsigned long long)acq[128+k]);
                fprintf(out,"]}\n");records++;bad+=!valid;
            }
            if(flags&1u)sender=p;
        }
    }
    fprintf(out,"xtrace_end die=%d records=%u invalid=%u clock=SR_GLOBALTIMER device_local=1\n",d,records,bad);free(data);
}
