#ifndef FLASH_TIMINGS_H
#define FLASH_TIMINGS_H
#include <math.h>
#include <stddef.h>
#include <stdio.h>
/* Host times use CLOCK_MONOTONIC. Device events use one die's
 * SR_GLOBALTIMER; their epoch must never be subtracted from a host epoch. */
typedef struct {
    unsigned long long request_id;
    int prompt, resumed, completed, ended, aborted;
    int first_valid, device_first_valid;
    double build_ms, run_ms, first_ms, device_first_ms;
    int counts_known, counts_final, error_code, stop_known;
    unsigned long long device_first_start_ns, device_first_publish_ns;
    unsigned long long device_last_publish_ns;
    unsigned device_traced_tokens;
} flash_timing;
typedef struct {
    int generated, executed, prompt_processed, decode_intervals;
    int post_valid, prompt_rate_valid, decode_rate_valid, device_rate_valid;
    double post_ms, prompt_rate, decode_rate, device_decode_ms, device_rate;
} flash_timing_values;
static inline int flash_timing_nonnegative(double x) { return isfinite(x) && x>=0.; }
/* Check before addition, including callers that received an INT_MAX input. */
static inline int flash_timing_request_fits(int prompt,int extra,int capacity) {
    return prompt>=1 && extra>=0 && prompt<=capacity && extra<=capacity-prompt;
}
static inline flash_timing_values flash_timing_derive(const flash_timing *t) {
    flash_timing_values v={0};
    int c=t->completed>0?t->completed:0;
    int p=t->prompt>0?t->prompt:1;
    int r=t->resumed>0?t->resumed:0;
    v.generated=c>=p?c-(p-1):0;
    v.executed=c>r?c-r:0;
    int pp=c<p?c:p;
    v.prompt_processed=pp>r?pp-r:0;
    v.decode_intervals=v.generated>1?v.generated-1:0;
    v.post_valid=t->first_valid && flash_timing_nonnegative(t->first_ms)
        && flash_timing_nonnegative(t->run_ms) && t->run_ms>=t->first_ms;
    if(v.post_valid)v.post_ms=t->run_ms-t->first_ms;
    v.prompt_rate_valid=t->counts_known && t->counts_final && !t->error_code
        && v.post_valid && t->first_ms>0.;
    if(v.prompt_rate_valid)v.prompt_rate=1000.*v.prompt_processed/t->first_ms;
    v.decode_rate_valid=t->counts_known && t->counts_final && !t->error_code
        && v.post_valid && v.generated>=2 && v.post_ms>0.;
    if(v.decode_rate_valid)v.decode_rate=1000.*v.decode_intervals/v.post_ms;
    v.device_rate_valid=t->counts_known && t->counts_final && !t->error_code
        && t->device_first_valid && v.generated>=2
        && t->device_traced_tokens==(unsigned)v.generated
        && t->device_last_publish_ns>t->device_first_publish_ns;
    if(v.device_rate_valid) {
        v.device_decode_ms=(t->device_last_publish_ns-t->device_first_publish_ns)*1e-6;
        v.device_rate=1000.*v.decode_intervals/v.device_decode_ms;
    }
    return v;
}
static inline void flash_timing_device(flash_timing *t,unsigned long long start,
        unsigned long long first,unsigned long long last,unsigned count) {
    t->device_first_valid=0;t->device_traced_tokens=0;
    if(!count || !start || first<start || !first || last<first)return;
    t->device_first_valid=1;t->device_first_start_ns=start;
    t->device_first_publish_ns=first;t->device_last_publish_ns=last;
    t->device_first_ms=(first-start)*1e-6;t->device_traced_tokens=count;
}
static inline const char *flash_timing_number(char out[48],double x,int valid) {
    if(valid && flash_timing_nonnegative(x))snprintf(out,48,"%.17g",x);
    else snprintf(out,48,"null");
    return out;
}
static inline const char *flash_timing_count(char out[48],int x,int known) {
    if(known)snprintf(out,48,"%d",x);else snprintf(out,48,"null");
    return out;
}
static inline const char *flash_timing_stamp(char out[48],unsigned long long x,int valid) {
    if(valid)snprintf(out,48,"\"%llu\"",x);else snprintf(out,48,"null");
    return out;
}
static inline int flash_timing_json(char *out,size_t cap,const flash_timing *t) {
    flash_timing_values v=flash_timing_derive(t);
    char b[48],r[48],f[48],p[48],pr[48],dr[48],d[48],dd[48],dv[48];
    char resumed[48],completed[48],executed[48],prompt[48],generated[48],intervals[48];
    char start[48],first[48],last[48];
    flash_timing_number(b,t->build_ms,1);flash_timing_number(r,t->run_ms,1);
    flash_timing_number(f,t->first_ms,t->first_valid);flash_timing_number(p,v.post_ms,v.post_valid);
    flash_timing_number(pr,v.prompt_rate,v.prompt_rate_valid);flash_timing_number(dr,v.decode_rate,v.decode_rate_valid);
    flash_timing_number(d,t->device_first_ms,t->device_first_valid);
    flash_timing_number(dd,v.device_decode_ms,v.device_rate_valid);flash_timing_number(dv,v.device_rate,v.device_rate_valid);
    flash_timing_count(resumed,t->resumed,t->counts_known);flash_timing_count(completed,t->completed,t->counts_known);
    flash_timing_count(executed,v.executed,t->counts_known);flash_timing_count(prompt,v.prompt_processed,t->counts_known);
    flash_timing_count(generated,v.generated,t->counts_known);flash_timing_count(intervals,v.decode_intervals,t->counts_known);
    flash_timing_stamp(start,t->device_first_start_ns,t->device_first_valid);
    flash_timing_stamp(first,t->device_first_publish_ns,t->device_first_valid);
    flash_timing_stamp(last,t->device_last_publish_ns,t->device_first_valid);
    int n=snprintf(out,cap,
        "{\"schema_version\":1,\"request_id\":\"%llu\",\"clock\":\"CLOCK_MONOTONIC\",\"prompt_tokens\":%d,"
        "\"resumed_positions\":%s,\"positions_completed\":%s,\"positions_executed\":%s,"
        "\"prompt_positions_processed\":%s,\"generated_tokens\":%s,\"decode_intervals\":%s,"
        "\"counts_known\":%s,\"counts_final\":%s,\"error_code\":%d,"
        "\"counter_semantics\":\"last_observed_completed_positions\","
        "\"stop_token_observed\":%s,\"aborted\":%s,"
        "\"build_ms\":%s,\"run_ms\":%s,\"first_observed_ms\":%s,\"post_first_ms\":%s,"
        "\"prompt_positions_per_second\":%s,\"decode_tokens_per_second\":%s,"
        "\"device_clock\":\"SR_GLOBALTIMER\",\"device_origin\":\"first_generated_position_start\","
        "\"device_first_start_ns\":%s,\"device_first_publish_ns\":%s,\"device_last_traced_publish_ns\":%s,"
        "\"device_traced_tokens\":%u,\"device_first_publish_ms\":%s,"
        "\"device_decode_ms\":%s,\"device_decode_tokens_per_second\":%s,"
        "\"prompt_n\":%d,\"cache_n\":%s,\"prompt_ms\":%s,\"prompt_per_second\":%s,"
        "\"predicted_n\":%s,\"predicted_ms\":%s,\"predicted_per_second\":%s,\"decode_ms\":%s}",
        t->request_id,t->prompt,resumed,completed,executed,prompt,generated,intervals,
        t->counts_known?"true":"false",t->counts_final?"true":"false",t->error_code,
        t->stop_known?(t->ended?"true":"false"):"null",t->aborted?"true":"false",
        b,r,f,p,pr,dr,start,first,last,t->device_traced_tokens,d,dd,dv,
        t->prompt,resumed,f,pr,generated,p,dr,p);
    return n<0 || (size_t)n>=cap?-1:0;
}
#endif
