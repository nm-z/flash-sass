#ifndef FLASH_TIMINGS_H
#define FLASH_TIMINGS_H
#include <math.h>
#include <stddef.h>
#include <stdio.h>
/* Host first_ms: run start -> first generated-token observation.
 * Device first_ms has its own kernel-start origin. Never mix clocks. */
typedef struct {
    int prompt, resumed, completed, ended, aborted;
    int first_valid, device_first_valid;
    double build_ms, run_ms, first_ms, device_first_ms;
} flash_timing;
typedef struct {
    int generated, executed, prompt_processed, decode_intervals;
    int post_valid, prompt_rate_valid, decode_rate_valid;
    double post_ms, prompt_rate, decode_rate;
} flash_timing_values;
static inline int flash_timing_nonnegative(double x) { return isfinite(x) && x>=0.; }
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
    v.prompt_rate_valid=v.post_valid && t->first_ms>0.;
    if(v.prompt_rate_valid)v.prompt_rate=1000.*v.prompt_processed/t->first_ms;
    v.decode_rate_valid=v.post_valid && v.generated>=2 && v.post_ms>0.;
    if(v.decode_rate_valid)v.decode_rate=1000.*v.decode_intervals/v.post_ms;
    return v;
}
static inline const char *flash_timing_number(char out[48],double x,int valid) {
    if(valid && flash_timing_nonnegative(x))snprintf(out,48,"%.9g",x);
    else snprintf(out,48,"null");
    return out;
}
static inline int flash_timing_json(char *out,size_t cap,const flash_timing *t) {
    flash_timing_values v=flash_timing_derive(t);
    char b[48],r[48],f[48],p[48],pr[48],dr[48],d[48];
    int n=snprintf(out,cap,
        "{\"schema_version\":1,\"prompt_tokens\":%d,\"resumed_positions\":%d,"
        "\"positions_completed\":%d,\"positions_executed\":%d,"
        "\"prompt_positions_processed\":%d,\"generated_tokens\":%d,"
        "\"decode_intervals\":%d,\"stop_token_observed\":%s,\"aborted\":%s,"
        "\"build_ms\":%s,\"run_ms\":%s,\"first_observed_ms\":%s,"
        "\"post_first_ms\":%s,\"prompt_positions_per_second\":%s,"
        "\"decode_tokens_per_second\":%s,\"device_first_publish_ms\":%s}",
        t->prompt,t->resumed,t->completed,v.executed,v.prompt_processed,
        v.generated,v.decode_intervals,t->ended?"true":"false",t->aborted?"true":"false",
        flash_timing_number(b,t->build_ms,1),flash_timing_number(r,t->run_ms,1),
        flash_timing_number(f,t->first_ms,t->first_valid),flash_timing_number(p,v.post_ms,v.post_valid),
        flash_timing_number(pr,v.prompt_rate,v.prompt_rate_valid),flash_timing_number(dr,v.decode_rate,v.decode_rate_valid),
        flash_timing_number(d,t->device_first_ms,t->device_first_valid));
    return n<0 || (size_t)n>=cap?-1:0;
}
#endif
