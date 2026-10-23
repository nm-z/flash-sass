#define _GNU_SOURCE
#include "engine.c"
#undef nd
#include <assert.h>

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
    assert(!last_timing.counts_known && !last_timing.counts_final);
    assert(!last_timing.first_valid && !last_timing.stop_known);
    assert(isnan(last_timing.build_ms) && isnan(last_timing.run_ms));
#endif
    free(output);
    core = NULL;
}
int main(void) {
    reject_case("chat nonsense");
    reject_case("gen -1");
    reject_case("chat");
    reject_case("gen 2147483648");
    reject_case("chat 999999999999999999999999999999999999999999");
    free(s_ids); free(s_toks); s_ids = s_toks = NULL;
    puts("trace_session_test: 5 malformed requests invalidate prior traces PASS");
    return 0;
}
