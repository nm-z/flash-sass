#define _GNU_SOURCE
#include <cuda.h>
#include <stdlib.h>
static CUresult test_context_entry(CUcontext context);
static CUresult unexpected_cuda_call(void);
static void *test_program_calloc(size_t count, size_t size);
#define calloc(count,size) test_program_calloc(count,size)
#define cuCtxSetCurrent(context) test_context_entry(context)
#define cuInit(...) unexpected_cuda_call()
#define cuLaunchKernel(fn,gx,gy,gz,bx,by,bz,shared,stream,params,extra) ((void)(params), unexpected_cuda_call())
#undef cuMemcpyHtoDAsync
#define cuMemcpyHtoDAsync(...) unexpected_cuda_call()
#define cuStreamSynchronize(...) unexpected_cuda_call()
#include "engine.c"
#undef calloc
#undef cuCtxSetCurrent
#undef cuInit
#undef cuLaunchKernel
#undef cuMemcpyHtoDAsync
#undef cuStreamSynchronize
#undef nd
#include <assert.h>
#include <sys/socket.h>

static unsigned calloc_fail_at, calloc_calls;
static void *test_program_calloc(size_t count, size_t size) {
    if (++calloc_calls == calloc_fail_at) return NULL;
    return calloc(count, size);
}

static int abort_driver_entries;
static CUresult unexpected_cuda_call(void) {
    fputs("trace_session_test: unexpected CUDA operation\n", stderr);
    abort();
}
static CUresult test_context_entry(CUcontext context) {
    (void)context;
    /* A callback-return timestamp must exist before abort_run enters CUDA. */
    assert(trace_host[0].callback_start != 0);
    assert(trace_host[0].callback_end >= trace_host[0].callback_start);
    abort_driver_entries++;
    return CUDA_SUCCESS;
}
static int request_cancel(void *arg, int position, uint32_t token) {
    assert(position == 4 && token == 123);
    assert(!trace_host[0].callback_end);
    (*(int *)arg)++;
    return 1;
}
static void callback_return_before_cancel(void) {
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
    h_abort = NULL; trace_n = 0;
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
    reject_case("chat nonsense");
    reject_case("gen -1");
    reject_case("chat");
    reject_case("gen 2147483648");
    reject_case("chat 999999999999999999999999999999999999999999");
    session_options();half_close_options();
    callback_return_before_cancel();
    program_allocation_failure();
    headless_arena_layout();
    free(s_ids); free(s_toks); s_ids = s_toks = NULL;
    puts("trace_session_test: request identity, timing options, half-close, callback-before-cancellation, and allocation failures PASS");
    return 0;
}
