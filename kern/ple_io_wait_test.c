/* CPU-only regression for the real host completion loop; no CUDA context. */
#define _GNU_SOURCE
#define FLASH_PLE_IO_TEST 1
#include "engine.c"
#undef nd
#undef ctx
#include <assert.h>

enum { TEST_DIES = 6, CTL_WORDS = 128 };
static unsigned fake_current, fake_completed, fake_launches, fake_syncs;
static unsigned fake_peer_queries, fake_peer_delay;
static int fake_peer_stuck;
static uint32_t fake_ctl[TEST_DIES][CTL_WORDS], fake_tok, fake_toks[8], fake_abort;
static phase_t fake_program[TEST_DIES];
static core_t fake_core;

CUresult CUDAAPI cuCtxSetCurrent(CUcontext c) {
    uintptr_t tag = (uintptr_t)c;
    assert(tag >= 1 && tag <= TEST_DIES);
    fake_current = (unsigned)tag - 1;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuLaunchKernel(CUfunction f, unsigned gx, unsigned gy, unsigned gz,
        unsigned bx, unsigned by, unsigned bz, unsigned shared, CUstream s,
        void **params, void **extra) {
    (void)f; (void)gx; (void)gy; (void)gz; (void)bx; (void)by; (void)bz;
    (void)shared; (void)s; (void)params; (void)extra;
    fake_launches++;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamQuery(CUstream s) {
    (void)s;
    if (fake_current == TEST_DIES - 1) {
        fake_peer_queries++;
        if (fake_peer_stuck || fake_peer_queries <= fake_peer_delay)
            return CUDA_ERROR_NOT_READY;
    }
    fake_completed |= 1u << fake_current;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuCtxSynchronize(void) {
    /* A blocking synchronization before all queries complete is the regression. */
    assert(fake_completed == 63u);
    fake_syncs++;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t n, CUstream s) {
    (void)s; memcpy(dst, (const void *)(uintptr_t)src, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t n, CUstream s) {
    (void)s; memcpy((void *)(uintptr_t)dst, src, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyDtoH(void *dst, CUdeviceptr src, size_t n) {
    memcpy(dst, (const void *)(uintptr_t)src, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemcpyHtoD(CUdeviceptr dst, const void *src, size_t n) {
    memcpy((void *)(uintptr_t)dst, src, n); return CUDA_SUCCESS;
}
CUresult CUDAAPI cuMemsetD32(CUdeviceptr dst, unsigned value, size_t n) {
    unsigned *p = (unsigned *)(uintptr_t)dst;
    for (size_t i = 0; i < n; i++) p[i] = value;
    return CUDA_SUCCESS;
}
CUresult CUDAAPI cuStreamSynchronize(CUstream s) { (void)s; return CUDA_SUCCESS; }
CUresult CUDAAPI cuGetErrorString(CUresult r, const char **s) {
    (void)r; *s = "mock CUDA error"; return CUDA_SUCCESS;
}
static void prepare(int failed, unsigned delay, int stuck) {
    memset(&fake_core, 0, sizeof fake_core);
    memset(fake_ctl, 0, sizeof fake_ctl);
    memset(db, 0, sizeof db);
    core = &fake_core; core->nd = TEST_DIES;
    fake_current = fake_completed = fake_launches = fake_syncs = 0;
    fake_peer_queries = 0; fake_peer_delay = delay; fake_peer_stuck = stuck;
    fake_tok = fake_abort = 0;
    for (unsigned d = 0; d < TEST_DIES; d++) {
        core->ctx[d] = (CUcontext)(uintptr_t)(d + 1);
        dCTL[d] = (CUdeviceptr)(uintptr_t)fake_ctl[d];
        dPROG[d] = (CUdeviceptr)(uintptr_t)&fake_program[d];
        db[d].toks = (CUdeviceptr)(uintptr_t)fake_toks;
        /* The token stream ended, but the final peer can still be running. */
        fake_ctl[d][2] = UINT32_C(0x80000000);
        fake_ctl[d][0x1c / 4] = failed ? 1 : 0;
    }
    h_tok = &fake_tok; h_toks = fake_toks; h_abort = &fake_abort;
    ple_disk = 0; ple_failed = failed; ple_failed_at = failed ? now() : 0;
    ple_io_launched_mask = ple_io_sync_mask = ple_io_error_mask = 0;
}
static void completed_case(int failed, unsigned delay) {
    unsigned nph[TEST_DIES] = {0};
    runctx r = {.T = 1};
    prepare(failed, delay, 0);
    alarm(3);
    if (setjmp(run_jb)) { fputs("unexpected watchdog\n", stderr); abort(); }
    double started = now();
    int rc = run_segment(&r, nph, 1);
    double elapsed = now() - started;
    alarm(0);
    assert(rc == (failed ? -5 : 0));
    assert(fake_launches == TEST_DIES && fake_syncs == TEST_DIES);
    assert(fake_completed == 63u && ple_io_sync_mask == 63u);
    assert(ple_io_launched_mask == 63u && ple_io_error_mask == (failed ? 63u : 0u));
    assert(fake_peer_queries == delay + 1);
    printf("PASS simulated CUDA %s: all six streams completed before sync; %.3f ms\n",
           failed ? "delayed failure" : "healthy", elapsed * 1e3);
}
static void terminal_error_case(void) {
    runctx r = {.T = 8, .count = 5};
    timing_request_start = now() - .050;
    timing_run_start = now() - .010;
    last_timing = (flash_timing){
        .request_id = 9, .prompt = 8, .completed = 4,
        .counts_known = 1, .counts_final = 1, .error_code = 0,
        .run_ms = 4.0
    };
    last_timing_valid = 1;
    snap_s = 13;
    trace_status = "running";
    trace_completed = trace_aborted = trace_ended = 0;
    ple_io_failure_state(&r);
    assert(snap_s == 0);
    assert(!strcmp(trace_status, "ple_io_error"));
    assert(trace_completed == 5 && !trace_aborted && !trace_ended);
    assert(last_timing_valid && last_timing.error_code == -5);
    assert(last_timing.completed == 5 && last_timing.counts_known);
    assert(!last_timing.counts_final && !last_timing.stop_known);
    assert(isfinite(last_timing.run_ms) && last_timing.run_ms >= 0);
    flash_timing_values values = flash_timing_derive(&last_timing);
    assert(!values.prompt_rate_valid && !values.decode_rate_valid && !values.device_rate_valid);
    puts("PASS PLE I/O terminal metadata: error code, partial counts and failed trace status");
}
static void stuck_case(void) {
    unsigned nph[TEST_DIES] = {0};
    runctx r = {.T = 1};
    prepare(1, 0, 1);
    alarm(3);
    double started = now();
    if (!setjmp(run_jb)) {
        (void)run_segment(&r, nph, 1);
        fputs("stuck peer returned without watchdog\n", stderr); abort();
    }
    double elapsed = now() - started;
    alarm(0);
    assert(fake_launches == TEST_DIES && fake_completed == 31u);
    assert(fake_peer_queries > 0 && fake_syncs == 0);
    assert(elapsed >= .03 && elapsed < 2);
    printf("PASS simulated CUDA stuck peer: watchdog %.3f ms; no blocking sync\n",
           elapsed * 1e3);
}
static void chunk_failure_case(void) {
    prepare(1, 0, 0);
    phase_t descriptor = {0};
    prog_t prefix[TEST_DIES] = {{0}};
    for (int d = 0; d < TEST_DIES; d++) {
        prefix[d].p = &descriptor; prefix[d].n = 1;
        /* A staged-ahead read failed, but this short chunk ends before that row. */
        fake_ctl[d][2] = 4; fake_ctl[d][0x1c / 4] = 0;
    }
    prefill_chunk = 4;
    runctx r = {.T = 16, .nprompt = 16, .sent = 15};
    alarm(3);
    if (setjmp(run_jb)) { fputs("chunk I/O error became a watchdog fault\n", stderr); abort(); }
    int rc = run_prefill_prefix(&r, prefix, prefix, NULL, 8);
    alarm(0);
    assert(rc == -5 && r.count == 4 && ple_io_segment_end == 4);
    assert(fake_launches == TEST_DIES && fake_syncs == TEST_DIES);
    assert(ple_io_sync_mask == 63u && ple_io_error_mask == 0u);
    puts("PASS simulated CUDA prefill boundary: I/O error returned before dispatching another chunk");
}
int main(void) {
    assert(!setenv("FLASH_WATCHDOG_S", "0.03", 1));
    completed_case(0, 0);
    completed_case(1, 8);
    terminal_error_case();
    stuck_case();
    chunk_failure_case();
    puts("ple_io_wait_test: CPU simulation only; GPU termination remains unmeasured");
    return 0;
}
