#define _POSIX_C_SOURCE 200809L
#include "request_meta.h"
#include <assert.h>
#include <stdlib.h>
static void contains(const char *s, const char *part) { assert(strstr(s, part)); }
int main(void) {
    request_meta m = {0};
    char *s = NULL; size_t n = 0;
    FILE *out = open_memstream(&s, &n); assert(out);
    request_meta_write(out, &m); fclose(out);
    contains(s, "requestmeta schema=1 clock=CLOCK_MONOTONIC request=0 status=none ");
    contains(s, "request_end_ns=0 "); free(s);
    request_meta_begin(&m, UINT64_C(1234567890123456789));
    assert(m.request == 1 && !strcmp(m.status, "running"));
    m.requested = 23; m.prompt = 20; m.resumed = 10; m.snapshot = 15;
    m.build_start_ns = m.request_start_ns + 1;
    m.build_end_ns = m.request_start_ns + 2;
    m.run_start_ns = m.request_start_ns + 3;
    m.engine_prefix_observed_ns = m.request_start_ns + 4;
    request_meta_observe(&m, m.request_start_ns + 5);
    request_meta_observe(&m, m.request_start_ns + 7);
    assert(m.engine_first_observed_ns == m.request_start_ns + 5);
    assert(m.engine_last_observed_ns == m.request_start_ns + 7);
    m.completed = 23; m.ended = 1;
    request_meta_finish(&m, "completed", m.request_start_ns + 8);
    s = NULL; n = 0; out = open_memstream(&s, &n); assert(out);
    request_meta_write(out, &m); fclose(out);
    contains(s, "status=completed aborted=0 ended=1 completed=23 requested=23 prompt=20 resumed=10 snapshot=15 ple_short_reads=0 ");
    contains(s, "request_start_ns=1234567890123456789 request_end_ns=1234567890123456797 ");
    contains(s, "engine_first_observed_ns=1234567890123456794 engine_last_observed_ns=1234567890123456796 engine_prefix_observed_ns=1234567890123456793\n");
    assert(n && s[n - 1] == '\n'); free(s);
    request_meta_begin(&m, 200);
    assert(m.request == 2 && m.request_start_ns == 200);
    assert(!m.completed && !m.requested && !m.prompt && !m.resumed && !m.snapshot);
    assert(!m.aborted && !m.ended && !m.ple_short_reads && !m.request_end_ns);
    assert(!m.build_start_ns && !m.build_end_ns && !m.run_start_ns);
    assert(!m.engine_first_observed_ns && !m.engine_last_observed_ns && !m.engine_prefix_observed_ns);
    request_meta_finish(&m, "argument_error", 201);
    assert(!strcmp(m.status, "argument_error") && m.request_end_ns == 201);
    request_meta_begin(&m, 300); m.completed = 21; m.aborted = 1;
    request_meta_finish(&m, "aborted", 301);
    assert(m.request == 3 && m.completed == 21 && m.aborted && !strcmp(m.status, "aborted"));
    request_meta_begin(&m, 400); m.ple_short_reads = 2;
    request_meta_finish(&m, "device_error", 401);
    assert(m.request == 4 && m.ple_short_reads == 2 && !m.aborted && !strcmp(m.status, "device_error"));
    puts("request_meta_test: ok");
    return 0;
}
