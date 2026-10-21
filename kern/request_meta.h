#ifndef FLASH_REQUEST_META_H
#define FLASH_REQUEST_META_H
#include <stdint.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

/* Host CLOCK_MONOTONIC observations. Zero means that a boundary was not
 * observed; these values never claim a device publication timestamp. */
typedef struct {
    uint64_t request, aborted, ended, completed, requested, prompt, resumed, snapshot;
    uint64_t ple_short_reads, request_start_ns, request_end_ns;
    uint64_t build_start_ns, build_end_ns, run_start_ns;
    uint64_t engine_first_observed_ns, engine_last_observed_ns, engine_prefix_observed_ns;
    const char *status;
} request_meta;

static inline void request_meta_begin(request_meta *m, uint64_t ns) {
    uint64_t request = m->request + 1;
    memset(m, 0, sizeof *m);
    m->request = request;
    m->status = "running";
    m->request_start_ns = ns;
}
static inline void request_meta_observe(request_meta *m, uint64_t ns) {
    if (!m->engine_first_observed_ns) m->engine_first_observed_ns = ns;
    m->engine_last_observed_ns = ns;
}
static inline void request_meta_finish(request_meta *m, const char *status, uint64_t ns) {
    m->status = status;
    m->request_end_ns = ns;
}
static inline void request_meta_write(FILE *out, const request_meta *m) {
    fprintf(out, "requestmeta schema=1 clock=CLOCK_MONOTONIC request=%" PRIu64
        " status=%s aborted=%" PRIu64 " ended=%" PRIu64
        " completed=%" PRIu64 " requested=%" PRIu64 " prompt=%" PRIu64
        " resumed=%" PRIu64 " snapshot=%" PRIu64 " ple_short_reads=%" PRIu64
        " request_start_ns=%" PRIu64 " request_end_ns=%" PRIu64
        " build_start_ns=%" PRIu64 " build_end_ns=%" PRIu64 " run_start_ns=%" PRIu64
        " engine_first_observed_ns=%" PRIu64 " engine_last_observed_ns=%" PRIu64
        " engine_prefix_observed_ns=%" PRIu64 "\n",
        m->request, m->status ? m->status : "none", m->aborted, m->ended,
        m->completed, m->requested, m->prompt, m->resumed, m->snapshot,
        m->ple_short_reads, m->request_start_ns, m->request_end_ns,
        m->build_start_ns, m->build_end_ns, m->run_start_ns,
        m->engine_first_observed_ns, m->engine_last_observed_ns, m->engine_prefix_observed_ns);
}
#endif
