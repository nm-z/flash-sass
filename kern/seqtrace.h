/* Optional ordered sequence observations. No CUDA calls or client I/O. */
#ifndef FLASH_SEQTRACE_H
#define FLASH_SEQTRACE_H
#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    unsigned long long request_id, last_turn_seq;
    int wanted, slot, nprompt, total;
    int from, count, limit, aborted, ended, resumed, snapshot, rc;
} seqtrace_request;
static FILE *seqtrace_file;
static pthread_mutex_t seqtrace_mutex = PTHREAD_MUTEX_INITIALIZER;
static unsigned long long seqtrace_event_id, seqtrace_request_id;

/* A dedicated validation core uses a fresh path. Append permits retaining
 * observations across reloads; event/request IDs are scoped to one engine load. */
static int seqtrace_open(const char *path) {
    if (!path || !*path) return 0;
    seqtrace_file = fopen(path, "a");
    if (!seqtrace_file) { perror("engine: FLASH_SEQUENCE_TRACE"); return -1; }
    seqtrace_event_id = seqtrace_request_id = 0;
    return 0;
}
static void seqtrace_close(void) {
    if (seqtrace_file) { fclose(seqtrace_file); seqtrace_file = NULL; }
}
static void seqtrace_emit(volatile seqtrace_request *r, const char *event) {
    if (!seqtrace_file) return;
    pthread_mutex_lock(&seqtrace_mutex);
    if (!r->request_id) r->request_id = ++seqtrace_request_id;
    unsigned long long event_id = ++seqtrace_event_id;
    if (!strcmp(event, "turn")) r->last_turn_seq = event_id;
    int rc = fprintf(seqtrace_file,
        "{\"event\":\"%s\",\"event_seq\":%llu,\"request_id\":%llu,\"pid\":%ld,"
        "\"wanted\":%d,\"slot\":%d,\"nprompt\":%d,\"total\":%d,"
        "\"from\":%d,\"count\":%d,\"limit\":%d,\"aborted\":%d,\"ended\":%d,"
        "\"resumed\":%d,\"snapshot\":%d,\"rc\":%d,\"last_turn_seq\":%llu}\n",
        event, event_id, r->request_id, (long)getpid(), r->wanted, r->slot,
        r->nprompt, r->total, r->from, r->count, r->limit, r->aborted,
        r->ended, r->resumed, r->snapshot, r->rc, r->last_turn_seq);
    if (rc < 0 || fflush(seqtrace_file)) {
        /* Missing/incomplete observations make the acceptance checker fail. */
        perror("engine: sequence observation write");
        clearerr(seqtrace_file);
    }
    pthread_mutex_unlock(&seqtrace_mutex);
}
#endif
