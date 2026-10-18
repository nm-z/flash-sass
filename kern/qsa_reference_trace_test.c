#define _POSIX_C_SOURCE 200809L
#include "qsa_reference_trace.h"
#include <assert.h>
#include <limits.h>

static unsigned checks;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); exit(1); } } while (0)
typedef struct { const unsigned char *data; size_t size, calls; int fail; } memory;
static int read_memory(void *opaque, void *dst, size_t at, size_t n) {
    memory *m = opaque;
    ++m->calls;
    if (m->fail || at > m->size || n > m->size - at) return -1;
    memcpy(dst, m->data + at, n); return 0;
}
static int exists(const qrt_state *s, const char *leaf) {
    char p[1800]; CHECK(!qrt_path(p, sizeof(p), s, leaf));
    return access(p, F_OK) == 0;
}
static void release(qrt_state *s) {
    if (s->catalog) { CHECK(!qrt_close(s->catalog)); s->catalog = NULL; }
}
int main(int argc, char **argv) {
    if (argc != 2) { fputs("usage: qsa_reference_trace_test NEW_DIR\n", stderr); return 2; }
    CHECK(!mkdir(argv[1], 0700));
    unsigned layer, category;
    CHECK(qrt_interest("indexer_score-3", &layer, &category) && layer == 3 && category == 8);
    CHECK(qrt_interest("indexer_top_k-47", &layer, &category) && category == 16);
    CHECK(qrt_interest("kq_mask_qsa-7", &layer, &category) && category == 32);
    CHECK(!qrt_interest("indexer_score-48", &layer, &category));
    CHECK(!qrt_interest("indexer_score-0", &layer, &category));
    CHECK(!qrt_interest("indexer_score-3x", &layer, &category));
    CHECK(!qrt_interest("indexer_score-", &layer, &category));
    CHECK(!qrt_interest("indexer_score-999999999999999999999999", &layer, &category));
    CHECK(!qrt_interest("unrelated-3", &layer, &category));

    const float values[] = {1,2,999,3,4,999};
    memory m = {(const unsigned char *)values, sizeof(values), 0, 0};
    qrt_tensor t = {.name="indexer_q-3", .dtype="f32", .element_size=4,
                    .ne={2,2,1,1}, .nb={4,12,24,24}, .span=sizeof(values)};
    qrt_state state;
    CHECK(!qrt_begin(&state, argv[1], 0));
    CHECK(!qrt_record(&state, &t, read_memory, &m));
    CHECK(m.calls == 2 && state.count == 1 && state.seen[3] == 2);
    char path[1800]; CHECK(!qrt_path(path, sizeof(path), &state, "tensor-000000.bin"));
    FILE *f = fopen(path, "rb"); CHECK(f != NULL);
    float packed[4]; CHECK(fread(packed, sizeof(float), 4, f) == 4);
    CHECK(fgetc(f) == EOF); CHECK(!fclose(f));
    CHECK(packed[0] == 1 && packed[1] == 2 && packed[2] == 3 && packed[3] == 4);
    CHECK(qrt_finish(&state) == -1);
    CHECK(!exists(&state, "complete.json"));

    /* Noncontiguous innermost elements must also exclude poisoned gaps. */
    CHECK(!qrt_begin(&state, argv[1], 1));
    t.ne[0]=2; t.ne[1]=1; t.nb[0]=12;
    m.calls=0;
    CHECK(!qrt_record(&state, &t, read_memory, &m) && m.calls == 2);
    CHECK(!qrt_path(path, sizeof(path), &state, "tensor-000000.bin"));
    f=fopen(path,"rb"); CHECK(f != NULL);
    CHECK(fread(packed, sizeof(float), 2, f) == 2 && packed[0] == 1 && packed[1] == 3);
    CHECK(fgetc(f) == EOF); CHECK(!fclose(f));
    release(&state);

    CHECK(!qrt_begin(&state, argv[1], 2));
    qrt_tensor bad=t;
    bad.span=4; CHECK(qrt_record(&state, &bad, read_memory, &m) == -1);
    bad=t; bad.ne[0]=SIZE_MAX; CHECK(qrt_record(&state,&bad,read_memory,&m) == -1);
    bad=t; bad.nb[0]=SIZE_MAX; CHECK(qrt_record(&state,&bad,read_memory,&m) == -1);
    bad=t; bad.ne[0]=QRT_MAX_BYTES; CHECK(qrt_record(&state,&bad,read_memory,&m) == -1);
    bad=t; bad.dtype="q2_0"; CHECK(qrt_record(&state,&bad,read_memory,&m) == -1);
    bad=t; bad.dtype="f16"; CHECK(qrt_record(&state,&bad,read_memory,&m) == -1);
    CHECK(state.count == 0 && state.seen[3] == 0);
    bad=t; bad.ne[0]=0; bad.span=0; m.calls=0;
    CHECK(!qrt_record(&state,&bad,read_memory,&m) && m.calls == 0);
    CHECK(!qrt_path(path,sizeof(path),&state,"tensor-000000.bin"));
    struct stat st; CHECK(!stat(path,&st) && st.st_size == 0);
    release(&state);

    CHECK(!qrt_begin(&state, argv[1], 3));
    m.fail=1; CHECK(qrt_record(&state,&t,read_memory,&m) == -1);
    CHECK(state.count == 0 && state.seen[3] == 0);
    CHECK(qrt_finish(&state) == -1 && !exists(&state,"complete.json"));
    m.fail=0;

    CHECK(!qrt_begin(&state, argv[1], 4));
    static const char *roles[] = {"indexer_k_raw", "indexer_q", "indexer_k", "indexer_score",
                                 "indexer_top_k", "kq_mask_qsa", "attn_gated", "l_last"};
    char name[96];
    t.ne[0]=2; t.ne[1]=1; t.nb[0]=4;
    for (unsigned il=3; il<48; il+=4) for (unsigned role=0; role<8; ++role) {
        snprintf(name,sizeof(name),"%s-%u",roles[role],il); t.name=name;
        CHECK(!qrt_record(&state,&t,read_memory,&m));
    }
    CHECK(state.count == 96);
    CHECK(!qrt_finish(&state));
    CHECK(exists(&state,"complete.json"));
    CHECK(!exists(&state,"complete.json.tmp"));
    CHECK(qrt_finish(&state) == -1);
    CHECK(qrt_begin(&state,argv[1],4) == -1);
    CHECK(exists(&state,"complete.json"));

    printf("PASS: %u synthetic trace checks; no checkpoint or GPU execution\n",checks);
    return 0;
}
