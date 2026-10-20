#include "prog.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { \
    if (!(x)) { \
        fprintf(stderr, "%s:%d: %s failed\n", __FILE__, __LINE__, #x); \
        return 1; \
    } \
} while (0)

enum {
    COPY_BYTES = 40960,
    TRACE_POSITIONS = 9,
    GUARD_BYTES = 64,
    SRC0_WORD = 2,
    SRC1_WORD = 3,
    DST0_WORD = 4,
    DST1_WORD = 5,
    POS_WORD = 6
};

#define POS_MASK (1u << POS_WORD)

static uint64_t descriptor_ptr(const phase_t *p, unsigned word) {
    return (uint64_t)p->w[word] | ((uint64_t)p->w[word + 1] << 32);
}

static unsigned resolved_position(const phase_t *p, unsigned token) {
    unsigned pos = p->w[POS_WORD];
    return (p->w[31] & POS_MASK) ? pos + token : pos;
}

static void add_trace_copy(prog_t *p, unsigned token_offset,
                           CUdeviceptr src, CUdeviceptr dst) {
    p->loop = 1;
    p->token_offset = token_offset;
    phase_t *x = prog_add(p, OP_XCOPY, 10);
    ph_ptr(x, 0x140, src);
    ph_ptr(x, 0x148, dst);
    ph_tokv(p, x, 0x150, 0, 0);
}

static char *read_source(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return NULL; }
    long size = ftell(f);
    if (size < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    char *text = malloc((size_t)size + 1);
    if (!text) { fclose(f); return NULL; }
    size_t got = fread(text, 1, (size_t)size, f);
    int bad = ferror(f) || got != (size_t)size;
    fclose(f);
    if (bad) { free(text); return NULL; }
    text[got] = 0;
    return text;
}

static int source_contract(const char *path) {
    char *s = read_source(path);
    if (!s) { fprintf(stderr, "cannot read assembly source: %s\n", path); return 1; }
    const char *required[] = {
        "param_POS  : c[0x0][0x150]",
        "ISCADD off, cta, tid, 0x8;",
        "LDG.E.CV.128 x0, [sp];",
        "SHL pbytes0, param_POS, 0xd;",
        "SHL pbytes1, param_POS, 0xf;",
        "IADD doff, pbytes0, pbytes1;",
        "IADD doff, off, doff;",
        "IADD dp0.CC, doff, param_DST0;",
        "IADD.X dp1, RZ, param_DST1;",
        "STG.E.128 [dp], x0;"
    };
    int bad = 0;
    for (size_t i = 0; i < sizeof required / sizeof required[0]; i++) {
        if (!strstr(s, required[i])) {
            fprintf(stderr, "assembly source missing: %s\n", required[i]);
            bad++;
        }
    }
    if (strstr(s, "<CODE>") || strstr(s, "my $") || strstr(s, "$o .=")) {
        fprintf(stderr, "assembly source must remain plain SASS, without a generator block\n");
        bad++;
    }
    free(s);
    return bad;
}

static int guard_layout(void) {
    const size_t span = (size_t)TRACE_POSITIONS * COPY_BYTES;
    const size_t total = GUARD_BYTES + span + GUARD_BYTES;
    uint8_t *dst = malloc(total);
    uint8_t *src[3] = { malloc(COPY_BYTES), malloc(COPY_BYTES), malloc(COPY_BYTES) };
    if (!dst || !src[0] || !src[1] || !src[2]) {
        free(dst); free(src[0]); free(src[1]); free(src[2]);
        fprintf(stderr, "guard fixture allocation failed\n");
        return 1;
    }
    memset(dst, 0xa5, total);
    for (size_t i = 0; i < (size_t)COPY_BYTES; i++) {
        src[0][i] = (uint8_t)(i * 13u + 7u);
        src[1][i] = (uint8_t)(i * 29u + 3u);
        src[2][i] = (uint8_t)(i * 37u + 11u);
    }
    const unsigned start = 7;
    const unsigned positions[3] = { 0, start, start + 1 };
    for (unsigned i = 0; i < 3u; i++) {
        size_t off = (size_t)positions[i] * COPY_BYTES;
        if (off > span || (size_t)COPY_BYTES > span - off) {
            free(dst); free(src[0]); free(src[1]); free(src[2]);
            fprintf(stderr, "trace position exceeds guarded allocation\n");
            return 1;
        }
        memcpy(dst + GUARD_BYTES + off, src[i], COPY_BYTES);
    }
    int bad = 0;
    for (size_t i = 0; i < (size_t)GUARD_BYTES; i++) {
        if (dst[i] != 0xa5 || dst[GUARD_BYTES + span + i] != 0xa5) bad++;
    }
    for (unsigned p = 0; p < (unsigned)TRACE_POSITIONS; p++) {
        const uint8_t *slot = dst + GUARD_BYTES + (size_t)p * COPY_BYTES;
        if (p == 0 || p == start || p == start + 1) {
            const uint8_t *expected = p == 0 ? src[0] : (p == start ? src[1] : src[2]);
            if (memcmp(slot, expected, COPY_BYTES)) bad++;
        } else {
            for (size_t i = 0; i < (size_t)COPY_BYTES; i++) if (slot[i] != 0xa5) { bad++; break; }
        }
    }
    free(dst); free(src[0]); free(src[1]); free(src[2]);
    if (bad) fprintf(stderr, "positioned copy overwrote a trace slot or guard (%d checks)\n", bad);
    return bad;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s validation/issue109/xcopy_trace.sass\n", argv[0]);
        return 2;
    }
    CHECK(source_contract(argv[1]) == 0);

    const CUdeviceptr src = (CUdeviceptr)UINT64_C(0x1122334455667788);
    const CUdeviceptr dst = (CUdeviceptr)UINT64_C(0x8877665544332211);
    prog_t legacy, a, b, pair;
    prog_init(&legacy); prog_init(&a); prog_init(&b); prog_init(&pair);

    phase_t *old = prog_add(&legacy, OP_XCOPY, 10);
    ph_ptr(old, 0x140, src);
    ph_ptr(old, 0x148, dst);
    CHECK(old->w[1] == 10);
    CHECK(old->w[SRC0_WORD] == 0x55667788u && old->w[SRC1_WORD] == 0x11223344u);
    CHECK(descriptor_ptr(old, DST0_WORD) == (uint64_t)dst);
    CHECK(old->w[POS_WORD] == 0 && old->w[31] == 0);
    for (unsigned t = 0; t < (unsigned)TRACE_POSITIONS; t++) CHECK(resolved_position(old, t) == 0);

    add_trace_copy(&a, 0, src, dst);
    add_trace_copy(&b, 1, src + COPY_BYTES, dst);
    CHECK(a.n == 1 && b.n == 1);
    CHECK(a.p[0].w[1] == 10 && b.p[0].w[1] == 10);
    CHECK(a.p[0].w[POS_WORD] == 0 && b.p[0].w[POS_WORD] == 1);
    CHECK(a.p[0].w[31] == POS_MASK && b.p[0].w[31] == POS_MASK);
    CHECK(prog_pair(&pair, &a, &b) == 0);
    CHECK(pair.n == 2 && pair.p[0].w[0] == OP_XCOPY && pair.p[1].w[0] == OP_XCOPY);
    CHECK((pair.p[0].w[30] & 8u) != 0 && (pair.p[1].w[30] & 8u) == 0);
    CHECK(pair.p[0].w[31] == POS_MASK && pair.p[1].w[31] == POS_MASK);
    for (unsigned t = 0; t + 1 < (unsigned)TRACE_POSITIONS; t++) {
        CHECK(resolved_position(&pair.p[0], t) == t);
        CHECK(resolved_position(&pair.p[1], t) == t + 1);
    }
    CHECK(guard_layout() == 0);

    prog_free(&legacy); prog_free(&a); prog_free(&b); prog_free(&pair);
    puts("xcopy_trace_test: legacy default, token mask, paired positions, and guarded slices PASS");
    return 0;
}
