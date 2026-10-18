/* Native logical-element snapshots for an independent backend callback.
 * The reader is supplied by the backend; tests use a bounded memory reader.
 * Padding bytes are never copied. Completion requires all twelve QSA layers.
 */
#ifndef FLASH_QSA_REFERENCE_TRACE_H
#define FLASH_QSA_REFERENCE_TRACE_H
#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define QRT_MAX_BYTES (64u * 1024u * 1024u)
#define QRT_REQUIRED 255u

typedef struct {
    const char *name, *dtype;
    size_t element_size, ne[4], nb[4], span;
} qrt_tensor;
typedef int (*qrt_read)(void *, void *, size_t, size_t);
typedef struct {
    char directory[1600];
    FILE *catalog;
    size_t row;
    unsigned count, seen[48];
} qrt_state;

static int qrt_close(FILE *f) {
    int bad = ferror(f);
    if (fflush(f)) bad = 1;
    if (fclose(f)) bad = 1;
    return bad ? -1 : 0;
}
static void qrt_quote(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 32) fprintf(f, "\\u%04x", (unsigned)c);
        else fputc(c, f);
    }
    fputc('"', f);
}
static int qrt_path(char *out, size_t cap, const qrt_state *s, const char *leaf) {
    int n = snprintf(out, cap, "%s/%s", s->directory, leaf);
    return n < 0 || (size_t)n >= cap ? -1 : 0;
}
/* A backend may expose aliases for the mask; require its logical role once.
 * Lowercase names and the layer suffix come from the pinned graph, not from
 * candidate code. Unrecognized versions cannot silently satisfy the gate. */
static int qrt_interest(const char *name, unsigned *layer, unsigned *category) {
    const char *dash = strrchr(name, '-');
    if (!dash || !dash[1]) return 0;
    unsigned n = 0;
    for (const char *p = dash + 1; *p; ++p) {
        if (*p < '0' || *p > '9' || n > 47) return 0;
        n = n * 10 + (unsigned)(*p - '0');
    }
    if (n >= 48 || n % 4 != 3) return 0;
    size_t len = (size_t)(dash - name);
    *layer = n; *category = 0;
#define QRT_NAME(text) (len == sizeof(text) - 1 && !memcmp(name, text, len))
    if (QRT_NAME("indexer_k_raw")) *category = 1;
    else if (QRT_NAME("indexer_q")) *category = 2;
    else if (QRT_NAME("indexer_k")) *category = 4;
    else if (QRT_NAME("indexer_score")) *category = 8;
    else if (QRT_NAME("indexer_top_k")) *category = 16;
    else if (QRT_NAME("indexer_sel") || QRT_NAME("kq_mask_qsa")) *category = 32;
    else if (QRT_NAME("attn_gated")) *category = 64;
    else if (QRT_NAME("l_last")) *category = 128;
#undef QRT_NAME
    return *category || !strncmp(name, "indexer_", 8) ||
           !strncmp(name, "Qcur", 4) || !strncmp(name, "Kcur", 4) ||
           !strncmp(name, "Vcur", 4) || !strncmp(name, "attn_", 5);
}
static int qrt_begin(qrt_state *s, const char *root, size_t row) {
    memset(s, 0, sizeof(*s)); s->row = row;
    int n = snprintf(s->directory, sizeof(s->directory), "%s/trace-row%06zu", root, row);
    if (n < 0 || (size_t)n >= sizeof(s->directory) || mkdir(s->directory, 0700)) return -1;
    char path[1800];
    if (qrt_path(path, sizeof(path), s, "tensors.jsonl")) return -1;
    s->catalog = fopen(path, "wbx");
    return s->catalog ? 0 : -1;
}
static int qrt_record(qrt_state *s, const qrt_tensor *t, qrt_read reader, void *opaque) {
    unsigned layer, category;
    if (!s->catalog || !qrt_interest(t->name, &layer, &category)) return -1;
    if (!t->dtype || !((t->element_size == 4 &&
        (!strcmp(t->dtype, "f32") || !strcmp(t->dtype, "i32"))) ||
        (t->element_size == 2 && (!strcmp(t->dtype, "f16") || !strcmp(t->dtype, "bf16")))))
        return -1;
    size_t elements = 1, last = 0;
    int empty = 0;
    for (unsigned d = 0; d < 4; ++d) if (!t->ne[d]) empty = 1;
    if (empty) elements = 0;
    else for (unsigned d = 0; d < 4; ++d) {
        if (t->ne[d] > SIZE_MAX / elements) return -1;
        elements *= t->ne[d];
        size_t steps = t->ne[d] - 1;
        if (steps && t->nb[d] > (SIZE_MAX - last) / steps) return -1;
        last += steps * t->nb[d];
    }
    if (elements > QRT_MAX_BYTES / t->element_size ||
        (elements && (last > t->span || t->element_size > t->span - last))) return -1;
    char leaf[64], path[1800];
    if (s->count == UINT32_MAX) return -1;
    snprintf(leaf, sizeof(leaf), "tensor-%06u.bin", s->count);
    if (qrt_path(path, sizeof(path), s, leaf)) return -1;
    FILE *f = fopen(path, "wbx");
    if (!f) return -1;
    int bad = 0;
    if (elements) {
        size_t row_bytes = t->ne[0] * t->element_size;
        unsigned char *row = malloc(row_bytes);
        if (!row) { fclose(f); return -1; }
        for (size_t i3 = 0; i3 < t->ne[3] && !bad; ++i3)
        for (size_t i2 = 0; i2 < t->ne[2] && !bad; ++i2)
        for (size_t i1 = 0; i1 < t->ne[1] && !bad; ++i1) {
            size_t base = i3*t->nb[3] + i2*t->nb[2] + i1*t->nb[1];
            if (t->nb[0] == t->element_size) bad = reader(opaque, row, base, row_bytes) != 0;
            else for (size_t i0 = 0; i0 < t->ne[0] && !bad; ++i0)
                bad = reader(opaque, row + i0*t->element_size, base + i0*t->nb[0], t->element_size) != 0;
            if (!bad && fwrite(row, 1, row_bytes, f) != row_bytes) bad = 1;
        }
        free(row);
    }
    if (qrt_close(f)) bad = 1;
    if (bad) return -1;
    FILE *m = s->catalog;
    fprintf(m, "{\"row\":%zu,\"visible_length\":%zu,\"layer\":%u,\"name\":", s->row, s->row + 1, layer);
    qrt_quote(m, t->name);
    fprintf(m, ",\"file\":\"%s\",\"dtype\":\"%s\",\"byte_order\":\"little\","
               "\"storage\":\"packed_logical_elements_ne0_fastest\",\"shape\":[%zu,%zu,%zu,%zu],"
               "\"source_strides_bytes\":[%zu,%zu,%zu,%zu],\"bytes\":%zu}\n",
            leaf, t->dtype, t->ne[0],t->ne[1],t->ne[2],t->ne[3],
            t->nb[0],t->nb[1],t->nb[2],t->nb[3],elements*t->element_size);
    if (ferror(m) || fflush(m)) return -1;
    ++s->count; s->seen[layer] |= category;
    return 0;
}
static int qrt_finish(qrt_state *s) {
    if (!s->catalog) return -1;
    FILE *m = s->catalog; s->catalog = NULL;
    if (qrt_close(m)) return -1;
    for (unsigned layer = 3; layer < 48; layer += 4) {
        if ((s->seen[layer] & QRT_REQUIRED) != QRT_REQUIRED) {
            fprintf(stderr, "QSA reference trace row %zu layer %u missing role bits 0x%x\n",
                    s->row, layer, QRT_REQUIRED & ~s->seen[layer]);
            return -1;
        }
    }
    char pending[1800], complete[1800];
    if (qrt_path(pending, sizeof(pending), s, "complete.json.tmp") ||
        qrt_path(complete, sizeof(complete), s, "complete.json")) return -1;
    FILE *f = fopen(pending, "wbx");
    if (!f) return -1;
    fprintf(f, "{\"schema\":\"flash-sass.independent-qsa-row.v1\",\"row\":%zu,"
               "\"visible_length\":%zu,\"qsa_layers\":12,\"tensors\":%u,"
               "\"independent_checkpoint_evidence\":false,\"complete\":true}\n",
            s->row, s->row + 1, s->count);
    if (qrt_close(f) || link(pending, complete) || unlink(pending)) return -1;
    return 0;
}
#endif
