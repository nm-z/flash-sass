/* Observe an independent llama.cpp forward through its public C API.
 * This is a capture tool, not an assertion that llama.cpp implements the pinned
 * Transformers rounding contract or that a GGUF has verified checkpoint lineage.
 * The optional build requires llama.cpp's b10705-era C headers and libraries.
 */
#define _POSIX_C_SOURCE 200809L
#include <llama.h>
#include <ggml-backend.h>
#include <errno.h>
#include <fcntl.h>
#include <fenv.h>
#include <float.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define LAYERS 48
#define VOCAB 248320
#define MAX_VISIBLE 262144
#define MAX_TENSOR_BYTES (512u * 1024u * 1024u)
#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif

typedef int (*reader)(void *, size_t, void *, size_t);
typedef struct {
    uint64_t ne[4];
    size_t nb[4], element_bytes, span;
    reader read;
    void *arg;
} view;
typedef struct {
    const char *dir;
    FILE *catalog;
    unsigned tensors, scores[LAYERS], selections[LAYERS], outputs[LAYERS];
    int capturing, failed;
} capture;

static void json_string(FILE *f, const char *s) {
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; ++p) {
        if (*p == '"' || *p == '\\') fprintf(f, "\\%c", *p);
        else if (*p < 32 || *p >= 127) fprintf(f, "\\u%04x", *p);
        else fputc(*p, f);
    }
    fputc('"', f);
}
static int path_join(char out[4096], const char *dir, const char *name) {
    int n = snprintf(out, 4096, "%s/%s", dir, name);
    if (n < 0 || n >= 4096) { errno = ENAMETOOLONG; return -1; }
    return 0;
}
static FILE *new_file(const char *dir, const char *name) {
    char path[4096];
    if (path_join(path, dir, name)) return NULL;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0) return NULL;
    FILE *f = fdopen(fd, "wb");
    if (!f) { int e = errno; close(fd); unlink(path); errno = e; }
    return f;
}
static int write_file(const char *dir, const char *name, const void *data, size_t n) {
    FILE *f = new_file(dir, name);
    if (!f) return -1;
    int failed = fwrite(data, 1, n, f) != n;
    if (fclose(f)) failed = 1;
    if (failed) {
        char path[4096]; if (!path_join(path, dir, name)) unlink(path);
        errno = EIO; return -1;
    }
    return 0;
}
static int parse_number(const char *text, unsigned max, unsigned *value) {
    if (!text || !*text) return -1;
    unsigned n = 0;
    for (const unsigned char *p = (const unsigned char *)text; *p; ++p) {
        if (*p < '0' || *p > '9') return -1;
        unsigned d = *p - '0';
        if (n > max / 10 || (n == max / 10 && d > max % 10)) return -1;
        n = n * 10 + d;
    }
    if (!n) return -1;
    *value = n; return 0;
}
static int view_size(const view *v, size_t *bytes) {
    size_t count = 1, end = 0;
    if (!v->read || !v->element_bytes || v->element_bytes > 8) return -1;
    for (int i = 0; i < 4; ++i) {
        if (!v->ne[i] || v->ne[i] > SIZE_MAX / count) return -1;
        count *= (size_t)v->ne[i];
        uint64_t steps = v->ne[i] - 1;
        if (v->nb[i] && steps > (SIZE_MAX - end) / v->nb[i]) return -1;
        end += (size_t)steps * v->nb[i];
    }
    if (count > MAX_TENSOR_BYTES / v->element_bytes ||
        end > SIZE_MAX - v->element_bytes || end + v->element_bytes > v->span) return -1;
    *bytes = count * v->element_bytes;
    return 0;
}
/* Serialize logical elements in ne[0]-fastest order, never the padding between
 * tensor rows. A non-contiguous view must not masquerade as a contiguous array. */
static int dump_view(const char *dir, const char *name, const view *v, size_t *nbytes) {
    if (view_size(v, nbytes)) { errno = EINVAL; return -1; }
    FILE *f = new_file(dir, name);
    if (!f) return -1;
    unsigned char buf[65536];
    int failed = 0;
    for (uint64_t i3 = 0; i3 < v->ne[3] && !failed; ++i3)
    for (uint64_t i2 = 0; i2 < v->ne[2] && !failed; ++i2)
    for (uint64_t i1 = 0; i1 < v->ne[1] && !failed; ++i1) {
        size_t base = (size_t)i3*v->nb[3] + (size_t)i2*v->nb[2] + (size_t)i1*v->nb[1];
        for (uint64_t i0 = 0; i0 < v->ne[0] && !failed;) {
            size_t count = v->nb[0] == v->element_bytes ?
                (size_t)(v->ne[0] - i0) : 1;
            if (count > sizeof buf / v->element_bytes) count = sizeof buf / v->element_bytes;
            size_t n = count * v->element_bytes;
            if (v->read(v->arg, base + (size_t)i0*v->nb[0], buf, n) ||
                fwrite(buf, 1, n, f) != n) failed = 1;
            i0 += count;
        }
    }
    if (fclose(f)) failed = 1;
    if (failed) {
        char path[4096]; if (!path_join(path, dir, name)) unlink(path);
        errno = EIO; return -1;
    }
    return 0;
}
static int tensor_read(void *arg, size_t offset, void *out, size_t n) {
    ggml_backend_tensor_get((const struct ggml_tensor *)arg, out, offset, n);
    return 0;
}
static const char *const names[] = {
    "indexer_k_raw", "indexer_k_pooled", "indexer_k", "indexer_q",
    "indexer_score", "indexer_score_tokens", "indexer_top_k",
    "Qcur_full", "Qcur_normed", "Kcur", "Kcur_normed", "Vcur",
    "kqv_out", "attn_pregate", "attn_gated", "attn_output", "l_last"
};
static int node_kind(const char *name, unsigned *layer) {
    for (size_t i = 0; i < sizeof names / sizeof *names; ++i) {
        size_t n = strlen(names[i]);
        if (strncmp(name, names[i], n) || name[n] != '-') continue;
        const char *p = name + n + 1;
        if (!*p) continue;
        unsigned b = 0;
        for (; *p >= '0' && *p <= '9'; ++p) {
            if (b >= LAYERS) break;
            b = 10*b + (unsigned)(*p - '0');
        }
        if (*p || b >= LAYERS) continue;
        if (strcmp(names[i], "l_last") && b % 4 != 3) continue;
        *layer = b; return (int)i;
    }
    return -1;
}
static bool observe(struct ggml_tensor *t, bool ask, void *arg) {
    capture *c = arg;
    unsigned layer;
    int kind = node_kind(t->name, &layer);
    if (ask) return c->capturing && !c->failed && kind >= 0;
    if (!c->capturing || kind < 0) return true;
    view v = { .span = ggml_nbytes(t), .read = tensor_read, .arg = t };
    switch (t->type) {
        case GGML_TYPE_F32: case GGML_TYPE_I32: v.element_bytes = 4; break;
        case GGML_TYPE_F16: case GGML_TYPE_BF16: v.element_bytes = 2; break;
        default: fprintf(stderr, "unsupported observed tensor type: %s\n", t->name); c->failed = 1; return false;
    }
    for (int i = 0; i < 4; ++i) { v.ne[i] = (uint64_t)t->ne[i]; v.nb[i] = t->nb[i]; }
    char filename[256];
    int len = snprintf(filename, sizeof filename, "%04u-%s.bin", c->tensors, t->name);
    size_t bytes;
    if (len < 0 || (size_t)len >= sizeof filename || dump_view(c->dir, filename, &v, &bytes)) {
        fprintf(stderr, "tensor capture failed for %s: %s\n", t->name, strerror(errno));
        c->failed = 1; return false;
    }
    FILE *f = c->catalog;
    fputs("{\"name\":", f); json_string(f, t->name);
    fputs(",\"file\":", f); json_string(f, filename);
    fputs(",\"type\":", f); json_string(f, ggml_type_name(t->type));
    fprintf(f, ",\"layer\":%u,\"bytes\":%zu,\"shape\":[%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 "],"
            "\"source_strides\":[%zu,%zu,%zu,%zu],\"layout\":\"contiguous_ne0_fastest\"}\n",
            layer, bytes, v.ne[0], v.ne[1], v.ne[2], v.ne[3], v.nb[0], v.nb[1], v.nb[2], v.nb[3]);
    if (fflush(f)) { c->failed = 1; return false; }
    ++c->tensors;
    if (!strcmp(names[kind], "indexer_score")) ++c->scores[layer];
    if (!strcmp(names[kind], "indexer_top_k")) ++c->selections[layer];
    if (!strcmp(names[kind], "attn_gated")) ++c->outputs[layer];
    return true;
}
static uint32_t *read_ids(const char *path, unsigned n) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    struct stat st;
    uint32_t *ids = NULL;
    if (fstat(fileno(f), &st) || !S_ISREG(st.st_mode) || st.st_size < 0 ||
        (uintmax_t)st.st_size % 4 || (uintmax_t)st.st_size < (uintmax_t)n * 4) goto done;
    ids = malloc((size_t)n * 4);
    if (!ids) goto done;
    if (fread(ids, 4, n, f) != n) { free(ids); ids = NULL; goto done; }
    for (unsigned i = 0; i < n; ++i) if (ids[i] >= VOCAB) {
        fprintf(stderr, "input ID outside vocabulary at position %u\n", i);
        free(ids); ids = NULL; break;
    }
    /* Save these exact copied IDs; later changes to the original cannot change
     * the prefix that is supplied to the independent forward. */
done:
    if (fclose(f)) { free(ids); ids = NULL; }
    return ids;
}
static int host_read(void *arg, size_t off, void *out, size_t n) {
    memcpy(out, (const unsigned char *)arg + off, n); return 0;
}
static int failing_read(void *arg, size_t off, void *out, size_t n) {
    (void)arg; (void)off; (void)out; (void)n; return -1;
}
static int self_test(const char *dir) {
    if (mkdir(dir, 0700)) return 2;
    uint32_t data[] = {0,1,2,3,4,5,6,7}, expected[] = {0,2,1,3};
    view v = { .ne = {2,2,1,1}, .nb = {8,4,16,16}, .element_bytes = 4,
               .span = sizeof data, .read = host_read, .arg = data };
    size_t bytes;
    if (dump_view(dir, "strided.u32", &v, &bytes) || bytes != sizeof expected) return 3;
    char path[4096]; path_join(path, dir, "strided.u32");
    FILE *f = fopen(path, "rb"); uint32_t got[4];
    if (!f) return 3;
    int bad = fread(got, 1, sizeof got, f) != sizeof got || memcmp(got, expected, sizeof got);
    if (fclose(f)) bad = 1;
    if (bad || !dump_view(dir, "strided.u32", &v, &bytes)) return 3;
    v.read = failing_read;
    if (!dump_view(dir, "read-failed.u32", &v, &bytes)) return 3;
    path_join(path, dir, "read-failed.u32"); if (!access(path, F_OK)) return 3;
    v.read = host_read; v.ne[0] = UINT64_MAX;
    if (!dump_view(dir, "overflow.u32", &v, &bytes)) return 3;
    v.ne[0] = 2; v.span = 3;
    if (!dump_view(dir, "bad-span.u32", &v, &bytes)) return 3;
    unsigned n;
    if (!parse_number("0", 64, &n) || !parse_number("-1", 64, &n) ||
        !parse_number("65", 64, &n) || !parse_number("2x", 64, &n) ||
        parse_number("64", 64, &n) || n != 64) return 3;
    if (node_kind("indexer_top_k-3", &n) < 0 || n != 3 ||
        node_kind("indexer_top_k-47", &n) < 0 || n != 47 ||
        node_kind("indexer_top_k-48", &n) >= 0 ||
        node_kind("indexer_top_k-3/../../bad", &n) >= 0 ||
        node_kind("indexer_top_k-1", &n) >= 0) return 3;
    puts("PASS_SYNTHETIC_IO: strided layout, exact sizes, exclusive files, failed-read cleanup, overflow, bounds, CLI and node names");
    return 0;
}
int main(int argc, char **argv) {
    uint32_t one = 1;
    if (*(unsigned char *)&one != 1 || sizeof(float) != 4 || FLT_RADIX != 2 || FLT_MANT_DIG != 24 ||
        FLT_EVAL_METHOD != 0 || fegetround() != FE_TONEAREST) {
        fputs("little-endian IEEE binary32, FLT_EVAL_METHOD=0 and FE_TONEAREST required\n", stderr); return 2;
    }
    if (argc == 3 && !strcmp(argv[1], "--self-test")) return self_test(argv[2]);
    unsigned visible, threads = 4;
    if ((argc != 5 && argc != 6) || parse_number(argv[3], MAX_VISIBLE, &visible) ||
        (argc == 6 && parse_number(argv[5], 64, &threads))) {
        fprintf(stderr, "usage: %s MODEL.gguf IDS.u32 VISIBLE NEW_DIR [THREADS=4]\n       %s --self-test NEW_DIR\n", argv[0], argv[0]);
        return 2;
    }
    uint32_t *ids = read_ids(argv[2], visible);
    if (!ids) { fputs("cannot read a complete, in-range token prefix\n", stderr); return 2; }
    const char *dir = argv[4];
    if (mkdir(dir, 0700)) { perror(dir); free(ids); return 2; }
    if (write_file(dir, "prefix_ids.u32", ids, (size_t)visible*4) ||
        write_file(dir, "row_ids.u32", ids + visible - 1, 4)) { perror("save IDs"); free(ids); return 2; }
    capture c = { .dir = dir, .catalog = new_file(dir, "tensors.jsonl") };
    if (!c.catalog) { free(ids); return 2; }
    struct llama_model *model = NULL;
    struct llama_context *ctx = NULL;
    struct llama_batch batch = {0};
    int batch_live = 0, rc = 2;
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    ggml_backend_dev_t no_accelerators[] = {NULL};
    mp.devices = no_accelerators; mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP; mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    mp.use_extra_bufts = false; mp.load_mtp = false;
    model = llama_model_load_from_file(argv[1], mp);
    if (!model) { fputs("independent model load failed\n", stderr); goto done; }
    char arch[128];
    int arch_length = llama_model_meta_val_str(model, "general.architecture", arch, sizeof arch);
    if (arch_length < 0 || (size_t)arch_length >= sizeof arch ||
        strcmp(arch, "qwen4exp") || llama_model_n_layer(model) != LAYERS ||
        llama_vocab_n_tokens(llama_model_get_vocab(model)) != VOCAB) {
        fputs("requires the 48-layer, 248320-token qwen4exp checkpoint\n", stderr); goto done;
    }
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = visible < 256 ? 256 : visible; cp.n_batch = 64; cp.n_ubatch = 64;
    cp.n_seq_max = 1; cp.n_threads = cp.n_threads_batch = (int32_t)threads;
    cp.type_k = cp.type_v = GGML_TYPE_F32;
    cp.offload_kqv = false; cp.op_offload = false;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval = observe; cp.cb_eval_user_data = &c;
    ctx = llama_init_from_model(model, cp);
    if (!ctx) { fputs("independent context creation failed\n", stderr); goto done; }
    batch = llama_batch_init(64, 0, 1); batch_live = 1;
    if (!batch.token || !batch.pos || !batch.n_seq_id || !batch.seq_id || !batch.logits) goto done;
    for (unsigned at = 0; at < visible;) {
        unsigned n = at == visible - 1 ? 1 : visible - 1 - at;
        if (n > 64) n = 64;
        batch.n_tokens = (int32_t)n;
        c.capturing = at == visible - 1;
        for (unsigned j = 0; j < n; ++j) {
            batch.token[j] = (llama_token)ids[at+j]; batch.pos[j] = (llama_pos)(at+j);
            batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0;
            batch.logits[j] = c.capturing ? 1 : 0;
        }
        int status = llama_decode(ctx, batch);
        if (status || c.failed) {
            fprintf(stderr, "independent decode/capture failed at %u: decode=%d capture=%d\n", at, status, c.failed); goto done;
        }
        at += n;
        fprintf(stderr, "teacher-forced positions %u/%u\n", at, visible);
    }
    const float *logits = llama_get_logits_ith(ctx, 0);
    if (!logits) goto done;
    for (size_t i = 0; i < VOCAB; ++i) if (!isfinite(logits[i])) {
        fprintf(stderr, "nonfinite independent logit at %zu\n", i); goto done;
    }
    if (write_file(dir, "logits.f32", logits, 4u*VOCAB)) goto done;
    if (fflush(c.catalog)) goto done;
    int complete = 1;
    for (int b = 3; b < LAYERS; b += 4)
        if (!c.scores[b] || !c.selections[b] || !c.outputs[b]) complete = 0;
    FILE *report = new_file(dir, "capture.json");
    if (!report) goto done;
    fprintf(report, "{\n\"schema\":\"flash-sass.independent-forward-capture.v1\",\n"
            "\"status\":\"%s\",\n\"source_revision\":", complete ? "CAPTURED_NOT_VALIDATED" : "INCOMPLETE_LAYER_DIAGNOSTICS");
    json_string(report, FLASH_REV);
    fputs(",\n\"independent_runtime\":\"llama.cpp public C API\",\n\"system_info\":", report);
    json_string(report, llama_print_system_info());
    fprintf(report, ",\n\"visible\":%u,\"logit_rows\":1,\"logit_position\":%u,\"vocab\":%d,\"model_layers\":%d,\n"
            "\"teacher_forced\":true,\"cpu_only\":true,\"kv_dtype\":\"f32\",\"threads\":%u,\n"
            "\"reference_contract\":\"llama.cpp runtime; NOT the pinned Transformers rounding contract\",\n"
            "\"original_checkpoint_lineage_verified\":false,\"independent_checkpoint_acceptance\":false,\n"
            "\"selection_index_unit\":\"llama.cpp cache cell; retain mask/padding semantics before comparison\",\n"
            "\"tensor_catalog\":\"tensors.jsonl\",\"tensor_count\":%u,\"per_layer\":[\n",
            visible, visible-1, VOCAB, LAYERS, threads, c.tensors);
    for (int b = 3; b < LAYERS; b += 4)
        fprintf(report, "%s{\"layer\":%d,\"score_tensors\":%u,\"selection_tensors\":%u,\"attention_output_tensors\":%u}",
                b == 3 ? "" : ",\n", b, c.scores[b], c.selections[b], c.outputs[b]);
    fprintf(report, "\n],\"required_layer_diagnostics_present\":%s,\"complete\":true\n}\n", complete ? "true" : "false");
    int report_bad = ferror(report); if (fclose(report)) report_bad = 1;
    if (report_bad) {
        char path[4096]; if (!path_join(path, dir, "capture.json")) unlink(path);
        goto done;
    }
    rc = complete ? 0 : 3;
    printf("%s: %s (capture only; no checkpoint acceptance claimed)\n", complete ? "CAPTURED_NOT_VALIDATED" : "INCOMPLETE_LAYER_DIAGNOSTICS", dir);
done:
    if (batch_live) llama_batch_free(batch);
    if (ctx) llama_free(ctx);
    if (model) llama_model_free(model);
    llama_backend_free();
    if (fclose(c.catalog)) {
        char path[4096]; if (!path_join(path, dir, "capture.json")) unlink(path);
        rc = 2;
    }
    free(ids);
    return rc;
}
