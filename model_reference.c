/*
 * Capture teacher-forced full-vocabulary logits from an independent llama.cpp
 * build through its C API. This is a same-GGUF backend comparison, not an
 * original-checkpoint quality attestation. No sampling or token insertion.
 *
 * Tested API: llama.cpp 7182ef71e6d5dd0c8ea76486fac134fcd25025d4.
 * Build with LLAMA_DIR pointing to that independent source/build.
 */
#define _POSIX_C_SOURCE 200809L
#include "llama.h"
#include <errno.h>
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
#include <time.h>

static double deadline;

static void die(const char *message) {
    fprintf(stderr, "model_reference: %s (%s)\n", message, strerror(errno));
    exit(2);
}
static double now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) die("clock_gettime");
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
static bool aborted(void *unused) { (void)unused; return now() > deadline; }
static bool loading(float progress, void *unused) {
    (void)progress; (void)unused; return !aborted(NULL);
}
static long positive(const char *s, long maximum) {
    char *end;
    errno = 0;
    long n = strtol(s, &end, 10);
    if (errno || !*s || *end || n < 1 || n > maximum) die("invalid positive argument");
    return n;
}
static FILE *output_file(const char *directory, const char *name) {
    size_t size = strlen(directory) + strlen(name) + 2;
    char *path = malloc(size);
    if (!path) die("allocate output path");
    if (snprintf(path, size, "%s/%s", directory, name) < 0) die("output path");
    FILE *f = fopen(path, "wbx");
    free(path);
    if (!f) die("create new output file");
    return f;
}
static void finish(FILE *f) {
    if (ferror(f) || fclose(f)) die("close output");
}
static void bytes(FILE *f, const void *data, size_t n) {
    if (n && fwrite(data, 1, n, f) != n) die("write output");
}
static void quoted(FILE *f, const char *s) {
    fputc('"', f);
    for (; *s; ++s) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') { fputc('\\', f); fputc(c, f); }
        else if (c < 32) fprintf(f, "\\u%04x", (unsigned)c);
        else fputc(c, f);
    }
    fputc('"', f);
}
static llama_token *read_ids(const char *path, size_t *count) {
    FILE *f = fopen(path, "rb");
    if (!f) die("open text IDs");
    size_t capacity = 32, n = 0;
    llama_token *ids = malloc(capacity * sizeof(*ids));
    if (!ids) die("allocate IDs");
    char word[64];
    while (fscanf(f, "%63s", word) == 1) {
        char *end;
        errno = 0;
        unsigned long id = strtoul(word, &end, 10);
        if (errno || word[0] == '-' || *end || id > INT32_MAX) die("invalid token ID");
        if (n == 262000) die("at most 262000 positions supported by this capture tool");
        if (n == capacity) {
            capacity *= 2;
            void *next = realloc(ids, capacity * sizeof(*ids));
            if (!next) die("grow IDs");
            ids = next;
        }
        ids[n++] = (llama_token)id;
    }
    if (ferror(f) || fclose(f) || !n) die("read nonempty text IDs");
    *count = n;
    return ids;
}
static void write_model_metadata(const char *out, const struct llama_model *model) {
    FILE *f = output_file(out, "model_metadata.json");
    int32_t count = llama_model_meta_count(model);
    fputs("{\n", f);
    for (int32_t i = 0; i < count; ++i) {
        char key[512];
        int32_t key_len = llama_model_meta_key_by_index(model, i, key, sizeof(key));
        if (key_len < 0 || (size_t)key_len >= sizeof(key)) die("model metadata key");
        int32_t length = llama_model_meta_val_str_by_index(model, i, NULL, 0);
        if (length < 0) die("model metadata value size");
        char *value = malloc((size_t)length + 1);
        if (!value) die("allocate metadata value");
        if (llama_model_meta_val_str_by_index(model, i, value, (size_t)length + 1) != length)
            die("model metadata changed");
        if (i) fputs(",\n", f);
        quoted(f, key); fputc(':', f); quoted(f, value);
        free(value);
    }
    fputs("\n}\n", f);
    finish(f);
}

int main(int argc, char **argv) {
    if (argc < 4 || argc > 6) {
        fputs("usage: model_reference MODEL_GGUF IDS_TXT NEW_OUTPUT_DIR [THREADS=2] [SECONDS=1800]\n", stderr);
        return 2;
    }
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1 || sizeof(float) != 4 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(llama_token) != 4)
        die("requires little-endian IEEE float32 host");
    int threads = argc > 4 ? (int)positive(argv[4], 64) : 2;
    long seconds = argc > 5 ? positive(argv[5], 86400) : 1800;
    size_t positions;
    llama_token *ids = read_ids(argv[2], &positions);
    if (mkdir(argv[3], 0700)) die("create new capture directory");
    const double started = now();
    deadline = started + (double)seconds;

    /* Deliberately do not dynamically discover/load GPU backends. */
    llama_backend_init();
    ggml_backend_dev_t no_devices[] = {NULL};
    struct llama_model_params mp = llama_model_default_params();
    mp.devices = no_devices;
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.lazy_mode = LLAMA_LAZY_MODE_ON;
    mp.use_extra_bufts = false;
    mp.load_mtp = false;
    mp.progress_callback = loading;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) die("independent model load failed");
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    int32_t v = llama_vocab_n_tokens(vocab);
    if (v <= 0) die("invalid vocabulary size");
    for (size_t i = 0; i < positions; ++i)
        if (ids[i] < 0 || ids[i] >= v) die("token ID outside vocabulary");
    write_model_metadata(argv[3], model);

    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = positions < 128 ? 128 : (uint32_t)positions;
    cp.n_batch = 1;
    cp.n_ubatch = 1;
    cp.n_seq_max = 1;
    cp.n_threads = threads;
    cp.n_threads_batch = threads;
    cp.offload_kqv = false;
    cp.op_offload = false;
    cp.type_k = GGML_TYPE_F32;
    cp.type_v = GGML_TYPE_F32;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.abort_callback = aborted;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("independent context creation failed");

    FILE *f = output_file(argv[3], "ids.u32");
    bytes(f, ids, positions * sizeof(*ids));
    finish(f);
    f = output_file(argv[3], "vocab_ids.u32");
    for (uint32_t i = 0; i < (uint32_t)v; ++i) bytes(f, &i, sizeof(i));
    finish(f);
    /*
     * Ordered vocabulary text: repeated [u32 byte length][raw bytes].
     * Hash alongside GGUF shards; a hash alone does not establish origin.
     */
    f = output_file(argv[3], "vocabulary_text.bin");
    for (int32_t i = 0; i < v; ++i) {
        const char *token = llama_vocab_get_text(vocab, i);
        if (!token || strlen(token) > UINT32_MAX) die("vocabulary text");
        uint32_t length = (uint32_t)strlen(token);
        bytes(f, &length, sizeof(length)); bytes(f, token, length);
    }
    finish(f);
    FILE *logits = output_file(argv[3], "logits.f32");
    for (size_t i = 0; i < positions; ++i) {
        if (aborted(NULL)) die("capture deadline exceeded");
        double tick = now();
        struct llama_batch batch = llama_batch_get_one(ids + i, 1);
        int32_t result = llama_decode(ctx, batch);
        if (result) {
            fprintf(stderr, "model_reference: decode failed at position %zu, result=%" PRId32 "\n", i, result);
            return 3;
        }
        const float *row = llama_get_logits_ith(ctx, -1);
        if (!row) die("missing output logit row");
        for (int32_t j = 0; j < v; ++j)
            if (!isfinite(row[j])) {
                fprintf(stderr, "model_reference: nonfinite logit at position %zu, token %" PRId32 "\n", i, j);
                return 3;
            }
        bytes(logits, row, (size_t)v * sizeof(float));
        if (fflush(logits)) die("flush logits");
        fprintf(stderr, "model_reference: position=%zu input_id=%" PRId32 " vocabulary=%" PRId32 " seconds=%.6f\n",
                i, ids[i], v, now() - tick);
    }
    finish(logits);
    /* Completion metadata is last; partial logits have no completion marker. */
    f = output_file(argv[3], "capture_info.json");
    fprintf(f, "{\n\"schema\":\"flash-sass.independent-gguf-capture.v1\","
               "\n\"teacher_forced\":true,\"positions\":%zu,\"vocab_size\":%" PRId32 ","
               "\n\"dtype\":\"<f4\",\"byte_order\":\"little\","
               "\n\"input_semantics\":\"row t predicts after supplied token t; no inserted tokens\","
               "\n\"independent_checkpoint_evidence\":false,\"quality_validation\":false,"
               "\n\"scope\":\"CPU llama.cpp same-quantized-GGUF numerical capture\","
               "\n\"gpu_layers\":0,\"offload_kqv\":false,\"op_offload\":false,"
               "\n\"kv_dtype\":\"f32\",\"flash_attention\":false,\"mmap\":true,"
               "\n\"lazy_marked_tensors\":true,\"extra_buffer_types\":false,\"mtp\":false,"
               "\n\"threads\":%d,\"context_capacity\":%" PRIu32 ",\"elapsed_seconds\":%.6f,"
               "\n\"system_info\":",
            positions, v, threads, llama_n_ctx(ctx), now() - started);
    quoted(f, llama_print_system_info());
    fputs(",\n\"command\":[", f);
    for (int i = 0; i < argc; ++i) {
        if (i) fputc(',', f);
        quoted(f, argv[i]);
    }
    fputs("],\n\"complete\":true\n}\n", f);
    finish(f);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(ids);
    return 0;
}
