/*
 * Capture teacher-forced full-vocabulary logits from an independent llama.cpp
 * build through its C API. This is a same-GGUF backend comparison, not an
 * original-checkpoint quality attestation. No sampling or token insertion.
 *
 * Derived from kern/model_reference.c in nm-z/flash-sass PR #46, commit
 * fcf435f728760b64de30214478524666093468ec. This adapter additionally captures
 * selected prefix lengths and the existing native graph's QSA diagnostics.
 * Tested API: llama.cpp fcac19079f3f2985fe74ed845f0880e7e9a8030a.
 * Build with QSA_LLAMA_DIR pointing to that independent source/build.
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

#define MAX_POSITIONS 262144
#define MODEL_LAYERS 48
#define MODEL_VOCAB 248320

static double deadline;

static void die(const char *message) {
    fprintf(stderr, "qsa_model_reference: %s (%s)\n", message, strerror(errno));
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
        if (n == MAX_POSITIONS) die("at most 262144 positions supported by this capture tool");
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

/* Names are the native runtime's graph callback names, not reimplementations
 * of its math. Kcur/Vcur also name the pre-reshape projections: the op checks
 * below select only the final current-token key and value. */
enum node_kind {
    RAW_INDEX_KEY, POOLED_KEYS, INDEX_QUERY, SCORES, TOP_POOLS, SELECTION_MASK,
    ATTENTION_QUERY, ATTENTION_KEY, ATTENTION_VALUE,
    ATTENTION_PREGATE, ATTENTION_GATED, ATTENTION_OUTPUT, BLOCK_OUTPUT,
    NEW_POOLED_KEYS, NODE_COUNT
};
static const char *const node_names[NODE_COUNT] = {
    "indexer_k_raw", "indexer_k", "indexer_q", "indexer_score",
    "indexer_top_k", "indexer_sel", "Qcur", "Kcur", "Vcur",
    "attn_pregate", "attn_gated", "attn_output", "l_last",
    "indexer_pool_k_new"
};
typedef struct {
    const char *out;
    FILE *index;
    bool *selected;
    size_t length, rows, tensor_count;
    uint32_t seen[MODEL_LAYERS];
} observer;

static bool *read_lengths(const char *text, size_t positions, size_t *rows) {
    bool *selected = calloc(positions + 1, sizeof(*selected));
    if (!selected) die("allocate selected lengths");
    const char *p = text;
    size_t previous = 0;
    *rows = 0;
    do {
        size_t length = 0;
        if (*p < '0' || *p > '9') die("lengths must be positive decimal integers");
        while (*p >= '0' && *p <= '9') {
            unsigned digit = (unsigned)(*p++ - '0');
            if (length > (MAX_POSITIONS - digit) / 10) die("visible length overflow");
            length = length * 10 + digit;
        }
        if (length <= previous || length > positions)
            die("visible lengths must increase strictly and fit the full input prefix");
        selected[length] = true;
        previous = length;
        ++*rows;
        if (!*p) break;
        if (*p++ != ',') die("lengths must be comma separated");
    } while (true);
    if (previous != positions) die("last selected length must equal the complete input length");
    return selected;
}

static size_t checked_product(size_t a, size_t b) {
    if (b && a > SIZE_MAX / b) die("tensor size overflow");
    return a * b;
}

/* Preserve native little-endian dtype bits while packing logical ne[0]-fastest
 * elements. Views can have gaps or transposed strides, so fwrite(t->data) is
 * not a valid capture of an arbitrary callback tensor. */
static unsigned char *pack_tensor(const struct ggml_tensor *t, size_t *size) {
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 &&
        t->type != GGML_TYPE_BF16 && t->type != GGML_TYPE_I32)
        die("unsupported diagnostic tensor dtype");
    size_t element = ggml_type_size(t->type), count = 1;
    for (int d = 0; d < 4; ++d) {
        if (t->ne[d] <= 0 || (uint64_t)t->ne[d] > SIZE_MAX) die("invalid tensor shape");
        count = checked_product(count, (size_t)t->ne[d]);
    }
    *size = checked_product(count, element);
    size_t extent = ggml_nbytes(t);
    /* Bound each individual readback; the intended largest pooled tensor is
     * about 32 MiB at 262144 tokens. Reject an accidental whole-weight capture. */
    if (!extent || extent > (size_t)512 * 1024 * 1024 || *size > (size_t)512 * 1024 * 1024)
        die("diagnostic tensor exceeds 512 MiB limit");
    unsigned char *raw = malloc(extent), *packed = malloc(*size);
    if (!raw || !packed) die("allocate diagnostic readback");
    ggml_backend_tensor_get(t, raw, 0, extent);
    for (size_t i = 0; i < count; ++i) {
        size_t remaining = i, offset = 0;
        for (int d = 0; d < 4; ++d) {
            size_t coordinate = remaining % (size_t)t->ne[d];
            remaining /= (size_t)t->ne[d];
            size_t term = checked_product(coordinate, t->nb[d]);
            if (term > SIZE_MAX - offset) die("tensor stride overflow");
            offset += term;
        }
        if (element > extent || offset > extent - element) die("tensor stride outside readback");
        memcpy(packed + i * element, raw + offset, element);
    }
    free(raw);
    return packed;
}

static void validate_values(const struct ggml_tensor *t, const unsigned char *data,
                            size_t size, enum node_kind kind) {
    if (t->type == GGML_TYPE_I32) {
        if (kind != TOP_POOLS) die("integer tensor outside top-pool diagnostics");
        return;
    }
    bool allow_negative_infinity = kind == SCORES || kind == SELECTION_MASK;
    size_t element = ggml_type_size(t->type);
    for (size_t i = 0; i < size; i += element) {
        uint32_t bits = 0, exponent, fraction, negative_infinity;
        if (t->type == GGML_TYPE_F32) {
            memcpy(&bits, data + i, 4);
            exponent = 0x7f800000u; fraction = 0x007fffffu; negative_infinity = 0xff800000u;
        } else {
            uint16_t half;
            memcpy(&half, data + i, 2);
            bits = half;
            exponent = t->type == GGML_TYPE_F16 ? 0x7c00u : 0x7f80u;
            fraction = t->type == GGML_TYPE_F16 ? 0x03ffu : 0x007fu;
            negative_infinity = exponent | 0x8000u;
        }
        if ((bits & exponent) == exponent &&
            (!allow_negative_infinity || (bits & fraction) || bits != negative_infinity))
            die("NaN or disallowed infinity in diagnostic tensor");
    }
}

static int identify(const struct ggml_tensor *t, int *layer) {
    const char *suffix = strrchr(t->name, '-');
    if (!suffix || !suffix[1]) return -1;
    char *end;
    long value = strtol(suffix + 1, &end, 10);
    if (*end || value < 0 || value >= MODEL_LAYERS) return -1;
    *layer = (int)value;
    size_t prefix = (size_t)(suffix - t->name);
    for (int kind = 0; kind < NODE_COUNT; ++kind) {
        if (strlen(node_names[kind]) != prefix || strncmp(t->name, node_names[kind], prefix)) continue;
        if (kind != BLOCK_OUTPUT && *layer % 4 != 3) return -1;
        if (kind == ATTENTION_KEY && t->op != GGML_OP_ROPE) return -1;
        if (kind == ATTENTION_VALUE && t->op != GGML_OP_RESHAPE) return -1;
        return kind;
    }
    return -1;
}

/* indexer_sel is a 0/-inf mask over cache cells, NOT a token-ID tensor. In this
 * single sequence, non-evicting decode, cache cell i is prefix position i.
 * Export the visible selected positions separately, in ascending set order. */
static size_t selected_ids(observer *o, const struct ggml_tensor *t,
                           const unsigned char *data, int layer) {
    if (t->type != GGML_TYPE_F32 || t->ne[1] != 1 || t->ne[2] != 1 || t->ne[3] != 1 ||
        (uint64_t)t->ne[0] < o->length) die("unexpected selection mask shape or dtype");
    char name[128];
    snprintf(name, sizeof(name), "L%06zu-B%02d-selected_ids.u32", o->length, layer);
    FILE *f = output_file(o->out, name);
    size_t count = 0;
    for (size_t i = 0; i < (size_t)t->ne[0]; ++i) {
        uint32_t bits;
        memcpy(&bits, data + i * 4, 4);
        if ((bits & 0x7fffffffu) == 0) {
            if (i >= o->length) die("selection includes a future or padding cell");
            uint32_t position = (uint32_t)i;
            bytes(f, &position, 4);
            ++count;
        } else if (bits != 0xff800000u) die("selection mask must contain only zero or negative infinity");
        if (i >= o->length / 4 * 4 && i < o->length && (bits & 0x7fffffffu) != 0)
            die("selection omitted an incomplete-tail position");
    }
    size_t pools = o->length / 4;
    if (pools > 512) pools = 512;
    if (count != pools * 4 + o->length % 4) die("selection count violates top-2048 plus incomplete tail");
    finish(f);
    return count;
}

static bool observe(struct ggml_tensor *t, bool ask, void *data) {
    observer *o = data;
    if (!o->length || !o->selected[o->length]) return ask ? false : true;
    int layer = -1, kind = identify(t, &layer);
    if (kind < 0) return ask ? false : true;
    if (ask) return true;
    uint32_t bit = (uint32_t)1 << kind;
    if (o->seen[layer] & bit) die("duplicate diagnostic node at one prefix length");
    size_t size;
    unsigned char *packed = pack_tensor(t, &size);
    validate_values(t, packed, size, (enum node_kind)kind);
    char name[160];
    snprintf(name, sizeof(name), "L%06zu-B%02d-%s.bin", o->length, layer, node_names[kind]);
    FILE *f = output_file(o->out, name);
    bytes(f, packed, size);
    finish(f);
    size_t count = kind == SELECTION_MASK ? selected_ids(o, t, packed, layer) : 0;
    free(packed);
    fprintf(o->index, "{\"visible_length\":%zu,\"position\":%zu,\"layer\":%d,\"node\":",
            o->length, o->length - 1, layer);
    quoted(o->index, node_names[kind]);
    fputs(",\"file\":", o->index); quoted(o->index, name);
    fputs(",\"dtype\":", o->index); quoted(o->index, ggml_type_name(t->type));
    fprintf(o->index, ",\"bytes\":%zu,\"ne\":[%" PRId64 ",%" PRId64 ",%" PRId64 ",%" PRId64 "]"
            ",\"source_nb\":[%zu,%zu,%zu,%zu],\"layout\":\"packed ne[0] fastest\"",
            size, t->ne[0], t->ne[1], t->ne[2], t->ne[3], t->nb[0], t->nb[1], t->nb[2], t->nb[3]);
    if (kind == SELECTION_MASK) fprintf(o->index, ",\"selected_positions\":%zu", count);
    fputs("}\n", o->index);
    if (fflush(o->index)) die("flush tensor index");
    o->seen[layer] |= bit;
    ++o->tensor_count;
    return true;
}

static void finish_position(observer *o) {
    if (!o->selected[o->length]) return;
    uint32_t required_qsa = ((uint32_t)1 << BLOCK_OUTPUT) - 1;
    for (int layer = 0; layer < MODEL_LAYERS; ++layer) {
        uint32_t required = (uint32_t)1 << BLOCK_OUTPUT;
        if (layer % 4 == 3) required |= required_qsa;
        if ((o->seen[layer] & required) != required) {
            fprintf(stderr, "qsa_model_reference: missing diagnostics at length=%zu layer=%d required=0x%x seen=0x%x\n",
                    o->length, layer, required, o->seen[layer]);
            die("incomplete per-layer diagnostics");
        }
    }
}

int main(int argc, char **argv) {
    if (argc < 5 || argc > 7) {
        fprintf(stderr, "usage: %s MODEL_GGUF IDS_TXT VISIBLE_LENGTHS_CSV NEW_OUTPUT_DIR [THREADS=2] [SECONDS=1800]\n", argv[0]);
        return 2;
    }
    uint16_t endian = 1;
    if (*(unsigned char *)&endian != 1 || sizeof(float) != 4 ||
        FLT_RADIX != 2 || FLT_MANT_DIG != 24 || sizeof(llama_token) != 4)
        die("requires little-endian IEEE float32 host");
    int threads = argc > 5 ? (int)positive(argv[5], 64) : 2;
    long seconds = argc > 6 ? positive(argv[6], 86400) : 1800;
    size_t positions;
    llama_token *ids = read_ids(argv[2], &positions);
    const char *out = argv[4];
    observer diagnostics = {.out = out};
    diagnostics.selected = read_lengths(argv[3], positions, &diagnostics.rows);
    if (mkdir(out, 0700)) die("create new capture directory");
    diagnostics.index = output_file(out, "tensors.jsonl");
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
    char architecture[64];
    if (llama_model_meta_val_str(model, "general.architecture", architecture, sizeof(architecture)) < 0 ||
        strcmp(architecture, "qwen4exp") || llama_model_n_layer(model) != MODEL_LAYERS || v != MODEL_VOCAB)
        die("capture contract requires qwen4exp with 48 blocks and vocabulary 248320");
    for (size_t i = 0; i < positions; ++i)
        if (ids[i] < 0 || ids[i] >= v) die("token ID outside vocabulary");
    write_model_metadata(out, model);

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
    cp.cb_eval = observe;
    cp.cb_eval_user_data = &diagnostics;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) die("independent context creation failed");

    FILE *f = output_file(out, "all_input_ids.u32");
    bytes(f, ids, positions * sizeof(*ids));
    finish(f);
    f = output_file(out, "ids.u32");
    for (size_t i = 0; i < positions; ++i)
        if (diagnostics.selected[i + 1]) bytes(f, ids + i, sizeof(*ids));
    finish(f);
    f = output_file(out, "positions.u32");
    for (uint32_t i = 0; i < positions; ++i)
        if (diagnostics.selected[i + 1]) bytes(f, &i, sizeof(i));
    finish(f);
    f = output_file(out, "vocab_ids.u32");
    for (uint32_t i = 0; i < (uint32_t)v; ++i) bytes(f, &i, sizeof(i));
    finish(f);
    /*
     * Ordered vocabulary text: repeated [u32 byte length][raw bytes].
     * Hash alongside GGUF shards; a hash alone does not establish origin.
     */
    f = output_file(out, "vocabulary_text.bin");
    for (int32_t i = 0; i < v; ++i) {
        const char *token = llama_vocab_get_text(vocab, i);
        if (!token || strlen(token) > UINT32_MAX) die("vocabulary text");
        uint32_t length = (uint32_t)strlen(token);
        bytes(f, &length, sizeof(length)); bytes(f, token, length);
    }
    finish(f);
    FILE *logits = output_file(out, "logits.f32");
    for (size_t i = 0; i < positions; ++i) {
        if (aborted(NULL)) die("capture deadline exceeded");
        double tick = now();
        diagnostics.length = i + 1;
        memset(diagnostics.seen, 0, sizeof(diagnostics.seen));
        struct llama_batch batch = llama_batch_get_one(ids + i, 1);
        int32_t result = llama_decode(ctx, batch);
        if (result) {
            fprintf(stderr, "qsa_model_reference: decode failed at position %zu, result=%" PRId32 "\n", i, result);
            return 3;
        }
        const float *row = llama_get_logits_ith(ctx, -1);
        if (!row) die("missing output logit row");
        for (int32_t j = 0; j < v; ++j)
            if (!isfinite(row[j])) {
                fprintf(stderr, "qsa_model_reference: nonfinite logit at position %zu, token %" PRId32 "\n", i, j);
                return 3;
            }
        finish_position(&diagnostics);
        if (diagnostics.selected[i + 1]) {
            bytes(logits, row, (size_t)v * sizeof(float));
            if (fflush(logits)) die("flush logits");
        }
        fprintf(stderr, "qsa_model_reference: position=%zu input_id=%" PRId32 " vocabulary=%" PRId32 " captured=%s seconds=%.6f\n",
                i, ids[i], v, diagnostics.selected[i + 1] ? "true" : "false", now() - tick);
    }
    finish(logits);
    finish(diagnostics.index);
    /* Completion metadata is last; partial logits have no completion marker. */
    f = output_file(out, "capture_info.json");
    fprintf(f, "{\n\"schema\":\"flash-sass.qsa-model-reference.v1\","
               "\n\"teacher_forced\":true,\"input_positions\":%zu,\"positions\":%zu,\"vocab_size\":%" PRId32 ","
               "\n\"dtype\":\"<f4\",\"byte_order\":\"little\","
               "\n\"input_semantics\":\"row r predicts after all_input_ids[positions[r]]; no inserted tokens\","
               "\n\"position_dtype\":\"<u4\",\"position_base\":0,\"prefix_decoded_in_full\":true,"
               "\n\"layers\":48,\"qsa_layers\":[3,7,11,15,19,23,27,31,35,39,43,47],\"diagnostic_tensors\":%zu,"
               "\n\"selection_semantics\":\"indexer_sel is a 0/-inf cell mask; selected_ids are ascending prefix positions, not vocabulary IDs\","
               "\n\"independent_checkpoint_evidence\":false,\"quality_validation\":false,"
               "\n\"scope\":\"CPU llama.cpp same-quantized-GGUF numerical capture\","
               "\n\"gpu_layers\":0,\"offload_kqv\":false,\"op_offload\":false,"
               "\n\"kv_dtype\":\"f32\",\"flash_attention\":false,\"mmap\":true,"
               "\n\"lazy_marked_tensors\":true,\"extra_buffer_types\":false,\"mtp\":false,"
               "\n\"threads\":%d,\"context_capacity\":%" PRIu32 ",\"elapsed_seconds\":%.6f,"
               "\n\"system_info\":",
            positions, diagnostics.rows, v, diagnostics.tensor_count, threads, llama_n_ctx(ctx), now() - started);
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
    free(diagnostics.selected);
    free(ids);
    return 0;
}
