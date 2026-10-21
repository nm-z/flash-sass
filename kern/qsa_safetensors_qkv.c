/* Project deterministic synthetic hidden states with original BF16 QSA weights.
 * This is a layer-component input generator, not a model forward or quality run. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef FLASH_REV
#define FLASH_REV "unknown"
#endif

#define FILE_BYTES UINT64_C(1935433136)
#define HEADER_BYTES UINT64_C(4584)
#define DATA_START (UINT64_C(8) + HEADER_BYTES)
#define HIDDEN 2560u
#define Q_ROWS 12288u
#define KV_ROWS 512u
#define NORM_SIZE 256u
#define MAX_POSITIONS 256u
#define CHECKPOINT_REV "de4b8e4d43b917e7706784d8bb445c9af86a3540"

typedef struct {
    const char *name;
    const char *shape;
    const char *offsets;
    uint64_t offset;
    size_t rows;
    size_t columns;
} tensor_spec;

static const tensor_spec Q_PROJ = {
    "model.language_model.layers.3.self_attn.q_proj.weight",
    "\"shape\":[12288,2560]", "\"data_offsets\":[1740745216,1803659776]",
    UINT64_C(1740745216), 12288u, 2560u
};
static const tensor_spec K_PROJ = {
    "model.language_model.layers.3.self_attn.k_proj.weight",
    "\"shape\":[512,2560]", "\"data_offsets\":[1706665984,1709287424]",
    UINT64_C(1706665984), 512u, 2560u
};
static const tensor_spec V_PROJ = {
    "model.language_model.layers.3.self_attn.v_proj.weight",
    "\"shape\":[512,2560]", "\"data_offsets\":[1803659776,1806281216]",
    UINT64_C(1803659776), 512u, 2560u
};
static const tensor_spec Q_NORM = {
    "model.language_model.layers.3.self_attn.q_norm.weight",
    "\"shape\":[256]", "\"data_offsets\":[1740744704,1740745216]",
    UINT64_C(1740744704), 256u, 1u
};
static const tensor_spec K_NORM = {
    "model.language_model.layers.3.self_attn.k_norm.weight",
    "\"shape\":[256]", "\"data_offsets\":[1706665472,1706665984]",
    UINT64_C(1706665472), 256u, 1u
};

static void fail(const char *message)
{
    fprintf(stderr, "qsa_safetensors_qkv: %s\n", message);
    exit(2);
}

static unsigned parse_unsigned(const char *value, unsigned minimum, unsigned maximum)
{
    unsigned result = 0;
    if (!value || !*value) fail("missing unsigned argument");
    for (const char *p = value; *p; ++p) {
        if (*p < '0' || *p > '9') fail("invalid unsigned argument");
        unsigned digit = (unsigned)(*p - '0');
        if (result > maximum / 10 ||
            (result == maximum / 10 && digit > maximum % 10))
            fail("unsigned argument overflow");
        result = result * 10 + digit;
    }
    if (result < minimum || result > maximum) fail("unsigned argument out of range");
    return result;
}

static uint64_t little_u64(const unsigned char *bytes)
{
    uint64_t value = 0;
    for (unsigned i = 0; i < 8; ++i) value |= (uint64_t)bytes[i] << (8 * i);
    return value;
}

static int has_between(const char *start, const char *end, const char *needle)
{
    size_t remaining = (size_t)(end - start);
    size_t length = strlen(needle);
    if (length > remaining) return 0;
    for (size_t i = 0; i + length <= remaining; ++i)
        if (!memcmp(start + i, needle, length)) return 1;
    return 0;
}

static void check_tensor_header(const char *header, const tensor_spec *spec)
{
    char key[256];
    int written = snprintf(key, sizeof key, "\"%s\"", spec->name);
    if (written < 0 || (size_t)written >= sizeof key) fail("tensor name too long");
    const char *start = strstr(header, key);
    if (!start) fail("required layer-3 tensor is absent from safetensors header");
    const char *end = strchr(start, '}');
    if (!end || !has_between(start, end, "\"dtype\":\"BF16\"") ||
        !has_between(start, end, spec->shape) || !has_between(start, end, spec->offsets))
        fail("pinned layer-3 tensor metadata differs from expected shape or offsets");
}

static float bf16_to_float(const unsigned char *bytes)
{
    uint32_t bits = (uint32_t)bytes[0] << 24 | (uint32_t)bytes[1] << 16;
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static uint32_t next_random(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static float hidden_value(uint32_t *state)
{
    return (((float)((next_random(state) >> 8) & 65535u) / 32768.0f) - 1.0f) * 0.05f;
}

static void project(const unsigned char *data, uint64_t offset, size_t rows,
                    const float *input, float *output)
{
    const unsigned char *weights = data + offset;
    for (size_t row = 0; row < rows; ++row) {
        const unsigned char *weight_row = weights + row * HIDDEN * 2u;
        float sum = 0.0f;
        for (size_t column = 0; column < HIDDEN; ++column)
            sum = sum + bf16_to_float(weight_row + column * 2u) * input[column];
        output[row] = sum;
    }
}

static void load_norm(const unsigned char *data, uint64_t offset, float *output)
{
    const unsigned char *weights = data + offset;
    for (size_t i = 0; i < NORM_SIZE; ++i) output[i] = bf16_to_float(weights + i * 2u);
}

static void write_bytes(const char *directory, const char *leaf,
                        const void *data, size_t bytes)
{
    char path[4096];
    int length = snprintf(path, sizeof path, "%s/%s", directory, leaf);
    if (length < 0 || (size_t)length >= sizeof path) fail("output path too long");
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW, 0600);
    if (fd < 0) fail("cannot create output file");
    const unsigned char *source = data;
    size_t written = 0;
    while (written < bytes) {
        ssize_t count = write(fd, source + written, bytes - written);
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) fail("output write failed");
        written += (size_t)count;
    }
    if (close(fd)) fail("cannot close output file");
}

static void write_json(const char *directory, const char *sha256, unsigned positions)
{
    char json[2048];
    int length = snprintf(json, sizeof json,
        "{\"schema\":\"flash-sass.issue22-original-layer3-qkv.v1\","
        "\"complete\":true,\"checkpoint_repo\":\"Qwen/Qwen3.8-Flash-Next\","
        "\"checkpoint_revision\":\"%s\",\"weights_file\":\"model-00082-of-00131.safetensors\","
        "\"weights_sha256\":\"%s\",\"weights_file_bytes\":%llu,\"layer\":3,"
        "\"weights_dtype\":\"BF16\",\"weight_accumulation\":\"FP32\","
        "\"positions\":%u,\"hidden_size\":2560,\"query_projection_rows\":12288,"
        "\"kv_projection_rows\":512,\"hidden_input_generator\":\"fixed_lcg_fp32_v1\","
        "\"hidden_input_seed\":4242,\"teacher_forced\":false,"
        "\"model_forward\":false,\"production_indexer_executed\":false,"
        "\"selection\":\"provided_fixture_ids\",\"full_model\":false}\n",
        CHECKPOINT_REV, sha256, (unsigned long long)FILE_BYTES, positions);
    if (length < 0 || (size_t)length >= sizeof json) fail("metadata too long");
    write_bytes(directory, "projection-complete.json", json, (size_t)length);
}

int main(int argc, char **argv)
{
    if (argc != 5) {
        fprintf(stderr, "usage: %s SAFETENSORS SHA256 POSITIONS NEW_DIRECTORY\n", argv[0]);
        return 2;
    }
    unsigned positions = parse_unsigned(argv[3], 1, MAX_POSITIONS);
    const char *sha256 = argv[2];
    if (strlen(sha256) != 64) fail("weights SHA-256 must have 64 hexadecimal digits");
    for (size_t i = 0; i < 64; ++i)
        if (!((sha256[i] >= '0' && sha256[i] <= '9') ||
              (sha256[i] >= 'a' && sha256[i] <= 'f')))
            fail("weights SHA-256 must be lowercase hexadecimal");

    int fd = open(argv[1], O_RDONLY | O_NOFOLLOW);
    if (fd < 0) fail("cannot open safetensors file");
    struct stat statbuf;
    if (fstat(fd, &statbuf) || !S_ISREG(statbuf.st_mode) ||
        (uint64_t)statbuf.st_size != FILE_BYTES)
        fail("unexpected pinned safetensors file size or type");
    const unsigned char *mapping = mmap(NULL, (size_t)FILE_BYTES, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapping == MAP_FAILED) fail("cannot map safetensors file");
    if (close(fd)) fail("cannot close safetensors descriptor");
    uint64_t header_bytes = little_u64(mapping);
    if (header_bytes != HEADER_BYTES) fail("unexpected pinned safetensors header size");
    char *header = malloc((size_t)HEADER_BYTES + 1u);
    if (!header) fail("header allocation failed");
    memcpy(header, mapping + 8, (size_t)HEADER_BYTES);
    header[HEADER_BYTES] = '\0';
    const tensor_spec *specs[] = {&Q_PROJ, &K_PROJ, &V_PROJ, &Q_NORM, &K_NORM};
    for (size_t i = 0; i < sizeof specs / sizeof specs[0]; ++i)
        check_tensor_header(header, specs[i]);
    free(header);
    const unsigned char *data = mapping + DATA_START;
    const tensor_spec *matrix_specs[] = {&Q_PROJ, &K_PROJ, &V_PROJ};
    for (size_t i = 0; i < sizeof matrix_specs / sizeof matrix_specs[0]; ++i) {
        uint64_t bytes = (uint64_t)matrix_specs[i]->rows * matrix_specs[i]->columns * 2u;
        if (matrix_specs[i]->offset + bytes > FILE_BYTES - DATA_START)
            fail("tensor exceeds safetensors data region");
    }
    if (Q_NORM.offset + NORM_SIZE * 2u > FILE_BYTES - DATA_START ||
        K_NORM.offset + NORM_SIZE * 2u > FILE_BYTES - DATA_START)
        fail("normalization tensor exceeds safetensors data region");

    char *output_directory = argv[4];
    if (mkdir(output_directory, 0700)) fail("output directory must be new");
    size_t hidden_count = (size_t)positions * HIDDEN;
    size_t kv_count = (size_t)positions * KV_ROWS;
    float *hidden = malloc(hidden_count * sizeof *hidden);
    float *qg = malloc(Q_ROWS * sizeof *qg);
    float *keys = malloc(kv_count * sizeof *keys);
    float *values = malloc(kv_count * sizeof *values);
    if (!hidden || !qg || !keys || !values) fail("projection allocation failed");
    uint32_t random_state = 4242u;
    for (size_t i = 0; i < hidden_count; ++i) hidden[i] = hidden_value(&random_state);
    project(data, Q_PROJ.offset, Q_ROWS,
            hidden + (size_t)(positions - 1u) * HIDDEN, qg);
    for (unsigned position = 0; position < positions; ++position) {
        const float *input = hidden + (size_t)position * HIDDEN;
        project(data, K_PROJ.offset, KV_ROWS, input, keys + (size_t)position * KV_ROWS);
        project(data, V_PROJ.offset, KV_ROWS, input, values + (size_t)position * KV_ROWS);
    }
    float qn[NORM_SIZE], kn[NORM_SIZE];
    load_norm(data, Q_NORM.offset, qn);
    load_norm(data, K_NORM.offset, kn);
    write_bytes(output_directory, "hidden_input.f32", hidden, hidden_count * sizeof *hidden);
    write_bytes(output_directory, "qg.f32", qg, Q_ROWS * sizeof *qg);
    write_bytes(output_directory, "raw_k.f32", keys, kv_count * sizeof *keys);
    write_bytes(output_directory, "raw_v.f32", values, kv_count * sizeof *values);
    write_bytes(output_directory, "qn.f32", qn, sizeof qn);
    write_bytes(output_directory, "kn.f32", kn, sizeof kn);
    write_json(output_directory, sha256, positions);
    printf("{\"schema\":\"flash-sass.issue22-original-layer3-qkv-result.v1\","
           "\"complete\":true,\"positions\":%u,\"hidden_size\":%u,"
           "\"query_projection_values\":%u,\"kv_projection_values_per_position\":%u,"
           "\"checkpoint_revision\":\"%s\",\"activation_source\":\"fixed_lcg_fp32_v1\","
           "\"full_model\":false,\"quality\":false}\n",
           positions, HIDDEN, Q_ROWS, KV_ROWS, CHECKPOINT_REV);
    free(values); free(keys); free(qg); free(hidden);
    if (munmap((void *)mapping, (size_t)FILE_BYTES)) fail("cannot unmap safetensors file");
    return 0;
}
