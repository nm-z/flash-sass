/* Project deterministic synthetic hidden states with original BF16 QSA weights.
 * This is a layer-component input generator, not a model forward or quality run. */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <fenv.h>
#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "qsa.h"

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
#define INDEX_DIM 128u
#define INDEX_HEADS 4u
#define INDEX_ROWS 640u
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
static const tensor_spec INDEX_QK = {
    "model.language_model.layers.3.self_attn.indexer.index_qk_proj.weight",
    "\"shape\":[640,2560]", "\"data_offsets\":[1703388160,1706664960]",
    UINT64_C(1703388160), 640u, 2560u
};
static const tensor_spec INDEX_Q_NORM = {
    "model.language_model.layers.3.self_attn.indexer.q_layernorm.weight",
    "\"shape\":[128]", "\"data_offsets\":[1706665216,1706665472]",
    UINT64_C(1706665216), 128u, 1u
};
static const tensor_spec INDEX_K_NORM = {
    "model.language_model.layers.3.self_attn.indexer.k_layernorm.weight",
    "\"shape\":[128]", "\"data_offsets\":[1706664960,1706665216]",
    UINT64_C(1706664960), 128u, 1u
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
    uint16_t bf16 = (uint16_t)bytes[0] | (uint16_t)((uint16_t)bytes[1] << 8);
    uint32_t bits = (uint32_t)bf16 << 16;
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
        if (!isfinite(sum)) fail("non-finite BF16 projection result");
        output[row] = sum;
    }
}

static void load_norm(const unsigned char *data, uint64_t offset, float *output)
{
    const unsigned char *weights = data + offset;
    for (size_t i = 0; i < NORM_SIZE; ++i) {
        output[i] = bf16_to_float(weights + i * 2u);
        if (!isfinite(output[i])) fail("non-finite BF16 norm weight");
    }
}

static void load_index_norm(const unsigned char *data, uint64_t offset, float *output)
{
    const unsigned char *weights = data + offset;
    for (size_t i = 0; i < INDEX_DIM; ++i) {
        output[i] = bf16_to_float(weights + i * 2u);
        if (!isfinite(output[i])) fail("non-finite BF16 indexer norm weight");
    }
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
        "\"model_forward\":false,\"indexer_reference_capture_generated\":true,"
        "\"production_indexer_executed\":false,"
        "\"selection\":\"provided_fixture_ids\",\"full_model\":false}\n",
        CHECKPOINT_REV, sha256, (unsigned long long)FILE_BYTES, positions);
    if (length < 0 || (size_t)length >= sizeof json) fail("metadata too long");
    write_bytes(directory, "projection-complete.json", json, (size_t)length);
}

static void write_indexer_capture(const char *output_directory,
                                  const unsigned char *data,const float *hidden,
                                  unsigned positions,const char *sha256)
{
    const unsigned budget=1;
    const unsigned blocks=positions/4;
    const unsigned selected=blocks<budget?blocks:budget;
    size_t projected_count=(size_t)positions*INDEX_ROWS;
    size_t raw_count=(size_t)positions*INDEX_DIM;
    size_t rope_count=(size_t)positions*32u;
    float *projected=malloc(projected_count*sizeof *projected);
    float *raw=malloc(raw_count*sizeof *raw);
    float *cs=malloc(rope_count*sizeof *cs),*sn=malloc(rope_count*sizeof *sn);
    uint16_t *pool=malloc((size_t)blocks*INDEX_DIM*sizeof *pool);
    float *scores=malloc((size_t)blocks*sizeof *scores);
    uint32_t *ids=malloc((size_t)selected*sizeof *ids);
    float *iq=malloc(INDEX_HEADS*INDEX_DIM*sizeof *iq);
    if(!projected||!raw||!cs||!sn||!pool||!scores||!ids||!iq)
        fail("indexer capture allocation failed");
    for(unsigned position=0;position<positions;position++) {
        project(data,INDEX_QK.offset,INDEX_ROWS,
                hidden+(size_t)position*HIDDEN,projected+(size_t)position*INDEX_ROWS);
        memcpy(raw+(size_t)position*INDEX_DIM,
               projected+(size_t)position*INDEX_ROWS+INDEX_HEADS*INDEX_DIM,
               INDEX_DIM*sizeof *raw);
        for(unsigned d=0;d<32;d++) {
            double angle=(double)position*pow(10000000.0,-(double)d/32.0);
            cs[(size_t)position*32+d]=(float)cos(angle);
            sn[(size_t)position*32+d]=(float)sin(angle);
        }
    }
    float iqn[INDEX_DIM],ikn[INDEX_DIM];
    load_index_norm(data,INDEX_Q_NORM.offset,iqn);
    load_index_norm(data,INDEX_K_NORM.offset,ikn);
    const float *iqk=projected+(size_t)(positions-1)*INDEX_ROWS;
    for(unsigned h=0;h<INDEX_HEADS;h++)
        idx_norm_rope_gpu(iqk+(size_t)h*INDEX_DIM,iqn,1e-6f,
            cs+(size_t)(positions-1)*32,sn+(size_t)(positions-1)*32,
            iq+(size_t)h*INDEX_DIM);
    for(unsigned block=0;block<blocks;block++)
        idx_pool_gpu(raw+(size_t)block*4*INDEX_DIM,ikn,1e-6f,
            cs+(size_t)block*4*32,sn+(size_t)block*4*32,
            pool+(size_t)block*INDEX_DIM);
    if(blocks) idx_scores_gpu(iq,pool,blocks,scores);
    unsigned count=idx_select_gpu(scores,blocks,budget,ids);
    if(count!=selected) fail("indexer reference returned an unexpected selection count");
    for(size_t i=0;i<(size_t)blocks*INDEX_DIM;i++)
        if(!isfinite(qsa_h2f(pool[i]))) fail("non-finite FP16 indexer pool");
    for(unsigned i=0;i<blocks;i++) if(!isfinite(scores[i])) fail("non-finite indexer score");

    char root[4096],snapshot[4096],path[4096];
    int n=snprintf(root,sizeof root,"%s/indexer-capture",output_directory);
    if(n<0||(size_t)n>=sizeof root||mkdir(root,0700)) fail("cannot create indexer capture");
    write_bytes(root,"raw_ik.f32",raw,raw_count*sizeof *raw);
    write_bytes(root,"qk_projected.f32",projected,projected_count*sizeof *projected);
    write_bytes(root,"rope_cs.f32",cs,rope_count*sizeof *cs);
    write_bytes(root,"rope_sn.f32",sn,rope_count*sizeof *sn);
    write_bytes(root,"iqn.f32",iqn,sizeof iqn);
    write_bytes(root,"ikn.f32",ikn,sizeof ikn);
    n=snprintf(snapshot,sizeof snapshot,"%s/n%u",root,positions);
    if(n<0||(size_t)n>=sizeof snapshot||mkdir(snapshot,0700)) fail("cannot create indexer snapshot");
    write_bytes(snapshot,"iqk.f32",iqk,INDEX_ROWS*sizeof *iqk);
    write_bytes(snapshot,"iq.f32",iq,INDEX_HEADS*INDEX_DIM*sizeof *iq);
    write_bytes(snapshot,"pool.f16",pool,(size_t)blocks*INDEX_DIM*sizeof *pool);
    write_bytes(snapshot,"selected_ids.u32",ids,(size_t)selected*sizeof *ids);
    if(blocks>budget) write_bytes(snapshot,"scores.f32",scores,(size_t)blocks*sizeof *scores);
    n=snprintf(path,sizeof path,"%s/complete.json",root);
    if(n<0||(size_t)n>=sizeof path) fail("indexer completion path too long");
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0) fail("cannot create indexer completion metadata");
    FILE *f=fdopen(fd,"w");
    if(!f) fail("cannot open indexer completion stream");
    if(fprintf(f,"{\"schema\":\"flash-sass.qsa-layer-capture.v1\","
        "\"positions\":%u,\"budget\":%u,\"epsilon\":1e-6,\"byte_order\":\"little\","
        "\"teacher_forced\":false,\"layer\":3,\"die\":0,\"FLASH_KV_HOST\":0,"
        "\"pool_dtype\":\"fp16\",\"rope_base\":10000000,\"snapshot_lengths\":[%u],"
        "\"evidence_kind\":\"synthetic_cpu_fixture\",\"weights_origin\":\"original_checkpoint_safetensors_layer3\","
        "\"weights_revision\":\"%s\",\"weights_file_sha256\":\"%s\","
        "\"indexer_projection\":\"original_bf16_index_qk_proj\",\"selection_source\":\"reference_qsa_c\","
        "\"production_gpu_indexer_executed\":false,\"full_model\":false}\n",
        positions,budget,positions,CHECKPOINT_REV,sha256)<0) fail("indexer completion write failed");
    if(fflush(f)||fclose(f)) fail("cannot close indexer completion metadata");
    n=snprintf(path,sizeof path,"%s/snapshot.json",snapshot);
    if(n<0||(size_t)n>=sizeof path) fail("indexer snapshot path too long");
    fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0) fail("cannot create indexer snapshot metadata");
    f=fdopen(fd,"w");
    if(!f) fail("cannot open indexer snapshot stream");
    if(fprintf(f,"{\"schema\":\"flash-sass.qsa-layer-snapshot.v1\","
        "\"length\":%u,\"position\":%u,\"budget\":%u,\"complete_blocks\":%u,"
        "\"selected_blocks\":%u,\"tail\":%u,\"attended_keys\":%u,"
        "\"sparse_exclusion\":%s,\"layer\":3,\"die\":0,"
        "\"evidence_kind\":\"synthetic_cpu_fixture\",\"teacher_forced\":false,"
        "\"scores_source\":\"reference_qsa_c\",\"selected_ids_source\":\"reference_qsa_c\"}\n",
        positions,positions-1,budget,blocks,selected,positions%4,
        selected*4+positions%4,blocks>budget?"true":"false")<0)
        fail("indexer snapshot metadata write failed");
    if(fflush(f)||fclose(f)) fail("cannot close indexer snapshot metadata");
    free(iq);free(ids);free(scores);free(pool);free(sn);free(cs);free(raw);free(projected);
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
    uint16_t endian_check = 1;
    if (*(const unsigned char *)&endian_check != 1) fail("little-endian host required");
    char *header = malloc((size_t)HEADER_BYTES + 1u);
    if (!header) fail("header allocation failed");
    memcpy(header, mapping + 8, (size_t)HEADER_BYTES);
    header[HEADER_BYTES] = '\0';
    const tensor_spec *specs[] = {
        &Q_PROJ, &K_PROJ, &V_PROJ, &Q_NORM, &K_NORM,
        &INDEX_QK, &INDEX_Q_NORM, &INDEX_K_NORM
    };
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
    if (INDEX_QK.offset + INDEX_ROWS * HIDDEN * 2u > FILE_BYTES - DATA_START ||
        INDEX_Q_NORM.offset + INDEX_DIM * 2u > FILE_BYTES - DATA_START ||
        INDEX_K_NORM.offset + INDEX_DIM * 2u > FILE_BYTES - DATA_START)
        fail("indexer tensor exceeds safetensors data region");

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
    write_indexer_capture(output_directory,data,hidden,positions,sha256);
    printf("{\"schema\":\"flash-sass.issue22-original-layer3-qkv-result.v1\","
           "\"complete\":true,\"positions\":%u,\"hidden_size\":%u,"
           "\"query_projection_values\":%u,\"kv_projection_values_per_position\":%u,"
           "\"checkpoint_revision\":\"%s\",\"activation_source\":\"fixed_lcg_fp32_v1\","
           "\"indexer_reference_capture_generated\":true,\"full_model\":false,\"quality\":false}\n",
           positions, HIDDEN, Q_ROWS, KV_ROWS, CHECKPOINT_REV);
    free(values); free(keys); free(qg); free(hidden);
    if (munmap((void *)mapping, (size_t)FILE_BYTES)) fail("cannot unmap safetensors file");
    return 0;
}
