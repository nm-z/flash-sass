/* GPU observations for the CUDA-free scalar oracle. Included by model_test
 * after its ordinary comparison helpers and production QSA descriptors. */
#ifndef FLASH_MODEL_ORACLE_GPU_H
#define FLASH_MODEL_ORACLE_GPU_H
#include "model_oracle.h"
#include <limits.h>
#include <math.h>

_Static_assert(sizeof(float) == 4 && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
               "model oracle GPU comparison requires little-endian F32");

static model_oracle_reader_t *oracle_gpu_reader;
static FILE *oracle_gpu_file;
static model_oracle_record_t oracle_gpu_record;
static uint32_t *oracle_gpu_predictions;

static const char *oracle_gpu_environment(const char *name) {
    const char *value = getenv(name);
    if (!value || !*value) fprintf(stderr, "GPU oracle requires %s\n", name);
    return value && *value ? value : NULL;
}

static int oracle_gpu_csv(const char *name, uint32_t *values, size_t count, unsigned maximum,
                          int positive) {
    const char *input = oracle_gpu_environment(name);
    if (!input || input[0] == ',' || input[strlen(input) - 1] == ',' || strstr(input, ",,")) return -1;
    char *copy = strdup(input), *save = NULL;
    if (!copy) return -1;
    size_t used = 0;
    int result = 0;
    for (char *word = strtok_r(copy, ",", &save); word; word = strtok_r(NULL, ",", &save)) {
        int value = decimal_arg(word, maximum, name);
        if (value < 0 || (positive && !value) || used == count) { result = -1; break; }
        values[used++] = (uint32_t)value;
    }
    if (used != count) result = -1;
    free(copy);
    return result;
}

static void oracle_gpu_close(void) {
    if (oracle_gpu_reader) model_oracle_reader_close(oracle_gpu_reader);
    oracle_gpu_reader = NULL;
    if (oracle_gpu_file) fclose(oracle_gpu_file);
    oracle_gpu_file = NULL;
    free(oracle_gpu_predictions);
    oracle_gpu_predictions = NULL;
}

static int oracle_gpu_open(const uint32_t *ids, unsigned tokens, float epsilon,
                           const int physical_dies[MAXD]) {
    model_oracle_config_t config = {0};
    model_oracle_identity_t identity;
    char error[512] = {0};
    const char *records = oracle_gpu_environment("MODEL_ORACLE_RECORDS");
    const char *receipt_path = oracle_gpu_environment("MODEL_ORACLE_RECEIPT");
    config.source_root = oracle_gpu_environment("MODEL_ORACLE_SOURCE_ROOT");
    config.source_sha256_manifest = oracle_gpu_environment("MODEL_ORACLE_SOURCE_MANIFEST");
    config.source_revision = oracle_gpu_environment("MODEL_ORACLE_SOURCE_REVISION");
    config.tile_cache_dir = fullcache;
    config.tile_cache_manifest = oracle_gpu_environment("MODEL_ORACLE_TILE_MANIFEST");
    config.tile_cache_sha256_manifest = oracle_gpu_environment("MODEL_ORACLE_TILE_HASHES");
    config.checkpoint_sha256_manifest = oracle_gpu_environment("MODEL_ORACLE_CHECKPOINT_HASHES");
    config.primary_checkpoint_path = oracle_gpu_environment("MODEL_ORACLE_PRIMARY");
    config.ple_shard_path = oracle_gpu_environment("MODEL_PLE");
    const char *offset = oracle_gpu_environment("MODEL_PLE_OFF");
    if (!records || !receipt_path || !config.source_root || !config.source_sha256_manifest ||
        !config.source_revision || !config.tile_cache_manifest || !config.tile_cache_sha256_manifest ||
        !config.checkpoint_sha256_manifest || !config.primary_checkpoint_path || !config.ple_shard_path || !offset)
        return -1;
    int parsed_offset = decimal_arg(offset, INT_MAX, "MODEL_PLE_OFF");
    uint32_t attention_map[MODEL_ORACLE_ATTN_LAYERS];
    if (parsed_offset < 0 || oracle_gpu_csv("SPLIT_SHARES", config.shares, MAXD, INT_MAX, 1) ||
        oracle_gpu_csv("ATTN_MAP", attention_map, MODEL_ORACLE_ATTN_LAYERS, MAXD - 1, 0)) {
        fprintf(stderr, "invalid GPU oracle offset or explicit topology\n"); return -1;
    }
    config.ple_data_offset = (uint64_t)parsed_offset;
    config.teacher_ids = ids;
    config.token_count = tokens;
    config.flash_nblk = model_budget;
    config.epsilon = epsilon;
    for (int die = 0; die < MAXD; die++) config.physical_dies[die] = (uint32_t)physical_dies[die];
    for (unsigned layer = 0; layer < MODEL_ORACLE_ATTN_LAYERS; layer++)
        config.attention_map[layer] = (int32_t)attention_map[layer];
    FILE *receipt = fopen(receipt_path, "rb");
    if (!receipt) { perror(receipt_path); return -1; }
    int verified = model_oracle_verify_inputs_from_receipt(&config, receipt, &identity, error, sizeof error);
    fclose(receipt);
    if (verified) { fprintf(stderr, "GPU oracle receipt: %s\n", error); return -1; }
    oracle_gpu_file = fopen(records, "rb");
    if (!oracle_gpu_file) { perror(records); return -1; }
    if (model_oracle_reader_open(&oracle_gpu_reader, oracle_gpu_file, &identity, error, sizeof error)) goto invalid;
    oracle_gpu_predictions = calloc(tokens, sizeof *oracle_gpu_predictions);
    if (!oracle_gpu_predictions) { snprintf(error, sizeof error, "prediction allocation failed"); goto invalid; }
    /* Validate complete framing and IDs before any CUDA initialization. A
     * second reader pass retains each live record through its GPU comparison. */
    for (unsigned position = 0; position < tokens; position++) {
        if (model_oracle_reader_next(oracle_gpu_reader, &oracle_gpu_record, error, sizeof error) != 1) goto invalid;
        if (oracle_gpu_record.position != position || oracle_gpu_record.teacher_id != ids[position]) {
            snprintf(error, sizeof error, "teacher ID/position mismatch at %u", position); goto invalid;
        }
        oracle_gpu_predictions[position] = oracle_gpu_record.prediction;
        const model_oracle_item_t *prediction = NULL;
        for (uint32_t item = 0; item < oracle_gpu_record.item_count; item++)
            if (!strcmp(oracle_gpu_record.items[item].label, "tok.0")) prediction = &oracle_gpu_record.items[item];
        uint32_t tuple[2];
        if (!prediction || prediction->raw_byte_count != sizeof tuple || !prediction->raw_bytes) {
            snprintf(error, sizeof error, "missing raw prediction tuple at %u", position); goto invalid;
        }
        memcpy(tuple, prediction->raw_bytes, sizeof tuple);
        if (tuple[0] != oracle_gpu_record.prediction || tuple[1] != oracle_gpu_record.prediction_score_bits) {
            snprintf(error, sizeof error, "prediction metadata mismatch at %u", position); goto invalid;
        }
    }
    if (model_oracle_reader_next(oracle_gpu_reader, &oracle_gpu_record, error, sizeof error) != 0) goto invalid;
    model_oracle_reader_close(oracle_gpu_reader);
    oracle_gpu_reader = NULL;
    if (model_oracle_reader_open(&oracle_gpu_reader, oracle_gpu_file, &identity, error, sizeof error)) goto invalid;
    printf("CPU oracle: %u complete teacher-forced records; input seal checked without large-file hashing\n", tokens);
    return 0;
invalid:
    fprintf(stderr, "invalid GPU oracle records: %s\n", error);
    oracle_gpu_close();
    return -1;
}

static int oracle_gpu_next(unsigned position, uint32_t teacher_id) {
    char error[512] = {0};
    if (model_oracle_reader_next(oracle_gpu_reader, &oracle_gpu_record, error, sizeof error) != 1 ||
        oracle_gpu_record.position != position || oracle_gpu_record.teacher_id != teacher_id) {
        fprintf(stderr, "GPU oracle position %u: %s\n", position, error);
        return -1;
    }
    return 0;
}

static int oracle_gpu_finished(void) {
    char error[512] = {0};
    if (model_oracle_reader_next(oracle_gpu_reader, &oracle_gpu_record, error, sizeof error) != 0) {
        fprintf(stderr, "GPU oracle completion: %s\n", error);
        return -1;
    }
    return 0;
}

typedef struct {
    model_oracle_observation_t *items;
    size_t count, capacity;
    int nonfinite;
} model_oracle_gpu_frame;

static void oracle_gpu_take(model_oracle_gpu_frame *frame, const char *label,
                            void *bytes, size_t size, unsigned type,
                            unsigned rank, const uint64_t *dimensions,
                            const char *control) {
    if (frame->count == frame->capacity) {
        size_t capacity = frame->capacity ? frame->capacity * 2 : 128;
        void *items = realloc(frame->items, capacity * sizeof *frame->items);
        if (!items) { fprintf(stderr, "oracle observation allocation failed\n"); exit(1); }
        frame->items = items;
        frame->capacity = capacity;
    }
    model_oracle_observation_t *item = &frame->items[frame->count++];
    memset(item, 0, sizeof *item);
    item->label = strdup(label);
    if (!item->label) { fprintf(stderr, "oracle label allocation failed\n"); exit(1); }
    item->bytes = bytes;
    item->byte_count = size;
    item->value_type = type;
    item->rank = rank;
    for (unsigned i = 0; i < rank; i++) item->dimensions[i] = dimensions[i];
    if (control && (!strcmp(control, "layer") || !strcmp(control, "logits"))) {
        const float *values = bytes;
        for (size_t i = 0; i < size / sizeof(float); i++) if (!isfinite(values[i])) {
            fprintf(stderr, "%s: non-finite model output at %zu\n", label, i);
            frame->nonfinite = 1;
            break;
        }
    }
    if (size && control && model_inject && !model_inject_reached &&
        !strcmp(model_inject, control)) {
        ((unsigned char *)bytes)[0] ^= 1;
        model_inject_reached = 1;
        printf("MODEL_INJECT=%s reached at %s\n", control, label);
    }
}

static void oracle_gpu_read(model_oracle_gpu_frame *frame, const char *label,
                            CUdeviceptr device, size_t size, unsigned type,
                            const char *control) {
    void *bytes = malloc(size);
    if (!bytes) { fprintf(stderr, "oracle GPU readback allocation failed\n"); exit(1); }
    CK(cuMemcpyDtoH(bytes, device, size));
    size_t width = type == MODEL_ORACLE_I8 ? 1 : type == MODEL_ORACLE_F16 ? 2 : 4;
    uint64_t dimensions[1] = { size / width };
    oracle_gpu_take(frame, label, bytes, size, type, 1, dimensions, control);
}

static void oracle_gpu_qsa(model_oracle_gpu_frame *frame, int die, int layer,
                           unsigned position) {
    dn_bufs_t *q = &model_qdev[die][layer];
    unsigned blocks = (position + 1) / 4, keys = qsa_nkeys(position, model_budget);
    char label[MODEL_ORACLE_LABEL_BYTES];
#define ORACLE_QSA(suffix, device, size, type) do { \
    snprintf(label, sizeof label, "b%d.%s", layer, suffix); \
    oracle_gpu_read(frame, label, device, size, type, NULL); \
} while (0)
    ORACLE_QSA("kq", q->kq + (size_t)position * 512, 512, MODEL_ORACLE_I8);
    ORACLE_QSA("vq", q->vq + (size_t)position * 512, 512, MODEL_ORACLE_I8);
    ORACLE_QSA("ks", q->ks + (size_t)position * 32, 32, MODEL_ORACLE_F16);
    ORACLE_QSA("vs", q->vs + (size_t)position * 32, 32, MODEL_ORACLE_F16);
    ORACLE_QSA("iqk", q->iqk, 640 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("iq", q->iq, 512 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("ring", q->ring, 128 * ((position & 3) + 1) * sizeof(float), MODEL_ORACLE_F32);
    if ((position & 3) == 3) {
        unsigned block = position / 4;
        uint16_t tile[4096], *row = malloc(128 * sizeof *row);
        if (!row) { fprintf(stderr, "oracle pool allocation failed\n"); exit(1); }
        CK(cuMemcpyDtoH(tile, q->pkt + (size_t)(block >> 5) * sizeof tile, sizeof tile));
        for (unsigned d = 0; d < 128; d++)
            row[d] = tile[(d >> 2) * 128 + (block & 31) * 4 + (d & 3)];
        uint64_t dimensions[1] = {128};
        snprintf(label, sizeof label, "b%d.pool", layer);
        oracle_gpu_take(frame, label, row, 128 * sizeof *row, MODEL_ORACLE_F16,
                        1, dimensions, NULL);
    }
    if (blocks > model_budget) {
        ORACLE_QSA("scores", q->sc, (size_t)blocks * sizeof(float), MODEL_ORACLE_F32);
        snprintf(label, sizeof label, "b%d.selected_ids", layer);
        oracle_gpu_read(frame, label, q->sel, (size_t)model_budget * sizeof(uint32_t),
                        MODEL_ORACLE_U32, "selection");
    }
    size_t size = (size_t)keys * 512 * sizeof(float);
    float *packed = malloc(model_kcf * sizeof(float)), *canonical = malloc(size);
    if (!packed || !canonical) { fprintf(stderr, "oracle gather allocation failed\n"); exit(1); }
    CK(cuMemcpyDtoH(packed, q->kc, model_kcf * sizeof(float)));
    for (unsigned slot = 0; slot < keys; slot++)
        for (unsigned kv = 0; kv < 2; kv++)
            for (unsigned d = 0; d < 256; d++)
                canonical[((size_t)slot * 2 + kv) * 256 + d] = packed[attn_kidx(slot, kv, d)];
    free(packed);
    uint64_t dimensions[3] = {keys, 2, 256};
    snprintf(label, sizeof label, "b%d.gather_k", layer);
    oracle_gpu_take(frame, label, canonical, size, MODEL_ORACLE_F32, 3, dimensions, NULL);
    canonical = malloc(size);
    if (!canonical) { fprintf(stderr, "oracle V gather allocation failed\n"); exit(1); }
    CK(cuMemcpyDtoH(canonical, q->vc, size));
    snprintf(label, sizeof label, "b%d.gather_v", layer);
    oracle_gpu_take(frame, label, canonical, size, MODEL_ORACLE_F32, 3, dimensions, NULL);
    ORACLE_QSA("qo", q->qo, 6144 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("pm", q->pm, 192 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("pd", q->pd, 192 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("po", q->po, 49152 * sizeof(float), MODEL_ORACLE_F32);
    ORACLE_QSA("aout", q->aout, 6144 * sizeof(float), MODEL_ORACLE_F32);
#undef ORACLE_QSA
}

static int model_oracle_gpu_compare(const model_oracle_record_t *record, int full_scope,
                                    const dn_bufs_t db[MAXD],
                                    split_t split[MAXD][MAXB],
                                    CUdeviceptr dn[MAXD][MAXB],
                                    CUdeviceptr conv[MAXD][MAXB],
                                    const CUdeviceptr ple[MAXD]) {
    if (!full_scope || nd != MAXD || !record || record->position >= 262144 ||
        record->visible_length != record->position + 1) {
        fprintf(stderr, "GPU oracle comparison requires a valid full-model record and six dies\n");
        return -1;
    }
    model_oracle_gpu_frame frame = {0};
    char label[MODEL_ORACLE_LABEL_BYTES], error[512] = {0};
    for (int die = 0; die < nd; die++) {
        CK(cuCtxSetCurrent(ctx[die]));
        for (int layer = 0; layer < MAXB; layer++) {
            snprintf(label, sizeof label, "b%d.S.d%d", layer, die);
            oracle_gpu_read(&frame, label, model_trace[die][layer], 10240 * sizeof(float),
                            MODEL_ORACLE_F32, "layer");
            if (BLK_IS_ATTN(layer)) {
                if (die == attn_die_of(layer)) oracle_gpu_qsa(&frame, die, layer, (unsigned)record->position);
            } else {
                snprintf(label, sizeof label, "b%d.dn_state.d%d", layer, die);
                oracle_gpu_read(&frame, label, dn[die][layer],
                                (size_t)split[die][layer].hn * 16384 * sizeof(float),
                                MODEL_ORACLE_F32, "dn");
                snprintf(label, sizeof label, "b%d.conv_state.d%d", layer, die);
                oracle_gpu_read(&frame, label, conv[die][layer],
                                (size_t)4 * SPLIT_NLOC(&split[die][layer]) * sizeof(float),
                                MODEL_ORACLE_F32, "conv");
            }
            if (layer == 1) {
                snprintf(label, sizeof label, "b1.ple_state.d%d", die);
                oracle_gpu_read(&frame, label, ple[die], 10 * 10240 * sizeof(float),
                                MODEL_ORACLE_F32, "ple");
            }
        }
        snprintf(label, sizeof label, "logits.d%d", die);
        oracle_gpu_read(&frame, label, db[die].hlo, (size_t)split[die][0].on * sizeof(float),
                        MODEL_ORACLE_F32, "logits");
    }
    for (int die = 0; die < nd; die++) {
        CK(cuCtxSetCurrent(ctx[die]));
        snprintf(label, sizeof label, "S.%d", die);
        oracle_gpu_read(&frame, label, db[die].S, 10240 * sizeof(float), MODEL_ORACLE_F32, "layer");
        snprintf(label, sizeof label, "tok.%d", die);
        oracle_gpu_read(&frame, label, db[die].tokout, 2 * sizeof(uint32_t), MODEL_ORACLE_U32, NULL);
    }
    int result;
    if (frame.nonfinite) {
        printf("{\"event\":\"qsa_oracle_gpu_compare\",\"position\":%llu,\"scalar_match\":false,\"reason\":\"nonfinite_model_output\"}\n",
               (unsigned long long)record->position);
        result = 1;
    } else {
        result = model_oracle_compare_record(record, frame.items, frame.count, stdout, error, sizeof error);
        if (result) fprintf(stderr, "oracle comparison at position %llu: %s\n",
                            (unsigned long long)record->position, error);
    }
    for (size_t i = 0; i < frame.count; i++) {
        free((void *)frame.items[i].label);
        free((void *)frame.items[i].bytes);
    }
    free(frame.items);
    return result;
}
#endif
