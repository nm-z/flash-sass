#include "../../kern/model_oracle.h"

#include <openssl/evp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int digest_bytes(const void *bytes, size_t byte_count, uint8_t digest[MODEL_ORACLE_SHA256_BYTES]) {
    unsigned int digest_len = 0;
    return EVP_Digest(bytes, byte_count, digest, &digest_len, EVP_sha256(), NULL) == 1 &&
           digest_len == MODEL_ORACLE_SHA256_BYTES;
}

static int compare_case(const char *name, const model_oracle_record_t *record,
                        const model_oracle_observation_t *observations, size_t observation_count,
                        int expected_result, const char *expected_json) {
    FILE *result = tmpfile();
    char error[256] = {0};
    char output[256] = {0};
    int actual, ok = 1;
    if (!result) {
        fprintf(stderr, "%s: tmpfile failed\n", name);
        return -1;
    }
    actual = model_oracle_compare_record(record, observations, observation_count,
                                         result, error, sizeof error);
    if (fflush(result) == EOF || fseek(result, 0, SEEK_SET) != 0) ok = 0;
    size_t got = fread(output, 1, sizeof output - 1, result);
    output[got] = '\0';
    if (expected_result < 0) {
        if (actual >= 0 || strstr(output, "scalar_match:true")) ok = 0;
    } else {
        if (actual != expected_result || !strstr(output, expected_json)) ok = 0;
    }
    if (fclose(result) != 0) ok = 0;
    if (ok) printf("PASS %s\n", name);
    else fprintf(stderr, "FAIL %s (result=%d, error=%s, output=%s)\n",
                 name, actual, error, output);
    return ok ? 0 : -1;
}

static void initialize_item(model_oracle_item_t *item, const char *label,
                            uint32_t value_type, uint64_t element_count,
                            uint64_t byte_count, const uint8_t *raw_bytes,
                            const uint8_t *digest) {
    memset(item, 0, sizeof *item);
    snprintf(item->label, sizeof item->label, "%s", label);
    item->layer = 0;
    item->logical_die = 0;
    item->physical_die = 3;
    item->value_type = value_type;
    item->rank = 1;
    item->dimensions[0] = element_count;
    item->byte_count = byte_count;
    memcpy(item->sha256, digest, MODEL_ORACLE_SHA256_BYTES);
    item->raw_bytes = raw_bytes;
    item->raw_byte_count = raw_bytes ? byte_count : 0;
}

static void initialize_observation(model_oracle_observation_t *observation,
                                   const char *label, uint32_t value_type,
                                   uint64_t element_count, const void *bytes,
                                   uint64_t byte_count) {
    memset(observation, 0, sizeof *observation);
    observation->label = label;
    observation->value_type = value_type;
    observation->rank = 1;
    observation->dimensions[0] = element_count;
    observation->bytes = bytes;
    observation->byte_count = byte_count;
}

int main(void) {
    static const uint8_t selected_bytes[8] = {1, 0, 0, 0, 2, 0, 0, 0};
    static const uint8_t scalar_bytes[4] = {0, 0, 128, 63};
    static const uint8_t changed_scalar_bytes[4] = {0, 0, 129, 63};
    model_oracle_item_t items[2];
    model_oracle_observation_t observations[3];
    model_oracle_record_t record;
    uint8_t selected_digest[MODEL_ORACLE_SHA256_BYTES];
    uint8_t scalar_digest[MODEL_ORACLE_SHA256_BYTES];
    int failed = 0;
    if (!digest_bytes(selected_bytes, sizeof selected_bytes, selected_digest) ||
        !digest_bytes(scalar_bytes, sizeof scalar_bytes, scalar_digest)) {
        fprintf(stderr, "failed to create test digests\n");
        return 1;
    }
    initialize_item(&items[0], "b0.selected_ids", MODEL_ORACLE_U32, 2,
                    sizeof selected_bytes, selected_bytes, selected_digest);
    initialize_item(&items[1], "b0.S.d0", MODEL_ORACLE_F32, 1,
                    sizeof scalar_bytes, NULL, scalar_digest);
    memset(&record, 0, sizeof record);
    record.position = 4;
    record.visible_length = 5;
    record.item_count = 2;
    record.items = items;

    /* Observation order is reversed to verify label-based matching. */
    initialize_observation(&observations[0], "b0.S.d0", MODEL_ORACLE_F32, 1,
                           scalar_bytes, sizeof scalar_bytes);
    initialize_observation(&observations[1], "b0.selected_ids", MODEL_ORACLE_U32, 2,
                           selected_bytes, sizeof selected_bytes);
    failed += compare_case("valid raw and digest comparison", &record,
                           observations, 2, 0, "\"scalar_match\":true");

    observations[0].bytes = changed_scalar_bytes;
    failed += compare_case("one-bit mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].bytes = scalar_bytes;

    observations[0].byte_count = 3;
    failed += compare_case("size mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].byte_count = sizeof scalar_bytes;

    observations[0].value_type = MODEL_ORACLE_U32;
    failed += compare_case("type mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].value_type = MODEL_ORACLE_F32;

    observations[0].dimensions[0] = 2;
    failed += compare_case("shape mismatch", &record, observations, 2,
                           1, "\"scalar_match\":false");
    observations[0].dimensions[0] = 1;

    failed += compare_case("missing label", &record, &observations[1], 1,
                           1, "\"scalar_match\":false");

    model_oracle_observation_t duplicate[2] = {observations[0], observations[0]};
    failed += compare_case("duplicate label", &record, duplicate, 2,
                           1, "\"scalar_match\":false");

    initialize_observation(&observations[2], "extra.label", MODEL_ORACLE_F32, 1,
                           scalar_bytes, sizeof scalar_bytes);
    failed += compare_case("extra label", &record, observations, 3,
                           1, "\"scalar_match\":false");

    model_oracle_record_t empty = record;
    empty.item_count = 0;
    empty.items = NULL;
    failed += compare_case("empty record rejected", &empty, NULL, 0, -1, NULL);

    if (failed) return 1;
    puts("oracle API checks passed");
    return 0;
}
