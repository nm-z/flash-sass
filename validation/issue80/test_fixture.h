#ifndef FLASH_SASS_ISSUE80_TEST_FIXTURE_H
#define FLASH_SASS_ISSUE80_TEST_FIXTURE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    unsigned char bytes[1024 * 1024];
    size_t length;
} issue80_bytes;

static inline void issue80_append(issue80_bytes *out, const void *data, size_t length) {
    if (out->length + length > sizeof out->bytes) return;
    memcpy(out->bytes + out->length, data, length);
    out->length += length;
}

static inline void issue80_u32(issue80_bytes *out, uint32_t value) {
    unsigned char bytes[4] = {
        (unsigned char)value, (unsigned char)(value >> 8),
        (unsigned char)(value >> 16), (unsigned char)(value >> 24),
    };
    issue80_append(out, bytes, sizeof bytes);
}

static inline void issue80_u64(issue80_bytes *out, uint64_t value) {
    unsigned char bytes[8];
    for (unsigned i = 0; i < 8; ++i) bytes[i] = (unsigned char)(value >> (i * 8));
    issue80_append(out, bytes, sizeof bytes);
}

static inline void issue80_string(issue80_bytes *out, const char *value) {
    size_t length = strlen(value);
    issue80_u64(out, length);
    issue80_append(out, value, length);
}

static inline void issue80_kv_string(issue80_bytes *out, const char *key, const char *value) {
    issue80_string(out, key);
    issue80_u32(out, 8);
    issue80_string(out, value);
}

static inline void issue80_kv_array_strings(issue80_bytes *out, const char *key,
                                             const char *const *values, uint64_t count) {
    issue80_string(out, key);
    issue80_u32(out, 9);
    issue80_u32(out, 8);
    issue80_u64(out, count);
    for (uint64_t i = 0; i < count; ++i) issue80_string(out, values[i]);
}

static inline void issue80_kv_token_types(issue80_bytes *out, const int32_t *values,
                                           uint64_t count) {
    issue80_string(out, "tokenizer.ggml.token_type");
    issue80_u32(out, 9);
    issue80_u32(out, 5);
    issue80_u64(out, count);
    for (uint64_t i = 0; i < count; ++i) issue80_u32(out, (uint32_t)values[i]);
}

static inline void issue80_kv_empty_string_array(issue80_bytes *out, const char *key) {
    issue80_string(out, key);
    issue80_u32(out, 9);
    issue80_u32(out, 8);
    issue80_u64(out, 0);
}

static inline int issue80_write_fixture(FILE *file) {
    const char *tokens[28] = {"<|im_end|>", "</think>"};
    char letters[26][2];
    for (int i = 0; i < 26; ++i) {
        letters[i][0] = (char)('a' + i);
        letters[i][1] = 0;
        tokens[i + 2] = letters[i];
    }
    int32_t token_types[28] = {3, 3};
    for (int i = 2; i < 28; ++i) token_types[i] = 1;

    issue80_bytes metadata = {0};
    issue80_kv_string(&metadata, "tokenizer.ggml.model", "gpt2");
    issue80_kv_string(&metadata, "tokenizer.ggml.pre", "qwen35");
    issue80_kv_array_strings(&metadata, "tokenizer.ggml.tokens", tokens, 28);
    issue80_kv_token_types(&metadata, token_types, 28);
    issue80_kv_empty_string_array(&metadata, "tokenizer.ggml.merges");
    issue80_string(&metadata, "tokenizer.ggml.eos_token_id");
    issue80_u32(&metadata, 4);
    issue80_u32(&metadata, 0);
    issue80_string(&metadata, "tokenizer.ggml.add_bos_token");
    issue80_u32(&metadata, 7);
    unsigned char no_bos = 0;
    issue80_append(&metadata, &no_bos, 1);
    issue80_kv_string(&metadata, "tokenizer.chat_template", "{{ messages[0].content }}");

    issue80_bytes fixture = {0};
    issue80_append(&fixture, "GGUF", 4);
    issue80_u32(&fixture, 3);
    issue80_u64(&fixture, 0);
    issue80_u64(&fixture, 8);
    issue80_append(&fixture, metadata.bytes, metadata.length);
    while (fixture.length % 32) {
        unsigned char padding = 0;
        issue80_append(&fixture, &padding, 1);
    }
    return fwrite(fixture.bytes, 1, fixture.length, file) == fixture.length ? 0 : -1;
}

static inline int issue80_write_acceptance_fixture(FILE *file) {
    enum { token_count = 9708 };
    const char *fixed_tokens[28] = {"<|im_end|>", "</think>"};
    char letters[26][2];
    char (*generated)[32] = calloc(token_count, sizeof *generated);
    const char **tokens = calloc(token_count, sizeof *tokens);
    int32_t *token_types = calloc(token_count, sizeof *token_types);
    if (!generated || !tokens || !token_types) {
        free(generated); free(tokens); free(token_types);
        return -1;
    }
    tokens[0] = fixed_tokens[0];
    tokens[1] = fixed_tokens[1];
    for (int i = 0; i < 26; ++i) {
        letters[i][0] = (char)('a' + i);
        letters[i][1] = 0;
        tokens[i + 2] = letters[i];
    }
    for (int i = 28; i < token_count; ++i) {
        snprintf(generated[i], sizeof generated[i], "token_%d", i);
        tokens[i] = generated[i];
    }
    tokens[9707] = "a";
    for (int i = 0; i < token_count; ++i) token_types[i] = i < 2 ? 3 : 1;

    issue80_bytes metadata = {0};
    issue80_kv_string(&metadata, "tokenizer.ggml.model", "gpt2");
    issue80_kv_string(&metadata, "tokenizer.ggml.pre", "qwen35");
    issue80_kv_array_strings(&metadata, "tokenizer.ggml.tokens", tokens, token_count);
    issue80_kv_token_types(&metadata, token_types, token_count);
    issue80_kv_empty_string_array(&metadata, "tokenizer.ggml.merges");
    issue80_string(&metadata, "tokenizer.ggml.eos_token_id");
    issue80_u32(&metadata, 4);
    issue80_u32(&metadata, 0);
    issue80_string(&metadata, "tokenizer.ggml.add_bos_token");
    issue80_u32(&metadata, 7);
    unsigned char no_bos = 0;
    issue80_append(&metadata, &no_bos, 1);
    issue80_kv_string(&metadata, "tokenizer.chat_template", "{{ messages[0].content }}");

    issue80_bytes fixture = {0};
    issue80_append(&fixture, "GGUF", 4);
    issue80_u32(&fixture, 3);
    issue80_u64(&fixture, 0);
    issue80_u64(&fixture, 8);
    issue80_append(&fixture, metadata.bytes, metadata.length);
    while (fixture.length % 32) {
        unsigned char padding = 0;
        issue80_append(&fixture, &padding, 1);
    }
    int result = fwrite(fixture.bytes, 1, fixture.length, file) == fixture.length ? 0 : -1;
    free(generated); free(tokens); free(token_types);
    return result;
}

#endif
