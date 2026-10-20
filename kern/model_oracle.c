#ifndef _XOPEN_SOURCE
#define _XOPEN_SOURCE 700
#endif
#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif
#include "model_oracle.h"
#include "../ref/attn.h"
#include "../ref/block.h"
#include "../ref/gguf.h"
#include "../ref/ple.h"
#include "../ref/qsa.h"
#include "../ref/split.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <openssl/evp.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define ORACLE_MAGIC "QSAORC1\0"
#define ORACLE_SEAL_MAGIC "QSASEAL1"
#define ORACLE_CHECKPOINT_MAGIC "QSACKP1\0"
#define ORACLE_RECORD_MAGIC "POS1"
#define ORACLE_MAX_TENSORS 8192u
#define ORACLE_MAX_SOURCE_FILES 4096u
#define ORACLE_MAX_ITEMS 1024u
#define ORACLE_IO_BYTES (1u << 20)
#define ORACLE_CACHE_HEADER "cache-only tile export; NOT A CUDA IPC MANIFEST"

typedef struct {
    char name[MODEL_ORACLE_LABEL_BYTES * 2];
    size_t bytes;
    uint32_t type;
    int rows, k, stride, experts;
    void *mapping;
    size_t mapping_bytes;
} oracle_tensor_t;

typedef struct {
    float *dn_state;
    float *conv_state;
    float *ple_state;
} oracle_layer_state_t;

typedef struct {
    model_oracle_item_t item;
    uint8_t *raw;
} owned_item_t;

struct model_oracle {
    model_oracle_config_t config;
    model_oracle_identity_t identity;
    char *owned_strings[9];
    uint32_t *ids;
    FILE *records;
    uint64_t next_position;
    uint64_t record_offset;
    uint32_t max_cache_positions;
    size_t compact_positions;
    size_t kcache_floats;
    size_t vcache_floats;
    oracle_tensor_t *tensors;
    size_t tensor_count;
    dn_hweights_t weights[MODEL_ORACLE_BLOCKS];
    dn_hbufs_t *buffers;
    oracle_layer_state_t state[MODEL_ORACLE_BLOCKS];
    model_qsa_t *qsa[MODEL_ORACLE_BLOCKS];
    uint8_t *ple_mapping;
    size_t ple_mapping_bytes;
    size_t ple_delta;
    const uint8_t *ple_table;
    float *ple_conv_weights;
};

struct model_oracle_reader {
    FILE *stream;
    model_oracle_identity_t identity;
    uint64_t next_position;
    model_oracle_item_t *items;
    uint8_t **raw_allocations;
    uint32_t allocated_items;
};

typedef struct {
    owned_item_t *items;
    uint32_t count;
    uint32_t capacity;
} record_builder_t;

typedef struct {
    char *path;
    uint8_t digest[MODEL_ORACLE_SHA256_BYTES];
} hash_line_t;

typedef struct {
    hash_line_t *items;
    size_t count;
    size_t capacity;
} hash_manifest_t;

typedef struct {
    char name[MODEL_ORACLE_LABEL_BYTES * 2];
    uint64_t offset;
    uint64_t bytes;
    uint32_t type;
    int rows, k, stride, experts;
    uint8_t digest[MODEL_ORACLE_SHA256_BYTES];
    int digest_found;
} tile_desc_t;

typedef struct {
    char *key;
    uint64_t size;
    uint64_t mtime_ns;
} seal_meta_t;

typedef struct {
    seal_meta_t *items;
    size_t count;
    size_t capacity;
} seal_meta_vector_t;

static int oracle_error(char *dst, size_t cap, const char *fmt, ...);
static uint32_t host_u32_bits(float f);
static uint64_t stat_mtime_ns(const struct stat *st);
static int hash_begin(EVP_MD_CTX **ctx, char *err, size_t errcap);
static int hash_bytes(EVP_MD_CTX *ctx, const void *data, size_t bytes, char *err, size_t errcap);
static int hash_finish(EVP_MD_CTX *ctx, uint8_t out[MODEL_ORACLE_SHA256_BYTES], char *err, size_t errcap);
static int sha256_memory(const void *data, size_t bytes, uint8_t out[MODEL_ORACLE_SHA256_BYTES], char *err, size_t errcap);
static int sha256_file(const char *path, uint8_t out[MODEL_ORACLE_SHA256_BYTES], struct stat *before, char *err, size_t errcap);
static void put_u32(uint8_t out[4], uint32_t v);
static void put_u64(uint8_t out[8], uint64_t v);
static uint32_t get_u32(const uint8_t in[4]);
static uint64_t get_u64(const uint8_t in[8]);
static int write_all(FILE *f, const void *data, size_t bytes, char *err, size_t errcap);
static int read_all(FILE *f, void *data, size_t bytes, char *err, size_t errcap);
static int write_u32(FILE *f, uint32_t v, char *err, size_t errcap);
static int write_u64(FILE *f, uint64_t v, char *err, size_t errcap);
static int read_u32(FILE *f, uint32_t *v, char *err, size_t errcap);
static int read_u64(FILE *f, uint64_t *v, char *err, size_t errcap);
static int identity_write(FILE *f, const model_oracle_identity_t *id, char *err, size_t errcap);
static int identity_read(FILE *f, model_oracle_identity_t *id, char *err, size_t errcap);
static int identity_equal(const model_oracle_identity_t *a, const model_oracle_identity_t *b);

static void hash_manifest_clear(hash_manifest_t *m) {
    for (size_t i = 0; i < m->count; i++) free(m->items[i].path);
    free(m->items); memset(m, 0, sizeof *m);
}

static int hex_nibble(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static int parse_sha256_hex(const char *s, uint8_t digest[MODEL_ORACLE_SHA256_BYTES]) {
    for (unsigned i = 0; i < MODEL_ORACLE_SHA256_BYTES; i++) {
        int hi = hex_nibble(s[i * 2]), lo = hex_nibble(s[i * 2 + 1]);
        if (hi < 0 || lo < 0) return -1;
        digest[i] = (uint8_t)((hi << 4) | lo);
    }
    return 0;
}

static int hash_manifest_push(hash_manifest_t *m, const char *path, const uint8_t digest[32], char *err, size_t errcap) {
    if (m->count == m->capacity) {
        size_t next = m->capacity ? m->capacity * 2 : 32;
        hash_line_t *p = realloc(m->items, next * sizeof *p);
        if (!p) return oracle_error(err, errcap, "allocate hash manifest entries");
        m->items = p; m->capacity = next;
    }
    m->items[m->count].path = strdup(path);
    if (!m->items[m->count].path) return oracle_error(err, errcap, "copy hash manifest path");
    memcpy(m->items[m->count].digest, digest, 32);
    m->count++;
    return 0;
}

static int read_hash_manifest(const char *path, hash_manifest_t *out, char *err, size_t errcap) {
    FILE *f = fopen(path, "r");
    char *line = NULL;
    size_t cap = 0;
    ssize_t n;
    int rc = -1;
    memset(out, 0, sizeof *out);
    if (!f) return oracle_error(err, errcap, "open hash manifest %s: %s", path, strerror(errno));
    while ((n = getline(&line, &cap, f)) >= 0) {
        char *p = line, *name;
        uint8_t digest[32];
        if (n >= 65 && p[0] == '\\') p++;
        if (strlen(p) < 65 || parse_sha256_hex(p, digest) < 0) {
            oracle_error(err, errcap, "malformed SHA-256 line in %s", path); goto done;
        }
        p += 64;
        if (*p != ' ' && *p != '\t') { oracle_error(err, errcap, "missing filename in %s", path); goto done; }
        while (*p == ' ' || *p == '\t') p++;
        if (*p == '*') p++;
        name = p;
        while (*p && *p != '\n' && *p != '\r') p++;
        *p = '\0';
        if (!*name || hash_manifest_push(out, name, digest, err, errcap) < 0) goto done;
    }
    if (ferror(f)) { oracle_error(err, errcap, "read hash manifest %s: %s", path, strerror(errno)); goto done; }
    rc = 0;
done:
    free(line); fclose(f);
    if (rc < 0) hash_manifest_clear(out);
    return rc;
}

static const char *path_basename(const char *path) {
    const char *slash = strrchr(path, '/');
    return slash ? slash + 1 : path;
}

static const hash_line_t *hash_by_basename(const hash_manifest_t *m, const char *base) {
    const hash_line_t *found = NULL;
    for (size_t i = 0; i < m->count; i++) {
        if (!strcmp(path_basename(m->items[i].path), base)) {
            if (found) return NULL;
            found = &m->items[i];
        }
    }
    return found;
}

static int hash_update_u32(EVP_MD_CTX *ctx, uint32_t value, char *err, size_t errcap) {
    uint8_t b[4]; put_u32(b, value); return hash_bytes(ctx, b, sizeof b, err, errcap);
}

static int hash_update_u64(EVP_MD_CTX *ctx, uint64_t value, char *err, size_t errcap) {
    uint8_t b[8]; put_u64(b, value); return hash_bytes(ctx, b, sizeof b, err, errcap);
}

static int hash_update_string(EVP_MD_CTX *ctx, const char *s, char *err, size_t errcap) {
    size_t n = strlen(s);
    if (hash_update_u64(ctx, n, err, errcap) < 0) return -1;
    return hash_bytes(ctx, s, n, err, errcap);
}

static int path_join(char *dst, size_t cap, const char *root, const char *relative, char *err, size_t errcap) {
    int n;
    if (relative[0] == '/') {
        n = snprintf(dst, cap, "%s", relative);
    } else {
        n = snprintf(dst, cap, "%s/%s", root, relative);
    }
    if (n < 0 || (size_t)n >= cap) return oracle_error(err, errcap, "input path is too long");
    return 0;
}

static int canonical_source_key(const char *root, const char *path, char *key, size_t keycap, char *err, size_t errcap) {
    char *real_root = realpath(root, NULL);
    const char *relative = path;
    int n;
    if (!real_root) return oracle_error(err, errcap, "realpath source root %s: %s", root, strerror(errno));
    if (path[0] == '/') {
        size_t rootlen = strlen(real_root);
        if (strncmp(path, real_root, rootlen) || path[rootlen] != '/') {
            free(real_root); return oracle_error(err, errcap, "source manifest path is outside source_root");
        }
        relative = path + rootlen + 1;
    }
    while (!strncmp(relative, "./", 2)) relative += 2;
    n = snprintf(key, keycap, "%s", relative);
    free(real_root);
    if (n < 0 || (size_t)n >= keycap || !*key || strstr(key, "../"))
        return oracle_error(err, errcap, "invalid relative source identity path");
    return 0;
}

static int source_path(const char *root, const char *manifest_path, char *out, size_t cap, char *err, size_t errcap) {
    if (manifest_path[0] == '/') {
        int n = snprintf(out, cap, "%s", manifest_path);
        if (n < 0 || (size_t)n >= cap) return oracle_error(err, errcap, "source path is too long");
        return 0;
    }
    return path_join(out, cap, root, manifest_path, err, errcap);
}

static int meta_push(seal_meta_vector_t *v, const char *key, uint64_t size, uint64_t mtime_ns, char *err, size_t errcap) {
    if (v->count == v->capacity) {
        size_t next = v->capacity ? v->capacity * 2 : 64;
        seal_meta_t *p = realloc(v->items, next * sizeof *p);
        if (!p) return oracle_error(err, errcap, "allocate seal metadata");
        v->items = p; v->capacity = next;
    }
    v->items[v->count].key = strdup(key);
    if (!v->items[v->count].key) return oracle_error(err, errcap, "copy seal metadata key");
    v->items[v->count].size = size;
    v->items[v->count].mtime_ns = mtime_ns;
    v->count++;
    return 0;
}

static void meta_clear(seal_meta_vector_t *v) {
    for (size_t i = 0; i < v->count; i++) free(v->items[i].key);
    free(v->items); memset(v, 0, sizeof *v);
}

static const seal_meta_t *meta_find(const seal_meta_vector_t *v, const char *key) {
    for (size_t i = 0; i < v->count; i++) if (!strcmp(v->items[i].key, key)) return &v->items[i];
    return NULL;
}

static int stat_file_meta(const char *path, uint64_t *size, uint64_t *mtime_ns, char *err, size_t errcap) {
    struct stat st;
    if (stat(path, &st) < 0) return oracle_error(err, errcap, "stat %s: %s", path, strerror(errno));
    if (!S_ISREG(st.st_mode) || st.st_size < 0) return oracle_error(err, errcap, "%s is not a regular file", path);
    *size = (uint64_t)st.st_size;
    *mtime_ns = stat_mtime_ns(&st);
    return 0;
}

static int check_or_add_meta(seal_meta_vector_t *current, const seal_meta_vector_t *sealed,
                             const char *key, const char *path, char *err, size_t errcap) {
    uint64_t size, mtime;
    const seal_meta_t *old;
    if (stat_file_meta(path, &size, &mtime, err, errcap) < 0) return -1;
    old = sealed ? meta_find(sealed, key) : NULL;
    if (sealed && (!old || old->size != size || old->mtime_ns != mtime))
        return oracle_error(err, errcap, "sealed input metadata changed for %s", key);
    return meta_push(current, key, size, mtime, err, errcap);
}

typedef struct {
    char key[PATH_MAX];
    char path[PATH_MAX];
    uint8_t digest[MODEL_ORACLE_SHA256_BYTES];
} source_file_t;

static int source_file_cmp(const void *a, const void *b) {
    return strcmp(((const source_file_t *)a)->key, ((const source_file_t *)b)->key);
}

static int tile_desc_cmp(const void *a, const void *b) {
    return strcmp(((const tile_desc_t *)a)->name, ((const tile_desc_t *)b)->name);
}

static int parse_source_files(const model_oracle_config_t *cfg, source_file_t **out,
                              size_t *count_out, uint8_t aggregate[32],
                              char *err, size_t errcap) {
    hash_manifest_t manifest;
    source_file_t *files = NULL;
    EVP_MD_CTX *hash = NULL;
    int rc = -1;
    if (read_hash_manifest(cfg->source_sha256_manifest, &manifest, err, errcap) < 0) return -1;
    if (!manifest.count || manifest.count > ORACLE_MAX_SOURCE_FILES) {
        oracle_error(err, errcap, "source SHA manifest has invalid entry count"); goto done;
    }
    files = calloc(manifest.count, sizeof *files);
    if (!files) { oracle_error(err, errcap, "allocate source file identities"); goto done; }
    for (size_t i = 0; i < manifest.count; i++) {
        char resolved[PATH_MAX];
        if (canonical_source_key(cfg->source_root, manifest.items[i].path, files[i].key, sizeof files[i].key, err, errcap) < 0 ||
            source_path(cfg->source_root, manifest.items[i].path, resolved, sizeof resolved, err, errcap) < 0) goto done;
        if (sha256_file(resolved, files[i].digest, &(struct stat){0}, err, errcap) < 0) goto done;
        if (memcmp(files[i].digest, manifest.items[i].digest, 32)) {
            oracle_error(err, errcap, "source SHA mismatch for %s", files[i].key); goto done;
        }
        if (snprintf(files[i].path, sizeof files[i].path, "%s", resolved) >= (int)sizeof files[i].path) {
            oracle_error(err, errcap, "source file path too long"); goto done;
        }
    }
    qsort(files, manifest.count, sizeof *files, source_file_cmp);
    for (size_t i = 1; i < manifest.count; i++) if (!strcmp(files[i - 1].key, files[i].key)) {
        oracle_error(err, errcap, "duplicate source identity path %s", files[i].key); goto done;
    }
    if (hash_begin(&hash, err, errcap) < 0) goto done;
    for (size_t i = 0; i < manifest.count; i++) {
        if (hash_update_string(hash, files[i].key, err, errcap) < 0 || hash_bytes(hash, files[i].digest, 32, err, errcap) < 0) goto done;
    }
    if (hash_finish(hash, aggregate, err, errcap) < 0) { hash = NULL; goto done; }
    hash = NULL;
    *out = files; *count_out = manifest.count; files = NULL; rc = 0;
done:
    if (hash) EVP_MD_CTX_free(hash);
    free(files); hash_manifest_clear(&manifest);
    return rc;
}

static int parse_tile_catalog(const model_oracle_config_t *cfg, tile_desc_t **out,
                              size_t *count_out, uint8_t manifest_sha[32],
                              uint8_t content_sha[32], int full_hash,
                              const seal_meta_vector_t *sealed, seal_meta_vector_t *current_meta,
                              char *err, size_t errcap) {
    FILE *f = NULL;
    char *line = NULL;
    size_t linecap = 0, count = 0, capacity = 0;
    tile_desc_t *tiles = NULL;
    hash_manifest_t hashes;
    EVP_MD_CTX *content_hash = NULL;
    int rc = -1;
    struct stat st;
    uint64_t meta_size, meta_mtime;
    memset(&hashes, 0, sizeof hashes);
    f = fopen(cfg->tile_cache_manifest, "r");
    if (!f) return oracle_error(err, errcap, "open tile manifest %s: %s", cfg->tile_cache_manifest, strerror(errno));
    ssize_t got = getline(&line, &linecap, f);
    if (got < 0) { oracle_error(err, errcap, "empty CPU tile cache manifest"); goto done; }
    while (got && (line[got - 1] == '\n' || line[got - 1] == '\r')) line[--got] = '\0';
    if (strcmp(line, ORACLE_CACHE_HEADER)) { oracle_error(err, errcap, "not a hold_cache CPU manifest"); goto done; }
    while ((got = getline(&line, &linecap, f)) >= 0) {
        char name[MODEL_ORACLE_LABEL_BYTES * 2];
        unsigned long long off, bytes;
        unsigned type;
        int rows, k, stride, experts, used = 0;
        tile_desc_t *t;
        size_t n;
        while (got && (line[got - 1] == '\n' || line[got - 1] == '\r')) line[--got] = '\0';
        if (!got) continue;
        if (sscanf(line, "%127s %llu %llu %u %d %d %d %d%n", name, &off, &bytes, &type, &rows, &k, &stride, &experts, &used) != 8 || used <= 0) {
            oracle_error(err, errcap, "malformed CPU tile manifest row"); goto done;
        }
        for (const char *p = line + used; *p; p++) if (*p != ' ' && *p != '\t') {
            oracle_error(err, errcap, "extra columns in CPU tile manifest row"); goto done;
        }
        if (count >= ORACLE_MAX_TENSORS || bytes > SIZE_MAX || rows < 0 || k < 0 || stride < 0 || experts < 0) {
            oracle_error(err, errcap, "invalid CPU tile manifest dimensions"); goto done;
        }
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 128;
            tile_desc_t *p = realloc(tiles, next * sizeof *p);
            if (!p) { oracle_error(err, errcap, "allocate CPU tile manifest rows"); goto done; }
            tiles = p; capacity = next;
        }
        t = &tiles[count++]; memset(t, 0, sizeof *t);
        n = strlen(name);
        if (!n || n >= sizeof t->name) { oracle_error(err, errcap, "invalid CPU tensor name"); goto done; }
        memcpy(t->name, name, n + 1);
        t->offset = off; t->bytes = bytes; t->type = type; t->rows = rows; t->k = k; t->stride = stride; t->experts = experts;
    }
    if (ferror(f)) { oracle_error(err, errcap, "read tile manifest: %s", strerror(errno)); goto done; }
    if (!count) { oracle_error(err, errcap, "CPU tile manifest contains no tensors"); goto done; }
    if (sha256_file(cfg->tile_cache_manifest, manifest_sha, &st, err, errcap) < 0) goto done;
    if (read_hash_manifest(cfg->tile_cache_sha256_manifest, &hashes, err, errcap) < 0) goto done;
    if (hashes.count != count) { oracle_error(err, errcap, "tile hash manifest tensor count mismatch"); goto done; }
    for (size_t i = 0; i < count; i++) {
        const hash_line_t *entry = hash_by_basename(&hashes, tiles[i].name);
        if (!entry) { oracle_error(err, errcap, "missing or duplicate tensor hash for %s", tiles[i].name); goto done; }
        memcpy(tiles[i].digest, entry->digest, 32); tiles[i].digest_found = 1;
    }
    qsort(tiles, count, sizeof *tiles, tile_desc_cmp);
    for (size_t i = 1; i < count; i++) if (!strcmp(tiles[i - 1].name, tiles[i].name)) {
        oracle_error(err, errcap, "duplicate tensor %s in CPU tile manifest", tiles[i].name); goto done;
    }
    if (hash_begin(&content_hash, err, errcap) < 0) goto done;
    for (size_t i = 0; i < count; i++) {
        char key[MODEL_ORACLE_LABEL_BYTES * 2 + 16], path[PATH_MAX];
        if (hash_update_string(content_hash, tiles[i].name, err, errcap) < 0 ||
            hash_update_u64(content_hash, tiles[i].bytes, err, errcap) < 0 ||
            hash_update_u32(content_hash, tiles[i].type, err, errcap) < 0 ||
            hash_update_u32(content_hash, (uint32_t)tiles[i].rows, err, errcap) < 0 ||
            hash_update_u32(content_hash, (uint32_t)tiles[i].k, err, errcap) < 0 ||
            hash_update_u32(content_hash, (uint32_t)tiles[i].stride, err, errcap) < 0 ||
            hash_update_u32(content_hash, (uint32_t)tiles[i].experts, err, errcap) < 0 ||
            hash_bytes(content_hash, tiles[i].digest, 32, err, errcap) < 0) goto done;
        if (snprintf(key, sizeof key, "tile/%s", tiles[i].name) >= (int)sizeof key ||
            path_join(path, sizeof path, cfg->tile_cache_dir, tiles[i].name, err, errcap) < 0) {
            oracle_error(err, errcap, "tile cache path too long"); goto done;
        }
        if (full_hash) {
            uint8_t got_hash[32];
            if (sha256_file(path, got_hash, &st, err, errcap) < 0) goto done;
            if (memcmp(got_hash, tiles[i].digest, 32)) { oracle_error(err, errcap, "tile SHA mismatch for %s", tiles[i].name); goto done; }
            meta_size = (uint64_t)st.st_size; meta_mtime = stat_mtime_ns(&st);
            if (meta_push(current_meta, key, meta_size, meta_mtime, err, errcap) < 0) goto done;
        } else if (check_or_add_meta(current_meta, sealed, key, path, err, errcap) < 0) goto done;
    }
    if (hash_finish(content_hash, content_sha, err, errcap) < 0) { content_hash = NULL; goto done; }
    content_hash = NULL;
    *out = tiles; *count_out = count; tiles = NULL; rc = 0;
done:
    if (content_hash) EVP_MD_CTX_free(content_hash);
    free(tiles); free(line); if (f) fclose(f); hash_manifest_clear(&hashes);
    return rc;
}

static int verify_production_topology(const model_oracle_config_t *cfg, char *err, size_t errcap) {
    static const uint32_t expected_shares[MODEL_ORACLE_DIES] = {112, 97, 81, 116, 97, 97};
    static const int32_t expected_map[MODEL_ORACLE_ATTN_LAYERS] = {1, 1, 1, 1, 4, 4, 4, 4, 5, 5, 5, 5};
    if (!cfg || !cfg->source_root || !cfg->source_sha256_manifest || !cfg->source_revision ||
        !cfg->tile_cache_dir || !cfg->tile_cache_manifest || !cfg->tile_cache_sha256_manifest ||
        !cfg->checkpoint_sha256_manifest || !cfg->primary_checkpoint_path || !cfg->ple_shard_path ||
        !cfg->teacher_ids || !cfg->token_count || cfg->token_count > UINT_MAX ||
        !cfg->flash_nblk || cfg->flash_nblk > 4096 || cfg->ple_data_offset > INT64_MAX ||
        host_u32_bits(cfg->epsilon) != host_u32_bits(1e-6f)) {
        return oracle_error(err, errcap, "incomplete or out-of-range model oracle configuration");
    }
    if (!*cfg->source_revision || strlen(cfg->source_revision) >= MODEL_ORACLE_REVISION_BYTES)
        return oracle_error(err, errcap, "invalid source revision identity");
    for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
        if (cfg->shares[d] != expected_shares[d]) return oracle_error(err, errcap, "production share mismatch at logical die %u", d);
        for (unsigned p = 0; p < d; p++) if (cfg->physical_dies[p] == cfg->physical_dies[d])
            return oracle_error(err, errcap, "duplicate physical die ID %u", cfg->physical_dies[d]);
    }
    for (uint64_t i = 0; i < cfg->token_count; i++) if (cfg->teacher_ids[i] > 248319u)
        return oracle_error(err, errcap, "teacher ID at position %" PRIu64 " exceeds vocabulary", i);
    for (unsigned i = 0; i < MODEL_ORACLE_ATTN_LAYERS; i++) {
        if (cfg->attention_map[i] != expected_map[i]) return oracle_error(err, errcap, "production attention map mismatch at layer index %u", i);
    }
    return 0;
}

static int production_environment_matches(const model_oracle_config_t *cfg, char *err, size_t errcap) {
    char shares[128], map[128];
    size_t used = 0;
    for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
        int n = snprintf(shares + used, sizeof shares - used, "%s%u", d ? "," : "", cfg->shares[d]);
        if (n < 0 || (size_t)n >= sizeof shares - used) return oracle_error(err, errcap, "format production share list");
        used += (size_t)n;
    }
    used = 0;
    for (unsigned i = 0; i < MODEL_ORACLE_ATTN_LAYERS; i++) {
        int n = snprintf(map + used, sizeof map - used, "%s%d", i ? "," : "", cfg->attention_map[i]);
        if (n < 0 || (size_t)n >= sizeof map - used) return oracle_error(err, errcap, "format production attention map");
        used += (size_t)n;
    }
    const char *env_shares = getenv("SPLIT_SHARES"), *env_map = getenv("ATTN_MAP");
    if (!env_shares || strcmp(env_shares, shares) || !env_map || strcmp(env_map, map))
        return oracle_error(err, errcap, "SPLIT_SHARES/ATTN_MAP must match explicit production oracle configuration");
    uint64_t cumulative = 0;
    for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
        uint64_t expected_lo = UINT64_C(1000000) * cumulative / 600u;
        cumulative += cfg->shares[d];
        uint64_t expected_hi = UINT64_C(1000000) * cumulative / 600u;
        unsigned lo = split_bound(1000000u, (int)d, MODEL_ORACLE_DIES);
        unsigned hi = split_bound(1000000u, (int)d + 1, MODEL_ORACLE_DIES);
        if (hi < lo || lo != expected_lo || hi != expected_hi)
            return oracle_error(err, errcap, "split share arithmetic mismatch at die %u", d);
    }
    for (unsigned i = 0; i < MODEL_ORACLE_ATTN_LAYERS; i++) if (attn_die_of(3 + 4 * (int)i) != cfg->attention_map[i])
        return oracle_error(err, errcap, "cached ATTN_MAP does not match oracle topology");
    return 0;
}

static int sha_source_identity(const model_oracle_config_t *cfg, uint8_t out[32], char *err, size_t errcap) {
    source_file_t *files = NULL;
    size_t count = 0;
    uint8_t unused[32];
    (void)unused;
    if (parse_source_files(cfg, &files, &count, out, err, errcap) < 0) return -1;
    free(files);
    return 0;
}

static int canonical_checkpoint_digest(const model_oracle_config_t *cfg, int full_hash,
                                       const seal_meta_vector_t *sealed, seal_meta_vector_t *current,
                                       uint8_t primary_out[32], uint8_t ple_out[32], uint8_t combined_out[32],
                                       char *err, size_t errcap) {
    hash_manifest_t hashes;
    const hash_line_t *primary_entry, *ple_entry;
    EVP_MD_CTX *ctx = NULL;
    const char *primary_name = path_basename(cfg->primary_checkpoint_path);
    const char *ple_name = path_basename(cfg->ple_shard_path);
    const char *paths[2] = {cfg->primary_checkpoint_path, cfg->ple_shard_path};
    const char *roles[2] = {"primary", "ple"};
    const hash_line_t *entries[2];
    uint8_t *outputs[2] = {primary_out, ple_out};
    if (!strcmp(primary_name, ple_name)) return oracle_error(err, errcap, "primary and PLE checkpoint basenames must differ");
    if (read_hash_manifest(cfg->checkpoint_sha256_manifest, &hashes, err, errcap) < 0) return -1;
    if (hashes.count != 2) { hash_manifest_clear(&hashes); return oracle_error(err, errcap, "checkpoint hash manifest must identify exactly primary and PLE shards"); }
    primary_entry = hash_by_basename(&hashes, primary_name);
    ple_entry = hash_by_basename(&hashes, ple_name);
    if (!primary_entry || !ple_entry) { hash_manifest_clear(&hashes); return oracle_error(err, errcap, "checkpoint hash manifest lacks primary or PLE shard"); }
    entries[0] = primary_entry; entries[1] = ple_entry;
    if (hash_begin(&ctx, err, errcap) < 0) { hash_manifest_clear(&hashes); return -1; }
    for (unsigned i = 0; i < 2; i++) {
        struct stat st;
        char key[32];
        snprintf(key, sizeof key, "checkpoint/%s", roles[i]);
        if (full_hash) {
            uint8_t actual[32];
            if (sha256_file(paths[i], actual, &st, err, errcap) < 0) goto fail;
            if (memcmp(actual, entries[i]->digest, 32)) { oracle_error(err, errcap, "checkpoint SHA mismatch for %s shard", roles[i]); goto fail; }
            if (meta_push(current, key, (uint64_t)st.st_size, stat_mtime_ns(&st), err, errcap) < 0) goto fail;
        } else if (check_or_add_meta(current, sealed, key, paths[i], err, errcap) < 0) goto fail;
        memcpy(outputs[i], entries[i]->digest, 32);
        if (hash_update_string(ctx, roles[i], err, errcap) < 0 || hash_bytes(ctx, entries[i]->digest, 32, err, errcap) < 0) goto fail;
    }
    if (hash_finish(ctx, combined_out, err, errcap) < 0) { ctx = NULL; goto fail; }
    ctx = NULL;
    hash_manifest_clear(&hashes);
    return 0;
fail:
    if (ctx) EVP_MD_CTX_free(ctx);
    hash_manifest_clear(&hashes);
    return -1;
}

static int validate_ple_extent(const model_oracle_config_t *cfg, char *err, size_t errcap) {
    struct stat st;
    uint64_t max_row = 0;
    if (stat(cfg->ple_shard_path, &st) < 0) return oracle_error(err, errcap, "stat PLE shard: %s", strerror(errno));
    if (!S_ISREG(st.st_mode) || st.st_size < 0 || cfg->ple_data_offset > (uint64_t)st.st_size)
        return oracle_error(err, errcap, "invalid PLE shard/offset");
    for (unsigned h = 0; h < PLE_HEADS; h++) {
        uint64_t end = ple_head_offsets[h] + ple_head_vocab[h];
        if (end > max_row) max_row = end;
    }
    if (max_row > UINT64_MAX / 90u || cfg->ple_data_offset + max_row * 90u > (uint64_t)st.st_size)
        return oracle_error(err, errcap, "PLE shard is too short for real table and configured data offset");
    return 0;
}

static int identity_from_config(const model_oracle_config_t *cfg, model_oracle_identity_t *id,
                                int full_hash, const seal_meta_vector_t *sealed,
                                seal_meta_vector_t *current, char *err, size_t errcap) {
    tile_desc_t *tiles = NULL;
    size_t tile_count = 0;
    EVP_MD_CTX *ids_hash = NULL;
    int rc = -1;
    memset(id, 0, sizeof *id);
    if (verify_production_topology(cfg, err, errcap) < 0 || production_environment_matches(cfg, err, errcap) < 0 ||
        validate_ple_extent(cfg, err, errcap) < 0) return -1;
    memcpy(id->source_revision, cfg->source_revision, strlen(cfg->source_revision) + 1);
    if (sha_source_identity(cfg, id->source_content_sha256, err, errcap) < 0) goto done;
    if (parse_tile_catalog(cfg, &tiles, &tile_count, id->tile_manifest_sha256, id->tile_content_sha256,
                           full_hash, sealed, current, err, errcap) < 0) goto done;
    if (canonical_checkpoint_digest(cfg, full_hash, sealed, current, id->primary_checkpoint_sha256,
                                    id->ple_checkpoint_sha256, id->checkpoint_content_sha256,
                                    err, errcap) < 0) goto done;
    if (hash_begin(&ids_hash, err, errcap) < 0) goto done;
    for (uint64_t i = 0; i < cfg->token_count; i++) if (hash_update_u32(ids_hash, cfg->teacher_ids[i], err, errcap) < 0) goto done;
    if (hash_finish(ids_hash, id->teacher_ids_sha256, err, errcap) < 0) { ids_hash = NULL; goto done; }
    ids_hash = NULL;
    id->ple_data_offset = cfg->ple_data_offset;
    id->token_count = cfg->token_count;
    id->flash_nblk = cfg->flash_nblk;
    id->epsilon_bits = host_u32_bits(cfg->epsilon);
    memcpy(id->physical_dies, cfg->physical_dies, sizeof id->physical_dies);
    memcpy(id->shares, cfg->shares, sizeof id->shares);
    memcpy(id->attention_map, cfg->attention_map, sizeof id->attention_map);
    rc = 0;
done:
    if (ids_hash) EVP_MD_CTX_free(ids_hash);
    free(tiles);
    (void)tile_count;
    return rc;
}

static int meta_write(FILE *f, const seal_meta_vector_t *meta, char *err, size_t errcap) {
    if (meta->count > UINT32_MAX || write_u32(f, (uint32_t)meta->count, err, errcap) < 0) return -1;
    for (size_t i = 0; i < meta->count; i++) {
        size_t n = strlen(meta->items[i].key);
        if (!n || n > UINT16_MAX || write_u32(f, (uint32_t)n, err, errcap) < 0 ||
            write_all(f, meta->items[i].key, n, err, errcap) < 0 ||
            write_u64(f, meta->items[i].size, err, errcap) < 0 ||
            write_u64(f, meta->items[i].mtime_ns, err, errcap) < 0) return -1;
    }
    return 0;
}

static int meta_read(FILE *f, seal_meta_vector_t *meta, char *err, size_t errcap) {
    uint32_t count;
    memset(meta, 0, sizeof *meta);
    if (read_u32(f, &count, err, errcap) < 0) return -1;
    if (count > ORACLE_MAX_TENSORS + 16u) return oracle_error(err, errcap, "seal has too many input metadata entries");
    for (uint32_t i = 0; i < count; i++) {
        uint32_t n;
        uint64_t size, mtime;
        char *key;
        if (read_u32(f, &n, err, errcap) < 0) goto fail;
        if (!n || n >= PATH_MAX) { oracle_error(err, errcap, "invalid seal metadata key length"); goto fail; }
        key = malloc((size_t)n + 1);
        if (!key) { oracle_error(err, errcap, "allocate seal key"); goto fail; }
        if (read_all(f, key, n, err, errcap) < 0 || read_u64(f, &size, err, errcap) < 0 || read_u64(f, &mtime, err, errcap) < 0) { free(key); goto fail; }
        key[n] = '\0';
        if (meta_push(meta, key, size, mtime, err, errcap) < 0) { free(key); goto fail; }
        free(key);
    }
    for (size_t i = 0; i < meta->count; i++) for (size_t j = i + 1; j < meta->count; j++)
        if (!strcmp(meta->items[i].key, meta->items[j].key)) { oracle_error(err, errcap, "duplicate key in verification seal"); goto fail; }
    return 0;
fail:
    meta_clear(meta);
    return -1;
}

static int seal_read(FILE *f, model_oracle_identity_t *identity, seal_meta_vector_t *meta,
                     char *err, size_t errcap) {
    char magic[8]; uint32_t version;
    if (fseek(f, 0, SEEK_SET) < 0) return oracle_error(err, errcap, "seek verification receipt: %s", strerror(errno));
    if (read_all(f, magic, sizeof magic, err, errcap) < 0 || memcmp(magic, ORACLE_SEAL_MAGIC, sizeof magic))
        return oracle_error(err, errcap, "invalid verification receipt magic");
    if (read_u32(f, &version, err, errcap) < 0 || version != MODEL_ORACLE_FORMAT_VERSION)
        return oracle_error(err, errcap, "unsupported verification receipt version");
    if (identity_read(f, identity, err, errcap) < 0 || meta_read(f, meta, err, errcap) < 0) return -1;
    return 0;
}

static int seal_write(FILE *f, const model_oracle_identity_t *identity, const seal_meta_vector_t *meta,
                      char *err, size_t errcap) {
    if (fseek(f, 0, SEEK_SET) < 0) return oracle_error(err, errcap, "seek verification receipt: %s", strerror(errno));
    if (write_all(f, ORACLE_SEAL_MAGIC, 8, err, errcap) < 0 ||
        write_u32(f, MODEL_ORACLE_FORMAT_VERSION, err, errcap) < 0 ||
        identity_write(f, identity, err, errcap) < 0 || meta_write(f, meta, err, errcap) < 0 ||
        fflush(f) == EOF || fsync(fileno(f)) < 0) return oracle_error(err, errcap, "write verification receipt: %s", strerror(errno ? errno : EIO));
    return 0;
}

static int verify_inputs_impl(const model_oracle_config_t *config, FILE *verification_receipt,
                              model_oracle_identity_t *identity, int *full_hash_performed,
                              int allow_full_hash, char *error, size_t error_capacity) {
    model_oracle_identity_t sealed_identity, current_identity;
    seal_meta_vector_t sealed_meta = {0}, current_meta = {0};
    long length;
    int full_hash, rc = -1;
    if (!config || !verification_receipt || !identity || !full_hash_performed)
        return oracle_error(error, error_capacity, "model_oracle_verify_inputs requires config, receipt and identity outputs");
    if (fseek(verification_receipt, 0, SEEK_END) < 0 || (length = ftell(verification_receipt)) < 0)
        return oracle_error(error, error_capacity, "seek verification receipt: %s", strerror(errno));
    full_hash = length == 0;
    if (full_hash && !allow_full_hash)
        return oracle_error(error, error_capacity, "verification receipt is empty; full hashing is disabled on this path");
    if (length && seal_read(verification_receipt, &sealed_identity, &sealed_meta, error, error_capacity) < 0) goto done;
    if (identity_from_config(config, &current_identity, full_hash, full_hash ? NULL : &sealed_meta,
                             &current_meta, error, error_capacity) < 0) goto done;
    if (!full_hash) {
        if (!identity_equal(&current_identity, &sealed_identity)) {
            oracle_error(error, error_capacity, "input content identity differs from verification receipt"); goto done;
        }
        if (current_meta.count != sealed_meta.count) {
            oracle_error(error, error_capacity, "input metadata inventory differs from verification receipt"); goto done;
        }
        for (size_t i = 0; i < current_meta.count; i++) {
            const seal_meta_t *old = meta_find(&sealed_meta, current_meta.items[i].key);
            if (!old || old->size != current_meta.items[i].size || old->mtime_ns != current_meta.items[i].mtime_ns) {
                oracle_error(error, error_capacity, "large input metadata differs for %s", current_meta.items[i].key); goto done;
            }
        }
    } else if (seal_write(verification_receipt, &current_identity, &current_meta, error, error_capacity) < 0) {
        goto done;
    }
    *identity = current_identity;
    *full_hash_performed = full_hash;
    rc = 0;
done:
    meta_clear(&sealed_meta); meta_clear(&current_meta);
    return rc;
}

int model_oracle_verify_inputs(const model_oracle_config_t *config, FILE *verification_receipt,
                               model_oracle_identity_t *identity, int *full_hash_performed,
                               char *error, size_t error_capacity) {
    if (!full_hash_performed)
        return oracle_error(error, error_capacity, "full_hash_performed output is required for initial verification");
    return verify_inputs_impl(config, verification_receipt, identity, full_hash_performed,
                              1, error, error_capacity);
}

int model_oracle_verify_inputs_from_receipt(const model_oracle_config_t *config,
                                             FILE *verification_receipt,
                                             model_oracle_identity_t *identity,
                                             char *error, size_t error_capacity) {
    int full_hash_performed = 0;
    if (verify_inputs_impl(config, verification_receipt, identity, &full_hash_performed,
                           0, error, error_capacity) < 0) return -1;
    if (full_hash_performed)
        return oracle_error(error, error_capacity, "cached verification unexpectedly performed full hashing");
    return 0;
}

static void builder_clear(record_builder_t *b) {
    for (uint32_t i = 0; i < b->count; i++) free(b->items[i].raw);
    b->count = 0;
}

static void builder_destroy(record_builder_t *b) {
    builder_clear(b); free(b->items); memset(b, 0, sizeof *b);
}

static int builder_add(record_builder_t *b, const char *label, uint32_t layer,
                       uint32_t logical_die, uint32_t physical_die, uint32_t value_type,
                       uint32_t rank, const uint64_t dims[4], const void *bytes,
                       uint64_t byte_count, int keep_raw, char *err, size_t errcap) {
    owned_item_t *owned;
    size_t label_len = strlen(label);
    if (!label_len || label_len >= MODEL_ORACLE_LABEL_BYTES || rank > 4 || byte_count > SIZE_MAX || (byte_count && !bytes))
        return oracle_error(err, errcap, "invalid oracle item %s", label);
    if (b->count == b->capacity) {
        uint32_t next = b->capacity ? b->capacity * 2 : 64;
        if (next > ORACLE_MAX_ITEMS) return oracle_error(err, errcap, "oracle position exceeds item limit");
        owned_item_t *p = realloc(b->items, (size_t)next * sizeof *p);
        if (!p) return oracle_error(err, errcap, "allocate oracle record items");
        b->items = p; b->capacity = next;
    }
    owned = &b->items[b->count]; memset(owned, 0, sizeof *owned);
    memcpy(owned->item.label, label, label_len + 1);
    owned->item.layer = layer; owned->item.logical_die = logical_die; owned->item.physical_die = physical_die;
    owned->item.value_type = value_type; owned->item.rank = rank; owned->item.byte_count = byte_count;
    if (dims) memcpy(owned->item.dimensions, dims, sizeof owned->item.dimensions);
    if (sha256_memory(bytes, (size_t)byte_count, owned->item.sha256, err, errcap) < 0) return -1;
    if (keep_raw && byte_count) {
        owned->raw = malloc((size_t)byte_count);
        if (!owned->raw) return oracle_error(err, errcap, "allocate raw bytes for %s", label);
        memcpy(owned->raw, bytes, (size_t)byte_count);
        owned->item.raw_bytes = owned->raw;
        owned->item.raw_byte_count = byte_count;
    }
    b->count++;
    return 0;
}

static int record_write(FILE *f, const model_oracle_record_t *r, const record_builder_t *b,
                        char *err, size_t errcap) {
    if (write_all(f, ORACLE_RECORD_MAGIC, 4, err, errcap) < 0 ||
        write_u64(f, r->position, err, errcap) < 0 ||
        write_u64(f, r->visible_length, err, errcap) < 0 ||
        write_u32(f, r->teacher_id, err, errcap) < 0 ||
        write_u32(f, r->prediction, err, errcap) < 0 ||
        write_u32(f, r->prediction_score_bits, err, errcap) < 0 ||
        write_u32(f, b->count, err, errcap) < 0) return -1;
    for (uint32_t i = 0; i < b->count; i++) {
        const model_oracle_item_t *x = &b->items[i].item;
        uint32_t label_len = (uint32_t)strlen(x->label);
        if (write_u32(f, label_len, err, errcap) < 0 || write_all(f, x->label, label_len, err, errcap) < 0 ||
            write_u32(f, x->layer, err, errcap) < 0 || write_u32(f, x->logical_die, err, errcap) < 0 ||
            write_u32(f, x->physical_die, err, errcap) < 0 || write_u32(f, x->value_type, err, errcap) < 0 ||
            write_u32(f, x->rank, err, errcap) < 0) return -1;
        for (unsigned d = 0; d < 4; d++) if (write_u64(f, x->dimensions[d], err, errcap) < 0) return -1;
        if (write_u64(f, x->byte_count, err, errcap) < 0 || write_all(f, x->sha256, 32, err, errcap) < 0 ||
            write_u64(f, x->raw_byte_count, err, errcap) < 0 ||
            (x->raw_byte_count && write_all(f, x->raw_bytes, (size_t)x->raw_byte_count, err, errcap) < 0)) return -1;
    }
    return 0;
}

static void reader_clear_record(model_oracle_reader_t *r) {
    for (uint32_t i = 0; i < r->allocated_items; i++) free(r->raw_allocations[i]);
    free(r->raw_allocations); free(r->items);
    r->raw_allocations = NULL; r->items = NULL; r->allocated_items = 0;
}

static const char *item_component(const char *label) {
    const char *dot = strrchr(label, '.');
    return dot ? dot + 1 : label;
}

static int item_is_gather(const char *label) {
    const char *component = item_component(label);
    return !strcmp(component, "gather_k") || !strcmp(component, "gather_v");
}

static int item_requires_raw(const char *label) {
    return !strcmp(item_component(label), "selected_ids") || !strncmp(label, "tok.", 4);
}

static uint32_t item_canonical_type(const char *label) {
    const char *component = item_component(label);
    if (!strcmp(component, "kq") || !strcmp(component, "vq")) return MODEL_ORACLE_I8;
    if (!strcmp(component, "ks") || !strcmp(component, "vs") || !strcmp(component, "pool")) return MODEL_ORACLE_F16;
    if (!strcmp(component, "selected_ids") || !strncmp(label, "tok.", 4)) return MODEL_ORACLE_U32;
    return MODEL_ORACLE_F32;
}

static uint64_t item_type_width(uint32_t value_type) {
    switch (value_type) {
    case MODEL_ORACLE_I8: return 1;
    case MODEL_ORACLE_F16: return 2;
    case MODEL_ORACLE_F32:
    case MODEL_ORACLE_U32: return 4;
    default: return 0;
    }
}

static int validate_item_layout(const model_oracle_item_t *item, char *err, size_t errcap) {
    uint64_t width, elements = 1;
    int gather;
    if (!item || !item->label[0] || !memchr(item->label, '\0', sizeof item->label))
        return oracle_error(err, errcap, "invalid oracle item label");
    if (item->layer != MODEL_ORACLE_GLOBAL_LAYER && item->layer >= MODEL_ORACLE_BLOCKS)
        return oracle_error(err, errcap, "item %s has invalid layer %u", item->label, item->layer);
    if (item->logical_die >= MODEL_ORACLE_DIES)
        return oracle_error(err, errcap, "item %s has invalid logical die %u", item->label, item->logical_die);
    width = item_type_width(item->value_type);
    if (!width || item->value_type != item_canonical_type(item->label))
        return oracle_error(err, errcap, "item %s has invalid value type %u", item->label, item->value_type);
    if (!item->byte_count || item->byte_count > (UINT64_C(1) << 34) ||
        (uint64_t)(size_t)item->byte_count != item->byte_count)
        return oracle_error(err, errcap, "item %s has invalid byte count", item->label);
    gather = item_is_gather(item->label);
    if (item->rank != (gather ? 3u : 1u))
        return oracle_error(err, errcap, "item %s has invalid rank %u", item->label, item->rank);
    for (uint32_t d = 0; d < item->rank; d++) {
        if (!item->dimensions[d] || elements > UINT64_MAX / item->dimensions[d])
            return oracle_error(err, errcap, "item %s has invalid dimensions", item->label);
        elements *= item->dimensions[d];
    }
    for (uint32_t d = item->rank; d < 4; d++) if (item->dimensions[d])
        return oracle_error(err, errcap, "item %s has non-flat trailing dimensions", item->label);
    if (gather && (item->dimensions[1] != 2 || item->dimensions[2] != 256))
        return oracle_error(err, errcap, "item %s has invalid gathered shape", item->label);
    if (elements > UINT64_MAX / width || elements * width != item->byte_count)
        return oracle_error(err, errcap, "item %s dimensions do not match byte count", item->label);
    if (item_requires_raw(item->label)) {
        if (!item->raw_bytes || item->raw_byte_count != item->byte_count)
            return oracle_error(err, errcap, "item %s requires complete raw bytes", item->label);
    } else if (item->raw_bytes || item->raw_byte_count) {
        return oracle_error(err, errcap, "item %s must use its digest, not raw bytes", item->label);
    }
    return 0;
}

static int validate_item_identity(const model_oracle_item_t *item,
                                  const model_oracle_identity_t *identity,
                                  char *err, size_t errcap) {
    if (validate_item_layout(item, err, errcap) < 0) return -1;
    if (item->physical_die != identity->physical_dies[item->logical_die])
        return oracle_error(err, errcap, "item %s physical die %u does not match logical die %u identity",
                            item->label, item->physical_die, item->logical_die);
    return 0;
}

int model_oracle_reader_open(model_oracle_reader_t **out, FILE *record_stream,
                             const model_oracle_identity_t *expected,
                             char *error, size_t error_capacity) {
    char magic[8]; uint32_t version;
    model_oracle_identity_t got;
    model_oracle_reader_t *r;
    if (!out || !record_stream || !expected) return oracle_error(error, error_capacity, "reader requires stream and expected identity");
    *out = NULL;
    if (fseek(record_stream, 0, SEEK_SET) < 0) return oracle_error(error, error_capacity, "seek oracle records: %s", strerror(errno));
    if (read_all(record_stream, magic, sizeof magic, error, error_capacity) < 0 || memcmp(magic, ORACLE_MAGIC, sizeof magic))
        return oracle_error(error, error_capacity, "invalid oracle record header");
    if (read_u32(record_stream, &version, error, error_capacity) < 0 || version != MODEL_ORACLE_FORMAT_VERSION)
        return oracle_error(error, error_capacity, "unsupported oracle record version");
    if (identity_read(record_stream, &got, error, error_capacity) < 0) return -1;
    if (!identity_equal(&got, expected)) return oracle_error(error, error_capacity, "oracle record identity mismatch");
    r = calloc(1, sizeof *r);
    if (!r) return oracle_error(error, error_capacity, "allocate oracle reader");
    r->stream = record_stream; r->identity = got;
    *out = r;
    return 0;
}

int model_oracle_reader_next(model_oracle_reader_t *reader, model_oracle_record_t *record,
                             char *error, size_t error_capacity) {
    uint8_t magic[4];
    int c;
    uint32_t count;
    if (!reader || !record) return oracle_error(error, error_capacity, "reader_next requires reader and record");
    reader_clear_record(reader);
    c = fgetc(reader->stream);
    if (c == EOF) {
        if (ferror(reader->stream)) return oracle_error(error, error_capacity, "read oracle record: %s", strerror(errno));
        if (reader->next_position != reader->identity.token_count)
            return oracle_error(error, error_capacity, "oracle record stream ended after %" PRIu64 " of %" PRIu64 " positions",
                                reader->next_position, reader->identity.token_count);
        return 0;
    }
    magic[0] = (uint8_t)c;
    if (read_all(reader->stream, magic + 1, 3, error, error_capacity) < 0 || memcmp(magic, ORACLE_RECORD_MAGIC, 4))
        return oracle_error(error, error_capacity, "invalid per-position oracle record marker");
    memset(record, 0, sizeof *record);
    if (read_u64(reader->stream, &record->position, error, error_capacity) < 0 ||
        read_u64(reader->stream, &record->visible_length, error, error_capacity) < 0 ||
        read_u32(reader->stream, &record->teacher_id, error, error_capacity) < 0 ||
        read_u32(reader->stream, &record->prediction, error, error_capacity) < 0 ||
        read_u32(reader->stream, &record->prediction_score_bits, error, error_capacity) < 0 ||
        read_u32(reader->stream, &count, error, error_capacity) < 0) return -1;
    if (record->visible_length != record->position + 1 || record->position != reader->next_position ||
        record->position >= reader->identity.token_count || !count || count > ORACLE_MAX_ITEMS)
        return oracle_error(error, error_capacity, "invalid oracle record position or item count");
    reader->items = calloc(count ? count : 1, sizeof *reader->items);
    reader->raw_allocations = calloc(count ? count : 1, sizeof *reader->raw_allocations);
    if (!reader->items || !reader->raw_allocations) return oracle_error(error, error_capacity, "allocate oracle record data");
    reader->allocated_items = count;
    for (uint32_t i = 0; i < count; i++) {
        model_oracle_item_t *x = &reader->items[i];
        uint32_t label_len;
        uint64_t raw_len;
        if (read_u32(reader->stream, &label_len, error, error_capacity) < 0) return -1;
        if (!label_len || label_len >= MODEL_ORACLE_LABEL_BYTES) return oracle_error(error, error_capacity, "invalid oracle item label length");
        if (read_all(reader->stream, x->label, label_len, error, error_capacity) < 0) return -1;
        if (memchr(x->label, '\0', label_len)) return oracle_error(error, error_capacity, "oracle item label contains an embedded NUL");
        x->label[label_len] = '\0';
        if (read_u32(reader->stream, &x->layer, error, error_capacity) < 0 ||
            read_u32(reader->stream, &x->logical_die, error, error_capacity) < 0 ||
            read_u32(reader->stream, &x->physical_die, error, error_capacity) < 0 ||
            read_u32(reader->stream, &x->value_type, error, error_capacity) < 0 ||
            read_u32(reader->stream, &x->rank, error, error_capacity) < 0) return -1;
        if (x->rank > 4) return oracle_error(error, error_capacity, "invalid oracle item rank");
        for (unsigned d = 0; d < 4; d++) if (read_u64(reader->stream, &x->dimensions[d], error, error_capacity) < 0) return -1;
        if (read_u64(reader->stream, &x->byte_count, error, error_capacity) < 0 ||
            read_all(reader->stream, x->sha256, 32, error, error_capacity) < 0 ||
            read_u64(reader->stream, &raw_len, error, error_capacity) < 0) return -1;
        if (x->byte_count > (UINT64_C(1) << 34) || raw_len > x->byte_count || raw_len > (UINT64_C(1) << 20))
            return oracle_error(error, error_capacity, "oracle item size exceeds format limit");
        if (item_requires_raw(x->label) && raw_len != x->byte_count)
            return oracle_error(error, error_capacity, "oracle item %s requires complete raw bytes", x->label);
        if (!item_requires_raw(x->label) && raw_len)
            return oracle_error(error, error_capacity, "oracle item %s must use its digest", x->label);
        x->raw_byte_count = raw_len;
        if (raw_len) {
            reader->raw_allocations[i] = malloc((size_t)raw_len);
            if (!reader->raw_allocations[i]) return oracle_error(error, error_capacity, "allocate raw oracle item");
            if (read_all(reader->stream, reader->raw_allocations[i], (size_t)raw_len, error, error_capacity) < 0) return -1;
            x->raw_bytes = reader->raw_allocations[i];
        }
        if (validate_item_identity(x, &reader->identity, error, error_capacity) < 0) return -1;
        for (uint32_t j = 0; j < i; j++) if (!strcmp(reader->items[j].label, x->label))
            return oracle_error(error, error_capacity, "duplicate oracle item label %s", x->label);
    }
    record->item_count = count;
    record->items = reader->items;
    reader->next_position++;
    return 1;
}

void model_oracle_reader_close(model_oracle_reader_t *reader) {
    if (!reader) return;
    reader_clear_record(reader);
    free(reader);
}

static int observation_has_label(const model_oracle_observation_t *observations, size_t count,
                                 const char *label, size_t *index) {
    size_t found = SIZE_MAX;
    for (size_t i = 0; i < count; i++) if (observations[i].label &&
        strnlen(observations[i].label, MODEL_ORACLE_LABEL_BYTES) < MODEL_ORACLE_LABEL_BYTES &&
        !strcmp(observations[i].label, label)) {
        if (found != SIZE_MAX) return -1;
        found = i;
    }
    if (found == SIZE_MAX) return 0;
    *index = found; return 1;
}

int model_oracle_compare_record(const model_oracle_record_t *record,
                                const model_oracle_observation_t *observations,
                                size_t observation_count, FILE *result_stream,
                                char *error, size_t error_capacity) {
    int match = 1;
    if (!record || (!observations && observation_count) || !result_stream)
        return oracle_error(error, error_capacity, "compare requires record, observations and result stream");
    if (record->item_count > ORACLE_MAX_ITEMS || (record->item_count && !record->items))
        return oracle_error(error, error_capacity, "invalid expected item array");
    if (!record->item_count)
        return oracle_error(error, error_capacity, "empty full-model oracle record");
    for (uint32_t i = 0; i < record->item_count; i++) {
        if (validate_item_layout(&record->items[i], error, error_capacity) < 0) return -1;
        for (uint32_t j = 0; j < i; j++) if (!strcmp(record->items[i].label, record->items[j].label))
            return oracle_error(error, error_capacity, "duplicate expected oracle label %s", record->items[i].label);
    }
    if (observation_count != record->item_count) match = 0;
    for (size_t i = 0; i < observation_count; i++) {
        const char *label = observations[i].label;
        if (!label || !*label || strnlen(label, MODEL_ORACLE_LABEL_BYTES) >= MODEL_ORACLE_LABEL_BYTES) {
            match = 0;
            continue;
        }
        for (size_t j = 0; j < i; j++) {
            const char *prior = observations[j].label;
            if (prior && *prior && strnlen(prior, MODEL_ORACLE_LABEL_BYTES) < MODEL_ORACLE_LABEL_BYTES &&
                !strcmp(prior, label)) match = 0;
        }
        size_t expected_matches = 0;
        for (uint32_t j = 0; j < record->item_count; j++) {
            const char *expected_label = record->items[j].label;
            if (expected_label[0] && memchr(expected_label, '\0', sizeof record->items[j].label) &&
                !strcmp(label, expected_label)) expected_matches++;
        }
        if (expected_matches != 1) match = 0;
    }
    for (uint32_t i = 0; i < record->item_count; i++) {
        const model_oracle_item_t *expected = &record->items[i];
        size_t found = 0;
        int has = observation_has_label(observations, observation_count, expected->label, &found);
        if (has != 1) { match = 0; continue; }
        if (observations[found].byte_count != expected->byte_count ||
            observations[found].value_type != expected->value_type || observations[found].rank != expected->rank ||
            memcmp(observations[found].dimensions, expected->dimensions, sizeof expected->dimensions) ||
            (expected->byte_count && !observations[found].bytes)) { match = 0; continue; }
        if (expected->raw_bytes) {
            if (expected->raw_byte_count != expected->byte_count ||
                memcmp(expected->raw_bytes, observations[found].bytes, (size_t)expected->byte_count)) match = 0;
        } else {
            uint8_t digest[32];
            if (sha256_memory(observations[found].bytes, (size_t)expected->byte_count, digest, error, error_capacity) < 0) return -1;
            if (memcmp(digest, expected->sha256, 32)) match = 0;
        }
    }
    if (fprintf(result_stream, "{\"position\":%" PRIu64 ",\"visible_length\":%" PRIu64 ",\"scalar_match\":%s}\n",
                record->position, record->visible_length, match ? "true" : "false") < 0 || fflush(result_stream) == EOF)
        return oracle_error(error, error_capacity, "write scalar comparison result: %s", strerror(errno ? errno : EIO));
    return match ? 0 : 1;
}

static const oracle_tensor_t *tensor_find(const model_oracle_t *o, const char *name) {
    for (size_t i = 0; i < o->tensor_count; i++) if (!strcmp(o->tensors[i].name, name)) return &o->tensors[i];
    return NULL;
}

static const oracle_tensor_t *tensor_need(const model_oracle_t *o, const char *name, char *err, size_t errcap) {
    const oracle_tensor_t *t = tensor_find(o, name);
    if (!t) oracle_error(err, errcap, "CPU tile cache lacks required tensor %s", name);
    return t;
}

static int map_tile_cache(model_oracle_t *o, char *err, size_t errcap) {
    tile_desc_t *catalog = NULL;
    size_t count = 0;
    uint8_t manifest_sha[32], content_sha[32];
    seal_meta_vector_t metadata = {0};
    if (parse_tile_catalog(&o->config, &catalog, &count, manifest_sha, content_sha, 0, NULL, &metadata, err, errcap) < 0) {
        meta_clear(&metadata); return -1;
    }
    meta_clear(&metadata);
    if (memcmp(manifest_sha, o->identity.tile_manifest_sha256, 32) || memcmp(content_sha, o->identity.tile_content_sha256, 32)) {
        free(catalog); return oracle_error(err, errcap, "tile cache content identity differs from verified receipt");
    }
    o->tensors = calloc(count, sizeof *o->tensors);
    if (!o->tensors) { free(catalog); return oracle_error(err, errcap, "allocate tile cache mappings"); }
    o->tensor_count = count;
    for (size_t i = 0; i < count; i++) {
        char path[PATH_MAX];
        int fd;
        struct stat st;
        oracle_tensor_t *t = &o->tensors[i];
        memcpy(t->name, catalog[i].name, sizeof t->name);
        t->bytes = (size_t)catalog[i].bytes; t->type = catalog[i].type;
        t->rows = catalog[i].rows; t->k = catalog[i].k; t->stride = catalog[i].stride; t->experts = catalog[i].experts;
        if (path_join(path, sizeof path, o->config.tile_cache_dir, t->name, err, errcap) < 0) { free(catalog); return -1; }
        fd = open(path, O_RDONLY);
        if (fd < 0) { free(catalog); return oracle_error(err, errcap, "open cache tensor %s: %s", t->name, strerror(errno)); }
        if (fstat(fd, &st) < 0 || st.st_size < 0 || (uint64_t)st.st_size != t->bytes) {
            close(fd); free(catalog); return oracle_error(err, errcap, "cache tensor size mismatch for %s", t->name);
        }
        t->mapping_bytes = t->bytes;
        if (t->bytes) {
            t->mapping = mmap(NULL, t->bytes, PROT_READ, MAP_PRIVATE, fd, 0);
            if (t->mapping == MAP_FAILED) { t->mapping = NULL; close(fd); free(catalog); return oracle_error(err, errcap, "mmap cache tensor %s: %s", t->name, strerror(errno)); }
        }
        close(fd);
    }
    free(catalog);
    return 0;
}

static int map_ple_table(model_oracle_t *o, char *err, size_t errcap) {
    struct stat st;
    long page = sysconf(_SC_PAGESIZE);
    int fd;
    uint64_t file_size, map_offset;
    size_t map_bytes;
    if (page <= 0) return oracle_error(err, errcap, "cannot determine mmap page size");
    fd = open(o->config.ple_shard_path, O_RDONLY);
    if (fd < 0) return oracle_error(err, errcap, "open PLE shard: %s", strerror(errno));
    if (fstat(fd, &st) < 0 || st.st_size < 0) { close(fd); return oracle_error(err, errcap, "stat PLE shard: %s", strerror(errno)); }
    file_size = (uint64_t)st.st_size;
    map_offset = o->config.ple_data_offset & ~((uint64_t)page - 1u);
    if (file_size < map_offset || file_size - map_offset > SIZE_MAX) { close(fd); return oracle_error(err, errcap, "PLE mapping is too large"); }
    map_bytes = (size_t)(file_size - map_offset);
    o->ple_mapping = mmap(NULL, map_bytes, PROT_READ, MAP_PRIVATE, fd, (off_t)map_offset);
    close(fd);
    if (o->ple_mapping == MAP_FAILED) { o->ple_mapping = NULL; return oracle_error(err, errcap, "mmap PLE shard: %s", strerror(errno)); }
    o->ple_mapping_bytes = map_bytes;
    o->ple_delta = (size_t)(o->config.ple_data_offset - map_offset);
    o->ple_table = o->ple_mapping + o->ple_delta;
    return 0;
}

static int qsa_cache_allocate(model_oracle_t *o, unsigned layer, char *err, size_t errcap) {
    model_qsa_t *q = calloc(1, sizeof *q);
    size_t m = o->max_cache_positions;
    if (!q) return oracle_error(err, errcap, "allocate QSA oracle state for layer %u", layer);
    o->qsa[layer] = q;
    q->nblk = o->config.flash_nblk;
    q->kq = calloc(m * 512u, 1); q->vq = calloc(m * 512u, 1);
    q->ks = calloc(m * 16u, sizeof(uint16_t)); q->vs = calloc(m * 16u, sizeof(uint16_t));
    q->pool = calloc(m / 4u * 128u, sizeof(uint16_t));
    q->pkt = calloc(m / 128u * 4096u, sizeof(uint16_t));
    q->score = calloc(m / 4u + 64u, sizeof(float));
    q->sel = calloc((size_t)q->nblk + 8u, sizeof(uint32_t));
    q->kc = calloc(o->kcache_floats, sizeof(float)); q->vc = calloc(o->vcache_floats, sizeof(float));
    if (!q->kq || !q->vq || !q->ks || !q->vs || !q->pool || !q->pkt || !q->score || !q->sel || !q->kc || !q->vc)
        return oracle_error(err, errcap, "allocate QSA oracle buffers for layer %u", layer);
    return 0;
}

static void *block_tensor(model_oracle_t *o, int block, const char *suffix,
                          uint32_t *type, size_t *bytes, char *err, size_t errcap) {
    char name[MODEL_ORACLE_LABEL_BYTES * 2];
    int n = snprintf(name, sizeof name, "blk.%d.%s", block, suffix);
    if (n < 0 || (size_t)n >= sizeof name) { oracle_error(err, errcap, "block tensor name too long"); return NULL; }
    const oracle_tensor_t *t = tensor_need(o, name, err, errcap);
    if (!t) return NULL;
    if (type) *type = t->type;
    if (bytes) *bytes = t->bytes;
    return t->mapping;
}

static void *named_tensor(model_oracle_t *o, const char *name, uint32_t *type,
                          size_t *bytes, char *err, size_t errcap) {
    const oracle_tensor_t *t = tensor_need(o, name, err, errcap);
    if (!t) return NULL;
    if (type) *type = t->type;
    if (bytes) *bytes = t->bytes;
    return t->mapping;
}

static int load_weights(model_oracle_t *o, char *err, size_t errcap) {
    for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
        dn_hweights_t *w = &o->weights[b];
        uint32_t type;
        size_t bytes;
#define SET_BLOCK(field, suffix, cast) do { \
    void *p_ = block_tensor(o, b, suffix, NULL, NULL, err, errcap); \
    if (!p_) return -1; \
    w->field = (cast)p_; \
} while (0)
#define SET_BLOCK_TYPED(field, typefield, suffix, cast) do { \
    void *p_ = block_tensor(o, b, suffix, &type, NULL, err, errcap); \
    if (!p_) return -1; \
    w->field = (cast)p_; w->typefield = type; \
} while (0)
        SET_BLOCK(hc_attn_norm, "hc_attn_norm.weight", const float *);
        SET_BLOCK(hc_attn_dj, "hc_attn_dj.weight", const uint16_t *);
        SET_BLOCK(hc_attn_up, "hc_attn_up.weight", const uint16_t *);
        SET_BLOCK(hc_ffn_norm, "hc_ffn_norm.weight", const float *);
        SET_BLOCK(hc_ffn_dj, "hc_ffn_dj.weight", const uint16_t *);
        SET_BLOCK(hc_ffn_up, "hc_ffn_up.weight", const uint16_t *);
        SET_BLOCK(ffn_gate_inp_x, "ffn_gate_inp_x.weight", const uint16_t *);
        SET_BLOCK(ffn_gate_exps, "ffn_gate_exps.weight", const uint8_t *);
        SET_BLOCK(ffn_up_exps, "ffn_up_exps.weight", const uint8_t *);
        SET_BLOCK(ffn_down_exps, "ffn_down_exps.weight", const uint8_t *);
        SET_BLOCK_TYPED(ffn_gate_shexp, ffn_gate_shexp_type, "ffn_gate_shexp.weight", const uint8_t *);
        SET_BLOCK_TYPED(ffn_up_shexp, ffn_up_shexp_type, "ffn_up_shexp.weight", const uint8_t *);
        SET_BLOCK_TYPED(ffn_down_shexp, ffn_down_shexp_type, "ffn_down_shexp.weight", const uint8_t *);
        if (BLK_IS_ATTN(b)) {
            SET_BLOCK_TYPED(attn_q, attn_q_type, "attn_q.weight", const uint8_t *);
            SET_BLOCK_TYPED(attn_k, attn_k_type, "attn_k.weight", const uint8_t *);
            SET_BLOCK_TYPED(attn_v, attn_v_type, "attn_v.weight", const uint8_t *);
            SET_BLOCK_TYPED(attn_output, attn_output_type, "attn_output.weight", const uint8_t *);
            SET_BLOCK(attn_q_norm, "attn_q_norm.weight", const float *);
            SET_BLOCK(attn_k_norm, "attn_k_norm.weight", const float *);
            SET_BLOCK(indexer_x, "indexer_x.weight", const uint16_t *);
            SET_BLOCK(indexer_qn, "indexer.q_norm.weight", const float *);
            SET_BLOCK(indexer_kn, "indexer.k_norm.weight", const float *);
        } else {
            SET_BLOCK_TYPED(attn_qkv, attn_qkv_type, "attn_qkv.weight", const uint8_t *);
            SET_BLOCK_TYPED(attn_gate, attn_gate_type, "attn_gate.weight", const uint8_t *);
            SET_BLOCK_TYPED(ssm_out, ssm_out_type, "ssm_out.weight", const uint8_t *);
            SET_BLOCK(ssm_ab, "ssm_ab.weight", const uint16_t *);
            SET_BLOCK(ssm_ab_bias, "ssm_ab.bias", const float *);
            SET_BLOCK(ssm_conv1d, "ssm_conv1d.weight", const float *);
            SET_BLOCK(ssm_a, "ssm_a", const float *);
            SET_BLOCK(ssm_norm, "ssm_norm.weight", const float *);
        }
        if (b == 1) {
            SET_BLOCK(ple_key, "ple_key.weight", const uint8_t *);
            SET_BLOCK(ple_value, "ple_value.weight", const uint16_t *);
            SET_BLOCK(ple_norm_key, "ple_norm_key.weight", const float *);
            SET_BLOCK(ple_norm_query, "ple_norm_query.weight", const float *);
            SET_BLOCK(ple_norm_conv, "ple_norm_conv.weight", const float *);
            const uint16_t *conv = block_tensor(o, b, "ple_conv1d.weight", &type, &bytes, err, errcap);
            if (!conv) return -1;
            if (type != GG_F16 || bytes != 10240u * 4u * sizeof(uint16_t)) return oracle_error(err, errcap, "PLE convolution tensor type/size mismatch");
            o->ple_conv_weights = malloc(10240u * 4u * sizeof(float));
            if (!o->ple_conv_weights) return oracle_error(err, errcap, "allocate PLE convolution weights");
            for (size_t i = 0; i < 10240u * 4u; i++) o->ple_conv_weights[i] = qsa_h2f(conv[i]);
            w->ple_conv_w = o->ple_conv_weights;
            w->ple_tab = o->ple_table;
            memcpy(w->ple_mult, ple_multipliers, sizeof w->ple_mult);
            memcpy(w->ple_off, ple_head_offsets, sizeof w->ple_off);
            memcpy(w->ple_voc, ple_head_vocab, sizeof w->ple_voc);
            w->ple_absent = PLE_EOS_TOKEN; w->ple_end = PLE_EOS_TOKEN;
        }
#undef SET_BLOCK_TYPED
#undef SET_BLOCK
    }
    dn_hweights_t *head = &o->weights[0];
    head->output_hc_norm = named_tensor(o, "output_hc_norm.weight", NULL, NULL, err, errcap);
    head->output_hc_down = named_tensor(o, "output_hc_down.weight", NULL, NULL, err, errcap);
    head->output_hc_up = named_tensor(o, "output_hc_up.weight", NULL, NULL, err, errcap);
    head->output = named_tensor(o, "output.weight", NULL, NULL, err, errcap);
    head->token_embd = named_tensor(o, "token_embd.weight", NULL, NULL, err, errcap);
    if (!head->output_hc_norm || !head->output_hc_down || !head->output_hc_up || !head->output || !head->token_embd)
        return -1;
    return 0;
}

static int identity_matches_config(const model_oracle_config_t *cfg, const model_oracle_identity_t *id,
                                   char *err, size_t errcap) {
    EVP_MD_CTX *ctx = NULL;
    uint8_t ids_hash[32];
    int rc = -1;
    if (verify_production_topology(cfg, err, errcap) < 0 || production_environment_matches(cfg, err, errcap) < 0) return -1;
    if (strcmp(cfg->source_revision, id->source_revision) || cfg->ple_data_offset != id->ple_data_offset ||
        cfg->token_count != id->token_count || cfg->flash_nblk != id->flash_nblk ||
        host_u32_bits(cfg->epsilon) != id->epsilon_bits ||
        memcmp(cfg->physical_dies, id->physical_dies, sizeof id->physical_dies) ||
        memcmp(cfg->shares, id->shares, sizeof id->shares) ||
        memcmp(cfg->attention_map, id->attention_map, sizeof id->attention_map))
        return oracle_error(err, errcap, "oracle config disagrees with verified identity");
    if (hash_begin(&ctx, err, errcap) < 0) goto done;
    for (uint64_t i = 0; i < cfg->token_count; i++) {
        if (cfg->teacher_ids[i] > 248319u) { oracle_error(err, errcap, "teacher ID exceeds vocabulary"); goto done; }
        if (hash_update_u32(ctx, cfg->teacher_ids[i], err, errcap) < 0) goto done;
    }
    if (hash_finish(ctx, ids_hash, err, errcap) < 0) { ctx = NULL; goto done; }
    ctx = NULL;
    if (memcmp(ids_hash, id->teacher_ids_sha256, 32)) { oracle_error(err, errcap, "teacher IDs differ from verified identity"); goto done; }
    rc = 0;
done:
    if (ctx) EVP_MD_CTX_free(ctx);
    return rc;
}

static int oracle_copy_config(model_oracle_t *o, const model_oracle_config_t *cfg,
                              const model_oracle_identity_t *id, char *err, size_t errcap) {
    const char *src[9] = {
        cfg->source_root, cfg->source_sha256_manifest, cfg->source_revision,
        cfg->tile_cache_dir, cfg->tile_cache_manifest, cfg->tile_cache_sha256_manifest,
        cfg->checkpoint_sha256_manifest, cfg->primary_checkpoint_path, cfg->ple_shard_path
    };
    const char **dst[9] = {
        &o->config.source_root, &o->config.source_sha256_manifest, &o->config.source_revision,
        &o->config.tile_cache_dir, &o->config.tile_cache_manifest, &o->config.tile_cache_sha256_manifest,
        &o->config.checkpoint_sha256_manifest, &o->config.primary_checkpoint_path, &o->config.ple_shard_path
    };
    o->config = *cfg;
    for (unsigned i = 0; i < 9; i++) {
        o->owned_strings[i] = strdup(src[i]);
        if (!o->owned_strings[i]) return oracle_error(err, errcap, "copy model oracle input path");
        *dst[i] = o->owned_strings[i];
    }
    o->ids = malloc((size_t)cfg->token_count * sizeof *o->ids);
    if (!o->ids) return oracle_error(err, errcap, "allocate model oracle teacher IDs");
    memcpy(o->ids, cfg->teacher_ids, (size_t)cfg->token_count * sizeof *o->ids);
    o->config.teacher_ids = o->ids;
    o->identity = *id;
    return 0;
}

static int oracle_allocate_state(model_oracle_t *o, char *err, size_t errcap) {
    uint64_t t = o->config.token_count;
    if (t > UINT_MAX - 127u) return oracle_error(err, errcap, "token count exceeds QSA cache index range");
    o->max_cache_positions = ((uint32_t)t + 127u) & ~127u;
    o->compact_positions = ((size_t)4u * (o->config.flash_nblk + 1u) + 2048u + 31u) / 32u * 32u;
    o->kcache_floats = ATTN_KC_FLOATS(o->compact_positions);
    o->vcache_floats = o->compact_positions * 512u;
    o->buffers = calloc(1, sizeof *o->buffers);
    if (!o->buffers) return oracle_error(err, errcap, "allocate scalar model buffers");
    o->buffers->logits = malloc(248320u * sizeof(float));
    if (!o->buffers->logits) return oracle_error(err, errcap, "allocate full scalar logits");
    o->buffers->ids = o->ids;
    for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
        if (BLK_IS_ATTN(b)) {
            if (qsa_cache_allocate(o, (unsigned)b, err, errcap) < 0) return -1;
        } else {
            o->state[b].dn_state = calloc(48u * 16384u, sizeof(float));
            o->state[b].conv_state = calloc(4u * 10240u, sizeof(float));
            if (!o->state[b].dn_state || !o->state[b].conv_state)
                return oracle_error(err, errcap, "allocate DeltaNet state for block %u", b);
        }
        if (b == 1) {
            o->state[b].ple_state = calloc(10u * 10240u, sizeof(float));
            if (!o->state[b].ple_state) return oracle_error(err, errcap, "allocate PLE state");
        }
    }
    return 0;
}

static void oracle_free_qsa(model_qsa_t *q) {
    if (!q) return;
    free(q->kq); free(q->vq); free(q->ks); free(q->vs); free(q->pool); free(q->pkt);
    free(q->score); free(q->sel); free(q->kc); free(q->vc); free(q);
}

void model_oracle_close(model_oracle_t *o) {
    if (!o) return;
    if (o->tensors) for (size_t i = 0; i < o->tensor_count; i++) if (o->tensors[i].mapping) munmap(o->tensors[i].mapping, o->tensors[i].mapping_bytes);
    free(o->tensors);
    if (o->ple_mapping) munmap(o->ple_mapping, o->ple_mapping_bytes);
    free(o->ple_conv_weights);
    for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
        free(o->state[b].dn_state); free(o->state[b].conv_state); free(o->state[b].ple_state);
        oracle_free_qsa(o->qsa[b]);
    }
    if (o->buffers) { free(o->buffers->logits); free(o->buffers); }
    free(o->ids);
    for (unsigned i = 0; i < 9; i++) free(o->owned_strings[i]);
    free(o);
}

static int oracle_record_header_write(FILE *f, const model_oracle_identity_t *identity,
                                      char *err, size_t errcap) {
    if (write_all(f, ORACLE_MAGIC, 8, err, errcap) < 0 ||
        write_u32(f, MODEL_ORACLE_FORMAT_VERSION, err, errcap) < 0 || identity_write(f, identity, err, errcap) < 0)
        return -1;
    return 0;
}

static int oracle_record_header_check(FILE *f, const model_oracle_identity_t *expected,
                                      char *err, size_t errcap) {
    char magic[8]; uint32_t version; model_oracle_identity_t actual;
    if (fseek(f, 0, SEEK_SET) < 0) return oracle_error(err, errcap, "seek oracle record stream: %s", strerror(errno));
    if (read_all(f, magic, sizeof magic, err, errcap) < 0 || memcmp(magic, ORACLE_MAGIC, sizeof magic))
        return oracle_error(err, errcap, "invalid oracle record file header");
    if (read_u32(f, &version, err, errcap) < 0 || version != MODEL_ORACLE_FORMAT_VERSION)
        return oracle_error(err, errcap, "unsupported oracle record version");
    if (identity_read(f, &actual, err, errcap) < 0) return -1;
    if (!identity_equal(&actual, expected)) return oracle_error(err, errcap, "oracle record header identity mismatch");
    return 0;
}

static int checkpoint_write_bytes(FILE *f, const void *data, size_t bytes, char *err, size_t errcap) {
    return write_all(f, data, bytes, err, errcap);
}

static int checkpoint_read_bytes(FILE *f, void *data, size_t bytes, char *err, size_t errcap) {
    return read_all(f, data, bytes, err, errcap);
}

static int checkpoint_write_state(FILE *f, const model_oracle_t *o, char *err, size_t errcap) {
    for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
        if (o->state[b].dn_state && checkpoint_write_bytes(f, o->state[b].dn_state, 48u * 16384u * sizeof(float), err, errcap) < 0) return -1;
        if (o->state[b].conv_state && checkpoint_write_bytes(f, o->state[b].conv_state, 4u * 10240u * sizeof(float), err, errcap) < 0) return -1;
        if (o->state[b].ple_state && checkpoint_write_bytes(f, o->state[b].ple_state, 10u * 10240u * sizeof(float), err, errcap) < 0) return -1;
        if (o->qsa[b]) {
            const model_qsa_t *q = o->qsa[b];
            size_t m = o->max_cache_positions;
            if (checkpoint_write_bytes(f, q->kq, m * 512u, err, errcap) < 0 ||
                checkpoint_write_bytes(f, q->vq, m * 512u, err, errcap) < 0 ||
                checkpoint_write_bytes(f, q->ks, m * 16u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_write_bytes(f, q->vs, m * 16u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_write_bytes(f, q->pool, m / 4u * 128u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_write_bytes(f, q->ring, sizeof q->ring, err, errcap) < 0) return -1;
        }
    }
    if (checkpoint_write_bytes(f, o->buffers->S, sizeof o->buffers->S, err, errcap) < 0) return -1;
    return 0;
}

static int checkpoint_read_state(FILE *f, model_oracle_t *o, char *err, size_t errcap) {
    for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
        if (o->state[b].dn_state && checkpoint_read_bytes(f, o->state[b].dn_state, 48u * 16384u * sizeof(float), err, errcap) < 0) return -1;
        if (o->state[b].conv_state && checkpoint_read_bytes(f, o->state[b].conv_state, 4u * 10240u * sizeof(float), err, errcap) < 0) return -1;
        if (o->state[b].ple_state && checkpoint_read_bytes(f, o->state[b].ple_state, 10u * 10240u * sizeof(float), err, errcap) < 0) return -1;
        if (o->qsa[b]) {
            model_qsa_t *q = o->qsa[b];
            size_t m = o->max_cache_positions;
            if (checkpoint_read_bytes(f, q->kq, m * 512u, err, errcap) < 0 ||
                checkpoint_read_bytes(f, q->vq, m * 512u, err, errcap) < 0 ||
                checkpoint_read_bytes(f, q->ks, m * 16u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_read_bytes(f, q->vs, m * 16u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_read_bytes(f, q->pool, m / 4u * 128u * sizeof(uint16_t), err, errcap) < 0 ||
                checkpoint_read_bytes(f, q->ring, sizeof q->ring, err, errcap) < 0) return -1;
        }
    }
    if (checkpoint_read_bytes(f, o->buffers->S, sizeof o->buffers->S, err, errcap) < 0) return -1;
    return 0;
}

int model_oracle_checkpoint_write(const model_oracle_t *oracle, FILE *checkpoint_out,
                                  char *error, size_t error_capacity) {
    model_oracle_t *o = (model_oracle_t *)oracle;
    uint32_t endian = UINT32_C(0x01020304);
    int records_fd;
    if (!o || !checkpoint_out) return oracle_error(error, error_capacity, "checkpoint requires oracle and output stream");
    if (fflush(o->records) == EOF) return oracle_error(error, error_capacity, "flush oracle records: %s", strerror(errno));
    records_fd = fileno(o->records);
    if (records_fd < 0) return oracle_error(error, error_capacity, "get oracle record descriptor: %s", strerror(errno));
    for (;;) {
        if (fsync(records_fd) == 0) break;
        if (errno != EINTR) return oracle_error(error, error_capacity, "sync oracle records: %s", strerror(errno));
    }
    long end = ftell(o->records);
    if (end < 0) return oracle_error(error, error_capacity, "tell oracle record position: %s", strerror(errno));
    o->record_offset = (uint64_t)end;
    if (fflush(checkpoint_out) == EOF || ftruncate(fileno(checkpoint_out), 0) < 0 || fseek(checkpoint_out, 0, SEEK_SET) < 0)
        return oracle_error(error, error_capacity, "reset oracle checkpoint stream: %s", strerror(errno));
    if (write_all(checkpoint_out, ORACLE_CHECKPOINT_MAGIC, 8, error, error_capacity) < 0 ||
        write_u32(checkpoint_out, MODEL_ORACLE_FORMAT_VERSION, error, error_capacity) < 0 ||
        write_u32(checkpoint_out, endian, error, error_capacity) < 0 ||
        identity_write(checkpoint_out, &o->identity, error, error_capacity) < 0 ||
        write_u64(checkpoint_out, o->next_position, error, error_capacity) < 0 ||
        write_u64(checkpoint_out, o->record_offset, error, error_capacity) < 0 ||
        checkpoint_write_state(checkpoint_out, o, error, error_capacity) < 0) return -1;
    if (fflush(checkpoint_out) == EOF || fsync(fileno(checkpoint_out)) < 0)
        return oracle_error(error, error_capacity, "flush oracle checkpoint: %s", strerror(errno));
    return 0;
}

static int checkpoint_read(model_oracle_t *o, FILE *checkpoint_in, char *err, size_t errcap) {
    char magic[8]; uint32_t version, endian; model_oracle_identity_t id;
    uint64_t next, record_offset;
    if (fseek(checkpoint_in, 0, SEEK_SET) < 0) return oracle_error(err, errcap, "seek oracle checkpoint: %s", strerror(errno));
    if (read_all(checkpoint_in, magic, 8, err, errcap) < 0 || memcmp(magic, ORACLE_CHECKPOINT_MAGIC, 8))
        return oracle_error(err, errcap, "invalid oracle checkpoint magic");
    if (read_u32(checkpoint_in, &version, err, errcap) < 0 || version != MODEL_ORACLE_FORMAT_VERSION ||
        read_u32(checkpoint_in, &endian, err, errcap) < 0 || endian != UINT32_C(0x01020304))
        return oracle_error(err, errcap, "unsupported oracle checkpoint version or byte order");
    if (identity_read(checkpoint_in, &id, err, errcap) < 0) return -1;
    if (!identity_equal(&id, &o->identity)) return oracle_error(err, errcap, "oracle checkpoint identity mismatch");
    if (read_u64(checkpoint_in, &next, err, errcap) < 0 || read_u64(checkpoint_in, &record_offset, err, errcap) < 0) return -1;
    if (next > o->identity.token_count) return oracle_error(err, errcap, "checkpoint position exceeds token_count");
    if (checkpoint_read_state(checkpoint_in, o, err, errcap) < 0) return -1;
    if (fgetc(checkpoint_in) != EOF || ferror(checkpoint_in)) return oracle_error(err, errcap, "trailing or unreadable checkpoint bytes");
    o->next_position = next;
    o->record_offset = record_offset;
    return 0;
}

int model_oracle_open(model_oracle_t **out, const model_oracle_config_t *config,
                      const model_oracle_identity_t *identity, FILE *record_stream,
                      FILE *checkpoint_in, char *error, size_t error_capacity) {
    model_oracle_t *o;
    long record_length;
    uint8_t endian_probe[4];
    if (!out || !config || !identity || !record_stream) return oracle_error(error, error_capacity, "model_oracle_open requires config, identity and record stream");
    *out = NULL;
    put_u32(endian_probe, UINT32_C(0x01020304));
    if (memcmp(endian_probe, "\x04\x03\x02\x01", 4)) return oracle_error(error, error_capacity, "oracle requires a little-endian host");
    if (identity_matches_config(config, identity, error, error_capacity) < 0) return -1;
    o = calloc(1, sizeof *o);
    if (!o) return oracle_error(error, error_capacity, "allocate model oracle state");
    if (oracle_copy_config(o, config, identity, error, error_capacity) < 0) goto fail;
    o->records = record_stream;
    if (oracle_allocate_state(o, error, error_capacity) < 0 || map_tile_cache(o, error, error_capacity) < 0 ||
        map_ple_table(o, error, error_capacity) < 0 || load_weights(o, error, error_capacity) < 0) goto fail;
    if (fseek(record_stream, 0, SEEK_END) < 0 || (record_length = ftell(record_stream)) < 0) {
        oracle_error(error, error_capacity, "seek oracle record stream: %s", strerror(errno)); goto fail;
    }
    if (!checkpoint_in) {
        if (record_length != 0) { oracle_error(error, error_capacity, "new oracle record stream is not empty"); goto fail; }
        if (oracle_record_header_write(record_stream, identity, error, error_capacity) < 0 || fflush(record_stream) == EOF) goto fail;
        long pos = ftell(record_stream);
        if (pos < 0) { oracle_error(error, error_capacity, "tell oracle header position"); goto fail; }
        o->record_offset = (uint64_t)pos;
    } else {
        if (record_length <= 0 || oracle_record_header_check(record_stream, identity, error, error_capacity) < 0) goto fail;
        if (checkpoint_read(o, checkpoint_in, error, error_capacity) < 0) goto fail;
        if ((uint64_t)record_length < o->record_offset) { oracle_error(error, error_capacity, "record stream is shorter than checkpoint receipt"); goto fail; }
        if ((uint64_t)record_length > o->record_offset) {
            if (ftruncate(fileno(record_stream), (off_t)o->record_offset) < 0 || fsync(fileno(record_stream)) < 0) {
                oracle_error(error, error_capacity, "truncate uncheckpointed oracle records: %s", strerror(errno)); goto fail;
            }
        }
        if (fseek(record_stream, 0, SEEK_END) < 0) { oracle_error(error, error_capacity, "seek append point in oracle record stream"); goto fail; }
    }
    *out = o;
    return 0;
fail:
    model_oracle_close(o);
    return -1;
}

static int oracle_error(char *dst, size_t cap, const char *fmt, ...) {
    va_list ap;
    if (dst && cap) {
        va_start(ap, fmt);
        vsnprintf(dst, cap, fmt, ap);
        va_end(ap);
    }
    return -1;
}

static uint32_t host_u32_bits(float f) {
    uint32_t v;
    memcpy(&v, &f, sizeof v);
    return v;
}

static uint64_t stat_mtime_ns(const struct stat *st) {
    return (uint64_t)st->st_mtim.tv_sec * UINT64_C(1000000000) + (uint64_t)st->st_mtim.tv_nsec;
}

static int hash_begin(EVP_MD_CTX **ctx, char *err, size_t errcap) {
    *ctx = EVP_MD_CTX_new();
    if (!*ctx) return oracle_error(err, errcap, "EVP_MD_CTX_new failed");
    if (EVP_DigestInit_ex(*ctx, EVP_sha256(), NULL) != 1) {
        EVP_MD_CTX_free(*ctx); *ctx = NULL;
        return oracle_error(err, errcap, "EVP_DigestInit_ex(SHA256) failed");
    }
    return 0;
}

static int hash_bytes(EVP_MD_CTX *ctx, const void *data, size_t bytes, char *err, size_t errcap) {
    if (bytes && EVP_DigestUpdate(ctx, data, bytes) != 1)
        return oracle_error(err, errcap, "EVP_DigestUpdate failed");
    return 0;
}

static int hash_finish(EVP_MD_CTX *ctx, uint8_t out[MODEL_ORACLE_SHA256_BYTES], char *err, size_t errcap) {
    unsigned int n = 0;
    int ok = EVP_DigestFinal_ex(ctx, out, &n) == 1 && n == MODEL_ORACLE_SHA256_BYTES;
    EVP_MD_CTX_free(ctx);
    if (!ok) return oracle_error(err, errcap, "EVP_DigestFinal_ex(SHA256) failed");
    return 0;
}

static int sha256_memory(const void *data, size_t bytes, uint8_t out[MODEL_ORACLE_SHA256_BYTES], char *err, size_t errcap) {
    EVP_MD_CTX *ctx = NULL;
    if (hash_begin(&ctx, err, errcap) < 0) return -1;
    if (hash_bytes(ctx, data, bytes, err, errcap) < 0) { EVP_MD_CTX_free(ctx); return -1; }
    return hash_finish(ctx, out, err, errcap);
}

static int sha256_file(const char *path, uint8_t out[MODEL_ORACLE_SHA256_BYTES],
                       struct stat *before, char *err, size_t errcap) {
    int fd = -1;
    EVP_MD_CTX *ctx = NULL;
    uint8_t *buffer = NULL;
    struct stat after;
    int rc = -1;
    fd = open(path, O_RDONLY);
    if (fd < 0) return oracle_error(err, errcap, "open %s: %s", path, strerror(errno));
    if (fstat(fd, before) < 0) { oracle_error(err, errcap, "fstat %s: %s", path, strerror(errno)); goto done; }
    if (!S_ISREG(before->st_mode) || before->st_size < 0) {
        oracle_error(err, errcap, "%s is not a regular file", path); goto done;
    }
    if (hash_begin(&ctx, err, errcap) < 0) goto done;
    buffer = malloc(ORACLE_IO_BYTES);
    if (!buffer) { oracle_error(err, errcap, "allocate file hash buffer"); goto done; }
    for (;;) {
        ssize_t n = read(fd, buffer, ORACLE_IO_BYTES);
        if (n < 0 && errno == EINTR) continue;
        if (n < 0) { oracle_error(err, errcap, "read %s: %s", path, strerror(errno)); goto done; }
        if (!n) break;
        if (hash_bytes(ctx, buffer, (size_t)n, err, errcap) < 0) goto done;
    }
    if (fstat(fd, &after) < 0) { oracle_error(err, errcap, "restat %s: %s", path, strerror(errno)); goto done; }
    if (after.st_size != before->st_size || stat_mtime_ns(&after) != stat_mtime_ns(before)) {
        oracle_error(err, errcap, "%s changed while it was being hashed", path); goto done;
    }
    if (hash_finish(ctx, out, err, errcap) < 0) { ctx = NULL; goto done; }
    ctx = NULL;
    rc = 0;
done:
    if (ctx) EVP_MD_CTX_free(ctx);
    free(buffer);
    if (fd >= 0) close(fd);
    return rc;
}

static void put_u32(uint8_t out[4], uint32_t v) {
    out[0] = (uint8_t)v; out[1] = (uint8_t)(v >> 8); out[2] = (uint8_t)(v >> 16); out[3] = (uint8_t)(v >> 24);
}

static void put_u64(uint8_t out[8], uint64_t v) {
    for (unsigned i = 0; i < 8; i++) out[i] = (uint8_t)(v >> (8u * i));
}

static uint32_t get_u32(const uint8_t in[4]) {
    return (uint32_t)in[0] | (uint32_t)in[1] << 8 | (uint32_t)in[2] << 16 | (uint32_t)in[3] << 24;
}

static uint64_t get_u64(const uint8_t in[8]) {
    uint64_t v = 0;
    for (unsigned i = 0; i < 8; i++) v |= (uint64_t)in[i] << (8u * i);
    return v;
}

static int write_all(FILE *f, const void *data, size_t bytes, char *err, size_t errcap) {
    if (bytes && fwrite(data, 1, bytes, f) != bytes)
        return oracle_error(err, errcap, "write failed: %s", strerror(errno ? errno : EIO));
    return 0;
}

static int read_all(FILE *f, void *data, size_t bytes, char *err, size_t errcap) {
    if (bytes && fread(data, 1, bytes, f) != bytes) {
        if (feof(f)) return oracle_error(err, errcap, "truncated oracle stream");
        return oracle_error(err, errcap, "read failed: %s", strerror(errno ? errno : EIO));
    }
    return 0;
}

static int write_u32(FILE *f, uint32_t v, char *err, size_t errcap) {
    uint8_t b[4]; put_u32(b, v); return write_all(f, b, sizeof b, err, errcap);
}

static int write_u64(FILE *f, uint64_t v, char *err, size_t errcap) {
    uint8_t b[8]; put_u64(b, v); return write_all(f, b, sizeof b, err, errcap);
}

static int read_u32(FILE *f, uint32_t *v, char *err, size_t errcap) {
    uint8_t b[4]; if (read_all(f, b, sizeof b, err, errcap) < 0) return -1; *v = get_u32(b); return 0;
}

static int read_u64(FILE *f, uint64_t *v, char *err, size_t errcap) {
    uint8_t b[8]; if (read_all(f, b, sizeof b, err, errcap) < 0) return -1; *v = get_u64(b); return 0;
}

static int identity_write(FILE *f, const model_oracle_identity_t *id, char *err, size_t errcap) {
    if (write_all(f, id->source_revision, sizeof id->source_revision, err, errcap) < 0 ||
        write_all(f, id->source_content_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->tile_manifest_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->tile_content_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->checkpoint_content_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->primary_checkpoint_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->ple_checkpoint_sha256, 32, err, errcap) < 0 ||
        write_all(f, id->teacher_ids_sha256, 32, err, errcap) < 0 ||
        write_u64(f, id->ple_data_offset, err, errcap) < 0 ||
        write_u64(f, id->token_count, err, errcap) < 0 ||
        write_u32(f, id->flash_nblk, err, errcap) < 0 ||
        write_u32(f, id->epsilon_bits, err, errcap) < 0) return -1;
    for (unsigned i = 0; i < MODEL_ORACLE_DIES; i++) if (write_u32(f, id->physical_dies[i], err, errcap) < 0) return -1;
    for (unsigned i = 0; i < MODEL_ORACLE_DIES; i++) if (write_u32(f, id->shares[i], err, errcap) < 0) return -1;
    for (unsigned i = 0; i < MODEL_ORACLE_ATTN_LAYERS; i++) if (write_u32(f, (uint32_t)id->attention_map[i], err, errcap) < 0) return -1;
    return 0;
}

static int identity_read(FILE *f, model_oracle_identity_t *id, char *err, size_t errcap) {
    uint32_t v;
    memset(id, 0, sizeof *id);
    if (read_all(f, id->source_revision, sizeof id->source_revision, err, errcap) < 0 ||
        read_all(f, id->source_content_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->tile_manifest_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->tile_content_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->checkpoint_content_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->primary_checkpoint_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->ple_checkpoint_sha256, 32, err, errcap) < 0 ||
        read_all(f, id->teacher_ids_sha256, 32, err, errcap) < 0 ||
        read_u64(f, &id->ple_data_offset, err, errcap) < 0 ||
        read_u64(f, &id->token_count, err, errcap) < 0 ||
        read_u32(f, &id->flash_nblk, err, errcap) < 0 ||
        read_u32(f, &id->epsilon_bits, err, errcap) < 0) return -1;
    for (unsigned i = 0; i < MODEL_ORACLE_DIES; i++) { if (read_u32(f, &id->physical_dies[i], err, errcap) < 0) return -1; }
    for (unsigned i = 0; i < MODEL_ORACLE_DIES; i++) { if (read_u32(f, &id->shares[i], err, errcap) < 0) return -1; }
    for (unsigned i = 0; i < MODEL_ORACLE_ATTN_LAYERS; i++) { if (read_u32(f, &v, err, errcap) < 0) return -1; id->attention_map[i] = (int32_t)v; }
    return 0;
}

static int identity_equal(const model_oracle_identity_t *a, const model_oracle_identity_t *b) {
    return !memcmp(a->source_revision, b->source_revision, sizeof a->source_revision) &&
        !memcmp(a->source_content_sha256, b->source_content_sha256, 32) &&
        !memcmp(a->tile_manifest_sha256, b->tile_manifest_sha256, 32) &&
        !memcmp(a->tile_content_sha256, b->tile_content_sha256, 32) &&
        !memcmp(a->checkpoint_content_sha256, b->checkpoint_content_sha256, 32) &&
        !memcmp(a->primary_checkpoint_sha256, b->primary_checkpoint_sha256, 32) &&
        !memcmp(a->ple_checkpoint_sha256, b->ple_checkpoint_sha256, 32) &&
        !memcmp(a->teacher_ids_sha256, b->teacher_ids_sha256, 32) &&
        a->ple_data_offset == b->ple_data_offset && a->token_count == b->token_count && a->flash_nblk == b->flash_nblk &&
        a->epsilon_bits == b->epsilon_bits &&
        !memcmp(a->physical_dies, b->physical_dies, sizeof a->physical_dies) &&
        !memcmp(a->shares, b->shares, sizeof a->shares) &&
        !memcmp(a->attention_map, b->attention_map, sizeof a->attention_map);
}

static void set_dims(uint64_t d[4], uint64_t a, uint64_t b, uint64_t c, uint64_t e) {
    d[0] = a; d[1] = b; d[2] = c; d[3] = e;
}

static int builder_move(record_builder_t *dst, record_builder_t *src, char *err, size_t errcap) {
    if (src->count > ORACLE_MAX_ITEMS - dst->count) return oracle_error(err, errcap, "oracle position exceeds item limit");
    uint32_t need = dst->count + src->count;
    if (need > dst->capacity) {
        uint32_t next = dst->capacity ? dst->capacity : 64;
        while (next < need) next *= 2;
        if (next > ORACLE_MAX_ITEMS) next = ORACLE_MAX_ITEMS;
        owned_item_t *p = realloc(dst->items, (size_t)next * sizeof *p);
        if (!p) return oracle_error(err, errcap, "grow oracle record item list");
        dst->items = p; dst->capacity = next;
    }
    memcpy(dst->items + dst->count, src->items, (size_t)src->count * sizeof *src->items);
    dst->count += src->count;
    src->count = 0;
    return 0;
}

static int add_layer_item(model_oracle_t *o, record_builder_t *b, const char *label, int layer,
                          int die, uint32_t value_type, uint32_t rank, const uint64_t dims[4],
                          const void *bytes, uint64_t nbytes, int raw, char *err, size_t errcap) {
    uint32_t logical = die < 0 ? UINT32_MAX : (uint32_t)die;
    uint32_t physical = die < 0 ? UINT32_MAX : o->config.physical_dies[die];
    return builder_add(b, label, layer < 0 ? UINT32_MAX : (uint32_t)layer, logical, physical,
                       value_type, rank, dims, bytes, nbytes, raw, err, errcap);
}

static int add_formatted_item(model_oracle_t *o, record_builder_t *b, const char *format,
                              int layer, int die, uint32_t value_type, uint32_t rank,
                              const uint64_t dims[4], const void *bytes, uint64_t nbytes,
                              int raw, char *err, size_t errcap) {
    char label[MODEL_ORACLE_LABEL_BYTES];
    int n = snprintf(label, sizeof label, format, layer, die);
    if (n < 0 || (size_t)n >= sizeof label) return oracle_error(err, errcap, "oracle comparator label too long");
    return add_layer_item(o, b, label, layer, die, value_type, rank, dims, bytes, nbytes, raw, err, errcap);
}

typedef struct {
    int h0, hn, klo, nk, y0, yn, o0, on;
} oracle_split_t;

static void oracle_split(oracle_split_t *s, int d, int blk) {
    s->h0 = (int)split_bound(48, d, MODEL_ORACLE_DIES);
    s->hn = (int)split_bound(48, d + 1, MODEL_ORACLE_DIES) - s->h0;
    s->klo = s->h0 % 16;
    s->nk = s->hn < 16 ? s->hn : 16;
    s->y0 = (int)split_bound(2560, d, MODEL_ORACLE_DIES);
    s->yn = (int)split_bound(2560, d + 1, MODEL_ORACLE_DIES) - s->y0;
    s->o0 = (int)split_bound(248320, d, MODEL_ORACLE_DIES);
    s->on = (int)split_bound(248320, d + 1, MODEL_ORACLE_DIES) - s->o0;
    (void)blk;
}

static int add_qsa_items(model_oracle_t *o, record_builder_t *b, int layer, unsigned t,
                         char *err, size_t errcap) {
    model_qsa_t *q = o->qsa[layer];
    unsigned nv = (t + 1u) / 4u;
    unsigned nkeys = qsa_nkeys(t, o->config.flash_nblk);
    uint64_t d1[4], d2[4];
    int die = attn_die_of(layer);
    size_t ring_bytes = (size_t)4 * 128 * ((t & 3u) + 1u) * sizeof(float);
    set_dims(d1, 512, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.kq", layer, die, MODEL_ORACLE_I8, 1, d1,
                           q->kq + (size_t)t * 512, 512, 0, err, errcap) < 0 ||
        add_formatted_item(o, b, "b%d.vq", layer, die, MODEL_ORACLE_I8, 1, d1,
                           q->vq + (size_t)t * 512, 512, 0, err, errcap) < 0) return -1;
    set_dims(d1, 16, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.ks", layer, die, MODEL_ORACLE_F16, 1, d1,
                           q->ks + (size_t)t * 16, 32, 0, err, errcap) < 0 ||
        add_formatted_item(o, b, "b%d.vs", layer, die, MODEL_ORACLE_F16, 1, d1,
                           q->vs + (size_t)t * 16, 32, 0, err, errcap) < 0) return -1;
    set_dims(d1, 640, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.iqk", layer, die, MODEL_ORACLE_F32, 1, d1, q->iqk, sizeof q->iqk, 0, err, errcap) < 0) return -1;
    set_dims(d1, 512, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.iq", layer, die, MODEL_ORACLE_F32, 1, d1, q->iq, sizeof q->iq, 0, err, errcap) < 0) return -1;
    set_dims(d1, 128u * ((t & 3u) + 1u), 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.ring", layer, die, MODEL_ORACLE_F32, 1, d1, q->ring, ring_bytes, 0, err, errcap) < 0) return -1;
    if ((t & 3u) == 3u) {
        unsigned block = t / 4u;
        set_dims(d1, 128, 0, 0, 0);
        if (add_formatted_item(o, b, "b%d.pool", layer, die, MODEL_ORACLE_F16, 1, d1,
                               q->pool + (size_t)block * 128u, 128u * sizeof(uint16_t), 0, err, errcap) < 0) return -1;
    }
    if (nv > o->config.flash_nblk) {
        set_dims(d1, nv, 0, 0, 0);
        if (add_formatted_item(o, b, "b%d.scores", layer, die, MODEL_ORACLE_F32, 1, d1,
                               q->score, (size_t)nv * sizeof(float), 0, err, errcap) < 0) return -1;
        set_dims(d1, o->config.flash_nblk, 0, 0, 0);
        uint8_t *ids_le = malloc((size_t)o->config.flash_nblk * sizeof(uint32_t));
        if (!ids_le) return oracle_error(err, errcap, "allocate selected-ID bytes for block %d", layer);
        for (uint32_t i = 0; i < o->config.flash_nblk; i++) put_u32(ids_le + 4u * i, q->sel[i]);
        if (add_formatted_item(o, b, "b%d.selected_ids", layer, die, MODEL_ORACLE_U32, 1, d1,
                               ids_le, (size_t)o->config.flash_nblk * sizeof(uint32_t), 1, err, errcap) < 0) { free(ids_le); return -1; }
        free(ids_le);
    }
    uint8_t *canonical_k = malloc((size_t)nkeys * 512u * sizeof(float));
    uint8_t *canonical_v = malloc((size_t)nkeys * 512u * sizeof(float));
    if (!canonical_k || !canonical_v) { free(canonical_k); free(canonical_v); return oracle_error(err, errcap, "allocate canonical gathered K/V for block %d", layer); }
    for (unsigned slot = 0; slot < nkeys; slot++) for (int kv = 0; kv < 2; kv++) for (int d = 0; d < 256; d++) {
        size_t ix = (size_t)slot * 512u + (size_t)kv * 256u + d;
        put_u32(canonical_k + 4 * ix, host_u32_bits(q->kc[attn_kidx(slot, kv, d)]));
        put_u32(canonical_v + 4 * ix, host_u32_bits(q->vc[ix]));
    }
    set_dims(d2, nkeys, 2, 256, 0);
    if (add_formatted_item(o, b, "b%d.gather_k", layer, die, MODEL_ORACLE_F32, 3, d2,
                           canonical_k, (size_t)nkeys * 512u * sizeof(float), 0, err, errcap) < 0) { free(canonical_k); free(canonical_v); return -1; }
    if (add_formatted_item(o, b, "b%d.gather_v", layer, die, MODEL_ORACLE_F32, 3, d2,
                           canonical_v, (size_t)nkeys * 512u * sizeof(float), 0, err, errcap) < 0) { free(canonical_k); free(canonical_v); return -1; }
    free(canonical_k); free(canonical_v);
    set_dims(d1, 6144, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.qo", layer, die, MODEL_ORACLE_F32, 1, d1, q->qo, sizeof q->qo, 0, err, errcap) < 0) return -1;
    set_dims(d1, 24u * 8u, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.pm", layer, die, MODEL_ORACLE_F32, 1, d1, q->pm, sizeof q->pm, 0, err, errcap) < 0 ||
        add_formatted_item(o, b, "b%d.pd", layer, die, MODEL_ORACLE_F32, 1, d1, q->pd, sizeof q->pd, 0, err, errcap) < 0) return -1;
    set_dims(d1, 24u * 8u * 256u, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.po", layer, die, MODEL_ORACLE_F32, 1, d1, q->po, sizeof q->po, 0, err, errcap) < 0) return -1;
    set_dims(d1, 6144, 0, 0, 0);
    if (add_formatted_item(o, b, "b%d.aout", layer, die, MODEL_ORACLE_F32, 1, d1, q->aout, sizeof q->aout, 0, err, errcap) < 0) return -1;
    return 0;
}

static int add_recurrent_items(model_oracle_t *o, record_builder_t *b, int layer, int die,
                               char *err, size_t errcap) {
    oracle_split_t s;
    uint64_t dims[4];
    oracle_split(&s, die, layer);
    if (!BLK_IS_ATTN(layer)) {
        char label[MODEL_ORACLE_LABEL_BYTES];
        int n = snprintf(label, sizeof label, "b%d.dn_state.d%d", layer, die);
        if (n < 0 || (size_t)n >= sizeof label) return oracle_error(err, errcap, "state comparator label too long");
        set_dims(dims, (uint64_t)s.hn * 16384u, 0, 0, 0);
        if (add_layer_item(o, b, label, layer, die, MODEL_ORACLE_F32, 1, dims,
                           o->state[layer].dn_state + (size_t)s.h0 * 16384u,
                           (size_t)s.hn * 16384u * sizeof(float), 0, err, errcap) < 0) return -1;
        unsigned nloc = 128u * (2u * (unsigned)s.nk + (unsigned)s.hn);
        float *packed = malloc((size_t)4 * nloc * sizeof(float));
        if (!packed) return oracle_error(err, errcap, "allocate packed convolution slice");
        for (unsigned slot = 0; slot < 4; slot++) {
            for (int j = 0; j < 2 * s.nk + s.hn; j++) {
                unsigned source;
                if (j < s.nk) source = 128u * (unsigned)((s.klo + j) % 16);
                else if (j < 2 * s.nk) source = 2048u + 128u * (unsigned)((s.klo + j - s.nk) % 16);
                else source = 4096u + 128u * (unsigned)(s.h0 + j - 2 * s.nk);
                memcpy(packed + (size_t)slot * nloc + 128u * (unsigned)j,
                       o->state[layer].conv_state + (size_t)slot * 10240u + source, 128u * sizeof(float));
            }
        }
        n = snprintf(label, sizeof label, "b%d.conv_state.d%d", layer, die);
        if (n < 0 || (size_t)n >= sizeof label) { free(packed); return oracle_error(err, errcap, "conv comparator label too long"); }
        set_dims(dims, (uint64_t)4 * nloc, 0, 0, 0);
        int rc = add_layer_item(o, b, label, layer, die, MODEL_ORACLE_F32, 1, dims,
                                packed, (size_t)4 * nloc * sizeof(float), 0, err, errcap);
        free(packed);
        if (rc < 0) return -1;
    }
    if (layer == 1) {
        char label[MODEL_ORACLE_LABEL_BYTES];
        int n = snprintf(label, sizeof label, "b1.ple_state.d%d", die);
        if (n < 0 || (size_t)n >= sizeof label) return oracle_error(err, errcap, "PLE comparator label too long");
        set_dims(dims, 102400, 0, 0, 0);
        if (add_layer_item(o, b, label, layer, die, MODEL_ORACLE_F32, 1, dims,
                           o->state[layer].ple_state, 10u * 10240u * sizeof(float), 0, err, errcap) < 0) return -1;
    }
    return 0;
}

int model_oracle_run_chunk(model_oracle_t *o, uint32_t max_positions, uint32_t *processed,
                           char *error, size_t error_capacity) {
    uint32_t done = 0;
    if (!o || !processed || !max_positions || max_positions > MODEL_ORACLE_MAX_CHUNK_POSITIONS)
        return oracle_error(error, error_capacity, "chunk size must be 1..%u positions", MODEL_ORACLE_MAX_CHUNK_POSITIONS);
    *processed = 0;
    while (done < max_positions && o->next_position < o->identity.token_count) {
        record_builder_t per_die[MODEL_ORACLE_DIES] = {{0}}, all = {0};
        uint64_t pos = o->next_position;
        uint32_t input_id = o->ids[pos];
        model_oracle_record_t record;
        dn_hbufs_t *hb = o->buffers;
        int failed = 0;
        hb->ids = o->ids;
        embed_gpu(&o->weights[0], hb, input_id);
        for (unsigned b = 0; b < MODEL_ORACLE_BLOCKS; b++) {
            int layer = (int)b;
            hb->dn_state = o->state[b].dn_state;
            hb->qsa = BLK_IS_ATTN(b) ? o->qsa[b] : NULL;
            if (o->state[b].conv_state) memcpy(hb->conv_state, o->state[b].conv_state, sizeof hb->conv_state);
            if (o->state[b].ple_state) memcpy(hb->ple_state, o->state[b].ple_state, sizeof hb->ple_state);
            if (b == 1) ple_prefix_gpu(&o->weights[b], hb, (unsigned)pos, o->config.epsilon);
            if (BLK_IS_ATTN(b)) attn_block_gpu(&o->weights[b], hb, (unsigned)pos, o->config.epsilon, MODEL_ORACLE_DIES, layer, 0);
            else deltanet_block_gpu(&o->weights[b], hb, (unsigned)pos, o->config.epsilon, MODEL_ORACLE_DIES, layer, 0);
            if (o->state[b].conv_state) memcpy(o->state[b].conv_state, hb->conv_state, sizeof hb->conv_state);
            if (o->state[b].ple_state) memcpy(o->state[b].ple_state, hb->ple_state, sizeof hb->ple_state);
            for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
                uint64_t dims[4];
                set_dims(dims, 10240, 0, 0, 0);
                if (add_formatted_item(o, &per_die[d], "b%d.S.d%d", layer, (int)d, MODEL_ORACLE_F32, 1,
                                       dims, hb->S, sizeof hb->S, 0, error, error_capacity) < 0) { failed = 1; goto token_done; }
                if (BLK_IS_ATTN(b) && (int)d == attn_die_of(layer) &&
                    add_qsa_items(o, &per_die[d], layer, (unsigned)pos, error, error_capacity) < 0) { failed = 1; goto token_done; }
                if (add_recurrent_items(o, &per_die[d], layer, (int)d, error, error_capacity) < 0) { failed = 1; goto token_done; }
            }
        }
        head_gpu(&o->weights[0], hb, o->config.epsilon, MODEL_ORACLE_DIES);
        for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
            int die = (int)d;
            oracle_split_t s;
            uint64_t dims[4];
            char label[MODEL_ORACLE_LABEL_BYTES];
            oracle_split(&s, die, 0);
            set_dims(dims, (uint64_t)s.on, 0, 0, 0);
            int n = snprintf(label, sizeof label, "logits.d%d", die);
            if (n < 0 || (size_t)n >= sizeof label ||
                add_layer_item(o, &per_die[d], label, -1, die, MODEL_ORACLE_F32, 1,
                               dims, hb->logits + s.o0, (size_t)s.on * sizeof(float), 0, error, error_capacity) < 0) { failed = 1; goto token_done; }
            if (builder_move(&all, &per_die[d], error, error_capacity) < 0) { failed = 1; goto token_done; }
        }
        {
            uint64_t dims[4];
            set_dims(dims, 10240, 0, 0, 0);
            for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) {
                int die = (int)d;
                char label[MODEL_ORACLE_LABEL_BYTES];
                int n = snprintf(label, sizeof label, "S.%d", die);
                if (n < 0 || (size_t)n >= sizeof label ||
                    add_layer_item(o, &all, label, -1, die, MODEL_ORACLE_F32, 1,
                                   dims, hb->S, sizeof hb->S, 0, error, error_capacity) < 0) { failed = 1; goto token_done; }
                uint8_t token_tuple[8];
                put_u32(token_tuple, hb->tok); put_u32(token_tuple + 4, host_u32_bits(hb->tokmax));
                set_dims(dims, 2, 0, 0, 0);
                n = snprintf(label, sizeof label, "tok.%d", die);
                if (n < 0 || (size_t)n >= sizeof label ||
                    add_layer_item(o, &all, label, -1, die, MODEL_ORACLE_U32, 1,
                                   dims, token_tuple, sizeof token_tuple, 1, error, error_capacity) < 0) { failed = 1; goto token_done; }
            }
            uint32_t score_bits = host_u32_bits(hb->tokmax);
            memset(&record, 0, sizeof record);
            record.position = pos; record.visible_length = pos + 1; record.teacher_id = input_id;
            record.prediction = hb->tok; record.prediction_score_bits = score_bits;
            record.item_count = all.count;
            record.items = NULL; /* record_write serializes the owned item wrappers directly. */
            if (record_write(o->records, &record, &all, error, error_capacity) < 0) { failed = 1; goto token_done; }
            if (fflush(o->records) == EOF) { oracle_error(error, error_capacity, "flush oracle record: %s", strerror(errno)); failed = 1; goto token_done; }
            long offset = ftell(o->records);
            if (offset < 0) { oracle_error(error, error_capacity, "tell oracle record stream: %s", strerror(errno)); failed = 1; goto token_done; }
            o->record_offset = (uint64_t)offset;
        }
        o->next_position++; done++;
token_done:
        for (unsigned d = 0; d < MODEL_ORACLE_DIES; d++) builder_destroy(&per_die[d]);
        builder_destroy(&all);
        if (failed) return -1;
    }
    *processed = done;
    return 0;
}

#ifdef MODEL_ORACLE_STANDALONE
static int cli_u64(const char *s, uint64_t max, uint64_t *out) {
    char *end = NULL;
    unsigned long long v;
    if (!s || !*s) return -1;
    for (const char *p = s; *p; p++) if (*p < '0' || *p > '9') return -1;
    errno = 0; v = strtoull(s, &end, 10);
    if (errno || !end || *end || v > max) return -1;
    *out = (uint64_t)v; return 0;
}

static int cli_physical_ids(const char *s, uint32_t ids[MODEL_ORACLE_DIES]) {
    const char *p = s;
    for (unsigned i = 0; i < MODEL_ORACLE_DIES; i++) {
        char *end = NULL;
        unsigned long value;
        if (!p || !*p) return -1;
        errno = 0; value = strtoul(p, &end, 10);
        if (errno || end == p || value > UINT32_MAX) return -1;
        ids[i] = (uint32_t)value;
        if (i + 1 == MODEL_ORACLE_DIES) return *end ? -1 : 0;
        if (*end != ',') return -1;
        p = end + 1;
    }
    return -1;
}

static int cli_read_ids(const char *path, uint32_t **ids_out, uint64_t *count_out) {
    FILE *f = fopen(path, "r");
    char word[64];
    uint32_t *ids = NULL;
    size_t count = 0, capacity = 0;
    if (!f) { perror(path); return -1; }
    while (fscanf(f, "%63s", word) == 1) {
        uint64_t id;
        if (cli_u64(word, 248319u, &id) < 0) { fprintf(stderr, "%s: invalid teacher ID\n", path); free(ids); fclose(f); return -1; }
        if (count == UINT_MAX) { fprintf(stderr, "%s: too many teacher IDs\n", path); free(ids); fclose(f); return -1; }
        if (count == capacity) {
            size_t next = capacity ? capacity * 2 : 256;
            uint32_t *p = realloc(ids, next * sizeof *p);
            if (!p) { perror("allocate teacher IDs"); free(ids); fclose(f); return -1; }
            ids = p; capacity = next;
        }
        ids[count++] = (uint32_t)id;
    }
    if (ferror(f)) { perror(path); free(ids); fclose(f); return -1; }
    fclose(f);
    if (!count) { fprintf(stderr, "%s: no teacher IDs\n", path); free(ids); return -1; }
    *ids_out = ids; *count_out = count;
    return 0;
}

static int cli_fail(const char *what, const char *error) {
    fprintf(stderr, "model_oracle: %s%s%s\n", what, error && *error ? ": " : "", error && *error ? error : "");
    return 1;
}

static int sync_parent_directory(const char *path, char *err, size_t errcap) {
    size_t path_len = strlen(path);
    char *parent = malloc(path_len + 2);
    char *slash;
    int fd, rc;
    if (!parent) return oracle_error(err, errcap, "allocate parent directory path");
    memcpy(parent, path, path_len + 1);
    slash = strrchr(parent, '/');
    if (!slash) strcpy(parent, ".");
    else if (slash == parent) parent[1] = '\0';
    else *slash = '\0';
    do { fd = open(parent, O_RDONLY); } while (fd < 0 && errno == EINTR);
    if (fd < 0) {
        int saved_errno = errno;
        free(parent);
        return oracle_error(err, errcap, "open parent directory: %s", strerror(saved_errno));
    }
    free(parent);
    do { rc = fsync(fd); } while (rc < 0 && errno == EINTR);
    if (rc < 0) {
        int saved_errno = errno;
        close(fd);
        return oracle_error(err, errcap, "sync parent directory: %s", strerror(saved_errno));
    }
    if (close(fd) < 0) return oracle_error(err, errcap, "close parent directory: %s", strerror(errno));
    return 0;
}

int main(int argc, char **argv) {
    static const uint32_t production_shares[MODEL_ORACLE_DIES] = {112, 97, 81, 116, 97, 97};
    static const int32_t production_map[MODEL_ORACLE_ATTN_LAYERS] = {1, 1, 1, 1, 4, 4, 4, 4, 5, 5, 5, 5};
    model_oracle_config_t cfg;
    model_oracle_identity_t identity;
    model_oracle_t *oracle = NULL;
    uint32_t *ids = NULL, processed = 0, physical[MODEL_ORACLE_DIES];
    uint64_t ple_offset, budget, chunk;
    int full_hash = 0, checkpoint_exists = 0, status = 1;
    FILE *records = NULL, *receipt = NULL, *checkpoint_in = NULL, *checkpoint_tmp = NULL;
    char error[512] = {0};
    char *checkpoint_next = NULL;
    struct stat st;
    if (argc != 18) {
        fprintf(stderr, "usage: %s SOURCE_ROOT SOURCE_SHA256_LIST SOURCE_REV TILE_DIR TILE_MANIFEST TILE_SHA256_LIST CHECKPOINT_SHA256_LIST PRIMARY_GGUF PLE_GGUF PLE_OFFSET IDS_FILE FLASH_NBLK PHYSICAL_DIES_CSV RECORDS CHECKPOINT RECEIPT CHUNK_POSITIONS\n", argv[0]);
        return 2;
    }
    memset(&cfg, 0, sizeof cfg);
    if (cli_u64(argv[10], UINT64_MAX, &ple_offset) < 0 || cli_u64(argv[12], 4096, &budget) < 0 || !budget ||
        cli_physical_ids(argv[13], physical) < 0 || cli_u64(argv[17], MODEL_ORACLE_MAX_CHUNK_POSITIONS, &chunk) < 0 || !chunk) {
        fprintf(stderr, "model_oracle: invalid offset, budget, die list or chunk size\n"); return 2;
    }
    if (cli_read_ids(argv[11], &ids, &cfg.token_count) < 0) return 2;
    cfg.source_root = argv[1]; cfg.source_sha256_manifest = argv[2]; cfg.source_revision = argv[3];
    cfg.tile_cache_dir = argv[4]; cfg.tile_cache_manifest = argv[5]; cfg.tile_cache_sha256_manifest = argv[6];
    cfg.checkpoint_sha256_manifest = argv[7]; cfg.primary_checkpoint_path = argv[8]; cfg.ple_shard_path = argv[9];
    cfg.ple_data_offset = ple_offset; cfg.teacher_ids = ids; cfg.flash_nblk = (uint32_t)budget; cfg.epsilon = 1e-6f;
    memcpy(cfg.physical_dies, physical, sizeof physical);
    memcpy(cfg.shares, production_shares, sizeof production_shares);
    memcpy(cfg.attention_map, production_map, sizeof production_map);
    if (setenv("SPLIT_SHARES", "112,97,81,116,97,97", 1) ||
        setenv("ATTN_MAP", "1,1,1,1,4,4,4,4,5,5,5,5", 1)) {
        perror("set production topology environment"); goto done;
    }
    /* Lock before opening the shared receipt or checkpoint. Append mode also
     * preserves an existing stream when no checkpoint has been published. */
    records = fopen(argv[14], "a+b");
    if (!records) { perror(argv[14]); goto done; }
    struct flock writer_lock = { .l_type = F_WRLCK, .l_whence = SEEK_SET };
    if (fcntl(fileno(records), F_SETLK, &writer_lock) < 0) {
        status = cli_fail("record stream is already in use or cannot be locked", strerror(errno));
        goto done;
    }
    receipt = fopen(argv[16], "a+b");
    if (!receipt) { perror(argv[16]); goto done; }
    if (model_oracle_verify_inputs(&cfg, receipt, &identity, &full_hash, error, sizeof error) < 0) {
        status = cli_fail("input verification failed", error); goto done;
    }
    if (stat(argv[15], &st) == 0) checkpoint_exists = 1;
    else if (errno != ENOENT) { perror(argv[15]); goto done; }
    if (checkpoint_exists) {
        checkpoint_in = fopen(argv[15], "rb");
        if (!checkpoint_in) { perror(argv[15]); goto done; }
    }
    if (model_oracle_open(&oracle, &cfg, &identity, records, checkpoint_in, error, sizeof error) < 0) {
        status = cli_fail("open producer failed", error); goto done;
    }
    if (model_oracle_run_chunk(oracle, (uint32_t)chunk, &processed, error, sizeof error) < 0) {
        status = cli_fail("CPU chunk failed", error); goto done;
    }
    size_t next_len = strlen(argv[15]) + sizeof ".next";
    checkpoint_next = malloc(next_len);
    if (!checkpoint_next) { perror("allocate checkpoint path"); goto done; }
    snprintf(checkpoint_next, next_len, "%s.next", argv[15]);
    checkpoint_tmp = fopen(checkpoint_next, "w+b");
    if (!checkpoint_tmp) { perror(checkpoint_next); goto done; }
    if (model_oracle_checkpoint_write(oracle, checkpoint_tmp, error, sizeof error) < 0) {
        status = cli_fail("checkpoint failed", error); goto done;
    }
    if (fclose(checkpoint_tmp) != 0) { checkpoint_tmp = NULL; perror("close checkpoint"); goto done; }
    checkpoint_tmp = NULL;
    if (sync_parent_directory(argv[14], error, sizeof error) < 0) {
        status = cli_fail("sync record directory", error); goto done;
    }
    if (rename(checkpoint_next, argv[15]) < 0) { perror("install checkpoint"); goto done; }
    if (sync_parent_directory(argv[15], error, sizeof error) < 0) {
        status = cli_fail("sync checkpoint directory", error); goto done;
    }
    printf("MODEL_ORACLE_CHUNK processed=%u complete=%s source_revision=%s\n",
           processed, (oracle->next_position == identity.token_count) ? "true" : "false", identity.source_revision);
    status = 0;
done:
    if (checkpoint_tmp) fclose(checkpoint_tmp);
    if (status && checkpoint_next) unlink(checkpoint_next);
    free(checkpoint_next);
    if (oracle) model_oracle_close(oracle);
    if (checkpoint_in) fclose(checkpoint_in);
    if (records) fclose(records);
    if (receipt) fclose(receipt);
    free(ids);
    (void)full_hash;
    return status;
}
#endif
