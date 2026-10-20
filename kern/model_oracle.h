/* CUDA-free, resumable scalar QSA/model oracle for full-model integration.
 * Records preserve the raw-byte contract used by model_qsa_compare and
 * model_compare_layers. This interface deliberately has no CUDA types. */
#ifndef FLASH_MODEL_ORACLE_H
#define FLASH_MODEL_ORACLE_H

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#define MODEL_ORACLE_FORMAT_VERSION 1u
#define MODEL_ORACLE_DIES 6u
#define MODEL_ORACLE_BLOCKS 48u
#define MODEL_ORACLE_ATTN_LAYERS 12u
#define MODEL_ORACLE_SHA256_BYTES 32u
#define MODEL_ORACLE_LABEL_BYTES 64u
#define MODEL_ORACLE_REVISION_BYTES 64u
#define MODEL_ORACLE_MAX_CHUNK_POSITIONS 32u
#define MODEL_ORACLE_GLOBAL_LAYER UINT32_MAX

typedef struct model_oracle model_oracle_t;
typedef struct model_oracle_reader model_oracle_reader_t;

/* Content identity contains no absolute execution paths: source-root and NFS
 * mountpoint differences do not change equality. Checkpoint/cache paths remain
 * caller-supplied provenance used only to locate bytes. */
typedef struct {
    char source_revision[MODEL_ORACLE_REVISION_BYTES];
    uint8_t source_content_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t tile_manifest_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t tile_content_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t checkpoint_content_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t primary_checkpoint_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t ple_checkpoint_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint8_t teacher_ids_sha256[MODEL_ORACLE_SHA256_BYTES];
    uint64_t ple_data_offset;
    uint64_t token_count;
    uint32_t flash_nblk;
    uint32_t epsilon_bits;
    uint32_t physical_dies[MODEL_ORACLE_DIES];
    uint32_t shares[MODEL_ORACLE_DIES];
    int32_t attention_map[MODEL_ORACLE_ATTN_LAYERS];
} model_oracle_identity_t;

/* Paths identify the exact inputs; shares/map and physical_dies are explicit
 * production topology, not inferred from a default environment. teacher_ids
 * are all teacher-forced input IDs for token_count positions. */
typedef struct {
    const char *source_root;
    const char *source_sha256_manifest;
    const char *source_revision;
    const char *tile_cache_dir;
    const char *tile_cache_manifest;
    const char *tile_cache_sha256_manifest;
    const char *checkpoint_sha256_manifest;
    const char *primary_checkpoint_path;
    const char *ple_shard_path;
    uint64_t ple_data_offset;
    const uint32_t *teacher_ids;
    uint64_t token_count;
    uint32_t flash_nblk;
    uint32_t physical_dies[MODEL_ORACLE_DIES];
    uint32_t shares[MODEL_ORACLE_DIES];
    int32_t attention_map[MODEL_ORACLE_ATTN_LAYERS];
    float epsilon;
} model_oracle_config_t;

enum model_oracle_value_type {
    MODEL_ORACLE_OPAQUE = 0,
    MODEL_ORACLE_F32 = 1,
    MODEL_ORACLE_F16 = 2,
    MODEL_ORACLE_I8 = 3,
    MODEL_ORACLE_U32 = 4
};

/* A per-position item is a byte-exact buffer from the scalar twin. SHA-256 is
 * over the exact compared bytes. raw_bytes is present for selected block IDs
 * and per-die prediction tuples; raw U32 values are little-endian and other
 * items are represented by the digest.
 *
 * Canonical QSA labels/order match kern/model_qsa_test.h:
 *   bN.kq, bN.vq, bN.ks, bN.vs, bN.iqk, bN.iq, bN.ring, bN.pool,
 *   bN.scores, bN.selected_ids, bN.gather_k, bN.gather_v,
 *   bN.qo, bN.pm, bN.pd, bN.po, bN.aout.
 * Within each position, items follow model_compare_layers order: logical die
 * ascending, then block ascending; each block contributes S, its QSA labels
 * in model_qsa_compare order, then applicable recurrent/PLE state labels;
 * that die's logits follow its blocks. Final S.D and tok.D items are emitted
 * in ascending die order. Readers may match GPU observations by label without
 * relying on this serialized order, but must reject missing, duplicate, or
 * extra labels.
 * bN.pool is only the newest completed 128-value FP16 row and is present only
 * when (position & 3) == 3. bN.gather_k and bN.gather_v are F32 little-endian
 * [nkeys][2][256], slot-major; K is read through attn_kidx and V is native
 * position-major. All compared items have rank 1 with a flat scalar count,
 * except gather_k/gather_v (rank 3). In particular ring is flat
 * 128*((position&3)+1), pm/pd are flat 192, po is flat 49152, qo/aout are
 * flat 6144, and recurrent/packed-conv/PLE/logits items are flat at their full
 * compared byte count. Codes are I8; scales and pool are F16; selected IDs and
 * tok tuples are U32; all remaining arrays are F32. The remaining comparison
 * labels are exactly bN.S.dD, bN.dn_state.dD, bN.conv_state.dD,
 * b1.ple_state.dD, logits.dD, S.D and tok.D, with D the logical die index. */
typedef struct {
    char label[MODEL_ORACLE_LABEL_BYTES];
    /* layer is MODEL_ORACLE_GLOBAL_LAYER for per-die logits/final items,
     * otherwise it is a block index in [0, MODEL_ORACLE_BLOCKS). */
    uint32_t layer;
    /* logical_die indexes identity.physical_dies; physical_die must match it. */
    uint32_t logical_die;
    uint32_t physical_die;
    uint32_t value_type;
    uint32_t rank;
    uint64_t dimensions[4];
    uint64_t byte_count;
    uint8_t sha256[MODEL_ORACLE_SHA256_BYTES];
    const uint8_t *raw_bytes;
    uint64_t raw_byte_count;
} model_oracle_item_t;

/* item pointers and raw_bytes returned by reader_next remain valid until the
 * next reader_next call or reader_close. position is zero-based; visible_length
 * is position + 1. prediction_score_bits preserves the scalar FP32 argmax value. */
typedef struct {
    uint64_t position;
    uint64_t visible_length;
    uint32_t teacher_id;
    uint32_t prediction;
    uint32_t prediction_score_bits;
    uint32_t item_count;
    const model_oracle_item_t *items;
} model_oracle_record_t;

/* GPU readbacks are supplied in the same canonical byte order and with the
 * comparator labels from model_qsa_compare/model_compare_layers. */
typedef struct {
    const char *label;
    uint32_t value_type;
    uint32_t rank;
    uint64_t dimensions[4];
    const void *bytes;
    uint64_t byte_count;
} model_oracle_observation_t;

/* Initial CPU-oracle verification may hash all inputs and write the persistent
 * seal; open the receipt as a+b. On resume, a nonempty receipt reuses its
 * digests, rechecks small source files, and checks large cache/checkpoint
 * inputs by logical key, size and mtime_ns. That check trusts large inputs
 * were not replaced while preserving those metadata values. Paths are not
 * part of content identity. full_hash_performed is 1 only after full hashing. */
int model_oracle_verify_inputs(const model_oracle_config_t *config,
                               FILE *verification_receipt,
                               model_oracle_identity_t *identity,
                               int *full_hash_performed,
                               char *error, size_t error_capacity);

/* Read-only consumer gate. Open verification_receipt as rb and require a
 * nonempty seal. An empty receipt fails before model/cache input traversal;
 * this function never writes the seal or performs full content hashing. */
int model_oracle_verify_inputs_from_receipt(const model_oracle_config_t *config,
                                             FILE *verification_receipt,
                                             model_oracle_identity_t *identity,
                                             char *error, size_t error_capacity);

/* Open validates config against the verified identity. With checkpoint_in==NULL
 * it starts at position 0 and writes a new record header; otherwise record_stream
 * must reference the existing record file (opened for read/append) and the
 * matching checkpoint restores scalar state. Caller owns the persistent record,
 * receipt, and checkpoint files (for example, under a DSH-owned NFS folder). */
int model_oracle_open(model_oracle_t **out, const model_oracle_config_t *config,
                      const model_oracle_identity_t *identity,
                      FILE *record_stream, FILE *checkpoint_in,
                      char *error, size_t error_capacity);

/* Process at most max_positions (1..MODEL_ORACLE_MAX_CHUNK_POSITIONS), writing
 * one self-describing record per completed position. processed receives the
 * actual number written; 0 means the configured token_count is complete. */
int model_oracle_run_chunk(model_oracle_t *oracle, uint32_t max_positions,
                           uint32_t *processed, char *error, size_t error_capacity);

/* Flush and fsync the record stream, then save current stream, DeltaNet,
 * packed-convolution, PLE and QSA cache state, identity and next position.
 * The caller owns persistence; after atomic rename, fsync the parent directory
 * before treating a checkpoint update as durable. */
int model_oracle_checkpoint_write(const model_oracle_t *oracle, FILE *checkpoint_out,
                                  char *error, size_t error_capacity);
void model_oracle_close(model_oracle_t *oracle);

/* Read and identity-check a record stream before GPU comparison begins. */
int model_oracle_reader_open(model_oracle_reader_t **out, FILE *record_stream,
                             const model_oracle_identity_t *expected,
                             char *error, size_t error_capacity);
/* Returns 1 for a record, 0 at clean EOF, and -1 on malformed/truncated input.
 * Rejects empty records, invalid item types/ranks/shapes, duplicate labels,
 * raw-byte contract violations, and die/layer fields inconsistent with the
 * stream identity. */
int model_oracle_reader_next(model_oracle_reader_t *reader,
                             model_oracle_record_t *record,
                             char *error, size_t error_capacity);
void model_oracle_reader_close(model_oracle_reader_t *reader);

/* Rejects an empty expected record. Compare every required item exactly: raw selected IDs use byte comparison;
 * remaining items compare SHA-256 of the readback bytes. After all labels,
 * sizes and values match, write a result line with scalar_match=true. On a
 * mismatch write scalar_match=false and return 1; I/O/format errors return -1. */
int model_oracle_compare_record(const model_oracle_record_t *record,
                                const model_oracle_observation_t *observations,
                                size_t observation_count, FILE *result_stream,
                                char *error, size_t error_capacity);

/* Defining MODEL_ORACLE_STANDALONE adds a one-chunk CPU producer CLI main;
 * without it this translation unit provides only the library API and can link
 * into model_test's GPU-only record consumer without a duplicate main. */

#endif
