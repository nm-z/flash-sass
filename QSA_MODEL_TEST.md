# Production QSA model integration test

model_test retains an explicit MODEL_ATTN=dense baseline and adds MODEL_ATTN=qsa.
The QSA path uses production dimensions and per-layer binding from qsa_state.h:
actual indexer projection/norm weights, INT8 KV and FP16 scales, raw-key rings,
pooled keys, score/select/gather, partial attention and merge.

The scalar path uses identical teacher-forced token IDs. It checks every
attention layer's append state, projections, ring/pool, scores, IDs, gathered
values, partials and output, plus layer streams, recurrent state and the complete
sharded vocabulary logits when the head is enabled. QSA opcode and requested
budget assertions reject accidentally dense/truncated programs. Attention owners
use the production attn_die_of() map. The embedding owner broadcasts its result.

## Run

    MODEL_ATTN=qsa FLASH_NBLK=512 FLASH_KV_HOST=0 \
      MODEL_IDS=/path/ids.txt MODEL_PLE=/path/ple-shard.gguf MODEL_PLE_OFF=192 \
      kern/model_test /path/full-tile-cache 0 48 TOKEN_COUNT eh MANIFEST0 ... MANIFEST5

IDs are whitespace-separated integers. A comparison requires every token up
front; do not label free-running divergent sequences as matching teacher forcing.
MODEL_NOTWIN=1 is a separate GPU smoke/timing mode. FLASH_NBLK must be 1..4096;
FLASH_KV_HOST selects the actual allocated placement. Long sparse exclusion begins
at visible length 4*(budget+1), so budget 512 first excludes at 2052.

MODEL_INJECT supports selection/layer/logits rejection tests. Selection injection
requires actual exclusion. The dense mode remains explicitly labeled as a
baseline and is not production-QSA evidence.

## CPU tile cache without reloading holders

    make -C kern hold_cache
    mkdir -p /new/cache/tensors
    kern/hold_cache CHECKPOINT 0 3 4 /new/cache/manifest cache=/new/cache/tensors

hold_cache compiles the exact holder packing/merging code with HOLD_CACHE_ONLY.
It does not link libcuda or call CUDA initialization, allocation, upload, IPC,
readiness publication or holder wait operations. nm -u verifies this in the
recorded build. It requires cache=, rejects split=, writes tensor files exclusively
and writes an exclusive manifest.new headed NOT A CUDA IPC MANIFEST. Existing
holder manifests/readiness files are not changed. The CPU test maps weight files
read-only rather than eagerly copying the full checkpoint into RAM.

A block-3 export supports the first attention-layer fixture. Full 0..48 tests
need all requested blocks, embedding and head tensors from the correct shard(s).
Keep the actual checkpoint hashes, cache manifest and exact source/binary
identity with any result.

## Status

model_test, dies_test, libengine.so and the matching cubin compile on archy.
The CPU-only cache exporter builds without CUDA symbol references. Compilation
does not establish model parity. Complete target-GPU teacher-forced comparisons,
mismatch-injection results and full-layer sparse-context evidence remain required.
Independent checkpoint quality is separate issue #22.
