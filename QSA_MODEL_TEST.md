# Production QSA model integration validation

`model_test` preserves `MODEL_ATTN=dense` as the named FP32 baseline and adds
`MODEL_ATTN=qsa`. The QSA path uses the production program builder and shared
capacity/per-layer binding helpers in `qsa_state.h`. It supplies actual indexer
projection and norm weights, INT8 K/V codes, FP16 scales, independent layer
rings and pools, score/select/gather, partial attention, and merge. The capacity
helper is a refactor of the existing formula; it is not a capacity correction.

## Comparison contract and scope

The scalar path uses the same supplied token IDs and the instruction-order
reference operations compiled with `-ffp-contract=off`. Codes, scales, selected
IDs, gathered values, partials, outputs, and state comparisons require byte
identity. This is implementation parity under that contract, not an independent
original-checkpoint quality measurement.

Per-token launches compare each layer's stream and QSA stages after every
position. They also compare each die's DeltaNet state slice, the correctly
packed Q/K/V convolution history, and the complete PLE ring when those blocks
are present. With the output head enabled, all vocabulary-logit shards are
compared. Loop launches compare every head prediction but retain detailed
layer/QSA/logit/recurrent buffers only for the final position. Matching earlier
predictions does not establish equality of the earlier full logit vectors.

`e` enables the embedding and `h` enables the output head. A run over a partial
block range is still a partial-model fixture, even with both flags enabled.
Without `e`, the initial stream is synthetic; token IDs do not drive an embedding
lookup. Without `h`, the reported zero prediction is not vocabulary validation.
A full model comparison requires `0 48` and `eh` (or `ehl` for loop launches).

The harness checks the executed QSA opcode counts and requested block budget.
`MODEL_INJECT=selection|layer|logits|dn|conv|ple` provides negative controls; an
injection must reach the named comparison and exit 3. Selection injection needs
actual exclusion, logits needs the head, DN/conv need a recurrent block, and PLE
needs block 1. Invalid input or unreachable controls must be rejected before
CUDA initialization, with exit 2. A malformed configuration is not an expected
numerical-mismatch result.

## Inputs and execution

The executable interface is:

```text
model_test FULL_TILE_CACHE B0 B1 TOKEN_COUNT FLAGS MANIFEST0 ... MANIFEST5
```

`MODEL_IDS` contains decimal vocabulary IDs. A QSA comparison requires at least
`TOKEN_COUNT` IDs and uses that prefix if the file is longer. Free-running
execution is separately selected with `MODEL_NOTWIN=1` and supplies no scalar
agreement evidence. `FLASH_NBLK` must be 1..4096. `FLASH_KV_HOST` selects the
actual allocated placement. Budget 512 first excludes a completed block at
visible length **2,052**, not 2,048 or 2,051.

Use the same `SPLIT_SHARES`, `ATTN_MAP`, numeric `CUDA_VISIBLE_DEVICES`, and
holder manifests as the weight layout. The existing `hold_ordinal()` accepts
numeric CUDA visibility, not UUID strings. Attention owners follow
`attn_die_of()`; the embedding owner broadcasts its stream to the other dies.
For real PLE rows, set `MODEL_PLE` to the table or second checkpoint shard and
set `MODEL_PLE_OFF` to its verified data offset (192 for the recorded shard).

On shared archy, acquire `/tmp/flash-sass-engine-validation.lock` before GPU
work, then inspect `nvidia-smi` and `ps`. Wait if another worker owns the lock or
the required GPU is busy. Check available VRAM before allocating a candidate
context. Release the lock after the run. Keep existing holders and the resident
engine intact: do not kill another worker's process or park, restart, reload,
or reconfigure the resident engine to make a test fit.

## CPU tile cache export

`make -C kern hold_cache` builds the exact holder packing/merging implementation
with `HOLD_CACHE_ONLY`. It does not link libcuda or call CUDA initialization,
allocation, upload, IPC, readiness publication, or holder waiting. Verify its
undefined symbols with `nm -u kern/hold_cache` when recording a build.

```sh
mkdir -p /new/cache/tensors
kern/hold_cache CHECKPOINT 0 0 48 /new/cache/manifest \
  token_embd output cache=/new/cache/tensors
```

The exporter requires a new cache, rejects `split=`, writes tensor files
exclusively, and writes an exclusive `manifest.new` headed
`NOT A CUDA IPC MANIFEST`. It leaves holder manifests/readiness files intact.
The scalar test maps packed tensor files read-only. A block-3 export cannot
supply a 48-layer test; include all requested blocks and embedding/head tensors.
Preserve checkpoint/cache hashes and exact executable identity with results.

## Recorded evidence

| Evidence | What was measured | Limits |
|---|---|---|
| [Archived block 3](validation/issue9/block3-1e299ef) | Source `1e299ef5e169ed9a88f2f6c8d0241831ab688f72`; actual block-3 weights across six resident arenas; nine sequential synthetic-stream positions; budget 1; both KV placements; 9/9 positions exact in each; selection/layer controls exited 3 in both placements. | No embedding, output head, additional model blocks, request reuse, or snapshot test. |
| [Integrated direct-position matrix](validation/issue9/review-e7921ae/direct/matrix/results.json) | Source `e7921ae`; 41 passing cases, comprising 146 positive checked positions and three negative controls. Both KV placements, budget 512, every modulo-four tail around required depths through 262,144; budget-8 short-history replay. Selection/gather/output controls exited 3. | Synthetic component history and fixture weights. It does not measure actual-weight full-model logits or request/snapshot behavior. |

The archived Python runner was removed from the current PR. Historical
`IDENTITIES.json` and raw command records retain the original runner hash;
these immutable records do not claim that the old capture used the new runner.
Build success alone is not GPU parity evidence, and older executable results
do not automatically validate subsequent source changes.

## Remaining issue #9 acceptance

Keep the PR linked with `Refs #9` and in draft until the remaining gates have
measured results:

- Actual-weight embedding → all 48 blocks → full vocabulary logits, with the
  scalar and GPU consuming identical teacher-forced IDs.
- Additional real-weight attention/recurrent/PLE layers and reachable state and
  logits negative controls on the revised harness.
- Full-model coverage at the production selection budget and explicit separation
  from the long synthetic component fixtures.
- Consecutive requests and snapshot/resume on the candidate implementation, with
  single-/multi-token launches and state validation. Restarting `model_test` as
  a new process does not exercise reuse or snapshot restoration.
- Both KV placements and the production die layout for the applicable model
  matrix, with exact source/library/cubin/cache/device identities and raw logs.

Independent checkpoint quality remains issue #22. That separate scope does not
waive the whole-model integration acceptance criteria in issue #9.
