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
needs block 1. The CLI boundary checks require malformed model arguments/IDs
and unreachable controls to be rejected before CUDA initialization, with exit
2. A setup/configuration failure is not an expected numerical-mismatch result.

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

`kern/run_qsa_model_matrix.pl` records a selected matrix, binary/source and input
identities, effective environment, GPU/process admission checks, and raw outputs.
It requires an exact successful comparison summary, the QSA banner and matching
opcode/budget assertions. Negative controls require a reachability marker,
comparison mismatch and exit 3. `--controls none` selects positives only;
`--controls all` selects reachable controls; an explicit list rejects unreachable
requests. A passing report says `SELECTED_CASES_PASSED` and leaves
`acceptance_complete` false.

```sh
perl kern/run_qsa_model_matrix.pl --self-test
bash kern/test_model_cli.sh kern/model_test
perl kern/run_qsa_model_matrix.pl --repo "$PWD" --out /new/results \
  --cache /full/cache --ids /teacher-ids --blocks 0:48 --tokens 2052 --flags eh \
  --manifest /holder0 --manifest /holder1 --manifest /holder2 \
  --manifest /holder3 --manifest /holder4 --manifest /holder5 \
  --cuda-visible 0,1,2,3,4,5 --budgets 1,512 --kv 0,1 --modes token,loop \
  --controls none --plan
```

`--plan` does no GPU work. Actual runs acquire the fixed shared lock and admit
only explicitly named holder PIDs whose executable and manifest match. An owned
holder wrapper may pass `--lock-fd 9`; the runner verifies that descriptor's
device/inode against the fixed lock and retains it through each child. Both
paths inspect GPU and process state before each run. Minimum free-memory
thresholds are admission checks, not guarantees that arbitrary allocations fit.
The CPU CLI checks intercept `cuInit`, include a positive interception control,
and require invalid/zero-length cases to exit 2 before reaching it.

For bounded private-engine fixtures, `FLASH_LAYERS` limits initialization,
request reset and snapshot copying to the executed blocks. The default remains
48. Partial-layer engine results must report the effective `config` and remain
labeled as partial-model results, regardless of head/logit comparisons.

`kern/check_qsa_engine_requests.pl` uses the same lock/admission helpers for
owned private cores. It checks fresh requests, prompt changes, snapshots at
positions 1/2/3/5, resume counts, final state/logit fingerprints and actual
paired/chunk counters. A separate `gen 1` request has its own full-head engine
reference; it is not a teacher-forced scalar-model comparison. `--self-test`
checks the parser on CPU and `--plan` displays the exact request sequence.
Actual execution requires an inherited reservation and an explicitly identified
holder, such as those supplied by `validation/issue9/with_private_holder.sh`
for the one-die fixture.

### Multi-die and variable-depth request checks

The request checker accepts one to six dies, with a repeated `--manifest` and
`--holder-pid` for each physical device in `--cuda-visible`. Manifest order
defines the logical die numbers in `statehash`. Multi-die runs require explicit
`--split-shares` and `--attn-map` (or their matching environment variables);
they must describe the supplied weight arenas, not an invented equal split.
The reported engine `config` must match both the topology and execution flags.
Recurrent state, PLE and vocabulary-shard fingerprints are required on every
die; attention state is required only on that layer's mapped owner. Missing,
duplicate, unexpected or changed remote-die records fail the comparison.

Defaults remain eight blocks, nine IDs, and snapshots at 1/2/3/5. `--layers 48`
requests all blocks. `--tokens N` supplies identical deterministic, valid IDs
to each launch/KV-placement mode; `--snapshots` selects exact interior snapshot
positions. For production-budget exclusion use `--nblk 512 --tokens 2052`,
for example with `--snapshots 1,2,3,5,2048,2051`. Capacity rounds up separately
from the requested visible length and leaves room for the separate generated
position. This replay matrix is not a replacement for direct-at-position
component checks at huge synthetic-cache depths.

`--plan` does not access CUDA, holders or the lock. Reports use schema
`flash-sass.private-qsa-request-check.v2` and preserve the requested die map,
snapshot positions, token count and capacity. `requested_selection_exclusion`
describes the plan, not a measured result. Actual runs also record every
manifest hash, an exact source diff and optional `--weight-manifest` and
`--build-log` identities. They retain `acceptance_complete=false`: neither a
synthetic parser fixture nor GPU-versus-GPU fingerprints establish full-model
CPU agreement or checkpoint lineage.

`perl kern/check_qsa_engine_requests.pl --self-test` runs
`kern/test_qsa_engine_requests.pl`, including six-die, 48-layer, 2,052-token
synthetic captures; corruption/missing-state checks on nonzero dies; topology
rejection; and independent chunk/snapshot accounting checks. These are CPU
parser/planner tests, not GPU results.

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
| [Integrated direct-position matrix](validation/issue9/review-523110f/README.md) | Source `523110f`; 41 passing cases, comprising 146 positive checked positions and three negative controls. Both KV placements, budget 512, every modulo-four tail around required depths through 262,144; budget-8 short-history replay. Selection/gather/output controls exited 3. | Synthetic component history and fixture weights. It does not measure actual-weight full-model logits or request/snapshot behavior. |
| [Eight-block model matrix](validation/issue9/review-4429cbd/README.md) | C source `4429cbd0e40c9338cb9e5cf809afd52cb71e0a5e`; actual embedding, blocks 0..7 and full output head; both KV placements; budgets 1/512; token/loop launches. Main batch: 12 positive cases and 12 deliberately rejected mismatches, plus one separate initial positive. | One-die partial model. Nine-ID budget-512 cases stay below exclusion. Loop runs retain only the final detailed buffers. |
| [Private-engine requests](validation/issue9/review-4429cbd/engine-batch-v2/matrices/) | 128 passing request checks across eight owned cores: budgets 1/512, host/VRAM, scalar/paired launches, reset, snapshots/resume at 1/2/3/5, and a separate short generation/chunk check. Actual `config` and state/logit fingerprints are recorded. | Eight blocks on one die; GPU-versus-GPU FNV fingerprints. This is not full-model CPU agreement or production-budget deep exclusion. |

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
- Full-model coverage at the production selection budget and explicit separation
  from the long synthetic component fixtures.
- Extension of the measured eight-block request/snapshot/generation matrix to
  the complete model and production die placement. Restarting `model_test` as a
  new process does not exercise reuse or snapshot restoration.
- Both KV placements and the production die layout for the applicable model
  matrix, with exact source/library/cubin/cache/device identities and raw logs.

Independent checkpoint quality remains issue #22. That separate scope does not
waive the whole-model integration acceptance criteria in issue #9.
