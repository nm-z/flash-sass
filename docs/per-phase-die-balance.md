# Six-die work balance within ten percent per phase

Active objective: each compute die performs useful work within±10% of the
six-die mean **for each phase**. A token-wide average, equal phase counts,
wait padding, or redundant unused arithmetic does not establish this target.
The full48-layer model and full head remain in scope. The production phase
limit remains below500. Complete the full balanced path before GPU execution.

The primary runtime report must show per-phase/per-token/per-die useful-body
time, separately accounting for exchange, local-grid and in-operation waits.
The work ledger must retain row/head counts, weight formats/bytes, selected
expert counts and actual CTA participation. Equal static row counts alone do
not prove equal elapsed body time on different GPUs. Every phase remains in
the report; leader-only or zero-work peers are not silently filtered out.

## Choice of execution layout

The existing split partitions whole experts and assigns one owner to each
attention layer. Changing only SPLIT_SHARES or rotating attention owners can
balance totals while still leaving individual phases badly unbalanced.

The implementation direction is row-striped experts and shared experts,
equal recurrent-head ownership, distributed attention query heads, and row
shards for dense projections. Keep local work contiguous where that preserves
complete activation-quantization groups. Preserve global expert IDs and every
selected expert's contribution. Storage ownership may differ from compute
ownership when a GPU has a fenced memory range.

| Component | Intended six-die useful work |
|---|---|
| GR down/injection | 54 of324 rows per die |
| GR up/mix and2560-row output projections | 416/448 columns or rows; within5% of mean |
| GDN | Eight of48 value heads per die; matching eight key heads |
| Routed/shared640-row gate/up projections | Eight-row stripes:104/112 rows per expert per die; within5% regardless of selected expert IDs |
| Routed/shared2560-row down projections | Eight-row stripes:424/432 rows per expert per die; within1.25% |
| QSA query/gate projection and attention | Four of24 query heads per die |
| QSA K/V | The one GQA K/V head consumed by that die's four query heads |
| Indexer projection | Eight-row stripes of640 rows, published before each die's local selection |
| PLE key | Complete32-element groups of10240 output rows |
| Output head | Complete32-row groups of248320 logits |
| Embedding | Column shards with exact original dequantization and publication of all four residual branches |

A32-row expert stripe is insufficient:640 rows contain only20 such groups;
a die can receive128 rows versus a106.667-row mean, a20% excess. Eight-row
stripes contain80 groups and bound the per-expert excess at5%, independently
of the router's selected IDs. This is an analytic preflight bound, not a GPU
qualification result.

## Implementation state

Branch `perf/per-phase-die-balance` starts at validated v39 commit00e0981.

Implemented:

- `kern/balance/layout.h`: shared holder/program row mapping, balanced dense
  ranges, GDN head ranges, attention query/KV head ranges, and layout marker.
- `hold balanced=d/6`: packs rows of all512 routed experts and every shared
  expert, dense/head shards, four attention query heads, one K/V head, and
  striped indexer projection rows. Each stored row retains the checkpoint's
  representation; the existing quantized row repacker is reused.
- `kern/balance/embedding.h`: packs independent32-column groups of the raw
  Q3_K embedding into16-byte code/mask/scale records. Both original six-bit
  scales and the FP16 multiplier are retained; no weight is requantized.
  Balanced holders include the embedding column shard and output head on all
  six dies.
- `kern/balance/embedding.lua`: the matching production SASS decoder and
  direct six-die residual publication, including image embeddings. Each die
  owns its complete32-column groups; every original code/scale is decoded in
  the original multiplication order. One embedding phase publishes disjoint
  columns and the next operation acquires their completion flags.
- `prog_embed_bcast` constructs the balanced embedding descriptor. Dense/GDN,
  PLE and head allocations/ranges use the shared execution split when the
  balanced mode is enabled; logit global offsets follow those same boundaries.
  Each causal position has its own six-die embedding destination table.
- `die_balanced.cubin` assembles successfully from production SASS with the
  replacement embedding and striped MoE bodies. Its distinct module marker prevents loading it
  with the legacy holder/program layout. It is an in-progress image, not a
  runnable complete balanced model yet.
- `kern/balance/program.h` constructs same-token gate/up and down chains. All
  ten selected global IDs are consumed on each die. Compact local row buffers
  are separate from the complete vectors published to peers, preventing local
  publication from overwriting rows that have not yet been sent.
- `kern/balance/moe.lua` retains production per-row dot arithmetic, changes
  routed expert strides to the local row counts, and executes eight useful
  rows per virtual unit. Shared-expert format bodies take the same marked
  row-shard path; ordinary dense descriptors retain their original row loops.
- `kern/balance/prepare.lua` performs SiLU/multiply and32-value activation
  quantization inside the existing down dispatch, then joins sixteen local
  CTA readiness flags. Epochs identify token/layer, and flags are cleared for
  every request, including a prefix-cache resume. The join remains an explicit
  data dependency, not an omitted wait or extra model dispatch.
- Down retains six numerical partials for each owned row. Experts are processed
  in the original owner group and router order, with the first contribution
  multiplied and later contributions accumulated as before. The shared expert
  is added to its original group. Publication reduces groups0–5 in order;
  the following GR consumer reads one assembled vector instead of summing six
  copies. The reference owner boundaries come from the retained SPLIT_SHARES.
- The engine now classifies all six layout markers, validates logical ranks,
  stripe grain and distinct physical GPUs, and rejects mixed arenas. The row
  path requires fused GR, producer quantization, matrix grouping, all48 layers
  and a single causal position. Kernel ABI markers must match the layout.
- Holder and engine build checks. No balanced holders have been loaded and no
  balanced kernel has been run. The serving baseline and persistent holders
  have not been changed.
- Four-head attention integration now assembles: each die projects four query
  heads and its one K/V head, publishes its striped indexer rows, updates its
  one-head persistent cache, selects keys, gathers and computes attention, and
  publishes quantized output at the global query-head offset. Two-head packets
  preserve the original eight key partitions and their reduction order while
  assigning useful attention work to all sixteen CTAs.
- `kern/balance/composite.lua` implements one complete local QSA operation with
  seven real child descriptors. Its normalization/indexer, gather and partition
  stages retain three local grid readiness joins. Selection completion uses
  one producer flag: score CTAs can enter the gather path as soon as the sole
  selection CTA publishes the selected indices. Score-producer readiness for
  that selection CTA remains a separate internal wait. The whole QSA operation
  restores the descriptor cursor before the production footer publishes output.
- The complete CPU production builder now emits **493 physical phases and980
  descriptors on every die**, including all48 layers and the full head. It
  reports90 phase-footer local grid boundaries and402 exchange publications/
  acquires. In addition, twelve QSA blocks contain36 local grid joins,12
  selection-producer joins and12 score-consumer joins; MoE preparation adds48
  local grid joins. Native `inner_grid_barriers=0` does not mean these internal
  waits are absent. This is descriptor/build evidence, not GPU timing evidence.

Still required for the complete operation loop:

1. Install the complete balanced engine and compatible holders only after
   physical sizing and concrete restoration artifacts are ready. The source
   now enables the balanced path for six validated matching holder markers;
   the live baseline has not been reloaded.
2. Verify complete dense/GDN/PLE/head descriptor coverage in the final program.
3. Qualify the implemented routed/shared path as part of the complete model;
   no partial-MoE GPU benchmark has been performed.
4. Verify the six-part accumulation/publication and assembled GR path against
   completed full-model outputs and state, after the first end-to-end run.
5. Qualify the integrated four-head attention bodies, one-head cache layout,
   striped indexer publication and global-offset output publication in the
   complete model. Their CPU/SASS build is complete; runtime is unmeasured.
6. Complete and budget the physical placement below. Use six fully resident
   balanced weight shards; no active-row cache is needed if the remap fits.
7. Per-phase useful-work timing, explicit wait accounting and complete reports;
   full-model end-to-end run, then repeatability/output gates and±10% audit.

## Storage constraint that must be solved in the implementation

The saved trace job73ed8630-36f3-47a5-9e7d-36fd7db0584f reports only9–88MiB
free on three physical GPUs at peak with replication/tracing, versus about
1.1GiB on the other compute GPUs. Holder metadata shows5.29GB of weights on
logical die2 but7.46–7.51GB on dies0/3. Do not infer the placement solely from
the nvidia-smi totals: retain UUID/process mapping.

The current process inventory confirms fence PID2210 on physical GPU2,
protecting its configured range2750–3769; it is not a spare allocation to
remove. Canonical document170646 identifies2110MiB for this fence,5116MiB for
die2's holder PID323394 and140MiB for its core context in the saved snapshot.
Equal expert-row storage on that compute GPU cannot be assumed to fit.
The spare physical GPU6 had only71MiB allocated in the saved trace snapshot.
The selected approach is to remap logical compute die2 to spare physical GPU6
and move the lightweight host-memory anchor to physical GPU2. The six logical
compute dies and every model contribution remain; physical compute placement
becomes0,1,6,3,4,5. The fence remains on physical GPU2. The core already creates
each compute context using the physical `die` recorded in its holder manifest,
independently of that manifest's logical position. A holder on physical GPU6
can therefore pack `balanced=2/6`, record physical `die6`, and occupy logical
slot2 without changing model head/expert identities.

Canonical read-topology job`c21a14c9-b3fb-4cf9-a0c6-70f6cabe155b` reports
peer reads OK for every directed GPU pair, including GPU6. The corresponding
write-capability query`3d508be0-cf6b-4bd3-8c7f-e5aa8137a633` also reports OK
for every directed pair.
These are capability queries, not kernel or partial-model benchmarks. No
holder/core placement has been changed yet. Budget the full new holders,
dense weights, one-head KV/indexer state and scratch before installation.

This replaces the more complex alternative of keeping compute on fenced GPU2
with backing weights on GPU6 and a local active-row cache. Cache misses would
introduce another source of per-phase imbalance. Remapping removes that
constraint while retaining six-way computation. Do not rely on the fenced band.

## Routed/shared implementation contract

The next production component must consume all ten selected global expert IDs
on every compute die. The holder's row mapping is independent of selected ID;
local row r maps to`((r/8)*6+die)*8+r%8`. Gate/up expert stride is the local
row count times the packed2560-column row stride, not the legacy fixed
640-row expert stride. Down uses the local2560-row shard and its640-column
packed stride. The existing SASS has literal full-expert strides that must be
replaced; changing only descriptor row counts is incorrect.

The gate/up phase can group the two routed matrices and two shared matrices,
publishing disjoint FP32 G/U rows to all six dies. At the existing down-stage
entry, acquire complete G/U, evaluate/quantize all eleven activations locally,
then execute disjoint down-output row shards. A local readiness join is needed
between parallel quantization and consumers; account for it explicitly. Run
the preparation only on each CTA's first virtual unit, so it is not repeated
or raced when a CTA loops over additional output-row units.

The implementation runs preparation from the routed-down SETUP label, before
its BEGIN/END virtual-unit loop. Same-token groups use the native chain flag32;
the routed→shared chain additionally uses the existing local-fence flag31.
The balanced image disables the old Q2MOEGN flag32→whole-expert GA shortcut,
so chain metadata cannot accidentally select a640-row legacy body.

Publish the completed output row shards directly. The next GR consumer must
consume one assembled output vector, not sum six copies of that vector. Retain
all selected contributions and document the accumulation order. Shared expert
formats vary by layer, so reuse their appropriate production format bodies.

Use eight-row work units where possible. Q2_0 K2560 uses16 lanes per row,
so an eight-row unit has128 participating threads; Q2_0 K640 uses four lanes,
so it has32. Keep other warps out of useless row arithmetic while retaining
their mandatory phase/CTA participation. Do not run the old minimum256-row
GEMV iteration on a104-row shard merely to make timings appear equal.

Distributing QSA to every die also changes physical dispatch counts: the old
499-phase programs have only two owner attention layers per die, even though
there are twelve attention layers in the full model. The complete new
production schedule must be counted before execution. Preserve the below500
limit using real dependencies and the complete model, not omitted attention
work or a relabeled count of descriptors. Any fused local readiness waits must
remain visible in the balance report.

## Qualification and publication

The `die_balanced_body.cubin` diagnostic image now builds with the production
operation graph. With `FLASH_BODY_TRACE=1 FLASH_XTRACE=N`, `bodytrace D` emits
every physical phase/CTA, body entry after incoming acquisition, body completion
after the local footer rendezvous, SM residency and individually timed explicit
polls. QSA records normalization/indexer, score-producer, selection-producer,
gather and partition waits; MoE preparation and the router consumer have their
own poll records. Existing `xtrace D` retains publication/arrival/acquire data.
Missing or malformed records remain in the output with `valid:false`.

The body-minus-poll field still includes local CTA barriers, publication,
dispatch and instrumentation overhead. It must not be called pure arithmetic
time. Resident SM IDs establish placement, not useful utilization; the work
ledger and operation participation are still needed. The trace has not run on
a GPU yet. Both normal and diagnostic images preserve493 physical phases.

On October22, the source checkpoint's NFS mount timed out on archy. Direct SSH
to Sentry's configured address returned `No route to host`. Existing GPU
holders and the serving core remain alive. Do not stop them before preserving
a usable source for replacement weights: their source GGUF is not currently
reachable. Repacking their existing resident native rows into a local backing
cache avoids depending on that mount. `kern/balance/native_weights.c` now
implements original-arena export, CPU redistribution, and a persistent cache
loader. It shares the exact holder row-selection code through
`kern/hold_selection.h`; native quantized rows are copied without conversion,
Q6 slices are redistributed separately, and embedding groups use the same
integer packer as checkpoint loading. Original arenas remain a separate local
restoration source. Cache metadata is published only after its arena is durable.

The complete CPU size plan passed in canonical job
`75442a5e-5e35-4d72-8f00-0f6745fe7179`:

| Logical die | Physical GPU | Balanced arena bytes |
|---|---|---|
| 0 | 0 | 6,798,874,112 |
| 1 | 1 | 6,798,874,112 |
| 2 | 6 | 6,494,398,720 |
| 3 | 3 | 6,476,552,960 |
| 4 | 4 | 6,476,552,960 |
| 5 | 5 | 6,494,398,720 |

Each arena contains1080 tensor records, including its layout marker. The
original arenas total39,456,608,768 bytes. Archy had88,408,420,352 available
disk bytes before export, enough for both original and redistributed arenas.
Runtime context/state/trace memory must still be included before installation.

The first plan run exposed an embedding catalog detail: only die0's original
manifest contains the replicated embedding. The native redistributor now
accepts that single source while preserving all global embedding rows. This
was a CPU metadata failure, not a model run. A subsequent queue capture failed
mid-plan; [gpuq#70](https://github.com/nm-z/gpuq/issues/70) is attached under#69
and remains open. With the known stale polling shell temporarily suspended,
the identical complete plan succeeded. No balanced holder has been loaded and
no balanced model run has occurred yet.

Original export is tracked as canonical job
`1be7314c-ba35-4acf-ac79-803d951cd241`. The original core and holders remain
live during its read-only device-to-host copies. It finished successfully with
all six original arenas (39,456,608,768 bytes). Observed export stalls were
Linux disk writeback waits (`rq_qos_wait`, `balance_dirty_pages` and `fsync`),
not evidence that the process stopped. Re-poll the same job/process rather than
starting another export solely because an observation timed out.

The first queued redistribution, job`25c7ea28-09a0-4007-8cb2-b67e790fd86d`,
failed before copying any rows: GPUQ had captured and removed the cache's
operational text policy/catalogs. Their exact contents are retained in canonical
artifact documents173167–173173. They were recovered from that database, and
job`68221e81-71e4-41ea-b022-33c4d79a4f75` sealed them into versioned binary
`policy.bin`/`metadata.bin` catalogs. All seven binary catalogs remain present
after collection. Arena bytes were unchanged; no second export was required.
[gpuq#71](https://github.com/nm-z/gpuq/issues/71) records the operational-file
cleanup issue. New native caches/loaders use the binary catalogs.

Disk I/O pressure exceeded80% during export. The original SSD setting
`/sys/block/sdb/queue/wbt_lat_usec` was2000; its saved value is
`/home/nate/work/flash-balanced-native-20261022-v1/wbt_lat_usec.original`.
It was temporarily0 during the final original export and initial redistribution;
it has now been restored to2000 before any model performance run. Resume the
temporarily stopped stale GPUQ polling shell after the long captures are finished.
No model timing result was collected under the temporary setting. The current
CPU redistribution is canonical job`823db2be-1032-44ec-92d2-905ae2965ce2`, using
the binary catalogs and the original unmodified arena bytes.

The existing `resident_decode_bench` now supports a private balanced core with
`BENCH_ROW_BALANCE=1 BENCH_HOLDER_DIR=... BENCH_CANDIDATE_ONLY=1 BENCH_NO_MTP=1`.
It reads `h0.ipc.bin` through `h5.ipc.bin` under that directory, uses physical
GPU2 for the anchor, and selects the complete scalar production path. The first
execution should use `BENCH_RUNS=1` without repeatability, state or reference
options. A single unchecked execution is explicitly reported without claiming
correctness/repeatability. After it finishes, the same path supports frozen
reference output, repeatability/state gates and scalar teacher-forced coverage.
`BENCH_BODY_TRACE=1 BENCH_XTRACE=N` additionally collects all six body reports.
These full-model commands have not been run on balanced holders yet.

Persistent CUDA IPC manifests use the`.ipc.bin` suffix so GPUQ treats them as
operational data, while retaining their normal arena/handle/tensor serialization
for the core's parser. Their readiness files use a versioned binary marker; the
core checks marker existence, not textual contents. This also prevents artifact
cleanup from removing a live holder's bootstrap files at the end of its launch
job. Source policy and tensor work remain visible in the queue's plan/log data.

The diagnostic stream also retains the actual ten selected global expert IDs
and selected count at each routed gate entry. Their metadata occupies an unused
second child record, separate from poll records. Metadata is captured before
starting the body clock; the per-CTA timestamp address is restored afterwards.

Reuse canonical GPUQ evidence before starting new work. All archy GPU execution
continues through `ssh archy gpuq run -- COMMAND`. Preserve one canonical
history.sqlite3, immutable source/images and the exact workload. The completed
candidate must execute the full production loop before extra diagnostic tests.
Then check same-image state/token repeatability before cross-image comparison.
The final evidence must include **every phase** and report actual TPS, memory,
cache traffic and residual imbalance; do not qualify only phases that pass.

This goal is not complete. No±10% runtime balance or TPS improvement is claimed.

## Next attention integration

Preserve four local query heads and one local K/V head on every die. Main
Q/gate rows are2048 per die; K/V projections have256 rows each. The640-row
indexer projection uses the existing marked row-shard publication path into
all six complete IQK vectors. Group these independent projections and publish
indexer completion before consuming IQK. Normalization and cache append must
use local head counts, and compact/persistent KV layouts must have one head.

The old owner program already groups ATTNQKQ and IDXQK as independent work.
Fusing IDXQK with scoring alone would therefore not remove a phase. A complete
local operation now runs normalization/cache append and indexer update, joins
their producers, then performs scoring/selection, KV gather, attention partitions
and gated merge. The sole selection CTA publishes completion directly to gather
consumers. The gather and partition producer joins remain explicit. An initial
two-operation version counted505 physical phases; carrying selection directly
into gather removed twelve dispatch boundaries and produces493. No attention
work was skipped, and the internal waits remain part of the required timing
audit. These counts come from the full production CPU builder, not a GPU run.

The existing score/select body loads its auxiliary selection descriptor using
pidx+1, while the part/merge body similarly loads an adjacent merge descriptor.
Composite stages must load the correct child parameter banks, apply each
child's own +TOK mask, preserve the primary phase flags/count, and restore any
temporary descriptor cursor before the physical phase footer. Do not reuse a
completion barrier across global/shared loads without explicit retirement.
Each readiness generation must be distinct per token/layer and cleared on all
request paths. Projection, append, selection, gather and merge must all execute
on every logical die; an empty owner-participation exchange is not useful work.
