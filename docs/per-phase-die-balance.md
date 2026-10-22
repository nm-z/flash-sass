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
  six dies. Its matching SASS decoder/publication path is still required.
- A distinct manifest marker prevents the legacy engine from accidentally
  interpreting these matrices as whole-expert shards. The balanced descriptor
  and SASS path must replace that temporary rejection before execution.
- Holder and engine build checks. No balanced holders have been loaded and no
  balanced kernel has been run. The serving baseline and persistent holders
  have not been changed.

Still required for the complete operation loop:

1. Balanced embedding dequantization and publication (packing is implemented).
2. Descriptor/buffer ownership for all dense, GDN, PLE and head stages.
3. Row-striped gate/up/shared SASS, publication of complete FP32 activations,
   and local quantization before row-striped down projections. The existing
   gate/up→down boundary carries this dependency; do not add placeholder phases.
4. Down projection accumulation preserving selected contributions and documented
   numerical order, with disjoint output-row publication instead of whole-expert
   partial ownership. The following GR consumer must read that revised layout.
5. Four-head attention bodies, one-head cache layout, striped indexer projection
   publication, local selection and output publication with global head offsets.
6. Memory placement/cache support for the constrained compute die, below.
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
Use separate backing storage and a local active-row cache for the constrained
compute die as necessary; computation remains on all six intended compute dies.
Budget cache, dense weights, one-head KV/indexer state and scratch together.
Check peer accessibility and account for cache fill time in phase measurements;
remote traffic must not be relabeled as useful arithmetic or excluded to force
an apparent±10% pass. Do not deploy an allocation that relies on the fenced band.

## Qualification and publication

Reuse canonical GPUQ evidence before starting new work. All archy GPU execution
continues through `ssh archy gpuq run -- COMMAND`. Preserve one canonical
history.sqlite3, immutable source/images and the exact workload. The completed
candidate must execute the full production loop before extra diagnostic tests.
Then check same-image state/token repeatability before cross-image comparison.
The final evidence must include **every phase** and report actual TPS, memory,
cache traffic and residual imbalance; do not qualify only phases that pass.

This goal is not complete. No±10% runtime balance or TPS improvement is claimed.
