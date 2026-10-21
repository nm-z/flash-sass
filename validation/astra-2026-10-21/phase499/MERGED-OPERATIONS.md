# Merged operation timing and useful block participation

This report reads the existing canonical GPUQ database. **No new GPU run or kernel modification was performed.** The source job is `03bff6fb-db1f-4844-9d91-0e2e381ea17f`. Its benchmark issued `prof DIE all` for both kernels on all six dies, equivalent to `PROF_ALL=1 kern/prof.sh`.

[Complete opcode table, including all six per-die tables](operations.md) · [CSV](operations.csv) · [Raw recorded output](03bff6fb-db1f-4844-9d91-0e2e381ea17f.json)

All times below are **µs per final token, averaged over six dies**. They are inclusive phase intervals, with synchronization and peer waits. They are not isolated arithmetic timings or request medians. Attention ownership changes from four layers each on dies 1/4/5 to two layers on every die. Averaging the totals over all six dies accounts for the same twelve layers.

## What the existing records can resolve

The DB contains complete **opcode totals**, not timestamps for every original descriptor. That distinction matters: `actq256` combines quantization for HCUP, GDN, PLE and attention; `xsend` combines many different producers; `bf16k2560` combines router, PLE and indexer projections. Grouped candidate phases are charged entirely to the first opcode. Their suboperations have no independent recorded duration.

Consequently an exact numeric baseline sum for **every individual merge** cannot be recovered from this run. The following table preserves the unknown terms rather than spreading an opcode's time evenly over unrelated operations. A term such as `A_HCUP` means the recorded time of that subset of baseline ACTQ phases, which is not separately retained. It is not an estimate.

| Merged candidate operation | Baseline operations replaced and recorded contribution | Candidate µs/token/die | Timing limit |
|---|---|---:|---|
| `grwrite_read` (47 phases) | 47 `hcwsum` + 47 `bf16hcrx` | 5,621.6 | Baseline mixes these with the final write and two retained reads. Full family: **3,541.3 → 5,767.7**, including retained operations on both sides. |
| `hcupq_x` (97) | `bf16hcup` + its ACTQ subset = **2,378.3 + A_HCUP** | 3,272.0 | ACTQ subset not separated. |
| `gdn_full` (36) | `conv1d` + `dnstep` + `dnnormx` + its ACTQ subset = **1,333.6 + A_GDN** | 1,176.2 | Includes recurrent update, normalization/gating and quantized publication. |
| `router_full` (48) | Router BF16 projection + `router` + `moesel` = **R_router + 1,296.1** | 2,579.1 | Router projection is mixed with PLE/indexer in baseline `bf16k2560`. |
| Routed gate/up/activation and shared gate/up/activation (`q2moegn`, 48) | Two routed `q2moegn` bodies + `ffnact` + shared gate/up GEMVs = **2,097.1 + S_gate_up** | 1,871.5 | Eight shared-expert layers per die; their matrix times are mixed with other same-format GEMVs. |
| Routed/shared down + publication (`q2moedn`, 48) | `q2moedn` + all shared down accumulation opcodes + their publication = **1,007.9 + X_MoE_down** | 1,771.0 | Includes `q2acc640`, `b32q40acc`, `b32q50acc`, `b32q80acc`, `b32iq4nlacc` in the baseline subtotal. |
| Four Q6 K1536 slices + publication (`q6kk1536`, 6) | `q6kk1536` + three `q6kk1536acc` + publication = **321.5 + X_Q6** | 1,088.5 | Baseline publication is in the shared XSEND pool. |
| `qsa_score_select` (2) | `idxscore` + five `idxsel` phases = **32.6** | 11.9 | Exact opcode-family sum across all dies. At this short context the scoring/radix work takes its early exit; this does not measure full-context radix performance. |
| `qsa_part_merge` (2) | `attnpartn` + `attnmerge` + attention publication + attention ACTQ = **143.7 + X_attn + A_attn** | 190.5 | Other dies' removed attention ACTQ also belongs to `A_attn`. |
| Cache append + indexer update (charged to `attnqkq`, 2) | `attnqkq` + `idxqk` = **105.4** | 306.0 | Exact opcode-family sum; includes changed wait attribution and ownership. |
| `ple_rowsq` (1) | `plerows` + PLE ACTQ = **55.6 + A_PLE** | 65.3 | Only the gather-producing warp now performs all ten quantization vectors. |
| PLE gate/norm + grouped PLE key/query norms | `plegate` + all three `rms2560` = **31.8** | **67.4** | Complete PLE family: 26.3 in `plegate_norm` + 41.1 in grouped `rms2560`. Individual norms not separately timed. |
| Output argmax + publication (`argmax`, 1) | `argmax` + its XSEND = **10.2 + X_argmax** | 13.0 | Final AMAX stays separate; no top-k was executed in this greedy run. |
| Embedding + direct publication | `embrow` + broadcast/copy | Not recorded | EMBROW is the first timestamp; PROF_ALL omits the preceding interval. The baseline broadcast/copy is mixed with other XSEND/XCOPY work. |
| Independent matrix groups (41 groups/die) | GDN input/gate/bias and QSA Q/K/V/indexer GEMVs | See complete opcode table | Primary opcode totals combine several group shapes and retained single operations. No per-group timing survives in this DB record. |

Baseline pool totals: **ACTQ = 2,800.8**, **XSEND = 1,562.5**, **BF16K2560 = 1,380.0** µs/token/die. These are shared pools, not additional terms to add to each row. Candidate producer publication also appears in unchanged projection names:

| Publication-bearing projection opcode | Baseline body subtotal | Candidate inclusive total | Baseline counterpart |
|---|---:|---:|---|
| `q4kk6144` | 404.7 | 1,719.8 | Body + its XSEND subset |
| `iq4k6144` | 473.4 | 1,300.8 | Body + its XSEND subset |
| `q5kk6144` | 267.9 | 1,189.5 | Body + its XSEND subset |
| `q3kk6144` | 99.2 | 245.2 | Body + its XSEND subset |

These rows identify substantial inclusive increases but do not by themselves isolate publication overhead. Matrix grouping and timing charged after peer exchanges also move waits between opcode buckets.

A broader quantization family can be summed without guessing the ACTQ allocation: baseline HCUP + convolution + GDN update/norm + PLE gather + ACTQ + attention partition/merge totals **6,712.0**, versus candidate HCUPQ + GDN_FULL + PLE_ROWSQ + QSA_PART_MERGE **4,704.1**. This subtotal excludes baseline attention XSEND. Its difference is not a causal fusion speedup.

## Blocks that perform useful work

Every phase belongs to **one persistent launch of 16 blocks × 256 threads per die**. There is no new grid launch per operation. A descriptor's `units` may mean blocks, warps, heads or virtual work items; it cannot be read as active SM count. Nonworking blocks still execute framework synchronization and may spin while waiting.

The counts below follow the production planner and SASS dispatch, using split `112,97,81,116,97,97`. They are **source-derived participation, not hardware activity measurements**. “At most N SMs” gives the maximum useful arithmetic footprint of N blocks; it does not mean those SMs stayed busy for the full phase. Per-phase SM IDs, busy cycles and active-warp counters are absent from the DB's documents and metrics. GPU utilization samples cannot supply them.

| Merged operation | Blocks with useful operation work | Useful SM footprint established by source | Work distribution / serialization |
|---|---|---|---|
| Residual write/sum + normalized GR read | 16 | At most 16 | Every block repeats the normalization/reduction prologue, then owns matrix rows. |
| HCUP + quantization/publication | Main column work: **14,13,11,16,13,13** on dies 0–5; setup runs in all 16 | Main column stage at most the corresponding count | Fused code assigns 32 columns per block. Quantization/publication uses warp 0 of producing blocks. Do not interpret 16 descriptor units as 16 useful column blocks on every die. |
| Full GDN | **8,8,7,9,8,8** | At most **8,8,7,9,8,8** | One local head per block. Convolution uses the first three warps; normalization/quantized publication uses warp 0 after the state update. |
| Router projection + router/selection | 16 projection blocks → **one warp in block 0** | At most 16 → 1 | The consumer waits on 16 producer-ready words, then routes/selects alone. |
| Routed gate/up/activation | Up to 16; routed GEMV arithmetic is skipped if this die has zero selected experts | At most 16; runtime expert-dependent | DB does not retain per-layer local expert counts. Shared gate/up bodies use 16 blocks on the eight shared-expert layers assigned to each die; shared activation ends in block 0, warp 0. |
| Routed/shared down + publication | Up to 16 routed blocks; shared body uses 16 where present | At most 16 | Routed body can skip with zero local experts; shared accumulation retains row ownership. |
| Four Q6 slices + publication | 16 | At most 16 | Same block owns the output rows across all four slices; slices execute in order. |
| QSA score + selection | Scoring: up to 16 when the selection threshold is exceeded; selection: block 0 | At most 16 → 1 | **The saved short-context run has no score-dot-product work:** number of pooled keys is below 512. All blocks still run control/ready signaling; block 0 runs the selection early-exit path. |
| QSA partition + merge/quantization | 16 physical blocks | At most 16 | 32 partition work items are handled as `cta` and `cta+16`; 24 merge heads are also replayed over the same 16 blocks. **Not 32 blocks/SMs.** Each merge head joins its four producing blocks. |
| QSA cache append + indexer update | 4 blocks for 26 append warp units; 1 for 6 indexer warp units | At most 4, then 1 | The grouped dispatcher replays the second body in the same persistent blocks. |
| PLE rows + quantization | **1 warp in 1 block** | At most **1** | All ten ACTQ vectors now run sequentially in the producing warp. Baseline ACTQ used two blocks (ten warp work items). |
| PLE gate + normalization | **4** → warp 0 of each of those 4 | At most **4** | One residual branch per block; each normalizes its own branch. Baseline normalization used four warps in one block. |
| Grouped PLE key/query RMS | **1**, four warps per normalization | At most **1** | Grouping does not distribute four vector units across four blocks. |
| Independent matrix groups | Typically 16; small row shapes can have fewer useful row-owning blocks | At most 16 | Grouping replays bodies; it does not combine their grids into more resident blocks. Format and row count determine work. |
| Projection + publication tails | Same row-owning blocks as producer; common K6144 projections use 16 | At most 16 | Publication is performed by valid subgroup leaders, with uniform block-level loop/barrier control. |
| Embedding direct publication | **10** on embedding owner; no embedding body on other dies | At most **10** on owner | All six destination residuals are written directly. Framework participation remains on receiving dies. |
| Argmax + direct publication | 16 scan blocks → one reducing block, then lane 0 publication | At most 16 → 1 | Reducer is the winning/last-arriving block, not necessarily block 0. Top-k uses a warp for publication but was not executed in this run. |

Source references: [`engine.c` launch and profile implementation](../../../kern/engine.c), [`phase/plan.h`](../../../kern/phase/plan.h), [`phase/quant.lua`](../../../kern/phase/quant.lua), [`phase/gdn.lua`](../../../kern/phase/gdn.lua), [`phase/router.lua`](../../../kern/phase/router.lua), [`phase/shared.lua`](../../../kern/phase/shared.lua), [`phase/score_select.lua`](../../../kern/phase/score_select.lua), [`phase/part_merge.lua`](../../../kern/phase/part_merge.lua), [`phase/ple.lua`](../../../kern/phase/ple.lua), [`phase/ple_rows.lua`](../../../kern/phase/ple_rows.lua), [`phase/edges.lua`](../../../kern/phase/edges.lua), [`die.sass`](../../../kern/die.sass), [`prog.c`](../../../kern/prog.c), [`ref/split.c`](../../../ref/split.c).

The offline production schedule compiler confirms the candidate descriptor units and group widths. It performs no GPU arithmetic. Its output is used only for structural counts, not timings or measured occupancy.
