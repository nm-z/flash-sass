# DeltaNet and QSA wait dependencies

This is the dependency deliverable requested before further merging. It describes the complete single-position production model, all 48 layers, six dies, no MTP or FP8. It distinguishes a data dependency from the mechanism currently used to enforce it. An in-operation spin is still a wait.

## Serving state

The live engine was still serving the 32.99 TPS phase-fused candidate when inspected. GPUQ job `0ae67de8-e4b3-47db-b329-aaca5463c685` switched serving back to the measured baseline cubin, FNV **`9abf8aafb9134d59`**, with `producer_quant=0`, `fused_gr=0`, `matrix_group=0`, `balanced=0`, MTP disabled and FP32 residuals. A subsequent `config` read confirmed that state. The same resident weights remain loaded. This was a reload, not a new benchmark; 40.11 TPS is its prior matched measurement.

## Full schedule census before revision

The offline compiler calls production `model_program()` with the same holder metadata and split. It performs no GPU arithmetic. Baseline counts reproduce the existing runtime counts; candidate counts reproduce 499 on every die.

| Mechanism per token per die | Baseline | Existing 499-phase candidate |
|---|---:|---:|
| Logical phases | 1,178 or 1,230 | 499 |
| End-of-phase local grid joins (`BSPIN`) | 1,042 or 1,082 | **499** |
| Exchange publications | **354** | **354** |
| Exchange acquire waits (`XWSPIN`) | **354** | **354** |
| Footer system-fence threads (16 per publication) | **5,664** | **5,664** |
| Requested inner grid joins in grouped chains | 0 | 0 |

The candidate did not reduce the number of cross-die exchanges. Its 499 phase ends still have local grid joins. These are structural counts; no wait durations are inferred from them.

**354 of those local grid joins are redundant with the following exchange acquire.** For every sending phase, the next logical phase has the exchange-wait flag. Each block publishes its own completion only after its output stores are fenced. The next phase polls **six dies × sixteen blocks**, including this die's own sixteen blocks, before reading the payload. The producer's additional local sixteen-block join establishes no data visibility that this acquire does not already establish.

The removable component is the producer's local grid polling, **not** its output fence, peer flags, generation update, CTA handoff, or the consumer's acquire. Keep the descriptor span and phase count unchanged. Empty senders still advance the exchange protocol and cannot unilaterally omit their flags.

There is one further removable local grid join in the PLE prefix: grouped key/query RMS → value GEMV. The value GEMV reads gathered PLE rows, not either normalized result. The later value publication/acquire includes block 0, which produced both normalized vectors, before PLE gating consumes them.

## DeltaNet attention half

Each row identifies the data which must become available, its producer and consumer, and the minimum scope. “Peer acquire” is the existing 96-flag `XWSPIN`; it already includes the local producers.

| Edge / wait | Data crossing | Producer → consumer | Required synchronization | Disposition |
|---|---|---|---|---|
| Previous MoE → GR read (`grwrite_read`) | Six 2,560-element MoE partials; residual S and injection gates | Six dies' row-owning down-projection blocks → every GR-read block | Peer acquire; preserve fixed-order sum | Keep acquire; remove prior sender's duplicate local grid join. Full sum cannot start before all six contributions are ready. |
| GR down read → HCUP | Distributed down/injection rows | 16 blocks per die producing this die's GR rows → all HCUP blocks | Peer acquire | Remove sender's duplicate local grid join. |
| HCUP → input projection group | Quantized Z, sums, scales; FP32 Z for BF16 alpha/beta | Each die's GR-up column blocks → QKV/gate/alpha-beta projection blocks | Peer acquire | Remove sender's duplicate local grid join. |
| Within input projection group | No payload between QKV, gate and alpha/beta projections | Independent row producers, all reading Z | CTA descriptor handoff only; no grid dependency between members | Already grouped without an inner grid join. Descriptor-bank reuse still requires cross-warp ordering. |
| Input projections → GDN body | Q/K/V inputs, output gate and alpha/beta rows | GEMV row-owning blocks → the block owning each recurrent head | Local grid join in existing ownership | Keep: the head block can need rows produced by other blocks. |
| GDN convolution → state update | Head Q/K/V convolution outputs | First three warps of head-owning block → its state-update warps | Shared-memory CTA rendezvous | Keep. No other block/die is involved, but other **warps** are; this is not an empty dependency. |
| State update → norm/gate/quantize | 128-element head output | State-update warps → warp 0 of the same head block | Shared-memory CTA rendezvous | Keep. |
| GDN output → output projection | Global 6,144-element normalized/gated activation, codes/sums/scales | Head-owning blocks on six dies → output GEMV blocks | Peer acquire | Keep acquire; remove GDN producer's duplicate local grid join. |
| Output projection → residual write / MoE GR read | Distributed output Y; injection gates from prior GR read | Each die's projection row owners → all residual-read blocks | Peer acquire | Keep acquire; remove sender's duplicate local grid join. |

## QSA attention half

GR-down → HCUP and HCUP → projection waits are the same as DeltaNet above. The QSA owner runs the core; five other dies produce Q shards and later consume the merged activation.

| Edge / wait | Data crossing | Producer → consumer | Required synchronization | Disposition |
|---|---|---|---|---|
| Q/K/V/indexer matrix group → QSA setup | Five dies' Q shards; owner's K/V/indexer projections | Row producers on five Q workers plus owner's local matrix blocks → owner cache/indexer warps | Peer acquire includes owner local completion | Keep acquire; remove the sending phase's duplicate local join. Owner's empty Q participation still publishes protocol flags. |
| Cache append → indexer update within group | None between these bodies | Four cache-append blocks and block 0 indexer warps operate on independent state | CTA descriptor handoff | No inner grid join; no producer-ready spin between these bodies. Four-to-one work assignment itself is not a cross-block wait. |
| Cache/indexer group → score/selection | Indexer query, updated key packet/ring, cleared histogram/ready area | Indexer warp group → score blocks | Local grid join | Keep. KV append also completes here before its later consumers. |
| `SCORE_JOIN_WAIT` | Scores, first-pass histogram and 16 ready words | Score blocks 0–15 → selection block 0 | Consumer warp polls producer-ready words; block-local parameter handoff | Keep. At short context, score arithmetic early-exits; ready signaling still occurs. |
| Within radix selection | Histogram/reduction results, threshold and tie-order state | Selection block's warps → later passes in the same block | Shared-memory CTA barriers | Keep; pass 5's old cross-block count is eliminated because only one selection block remains. |
| Selection → KV gather | Selected key-block IDs | Selection block 0 → gather blocks | Local grid join | Keep. |
| KV gather → attention partitions | Compact K/V and quantization scales | Gather blocks → all 16 partition blocks | Local grid join | Keep. |
| `PM_HEAD_WAIT` | Partition maxima, denominators and partial output | Four relevant partition blocks per head → merge-head block | Four-producer ready join | Keep; not an all-grid join. 32 work items replay on 16 physical blocks. |
| Merge → attention output projection on every die | Gated merged activation codes/sums/scales | Owner's merge blocks → all dies' output projection blocks | Peer acquire | Keep acquire; remove producer's duplicate local grid join. Empty nonowners retain their participation flags. |
| Output projection → residual write / MoE GR read | Distributed Y rows | Six dies' output GEMV row owners → GR-read blocks | Peer acquire | Keep acquire; remove duplicate local join. |

## Common MoE half

| Edge / wait | Data crossing | Producer → consumer | Required synchronization | Disposition |
|---|---|---|---|---|
| GR read → HCUP → router projection | Down/injection rows, then Z codes/sums/scales and FP32 Z | GR row/column producers across six dies → local router projection | Two peer acquires | Keep acquires; remove duplicate producer local joins. |
| `ROUTE_JOIN` | 513 router logits and 16 completion words | Router GEMV blocks → warp 0 of block 0 | Producer-ready join | Keep: router selection reads logits from other blocks. Parallelizing/restructuring selection is a separate arithmetic change. |
| Router/select → routed gate/up group | Local expert IDs, weights, count, initialized partials | Block 0 router/selection warp → all routed expert blocks | Local grid join | Keep. |
| Routed gate/up → activation within fused body | Gate/up intermediates for the same expert rows | Same row-owning block's warps → its activation consumer | Shared-memory CTA barriers | Keep; preserves arithmetic/row ownership. |
| `SHARED_ACT_WAIT` (eight layers per die) | Shared gate/up matrix outputs and producer-ready words | Shared projection blocks 0–15 → block 0 activation warp | Producer-ready join | Keep: activation reads shared-expert rows from all producer blocks. |
| Gate/up/activation group → down | Expert activation codes/sums/scales | Activation producers → down-projection row blocks | Local grid join | Keep. |
| Routed down → shared down accumulation | Same output rows and their routed partial | Same block across both descriptors | Local visibility fence / CTA handoff, no grid join | Keep cross-warp/descriptor ordering; no cross-block join is requested. |
| Four Q6 slices | Previous slice's output-row accumulation | Same row-owning block across four descriptors | Local visibility fence / CTA handoff, no grid join | Keep original addition order. No inter-block payload is introduced. |
| Down publication → next residual sum/read | All six dies' MoE partial vectors | Six dies' row producers → all GR-read blocks | Peer acquire | Keep acquire; remove duplicate sender local join. Final layer instead feeds retained HCWSUM then output head. |

## PLE, output and framework waits

| Wait | Data / producer → consumer | Required scope | Disposition |
|---|---|---|---|
| `plegatherq_WAITRDY` | Host-prefetched PLE rows and request-generation readiness → gather warp | Host/device acquire | Keep. This is not a GPU block-barrier problem. |
| PLE gather → key projection | Gathered/quantized activation from one warp → distributed key GEMV blocks | Local grid join | Keep. The fused gather's sequential ten-vector quantizer is a work-distribution cost. |
| PLE key publication → key/query RMS | Gathered key from six dies; residual S | Peer acquire | Keep acquire; remove sender's duplicate local join. |
| Grouped key/query RMS → PLE value GEMV | **None:** value GEMV reads raw gathered PLE rows | No local grid join | Remove this join; later value exchange also joins the RMS-producing block before gating uses its results. |
| PLE value publication → gate/norm | Gathered V plus previously normalized K/Q | Peer acquire | Keep acquire; remove sender's duplicate local join. |
| Gate/norm → PLE convolution | Four px residual branches produced by four gate blocks | Local grid join | Keep. Warp-0 normalization within each gate block needs its local CTA handoff. |
| PLE convolution → next GR read | Updated residual S and local convolution state | Local grid join | Keep. |
| Embedding → first GR read | Owner writes four residual branches directly to every die | Peer acquire | Keep acquire; remove sender local join. Empty receiver participation is still required by the current protocol. |
| Output projection → argmax | Logits written by all local row blocks → argmax scan blocks | Local grid join | Keep. |
| Argmax/top-k internal final reducer | Per-block maxima/top-k lists and completion counter → winning reducer block | Existing atomic completion/reducer protocol and local scratch barriers | Keep; these carry real cross-block data. Top-k was not exercised in the last greedy run. |
| Argmax/top-k publication → AMAX/sample | Six dies' token/value or top-k lists | Peer acquire | Keep acquire; remove sender local join. |
| AMAX/sample → next token | Chosen ID, output token and stop flag from one warp | Local grid join and token-state handoff | Keep: embedding and PLE hash need the chosen ID. |
| Descriptor install / grouped replay | Warp 0 writes descriptor banks and token-adjusted words → all operation warps | CTA shared-memory handoff | Keep. No model activation crosses, but control metadata does. |
| `POSITION_END` / group cursor handoff | All warps finish body and sample old mode/cursor before warp 0 overwrites it | CTA handoff | Keep. Removing it based only on absence of inter-block data would race descriptor/mode reuse. |
| `PHASE_END` entry CTA barrier | No new shared-memory stores after the preceding footer rendezvous on the ordinary/group paths | Potential redundant CTA join | Separate control-flow proof needed for every specialized entry; not removed merely by textual adjacency. |
| `CHAIN_BSPIN` / legacy `xwait_SPIN` | Legacy generic grid/peer protocol | Not executed by this candidate schedule | Candidate requests zero inner grid joins and no XWAIT opcode. Do not count dormant code as runtime waits. |

## Fence hypothesis and occupancy

The old XSEND path and the fused publication path both finish through a footer where **thread 0 of each of sixteen blocks executes `MEMBAR.SYS` once per exchange**. Both schedules have 354 exchanges and therefore 5,664 such fence-thread executions per token per die. The producer publication emitter adds `MEMBAR.GL` and a row/peer copy loop, not a per-row `MEMBAR.SYS`. The hypothesis that fusion changed one system fence per phase into sixteen is not supported by this code. These are static executed-path counts, not hardware fence-duration measurements.

The candidate's 249 registers/thread at 256 threads/block require 63,744 registers before allocation rounding. Maxwell provides 65,536 registers and up to 64 resident warps per SM; thus two such blocks cannot coexist on an SM. Sixteen persistent blocks supply at most **eight resident warps per SM**, or 12.5% of the architectural warp limit. Source-derived occupancy is not measured issue efficiency. Reducing only the block count cannot improve this. Increasing blocks at the same register footprint can also deadlock the software grid barrier if some blocks cannot become resident. A higher-occupancy design requires a register/thread-layout and barrier redesign, not an unchecked launch-size edit. [NVIDIA Maxwell Tuning Guide](https://docs.nvidia.com/cuda/archive/11.6.1/maxwell-tuning-guide/index.html)

## Evidence and next implementation

The per-phase CSV identifies all 499 boundaries on all six dies, each group's actual member opcodes, producer payload, and remove/retain decision. It complements these template tables; template rows repeat for all 36 DeltaNet and 12 QSA layers, with eight shared-expert layers on each die.

The first complete revision can remove **354 redundant producer local grid joins plus the independent PLE RMS join per token per die**, leaving 144 end-of-phase local grid joins and all 354 necessary peer acquires. It must retain peer publication, local descriptor handoffs, generation accounting, and the sub-500-phase admission guard. This is wait removal with unchanged model operation grouping. No throughput or output-match claim follows from this audit; the whole revised kernel must be built and run before those results are reported.
