# Production decode toward 60 TPS

Target: at least60 committed target tokens/s, retaining the complete model,
correct outputs, fewer than500 physical phases/token and the pairwise
assigned-work bound. Physical GPUs2 and7 remain reserved for Nate.
**The target is not achieved.**

Current default candidate: **v91,24.793978201 TPS**, with492 phases/token.
Short-context reference outputs, all288 retained state hashes, and the
long-context baseline outputs pass. The unsuccessful compact-Q2 and local-GR
experiments are retained below; they are not enabled in the default image.

Priority scores for this work: throughput1.0, fidelity1.0, retaining the
established work/phase requirements1.0, engineering time0.8, cost0.8.
The initial lower-effort approach increases down-projection CTA width without
changing weight ownership; its expected benefit is fewer serial virtual-unit
iterations. The larger approach changes expert publication layout so several
ranks can travel in contiguous peer transactions; it requires coordinated
producer and consumer changes. The former has measured benefit; the latter
is the next step after the scatter-publication regression below.

All results are full64-token prompt/128-generated requests using the same
frozen original reference, scalar decode, MTP off and the six persistent
balanced holders. These are individual observations, not matched randomized
A/B estimates. Raw output and metadata remain in archy's canonical GPUq DB
`/home/nate/.local/state/gpuq/history.sqlite3`.

| Version | Canonical job | Decode TPS | Result |
|---|---|---:|---|
| v79, prior kernel | `89cebaae-2527-4d69-832e-c54c9f093664` |20.214920064|128 reference tokens and head audit pass |
| v81,32-row down tiles | `f435f4cd-4128-47e5-9cd9-382a0d4d0e61` |22.076207033|128 reference tokens, head audit and work readback pass;288 state hashes match v78 |
| v82,dual-expert gate execution plus packed peer scatter | `a6abb0b1-9ae7-4aeb-92a2-e25ff9b1cb99` |20.595120176|Correct128-token output, slower |
| v83,single-expert gate execution, packed peer scatter, parallel GR quant publication | `78649a44-ddd4-4196-934b-9cd700e8c102` |20.644140115|Correct128-token output, still slower |
| v84,rank-contiguous gate publication, wide down/shared publication and parallel GR quant publication | `14f3aa0b-a173-4601-8592-c1ce73e8a85a` |22.590515684|128 reference tokens, head audit and work readback pass;288 state hashes match v78 |
| v85,contiguous packed gate exchange with local consumer remapping | `b8d8433c-2ba9-4af0-b1cf-9cb9ab4bfc50` |23.437196538|128 reference tokens, head audit and work readback pass;288 state hashes match v78 |
| v86,single expert scan with original per-owner sums | `3d14d991-de83-4276-bfa2-0160772649fe` |23.665795418|128 reference tokens, head audit and work readback pass |
| v87,two gate-expert streams with contiguous publication | `50bd52a9-62e9-4463-ad0c-fb23711889ef` |23.899208898|128 reference tokens and all288 v78 state hashes match |
| v88,compact routed Q2 arithmetic and wide-tile tail guards | `825f9ea3-7ccc-4e75-8787-b5b9f7e6382f` |23.649676216|128 reference tokens and head audit pass; compaction not promoted |
| v89,complete local GR-up matrices | `2da84d75-b6ba-4d0c-9599-b8269e0dfd47` |23.449704315|128 reference tokens and head audit pass;493 phases,306 publications; replication not promoted |
| v90,unrolled default with tail guards | `3ea9c188-f9b1-4f5a-b36d-dcda6ea76c44` |23.794250711|128 reference tokens and head audit pass |
| v91,independent load bursts | `6237e3e2-a49f-44a5-86f3-8662d22b6d83` |24.793978201|128 reference tokens, head audit and work readback pass;288 state hashes match v78 |

The down-projection phase contains both routed and shared projections. On
logical die5 its accumulated profile time falls from10.670ms/token in v79 to
6.290ms in v81 and5.755ms in v84. GR up/quant phases fall from6.290ms in v79
to5.939ms in v84. These profile times include boundary effects; they are not
isolated arithmetic timings. The full request timing is the TPS authority.

## Changes retained at v84

Eight-row weight ownership is unchanged. K640 down projections instead pack
four ownership stripes into one32-row CTA unit. Routed and shared projections
use identical ownership and retain the original six-group FP32 accumulation
order. Their last tiles mask invalid rows. The routed down body exposes four
arithmetic warps per CTA instead of one, with14 units per die rather than53/54.
This changes no physical phase or selected expert.

Single-rank shared/down publication uses one warp per peer. GR quantization
likewise has six warps read the same completed shared-memory tile, perform
identical quantization and publish one peer each. The work audit counts those
replicated quantization evaluations. The ABI marker prevents loading the
32-row descriptors with the old8-row production image.

v84's actual retained program passes all7,188 geometry comparisons over492
phases and979 descriptors per die. Maximum pairwise work ratio remains
1.076923077. See [the actual work table](work-v84.tsv) and
[summary](work-v84-summary.txt). The actual readback includes all compound
members. All six last-layer query vectors and expert-ID lists agree.

## Regression retained as evidence

Packing four expert ranks into one warp while retaining the destination
layout `[rank][640]` made the writes scatter across rank strides. Gate/up
profile time increased from about7ms/token to9.4–11.3ms across dies. Removing
dual-expert compute did not remove that regression. v84 restores contiguous
per-rank gate writes while retaining the wider down execution and other
publication changes. All immutable candidates remain in their numbered
directories under `/home/nate/work/flash-per-phase-balance-vNN` on archy.

## Contiguous exchange layout introduced in v85

The routed gate/up exchange destination is now indexed by owner, local
eight-row tile, rank and row. Each producer CTA's ten ranks are80 consecutive floats. Six
warps each publish one peer; a warp carries four ranks in contiguous remote
addresses. The local source remains rank-major. There is no padding in the
6400 routed values, and the640 shared values remain at their original offset.

Before activation quantization, the down-projection preparation reads a static
7040-entry mapping table to restore the original rank/neuron order. The
quantized down input and all floating-point arithmetic keep their original
order. Each die adds28,160 mapping bytes and7040 coalesced index reads per MoE
layer, with no extra phase. A new module ABI marker requires the matching
consumer layout and rejects the previous engine/image combination.

The v85 full-model run measures23.437196538 TPS. Gate/up profile time on die5
falls from7.038ms/token in v84 to5.281ms; the entire timestamped token interval
falls from44.24ms to42.59ms. The actual descriptor readback passes all7,236
work-geometry comparisons, including the mapping reads and replicated GR
quantization, with the same492 phases and1.076923077 maximum pairwise ratio.
See [v85 work table](work-v85.tsv) and [summary](work-v85-summary.txt).

All128 generated IDs match the original frozen reference, all288 retained
per-die/per-layer state hashes match v78, and the head audit has zero failures
over248,320 logits. Removing unreachable generator branches after this run
produced byte-identical generated MoE SASS, so no redundant GPU run was added.

## Single expert scan and paired gate execution

The routed-down loop now scans each selected ID once. A512-entry immutable
owner lookup selects its original partial-sum group. A six-bit per-thread
mask preserves the first-product multiplication versus subsequent FMA choice
for each owner. The six partials are still reduced in the original order,
including the shared expert in its original group. The v86 down phase totals
about5.3ms/token, versus about5.9ms in v85.

Gate/up now uses the two CTA halves for even and odd expert ranks. Each warp
executes a uniform five-rank loop; publication joins all ten results at the
existing CTA barrier. This is the previously tested concurrency change
re-evaluated after fixing the remote destination layout. v87 measures23.90TPS.

Wide down tiles also now skip completely invalid warps before entering native
unpredicated loads. Every supported K640 format has four lanes per row and
row counts divisible by eight, so the guard is warp-uniform. It preserves
the existing publication/chain rendezvous and avoids relying on tensor or
allocation padding for the last tile.

## Load scheduling retained in v91

`kern/balance/load_bursts.lua` combines adjacent unpredicated LDG/LDS loads
only when their physical register destinations and address operands are
independent, their scoreboard slots and memory pipeline agree, and no
existing wait consumes an earlier load. It retains the last completion wait
and all original wait bits. The elapsed minimum instruction spacing is
preserved by stalls, including otherwise idle static spacing. Unknown address
aliases are excluded. No arithmetic, memory operation, label or phase is
reordered.

The complete image contains72 eligible bursts, removing589 intermediate
completion waits. Register count remains249. On logical die5 the QSA phase
total is2.035ms/token in v91; GDN totals1.981ms. Profile totals include boundary
effects and are not isolated arithmetic timings.

v91's retained program passes all7,332 work-geometry comparisons, with492
phases,979 descriptors per die and maximum pairwise ratio1.076923077. See
[short work table](work-v91.tsv), [short summary](work-v91-summary.txt),
[long work table](work-v91-long.tsv), and [long summary](work-v91-long-summary.txt).

Canonical long-context job `006cca44-f281-4153-988e-4be2af748492` processes a
4096-token prompt and executes4097 causal positions. Its two output IDs match
the prior v80 balanced baseline (`44552a00-9342-4ec0-a0de-5ba61c4a76bd`), whose
IDs are retained in [v80-long-reference.ids](v80-long-reference.ids). The final
head audit passes. All six dies have identical512-value indexer queries,
1024 selection scores,512 selected block IDs and ten distinct expert IDs.
This is a long-context baseline-preservation check, not an independent
original-model oracle or a two-token throughput benchmark.

## Completed experiments not enabled by default

`balance/compact_q2.lua` folds five identical68-instruction half-block bodies
into one loop, binding operands into free registers while preserving all
original values and arithmetic order. It passed the full model, but its
operand-copy and loop overhead did not demonstrate a throughput gain.

`FLASH_BALANCED_LOCAL_UP=1` with `die_balanced_localup.cubin` replicates all97
GR-up matrices, preserving their BF16 bits. It uses about606MiB more VRAM per
die and replaces97 peer acquires with local completion. The complete v89
schedule has493 phases,306 publications and305 peer acquires; the extra phase
is the existing parity guard. GR-up time rises to about6.9ms/token and total
decode is23.45TPS, so this experiment is opt-in. Its distinct module marker
rejects a local-up program paired with an ordinary image.

The existing v78 exchange trace was also read from the canonical database;
no GPU run was repeated for this analysis. It contains19296 valid records
over eight token positions. Sum of local body finish to acquisition is
4.08–4.62ms/token across dies. This interval includes publication, transport,
polling and rendezvous; it is not isolated link latency or the complete cost
of producing peer payloads. [Operation-level bounds](exchange-v78-operations.csv)
retain causal-clock uncertainty rather than subtracting unaligned GPU clocks.

60 TPS remains the active objective. Further work must address substantial
matrix execution and expert data movement; these measurements do not support
claiming that reducing the remaining completion waits alone reaches the target.

Jev search failed with `Search backend uses legacy retrieval; rebuild and
deploy the unified source-context-v1 index`. Feedback was filed as
[nm-z/ai-mux#627](https://github.com/nm-z/ai-mux/issues/627); repository source
and the canonical GPUq DB supplied the evidence used here.
