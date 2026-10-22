# Production decode toward 60 TPS

Target: at least60 committed target tokens/s, retaining the complete model,
correct outputs, fewer than500 physical phases/token and the pairwise
assigned-work bound. Physical GPUs2 and7 remain reserved for Nate.
**The target is not achieved.**

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

## Current v85 layout

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

The next substantial cost is the routed-down six-owner loop, which scans the
ten selected IDs six times to preserve original partial-sum association. A
single expert scan with per-owner accumulators can preserve that association
without repeating the expert-ID dispatch work. This remains unimplemented;
60 TPS remains the active objective.

Jev search failed with `Search backend uses legacy retrieval; rebuild and
deploy the unified source-context-v1 index`. Feedback was filed as
[nm-z/ai-mux#627](https://github.com/nm-z/ai-mux/issues/627); repository source
and the canonical GPUq DB supplied the evidence used here.
