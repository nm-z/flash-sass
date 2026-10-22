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

Jev search failed with `Search backend uses legacy retrieval; rebuild and
deploy the unified source-context-v1 index`. Feedback was filed as
[nm-z/ai-mux#627](https://github.com/nm-z/ai-mux/issues/627); repository source
and the canonical GPUq DB supplied the evidence used here.
