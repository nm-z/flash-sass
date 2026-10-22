# Pairwise work geometry of the complete production kernel

**Result:** the complete current kernel satisfies the clarified allocation
target: 492 phases/token, with maximum pairwise assigned-work ratio
1.076923077. Both short-context and long-context retained production programs
pass all 7,091 component comparisons. This does not claim ten-percent timing
balance or a throughput speedup.

Nate's clarified requirement is fewer than 500 physical phases per token and,
inside each phase, no die assigned more than 1.10 times another die's work.
The criterion is pairwise, not deviation from the mean and not an average
across token positions. Elapsed time remains separate performance evidence.

## Retained production program

Canonical GPUq job `89cebaae-2527-4d69-832e-c54c9f093664` completed the full
48-layer model and full output head, with MTP and FP8 off. After inference,
`workdump` copied the actual retained `dPROG` descriptors from each compute
GPU. It did not launch an audit kernel or alter the program.

Each of six dies has **492 physical phases and 979 descriptors**. The audit
includes every child of each compound phase, including QSA's seven child
stages, GDN's convolution descriptor, routed/shared projections, and all four
K1536 slices of Q6 K6144 projections. It covers 44 operation types, 5,874
descriptor records and 7,091 six-die component comparisons. All pass the
pairwise geometry bound; the largest ratio is **1.076923077**.

The actual geometry table is byte-for-byte identical to the offline production
builder's table compiled using the six active holders' metadata. Addresses
and ownership offsets are not compared as work quantities.

- [Every phase](work-phases-v79.tsv)
- [Every operation component and six die values](work-runtime-v79.tsv)
- [Actual retained program and run output](work-runtime-v79.log)
- [Geometry summary](work-runtime-v79-summary.txt)
- [Offline geometry](work-offline-v79.tsv)

The readback run matched all 128 frozen original-reference output IDs, and the
head audit checked all 248,320 logits, 96 CTA winners and six die winners with
zero failures. Decode was **20.214920064 tokens/s** for 127 decode intervals
after the first token. This is one observation, not a demonstrated speedup.

The local and deployed production cubins both have SHA256
`32e973cddd0b297dcda7caae016950070a46af56ca38f7e3dcc2d0373606d8f7`.
The generated SASS used for tile decoding has SHA256
`596764e606ee967c7c08be8d17b0cbd710955affe3f9f19893310b18efc9156d`.
Reproduce the geometry table from the retained log with:

```sh
lua5.4 kern/balance/work_report.lua \
  validation/astra-2026-10-22/per-phase-balance/work-runtime-v79.log \
  kern/die_balanced.sass 492
```

## What the geometry comparison establishes

`kern/balance/work_report.lua` compares corresponding operations on all six
dies, checks opcode/dispatch-path/token-mask equality, requires complete
contiguous phase/member/descriptor coverage, and rejects unknown operations.
The compound width comes from the root descriptor's actual encoded width.
For quantized matrices it checks useful rows, K, expert ranks, dot-product
elements, row tiles and padded dot-product elements. The default tile width
comes from the generated production SASS row-step instruction; ROW8 uses the
explicit eight-row specialization. Publication, activation quantization,
reduction, embedding, residual and head work are compared separately.

Each corresponding positive-cost component is bounded independently. With
the same operation and format on every die, adding those components preserves
the bound; a heavily imbalanced small child cannot hide behind another child
or a phase-wide average. Equal fixed work and dispatch overhead do not enlarge
the ratio. This is an allocation audit, not a count of memory-latency cycles,
barrier polling iterations, cache misses, or hardware instructions retired.

| Work assignment | Minimum | Maximum | Maximum/minimum |
|---|---:|---:|---:|
| Embedding / GR output columns | 416 | 448 | 1.076923077 |
| Routed and shared gate/up rows per expert | 104 | 112 | 1.076923077 |
| Routed and shared down rows | 424 | 432 | 1.018867925 |
| GDN heads | 8 | 8 | 1 |
| QSA query / local KV heads | 4 / 1 | 4 / 1 | 1 |
| PLE key projection rows | 1,696 | 1,728 | 1.018867925 |
| Output vocabulary rows | 41,376 | 41,408 | 1.000773395 |

GDN has eight useful head CTAs per die; sixteen resident CTAs must not be
reported as sixteen useful SMs for that operation. The same caution applies
to QSA's single-producer selection stage and other small operations.

The reporter was challenged with a real dump modified to assign 128 instead
of 112 gate rows to one die (and matching ROW8 units). It rejected six
components, with maximum ratio 1.230769231. Removing the final descriptor
from one die was rejected as missing die coverage.

## Input-dependent work and timing

Every selected routed expert exists as a row shard on every die. The same ten
global IDs select weights, rather than selecting which dies participate.
`balance/moe.lua` executes the ten projections within the same six original
owner groups on all dies, preserving the original FP32 addition association.
`balance/prepare.lua` prepares all eleven 640-value activation vectors on
every die. The shared result is reduced from the same six partial groups;
the subsequent assembled residual read consumes one vector, not six copies.

The completed eight-position trace, canonical job
`8d8d6623-b009-4bcf-9fab-40a788d2fa0e`, contains 377,856 valid CTA records and
2,304 valid expert records. At all 384 layer/token coordinates, every die
selected the same ten distinct global IDs. The [trace summary](body-v78-summary.txt)
retains the result. This capture has **33 nanosecond-timing failures and 26
cycle-timing failures out of 3,936 comparisons** under the older mean-relative
timing gate. Those failures are not converted into work passes or erased.

QSA's score count and gather/attention lengths depend on the common token
position and common 512-block cap. Its selection stages use fixed radix
passes and position-sized scans; histogram predicates additionally depend
on scores. The short v79 run never exceeded the 512-block cap.

Canonical job `44552a00-9342-4ec0-a0de-5ba61c4a76bd` then completed a
4,096-token prompt plus two generated tokens, executing 4,097 causal positions
through the full production model. At final layer47/position4096, readback
confirmed bitwise-identical 512-value indexer queries, all 1,024 selection
scores, and all 512 selected block IDs across the six dies. Thus the
data-dependent selection branch was actually exercised, rather than inferred
from the short-context bypass. The ten routed expert IDs also matched and
were distinct on every die. All six descriptor tables again passed, as did
the complete head audit. The source builds the same replicated selection
operation for every QSA layer: common indexer input and score history, common
token index, common scan lengths and radix passes. Sharded Q/K/V work has
equal head counts and does not change the selector's input shape.

See [long-context raw readback](work-runtime-v80-long.log),
[all geometry components](work-runtime-v80-long.tsv), and
[summary](work-runtime-v80-long-summary.txt). This run has no frozen original
long-context output reference; its evidence is work allocation, cross-die
dynamic-input equality and internal head consistency, not an independent
long-context model-correctness qualification. Its one decode interval is not
used as a throughput benchmark.

## PLE handoff change retained with these results

The runtime now has six persistent CPU DMA submission workers bound to the
six compute contexts. They wake before host row preparation, acquire the
published host epoch, then enqueue the 1,440-byte payload followed by its
four-byte epoch in the same existing per-die stream. The caller waits for
submission, not DMA completion. No GPU operation or physical phase is added.
The pinned source slots remain live until the request's GPU work completes.

The warmed-worker v78 runs matched both 128-token original-reference outputs,
both complete head audits and all 216 repeated GDN hashes. Mean host enqueue
completion spread was 17.28 / 19.61 microseconds, versus 34–41 microseconds in
the earlier worker implementation. TPS was 20.06 / 20.21; no throughput gain
is established. See [v78 output](decode-v78.log). Host enqueue completion is
not a GPU DMA-completion timestamp.

Compute uses physical GPUs **0, 1, 6, 3, 4, 5**. GPUs **2 and 7** remain
reserved for Nate. All GPU runs are recorded in the single canonical database
`/home/nate/.local/state/gpuq/history.sqlite3` on archy.
