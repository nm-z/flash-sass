# Distributed activation preparation experiment

This branch contains the complete production-kernel experiment. It does not
achieve60 TPS and is not promoted over the24.793978201-TPS v91 checkpoint.
Every run below executes all48 target layers and the full output head, with
MTP and FP8 disabled. Physical GPUs2 and7 remain excluded.

| Version | Canonical GPUq job | Decode TPS | Change |
|---|---|---:|---|
| v92 | `c3d53bed-3145-4482-b960-afeb666ee8b0` |18.718191775|Remote raw-input reads, packed quantized records, remote readiness polling |
| v93 | `5a0443f2-8f05-4d1e-aba8-c15c85b96a88` |23.215103729|Push readiness flags to each consumer; poll locally |
| v94 | `7431de13-5d09-41b2-9256-8359980558ef` |23.219708022|Push raw gate/up stripes only to their quantization owner |
| v95 | `d5dbde50-6f05-4c92-95ba-dd76ff732c34` |23.635491116|One publishing warp and one system release per CTA; stage packed records in shared memory |

These are individual full64-token-prompt/128-generated observations, not a
randomized A/B estimate. All four requests match the128 frozen original
reference tokens and pass the full head audit. v95 also matches all288
retained per-die/per-layer state hashes from v78. Raw records and input/output
capture remain in archy's single canonical database:
`/home/nate/.local/state/gpuq/history.sqlite3`.

## Complete data path

Each32-value activation group has owner `group_index % 6`. Of220 groups,
four dies prepare37 and two prepare36. Gate/up weight ownership remains the
existing eight-row stripe layout. Host-built routing tables send each raw
eight-row stripe only to the owner of its32-value group. The four stripes
arrive in that owner's ordinary rank-major FP32 buffers.

The owning warp performs the original SiLU/up multiplication and quantization
without reassociating arithmetic. Each128-byte-stride packet contains64 bytes
of signed16-bit codes, S32 at byte64, the two S16 sums at68/72, and the FP32
scale at76. v95 stages the20 useful words per warp at shared address
`0x2000 + warp*128`, then warp0 publishes every valid CTA record to all six
dies. All K640 routed/shared consumers directly read this layout, including
Q2 and all four B32 shared-down formats. Mutable packet loads remain coherent
`.CG` accesses.

Producer stores are system-fenced before readiness publication. Each
destination has96 local readiness words, one for each die/CTA producer. Flags
are pushed to every destination, and six consumer warps each acquire one
producer die's16 flags before their CTA rendezvous. Request startup clears all
96 flags. The epoch is `64*(token+1)+layer`; subsequent layer progress also
depends on the existing down-output exchange, protecting packet reuse.

This replaces a local preparation join with a real cross-die join inside the
existing down phase. It adds48 internal peer joins; it does not claim fewer
communication events by counting only dispatches. All six programs still
contain492 physical phases and979 descriptors. v95 passes all7,476 component
comparisons with maximum pairwise assigned-work ratio1.076923077. See
[work table](work-v95.tsv) and [summary](work-v95-summary.txt).

## Findings

Remote readiness polling was particularly expensive: the down-phase total on
logical die5 fell from19.598ms/token in v92 to9.188ms in v93 after mirroring
flags locally. Targeted input pushes reduced it to7.754ms in v94, while
gate/up grew from3.946ms to5.186ms. Consolidated publication gives7.480ms down
and4.742ms gate/up in v95. These phase profiles include boundary effects.
The full-request timing remains the throughput authority.

The reduced FP32 fanout and repeated quantization did not outweigh packet
construction, publication, the added readiness dependency and consumer access
costs. v91 remains the faster measured default. The complete implementation,
ABI marker and immutable artifacts are retained for follow-up work rather
than representing this as an achieved optimization.

The existing body tracer samples preparation polling on warp0. This
experiment distributes that polling across six warps; its old
body-minus-poll timing must not be interpreted as excluding every wait without
extending the trace format. The reported measurements here use full request
timing and ordinary full-phase profiles.
