# Native five-position GR up/mix/quantization

The first implementation (v112) shares one BF16 weight stream across five
position accumulators, retaining the scalar lane0 dot association. It stages
the original SiLU-transformed input, computes the five projections, and uses
the original gate/mix/quantization/publication body on the resulting raw rows.
It remains inside the complete production kernel; no standalone matrix test
substitutes for the full model. Register allocation remains249.

Canonical job `88e8ada9-1b46-49b8-841d-2689ba6f7b0a` executes the full model,
128 generated tokens, four-draft prompt lookup, acceptance and recovery.
All128 frozen output tokens and288 retained state hashes match; the complete
head audit passes. The new body executes4,753 times (49 five-position passes
x97 GR-up phases). End-to-end committed throughput is10.694023488TPS. The49
five-position target calls total6725.040095ms, or137.245716ms each, versus
about136.18ms for the retained generic five-position target. This observation
shows no throughput gain.

The scalar default is unchanged. Enable this experimental target with
`FLASH_BALANCED_HCUP5=1` and a batch module carrying
`flash_balance_hcup5_abi_v1`. The builder requires five positions, matching
weights/row counts/offsets, sixteen CTAs, and complete128-row tiles. The original
per-die row ownership is retained.

## Communication change retained in v118

Subsequent versions interleave five SiLU/projection chains, evaluate each gate
once per column, quantize once per position, and publish through six peer warps.
Those changes alone did not improve the complete pass. A stage trace measured
about227.328µs publishing the final GR-up operation on active die0 CTAs, versus
17.4–18.4µs for projection. This is one instrumented head GR-up operation,
not an average over all97 operations or a request-wide cost allocation.

v118 sends only FP32 Z. After the existing peer acquire, the next consumer
reconstructs the exact quantized representation locally on each die, avoiding
transmission of both FP32 values and their derived codes/scales/sums. Opcode110
prefixes the existing consumer with one local grid join. It adds no peer
exchange. The builder enforces producer/consumer adjacency; the receiver uses
five immutable producer descriptors cached in shared memory. Dedicated readiness
storage is cleared before each target/recovery launch. Enable
`FLASH_BALANCED_HCUP5_LOCALQ=1` in addition to native HCUP5.

| Version / condition | Canonical job | Committed TPS | Five-position target ms/call |
|---|---|---:|---:|
| v112, initial shared projection | `88e8ada9-1b46-49b8-841d-2689ba6f7b0a` |10.6940|137.2457|
| v113, column gate/quantization | `1d66b97c-e1cb-4416-898f-999c6c0fa408` |10.0677|149.1822|
| v114, generic body, profile enabled | `e87e07d5-7d82-4e74-9e1f-f1da9e2e330d` |10.6280|135.8756|
| v114, native body, same image/library | `ed1ecc89-60f1-4835-ab47-5116009e06b9` |10.2156|149.2417|
| v115, six peer warps | `0386ca3b-56c6-46d5-bf0c-6e3ce9d74ed3` |10.4747|141.6151|
| v116, interleaved SiLU | `92a1a63c-c346-49f6-b131-bdb3d42d6d2b` |10.4779|141.2126|
| v118, receiver quantization, profile enabled | `06793fe8-3008-4abc-b98a-65a21db8feaf` |11.1164|126.5246|
| v118, no first-pass profile readback, request1 | `e87a41ba-c362-42c6-85cc-03a1fc30b4df` |11.0591|126.4733|
| same job, request2 | same |11.1931|126.4662|

These full48-layer/head requests use64 prompt tokens and128 committed outputs,
with actual prompt lookup and greedy acceptance/recovery. Every row's outputs
and288 state hashes per request match the frozen scalar fixtures. The final
two requests were externally checked against all256 frozen IDs and576 hashes;
GPU/stream/next-input and full-head audits pass. Profile readback is included in
committed throughput when enabled. Target-call timers include program upload,
execution and synchronization, but exclude subsequent profile readback and other
controller/recovery costs. `decode_work_tps` includes the complete decode work.

Each final request proposes205 draft tokens, accepts16, and performs51 recoveries:
317 verification and59 recovery positions for128 committed tokens. v118 counts
80,196 outer phases and8,029 local grid joins, totaling **689.2578125
synchronization phases/committed token**. It fails both60TPS and below500.
It remains opt-in; prompt lookup is not a learned MTP/DFlash2 model.

[Saved per-operation comparison](operations.md) and [per-die CSV](operations.csv)
come directly from the canonical database without another profiling run.
The receiver prefix is charged to the next consumer, so the GR-up-only reduction
must not be presented as the entire verifier improvement.

## Stage trace and SM coverage

The first trace, job `96937265-990a-4c5f-9c22-0f2eb20c7ea2`, used `SR_SMID`.
Legacy MaxAs silently encoded lane ID; its all-zero SM field is invalid.
Stage timestamps remain usable. The correction uses the existing ptxas11.4
Maxwell convention: `SR_VIRTID` followed by `BFE.U32 ...,0x914`.
Job `7cbe9806-f660-45d9-a0ec-f51da5f644e5` uses this correction and v118
receiver quantization;128 reference outputs and288 state hashes match.
Only the final native HCUP of the first five-position pass is captured.

On die0 the corrected trace maps CTA0..15 to SM0..15. Thirteen CTAs/SMs own
projection rows; three still perform input preparation and joins. Active
publication measures4.096µs, while projection remains18.432µs. This is a
diagnostic observation, not a claim that all97 phases save223µs or that every
operation uses thirteen SMs.

The [per-CTA CSV](hcup-stages.csv) retains the corrected trace. Logical die2
below is physical GPU6, not reserved physical GPU2.

| Logical die | Physical GPU | Resident CTAs / SMs | Matrix CTAs / SMs | Mean matrix µs | Mean publish µs |
|---:|---:|---:|---:|---:|---:|
|0|0|16 /16|13 /13|18.432|4.096|
|1|1|16 /16|13 /13|18.117|5.120|
|2|6|16 /16|14 /14|17.920|5.120|
|3|3|16 /16|13 /13|18.353|4.096|
|4|4|16 /16|13 /13|17.959|5.120|
|5|5|16 /16|14 /14|17.554|6.290|

Stage means include only CTAs with matrix rows. All sixteen perform preparation
and synchronization. Different dies' clocks are never subtracted.
