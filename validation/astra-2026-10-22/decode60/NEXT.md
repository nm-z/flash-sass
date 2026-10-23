# Current work toward60 TPS

Retained scalar default: v104,25.72–25.79TPS,492 phases/token and maximum
pairwise assigned-work ratio1.076923077. GPUs2 and7 remain reserved.
The60TPS goal remains active and unachieved.

The balanced speculative loop is now operational for greedy text using prompt
lookup. It includes real target verification, acceptance/rejection, block-entry
snapshots, recovery replay, PLE restaging and counters for discarded/replayed work.
The corrected v111 frame matches both128-token requests, all576 short state
hashes, the two long-context tokens and all288 long state hashes. It also passes
GPU/stream/next-input and full-head audits. See SPECULATIVE.md for canonical jobs,
artifact versions and the earlier failed state comparisons.

Its performance is10.74 committed TPS with16/205 draft tokens accepted and
652.125 synchronization phases per committed token. This fails both performance
requirements. The final scalar replay's492-phase readback does not establish
that the full speculative request meets the phase limit.

The new causal handoff must remain before the batch-end decision, including the
last position. Omitting it caused282 state-hash differences despite identical
output tokens. The final path does not enable the diagnostic all-phase grid joins.
The existing CPU timing test passes with speculative execution-count coverage.

Next: native multi-position matrix execution and a stronger balanced proposer.
The measured five-position target takes136.18ms per verification call, so a
better proposer alone cannot achieve60TPS with this implementation. The present
wrapper is an exact causal replay path; it does not yet share matrix weights
and decoding between positions. The largest inclusive scalar family is hcupq_x
(GR up/mix/quantization), followed by routed gate/up/down and GR reads.

Objective weights: throughput1.0, fidelity1.0, phase/work constraints1.0,
engineering time0.8, cost0.8. The lower-effort option is tightening prompt-lookup
matching; it may avoid bad guesses but leaves the measured verifier bottleneck.
The higher-effort option is native batched matrix bodies plus learned drafting.
Select the latter. Reuse the complete controller and preserve its work accounting;
do not return to unchecked legacy MTP/DFlash output or omit recovery cost.

Cached DFlash2 config: hidden2560, intermediate7680, five draft layers, block8,
target48 layers and feature taps5/15/25/35/45. The current DFlash engine uses a
single extra device, CPU elementwise/attention work and unchecked output; it
cannot simply be enabled under the current GPU reservation/work-balance rules.
Native MTP's current expert split is also unequal. Required future changes must
cover all participating dies and all model work before the next full-model run.
