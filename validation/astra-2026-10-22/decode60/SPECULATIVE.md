# Lossless speculative decoding: implementation in progress

Nate explicitly authorized lossless MTP/speculative decoding for the60TPS
objective, with draft and verification work included in throughput and phase
accounting. The complete target model, per-phase pairwise work bound and
physical GPU2/7 reservation remain requirements. The active goal is unchanged.

The existing `mtp_four_run` and DFlash2 loops emit unchecked blocks. They do
not implement lossless acceptance, rejection or recovery, and are not suitable
for this objective. They remain disabled on balanced holders.

## Code implemented so far

`kern/balance/batch.c` packs complete already-fused position programs into
phase-major wrappers. Each wrapper retains every native child descriptor,
including QSA, GDN, router and independent matrix chains. The outer wrapper
owns one incoming acquire and one publication for its complete position block.
Inputs must agree in opcode, geometry, token-mask layout and structural flags.
Nested native phase widths and all child token offsets remain intact.

`kern/balance/batch_frame.lua` adds opcode108 to the production balanced image.
The real inner primary descriptor index remains in `pidx`, so composite bodies
still locate their own child descriptors. Wrapper state lives at shared
0xba00..0xba20, below the legacy B8 parameter banks and away from composite
bf00 scratch. At the last inner position the native outer framework state is
restored and its normal system release/publication footer runs. The new image
has a distinct `flash_balance_batch_abi_v1` marker.

QSA requires a complete local grid join between positions because rolling
indexer state and some gather buffers can be shared. The builder explicitly
retains that join. Other stateful bodies keep their established CTA ownership.
The work structure reports outer phases, causal phase replays, inner grid
joins, publications and acquires separately; the outer phase count alone is
not an end-to-end speculative-work claim.

The C object and `die_balanced_batch.cubin` compile. The batch image has249
registers. This is **not runtime validation**. No partial GPU run was launched,
and the ordinary engine still rejects balanced multi-position execution.

## Work still required before the first GPU run

1. Wire the new program builder and marker into the balanced engine with
   disjoint per-position head scratch, correct buffer lifetimes and five-position
   capacity. Keep unchecked MTP/DFlash modes rejected.
2. Build a real draft/verify controller. Publish only the longest accepted
   prefix and the target's replacement/bonus token. Initially greedy decoding
   needs exact token equality; sampled decoding requires a separate correct
   distribution-preserving acceptance rule.
3. Restore recurrent GDN, convolution, PLE and unfinished-indexer-ring state
   on rejection. A bounded first implementation can snapshot at block entry
   and replay the committed input prefix; all replay time/work must be counted.
   Future positional KV rows must remain causally invisible until overwritten.
4. Reset transient readiness generations on same-position replay and restage
   PLE rows when proposed input IDs change. Drain prior DMA submissions before
   clearing/reusing readiness flags.
5. Count verified, rejected, replayed and committed positions separately.
   Report a decode interval that includes the complete first speculative block,
   drafting, snapshots, verification and recovery; do not inflate TPS by
   excluding a multi-token first block while counting its later tokens.
6. Run the complete production model with acceptance and recovery enabled,
   including rejection and block-boundary cases, then compare committed output
   and recurrent state with the retained scalar fixtures. Do not run a partial
   layer model or present unchecked proposals as decode throughput.

The final drafter must also obey the six-die work/resource constraints. The
existing DFlash initializer chooses one extra device and the native MTP loader
uses unequal whole-expert ownership; neither placement can simply be enabled
under the current reservation and work-balance requirements. Prompt lookup is
an available lightweight speculative proposal method, but any result must name
its actual drafter, measure real acceptance, and retain the full60TPS objective.
