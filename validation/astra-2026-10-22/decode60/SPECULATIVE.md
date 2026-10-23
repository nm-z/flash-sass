# Balanced lossless speculative decoding

The greedy speculative controller is implemented and has completed full-model
runs with acceptance, rejection and state recovery. Its current proposer is
**prompt lookup**, not the legacy neural MTP or DFlash2 implementation. The
60-TPS objective is not achieved, and this configuration exceeds the end-to-end
phase target. The retained scalar default remains v104 at25.72–25.79TPS.

Nate explicitly authorized lossless speculative decoding with drafting,
verification and recovery included in throughput and phase accounting.
Physical GPUs2 and7 remain reserved; compute uses0,1,6,3,4,5.

## Implementation

`balance/batch.c` packs complete already-fused position programs into
phase-major wrappers. Every native child descriptor remains, including QSA,
GDN, router and independent matrix chains. The outer wrapper owns the incoming
peer acquire and final publication for its position block. QSA additionally
joins the local grid between positions to protect rolling state and aliased
attention scratch. The builder reports outer phases, causal replays, inner
grid joins, publications and acquires separately.

`balance/batch_frame.lua` adds opcode108 and preserves the real inner primary
index in `pidx`. Composite bodies therefore continue finding their own child
parameters. Wrapper metadata occupies shared0xba00..0xba20, below legacy B8
parameter banks. The new module requires `flash_balance_batch_abi_v1` as well
as the ordinary balanced markers. Register allocation remains249.

`balance/speculate.c` implements the complete loop:

- Search suffixes of1–16 tokens in actual prompt and committed-input history;
  propose up to four already-seen continuation tokens. Reference fixture IDs
  are held by the benchmark process and are never a proposal source.
- Execute one to five causal positions through the full48-layer target/head.
- Accept the matching draft prefix using exact greedy token equality and
  commit the target replacement/bonus token. Stop handling truncates the prefix.
- On rejection/truncation, restore the block-entry GDN, convolution, PLE and
  unfinished-indexer-ring snapshot, then replay only the committed input prefix.
  No rejected output reaches the token callback.
- Drain prior PLE DMA submissions, clear reused readiness slots, and restage
  the changed input span before verification or replay. GPU next-input IDs,
  final control counters and host committed IDs are kept consistent.

Per-position head scratch is disjoint. Post-request head/QSA/logit readback
uses the final accepted position's buffer slot. Sampling and image/new-snapshot
requests are explicitly unsupported in this initial greedy mode; the unchecked
legacy MTP/DFlash paths remain rejected on balanced holders.

Enable with `FLASH_BALANCED_SPECULATE=lookup`, five-position capacity and
`FLASH_EXACT_CUBIN=.../die_balanced_batch.cubin`. The existing benchmark exposes
`BENCH_BALANCED_SPEC=1` and `BENCH_BALANCED_BATCH_CUBIN`. It retains the six-die
reservation. `FLASH_SPEC_MAX_DRAFTS=0` is a diagnostic single-position setting;
`FLASH_BALANCED_BATCH_STRICT=1` is a diagnostic join after every inner phase.
Neither diagnostic is used for the final speculative results below.

## Memory handoff defect and correction

The first complete run matched128 output tokens but differed in282 of288
retained state hashes. A single-position run through the same controller
matched all hashes. Strict inner grid joins also matched, isolating the failure
to the new multi-position handoff.

The no-grid inner footer bypassed the fence/invalidation performed by native
`GROUP_NEXT`. Adding `MEMBAR.GL.IVALLD` between positions fixed the short case,
but the4K rejection case still differed. Moving the handoff before the group-end
decision, so it also covers the final position, fixed both cases. The final
path keeps the original QSA-only inner grid joins; the all-phase diagnostic
joins are not enabled. The affected payload load was not isolated to one
cache-policy instruction; PLE history loads are already coherent L2 loads.

## Canonical evidence

All jobs reside in archy's single GPUq history DB. Library v110, scalar image
v106 (byte-identical v104), batch image v111 and benchmark v106 form the final
validated artifact tuple. Source changes are version-controlled in this repo.

| Version / condition | Canonical job | Evidence |
|---|---|---|
| v106, initial full loop | `8b8f1d29-def8-4149-b495-cf921943867b` |128 outputs match;282 state hashes differ; not a valid lossless result |
| v107, proposals disabled | `6fe8e1a8-66ea-4159-96ce-fb7238fa26e9` |128 outputs and288 hashes match |
| v108, all inner joins | `ad75eb68-8c36-4af3-b09f-2a904afed598` |128 outputs and288 hashes match; diagnostic phase count1675.875/committed token |
| v109, between-position handoff | `8f8d89ec-c6ba-4c3d-9077-a0075aaf554a` |Short outputs and288 hashes match |
| v110, repeated short requests | `a562bcda-5d28-4bc7-860a-4f13a968c667` |Both128-token outputs and576 hashes match |
| v110,4K prompt | `b9daec02-3955-433d-8452-c84498c52d6c` |Both output IDs match;282 state hashes differ |
| v110,4K with proposals disabled | `0287a82b-ca7a-460c-938e-5688b4ded7ed` |Both output IDs and288 hashes match the scalar long fixture |
| v111, complete handoff,4K prompt | `ac9d86f7-6f83-4130-8100-9f06fc807723` |Both output IDs760/1156 and288 hashes match; full head audit passes; rejection/recovery exercised |
| v111, complete handoff, repeated short requests | `fef02272-c98a-491d-873f-f8ba996a1d78` |All256 outputs externally compared with frozen scalar IDs; all576 hashes match; both full head and GPU/stream/next-input audits pass |

The short state oracle is the retained v78 fixture. The long state oracle is
canonical scalar v104 job `989775df-8bbb-4503-b55f-22a1ec386c18`.
The final short benchmark used its repeatability mode, so the additional
external comparison of both token streams to the frozen128-ID fixture is
necessary; the harness's first-run `correct` flag alone is not that evidence.

The [last-program work summary](speculative/last-program-work-summary.txt)
reports492 phases and maximum pairwise assigned-work ratio1.076923077. It
covers the final retained scalar replay, not every earlier wide program.
Wide plans copy the same native geometries across positions and assert matching
structure; additional end-to-end work is reported below rather than hidden
behind that492-phase readback.

The existing CPU `timings_test` also passes with a new case ensuring discarded
and replayed positions increase execution counts without increasing generated
output counts.

## Performance and accounting

Final short requests measured **10.744281302 and10.737048870 committed TPS**.
`decode_work_ms` starts before the first proposal and includes the complete first
block, snapshots, staging, target execution, recovery, callbacks and final control
publication. Program construction and prompt processing are reported separately.
The ordinary post-first-token API rate is not the speculative performance authority.

Each128-token request performed:

| Work | Count |
|---|---:|
| Verification blocks |112|
| Proposed tokens |205|
| Accepted draft tokens |16 (7.80%)|
| Rejected/recovered blocks |51|
| Verified target positions |317|
| Replayed target positions |59|
| Committed output tokens |128|
| Outer target phases, verification plus replay |80,196|
| Inner whole-grid joins |3,276|
| Causal phase replays |184,992|
| Accounted synchronization phases per committed token |**652.125**|
| Snapshot bytes, all six dies |6,702,809,088|
| Restore bytes, all six dies |6,449,872,896|

Both throughput and the end-to-end phase bound therefore fail the goal.
The two-token4K case is a correctness check, not a decode benchmark.

The first final short request's inclusive target-call timings (program upload,
launch, polling and synchronization; excluding snapshot/draft/staging) are:

| Width | Verification calls | Mean ms/call | Recovery calls | Mean ms/call |
|---|---:|---:|---:|---:|
|1|59|39.307|44|39.377|
|2|0|—|6|64.945|
|3|3|88.398|1|88.354|
|4|1|112.474|0|—|
|5|49|136.181|0|—|

The five-position target cost requires native matrix work sharing in addition
to a better proposer. Merely replacing prompt lookup does not remove that
measured verifier cost. The next implementation should reuse weight loads and
decoding across positions, starting with the dominant GR matrix operations,
while retaining the established floating-point accumulation order.

The cached DFlash2 checkpoint is
`/home/nate/models-cache/flash-sass/dflash2/model.safetensors`; its adjacent
config declares hidden2560, five draft layers, target48 layers, vocabulary248320,
block size8 and target feature layers5/15/25/35/45. Its current engine code uses
one extra device and unchecked outputs. A proper integration still needs the
balanced six-die placement, target-feature capture and this lossless controller.
The native Q8 MTP checkpoint also exists at
`/home/nate/models-cache/flash-sass/mtp-Qwen3.8-Flash-Next-Q8_0.gguf`;
its current unequal whole-expert ownership does not satisfy the work bound.
