# Exchange measurements, split prototype, and remaining determinism failure

This is work in progress for the three requested items. The full objective is
not complete: the candidate still has repeatability failures, and the proposed
full FFN GR replication does not fit archy's current memory placement. No new
cross-kernel correctness comparisons have been used as an oracle.

## Per-exchange observations

Canonical GPUQ job: `ef3b3b95-a792-49e9-902e-ccb73d45e2a0`.
The compressed JSON export preserves the raw records in that single database.
Both full-model arms completed 64 prompt /16 generated tokens and captured the
last four executed positions. Each arm has 354 exchanges/token/die; all 16,992
records passed epoch, nonzero, coverage, and local monotonicity checks.

- `exchanges.csv`: every exchange, per token and die, with exact local intervals
  and causal clock-offset bounds for producer skew and communication/polling.
- `exchange-summary.md`: summed intervals per token on each die.
- `exchange-operations.csv`: operation-family aggregates. These normalize each
  family by the token/die samples on which it occurs; attention-family rows
  must not be added as if every die owns every attention layer.
- `trace-v3.tsv.gz`: the report input; generate the report with
  `lua5.4 kern/phase/exchange_report.lua INPUT_TSV OUTPUT_DIRECTORY`.

The measurement points are body completion after the CTA join, publication
completion observation, acquire-entry observation, first flag observations,
and acquire exit after the CTA join. A sender's inline payload publication is
already part of its body; the measurement does not recover pure physical wire
latency. Every producer CTA flag is observed by CTA0 on each receiver.

The trace uses device-local GLOBALTIMER clocks. For each token, all 30 directed
GPU-pair causal inequalities bound relative offsets; shortest-path closure
combines them. It assumes relative drift is negligible within one token and
rejects inconsistent bounds. No raw timestamps are subtracted across devices
as if their origins were shared. Tracer overhead and polling resolution remain
in the communication/polling bounds. The instrumented decode rates were
39.087 baseline and39.912 candidate TPS; these are diagnostic, not a valid
correctness-qualified speedup. A matched uninstrumented overhead experiment
for the final fixed kernel remains outstanding.

In the candidate's routed-down → GR-read family, the observed six-die average
is2,810 µs/token/die, with2,281–2,598 µs attributable to producer skew and
212–529 µs remaining for communication/publication/polling. This supports
prioritizing the die imbalance over treating the entire wait as link latency.

## Actual split change

`FLASH_GR_REPLICATE_FFN=0..48` materializes complete BF16 FFN GR down/injection
and up matrices on every die, then runs those operations locally. It preserves
48 model layers, the full head, FP32 residual storage, and all499 phases.
It does not add another fusion or bypass model computation. Use the
`die_phase_replica.cubin` full production image, whose HCUP body handles all
2,560 columns and whose marker guards the layout.

The production schedule compiler, using real holder metadata, proves that
48 replicated FFN GR stages replace96 peer exchanges with local boundaries:
354 →258 exchanges/token/die (7.375 →5.375 per layer), with499 phases retained.
See `replica-schedule.txt` and `original-schedule.txt`.

This costs13,189,120 extra bytes per replicated stage per die, or633,077,760
bytes for all48. Full-model GPUQ runs `c3204f3d-8dd1-4457-a753-4964b1967f62`
(48 stages) and `bbb43f91-de45-45c6-9b20-65c70e201113` (24 stages) both failed
allocation before candidate decode. They establish no candidate TPS. The
service automatically restarted; the serving baseline was then restored by
job `4be120bc-dff8-4a75-8949-3c57a403ef53`.

The memory snapshot in job `473aaf77-2130-47ef-b070-ab80104520c8` reports7,618
MiB per GPU, with175–266 MiB free on the six inference GPUs in recovered startup
configuration. The balanced runtime also copies relocated attention weights.
A feasible measured split, rather than merely an offline descriptor reduction,
is still required. The router is already fully replicated and has no peer
exchange of its own to eliminate.

## Determinism evidence and fixes

All comparisons below are repetitions of the **same** kernel against its own
first request. The diagnostic audits the final token arrays on all six GPUs
and the delivered host stream. Those agree even when repeats diverge.

| Job | Change / workload | Result |
|---|---|---|
| `40e19917-cd92-4fda-af13-fff67a1e8158` | Tail-masked full images;4096/16 | Baseline second request faulted during prefill; candidate never ran. This image did not contain the later frame-lifetime fix. |
| `309c3265-b003-4f2a-b3c0-5a8075f58e5b` | Frame-lifetime and timeout-predicate fixes;64/128 ×4 | Baseline repeats; candidate first request differs at token102 from the next three. |
| `665a2b0c-7587-4e39-9aa7-23111f4fb6f3` | Separate AMAX input-load completion barriers;64/128 ×4 | Baseline repeats; one candidate repeat has97 mismatches. |
| `e97904a9-ef91-4919-9211-9a7e219984f0` | Per-publishing-thread system fences;64/128 ×4 | Baseline repeats; candidate209 total mismatches. |
| `bdc913cf-01d9-4d76-8f34-29e009c88346` | Disjoint router/shared epochs;64/128 ×4 | Baseline repeats; candidate271 total mismatches. Epoch alias was real but not the entire fault. |
| `aa171856-a140-4fd9-9cf7-a00acd38b0fa` | Fixed128-token prompt, one output, per-layer state readback ×4 | Both arms emit repeatable tokens. Candidate run2 has internal GDN state differences beginning at layer26. Output-only repeatability is insufficient. |

[Issue136](https://github.com/nm-z/flash-sass/issues/136) records the definite
router/shared readiness alias: shared(token,L)=token*49+L+49 equaled
router(token+1,L). The fix reserves98 epochs per token and uses full32-bit
shift/add arithmetic. All fusions remain enabled. It must not be described as
a complete determinism fix.

Other retained fixes prevent KV gather from dequantizing unwritten causal-tail
rows, protect footer timestamp address/data lifetimes from descriptor prefetch,
and pad two fixed predicate hazards in the candidate footer. The predicate
scanner is a straight-line audit, not a full control-flow or scoreboard proof.

`BENCH_STATE_HASH=1` now compares all216 GDN state hashes against the first
request of the same arm, in addition to output and six-GPU/stream consensus.
QSA combined hashes include unused positional padding, so those are printed
but are not used as an equality oracle. Further cross-kernel correctness tests
remain deferred until self-repeatability is established.

## Further isolation after the first publication

The all48/24-stage attempts above were not reduced model runs: they failed
allocation while retaining the complete48-layer model. Eight replicated stages
**do** fit. Job `3a2671b8-b5dd-40f5-842e-f056a28339be` executes all48 layers/head
and reports **499 phases,160 local grid joins,338 peer acquires on every die**.
That is16 fewer actual exchanges. Its three post-warmup requests average37.003
TPS, against37.090 for its matched baseline, but one candidate request diverges
and has216 different GDN states after autoregressive feedback. These numbers
are diagnostic; the split is opt-in and is not claimed as a validated speedup.
The canonical memory samples peak at7,601 of7,618 MiB on the tightest GPU, so
another stage cannot be assumed to fit with a useful allocation margin.

Long-context job `ac49077b-ef12-4395-a3b3-51966dae307d` completes two4096/16
requests per arm. Each kernel repeats its own output, and all six GPUs agree
with the delivered stream. This is output repeatability on that workload;
it does not erase the short-workload failures or establish internal-state
repeatability at long context.

Further full-model diagnostics retain all fusions and499 phases:

- `7151cde2-b023-41d5-bc87-aca02e624e81`: coherent GR scratch/residual loads
  still fail64/128 repeatability. Immutable weights/norms retain cached loads.
- `8f3b1bdf-193b-4a3d-8c69-074d89958dad`: forcing499 local grid joins while
  retaining all merged bodies still fails. Local join elision alone does not
  explain the failure; this mode is diagnostic and off by default.
- `bfb4f242-d766-437f-995d-21289a439a29`: separate descriptor payload/mask
  completion barriers, independent S2R thread/block IDs and mixed timeout-read
  completion barriers still do not establish candidate repeatability.
- `3094c653-53d6-4386-80f6-cda75791e37a`: a traced fixed128-token prompt,
  four repeated requests, matches all216 GDN state hashes and GR input
  fingerprints in positions94..97. Instrumentation changes timing; this is
  not evidence that the uninstrumented failure is fixed.

Trace ABI v2 uses a fixed4096-descriptor stride, allowing selected prompt
windows across the prefill/decode program boundary. `BENCH_XTRACE_HASH_ONLY=1`
prints the GR input fingerprints without incorrectly treating unexecuted
prefill head descriptors as missing exchange records. Fingerprints are
observational diagnostics, not an independent model-correctness oracle.
The original v1 timing export and report input above remain bound to their
original source snapshot and layout.

GPUQ interrupted `1dd224db-acb0-456c-b496-7b8edd358bad` during a baseline
request with `SQLite3::BusyException: database is locked`, terminating the
client with SIGTERM. [GPUQ69](https://github.com/nm-z/gpuq/issues/69) records
that failure; the terminal job was rerun as7151cde2 after restoration. No result
is claimed for the interrupted run. A separate early-closed metrics pipe is
[GPUQ68](https://github.com/nm-z/gpuq/issues/68).

## Host readiness lifetime and uninstrumented failures

[Issue137](https://github.com/nm-z/flash-sass/issues/137) records a second
definite lifetime bug: core-owned PLE readiness flags survive a library reload,
while the library-local request epoch restarts. Job
`6c164217-8cd9-4a94-b05a-7d2cb357931a` observed191 old flags equal to the new
epoch before launch. The engine now clears the request's readiness span before
launch. This removes that alias; it does not establish candidate determinism.

The same job includes coherent loads for mutable device data and invalidating
loads for CPU-written compact PLE payloads. Baseline outputs and all216 GDN
states repeat across four requests; candidate output still varies. Explicitly
zeroing fresh-sequence scratch in `1e4f4a4b-52b3-419d-bf29-3da3d10cdd67` also
does not resolve the candidate failure.

Fixed-input full-head job `6aaba820-27ba-4889-b26f-809436fa4fe4` evaluates256
supplied positions without autoregressive feedback. Candidate predictions and
internal states vary. This isolates a fault before feedback amplification.

Updated-residual observer job `655b56b7-51a8-4ad0-95de-71f059b4c7eb` captures
583,680 fingerprints:95 GR writes at256 positions, six dies, four requests.
There are no temporal or cross-die differences. This is an observer result:
the added instructions change timing and cannot qualify the uninstrumented
kernel. See `gr-hash-first-differences.md` and `phase/gr_hash_report.lua`.
The initial v14 observer faulted because a branch used the distributed-output
predicate P6, which is not warp-uniform. The corrected observer computes on
all lanes and predicates only its final stores.

Retiring residual stores before reuse in
`e89ee2dc-22eb-4e7f-98d2-b568e16a1a07` makes the four output streams agree,
but one request still differs in210 GDN states, beginning at layer1.
Explicitly completing each six-word host PLE read packet in
`cdbe6957-d73a-4cc8-91d7-6075f2b9bfad` passes all four output/state repeats.
The latter job measures **34.7464 TPS** averaged over three warm requests,
64 prompt positions and128 generated tokens, no MTP, no trace. All six dies
report499 phases,144 local grid joins and354 peer exchanges. This is slower
than the earlier approximately40 TPS kernel.

The subsequent fixed-input gate
`ec1db389-9bdd-43d6-94bf-86b7094a6535` contradicts a general determinism claim:
baseline repeats, but candidate request2 differs at positions177 and241 and
in210 GDN states beginning at layer1. The ordinary pass therefore remains
unqualified. Diagnostic `c3a2d851-3fbc-41cf-bcbc-1a853dfa1ba5`, which waits
immediately after variable-latency operations throughout operator bodies,
also fails. It preserves the complete model and499 phases; serializing those
bodies alone does not explain the race. Frame-inclusive completion isolation
is the next diagnostic, not a proposed performance configuration.

Whole-kernel variable-completion job
`9c36232f-7393-4997-a71d-a84f6a4a7bd3` also fails:77,13 and2 output
mismatches in the three repeats,216 GDN state mismatches in each. GPU/host
stream and next-input agreement remain exact. Adding a13-cycle minimum
scheduled gap to every original instruction as well as immediate variable
completion waits in `e8ababfd-be5a-4192-ae35-e4e098df95b4` passes the four
ordinary64/128 requests, including216 states each. These deliberately slow
images retain all499 phases. A fixed-input gate is required before drawing a
scheduling conclusion; changing issue timing can also mask a data race.

The follow-up fixed-input job `9accb9bb-a79a-40ab-8a85-e3d33e549d40` passes
all four256-position candidate requests, including all216 states. The job as
a whole **fails** because the baseline's third request differs at positions183
and236 and in162 GDN states. Earlier baseline self-repeats are therefore
insufficient to qualify the baseline as an oracle. No cross-kernel correctness
claim is made. The next diagnostic keeps immediate variable-completion waits
throughout but adds the13-cycle spacing only to the persistent frame, to
distinguish frame and operator scheduling without changing the model.

Frame-only spacing `4daa6b0b-23eb-46fd-bf66-559cdcdbf4fa` fails all three
fixed-input repeats:7,3 and6 prediction mismatches, with210 state differences
each. It keeps immediate variable completions across the entire image and
changes only the fixed-spacing region relative to v20. Operator-only spacing
is tested separately; neither image is a production speed candidate.

Operator-only spacing `19f8c751-6c91-4ec4-b84e-0c20f4a61217` passes all four
fixed256-position requests and216 GDN states per request. Thus the passing
configuration does not require the frame's added fixed spacing. This remains
a timing-sensitive observation, not identification of a faulty instruction.

`92ea972d-f619-416f-8cf6-a3f9cf4579d8` uses the unchanged v17 production SASS
with a diagnostic host schedule that acquires and publishes every physical
phase. All six dies confirm499 phases,499 local joins and499 peer acquires;
there are no joined-publication shortcuts. It still fails all three repeats:
5,5 and1 prediction differences,210 GDN state differences each. Adding peer
ordering at every existing phase does not establish determinism. The normal
schedule remains354 peer acquires; lockstep is off by default and explicitly
cleared on benchmark restoration.

`phase/register_latency.lua` reports two same-name straight-line arithmetic
RAW candidates with minimum age5 cycles: the final accumulator clamp in
`idxscore` and its `scoreselect` clone. The report assumes6-cycle fixed
arithmetic latency and does not model register aliases, CFG edges, or extra
stall cycles supplied by variable completions. It is a review aid. These QSA
sites do not by themselves explain state drift beginning at layer1.

`phase/predicate_cfg.lua` audits minimum predicate age through labels and
direct branch edges, terminating paths at unknown BRX edges. It reports zero
sub13-cycle fixed predicate candidates in the current production image. This
extends the straight-line audit but does not prove full instruction scheduling
or indirect control-flow correctness.
