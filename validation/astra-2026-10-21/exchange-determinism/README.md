# Exchange measurements, split change, and repeatability qualification

This is work in progress for the three requested items. The full objective is
not complete. The latest v33 candidate passes the ordinary and fixed-input
self-repeatability gates; long-context qualification and a fresh matched split
measurement remain pending. Earlier failures below are retained as evidence.
Full48-stage FFN GR replication does not fit archy's current memory placement;
the eight-stage variant fits and reduces actual exchanges, but its earlier run
was not qualified. No cross-kernel correctness oracle is claimed.

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
The later eight-stage job below establishes a feasible actual split, but fails
repeatability. A matched qualification on the current fixed source is still
required. The router is already fully replicated and has no peer exchange of
its own to eliminate.

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

The audit also models8-cycle conversion/SFU results and delayed source reads
(4 cycles for conversion/SFU/global memory,2 for shared memory/shuffles),
using the installed MaxAs grammar's scheduling classes. Retired completion
barriers remain retired for subsequent uses. The same two candidates remain.

GR-only fixed spacing `4a216054-6519-4455-85a4-c56eeecd904c` fails all three
fixed-input repeats:17,12 and9 prediction differences and210 GDN state
differences each. It retains full-image immediate variable completions, but
applies extra fixed spacing only to `bf16hcrwx` and `grreadsum`. The next
isolation region is the PLE path, which executes immediately before layer1,
the first recurrent-state layer that repeatedly differs in these jobs.

PLE-path spacing `c6e5eb17-7945-4cdd-aafb-31c7dd9db19d` passes all four
fixed-input requests and all216 GDN states. This region includes
`plegatherq,plenormgate,pleconv,rms2560,q2k2560,bf16k2560`; the generic bodies
also execute outside PLE. The next narrowing retains extra fixed spacing
only in the three PLE-specific bodies, with full-image variable completions
unchanged. Passing this timing diagnostic is not yet a production fix.

PLE-specific spacing `e076993b-b61a-499a-a8cd-1578d2a77e5f` also passes all
four requests and216 GDN states per request. Extra spacing in the generic
matrix/norm bodies is therefore not required for this observed pass. The
next region is only `plegatherq`, which contains compact row gather followed
by quantization. Immediate variable completions elsewhere are retained for
the controlled isolation and must later be removed from a production fix.

Row/quantization-only spacing `1362c5c8-2245-42f2-9832-0b07a05e2d12` fails
the three repeats:11,19 and11 prediction differences and210 GDN states each.
The three-body PLE region remains the narrowest observed passing region.

The production candidate now applies explicit operand completions and
conservative fixed spacing only to the PLE-specific bodies. It contains597
explicit completion waits, versus16,384 in the full-image diagnostic. The
unfused baseline applies the analogous scheduling to its separate row/gate/
convolution bodies and the generic norm/quantization bodies used between
them (518 completion waits). No phase or fusion is removed. The full images,
split images and exchange trace images have been rebuilt together. This is
a candidate requiring end-to-end qualification, not a demonstrated root-cause
instruction fix or a measured performance improvement.

Production PLE-only end-to-end job
`36730b1e-61cb-44a6-96cf-960cd1c1442d` repeats all output tokens across four
requests per arm. Baseline also repeats all216 GDN states, but candidate
request1 differs in210 states. The job **fails**, despite exact output and
GPU/host/next-input agreement. Thus the PLE scheduling change alone is not
sufficient after removing model-wide completion waits. The next production
candidate retains PLE scheduling and adds immediate operand completions in
the GR read/write bodies, without extra fixed spacing there, phases, or
model-wide serialization. This remains an unqualified candidate.

The v28 warm decode means are34.0386 TPS for the candidate and33.9560 TPS
for its matched baseline (64 prompt tokens,128 generated,no MTP,no trace).
These are diagnostic rates because the candidate's state gate failed.

The v29 candidate keeps the PLE changes and adds442 GR operand-completion
waits, for1,039 total explicit waits. The baseline has727. These are static
instructions in the kernel, not extra dispatch phases or peer exchanges.
Both traced and untraced full-model/split images were rebuilt together.

V29 end-to-end job `b6f17fe1-28d7-460a-b159-eeec086fc79e` still fails.
Both arms repeat every output token, and baseline repeats all216 GDN states.
Candidate request1 differs in210 states; its other requests match. Candidate
warm decode averages31.3355 TPS, baseline32.8099 TPS. The added GR completions
cost time without resolving that state failure. The normal six-die schedule
remains499 phases and354 peer acquires. No speedup or determinism claim is made.

`statehash detail` adds post-completion host comparisons of the actual DN and
convolution values: changed element count, first differing bits/value, maximum
absolute/relative/ULP difference and nonfinite differences. It reads no data
while the kernel runs and adds no kernel instructions. References are captured
from the first completed request in the current library instance and freed on
unload. `BENCH_STATE_DETAIL=1` requires all432 component rows in the full model.
This will distinguish numerical differences from stale/corrupt values without
relying solely on the combined hashes or changing the device instruction stream.

Post-completion job `efcd838e-70c3-45ea-b14a-b2ce86524ce4` reproduces210
state differences in all three repeats while outputs agree. On die0/layer1,
51,603 of131,072 recurrent values differ, maximum absolute difference
4.0423125e-5, with no nonfinite differences. That layer's convolution buffer
matches exactly. Subsequent layers show larger propagated differences. This
is valid state data, not the unused QSA padding excluded from the gate.

The next candidate retains the PLE/GR changes and adds explicit variable
completions to the fused GDN body (and analogous unfused baseline bodies).
The per-request JSON now includes state mismatch counts and marks a request
incorrect when its state gate fails; older records' `correct` field referred
only to output tokens, while their stateaudit records and final job status
already recorded the failure. Generation recipes depend on their Makefile
include so edits to scheduling stages rebuild the images instead of silently
retaining an older binary.

V31 `fb8ec819-0f6b-4ddf-a96a-92e7be32fa4b` adds189 completion waits in
the fused GDN body (1,228 total;baseline903). It still fails one candidate
repeat with210 state mismatches while all output tokens agree. Die0/layer1
has53,505 changed recurrent values, maximum absolute difference1.92848966e-5,
and its convolution buffer still matches. Baseline repeats all outputs and
values on this workload. Candidate warm decode averages31.0001 TPS;
baseline32.4847 TPS. No additional fixed-input, long-context or cross-kernel
qualification is claimed for this candidate. GDN completions are therefore
also insufficient to identify or resolve the remaining fault.

## Physical dependency and publication audit

`phase/physical_dependencies.lua` resolves hardware register aliases and vector
operands before inspecting fixed-result, pending-variable and delayed-read
dependencies. Labels and branches terminate its straight-line analysis. It
does not prove CFG paths or the completion time of a later variable writer.
Its WAR rows are review candidates, not automatically applied fixes.

Against v31, the audit reports364 short fixed-result reads and44 delayed-read
candidates. Register-result `ICMP` feeds many router-selection consumers after
six scheduled cycles while the installed
[MaxAs grammar](https://github.com/NervanaSystems/maxas/blob/master/lib/MaxAs/MaxAsGrammar.pm)
models that compare class at13. Two QSA final accumulator clamps also have a
five-cycle gap instead of six. `phase/fixed_latency.lua` changes control fields:
732 register compares and64 unrolled accumulator sites in the candidate,
412 and32 in the baseline. It adds no instruction or phase. The corrected
physical audit has no short fixed-result rows in either image; remaining WAR
rows are not claimed to be definite hardware violations.

V32 `ff4a7085-7869-4a9c-a25c-c6e51683e461` still fails: baseline repeats,
candidate request2 differs at generated index33 and93 output tokens in total,
with216 different GDN state hashes. Those control-field corrections are not a
complete determinism fix.

The subsequent frame change orders final token/ID/state stores with
`MEMBAR.SYS` before publishing the progress counter polled by the host during
the persistent launch. This is a system publication boundary, distinct from
the conditional peer-send fences. NVIDIA describes the relevant cross-host/
device ordering scope in the
[CUDA memory-fence documentation](https://docs.nvidia.com/cuda/archive/11.0/cuda-c-programming-guide/index.html#memory-fence-functions).
`pleaudit` also compares the token history used for host row staging against
the completed GPU IDs, after execution.

The selected-body scheduling helper previously kept the last operator name
after `POSITION_END`, unintentionally applying operand-completion waits to
the footer. Selected mode now requires being inside an operator body. This
corrects the intended scope; old manifests retain the exact binaries actually
tested. All full/split/trace images were rebuilt for the new publication test.

V33 ordinary job `dd7a2119-eeb3-4a45-b1bd-88eedabe5336` passes four64/128
requests per arm: output tokens,216 GDN states,432 component value comparisons,
six-GPU/host/next-input agreement and PLE staging history all match. Candidate
warm decode averages31.6741 TPS, versus32.4113 for its matched baseline.
Fixed-input gate `a59acb74-24fb-4d75-8edd-a83cf65ba775` also passes all four
256-position requests per arm and all state comparisons. These are qualified
observations for those workloads, not proof for longer contexts or evidence
of a performance improvement. The4,096/16 long-context gate is running next.

The profile extraction tool now accepts explicit arm labels/workload text and
derives phase ranges from the captured headers, so a499-versus499 split report
does not inherit hardcoded unfused-baseline counts. Replaying the saved original
PROF_ALL fixture produces the identical numeric CSV.

`phase/predicate_cfg.lua` audits minimum predicate age through labels and
direct branch edges, terminating paths at unknown BRX edges. It reports zero
sub13-cycle fixed predicate candidates in the current production image. This
extends the straight-line audit but does not prove full instruction scheduling
or indirect control-flow correctness.
