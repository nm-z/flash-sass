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
