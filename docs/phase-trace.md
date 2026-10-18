# Bounded phase and host traces

Tracing is opt-in and uses separate kernel images. Build:

```bash
make -C kern die.cubin die_trace.cubin die_trace16.cubin libengine.so trace_layout_test
```

Select an absolute `FLASH_CUBIN` path and matching `FLASH_TRACE_CTAS`:
`die_trace.cubin` records thread 0 of CTA 0 (`FLASH_TRACE_CTAS=1`);
`die_trace16.cubin` records thread 0 of each of the 16 CTAs
(`FLASH_TRACE_CTAS=16`). Set `FLASH_TRACE` to the maximum output positions
to capture. It defaults to zero. The product
`FLASH_TRACE * FLASH_TRACE_CTAS` must be at most 256, so CTA 0 accepts
1–256 positions and the all-CTA image accepts 1–16.

Schema 2 reserves 4096 phases × 128 bytes × positions × CTAs per die,
at most 128 MiB per die. Only the actual phase count and completed bounded
position range are exported. Both CTA count and 16-word record stride are
checked on device; a mismatched image produces no valid trace. The host
rejects incomplete, nonmonotonic or mismatched records. Ordinary
`die.cubin` has no trace instructions or publication fence.

## Collection and identity

After a response, send `trace <die>` for **every** participating die and
`hosttrace`, before another request replaces the records. Each export
identifies the loaded library's request, position window, dimensions and
lifecycle. Save the `config` line, all command/completion records, exact source
commit, library/cubin SHA-256 hashes, launch/reload environment and raw exports.
The `config` command also reports the selected cubin's FNV identity.

The trace starts at the **last prompt position** (the first output token).
Earlier prefill positions cannot enter its phase-zero bucket. Each record
contains token position, phase, transformer layer, opcode, work units,
exchange flags, selected-key count, CTA and timestamps. Layer -1 is embedding,
output-head or framework work outside a transformer block. All program phases
are exported, including every QSA projection, normalization, pooling, scan,
selection pass, KV append, gather/dequantization and attention partial/merge.

## Device event meanings

GPU records use 64-bit global timers. HI/LO/HI reads retry across low-word
rollover. Keep them as integers; absolute values can exceed the exact integer
range of a floating-point number. Devices have independent epochs. CTAs on
one die share that die's timer; the host is a separate clock domain.

| CSV field | Sample location |
| --- | --- |
| `start_ns` | Phase entry, including phase zero |
| `wait_start_ns` | Before the incoming exchange readiness path |
| `wait_end_ns` | After incoming exchange readiness |
| `body_start_ns` | Entry to the selected opcode, after the dispatch comparisons/branches |
| `body_end_ns` | Selected opcode return, before framework position/grid barriers |
| `end_ns` | After the framework barrier and send path |
| `ple_wait_start_ns`, `ple_wait_end_ns` | PLE-row readiness wait nested inside that opcode's body |
| `send_before_ns`, `send_after_ns` | Bounds around peer flag stores; the latter follows a profiling-only system fence |
| `exchange_generation` | Incoming generation, an integer rather than a timestamp |
| `publication_ns` | After the completed-token count store, on the final phase only |

Flags bit 0 sends generation `exchange_generation + 1`; bit 1 waits for
`exchange_generation`. A phase may wait for one generation and send the next.
The send brackets preserve the live peer pointers, generation and barrier
predicates. Their branch guards are uniform within a warp.

These are **sampled opcode-body intervals**, not summed execution times of all
threads. Other warps or CTAs can still be finishing when a sampled thread
reaches a barrier. Their remaining work is represented in its barrier interval.
The all-CTA image exposes CTA skew; adding its 16 durations would double-count
overlap. Gather already contains its mapped-memory accesses: do not add a
separate PCIe estimate to it.

## Host event meanings

Host samples use `CLOCK_MONOTONIC`. PLE staging spans row selection, advisory
cache hints and the `pread` loop; read start/end isolates that loop. These
durations include cache hits and do not establish physical disk latency.
The readiness store is between `ple_ready_start_ns` and `ple_ready_ns`.
Bytes read and incomplete-row counts are reported per traced token.

For a delivered batch, polling samples bracket the stream/count query and
readback samples bracket the token transfer. Multiple tokens may share a batch:
deduplicate identical intervals before totaling their cost. The header's
`request_poll_calls/ns` and `request_readback_calls/ns` include **every**
poll/readback in the complete request, including prefill and positions outside
the bounded trace; these request totals have a different scope from per-token
samples.

Callback entry/return describes the core callback, not delivery to an HTTP
client. On cancellation, `callback_end_ns` is sampled immediately on return,
before `abort_run`. The header separately brackets abort publication using
`abort_start_ns`, `abort_end_ns`, `abort_position` and `abort_reason`. A disconnect may
instead be noticed by the periodic liveness callback, without a failing token
callback. Host/GPU after-store samples are observations, not synchronized
cross-domain instants.

## Lifecycle

`trace_begin` and `hosttrace_begin` contain `status`, `aborted`, `ended`,
`completed` and `requested`. Completed/requested count full request positions,
including the prompt; they are not bounded row counts. Completed is finalized
after all streams synchronize on a completed or aborted request.

`status=completed aborted=0` identifies a completed request. A normal stop token
can set `ended=1`; it does not imply cancellation. A disconnect is
`status=aborted aborted=1`, even if the device end bit is also set. An aborted
completed-prefix trace is diagnostic evidence, not a throughput sample.

Every call to `engine_run`, including argument/image validation failures,
advances request identity and invalidates prior data. Errors return
`argument_error`, `image_error` or `device_error`. Unavailable trace queries
carry current identity/lifecycle fields instead of exposing a prior request.
A malformed transport command rejected before `engine_run` is not a new model
request. IDs are local to the loaded library; a reload starts a new identity
domain. The immediate device-fault line retains the failed request's identity
before core recovery. Faulted GPU buffers are never salvaged as valid traces.

Snapshots, paired known-prefix work and chunked prefill end before the bounded
generation window. The generation program uses its own phase count when the
buffer is cleared and indexed.

## Reconciliation and overlap

For each sampled CTA/token, partition phase entry through publication into:

- setup: phase start to wait start;
- exchange readiness: wait start to wait end when the wait flag is set
  (otherwise classify this interval as dispatch/framework work);
- opcode dispatch: wait end to selected-body entry;
- sampled body excluding nested PLE readiness;
- PLE readiness;
- barrier/send: body end to phase end;
- gaps between phases and the final publication tail.

The sum must equal `publication_ns - first_phase.start_ns` exactly in integer
nanoseconds. Inter-publication duration also includes the gap from the previous
token's publication to this token's first phase. The first captured token has
no previous publication; its missing predecessor is reported explicitly.

For cross-die analysis, match exchange generations. A sender's
`send_before_ns` must causally precede every receiver's matching
`wait_end_ns`. Do not substitute `send_after_ns`: a receiver can observe the
store before that upper sample. Generations preceding the captured window are
external predecessors. A DAG combines local elapsed intervals with these
causal edges. Its longest observed chain uses maximum at merges, preserving
concurrent branches rather than adding their durations.

Relative clock offsets are constrained by the same send/wait relationships.
The analyzer reports feasible offset ranges, timer tolerance and the resulting
span/chain uncertainty. Bounds assume constant relative timer offsets over the
bounded capture and a 1024 ns allowance on pairwise timestamp ordering. Each
crossing uses that pairwise allowance. If each timestamp instead has an
independent error of ±e, use `--quantum-ns` of at least 2e. These bounds do not
include an independently measured clock-drift bound. A trace does not identify
a unique last sender whenever publication brackets and poll granularity leave
alternatives. Report the bounded cross-die GPU critical path, its span bounds
and their interval width separately from the exact local arithmetic residual.
The interval width is uncertainty, not unaccounted execution time. Host timings
remain a separate clock domain.

Analyze the entire capture window to bound its span. Do not add individual
token envelopes or bounds: one die can begin the next position while another
finishes the current one. The all-CTA graph also joins every sampled body end
to every phase end at an actual grid barrier, preserving same-die CTA skew.

## Validation and measured scope

`trace_layout_test` executes the actual image on one GPU using empty opcode
bodies, six distinct peer-flag banks and two tokens. It checks phase zero,
dispatch/body ordering, exchange generations, send brackets, both layout
mismatches, canaries and exact per-token reconciliation. It is a trace ABI and
control-flow test, not a six-GPU model or numerical-parity result.

Before any target-GPU run, inspect both `nvidia-smi` and `ps`, acquire the
cooperative validation lock, and wait while another worker is active. Never
terminate another worker's process. Record idle resident holders separately
from active workloads. Retain raw before/after checks.

Overhead requires identical prompt IDs, continuation length, sampling, model
layers, placement, QSA size, prefill/snapshot settings and executable source.
Use an ordinary image with tracing disabled and a matching tracing image with
the bounded buffer enabled. Warm each configuration, alternate measured runs
in ABBA order, verify identical output tokens, and keep unrounded client events
alongside the engine's rounded completion fields. State the metric, repeat
count and observed spread. All-CTA tracing is a separate configuration with
its own overhead. Do not treat traced throughput as production throughput.

The completed schema-2 acceptance run is in
[`schema2-c7d0a6e`](../validation/issue13/schema2-c7d0a6e/README.md): 10 real
lifecycle cases, matched CTA-0 and 16-CTA ABBA measurements, exact per-die
reconciliation and bounded cross-die GPU critical paths. It includes raw
captures, commands, source/image identities, uncertainty and the documented
memory failure and restoration.

The historical lifecycle evidence at
[`lifecycle-7262065`](../validation/issue13/lifecycle-7262065/README.md)
retains its original executable identity and scope. The implementation and new
evidence for this revision are tracked in PR #30, linked to issue #13.

## Tools

`ruby kern/trace_reconcile_test.rb` runs synthetic CPU regressions for exact
partitions, independent epoch shifts above 2^53, known makespan bounds and
malformed coverage. These fixtures are not GPU evidence.

`ruby kern/trace_lifecycle_check.rb --socket SOCKET --out NEW_DIR
--source-sha REV --pair-check --preflight EXECUTABLE` checks completed single
and multiple outputs, both stop slots, snapshots, callback cancellation,
validation failures, recovery and paired prefix/resume. The optional preflight
executable receives an evidence-directory argument before each request.

`ruby kern/trace_validation.rb --config CONFIG.json --output NEW_DIR` runs
the fixed-protocol ABBA comparison. The JSON names exact artifacts, resident
process identities, runtime fields and explicit preparation/restoration helpers.
`--check-config`, `--plan` and `--preflight DIR` expose its checks independently.
`trace_control.rb` is the optional explicit idle-core reload/restore helper;
its caller holds the cooperative lock throughout the operation. Before
enabling tracing, it requires matching ordinary runtime settings and checks
free memory plus the trace allocation released on reload against the requested
buffer and a configured reserve (8 MiB by default) on every declared GPU.

`ruby kern/trace_reconcile.rb --dies 6 --out REPORT.json TRACE_FILES HOST_FILE`
analyzes complete schema-2 wire exports, including gzip files. Add
`--position N` for a single token or `--allow-incomplete` for a stopped/aborted
diagnostic prefix. Original metadata remains in the report. It reports exact
per-CTA partitions, same-die envelopes, opcode totals, host events, a causal
witness, offset/span bounds and the unresolved span gap. It requires all
participating dies and their matching host export.
