# Bounded phase and host traces

Build `make -C kern die_trace.cubin libengine.so`, then select the explicit `FLASH_CUBIN=/absolute/path/die_trace.cubin` and `FLASH_TRACE=129` in the engine startup or reload environment. Tracing defaults to zero and accepts at most 256 output positions. It reserves 4096 phases × 64 bytes × positions per die (32.25 MiB for 129).

The ordinary image and tracing images have separate targets. `die_trace16.cubin` records all sixteen CTAs and requires a matching diagnostic allocation. The engine records CTA 0. A device-side mode guard refuses mismatched record layouts.

After a response, collect `trace <die>` for every participating die and `hosttrace` before another response replaces the records. Each CSV has request ID, position range, and dimension metadata. Missing or nonmonotonic samples fail explicitly. Layer -1 denotes embedding, output-head, or framework work outside a transformer block.

## Events

Five 64-bit GPU samples divide each phase into dispatch, exchange wait, local operation-body completion, and barrier/send intervals. Phase zero is included. The PLE-row phase also samples its readiness wait; that duration is a subset of its body interval. The final phase has a publication sample after the completed-token count store.

CTA 0's body-end sample can precede completion on other CTAs; its barrier interval includes their remaining work. Use the sixteen-CTA image for physical work distribution. HI/LO/HI timer reads retry across a low-word rollover.

Host records use CLOCK_MONOTONIC for token readback observation, callback entry/return, PLE staging start/end and readiness publication. Bytes and incomplete-row counts accompany staging. Multiple tokens observed in one poll can share a timestamp. Callback completion means the core callback returned, not that an HTTP client received bytes.

## Interpretation and validation

Subtract timestamps only within one clock domain. GPU epochs are independent, and the host clock is another domain. Compare durations and corresponding request/token/phase IDs; do not sum overlapping die times. Gather already includes mapped-memory access, so PCIe cost must not be added again.

The trace starts at the last prompt position and covers a bounded generation window. Prefix snapshot segments retain position-indexed records. Both trace_begin and hosttrace_begin include status, aborted, ended, completed, and requested. The latter two count full request positions, including prompt positions; they are not the bounded trace-row count. Completed is the last observed die-0 publication count, finalized after all streams synchronize on a completed or aborted request.

Status completed with aborted=0 identifies a completed request. Ended=1 can represent an ordinary stop token, so it does not establish cancellation by itself. A client disconnect is exported as status=aborted and aborted=1, even when the dispatcher's shared end bit also makes ended=1. Its completed-prefix trace is diagnostic evidence, not a successful throughput sample. Legacy done fields remain unchanged.

Every call to engine_run, including validation failures, advances the request ID and invalidates the previous trace. Argument, image and device failures use argument_error, image_error, and device_error. Unavailable trace commands return an error carrying the current request ID and lifecycle fields; they do not expose prior records or salvage partially written GPU records. Requested=0 means invalid dimensions prevented a meaningful requested-position count. A transport-level command rejected before engine_run is not a model request.

Request IDs are local to one loaded library. The immediate device-error protocol line retains the failed request's identity before core recovery; subsequent trace queries refer to the replacement library. No faulted GPU trace is salvaged.

Historical traces predating lifecycle fields can still be analyzed as historical evidence, with request status explicitly unknown. New traced benchmarks require complete lifecycle metadata and must reject aborted/error records and any request that stopped before the requested continuation.

Measure overhead using identical prompts, continuations and runtime settings with ordinary and tracing images. Instrumented throughput is a separate result. Keep all raw CSVs and completion records. Assembly and scalar parity alone do not establish target-GPU trace correctness.

This implements profiling infrastructure for #13; the matched investigation is #10, the runner is #14, and shared CLI/HTTP timing definitions are #15.

## Request lifecycle and paired-prefill regression

The eight-case owned-core regression passed at executable source
`72620653d4bb3dc10e3cb405695550db7152bae7`. Normal completion, stop/resume,
client disconnect, argument error, image error, recovery and actual paired
prefill/resume have matching lifecycle metadata on all six dies and the host.
The request IDs advance on failed attempts and stale trace data is unavailable.

Run `python3 kern/trace_lifecycle_check.py --socket OWNED_SOCKET --source-sha REV
--out NEW_DIRECTORY --pair-check` against an owned, traced engine with
`FLASH_TWO_POSITION=1`. The committed [raw evidence and exact scope](../validation/issue13/lifecycle-7262065/README.md)
include build identities, full compressed traces and the launch environment.
Device-fault injection, cross-reload ID continuity and controlled tracing overhead
are not established by this run.
