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

The trace starts at the last prompt position and covers a bounded generation window. Prefix snapshot segments retain position-indexed records. A failed or aborted response remains distinguishable through its completion/error record; a trace of its completed prefix does not mean the request succeeded.

Measure overhead using identical prompts, continuations and runtime settings with ordinary and tracing images. Instrumented throughput is a separate result. Keep all raw CSVs and completion records. Assembly and scalar parity alone do not establish target-GPU trace correctness.

This implements profiling infrastructure for #13; the matched investigation is #10, the runner is #14, and shared CLI/HTTP timing definitions are #15.
