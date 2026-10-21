# Issue 15 timing validation

## Accepted target run — 2026-10-18

**The real target-GPU request, event reconciliation, and separate observation/transport measurement gate passed.** The frozen implementation is [`8ad9d6efded1be7aecb2398fcabd7faa201667c3`](https://github.com/nm-z/flash-sass/commit/8ad9d6efded1be7aecb2398fcabd7faa201667c3). Later documentation and evidence commits do not change the measured runtime or checker source.

On archy, the queued validator acquired `/tmp/flash-sass-engine-validation.lock` at **07:10:00 UTC**, selected the current resident core while holding that lease, and completed at **07:10:12 UTC**. Both scalar and paired CLI suites passed. HTTP JSON and SSE passed in both modes. The runner verified the original library bytes and every originally reported effective configuration field before releasing the lease. The [accepted evidence](20261018-accepted/README.md) includes the command journal, raw events, responses, artifact hashes, CPU checks, and reproducible measurement summary.

| Gate | Result | Evidence |
| --- | --- | --- |
| Frozen CUDA build, ordinary and traced cubins, engine/core/frontend and validation binaries | PASS; final build exit 0 | [Build record](20261018-accepted/build-results.json), [artifact hashes](20261018-accepted/artifact-sha256.txt) |
| Real CLI, scalar mode | PASS: four outputs, snapshot creation/resume, first-output stop, oversized argument rejection | [Results](20261018-accepted/evidence/cli-pair0/results.json) |
| Real CLI, paired mode | PASS: the same cases plus the paired-prompt request | [Results](20261018-accepted/evidence/cli-pair1/results.json) |
| Real HTTP JSON and SSE, scalar mode | PASS | [Results](20261018-accepted/evidence/http-pair0/results.json) |
| Real HTTP JSON and SSE, paired mode | PASS | [Results](20261018-accepted/evidence/http-pair1/results.json) |
| Device endpoints, generated counts, G−1 rates, host callback order, and request identities | PASS for all measured requests | Raw protocols beside each result; large traces are gzip-compressed |
| Baseline library and effective configuration restoration | PASS at 07:10:12 UTC | [Supervisor result and protocol journal](20261018-accepted/evidence/results.json), [queue log](20261018-accepted/queue.log) |
| Actual session, callback/cancellation, and forced allocation failures on archy | PASS, exit 0 | [Session fixture](20261018-accepted/evidence/session-fixtures.log) |
| Actual tokenizer output-token regression on archy | PASS, exit 0 | [Tokenizer fixture](20261018-accepted/evidence/output-token-regression.log) |
| Actual frontend fixtures | PASS: 35 groups, including 32 JSON/SSE transport cases | [TAP](20261018-accepted/cpu/sse-after.tap.gz) |
| Canonical timing and prefill planner fixtures | PASS | [Timing](20261018-accepted/cpu/timing-after.log), [planner](20261018-accepted/cpu/prefill-after.log) |
| Shared-lease/process guards and resident restoration plan | PASS: 36 and 62 assertions | [Guard](20261018-accepted/cpu/guard-after.log), [config](20261018-accepted/cpu/config-after.log) |
| Merged PLE reader/reload/launch fixtures and bounded quantization fixtures | PASS | [CPU build and tests](20261018-accepted/cpu/merge-cpu-build.log) |

The live checker matrix records **15 passing validation cases**, including two oversized-argument rejections. Thirteen successful requests contribute device and host events. The scalar suite correctly skips the extra paired-prompt case because pairing is disabled; that case passes in the paired suite. One-output stop cases retain their generated count and expose null decode rates. Rejected requests expose unknown counts instead of stale prior-request values.

## Measured observation and transport intervals

These ranges are derived from [the accepted summary](20261018-accepted/summary.json), which can be regenerated with [summarize.pl](20261018-accepted/summarize.pl). All rows use **host `CLOCK_MONOTONIC`**, and the sample counts are observations from one instrumented acceptance run.

| Measured interval | Observations | Minimum–maximum, µs |
| --- | ---: | ---: |
| Core callback execution | 38 | 4.642–23.339 |
| Host readback observation → callback return | 38 | 4.755–23.679 |
| Host readback observation → local Unix-socket parse | 30 | 46.665–151.872 |
| First SSE token observation → successful text-write return | 2 | 56.773–92.874 |
| Completion-record construction → response-write return | 4 | 132.895–295.042 |

The socket-parse interval includes scheduling and checker parsing. The SSE text interval includes decoding, buffering, and the successful server write. The final-write interval includes serialization and logging. These measurements do not subtract GPU and host epochs, establish client receipt, or measure absolute GPU-to-host latency. Tracing is enabled, so these samples do not establish uninstrumented throughput or a general performance improvement. [TIMINGS.md](../../TIMINGS.md) defines the numerators, denominators, start/end events, inclusions, null conditions, and legacy aliases.

## Gaps found and fixed

The first real resident run exposed an HTTP SSE framing defect: the frontend copied the terminal LF of the engine's `timings` protocol line into the nested JSON. Plain JSON accepted that whitespace, but an SSE client ignored the following unprefixed line and received incomplete JSON. The fixture had incorrectly treated all lines of an event as JSON, masking the defect.

The frontend now removes only terminal LF/CRLF framing and forwards the remaining canonical JSON bytes unchanged. The fixture parses only SSE `data` fields and checks the terminal `[DONE]`. It reproduced the failure before the fix (exit 255) and passes afterward (exit 0), including a CRLF variant. [Before-fix TAP](20261018-accepted/cpu/sse-before.tap.gz) and [selected first-run evidence](20261018-accepted/sse-regression-before/README.md) preserve that failure. The accepted real JSON/SSE runs above validate the corrected transport.

The earlier allocation review also fixed all three host program-allocation failures: they now return code `-4`, report `allocation_error`, retain measured build time, leave unobserved counters/run time null, emit an explicit CLI error, and keep the session available. The session regression fails on the old implementation and passes on the corrected implementation. Callback-return events precede synchronous cancellation propagation, and HTTP half-close cancellation drains final counts under an absolute deadline.

The merge with current master retains parallel PLE reads, their byte/short-read timing counters, the KV diagnostic fields, attention/quantization fixes, and the prefill planner. Targets that compile the combined engine retain both timing/PLE dependencies and pthread support.

## Shared resident validation

[`run_resident.pl`](run_resident.pl) holds the inherited locked open-file description across baseline capture, candidate reload, all requests, private frontend cleanup, and baseline restoration. It verifies the Unix peer PID/UID and process start tick, the mapped library's SHA-256, the cubin fingerprint, and effective configuration. The target selector records the current resident identity **after** acquiring the lock, avoiding a stale PID captured while another validator is still running.

If an older baseline does not report `trace_tokens`, the runner reads it from the candidate after an initial reload with no environment assignments. Only after this readback and a saved restore command does it change the five declared validation settings. Restoration checks the original library bytes and the original configuration fields. Reload clears engine snapshots and caches; it preserves the persistent core, contexts, IPC weights, and memory fence. A restoration failure retains the lease and its recovery journal while retrying.

The accepted run used 48/48 layers on six Tesla M60 dies, driver 470.256.02, `nblk=512`, device KV, disk-backed PLE with one reader, `kv_trace=0`, `maxt=512`, one image row, four traced output positions, and scalar/paired modes. The complete source/configuration/artifact identities are in the accepted evidence. The individual checkers require the inherited lease and verified resident identity for `/tmp/flash.sock`; the private frontend also checks its actual backend peer.

[`run_target.pl`](run_target.pl) retains its guarded owned-core path. Its explicit IPC-layout option can reuse approved existing weight holders when the shared core is absent; each launch rechecks PID birth/UID, manifest hashes, GPU process types, and competing workers. The default policy still rejects resident allocations. Invocation and recovery details are in [TIMINGS.md](../../TIMINGS.md).

## Historical checkpoints

The following earlier attempts were blocked or incomplete when recorded. They are superseded by the accepted run above; their original results remain available for audit.

| Checkpoint | Result at that time | Evidence |
| --- | --- | --- |
| `5f6c120`, 00:21 UTC | CPU/build checks passed; target preflight blocked by existing allocations | [Earlier follow-up](20261018-followup/) |
| Host-allocation fix `bec699c`, 03:21–03:32 UTC | CPU regression passed; target lock held by the issue-19 validator | [Resumed review](20261018-resume/) |
| Frozen `7f52b08`, 05:20–05:26 UTC | CUDA build passed; owned-core preflight found occupied GPUs, later attempts found the lock held | [Queued retries](20261018-queued/) |
| Frozen `0f8b09c`, 06:57 UTC | Resident CLI and HTTP JSON passed; HTTP SSE failed; baseline restoration passed | [Selected SSE regression evidence](20261018-accepted/sse-regression-before/README.md) |
| Frozen `8ad9d6e`, 07:10 UTC | Scalar/paired CLI and HTTP JSON/SSE, event reconciliation, overhead measurements, and baseline restoration all passed | [Accepted run](20261018-accepted/README.md) |
