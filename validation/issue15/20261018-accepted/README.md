# Accepted timing validation — 2026-10-18

**PASS**, on the real target GPUs, from **07:10:00 to 07:10:12 UTC**.

Tested implementation: [`8ad9d6efded1be7aecb2398fcabd7faa201667c3`](https://github.com/nm-z/flash-sass/commit/8ad9d6efded1be7aecb2398fcabd7faa201667c3). The branch's later documentation/evidence commit preserves the measured engine, frontend, device, and checker source. [Artifact SHA-256 hashes](artifact-sha256.txt) identify the binaries in the frozen target directory. [The supervisor result](evidence/results.json) records its own hash, source SHA, effective configurations, resident identity, every command/exit code, reload protocol, and restoration result.

## Execution and restoration

[queue-adaptive.sh](queue-adaptive.sh) waited on `/tmp/flash-sass-engine-validation.lock` using descriptor 9. [select-target.pl](select-target.pl) resolved the current Unix peer, process start tick, and mapped library hash **under the acquired lease** and selected resident mode. This run used PID **1369637**, start tick **8980361**; these are recorded historical identities, not values to reuse for a later run. [selection.json](selection.json) records the exact selected invocation. The supervisor and its children shared the same locked open-file description throughout validation.

The runner first loaded the candidate without changing environment values to read an older baseline's otherwise unavailable trace setting. It then measured scalar and paired configurations, cleaned up each private frontend, and restored the original logical engine path and effective configuration. Restoration verified library SHA-256 `8ecd2476d48082be2655e1ad4b20a23163a9495a2ebbfd5126ff2990ee478b1b` and every originally reported configuration field at 07:10:12 UTC. [The queue log](queue.log) ends with `RUN_EXIT=0`. The persistent core, contexts, IPC weight holders, memory fence, and existing frontend remained in place. Reload clears engine snapshots and caches.

Both candidate configurations used the complete 48-layer model, six Tesla M60 dies, driver **470.256.02**, `nblk=512`, device-resident KV, disk-backed PLE, one PLE reader, `kv_trace=0`, `maxt=512`, `img_rows=1`, `trace_tokens=4`, and `prefill_chunk=0`. The sole mode difference was `two_position=0` or `1`. Full split/attention maps and fingerprints are in the supervisor result; [GPU inventory before](evidence/gpu-before.log) and [after](evidence/gpu-after.log) retain device identities and observed clocks.

## Request and event checks

| Suite | Cases | Result |
| --- | --- | --- |
| [CLI scalar](evidence/cli-pair0/results.json) | Four outputs; snapshot creation; resumed prefix; first-output stop; oversized request rejection | 5 PASS; paired-only case correctly skipped |
| [CLI paired](evidence/cli-pair1/results.json) | The same five cases and a paired prompt | 6 PASS |
| [HTTP scalar](evidence/http-pair0/results.json) | JSON and SSE using the actual frontend/tokenizer | 2 PASS |
| [HTTP paired](evidence/http-pair1/results.json) | JSON and SSE using the actual frontend/tokenizer | 2 PASS |

The 15 passing validation cases include two argument rejections. Thirteen successful requests contribute **38 generated outputs with host callback events**. The CLI captures 30 local socket observations; the HTTP requests capture eight output events, two first successful SSE text writes, and four final response writes.

For each traced request, the checkers match source/request identities, complete-position and generated-output counts, all relevant raw device rows, integer start/publication endpoints, host observation/callback ordering, and rate formulas. Four-output CLI requests report three decode intervals. First-output stops report one generated output, zero decode intervals, and null decode rates. Oversized requests reject execution and expose null unknown counts rather than the preceding request's data. HTTP aliases equal the canonical nested engine values; response writes are correlated through the frontend request ID and post-write log.

[summary.json](summary.json) contains all case-level counts/rates and measured host ranges. Generate it from the captured result files with:

```sh
perl validation/issue15/20261018-accepted/summarize.pl
```

See [the parent validation report](../README.md#measured-observation-and-transport-intervals) for the measurement table and [TIMINGS.md](../../../TIMINGS.md) for exact metric definitions. These are observations from one instrumented acceptance run. They do not subtract host/device epochs or establish client receipt, absolute GPU-to-host latency, or uninstrumented throughput.

## Raw evidence and checksums

The `evidence/` directories retain raw CLI commands/responses, HTTP request/response bodies and metadata, per-request device/host traces, canonical JSON, and correlated post-write logs. Large CLI journals and device traces are gzip-compressed without changing their decompressed bytes:

```sh
gzip -dc validation/issue15/20261018-accepted/evidence/cli-pair0/protocol.jsonl.gz
gzip -dc validation/issue15/20261018-accepted/evidence/http-pair1/sse.trace.protocol.gz
```

Baseline binary copies remain in the original private run directory on archy; their verified hashes and restoration protocol are published here. The captured IPC layout is unused in this resident-selected run and is retained to record the selector's fallback input. No unrelated process command-line snapshot is included.

`SHA256SUMS` covers the published package. Verify it from this directory with `sha256sum -c SHA256SUMS`. The separate `artifact-sha256.txt` records the original target binary paths and is verified against those binaries on archy.

## Build and deterministic regression evidence

[build-target.sh](build-target.sh) transferred a `git archive` of the exact implementation commit and built on archy with CUDA and its installed driver library. The initial command deadline interrupted the host executable portion after the cubins and engine library had built; this partial attempt is recorded as **exit 124**, not a complete build pass. [resume-build-and-queue.sh](resume-build-and-queue.sh) forced a fresh build of all host binaries and validation executables, recorded all ten artifact hashes, checked the resident runner's syntax, and queued the request validation. This completion exited **0**. [build-results.json](build-results.json) and both compressed build logs retain that distinction. Existing compiler warnings are preserved in the logs.

The CPU checks on engi passed: canonical timing, prefill planner, PLE reader/reload/benchmark/launch fixtures, bounded quantization, 36 guard assertions, 62 resident-plan assertions, and **35 HTTP fixture groups covering 32 transport cases**. The actual session and real-tokenizer regression also passed on archy without initializing CUDA in those fixture executables. Commands and logs are recorded in this package.

The real run preceding the SSE fix failed because a protocol LF entered an SSE JSON data field. [Selected original failure evidence](sse-regression-before/README.md) and [the stricter before-fix fixture](cpu/sse-before.tap.gz) reproduce it. The corrected fixture's [after-fix TAP](cpu/sse-after.tap.gz) passes, including LF/CRLF framing, exact canonical JSON forwarding, and normal/error/cancellation transport behavior. The passing real JSON and SSE requests above validate the production fix.
