# Matched decode benchmark

The runner connects to two already loaded cores that differ only in KV placement.
It does not reload either core or any weight holder. Supply a host-pinned socket
and a VRAM socket built from the same engine and cubin. Both must report all
48 layers, a 512-block budget and identical remaining config fields.

    python3 kern/paired_depth_bench.py \
      --host-socket /tmp/flash-host.sock --vram-socket /tmp/flash-vram.sock \
      --source-sha FULL_COMMIT_SHA --artifact /path/libengine.so \
      --artifact /path/die.cubin --artifact /path/flashcore \
      --weight-manifest weights.json --build-record build.json --out NEW_DIRECTORY

weights.json is an array of {path, bytes, mtime_ns, sha256}. Compute full checkpoint
hashes once before timing; the runner checks file size and mtime against that
record. build.json records exact compiler/assembler invocations, flags, conversion
provenance and anything not embedded in the runtime config. Unknown identities
must be null or explicitly unknown; a Git SHA alone does not identify an
uncommitted binary. --dirty-patch records a reviewed patch when applicable.

## What is measured

Fixed deterministic post-template IDs are identical across placements and runs.
They are synthetic prompts through the actual checkpoint, not an answer-quality
evaluation. Stop IDs are cleared; sampling is greedy. Warmup precedes at least
five interleaved pairs. Each measured request generates 129 outputs, giving 128
post-first-output intervals. The next request restores an exact prompt-prefix
snapshot; that restoration is checked, as are output count, stop status and
matching generated IDs. Increasing depths reuse existing prefixes. The requested
depth is the prompt length; individual decode positions are recorded.

Default depths cover 2048, 2051, 2052, 3100, 8000, 65000 and 262000. Capacity
failures are explicit skipped rows, not shorter prompts relabeled as long runs.
Allocation, uploads, compile/setup, warmup, prompt processing and client TTFT
remain outside the named steady-decode interval rate.

Every response's original protocol, generated IDs, client monotonic timestamps
and measured intervals is retained. Summary includes median per-run latency/rate,
p95 individual interval latency and sample counts. A short/stopped/error response,
mismatched manifest, missing trace event or different output sequence fails the
run and cannot enter a passing aggregate.

## Phase attribution

With the trace-enabled engine in PR #30, add --trace and allocate at least 129
trace tokens in both cores. The runner saves compressed CSV for every die and the
host. It includes phase zero, validates every position/phase and uses the final
publication event, not an earlier phase-body timestamp. Dispatch, exchange wait,
body work, PLE wait and barrier/send work form disjoint local-die categories.
It does not sum overlapping work from separate dies.

GPU timestamps are integer nanoseconds. Only intervals within the same die clock
are compared. Host observation and callback boundaries use CLOCK_MONOTONIC; their
absolute epochs are not subtracted from GPU epochs. Client versus engine and
engine versus GPU differences compare complete interval lengths and do not
pretend to give an absolute cross-clock delivery latency.

Instrumented runs measure attribution. Run an otherwise matched untraced pair
to quantify trace overhead before treating traced throughput as production
throughput. Separate profiler runs are required for memory transactions/stalls;
phase time already includes its own memory-access stalls.

## Manifest and failures

Records include engine config, source/patch, binary hashes, exact checkpoint
hashes, fixed IDs, requested count, warmups/order, placement, capacity, PLE mode,
runtime environment allowlist, GPU UUIDs, driver, clocks, temperature, ECC/PCIe
status and host identity. Full command output is preserved even for unsupported
driver counter queries. Successful summaries have status.json and exit.json;
errors leave FAILED.json, exit.json and all collected raw records.

The runner reports the legacy engine run/first fields as reported_post_first_ms.
That is a distinct metric from first-to-last client observation or device
publication. Complete shared CLI/HTTP timing primitives remain issue #15.

## Validation

    python3 kern/paired_depth_bench_test.py

Ten deterministic tests cover phase-zero inclusion, large timestamp epochs,
layout/missing/duplicate/publication errors, PLE bounds, one-output absence of
decode intervals, manifest mismatches and PLE short-read rejection.
The target-GPU paired experiment and raw evidence belong to issue #10, so runner
changes can be reviewed separately from conclusions about historical throughput.
