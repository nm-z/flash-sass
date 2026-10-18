# Issue 54: placement measurement and selection

This PR supplies capture and comparison tools for
[issue #54](https://github.com/nm-z/flash-sass/issues/54). It does not select or
install a new placement. The resident configuration must remain available to
other workers, and changing its split requires matching weight holders.

## What the tools resolve

`kern/placement_capture.rb` runs the request used by `linkmatrix_test.sh`:
the capital-of-France prompt, temperature zero, and at most 40 generated tokens.
Its defaults reproduce the published 35.9 tok/s retest's sample counts: two
rounds, one warm-up per round, and five measured requests per round.

The collector takes the exclusive
`/tmp/flash-sass-engine-validation.lock` itself. It checks `nvidia-smi` and
`ps`, validates explicitly allowed resident PIDs including their start times,
and waits for an idle engine before starting the sequence. Monitoring continues
during the sequence. It does not insert GPU-idle cooldown waits between
requests. The collector releases the lock after closing its own connections,
stopping its own frontend, and observing idle GPUs. A monitoring failure during
cleanup keeps the lock while the idle check retries.

A temporary instance of the existing C `flashd` frontend listens on a private
loopback port. Its engine socket goes through a recording proxy. The proxy
accepts only `info`, `stop`, `sample`, `snap`, and `chat` from that
frontend. It records the actual rendered prompt IDs, stop IDs, sampling command,
all generated token IDs (including hidden controls), and native `done` line.
After `done`, it reads `config` and `prof 0 all` through `prof 5 all`
on the same resident-engine connection, then releases the buffered `done` to
the frontend. The serial native connection keeps those six profiles associated
with that response.

The chat request performs the frontend's normal per-request stop/sampler setup
and prefix-cache operations. It does not change the loaded weights, placement,
kernel, capacity, services, or host settings.

The native timing was fixed before the profile reads, so the HTTP decode metric
does not include those reads. Proxy forwarding can still add token-callback
overhead, and profiling adds a gap between responses. This is an instrumented
version of the request protocol; compare candidates with the same collector.
HTTP wall time is recorded separately. The before/after sample telemetry is the
most recent background observation at each boundary; its UTC timestamps and
the complete telemetry sequence are retained.

Raw native attempts, including partial failures, are retained alongside complete
HTTP responses, profiles, metadata, process checks, GPU telemetry, and root-port
link observations. HTTP failures retain the complete available status and body.
The collector's `status: complete` means collection completed; the selector
performs the numerical and acceptance checks.

`kern/placement_select.rb` compares a baseline with measured candidate
captures. Its objective is the mean, over measured responses, of the largest
six-die **last-token phase span**. It only ranks the configurations actually
measured. It also checks exact output IDs, native/HTTP consistency, configuration
and host comparability, the 0.5 ms spread limit, and the 35.9 tok/s throughput
floor. Baseline-only evidence cannot select a new placement or close the issue.

## Running it

Run the collector on the GPU host, using a separate work directory and an
explicit list of already inspected core/holder/fence PIDs. The PID list must
include all resident GPU users on the six selected dies; any additional GPU
process makes the initial idle check wait. The six NVML indices must be in
engine-die order. Verify the three root-port addresses against `lspci -D -t`;
the defaults are archy's documented card-to-CPU ports.

Build the frontend from the same source for every compared capture:

```bash
make -C kern flashd
ruby kern/placement_capture.rb \
  --flashd "$PWD/kern/flashd" \
  --model /path/to/shard-1.gguf \
  --resident-pids CORE_PID,HOLDER_PIDS,FENCE_PID \
  --output baseline.json --label baseline
ruby kern/placement_select.rb baseline.json candidate-a.json candidate-b.json
```

Substitute numeric comma-separated PIDs and an actual model path. Do not wrap
the collector in another exclusive acquisition of the same lock. It never
runs `linkmatrix_test.sh`, `setpci`, a service restart, a kernel/library
reload, or a holder launch. The temporary frontend uses the caller's
`FLASHD_*` environment; keep it fixed across captures and consistent with the
desired HTTP frontend. Text benchmark requests do not invoke the vision service.

The collector refuses to overwrite an output file. Capture exit codes are 0
for complete collection, 1 for a failed attempted capture, and 2 for invalid
arguments or setup. Selector exit codes distinguish a selected passing
candidate (0), valid evidence with no passing new candidate (1), and invalid
or incomparable evidence (2).

The model path, size, and modification time are recorded as filesystem
identity metadata, not a full cryptographic model-content hash. The native
prompt IDs and generated IDs, frontend executable hash, collector hash, engine
revision and kernel identity provide additional checks; model provenance still
needs review before accepting a placement.

## Why the issue remains open

Current `SPLIT_SHARES=112,97,81,116,97,97` weights row, head, and contiguous
routed-expert partitions together. `ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5` places
attention weights, KV, and indexer state. The shared expert remains on
`block % 6`. These are the existing values, not a newly measured solution.

The observed six resident GPU allocations leave insufficient capacity for a
second equivalent engine instance. Loading a candidate would require replacing
the matching holders and engine, which the assignment expressly prohibits.
No numerical replacement is inferred from the issue's six aggregate waits.

The legacy profile is a CTA-0 phase-end interval, including dispatch, compute,
synchronization, and exchange waiting. Its first phase is omitted. `actq256`
is a wait-heavy consumer interval rather than an isolated pure-wait timer;
MoE exchange waiting also appears in `hcwsum`. Equalizing only `actq256`
does not establish a faster critical path.

A final placement requires authorized candidate loading and matched measurements
on the target host, adequate fenced-memory headroom, unchanged output tokens,
at most 0.5 ms `actq256` spread, at least 35.9 tok/s, and the chosen shares/map
with their profile in `SPEC.md`. Cooling and card-to-CPU link behavior are
possible confounders; this PR does not establish that hardware changes are
necessary. It remains linked with `Refs #54` and remains draft.

## Validation

The CPU capture checks exercise real local sockets with a fake engine: startup
connection release, pipelined commands, exact IDs and six-die association,
refusal of resident-control commands, cancellation during silent prefill,
bounded timeout, partial failure evidence, token-count disagreement, rejection
of foreign GPU processes at zero utilization, and complete HTTP error retention.

Run the CPU checks without a model, CUDA, or an engine:

```bash
ruby kern/placement_capture_test.rb
ruby kern/placement_select_test.rb
```

The final CPU run passed 9 collector checks and 41 selector checks; the same
suites are available through `make -C kern test-placement`. See
[capture-cpu.tap](capture-cpu.tap) and [selector-cpu.tap](selector-cpu.tap).
Selector checks cover passing synthetic candidates, the exact numerical
boundaries, a single outlying response, exclusion of warm-ups, token and reasoning
parity, malformed native/HTTP records, incomplete profiles, changed host or
configuration evidence, and both plain JSON and gzip input.

## Recorded baseline

[archy-baseline.json.gz](archy-baseline.json.gz) is the completed capture from the
current collector (SHA256
`ed85e0e6d7d5cea686cf75b36a50b17eb95339e41a9bfc70cd057a365501b9b4`).
[commands.md](commands.md) records execution and
[capture-final.log](capture-final.log) / [capture-final.exit](capture-final.exit)
record the successful return (exit 0).

[baseline-analysis.json](baseline-analysis.json) is the offline selector report.
Reproduce it with:

```bash
ruby kern/placement_select.rb validation/issue54/archy-baseline.json.gz
```

The expected exit status is **1**,
with `measurement_gates_passed: false` and no selected candidate. Each measured
request's spread and throughput is retained in the report. A selector result
does not by itself satisfy the SPEC or review requirements.

The final run, 00:54:53–00:55:09 UTC on 2026-10-18, measured 35.915 tok/s
(range 35.89–35.96), with the same 38 generated IDs in all ten measured replies.
Its maximum `actq256` spread was 2.3931 ms, so the existing placement still fails
the issue's balance requirement. All observed SM clocks were 1,177 MHz; die 4
was 81–83 °C. Root-port links were Gen3 x8, Gen3 x16, and Gen2 x16.

[archy-baseline-initial.json.gz](archy-baseline-initial.json.gz) retains the earlier
completed run from the preceding collector revision. It measured 35.923 tok/s
and a maximum spread of 2.388 ms. A subsequent CPU cancellation check identified
Ruby's `nil` EOF return; the final collector handles it immediately instead of
waiting for the socket deadline. The final baseline above was repeated after
that correction. Neither run loads a different placement.
