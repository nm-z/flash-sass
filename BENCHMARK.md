# Matched decode benchmark

The runner connects to two already loaded cores **on the GPU host**: one
using host KV and one using VRAM KV. It checks their actual processes and
loaded engine copies before sending inference requests. Both must report
48 layers, a 512-block budget and equal remaining configuration. It does
not start, stop or reload a shared core.

[PR #32](https://github.com/nm-z/flash-sass/pull/32) supplied the original
runner and a metadata smoke. It did not close
[issue #14](https://github.com/nm-z/flash-sass/issues/14). The follow-up's
[validation record](validation/issue14/20261018) separates CPU checks and
the shared-host preflight from the still-required target-GPU campaign.

## Collect a run

Run from the source checkout on the GPU host. Compiler tools, source,
weights and both core processes must be readable there.

~~~bash
perl kern/paired_depth_bench.pl \
  --host-socket /tmp/flash-host.sock --vram-socket /tmp/flash-vram.sock \
  --source-dir "$PWD" --source-sha "$(git rev-parse HEAD)" \
  --artifact executable=/absolute/path/flashcore \
  --artifact library=/absolute/path/libengine.so \
  --artifact cubin=/absolute/path/die.cubin \
  --weight-manifest /absolute/path/weights.json \
  --build-record /absolute/path/build.json \
  --core-records /absolute/path/cores.json \
  --prompt-file /absolute/path/post-template-ids.json \
  --answer-reference /absolute/path/answer-reference.json \
  --attribution-run /absolute/path/separate-attribution-run \
  --profiler-record /absolute/path/separate-profiler-record.json \
  --out /absolute/path/new-throughput-run
~~~

The command acquires **/tmp/flash-sass-engine-validation.lock**
nonblockingly, then checks ps and nvidia-smi. It holds the lock through
metadata capture, warmups, paired requests and final driver snapshots,
then releases it on success, caught failure or interruption. A busy lock
sends no requests. Do not wrap the runner in another acquisition of that lock.

Any GPU validation or profiler command outside this runner must also acquire
the same lock, inspect nvidia-smi and ps, and release it afterward. Do not
kill a process you did not start. The lock coordinates participating workers;
it does not make an occupied service or its memory available.

### Separate attribution mode

Use the same arguments with **--trace**, a separate output directory, and
the instrumented cubin. Omit --attribution-run for this collection itself.
This branch includes the metadata protocol and schema-2 trace producer,
integrated from [PR #30 at 257ea63c](https://github.com/nm-z/flash-sass/pull/30/commits/257ea63c6cc431e3a3c5b32f52ce0cfdfb8bfdff).
Build the ordinary and instrumented images with `make -C kern die.cubin
die_trace.cubin`; see [phase trace documentation](docs/phase-trace.md) for
the full trace ABI and build prerequisites.

The runtime setting is FLASH_TRACE, reported as trace_tokens. For 129 outputs,
use FLASH_TRACE=129 and FLASH_TRACE_CTAS=1. The runtime limits
FLASH_TRACE * FLASH_TRACE_CTAS to 256. Sixteen-CTA captures therefore
support short diagnostic windows, rather than the full acceptance window.

Authoritative throughput requires trace_tokens=0 or an engine without phase
tracing. Attribution and profiler files are revalidated and copied into the
throughput directory's attribution/ and profiler/ subdirectories. Their
instrumentation and clocks remain separately identified.

## Required input records

Use real identities and execution history. Placeholder hashes, an empty
object or a list of filenames are insufficient. Commands are argv arrays,
with individual arguments preserved, rather than shell strings.

### Source and build

The runner captures actual full HEAD, Git status and binary diff.
--source-sha must match HEAD. An optional --dirty-patch must be byte
identical to the captured diff. Untracked source files are rejected; commit
them before collection. Only the designated run-output subtree is excluded
from status/diff capture.

**build.json** is an object with these fields:

| Field | Required content |
| --- | --- |
| source_sha | Full 40-character source SHA |
| source_dirty_sha256 | SHA256 of exact git diff HEAD --binary --no-ext-diff bytes; hash the empty diff for clean source |
| steps | Nonempty compiler and assembler records, identifying all output roles |
| conversion | The same conversion record supplied in weights.json |

Each build step contains kind (compiler or assembler), exact argv, explicit
flags array, tool, exit, elapsed_ns and outputs. The tool has a readable
path and matching sha256; optional bytes and mtime_ns are also checked.
Each output has role (executable, library or cubin) and sha256, matching
the actual artifact. Recorded flags must appear in argv. Failed steps
are rejected.

Each conversion record has argv, an independently hashed tool, successful
exit, elapsed_ns, input_sha256 and output_sha256. Build/conversion history
remains operator supplied: checking hashes verifies file identities, not
that a particular historical invocation occurred.

Process verification reads /proc/PID/exe, cmdline, environ and maps.
The actual private /tmp/flashcore-PID-genN.so is hashed; the current file
named libengine.so alone is insufficient. SO_PEERCRED must identify the
verified PID. Runtime revision and dirty state must match the source.
The runtime's loaded-cubin FNV is compared with the SHA256-identified
artifact. This is a runtime report and file comparison, not GPU attestation.

### Model and tensors

**weights.json** is an object:

| Field | Required content |
| --- | --- |
| model | name and sha256 identifying a verified model file |
| files | Nonempty path, bytes, mtime_ns, sha256 records, including conversion input and output |
| tensors | Unique name, file_sha256 and conversion_sha256 records |
| conversion | Conversion record described above |

Every file is rehashed before measurement; size and nanosecond mtime are
checked before and after hashing. Tensor records must refer to verified
files and the conversion output.

The new **tensormeta** command exports actual in-memory arena/tensor tables:
logical/physical die, allocation size, name, offset, bytes, type, rows, K,
stride and experts. It omits CUDA IPC handles. Every loaded tensor must
appear in the model record. Bounds, duplicate names and counts are checked,
and the full loaded inventory must match across placements. Source-file
and conversion attribution remain distinct from that observed inventory.

### Core launch and PLE/cache conditions

**cores.json** names exactly host and vram. Each record contains:

- pid, exact argv and launch_environment;
- engine_reload_commands, an explicit array of argv arrays;
- limitations, an explicit array of descriptions;
- ple_cache_conditions, with mode, disk_identity, page_cache_policy,
  engine_cache_policy, staging_allocation_bytes and preparation_argv
  (an array of preparation argv arrays).

The environment must explicitly name every key in BenchManifest::ENV_KEYS.
JSON null means **unset**; an empty string means **set to empty**.
Several runtime switches test presence, so these are different.
The allowlist covers current engine/core/program/split settings, including
prefill, two-position execution, experimental switches, trace settings,
CUDA visibility/order and PLE behavior. Unknown variables and recognizable
credential-bearing fields are rejected rather than copied into public records.

The initial launch environment is compared with the relevant /proc
environment. Later setenv changes from reload commands are not falsely
described as visible there. Host/VRAM records may differ in FLASH_KV_HOST
and FLASH_SOCK; other relevant launch/reload arguments and cache conditions
must match. A limitations string does not excuse an unexplained difference.
Unknown cache policies are not complete records.

Throughput versus attribution must match executable/library, source/diff,
weights, tensors, prompt, sampling, policy, runtime and placement-specific
launch/cache records. Documented instrumentation differences are FLASH_TRACE,
FLASH_TRACE_CTAS, FLASH_CUBIN, sockets and corresponding runtime trace/cubin fields.

### Prompts and validated answers

**post-template-ids.json** is the complete array of token IDs, in the
vocabulary range 0 through 248319. The runner records its canonical JSON
SHA256, exact request, visible lengths, generated IDs and sampling/stop state.
Without --prompt-file, the historical synthetic fixture remains available
for diagnostics and cannot produce an accepted throughput aggregate.

**answer-reference.json** has schema=1, prompt_ids_sha256, provenance and
depths. Provenance contains method=independent_reference_tokens, a full
reference source_sha, artifact_sha256 and a description of the independent
reference/answer validation. Each supported depth maps to prompt_sha256
for that exact prefix, generated_ids containing 129 validated outputs,
and JSON valid=true.

Canonical hashes use JSON::PP->new->canonical->encode(...) without pretty
printing or a trailing newline. Reference evidence is copied into the run.
Host/VRAM agreement alone is insufficient. The collector requires an
independently validated reference and rejects output mismatches.
The operator must supply genuine evidence; a valid field is not an
automatic semantic judge.

### Separate profiler evidence

The profiler record has schema=2, scope=separate_profiler and matching
source_sha, source_dirty_sha256, prompt_ids_sha256, weights, sampling,
policy, tensors, per-placement configs and provenance. Executable/library
hashes, launch environment, reload history and PLE cache conditions must match
throughput; only the explicitly named tracing settings may differ. It also contains:

- explicit instrumentation settings;
- commands with exact argv arrays and successful exit codes;
- raw_files with readable paths, bytes and sha256 identities;
- nonempty counters with name, unit and an evidence reference to a raw record.

Each counter's evidence names raw_sha256 from raw_files and a JSON pointer
selecting a name/unit/value object in those verified bytes. For example:

~~~json
{"name":"dram_read_bytes","unit":"bytes",
 "evidence":{"raw_sha256":"<actual report SHA256>","json_pointer":"/counters/0"}}
~~~

The raw JSON record supplies the finite numeric value; a decimal string
preserves integers above 2^53. The runner rehashes the bytes it parses, verifies
name/unit, rejects duplicate references and writes the extracted value to its
accepted record. Input counters must not contain a separate declared value.
Schema 1's unbound declarations cannot satisfy acceptance.

Use a profiler's compatible JSON output directly, or preserve a small derived
JSON counter export together with the original report and the adapter's identity,
exact command and exit status. Describe derived exports in instrumentation.
References identify which retained bytes supplied each counter; the operator
remains responsible for supplying actual collector output.

Preserve actual raw reports and available-metric queries. Collect counters
separately from throughput. Phase durations already include memory stalls;
do not add profiler counters as extra elapsed time. Broader counter and
critical-path investigation remains in [issue #10](https://github.com/nm-z/flash-sass/issues/10).

## Measurement policy

Standard depths are 2048, 2051, 2052, 3100, 8000, 65000 and 262000.
A request at depth D generates 129 outputs at visible lengths D through
D+128, giving 128 intervals. The protocol command is **chat 128**:
the final prompt position produces the first output.

Each depth has a fixed nine-output warmup per placement, then at least
five pairs alternating host/VRAM and VRAM/host order. Sampling is greedy:
seed=0, temperature=0, top_k=0, top_p=1, with empty stop IDs.
Each measured request must restore the exact D-1 prefix snapshot.
Warmups are retained separately and excluded from aggregates.

Insufficient equal capacity produces an explicit capacity_skip with
required positions, capacity and reason. Neither prompt nor continuation
is shortened. No supported depths is a failure. Smaller --outputs and
custom --depths are diagnostic and do not satisfy the complete matrix.

Summary contains median per-run mean interval latency and reciprocal rate,
p95 of raw individual intervals, and sample counts. Incomplete, stopped,
aborted, errored, mismatched or unvalidated answers cannot enter summary.json.

## Timing definitions

Durations use milliseconds; counts and raw timestamps accompany them.

| Metric | Boundaries |
| --- | --- |
| timing.build_ms | Engine request construction/setup and uploads before execution |
| timing.engine_prefix_observed_ms | Run start to host-observed completion of known-prefix work, excluding the final prompt position |
| timing.engine_first_observed_ms | Run start to completed host readback of the first generated batch |
| request_ttft_ms | Client request boundary to first client-observed output |
| engine_observed_decode_ms | Last minus first generated host-readback observation |
| client_decode_ms | Last minus first client-observed output, with raw individual intervals |
| device_publication_decode_ms_by_die | Last minus first publication within each die's clock |
| request_completion_ms | Client request boundary to complete terminal response |
| timing.engine_request_ms | Engine entry to cleanup and terminal lifecycle record |

A prefix supplied entirely by a snapshot has zero remaining known-prefix
work. An unavailable observation stays null. prefill_device_ms is explicitly
null: a host prefix boundary is not a device prefill measurement. Build
here means request construction; offline compiler time is in build.json.

**requestmeta** uses integer uint64_t nanoseconds from CLOCK_MONOTONIC,
independent of phase tracing. Each engine attempt clears the prior record
and advances its ID, including validation failures. It records requested
and completed positions, prompt/resume/snapshot, abort/stop/error state,
PLE short reads and host boundaries. The legacy done response is unchanged.
Its rounded run_ms-first_ms remains reported_post_first_ms, distinct from
a first-to-last observation interval.

GPU epochs are local to each die. Do not subtract host/GPU timestamps or
add overlapping die/CTA times. Interval-length differences do not establish
absolute delivery latency. Shared CLI/HTTP work remains tracked by
[issue #15](https://github.com/nm-z/flash-sass/issues/15).

## Trace interpretation

Schema-2 parsing requires phase zero, every phase/CTA and final publication.
Missing, duplicate, out-of-order and nonmonotonic records, invalid PLE/send
brackets, incomplete lifecycle metadata and PLE short reads are rejected.

Disjoint local categories are dispatch setup, no-exchange dispatch,
exchange wait, opcode dispatch, body excluding PLE wait, PLE wait and
barrier/send. Phase gaps, publication tail and prior-publication transition
are retained separately. op_mean_us uses CTA 0 and excludes the first
captured output. Other lanes remain independent.

Actual attnpart identifies dense attention and attnpartn identifies QSA.
The runtime selected_keys field appears on every phase and represents the
QSA budget formula; for dense attention effective keys are the full visible
length, and both values are retained. Only attention partials count as
layer executions. Accepted default-QSA traces require all 12 attention
layers at every captured position, on exactly one die, with budget 512.
QSA before the cap may attend all keys; capped is recorded per execution.

The throughput collector rehashes/reparses attribution CSV, checks request
identity/outputs and recomputes interval durations. It records measured
trace overhead without clamping negative differences.
context-increments.json compares supported depths to the lowest supported
depth and includes 3100-to-8000 explicitly when present. Per-die opcode
deltas and the remaining between-record/publication term appear beside
host interval deltas. This local accounting is not a causal cross-die
critical-path result or a reproduction of the old 36/28/19 tok/s claim.

## Outputs and failure states

| Exit | Status | Meaning |
| --- | --- | --- |
| 0 | PASS | Complete standard collection and all supplied-evidence gates passed |
| 2 | COLLECTED_INCOMPLETE | Raw diagnostic/attribution collection complete, with acceptance gaps |
| 1 | FAILED | Initialization, metadata, request, identity or validation failed |

Only exit 0 writes summary.json. Diagnostics write diagnostic-summary.json.
Attribution alone remains incomplete because it is not authoritative
throughput. Failures retain FAILED.json, status.json, exit.json, the
manifest and raw requests. Response data is checkpointed before validation.
Caught errors/disconnects preserve partial output and the actual error line.
An uncatchable kill or power loss can leave an initial incomplete checkpoint;
it does not create a successful aggregate.

The manifest retains full nvidia-smi -q -x snapshots before/after collection,
plus per-request queries for UUID, driver, clocks, temperature, ECC mode,
PCIe generation/width, throttle flags and power. ECC/PCIe error counters
remain in the original XML. Failed metadata capture cannot pass.

## CPU verification

~~~bash
prove -v kern/paired_depth_bench_test.pl kern/bench_manifest_test.pl \
  kern/bench_trace_test.pl kern/bench_acceptance_test.pl
cc -std=c11 -Wall -Wextra -Werror -pedantic \
  kern/request_meta_test.c -o /tmp/flash-request-meta-test
/tmp/flash-request-meta-test
make -C kern dieops.h
cc -std=gnu99 -O2 -ffp-contract=off -Wall -Werror -fsyntax-only \
  -Iref -Ikern -I/opt/cuda/include kern/engine.c
~~~

These CPU checks do not validate model answers or target-GPU timings.
The real campaign must still supply matched raw outputs, manifests, phase
attribution, trace overhead and separate profiler evidence before #14 closes.
