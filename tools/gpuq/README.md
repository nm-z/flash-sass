# GPU queue history

`gpuq run` keeps the FIFO queue and inherited GPU lock, and records every job in
`~/.local/state/gpuq/history.sqlite3` on the execution host. Override with
`GPUQ_DB` or `--db PATH`. The database and its directory are private to the user.
Clients using the same database share history. A client using another database
has a separate history even when both clients share the queue and lock. This
replaces the old print-only queue wrapper on archy.

## Run and inspect

```sh
ssh archy gpuq run --name decode-short -- COMMAND ARGS...
ssh archy gpuq latest
ssh archy gpuq logs latest
ssh archy gpuq metrics JOB_ID
ssh archy gpuq search 'decode-short'
ssh archy gpuq rerun JOB_ID
ssh archy gpuq compare decode.engine.tokens_s JOB_A JOB_B
ssh archy gpuq query 'SELECT * FROM job_metric_status WHERE job_id="JOB_ID"'
```

### Visibility across history databases

`gpuq ls` lists active jobs from only the selected history database. It does not
enumerate sibling databases or reconstruct jobs from the shared FIFO. For a job
recorded in another database, use that client's exact database path with
`gpuq --db PATH ls` (or set `GPUQ_DB` to that path). For example, the history
file `/home/nate/.local/state/gpuq/issue9-history.sqlite3` is inspected with:

```text
gpuq --db /home/nate/.local/state/gpuq/issue9-history.sqlite3 ls
```

If the path is not known, check the submitting client's `GPUQ_DB` setting or
the `--db PATH` argument it used. Other clients' history paths are not
discoverable from the current `ls` output.

The shared queue directory contains `.job` ticket files for clients waiting for
the lock. Each ticket contains a PID and job ID; the runner removes its ticket
after acquiring the lock. The FIFO therefore helps explain queue order, but it
does not list jobs that are already running. To see those jobs, inspect their
history database with `gpuq --db PATH ls`.

Run/emit commands print no normal output. Command stdout and stderr are
captured verbatim, including binary bytes and unterminated lines, into compressed
database documents. Noninteractive stdin is captured as it is forwarded. Rerun
reuses the recorded argv, working directory, exclusions and stdin; it records the
current executable/source identities, so a changed implementation is visible.
Supplied script files are saved as source documents. `document ID --raw` is an
explicit database read for feeding an existing analysis tool; ordinary queries
return JSON. Integers outside JSON's exact binary64 range are displayed as decimal
strings. Original integer values remain exact in SQLite and raw records.

The child exit code is returned. Queue cancellation, signals, missing commands,
capture failures and interrupted owners are recorded. `jobs`/`ls` reconcile dead
owners as `lost`, never successful. Nested queue jobs inherit the same lock and
record their parent job. A storage failure prevents an unrecorded command from
starting; a storage failure during capture terminates the owned command. If the
database itself cannot be opened, run fails without executing the command; use
`gpuq doctor` to inspect that bootstrap error.

## Default collectors and exclusions

```sh
ssh archy gpuq run --exclude gpu,trace -- COMMAND ARGS...
ssh archy gpuq run --exclude all -- COMMAND ARGS...
```

| Collector | Default data |
| --- | --- |
| `timing` | Command wall time and normalized producer timing records. Engine, client and legacy reported rates have separate names. |
| `process` | Waited CPU time, maximum RSS, faults, I/O operation counts and context switches; sampled process-tree RSS/I/O/CPU and host memory/pressure. |
| `gpu` | Timestamped GPU utilization, memory, clocks, temperature, power, PCIe link and throttle masks; full before/after driver XML and process inventory. Device totals include concurrent workloads. |
| `engine` | Before/after resident-core configuration and process identity, when the configured Unix socket exists. |
| `trace` | Job-provided phase/host CSVs and available instrumented engine traces. Shared-engine snapshots are explicitly unattributed context. |
| `profiler` | Counter records supplied by the command/profiler. |
| `provenance` | Host identity, executable/argument-file hashes, Git revision, status and dirty diff. Large model inputs retain metadata and require their existing hash manifest for full content identity. |

Collectors run by default at a one-second sampling interval. `--interval SECONDS`
changes it. `--engine-socket PATH` selects the core. `--timeout SECONDS` optionally
terminates a command. No new kernel watchdog is imposed by the recorder.

The generic queue cannot manufacture hardware counters or kernel phase traces
from an executable that does not expose them. Enable the existing trace producer
as described in `docs/phase-trace.md` for per-phase measurements; supply profiler
output for hardware counters. Their collectors are enabled by default, but an
absent producer is recorded as **unavailable**, not as measured zero or an implicit
exclusion. The queue does not reload a shared model or replace its cubin to obtain
a trace. `job_metric_status` exposes this distinction for every standard metric.

Identity, lifecycle timestamps, exclusions, recorder implementation hashes,
stdin/stdout/stderr, supplied scripts, collection errors and generated text
artifacts are mandatory. `--exclude all` disables optional measurements while
retaining those records. Queue/setup/final collection durations are separate from
the command's wall time. Sampling and output capture can still affect workloads;
exclusions and the recorded measurement profile must accompany comparisons.

## Text and existing producers

Prefer direct records:

```sh
gpuq emit decode.engine.tokens_s tokens/s 40.1
printf '%s' '{"context":2048,"cache":"warm"}' | gpuq put workload
```

These run inside a job using `GPUQ_JOB_ID`/`GPUQ_DB`. A command can also print a
JSON object with `gpuq_metric`, `value`, `unit`, and optional `dimensions`; the
output is captured and normalized. Existing `engine_decode_tps`, `decode_tps`,
HTTP timing fields, token intervals, legacy `done` records and phase/host CSVs
have adapters. Raw records are retained even when not understood by an adapter.
The observations from two sources are not silently deduplicated or averaged.

Legacy programs may write into `$GPUQ_RUN_DIR` (also `$GPUQ_ARTIFACT_DIR` and
`TMPDIR`), or use `--capture PATH`. Nonexistent path arguments are also inspected
as possible output destinations. Newly created text files there are committed to
the database and then removed. Existing files, source-code files and Git
repositories are preserved. Writes to arbitrary unlisted filesystem paths and
files deleted by the program before collection cannot be intercepted by a
generic subprocess wrapper; use the direct API or managed output directory.
This is an agent workflow, not a filesystem security sandbox.

## Old evidence and new history

Old traces and run logs are kept in one ZIP archive. They are not imported into
this database. History starts with new queued jobs.

`metric_history`, `job_metric_status`, and `phase_intervals` are reusable SQL
interfaces. `query` uses a read-only connection; full text search covers captured
output, scripts and artifacts. Failed jobs remain visible. Phase intervals
subtract within one device clock; `body_ns` includes post-wait dispatch and PLE
wait is a subset of body time. Do not sum concurrent die intervals.

```sh
gpuq query 'SELECT job_id,metric,unit,value_json FROM metric_history WHERE metric="decode.engine.tokens_s"'
gpuq compare phase.body_ns JOB_A JOB_B
```

## Install and check

Ruby, the `sqlite3` and `csv` gems, and GNU `time` are required. Archy uses gems in
`/usr/local/lib/gpuq/gems`; the launcher adds that path only inside the recorder.
Use `install.sh SOURCE_DIRECTORY FULL_COMMIT_SHA EXPECTED_OLD_WRAPPER_SHA256`
as root. Installation preserves the original wrapper and uses an immutable
version directory plus an atomic launcher replacement.

```sh
ruby tools/gpuq/gpuq.rb run --name gpuq-integration --exclude gpu,engine,trace,profiler -- ruby tools/gpuq/test.rb
gpuq logs latest
```

The integration suite exercises database-only output, exact exits, stdin replay,
queue ordering/cancellation, inherited leases, artifacts, precision, historical
import, read-only queries and interrupted-owner recovery. Real device collection
is checked separately on archy through the installed queue.

## Initial deployment on archy

Installed recorder source: `38f13b5` (October 18, 2026). The installed integration
job `cfb49df7-abdd-45d3-93e7-06195d0d36dd` passed all 21 checks. Its database
records include real GPU samples, process resource usage, captured output and
explicit trace unavailability for the uninstrumented resident engine. An earlier
cancellation race was fixed and its failed job remains in history.

Old evidence is archived separately at
`/home/nate/.local/state/gpuq/archives/flash-sass-old-traces-20261018.zip` on archy
(51,508,951 bytes; SHA-256
`aeff637b04e082d1f1f0cbeeecb3471f3b28584d09eb7194ae91e8d8c768613e`).
Archive job `c85e7e3d-2e1f-4765-a751-670bda35199a` created it and successfully read
it back. It includes repository validation evidence, phase-trace and benchmark
follow-up PR evidence, and performance-run records from engi and archy. Model
GGUFs and generated library/cubin copies are excluded. No old run records were
imported into the fresh history database.
