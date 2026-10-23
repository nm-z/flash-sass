# Parallel PLE row reads

Disk-backed PLE uses sixteen 90-byte rows per position. The host hashes their row
numbers, issues the existing `POSIX_FADV_WILLNEED` hints, and reads the rows into
the compact pinned table. A persistent pthread pool lets the row reads overlap.
The producer waits for the whole batch before publishing the position's ready flag.

## Configuration and lifetime

- `FLASH_PLE_DISK=1` enables the existing disk-backed PLE path.
- `FLASH_PLE_READERS` defaults to `1`. Set it to `16` for sixteen concurrent reads.
  Decimal values `1..16` are accepted;
  empty, malformed, and out-of-range values fail engine initialization.
- `FLASH_PLE_READERS=1` executes the reads inline, without worker threads. Use
  this control to compare I/O timing or when warm-cache wake-up overhead matters.
- The `config` response reports `ple_readers`; it is `0` when disk PLE is inactive.

The engine creates workers once during initialization and joins them during
teardown, before the library is unloaded. Workers borrow the core's file descriptor
and write disjoint row slots. They do not call CUDA, retain a private file cache,
or pin the full table. The existing compact buffers and their size are unchanged.
Partial reads are completed and `EINTR` is retried. The pump retains responsibility
for short-read diagnostics and ready-flag publication.

## Correctness checks

```sh
make -C kern ple_read_test
./kern/ple_read_test
```

The C test compares actual file bytes for all reader counts, first/last and
page-straddling rows, duplicates, and 2,048 batches that reuse the same pool.
It also checks guards, partial reads, `EINTR`, EOF, `EIO`, invalid descriptors,
double teardown, and failure at each of the sixteen thread-creation points.

## File-I/O measurements

Run the benchmark before pre-warming the file:

```sh
make -C kern ple_read_bench
./kern/ple_read_bench "$GGUF2" 192 128 20261017 > ple-reads.csv 2> ple-reads.log
```

The arguments are the file, table byte offset, batches per arm, seed, and an
optional parallel reader count (default 16). Offset `192` applies to the model's
second GGUF shard; use `0` for an extracted table.

The unprimed arms use disjoint, shuffled 1 MiB regions and alternate serial/parallel
execution order. Each arm includes all sixteen existing `WILLNEED` hints and the
complete batch read. Separate warm arms use identical, already-read row sets.
An untimed scalar read checks every measured batch byte for byte.

CSV records contain the row offsets, reader count, pre-hint `mincore` residency,
hint/read/total duration in nanoseconds, bytes, errors, and output fingerprint.
The stderr record includes hostname, revision, seed, CPU time, context switches,
and peak RSS. Thread creation is outside the measured batches.

`resident_pages=0` describes the client's page cache. It does not establish
cold NFS server caches or cold physical media. Mixed-residency samples must be
reported separately. The benchmark never drops system caches or evicts file pages.
Existing hints may already overlap storage work, and worker synchronization adds
overhead to warm reads; parallelism does not imply a sixteen-fold speedup.

For a separate audit of actual read-call intervals:

```sh
make -C kern ple_read_audit
./kern/ple_read_audit "$GGUF2" 192 1 20261018 > ple-audit.csv 2> ple-audit.log
```

The audit records thread IDs and timestamps around actual `pread` calls, plus
the maximum overlapping call regions. Those regions can include scheduling time;
they do not prove that storage serviced every request simultaneously. Audit
instrumentation affects timing, so use the normal benchmark for latency comparisons.

## Shared-host collection

Build both tools before taking the shared validation lock, then run:

~~~sh
./kern/ple_io_run.sh NEW_RUN_DIRECTORY LOCAL_SHARD NFS_SHARD 192 128 20261017 16
~~~

The wrapper holds /tmp/flash-sass-engine-validation.lock and checks nvidia-smi
and ps before each benchmark. A held lock, another GPU process, active GPU
utilization, or a recognized disk benchmark causes exit 75 before submission.
It sends no signals and never restarts a service. The lock is released on success
and failure. Run it again when the shared host is available.

The directory contains the exact command arguments, individual command exits,
CSV and stderr, source revision and dirty diff, binary SHA-256 identities,
file stat identities, GPU metadata, and process observations. The local and NFS
files must have the same length so the seed constructs the same row sets.
The audit follows the timing experiments and has separate files. A completed
collection is raw evidence; validate its records before drawing conclusions.

## Interpreting the comparisons

An unprimed row set has not been read earlier by this benchmark. Its serial
and parallel arms use disjoint shuffled regions, so this is a randomized
comparison of different inputs with the same workload shape. It is not a
same-offset cold-cache pair. Client-cache absence says nothing about the NFS
server cache or physical media.

Warm serial/parallel arms use identical offsets and require identical output
fingerprints. Repeated uses of each row set alternate which arm runs first.
Both members of a fully resident paired summary must actually have all requested
client-cache pages resident. Classify all other samples by the recorded
pre-hint residency instead of treating the phase name as a cache guarantee.

The current CSV explicitly records validity, reference/parity failures, observed
cache class, comparison type, sample-set/repetition/order, instrumentation,
and audit event retention. Failed measurements remain visible and return a
nonzero process exit; they cannot be included in a successful summary.
Reference byte checks remain outside the measured hint/read durations.
Instrumented audit rows and audits with dropped events cannot establish
uninstrumented latency or complete read-event coverage.

## CPU regression scope

The reader tests cover byte/guard parity and lifetime, plus deterministic
concurrent entry and delayed-row completion checks. An invalid reader-count
reload is rejected before modifying the environment or unloading the engine.
Small startup tests use controlled launch/read stubs to verify that cache
warming completes before core launch and that a read failure suppresses launch.
The shared-host wrapper tests use inert GPU and benchmark commands and a
private advisory lock. They never acquire the production coordination lock,
so the CPU suite can itself run under the shared validation lock.

These CPU checks do not replace real-model parity, reader reload tests on archy,
or measurements of the local shard, NFS, and full-file cache warming at a
declared source revision. Historical records keep their original tested
commits; new implementation changes require appropriately scoped new evidence.

Run the committed CPU suite with:

~~~sh
make -C kern test-ple
~~~

The collection wrapper also runs `ple_io_validate.pl` before its separate
audit runs. The validator requires schema 2, successful untimed validation,
complete trial counts and order, stable row sets, and identical sampled
offsets/fingerprints across local and NFS. Only warm pairs resident in both
arms enter its descriptive latency medians. It rejects an instrumented CSV,
a truncated/failed record, or a mismatched shard; it does not infer cold
same-input speedups from disjoint unprimed samples.

To replay a newly collected pair without disk or GPU work:

~~~sh
perl kern/ple_io_validate.pl OUT/local.csv OUT/nfs.csv 128 16
~~~

Original measurements and their exact source revisions are preserved under
[validation/issue49](validation/issue49/README.md). Their earlier CSV schema
and arm ordering are retained and checked by a separate historical replay.
