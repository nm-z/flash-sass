# Parallel PLE row reads

Disk-backed PLE uses sixteen 90-byte rows per position. The host hashes their row
numbers, issues the existing `POSIX_FADV_WILLNEED` hints, and reads the rows into
the compact pinned table. A persistent pthread pool lets the row reads overlap.
The producer waits for the whole batch before publishing the position's ready flag.

## Configuration and lifetime

- `FLASH_PLE_DISK=1` enables the existing disk-backed PLE path.
- `FLASH_PLE_READERS` defaults to `16`. Decimal values `1..16` are accepted;
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
