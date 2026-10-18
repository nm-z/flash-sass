# Issue 49 validation

This directory preserves the original measured evidence for PRs
[#47](https://github.com/nm-z/flash-sass/pull/47) and
[#48](https://github.com/nm-z/flash-sass/pull/48), plus a reproducible replay.
Historical records are unchanged; their SHA-256 checksums are in
[historical/SHA256SUMS](historical/SHA256SUMS).
They were recovered from the original validation directory on engi.
The new host regressions are separate from those historical measurements.

## Acceptance evidence

| Requirement | Evidence and scope |
| --- | --- |
| Bounded persistent readers and completed rows | The production pool has 1..16 readers, one batch producer, persistent workers, and synchronous completion. `ple_read_test` checks all reader counts, 2,048 reused batches, row/guard bytes, partial reads, EINTR, EOF, EIO, EBADF, and every worker-start failure. The added timed barrier requires sixteen distinct workers, retains them for another generation, and withholds one row to check that the batch cannot return early. |
| Startup, teardown, reload, real-model parity | Original `flash-ple-7f37f6b-kv0/` and `kv1/` contain commands, runtime settings, stdout/stderr, binary hashes, and results from archy. Across both KV modes they report 144 checks, including 72 full-state parity and 72 byte-count checks, plus six reload checks. New host tests exercise reader-setting preflight without mutating the command or environment on rejection. |
| Compact disk PLE | The pool borrows the file descriptor and row buffers; it neither maps nor pins the PLE table. The model runs used `FLASH_PLE_DISK=1`. |
| Matched local/NFS I/O | Each original filesystem CSV has 2,304 successful batches. Replay checks every sampled offset and fingerprint across local/NFS, and 1,024 exact-input, fully client-resident warm pairs per filesystem. Unprimed serial/parallel arms used disjoint shuffled regions and are descriptive distributions, not same-offset cold pairs. |
| Optional startup warming | PR #48 is merged and its `serve.sh` implementation is retained. Warming is off by default, precedes core launch when enabled, has a standalone command, and fails core startup on read failure. `ple_prewarm_test.sh` checks the real script's control flow with isolated launch stubs and a 12,345-byte file. |
| Actual whole-file warm duration and residency | The original `flash-prewarm-cea6c79.log` records 28,800,138,432 bytes read in 70.7769 seconds, exit 0, and maximum RSS 10,264 KiB. The before/after `fincore` records show 14,385,856,512 and 20,611,133,440 resident bytes. Pages remain reclaimable; this is not a claim of permanent or complete residency. |

## Historical source identities and measured limits

- I/O and model runs: `7f37f6bdc69a0286869ead4d0e8231fa601b23c1`.
- Whole-file warming: PR #48 head `cea6c79a42d78dfd5d1580c11b1869c37c598c7b`,
  merged by `02a21508be1d6c893bbfb26c9e59651a014ba1a0`.
- The model checks use six dies and all 48 layers, with capacity 128,
  `FLASH_NBLK=4`, and `FLASH_MAXIMG=0`. They are bounded parity checks,
  not production-depth or external-quality evaluation.
- The original I/O CSVs have the legacy 13-column schema and earlier arm order.
  Schema 2, per-set balanced order, and cross-arm full-byte validation are
  new runner improvements. Do not relabel old timings as new-runner results.
- Original model commands and results are preserved; the external model harness
  used for those runs is not part of this repository.
- The historical I/O archive lacks standalone command-exit files. Its recorded
  final `pass=1` and complete raw rows are checked; missing exit records are not
  reconstructed. The new collection wrapper records individual exits.
- `mincore` measures the client page cache. It does not establish NFS server-cache
  or physical-media state. Separate audit call-region overlap is not proof of
  simultaneous physical storage requests and must not supply performance timing.

The historical total-latency medians for fully resident, exact-input warm pairs
were 23.297/60.947 microseconds on local storage and 24.990/68.249 microseconds
on NFS (serial/16 readers). These observations support keeping parallel readers
opt-in; no 16x speedup is claimed. Complete groups and quantiles are in
[historical/replay.json](historical/replay.json).

## Reproduce the replay and host checks

From the repository root:

~~~bash
(cd validation/issue49/historical && sha256sum -c SHA256SUMS)
perl validation/issue49/replay_historical.pl validation/issue49/historical
bash validation/issue49/replay_historical_test.sh validation/issue49/historical
make -C kern test-ple
~~~

The replay rejects truncated rows, bad read/error counts, incomplete trials,
warm-pair byte fingerprints that differ, mismatched local/NFS inputs, and a
failed final status. Its negative tests mutate copies of the archived records.

Host tests do not run a model or a disk-heavy benchmark. Before any new GPU or
disk-heavy validation on archy, acquire
`/tmp/flash-sass-engine-validation.lock`, inspect `nvidia-smi` and `ps`, and
release the lock afterward. Never stop another worker's process. The collection
wrapper in [PLE_READS.md](../../PLE_READS.md) records these checks and refuses
observed contention.
