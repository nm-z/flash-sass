# Resumed validation and lock-isolation fix

Tested implementation: `78eba07942ab9825b2c2d4bc18b4f31be83201e6`.
This directory is an evidence-only follow-up to that clean commit.

## Fixed defect

The CPU wrapper regression used the production GPU coordination lock. Running
it with that lock already held reproducibly failed with
`failed_read: expected exit 9, got 75` (exit 1). The test now rewrites only its
private copy of the runner to use a private advisory lock. It still exercises
real `flock` contention and release, checks that exactly one lock declaration
was rewritten, and leaves the production runner's lock path unchanged.

The complete CPU suite now passes with its caller holding
`/tmp/flash-sass-engine-validation.lock` on engi. Two concurrent wrapper test
instances and a TMPDIR containing spaces, `&`, `@`, `$`, and a quote also pass.
The production reader pool is byte-identical to the original model/I/O-tested
`7f37f6bdc69a0286869ead4d0e8231fa601b23c1` file.

## Fresh checks

- **engi:** rebuild all five PLE test/measurement executables, run the complete
  `make -C kern test-ple` suite under the shared lock, and check engine C syntax.
  The suite covers all reader counts, 2,048 reused batches, row/guard parity,
  partial reads and errors, worker startup failures, reload preflight, sixteen
  persistent concurrent workers, delayed completion, benchmark failures,
  eighteen startup-warming checks, wrapper guards, and 34 CSV validation checks.
- **archy:** compile/link `libengine.so`, rebuild the five PLE executables, and
  pass the same CPU suite. The successful run used GCC 16.2.1 and the installed
  CUDA 12.9 headers under `/opt/cuda/include`; it is not a CUDA 11.4 rebuild or
  a GPU execution result. Compilation was serial and at low CPU/I/O priority.
- **Evidence replay:** all 48 historical checksums, I/O replay, and negative
  replay tests pass. Both original model JSON records still identify `7f37f6b`,
  contain 72 passing comparisons each, have successful child exits, and contain
  six passing reload records in total, including thread-count checks.
- **Reader sanitizers:** AddressSanitizer and UndefinedBehaviorSanitizer passed
  on the unchanged reader test/pool at `d4e8320`, before the shell-fixture fix.
  This is separately scoped from the clean `78eba07` builds above.

## Acceptance scope

| Requirement | Evidence |
| --- | --- |
| Bounded persistent readers; every row completes before publication | Unchanged production pool and engine integration; fresh reader/barrier tests on both hosts. |
| Compact buffers without pinning the full PLE file | Unchanged borrowed descriptor/row-buffer design; no table mapping or pinning added. |
| Startup/teardown/reload and real-model parity on archy | Original 144 comparisons and six reload checks, with source identities, commands, full-state results, and checksums retained in `../historical`; fresh target engine link and CPU lifetime tests. |
| Matched local/NFS I/O with cache classes | Historical replay verifies 4,608 batches, matching sampled bytes across filesystems, and 1,024 exact-input fully client-resident warm pairs per filesystem. Disjoint unprimed inputs remain descriptive distributions, not same-offset cold pairs. |
| Optional startup and standalone full-file warming | Merged PR #48; eighteen fresh control-flow checks per host; original whole-file duration and before/after residency records retained. |
| Conflict resolution and reviewer-requested reader retest | Tested source contains the merge of master; `ple_read_test` and the complete CPU suite pass on engi and archy. |
| No Python additions | The resumed code change is a Bash test fixture with inline Perl; no Python was written. |

## Lock and attempt accounting

`archy-check.log` records an earlier attempt that acquired the shared lock but
failed because `/opt/cuda-11.4/include/cuda.h` is not installed. Its exit is 2;
it is not passing validation. `archy-final.log` records a corrected attempt
that waited five minutes and exited 75 without submitting build/test work.
Another worker retained the GPU lock.

The successful `archy-cpu.log` / `archy-cpu.exit.json` run is explicitly CPU-only:
it neither acquired nor released another worker's GPU lock, made no CUDA
execution calls, loaded no model, and read no real shard. Its tests use small
temporary fixtures. The engi suite separately demonstrates operation while
its caller holds the shared lock. GPU/disk-heavy validation still uses the
unchanged production coordination lock.

No new GPU/model, real-shard I/O, or full-file warming experiment was run during
this resume. Those acceptance measurements retain their original revisions
and methods in `../historical`; no new performance or residency claim is made.
No resident core, holder, fence, or other worker was stopped or reloaded.

## Reproduction

From the repository root at the tested implementation commit:

```bash
make -C kern -B ple_read_test ple_read_bench ple_read_audit ple_read_bench_test ple_read_audit_test
flock /tmp/flash-sass-engine-validation.lock make -C kern test-ple
make -C kern dieops.h
cc -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -fsyntax-only \
  -Ikern -Iref -I/opt/cuda/include kern/engine.c
(cd validation/issue49/historical && sha256sum -c SHA256SUMS)
perl validation/issue49/replay_historical.pl validation/issue49/historical
bash validation/issue49/replay_historical_test.sh validation/issue49/historical
```

`target-cpu-check.sh` records the exact target-host CPU commands. Its arguments
are an isolated directory containing the tested `kern/` and `ref/` trees and
the full tested commit SHA. It never runs a GPU/model or real-file benchmark.
Command outcomes, binary hashes, and failure/timeout records are retained here;
`SHA256SUMS` covers every evidence file.
