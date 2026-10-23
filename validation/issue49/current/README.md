# Follow-up host validation

Tested source: `52bf9bc20239a2a91d8412a3e65c2bbe2fc51949`.
The source tree was clean. This directory adds evidence only; it does not
change the tested implementation.

## Completed on engi

- All five PLE test/benchmark binaries rebuilt from that commit.
- `make -C kern test-ple` passed: the reader pool, normal/audit benchmark
  failure regressions, 18 startup-warming checks, the shared-host wrapper,
  and 34 CSV validation checks.
- The engine C syntax check passed against the installed CUDA headers.
- All 48 historical artifact checksums and the historical replay's negative
  tests passed.
- The production reader-pool file is identical to the model/I/O-tested
  `7f37f6bdc69a0286869ead4d0e8231fa601b23c1` version.

Commands run from the repository root:

~~~bash
make -C kern -B ple_read_test ple_read_bench ple_read_audit ple_read_bench_test ple_read_audit_test
make -C kern test-ple
cc -std=gnu99 -O2 -ffp-contract=off -Wall -fPIC -fsyntax-only \
  -Ikern -Iref -I/opt/cuda/include kern/engine.c
bash validation/issue49/replay_historical_test.sh validation/issue49/historical
(cd validation/issue49/historical && sha256sum -c SHA256SUMS)
~~~

The metadata records the tested commit/tree, host, compiler, and successful
exits. Binary hashes refer to the five rebuilt host executables.

## Additional archy build attempts

An additional CPU-only engine-link/reader-test run was attempted twice in a
new private directory on archy. Both attempts exited 75 because another worker
held `/tmp/flash-sass-engine-validation.lock`. The second waited up to
45 seconds. Neither reached the build or test command. Their logs and exits
are retained; they are not successful target-host validation.

No new GPU or disk-heavy benchmark was run for this follow-up. The original
archy model parity, matched warm I/O, and full-file warming measurements
remain in [../historical](../historical), with their original source identities
and methods. The new host regressions do not relabel or replace those runs.
