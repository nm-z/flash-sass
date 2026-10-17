# Issue 18 validation

## Reproduce the current tests

The implementation is C and SASS; the orchestration is Bash, jq and the
repository's existing Perl build tools. The two historical Python runners were
removed. Historical JSON and transcripts keep their original source revisions
and test scope.

Build on the M60 host using its CUDA 11.4 toolchain and the repository's patched
MaxAs checkout:

```bash
make -B -C kern attn_test qsa_test qsa_direct_test \
    die.cubin attnpart.cubin attnmerge.cubin libengine_reuse_test.so
make -C ref test-attn-contract
make -C kern checked_test
./kern/checked_test
```

The parameterized runner requires explicit paths and a GPU UUID. Each output
directory must be new. It takes the specified advisory lock, records
`nvidia-smi` and `ps` before every GPU process, and waits while there is
active work. It never kills another worker's process. The named resident
weight holders and a sleeping flashcore may remain loaded; an active request,
another test or another compute owner blocks the next run.

```bash
bash validation/issue18/run.sh --list --suite padding
bash validation/issue18/run.sh --run --suite padding \
    --repo "$CANDIDATE" --out "$NEW_OUTPUT/padding" \
    --gpu "$GPU_UUID" --lock "$GPU_LOCK"
bash validation/issue18/run.sh --run --suite direct \
    --repo "$CANDIDATE" --out "$NEW_OUTPUT/direct" \
    --gpu "$GPU_UUID" --lock "$GPU_LOCK"
```

The padding plan contains 1536 cases: independent K-only and V-only NaN,
positive infinity, negative infinity and bounded finite values, both memory
placements, dense tails 1..33, quarter/tile/chunk boundaries, 2048..2055,
small-budget sequential QSA, production-budget capped boundaries, budget 4096,
and maximum visible position 262144 in capped attention. Direct compact
fixtures intentionally do not claim full-history indexer/gather coverage.
The 10 direct-QSA cases use this repository's complete driver and cover
34 visible positions, including the configured maximum, with synthetic history.

The `controls` and `timing` suites also require `--parent "$CONTROL"`.
Construct CONTROL from the same candidate source and test drivers, replacing
only `kern/attnpart.sass` and `kern/attnpartn.sass` with their unmasked
pre-fix versions. Rebuild both images with `make -B`; record the exact source
and cubin hashes. K-only controls can pass on unmasked kernels because the
later score mask overwrites their effect. V-only controls must expose the
unmasked IEEE multiplication failure. A deliberately corrupted allocation
guard must be rejected separately from arithmetic comparisons.

The timing suite performs five alternating candidate/control pairs for each
memory placement and for dense and capped attention, 200 repeated iterations
per process, at visible count 2051 and capped budget 512. These are component
timings; whole-model request timings below have a different scope.

## Actual engine acceptance

`kern/engine_reuse_test.c` is a validation-only library built from the actual
engine. It requires all 48 layers and the actual PLE table. A fresh reload
followed immediately by `reusecheck 96 64 16` performs 20 real model requests:
four short references, four longer requests, four identical short requests
after the longer request, four longer requests with a divergent suffix and a
snapshot, and four identical short requests resumed from that snapshot.

Snapshot positions 16, 17, 18 and 19 cover every prefix residue modulo four.
Ordinary reuse must report `resumed=0`; rollback must report the requested
snapshot. Each comparison checks all three emitted token IDs and every final
logit on all dies for bitwise equality and finiteness. It does not compare
every generation step's full logit buffer.

Run both `FLASH_KV_HOST=0` and `1`, with `FLASH_LAYERS=48`,
`FLASH_NBLK=512`, `FLASH_TWO_POSITION=0`, `FLASH_PREFILL_CHUNK=0` and
a sufficient `FLASH_ENGINE_MAXT` such as 256. Keep the core's original
configuration and library path, reserve its shared engine-validation lock,
and check GPU/process idleness before each request. Reload the validation
library with the candidate cubin, record the acknowledged config, then send
the fixture command on a fresh socket connection. A matching control run uses
the same library/configuration with only the unmasked cubin changed.
Always reload the original library/settings afterward and compare the full
config response with the recorded original.

A successful socket return is not acceptance: require the terminal
`reusecheck PASS`, eight expected comparison PASS lines, 20 completed run
lines, 21 nonzero allocation-guard PASS lines, correct snapshot/resume values
and no failure markers. Fresh allocation additionally requires a successful
reload acknowledgement before the fixture.

The guards surround every engine device allocation and every allocation made
through `cuMemHostAlloc`, including the outer mapped host KV arena.
They do not surround individual logical subarrays inside that shared arena,
core-owned weights/PLE, or polling arrays allocated with `cuMemAllocHost`.
The separate component guards and documented load derivation cover those
different boundary questions. Guard checks detect writes, not speculative
read extent.

The production engine reports each request's `run_ms` and `build_ms`
before guard checks and logit readback. Compare matched candidate/control
requests to measure masking cost. The 96/64-token fixture establishes these
reuse transitions; it is not a long-context end-to-end benchmark.

## Evidence policy

Raw outputs, exit codes, explicit environments, commands, source status and
source/binary/cubin hashes accompany each measured suite. A partially completed
plan cannot report a suite pass. Summary documents identify the implementation
commit tested; later evidence-only commits do not change those tested sources.
Historical `archy-final` logs describe the old matrix, whose combined-infinity
arms did not establish independent K/V coverage. They are retained as history,
not relabeled as a current-head result.
