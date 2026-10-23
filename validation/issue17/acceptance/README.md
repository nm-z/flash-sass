# Issue #17: direct-position QSA acceptance evidence

This is the completed M60 acceptance run for [issue #17](https://github.com/nm-z/flash-sass/issues/17), following [PR #41](https://github.com/nm-z/flash-sass/pull/41) and its [changed files](https://github.com/nm-z/flash-sass/pull/41/files).

## Result and source identity

**PASS: 59/59 cases, 298 positive positions, and three expected failure-injection positions.** The matrix exited **0**. Each positive position reports `EXACT`; the injected selection, gather and final-output mismatches each report `FAIL` and exit **3**, as required by the negative controls.

The tested source is **`cdc3118e9d1d641fd8fc2433f93bad6510c01216`**. This evidence was committed afterward without changing the tested code. The receipt fingerprints the repository source inputs, generated dispatcher sources, patched MaxAs sources, executable and cubin. The runner rechecks those hashes after waiting for the GPU and before each case.

| Item | Recorded value |
| --- | --- |
| Host / GPU | archy / Tesla M60, device 6 |
| GPU UUID | `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` |
| Driver | 470.256.02 |
| Build UTC | 2026-10-18 00:45:41–00:45:51 |
| Matrix UTC | 2026-10-18 00:49:40–00:56:03 |
| Source-input SHA-256 | `065ec944c48a6c671f2f80c75f79232b3d031ee8836bed3fca25244bf8ebec26` |
| Executable SHA-256 | `2c6c3add558514bf64b05ac2be7717adf9883e053137eee4edd1f318e422fa1e` |
| Cubin SHA-256 | `116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14` |

The build used GCC 16.2.1, CUDA headers/libraries from `/opt/cuda` (nvcc reports 12.9), and **ptxas 11.4.152** for the Maxwell cubin. MaxAs was copied into the isolated checkout from the host's existing working dependency: upstream commit `54eda7af086a46c9dae1688b691968235d560164`, with the exact local Perl changes preserved in [build2/maxas.patch](build2/maxas.patch). The patch SHA-256 is `b34d3f663cb1778eefd0feb0734c28e67c1f73c1bfed6ed6a368ebc59bab162e`. The original dependency checkout was not modified.

## What remained after PR #41

PR #41 delivered the complete persistent-dispatcher path, deterministic historical cache construction, intermediate comparisons, both cache placements and a broad long-position matrix. Its saved logs identified source `eee917c5`, before dispatcher changes included in the merged source. That run therefore did not validate the final dispatcher delivered by the PR.

Its alternating adjacent-score pattern also left both sides of rank 512 on the same score plateau. The follow-up places exactly the budget of blocks above the lower adjacent score and independently ranks actual GPU scores. Prefix replay now compares ring and pooled state immediately, before the current append can replace them. The matrix explicitly includes the 4096-to-4097 complete-block scoring chunk transition and every attention tile boundary through the production budget.

## Acceptance coverage

| Issue acceptance item | Evidence |
| --- | --- |
| Direct visible position, budget, fixed seed, placement and pattern controls | Each case in [matrix/results.json](matrix/results.json) records the exact executable arguments and QSA environment. Seeds are 4242, 0 and 77. |
| Complete current-position chain through the persistent dispatcher | All positive logs identify the original-input projection mode and compare projection outputs, normalized queries, current KV codes/scales, ring, completed pool, score bits, selected IDs, compact K/V, attention partials and merged output. The descriptor assertions cover all five selection passes. |
| Required long positions and tails | Both placements cover N=1..4, 2047..2058, 3100..3103, 8000..8003, 65000..65003, 262000..262003 and 262141..262144. All use production block budget 512. |
| Packed-pool, attention-tile and scoring-chunk transitions | Both placements cover 127..134, eight-position windows beginning at 255, 511, 767, 1023, 1279, 1535 and 1791, plus 16383..16391. |
| Real sparse exclusion and raw key counts | At N=2052, 513 complete blocks yield 512 selected blocks and 2048 keys. At N=262144, 65536 complete blocks yield 512 selected blocks and 2048 keys. Tail positions report 2049, 2050 or 2051 keys. The runner validates the arithmetic and exclusion flag on every row. |
| Distinguishable KV and oldest/newest/clustered/scattered selection | Four connected positions per pattern, per placement, at N=3101..3104. Hash fixtures retain varied attention values; partial maxima, denominators and weighted sums are compared before merge. |
| Equal and adjacent scores at the actual cutoff | The four cutoff records below come from independently sorting the actual GPU score readback and verifying selected membership. |
| Direct seed versus sequential prefix | [VRAM replay](matrix/018-n65-b8-s4242-h0-hash-replay.log) and [pinned-host replay](matrix/043-n65-b8-s4242-h1-hash-replay.log) both report `prefix history=64 kv=EXACT ring=EXACT pool=EXACT`, then pass all four current positions at budget 8. Existing sequential test modes are unchanged. |
| Future slots, guards and nonzero failures | Every position checks finite future-slot sentinels through expected gathered values and all 36 guarded allocations. The three injected faults each exit 3. |
| Reproducible commands, source/cubin/GPU identity and raw logs | [Build receipt](build2/build.json), [build stdout](build2/build.stdout.log), [build stderr](build2/build.stderr.log), [matrix manifest](matrix/results.json), all 59 raw logs, and every attempted `nvidia-smi`/`ps` preflight are retained. |

The pool-pattern overrides are explicitly synthetic adversarial fixtures. The normal hash fixture uses consistent historical raw KV, pooled keys and ring state. Correctness timings here are command wall durations, not performance measurements; this evidence makes no whole-model or checkpoint-quality claim.

### Actual cutoff records

These results were identical for VRAM and pinned-host KV:

| Pattern | Last selected ID / score bits | First excluded ID / score bits | Assertion |
| --- | --- | --- | --- |
| equal | 511 / `4007df26` | 512 / `4007df26` | Equal scores; lower ID wins |
| adjacent | 774 / `4007df26` | 0 / `4007df25` | Positive scores differ by exactly one FP32 bit |

Raw logs: [VRAM equal](matrix/023-n3101-b512-s4242-h0-equal.log), [VRAM adjacent](matrix/024-n3101-b512-s4242-h0-adjacent.log), [pinned equal](matrix/048-n3101-b512-s4242-h1-equal.log), [pinned adjacent](matrix/049-n3101-b512-s4242-h1-adjacent.log).

## Exact successful commands

These commands ran on **archy**, reached by SSH from engi. The paths in the receipts are the original execution paths and intentionally remain unchanged in this copied evidence.

```bash
cd /home/nate/codex/flash-issue17-acceptance-20261017
perl kern/build_qsa_direct.pl --out /home/nate/codex/flash-issue17-results-20261017/build2
CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b perl kern/run_qsa_direct_matrix.pl --build-manifest /home/nate/codex/flash-issue17-results-20261017/build2/build.json --gpu-lock /home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock --out /home/nate/codex/flash-issue17-results-20261017/matrix
```

Build wrapper exit **0**; matrix exit **0** (also saved in [matrix.exit](matrix.exit)). The wrapper's exact forced rebuild command, preserved in its receipt, was:

```bash
make -B -C /home/nate/codex/flash-issue17-acceptance-20261017/kern CC=/usr/bin/gcc CUDA=/opt/cuda FLASH_REV=cdc3118e9d1d641fd8fc2433f93bad6510c01216 PTXAS11=/opt/cuda-11.4/bin/ptxas qsa_direct_test die.cubin
```

The runner acquired the shared per-UUID reservation and waited when it was held by another worker. Before every GPU case it saved both `nvidia-smi` device/compute-process output and `ps -eo pid,ppid,etimes,stat,comm`. It also waited for utilization to return to zero between cases. No other worker's process was signaled or killed.

## Supporting host-only checks

```bash
perl -c kern/build_qsa_direct.pl
perl -c kern/run_qsa_direct_matrix.pl
perl kern/QsaEvidence_test.pl
```

Both Perl syntax checks exited **0**. The existing evidence helper passed **10/10** tests, exit **0**; [helper.tap](helper.tap) preserves the output.

Two copied build receipts were deliberately changed without altering the executable or source: one replaced the executable hash with 64 zeroes, and one replaced the recorded hash for `kern/die.pl` with 64 zeroes. The following commands each exited **1** before creating an output directory or starting GPU work:

```bash
CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b timeout 10 perl kern/run_qsa_direct_matrix.pl --build-manifest /home/nate/codex/flash-issue17-results-20261017/stale-binary.json --gpu-lock /home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock --out /home/nate/codex/flash-issue17-results-20261017/invalid-binary
CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b timeout 10 perl kern/run_qsa_direct_matrix.pl --build-manifest /home/nate/codex/flash-issue17-results-20261017/stale-source.json --gpu-lock /home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock --out /home/nate/codex/flash-issue17-results-20261017/invalid-source
```

The diagnostics are [`binary changed after build`](stale-binary.log) and [`source changed after build: kern/die.pl`](stale-source.log). These are receipt-integrity negative controls, separate from the three GPU fault injections.

All authored implementation changes are C and Perl. No Python was added.
