# Issue #12: completed GPU component validation

The shared issue #12 validator completed **144 component cases and the actual
uninstrumented-image rejection test** on archy. This follow-up independently
checks the retained results; it did not launch those 144 experiments again.
**The separate matched six-die whole-model experiment remains outstanding.**

PR #29 is open and non-draft at the owner's request. That state is not a claim
that the outstanding whole-model acceptance requirements have passed.

## Experiment and audit

The GPU run began at `2026-10-18T06:00:57Z` and completed successfully before the
`06:09:51Z` status check. It used native harness source
`6ae0a092f2a5a0f6a472066152bcdd12daef8303`, including the legacy CUDA event-timing
ABI fix. This tested commit is preserved in the PR's ancestry. The current
`attnpartn_regress.c`, `cta_trace.h`, `cta_remap_bench.c`, and `die.pl` were
byte-compared with the tested build inputs and matched.

Each of the historical, matched-current, and traced mapping pairs ran 48 cases:
16 key counts, seeds 0/77/4242, and 32 measured repetitions per arm. Key counts
were 1, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051, 2052, 4095, 4096, 4097,
8191, 8192, and 8193. The persistent harness checks finite-input partial maxima,
denominators, weighted-value partials, final outputs, and allocation canaries.

The independent CPU-only auditor passed:

- All 144 case identities, successful exits, exact-output/canary coverage, and
  27,648 retained event samples, including recomputed medians and p95 values.
- All 12,288 saved raw CTA records: identity, ABBA mapping order, launch nonce,
  operation code, physical work counts, timestamp order, and timing terms.
- All 656 telemetry samples against the shared runner's declared monitoring
  contract, plus the required negative test's exit and trace-rejection message.

The auditor itself rejected deliberately changed timing statistics, a wrong
trace nonce, and off-device GPU activity in separate copied fixtures. The
original evidence was not modified. There were also 47 successful checks of
recorded native build/source/binary and runner/harness hashes.

## Measured component result

These are partial-plus-merge GPU-event medians, first computed within each
seed's 32 samples and then taking the median across three seeds. Trace-mode
measurements remain separate and are not used as uninstrumented performance.

| Pair | Compact keys | Old, microseconds | New, microseconds | Latency reduction |
| --- | ---: | ---: | ---: | ---: |
| Exact historical parent/remap | 2,048 | 245.634 | 141.746 | 42.294% |
| Exact historical parent/remap | 2,051 | 249.436 | 154.740 | 37.964% |
| Matched current engine | 2,048 | 246.782 | 142.328 | 42.326% |
| Matched current engine | 2,051 | 249.834 | 155.436 | 37.784% |
| Matched current engine | 8,192 | 339.221 | 333.911 | 1.565% |

Observed physical work matches the intended remap. At 2,048 keys, eight nonempty
units use four CTAs with two units each before, versus eight CTAs with one unit
each after. At 2,051 keys, twelve units use six CTAs with two units each before,
versus twelve CTAs with one unit each after. All physical-CTA records, including
those with no nonempty work, remain in the raw traces.

The measured GPU was Tesla M60
`GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4`, driver `470.256.02`.
Across retained samples its temperature ranged from 33 to 55 C, SM clocks from
405 to 873 MHz, and memory clocks from 324 to 2505 MHz. Those ranges include idle
samples; they do not assert a fixed clock for every timed event. Other GPUs
reported zero utilization in all 656 saved snapshots. Their idle weight-holder
and core allocations remained resident. One-second sampling cannot rule out
sub-second interference.

These results establish a component gain, not a model-level token-latency gain.
Do not multiply the component delta by a layer count or substitute it for the
outstanding matched whole-model measurement. The earlier historical evidence in
the parent directory remains unchanged for comparison.

## Runner and driver fixes

The original SDK-built harness fails eager binding on the target driver with
`undefined symbol: cuEventElapsedTime_v2`. The corrected native harness binds
`cuEventElapsedTime` and passes the CPU-only self-test with `LD_BIND_NOW=1`.
The matrix runner now performs and retains that preflight before its direct
reservation wait. The recorded negative preflight failed with zero GPU launches.

The runner also supports a bounded blocking lease wait, an explicit unused-GPU
mode for verified idle off-device residents, and safe adoption of `gpuq`'s
already-held descriptor (`--host-lock-fd 9`). Descriptor device/inode and the
exclusive flock are verified; this is not a lock bypass.

The first idle-residency implementation could falsely reject its own recently
completed work because target utilization remained nonzero. In the completed
shared dataset, **27 of 145 after-case snapshots** still reported target
utilization. The corrected post-run check preserves the immediate telemetry and
allows bounded target-counter settling, while continuing to reject unexpected
target compute apps, other-device activity, active requests, and failed queries.
Its **61 CPU self-tests passed**, including counter settling, lock inheritance,
real lock timeout/release, and driver-preflight rejection tests.

A three-case native-runner attempt passed the real-driver preflight and adopted
the `gpuq` reservation, but then exited 75 before any GPU launch: three resident
`llama-server` processes were classified as GPU workers from their names alone.
Their inspected command lines explicitly selected zero GPU layers, and none
appeared in the GPU compute-app inventory. The retained
`runner-initial-smoke/` files record that unsuccessful attempt, not a GPU pass.
The opt-in idle-resident policy now separately records such zero-offload
processes rather than treating their names alone as GPU work; conflicting flags,
reported GPU contexts, and other-device activity still reject. This classification
has CPU regression coverage but is not claimed as another completed target-GPU
matrix. CPU workload is not reserved or isolated by this component gate.

The shared 144-case run used the separately retained Bash `component-run.sh`,
not the subsequently revised Perl runner. Its successful result is not presented
as an end-to-end GPU result for that newer runner.

## Reproduction and files

From this directory, verify and unpack the archive into a new directory, then
run the CPU-only audit:

```bash
sha256sum -c raw-component-evidence.sha256
mkdir raw
tar -xzf raw-component-evidence.tar.gz -C raw
nu --no-config-file validation/issue-12/gpu-component-20261018/audit-component.nu \
    raw/component-matrix /tmp/cta-remap-audit.json
```

The archive contains every raw case JSON, trace, telemetry sample, exit table,
child identity, the actual shared runner, native source inputs, and build
metadata. `audit.json` contains all key-count summaries and observed CTA work.
`build-artifact-check.log`, `native-toolchain.txt`, `native-build-status.txt`,
and the preflight and self-test logs retain the additional checks.

## Remaining issue #12 acceptance

The exact builds, updated GPU component matrix, physical-CTA traces, and real
uninstrumented-image rejection are now evidenced. Still required are matched
whole-model decode with prefill excluded, attention/gather/indexer/synchronization
and total token timing, and a measured model-level improvement, regression, or
lack of material change. Issue #12 remains open until that evidence is retained.
