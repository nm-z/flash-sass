# Issue #12: attention CTA remap validation

PR #29 is open and non-draft at the user's request. This state change does not
establish that the remaining target-GPU and whole-model acceptance checks passed.
The original component results below are retained
as historical evidence; they do not validate the later trace or canary changes.

The [2026-10-18 review checkpoint](review-20261018/README.md) records six successful
isolated builds, the current CPU checks, and the latest shared-GPU lock blocker.
It contains no new target-GPU measurement.

## Historical component measurement

The 48 JSON cases in this directory compare exact parent
`c6d3de8180d8b87caf0d2f61fa52edc9247daded` with remap
`66725432f86cc193b75cd51230216aed44c60d3f`. Their original manifest identifies
harness `4aa4fc2aa7477e8491e50fb2094e8beff4626d71`, toolchain, GPU, cubins,
libraries, fixed seeds, and raw samples.

At 2,051 keys the historical median across three seeds was 201.451 us before
and 132.434 us after for the partial-plus-merge component. That is a measured
component result, not a whole-model speedup. The old `status.json` reports
those 48 historical cases only.

The earlier phrase “output guards” described NaN initialization inside the
output arrays. Those runs had no boundary canaries. They therefore establish
neither writes staying within the allocation nor correctness of the new guards.

## Current validation tools

- `kern/attnpartn_regress.c` compares all partial maxima, denominators,
  weighted-value partials, and final outputs through the persistent dispatcher.
  Leading and trailing canaries surround each output allocation. Output-content
  checks and boundary checks have separate meanings.
- `kern/cta_remap_matrix.pl` replaces the Python runner. Its shared lock paths
  are configurable. It records GPU/process checks, refuses an occupied host,
  retains raw command results, and validates the harness's required coverage.
- `DIE_CTA_TRACE=1` builds physical-CTA instrumentation in `kern/die.pl`.
  Untraced builds retain the current production dispatcher's instruction stream.
  The trace hooks follow the descriptor reload in the paired dispatcher; tracing
  records scalar decode passes only and leaves paired-prefix records unwritten.
- `kern/cta_remap_bench.c` builds the test-only
  `libengine_cta_remap.so`. It includes the production engine and compares
  mapping variants in a validator-owned six-die core.
- `kern/cta_remap_summary.pl` summarizes a complete whole-model output
  directory, including both warmup pairs. It validates every raw trace record's
  identity, opcode, work count, nonce, and timestamp order, then derives all
  device timing terms from the CSV and checks the reported JSON against them.
  Missing samples, output differences, invalid callback timing, incomplete or
  replayed traces, missing dies, and inconsistent timing terms fail validation.

All new code uses C, Perl, or Bash. The old Python runner is removed from the PR.

## Trace contract

The control block holds the trace pointer at `0xd0`, first token at `0xd8`,
token capacity at `0xdc`, physical CTA count (16) at `0xe0`, and nonzero launch nonce
at `0xe4`. Records are ordered by token, phase, and physical CTA.

Each 64-byte record contains five unsigned 64-bit GPU timestamps, followed by
six unsigned 32-bit fields:

| Offset | Field |
| --- | --- |
| 0 | Exchange-wait interval start, after descriptor load |
| 8 | Exchange-wait interval end, before opcode dispatch |
| 16 | Body start, after dispatch reaches the selected operation |
| 24 | Body end, after the local CTA drain barrier |
| 32 | Phase end, after the grid barrier |
| 40 | Absolute token position |
| 44 | Phase index |
| 48 | Physical CTA index |
| 52 | Opcode |
| 56 | Number of nonempty ATTNPARTN virtual units actually entered |
| 60 | Launch nonce, written by the kernel |

The work counter increments on the executed nonempty attention path, once per
virtual unit. Other opcodes report zero for that counter. Empty attention units
still execute setup and output stores; “zero nonempty units” does not mean the
CTA executed no instructions.

The harness clears the buffer before every traced launch, supplies a fresh nonce,
checks every expected record's identity and work count, and rejects timestamps
from an earlier launch. An uninstrumented image cannot satisfy this contract.
Ordered nonzero timestamps alone are insufficient.

The phase intervals distinguish exchange waiting, opcode dispatch, operation
body, and grid waiting. A uniform trace barrier holds all warps until body start
has been recorded. The body includes that barrier, operation-internal waits, and
the local CTA drain barrier. Times between consecutive records account for descriptor and
token transitions plus trace bookkeeping; they are retained as a separate term.

## Reproducible builds

From the branch checkout, on a host with the Maxwell-capable toolchain:

```bash
PR29_BUILD=/tmp/pr29-build
PTXAS11=/opt/cuda-11.4/bin/ptxas CUDA=/opt/cuda \
    MAXAS_ROOT=/path/to/patched/maxas bash kern/cta_remap_build.sh "$PR29_BUILD"
```

The output directory must be new. The builder makes six isolated source trees:

| Tree | Source and purpose |
| --- | --- |
| `historical-old` | Exact parent commit for the original component comparison |
| `historical-new` | Exact remap commit for the original component comparison |
| `matched-old` | Current engine/kernel with only the attention mapping reverted |
| `matched-new` | Current engine/kernel with the committed mapping |
| `trace-old` | The matched old mapping with CTA tracing |
| `trace-new` | The matched new mapping with identical CTA tracing |

The mapping patch, compiler/assembler versions, build transcripts, generated
source/cubin/library hashes, and commit identities remain in the build output.
The builder snapshots MaxAs's `bin` and `lib`, records their hashes and upstream
commit/local patch, and uses that same snapshot for every variant.
The builder requires byte-identical old/new engine libraries within each matched
pair. Historical cubins must not be loaded into the newer disk-PLE engine; the
compatible matched pair isolates the CTA mapping while keeping that protocol
and all other current operations fixed.

Before GPU execution, run `LD_BIND_NOW=1 attnpartn_regress --self-test` against
the target host's real CUDA driver library. Eager symbol binding checks driver
ABI compatibility even though the self-test does not initialize CUDA. A CPU
self-test linked to SDK stubs alone cannot establish target-driver compatibility.

The component runner accepts explicit paths to both cubins, both libraries, the
harness, its full source commit, the GPU UUID, and a new evidence directory:

```bash
perl kern/cta_remap_matrix.pl \
    --old "$PR29_BUILD/historical-old/kern/die.cubin" \
    --new "$PR29_BUILD/historical-new/kern/die.cubin" \
    --old-library "$PR29_BUILD/historical-old/kern/libengine.so" \
    --new-library "$PR29_BUILD/historical-new/kern/libengine.so" \
    --harness "$PR29_BUILD/matched-new/kern/attnpartn_regress" \
    --harness-source "$(cat "$PR29_BUILD/matched-new/source-commit.txt")" \
    --gpu "$PR29_GPU_UUID" --out "$PR29_RESULTS" --wait-seconds 180
```

For physical-CTA attribution, use the `trace-old` and `trace-new` cubins/libraries,
set both `--old-source` and `--new-source` to the build's current source commit,
and add `--trace --instrumentation DIE_CTA_TRACE=1`. The historical commit IDs are
the untraced runner defaults. Keep traced and untraced timings in separate
directories. Raw `TRACE` lines from the four ABBA correctness launches are
retained in each case's stderr; every timed launch's trace is validated too.

The shared host lock defaults to `/tmp/flash-sass-engine-validation.lock`;
GPU locks default to `/tmp/flash-sass-gpu-locks`. `--host-lock`/`--lock-dir` or
`FLASH_GPU_HOST_LOCK`/`FLASH_GPU_LOCK_DIR` can override them. Use the same host
reservation as the other workers. `--check-only --gpu UUID --out NEW_DIRECTORY`
records availability without starting the harness. Busy waits end with exit 75
and `BLOCKED_BUSY`; failed inventory queries fail closed.

Lock contention now uses a bounded blocking `flock`, so the validator is actually
queued for the reservation rather than repeatedly polling with `LOCK_NB`.
It never deletes a lock file, takes another worker's reservation, or kills a
holder. The default availability policy remains strict.

For a component experiment on an **unused, separate GPU**, an explicit option
can distinguish idle weight residency from active GPU work:

```bash
# Add to the component matrix command; --gpu must be a UUID in this mode.
--idle-resident-queue http://127.0.0.1:18090/queue
```

This mode still acquires both shared locks. It requires zero compute and memory
utilization on every GPU, zero running/waiting HTTP requests, no compute context
on the target GPU, and matching process identities for off-device `hold`,
`fence`, and `flashcore` contexts. Only the `flashd` serving that exact loopback
queue is exempted from the worker-process check. Unknown workers, incomplete or
failed inventory, target residency, or active requests fail closed. Raw identity
and utilization evidence is retained before and after each case; detected
post-run contention invalidates that case. These are sampled boundary checks,
not a guarantee against an unrelated client arriving between samples.

This option does **not** authorize reloading the resident core, sharing a device
with it, or treating component timing as a matched whole-model measurement.

## Whole-model procedure and timing definitions

The benchmark library must run in a core and weight-holder group owned by the
validation job. Acquire the shared host reservation and check `nvidia-smi` and
`ps` before starting GPU work. If another holder, fence, server, or validation
process is present, wait. Do not stop or reload another worker's process.

The builder also produces a matching `flashcore` and `hold` in `matched-new/kern`.
Use `FLASH_CUBIN` to point the validation core at the compatible untraced image
when it starts. After initializing that core with real PLE data, its own six
weight manifests, the test library, and enough context capacity, send this command
on its stdin. Paths must be absolute and contain no whitespace:

```text
ctabench OUT_DIR MATCHED_OLD_CUBIN MATCHED_NEW_CUBIN TRACE_OLD_CUBIN TRACE_NEW_CUBIN 2048 32 8
```

The command requires six dies, all 48 layers, production `NBLK=512`, real PLE
data, at least two decode intervals, and an even number of measured pairs.
Use identical split, attention assignment, KV placement, image capacity, and
context capacity throughout. The configuration also records the paired-position
and chunked-prefill settings. Save the core/holder launch commands, checkpoint
identity, and clock/temperature/load telemetry with the output.

The fixed prompt consists of token 32 repeated 2,048 times. The benchmark first
creates a snapshot just before its last position. Every measured response
restores the same snapshot and must produce exactly the same generated token
IDs. This is a fixed-input timing and parity experiment, not a quality benchmark.

Each response produces 33 callbacks for 32 decode intervals. The host decode
time is the last callback time minus the first callback time, divided by 32.
It excludes prefill, snapshot construction, program construction, snapshot
restoration, and the first generated-token interval. Individual callback times
and token IDs are retained.

Two warmup pairs precede the measured pairs. Old/new order and trace mode order
alternate, with the same allocations and construction capacities in all four
cells. This yields a mapping comparison and a matched trace-overhead comparison.

Traced responses capture the final two decoded positions on all physical CTAs
of all six dies. For each die, the last position's CTA-0 final-phase timestamp
minus the preceding position's final-phase timestamp gives its device decode
interval. Attention, gather, indexer, other bodies, exchange wait, dispatch,
grid wait, and transitions reconcile to that interval. This is the final token
of each response; the host metric averages all requested decode intervals.
The raw CSV contains the
other CTAs' work and wait times as well. Times from different dies are never
subtracted or summed into one serial token latency.

After a successful complete run:

```bash
perl kern/cta_remap_summary.pl OUT_DIR
```

Run the summary from the same source checkout used to build the benchmark. It
obtains the descriptor opcode names from that checkout's `die.pl header` output
and checks that every sample uses the same phase plan. Trace freshness is checked
in launch-nonce order separately for each die.

The summary reports distributions, paired old/new changes, instrumentation
overhead, and per-die phase contributions. Its inputs are the saved raw sample
JSON and trace files. Keep those files with the summary.

## Remaining acceptance

1. Rerun the component matrix with the boundary canaries on M60.
2. Run the physical-CTA traces and the rejection test using an uninstrumented
   image; verify the observed work distribution for both mappings.
3. Run and retain the matched six-die whole-model experiment, with the telemetry
   and raw files described above.
4. Replace the estimated whole-model benefit with that measured result, including
   a regression or no material gain if that is what the measurements show.

CPU compilation, assembler success, and synthetic rejection tests do not close
these GPU acceptance items.
