# Target-GPU component validation — 2026-10-18

**All 144 component cases passed on archy. The six-die whole-model measurement
has not run: the resident allocations leave insufficient VRAM for a private
validation core.** The PR's review state does not establish whole-model acceptance.

## Results and scope

The measured runtime is commit
[6ae0a092f2a5a0f6a472066152bcdd12daef8303](https://github.com/nm-z/flash-sass/commit/6ae0a092f2a5a0f6a472066152bcdd12daef8303).
It includes master 3d57d734d63aa74f167a6b5a02c616ddc018d2ff. The subsequent
423b158/c530354 changes and their merges affect the matrix runner and documentation,
not the C/SASS runtime used here. This records a specific source/toolchain/GPU
experiment; it does not claim that later changes to master were executed.

The matrix compared three pairs:

| Mode | Old image | New image |
| --- | --- | --- |
| Historical | Exact parent c6d3de8180d8b87caf0d2f61fa52edc9247daded | Exact remap 66725432f86cc193b75cd51230216aed44c60d3f |
| Matched untraced | Runtime 6ae0a09 with only the CTA mapping reversed | Runtime 6ae0a09 with the committed mapping |
| Matched traced | Same old mapping, DIE_CTA_TRACE=1 | Same new mapping, DIE_CTA_TRACE=1 |

All six isolated native builds passed. Old/new host libraries were byte-identical
within each matched pair. Source, generated SASS, cubin, library, harness, compiler,
assembler and patched MaxAs identities are retained in the build records.

Coverage was 16 key counts × 3 seeds × 3 modes = **144 cases**. The keys were
1, 1023, 1024, 1025, 2047, 2048, 2049, 2050, 2051, 2052, 4095, 4096, 4097,
8191, 8192 and 8193. Seeds were 0, 77 and 4242; seed 0 supplies equal scores,
and the other seeds supply nonuniform finite inputs.

The actual persistent dispatcher produced bit-identical partial maxima,
denominators, weighted-value partials and final outputs against the declared
reference/reduction contract and between mappings. Leading/trailing boundary
canaries were checked on every positive launch.

- **21,312 guard-checked launches passed.**
- **344,064 trace records were validated by the harness**, including timed launches.
- **12,288 raw trace records are retained** from the initial ABBA correctness
  launches. Timed-launch traces were checked in memory but are not all retained.
- Requiring tracing from the actual uninstrumented matched cubins failed with
  exit 1 and a specific missing launch-nonce rejection. A loader failure or
  arbitrary nonzero exit would not satisfy this negative check.

The independent readback recomputes sample distributions, verifies all retained
raw work/timestamp intervals, and checks telemetry. It passed the complete
matrix and rejected two deliberately corrupted copies: an altered raw body
interval and an altered reported median.
The integrated general matrix runner also passed all **41 CPU checks**,
including driver-ABI preflight, queued locking, resident identity checks and
failure handling; its transcript is retained in matrix-self-test.log.

## Measured component latency

Each case has four warmup pairs followed by 32 balanced old/new measured pairs.
A CUDA event sample covers 64 fixed-input attention-plus-merge positions through
the persistent dispatcher and is divided by 64. These times include that
component program's dispatch and synchronization overhead.

The following are medians over the 96 retained samples per arm
(three seeds × 32 samples), in microseconds per position. The raw files preserve
seed/run grouping; pooled samples are not claimed to be independent replicates.

| Keys | Matched old | Matched new | Latency change | New faster pairs |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.783 | 57.458 | -38.73% | 96/96 |
| 1024 | 240.797 | 130.219 | -45.92% | 96/96 |
| 2048 | 246.856 | 142.325 | -42.34% | 96/96 |
| 2049 | 249.932 | 155.244 | -37.89% | 96/96 |
| 2050 | 249.752 | 155.218 | -37.85% | 96/96 |
| 2051 | 249.813 | 155.400 | -37.79% | 96/96 |
| 4096 | 262.421 | 176.713 | -32.66% | 96/96 |
| 8192 | 339.115 | 333.846 | -1.55% | 96/96 |
| 8193 | 512.222 | 368.793 | -28.00% | 96/96 |

At 2051 keys, the matched old/new p95 values were 250.236 and 155.727 us.
The paired-difference median was -94.433 us, with a range of -95.054 to
-93.815 us. This differs slightly from subtracting the two marginal medians,
which gives -94.413 us.

The new historical-pair run measured 249.3905 → 154.7085 us at 2051 keys.
The earlier historical result, 201.451 → 132.434 us, remains in the parent
directory. It came from a different clock session and an earlier harness without
boundary canaries. Its samples are not pooled with this run.

The matched traced median at 2051 keys was 268.0965 → 171.3835 us. Mode batches
ran sequentially, so their traced/untraced difference is descriptive and is not
a balanced whole-model instrumentation-overhead estimate. The complete summary
also retains attention-only program timing and CTA0 merge-phase-end timing;
these have their own definitions and should not be added to the event interval.

**These are component measurements. No whole-model milliseconds/token saving
or tokens/second improvement is inferred from them.**

## Physical CTA work and waiting

The retained records count nonempty units on the executed attention path.

| Keys | Old physical distribution | New physical distribution |
| ---: | --- | --- |
| 2048 | CTAs 0, 1, 8, 9 each execute two nonempty units | CTAs 0–7 each execute one |
| 2049–2051 | CTAs 0, 1, 2, 8, 9, 10 each execute two | CTAs 0–11 each execute one |

Other CTAs have zero nonempty units, while still executing setup and empty-output
stores. The raw data includes all 16 CTAs' body and grid-wait intervals, and the
summary retains per-CTA distributions. Those retained intervals come from the
initial ABBA correctness launches; **they are not warmed production CTA latency
distributions**. Warmed event timing and trace-validation counts are reported
separately above.

## Target driver compatibility

The first attempt with the engi-built binary exposed a real target ABI failure:
CUDA 13 headers redirected event timing to cuEventElapsedTime_v2, which driver
470 does not export. Its lazy symbol failure happened after initial kernel work;
that failed attempt is not accepted as a measurement.

Commit 6ae0a09 explicitly binds the harness to the original cuEventElapsedTime
driver entry point. Both the newer-SDK build and the native archy build passed
LD_BIND_NOW=1 --self-test against archy's actual driver: 16 negative canary cases,
14 negative trace cases and 64 mapping cases.

The successful six native builds used **CUDA 12.9 headers/libraries** in
/opt/cuda and the **CUDA 11.4.152 Maxwell assembler** at
/opt/cuda-11.4/bin/ptxas, with GCC 16.2.1 and the frozen patched MaxAs snapshot.
Driver version was 470.256.02. The ABI diagnostic, failed-attempt stderr and
successful eager-binding outputs are retained.

## Reservation and telemetry

The full run held /tmp/flash-sass-engine-validation.lock continuously and also
held the GPU UUID lock. It used only GPU
GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4, a Tesla M60, through CUDA_VISIBLE_DEVICES.
The harness created only its own context, buffers and modules. It did not import
model weights, contact the resident engine, or stop/reload another process.

There are **656 telemetry snapshots**, from 2026-10-18 06:00:57 UTC through
06:09:27 UTC. GPU 7's active SM clocks were 872–873 MHz; memory clock was
2505 MHz during the active workload. Temperature over all snapshots was
33–55 C. Every sampled off-device GPU utilization was zero. The target GPU's
compute-process inventory permitted only the currently owned harness PID.

These are one-second and case-boundary samples. They can miss subsecond
interference from a process that does not respect the shared reservation.
The raw inventories, time stamps, child PIDs and exit codes are retained.

## Whole-model capacity blocker

The read-only capacity snapshot is in [whole-model-memory.csv](whole-model-memory.csv).
GPUs 0–5 were idle but retained the existing model/core allocations:

| Die | Free MiB | Private recurrent state + snapshot alone, MiB |
| ---: | ---: | ---: |
| 0 | 52 | 36 |
| 1 | 16 | 36 |
| 2 | 77 | 31.5 |
| 3 | 7 | 40.5 |
| 4 | 3 | 36 |
| 5 | 65 | 36 |

The stated split, 112,97,81,116,97,97, assigns 8,8,7,9,8,8 value heads.
The lower bound follows directly from engine_init's 36 recurrent layers:

    36 layers × local_heads × 16384 floats × 4 bytes × 2 copies

Dies 1, 3 and 4 cannot fit even these state buffers. Contexts, KV, compact caches,
programs, tracing buffers and scratch require additional memory. Keeping the
resident image capacity of 4096 would add another 20 MiB per die. Removing that
image allocation would not remove the state-memory blocker.

A private six-die core can use the existing holders as weight-only CUDA IPC
imports: hold_open reads manifests and opens handles; hold_close closes only the
private imports. The reviewed engine allocates its mutable state privately.
The intended configuration uses real disk PLE with offset 192, KV_HOST=0,
all 48 layers, NBLK=512, the existing split and ATTN_MAP, scalar positions,
and capacity 2080 for a 2048-token prompt plus 32 decode intervals.
FLASH_SOCK and FLASH_PLE_CONSUME must be unset for the private stdio core.

**No private whole-model core was started, and no whole-model sample exists.**
This validation job did not stop or reload the resident core, holders or fence.
The initial snapshot recorded resident core PID 3575268. Later checks observed
replacement resident cores, and briefly enough free memory for a private core.

The private benchmark acquired the shared reservation at **06:38:02 UTC**, with
lock-owner shell PID 1244313. Before its launch, another resident core, PID
1249426, appeared and used GPUs 0–5 at 100% utilization, again leaving only
3–77 MiB free. At **06:44:04 UTC**, the private runner returned **exit 75 /
BLOCKED_ACTIVITY_OR_MEMORY before starting flashcore or launching GPU work**.
Its reservation was released at **06:45:24 UTC**. This demonstrates why a held
cooperative lock is not itself proof of an uncontended production device.

[whole-model-attempt.tar.gz](whole-model-attempt.tar.gz) retains that attempt's
preflight inventories, exit/status files and exact runner. The prepared
[run-whole-model.nu](run-whole-model.nu) uses a private stdio core, imports the
existing weights, and sets image capacity to zero for this text-only experiment
to fit the tightest die. All four benchmark cells use that same configuration.
The GPU launch path has not yet run and is not reported as validated. The runner
monitors process activity and terminates only its own child on failure.

The user's earlier prohibition on stopping/reloading another worker's core
prevents freeing those allocations without authorization. A sufficiently long
six-die reservation, with the resident core kept off those GPUs, is still needed
to complete the matched decode experiment and measured whole-model conclusion.

## Reproduce and inspect

[run-components.nu](run-components.nu) is the ported isolated runner.
It requires the caller's inherited FD 9 to hold the shared host reservation.
It monitors GPU/process inventory, validates all cases, and executes the specific
uninstrumented-image rejection check. It is retained as the run procedure,
alongside the general matrix runner in kern.

After producing the six builds with kern/cta_remap_build.nu and passing the
target-driver eager-binding check:

~~~bash
ssh archy gpuq run -- nu --no-config-file \
    validation/issue-12/gpu-20261018/run-components.nu \
    "$PR29_BUILD" "$PR29_NEW_RESULTS" "$PR29_GPU_UUID"
~~~

[component-raw.tar.gz](component-raw.tar.gz) contains every case JSON/stderr,
telemetry snapshot, exit/PID record, the exact run script and build hash files.
[build-records.tar.gz](build-records.tar.gz) contains the native build logs,
toolchain/MaxAs identities and patch, target ABI checks, three smoke cases and
the capacity/process snapshot. No CUDA IPC handle contents are published.

To recompute the checked summary from the raw archive:

~~~bash
mkdir /tmp/pr29-component-readback
tar -xzf validation/issue-12/gpu-20261018/component-raw.tar.gz \
    -C /tmp/pr29-component-readback
nu --no-config-file validation/issue-12/gpu-20261018/summarize-components.nu \
    /tmp/pr29-component-readback/component-matrix > /tmp/pr29-component-summary.json
~~~

[component-summary.json](component-summary.json) contains all 16 key counts,
three modes, measured distributions, paired differences, raw physical CTA
summaries and telemetry statistics. [SHA256SUMS](SHA256SUMS) identifies the
retained artifacts.
