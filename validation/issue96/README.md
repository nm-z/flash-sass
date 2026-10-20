# Issue #96: mandatory GPU Flash Attention and full resolution images

[Issue #96](https://github.com/nm-z/flash-sass/issues/96),
[PR #98](https://github.com/nm-z/flash-sass/pull/98), and the
[required GPU/FA review](https://github.com/nm-z/flash-sass/pull/98#issuecomment-6285100011).

## Implementation

The service requires a GPU and GPU Flash Attention. CPU mode, CPU retry,
standard attention and memory-based resolution reduction are removed. Errors
are reported before publishing or caching a successful embedding response.
Every actual compute node's buffer is checked against the selected GPU, and
every FA node must match an observed GPU dispatch.

The parent starts before CUDA initialization and fork+execs a persistent GPU
worker. It buffers complete bounded responses before publishing success. Native
CUDA aborts, worker EOF, malformed headers and incomplete bodies return an
explicit worker-termination error. In-band errors are forwarded exactly. The
worker is retired after any error; a later request creates a new GPU worker.
There is no automatic resubmission of the failed image. CPU-only protocol
regressions cover startup exit 2, partial/invalid responses, errors, worker
replacement and no-replay behavior. The GPU fault test also aborts the native
FA dispatch once and requires an explicit error followed by a successful new
GPU request.

The installed M60 FA tile kernel fails for a partial KV tile. A finite-input
operator probe and separate Q/K lengths isolate that failure from preprocessing,
query tails and upstream model overflow. We have not identified the exact faulty
instruction in the installed kernel. The service uses its validated GPU FA path
for KV lengths divisible by 256 and a C/PTX online-softmax GPU FlashAttention
kernel for the remaining lengths. The choice is made before execution; a runtime
failure does not trigger another kernel or backend.

The PTX kernel visits every key using a 16-query × 32-key tile, predicated tails,
FP32 scores/probabilities/maxima/sums/PV, and online softmax. It allocates no global
attention matrix or workspace. The tested sm_52 cubin uses 25,376 bytes of shared
memory per CTA and 86 registers per thread, with zero spills. The installed FA
kernel manages its own scratch; its workspace/shared/register counts are not
measured by this adapter.

The model's published [processor configuration](https://huggingface.co/nopmobiel/Qwen3.8-Flash-Next-MLX-4bit/blob/main/processor_config.json)
and checked-in `kern/vision/preprocessor_config.json` specify 65,536–16,777,216
pixels, patch 16 and merge 2. We set 64–16,384 tokens explicitly, raise the
engine/launcher image capacity, and version the cache namespace. A 4096 × 4096
input retains 16,384 output tokens and a 128 × 128 M-RoPE grid. Inputs above the
pixel cap are rejected instead of being downscaled. Normal model patch alignment
still applies to dimensions that are not multiples of 32.

## Build and measurement identity

- Production code: `7cc72a6` (GPU FA implementation from `8059465`, followed by supervised GPU error handling).
- Isolated source/build on Archy: `/home/nate/work/fs96-sol`.
- Installed CUDA llama.cpp source: `fcac19079f3f2985fe74ed845f0880e7e9a8030a`,
  `/home/nate/claude/llama.cpp-cuda`. Existing CMake changes and binary hashes are
  recorded in the queue identity output; the prebuilt C++ libraries are reused.
- New authored/built code is C11 and PTX assembly. CUDA 11.4 `ptxas` assembles sm_52.
- GPU: Tesla M60, `CUDA6`, CUDA 11.8 runtime, driver 470.256.02.
- Projector: `/home/nate/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf`.
- Text model opened vocab-only:
  `/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`.
- Issue screenshot: 1387 × 452, SHA-256
  `007bb34ed767f38fbd703883af7be07422a84d81db2297cb4489e5a65b8bc013`.
- Direct, private encoder sockets and original file bytes; no image cache.
- Ten host threads. File decoding and model image preprocessing are host work;
  all observed projector compute operations and FA operations run on the GPU.

The queue database on Archy is the primary record of source snapshots, hashes,
GPU observations, command output, numerical records and timing records. All
collectors remain enabled. The unrelated resident text engine is unavailable
in these runs, and its trace collector errors are recorded as unavailable rather
than attributed to the image encoder.

## Current validation records

| Queue job | Purpose | Result |
|---|---|---|
| `146d9c41-9269-41be-a6b8-639b50208e3f` | First installed GPU FA layer with finite Q/K/V | Reproduces non-finite FA output |
| `c111c4bd-d403-42de-86d3-093d7a0c9dce` | Installed FA lengths 1024/2048/2304/2408/2432/4096 | Only 2408 fails |
| `db24b57e-9340-4ef3-93f1-fc7170c7a6d2` | Q=2408/K=2560 versus Q=2432/K=2408 | Isolates the KV tail |
| `67b8b74f-023a-4c8d-97a7-baae1c53a2b5` | C/PTX FA versus sampled float64 reference, every head, output guards | Pass |
| `0f0f2d20-d3f7-4020-9156-6fd8a76161f7` | Installed aligned GPU FA versus sampled float64 reference | Pass |
| `63fbc0f0-901d-4feb-ada5-a55d6b4c1d9e` | Actual placement of all screenshot compute nodes | 617 GPU compute operations, including 27 GPU FA operations |
| `78285269-a28f-4378-ae00-566d918c0fce` | Invalid/absent GPU, malformed input, injected error and NaN | Explicit errors; subsequent GPU FA requests succeed |
| `b1c3a80a-0caa-482c-8ce1-4466da720162` | Final production build, fault checks and full-size encodes | Inference checks pass; recorder postprocessing fails, as detailed below |
| `5f77f1c1-a8e5-4177-8ed9-4dc9347cae74` | Changed marker, native abort/error checks, sticky GPU reference and retained tensors | Pass, exit 0 |
| `162c3cbc-b217-4c62-adca-a9f8ee6d7065` | C-only client migration gate | Interrupted by recorder SQLite lock; capture_failed / exit 70 |
| `56050c63-1232-49db-8270-a9c5d78f497f` (separate issue96 history) | C-only recovery, dense GPU oracle, screenshot and minimum-size encodes | Pass, succeeded / exit 0 |

The first raw-kernel reference test (`f869d89f-3c9f-47d9-8414-ae3ac2dc75b8`)
exposed an out-of-range test sample when Nq=1. The corrected test passes. No
kernel change or tolerance relaxation was needed.

The initial CPU whole-model reference (`acebab20-c7f7-475b-a109-709a6656949f`)
and comparison (`944ef1d2-b3a6-4509-af15-9b1fa4348cce`) failed the strict per-token
limits. A CPU F16→F32 K/V promotion experiment (`20e67e2b-b25d-49f7-8823-426b8e908477`)
produced byte-identical CPU embeddings and also failed. These results do not
isolate the CPU/GPU numerical difference. They are retained as failed diagnostics
and are not an acceptance baseline. The separate `gpu_reference.c` executable
uses independent GPU F32 matrix products and softmax with the same rounded F16
K/V inputs. It verifies both reference attention matrix products execute on GPU
in every layer. The production service has no dense-attention mode or fallback.
Per-token acceptance limits remain cosine ≥ 0.999 and relative L2 ≤ 0.02.

### Kernel numerical checks

`fa_kernel_test.c` tests Q/K lengths 1/1, 13/35, 64/64, 2408/2408 and 13/65536,
with strided Q/output, finite random F16 K/V, a query scaled by 1024 to exercise
stable softmax, finite output checks and output allocation guards. Four sampled
queries in every one of 16 heads are compared with a double-precision softmax
reference over every key. Maximum observed absolute error: **1.5316 × 10⁻⁶**.
Acceptance limit: 2 × 10⁻⁴. Guard bytes remain intact.

`fa_native_test.c` tests aligned K lengths with Q/K 13/256, 512/512, 4096/4096,
16384/16384 and 13/65536 against the same style of double-precision reference.
Every output is finite. Maximum observed absolute error: **8.56692 × 10⁻⁸**.
At 16384/16384 the native test measured 1305.010 ms using the host monotonic
clock around graph execution and synchronization. The earlier C/PTX full-image
run measured about 3380 ms per layer using CUDA events. These are different
clock domains and exclude the rest of the projector and socket transfer. The
adapter's `ms` and `kernel_ms` fields are CUDA event intervals around dispatch;
they can include host launch gaps and native scratch-allocation delays. They
measure a dispatch interval, rather than isolated GPU busy time.

## Final image measurements

Final inference used source `7cc72a61511b3ff5ec1e07d438556046de85ed46` on Archy.
The encoder/client checks completed for every image:

| Input | Output tokens / grid | Encoder encode, host clock | Client + verification, host clock |
|---|---|---:|---:|
| Issue screenshot, 1387 × 452 | 602 / 43 × 14 | 3.272 s | 3.526162 s |
| 2048 × 2048 | 4,096 / 64 × 64 | 42.256 s | 43.837886 s |
| 4096 × 4096 | 16,384 / 128 × 128 | 595.711 s | 601.261089 s |

For each image the observer recorded **617 GPU compute operations**, including
**27 GPU FA dispatches**. Every output float was finite and every M-RoPE tuple
was checked. Encoder encode timing excludes image preprocessing and response
transfer. Client timing includes transfer, finite/position checks, SHA-256 and
writing the retained tensor. These measurements follow the fault checks, with
host/driver caches warm and the image cache bypassed. The first valid screenshot
encode in an earlier fresh-worker fault case took **56.861 s**; the cause of that
cold observation was not isolated. Startup/cold behavior remains distinct from
the measured image sequence above.

The independent GPU dense reference (`b5ad6bf1-031b-4c87-bdae-5a59423aaa5b`)
verified 779 GPU compute operations, including 27 KQ and 27 PV matrix products.
Comparison of all 602 output rows passed the unchanged limits:

- Global cosine: **0.999999947215**.
- Global relative L2: **0.000326004**.
- Minimum per-token cosine: **0.999994852832** (limit ≥ 0.999).
- Maximum per-token relative L2: **0.004566204** (limit ≤ 0.02).
- Maximum absolute component difference: **0.000391201**.

This whole-model reference covers the screenshot. Large-key operator references
cover every head and up to 65,536 keys; full-size image checks cover all output
values and positions. We have not run a whole-model float64 reference for the
4096 × 4096 image.

### VRAM and build identity

GPU6 (`GPU-431371bd-b482-96b2-4b70-04f83c6d075b`) had **3646 MiB** maximum
observed total device memory use in 637 NVML samples at the recorder's default
one-second interval; its minimum observed use was 71 MiB. This includes the
existing small text-core context on that GPU. The adapter's maximum post-FA
`total-free` sample was **3,823,632,384 bytes** (3646.5 MiB). These are sampled
device totals, with their sampling scope stated, rather than an isolated-process
or transient-allocation high-water mark. Other device samples include unrelated
resident holders and services.

| Artifact | SHA-256 |
|---|---|
| `kern/flashvis` | `ca18fb8e9bdfcd0a338b2cb299782495f3d0b4207c4213d975f8a1aa135e9905` |
| `kern/vis_fa.cubin` | `0d37f7fe72853ecd43862c8b65e383998241955bfaeb38518b090ddcb2027e54` |
| Installed `libmtmd.so` | `78a1ab2a4aaf235819b3537113709e0ad454b53fdc94814aa77aadcbca7b43e8` |
| Installed `libllama.so` | `bae4635cff4a280d77ac2a750335fd122ab25ea2560c8f7d1f20d804e0755601` |
| Installed `libggml-cuda.so` | `58716b3345fcbfa6cc4363a186524c5f752c7c0aa7e992f49d2d8dbe088eb741` |

Compiler: GCC 16.2.1 (20260810), GNU C11. Assembler: CUDA 11.4.152 `ptxas`,
sm_52. Remote DSH build `build-3488588-1792482507039570892` passed the CPU IPC
suite and returned artifacts from Archy. The changed test-only oracle and marker
were built by `build-3588364-1792482907253145470`. Both retain the same production
encoder/cubin hashes. No local compilation is needed on engi.

### Recorder postprocessing and retained output

The inference/fault command in `b1c3a80a-0caa-482c-8ce1-4466da720162` completed
all checks and recorded the successful image events above. Its final artifact
import then failed on an empty O_EXCL fault marker, so the **recorder reports
exit 70 / failed**. This is retained as a failed recorder job; it is not relabeled
as a successful queue run. The bug is tracked in [issue #101](https://github.com/nm-z/flash-sass/issues/101).
The test shim now writes a nonempty marker. A follow-up gate verifies those
errors with the changed marker, the sticky GPU-reference observer, and the
retained tensor byte counts, SHA-256 identities and all float values.

That follow-up, `5f77f1c1-a8e5-4177-8ed9-4dc9347cae74`, **succeeded with exit 0**.
It repeated the actual GPU failure checks, passed the sticky-observer numerical
reference, and verified 1,541,120 / 10,485,760 / 41,943,040 finite retained float
values with the exact hashes below. The completed full-resolution inference was
not recomputed to validate the artifact-marker change. Optional process-tree and
resident-engine trace collectors reported unavailable observations; the command,
numerical results, retained outputs and waited resource usage are recorded.

| Image | Retained embedding SHA-256 |
|---|---|
| Screenshot | `d9b7be627eb9828045805efc651e373127d29850acb11152a7d4c188f1beebab` |
| 2048 × 2048 | `eb0a3449d67a8265c93ccea9b4206f2af4a2b0c9098e961ac908b8fc6d0c188f` |
| 4096 × 4096 | `807b537caf63c6e4e9c898e544e36b199f332814e435070f411cc31eb91ba93a` |

Earlier CPU/standard-attention measurements are preserved in
[HISTORICAL.md](HISTORICAL.md); they do not validate this implementation.

## C and Make validation migration

The validation clients and orchestration now use C and Make. The added Bash
and Ruby files are removed, and the inherited launcher is restored. The Make
`vision-all` and `vision-core` targets provide the 16,384-token image setting.

Remote DSH build `build-257361-1792486269231372174` passed the C clients and
mock-worker suite with `-Wall -Wextra -Werror`. It checked exact protocol errors,
complete response buffering, fresh workers after failure, no replay, and group
cleanup. Startup cancellation under SIGINT, SIGTERM and SIGHUP left neither
mock supervisor nor worker alive. The C client blocks cancellation signals
across process-group setup, uses bounded nonblocking socket transfers, reaps
its child on exit, and publishes output by rename after the complete finite
tensor and positions have been validated.

The C checker reproduced the numerical comparison above, checked every value
and SHA-256 of all three retained tensors, and generated byte-identical 2048
and 4096 fixtures. The production encoder and cubin retain the hashes above.
The image timings above were collected with the earlier client; migrating the
checker does not constitute a new full-resolution latency measurement.

The first C-only GPU gate, `162c3cbc-b217-4c62-adca-a9f8ee6d7065`, recorded
the expected encoder/client identities and the three unavailable-GPU startup
checks, then GPUQ interrupted the command with SIGTERM. Its storage/capture
document `48518` reports `SQLite3::BusyException: database is locked`; the job
is `capture_failed`, exit 70. This is tracked in [issue #102](https://github.com/nm-z/flash-sass/issues/102).
The client's signal cleanup left no encoder process from that interrupted run.

The replay attempt `bd702439-1f10-43d8-8efb-51ab4f2e4720` and fresh submission
`b2ac6979-4443-4f3c-a054-91976a721f12` exited before command startup and were
reconciled to `lost`, tracked in [issue #103](https://github.com/nm-z/flash-sass/issues/103).
None of these interrupted attempts is counted as a pass.

The unchanged gate **succeeded with exit 0** as
`56050c63-1232-49db-8270-a9c5d78f497f`, using
the documented `--db /home/nate/.local/state/gpuq/issue96-history.sqlite3`
option. Source inspection verifies that this changes history storage while
retaining the shared `/tmp/gpuq` FIFO tickets and
`/tmp/flash-sass-engine-validation.lock` GPU mutex. All collectors remain enabled.
It passed all three unavailable-GPU startup cases and exact malformed,
over-limit, reported-error, NaN and native-abort rejections. Each image error
was followed by a successful GPU FA response. The new C client encoded the
screenshot in 3.555015250 s of host round-trip time and the 256 × 256 fixture
in 0.196587219 s. Both responses recorded 617 GPU compute operations and 27
GPU FA dispatches, with finite tensors and verified positions. The screenshot
tensor and independent dense GPU reference hashes exactly match the earlier
accepted results, and the unchanged per-token numerical limits pass.

The C client SHA-256 was
`6472628217029f210f16ccc3f43baa617824c9b040f17a6f1c7a325e03be4ba2`.
The source revision was `313916b`; the production encoder/cubin identities
were unchanged. Optional resident-engine/trace collection was unavailable,
and a process-tree sample missed command exit; waited resource usage and the
command's output remained recorded. No private encoder process remained after
the successful gate.

## Reproduce

Use the pinned installed CUDA libraries. Place this checkout at
`/home/nate/work/fs96-sol` on Archy before invoking the remote Makefile below.
The encoder target compiles C and assembles PTX; it reuses the installed
llama.cpp libraries. `LLAMA_CPP`/`LLAMA_BUILD` select matching source/build trees.
The CPU checks below use retained queue artifacts at the recorded paths; set
`ORIGINAL_RUN` and `ORACLE_RUN` when those artifacts live elsewhere.

```console
dsh build --host archy --require-tool cargo --require-tool cc -- make -f /home/nate/work/fs96-sol/validation/issue96/Makefile all cpu-check --artifact vision-artifacts
scp -r vision-artifacts archy:/home/nate/work/fs96-sol/
ssh archy gpuq --db /home/nate/.local/state/gpuq/issue96-history.sqlite3 run --name vision-checks-c -- make -f /home/nate/work/fs96-sol/validation/issue96/Makefile OUT_DIR=/home/nate/work/fs96-sol/vision-artifacts gpu-check
ssh archy gpuq --db /home/nate/.local/state/gpuq/issue96-history.sqlite3 run --name vision-full-c -- make -f /home/nate/work/fs96-sol/validation/issue96/Makefile OUT_DIR=/home/nate/work/fs96-sol/vision-artifacts full
```

The C-only `vision_check encode` command generates deterministic full-size BMPs, checks exact dimensions,
checks every output embedding for finiteness, verifies every M-RoPE tuple,
and records input/output SHA-256 and round-trip time. It stores binary output
under `$GPUQ_RUN_DIR`. `vision_check recover` verifies rejection of unsupported devices,
malformed/oversized inputs, a reported encode failure, native GPU abort and a non-finite output,
then requires another successful GPU FA response on the same daemon.
The fault interposer is test-only and is not linked into the service.

The Make `identity` target records source commits, compiler versions and executable,
cubin and installed-library hashes. The C `compare` and `retained` commands
perform numerical and retained-output checks. The C supervisor runner uses a
separately compiled mock worker; it invokes no shell or interpreter.
Retrieve results using `gpuq logs JOB_ID`,
`gpuq document ID --raw`, `gpuq metrics JOB_ID`, and read-only `gpuq query`.
Use the same `--db` option for records stored in the separate history file.
