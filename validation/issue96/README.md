# Issue #96: GPU image encoding

Issue: https://github.com/nm-z/flash-sass/issues/96

## Change

The old daemon forced `cp.use_gpu = false` and its default build linked the
CPU-only llama.cpp tree. The new default prefers the installed CUDA tree,
loads its backends explicitly, and selects the visible GPU with the most free
memory. `FLASHVIS_DEVICE=cpu` and explicit ggml GPU names remain available.

GPU encoding uses standard attention. The M60 CUDA Flash Attention path produced
non-finite embeddings in the initial experiment. Disabling that path produced
finite output; this does not isolate a particular kernel instruction or prove
FP16 overflow.

Every embedding is checked before the successful response header is sent.
Reported GPU allocation/encoding errors and non-finite output cause one retry
from the original image bytes with a fresh CPU context and the original CPU
attention defaults. The daemon remains on CPU until restarted. Malformed images
do not cause this transition.

For the Qwen3-VL projector, a memory preflight also avoids oversized dense
attention allocations. It reads the head count and spatial merge size from the
GGUF, estimates one FP32 attention-score tensor, and keeps half the currently
free GPU memory for other buffers. This is a conservative heuristic, not a full
graph-memory bound. Unknown projector families do not use this preflight.
Backend aborts, including memory races or allocations this estimate misses,
remain outside the in-process recovery path and require a daemon restart.

## Workload and provenance

All runtime checks use Archy through Engi and `gpuq`. The queue database retains
source snapshots, executable hashes, stdout/stderr and hardware observations.

- Archy boot ID: `bfba2bdc-c2d5-4bff-af28-edf8f2bbbedf`.
- Tesla M60, CUDA device `CUDA6`; CUDA 11.8 runtime, driver 470.256.02.
- Matching llama.cpp CUDA source/build:
  `fcac19079f3f2985fe74ed845f0880e7e9a8030a`,
  `/home/nate/claude/llama.cpp-cuda`.
- Projector: `/home/nate/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf`.
- Text model opened vocab-only:
  `/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`.
- Image: issue #96's attachment, 1387 × 452 pixels,
  SHA-256 `007bb34ed767f38fbd703883af7be07422a84d81db2297cb4489e5a65b8bc013`.
- Output: 602 image tokens, 43 × 14 grid, 43 sequence positions, width 2560.

Both production backends use the same binary, library build, projector, image
bytes and ten CPU threads. The CPU keeps its original Flash Attention default;
the GPU uses standard attention. The benchmark talks directly to a private
encoder socket, so an image cache cannot skip encoding.

## Final latency comparison

One warmup and three measured encodes per production backend:

| Host observation | Original CPU configuration | Automatic GPU (`CUDA6`) |
|---|---:|---:|
| Warmup encode | 14.075 s | 2.264 s |
| Median measured encode | 14.038 s | 1.871 s |
| Median measured image round trip | 14.073923 s | 1.910781 s |

The median image round trip was **7.3655× faster**. Startup, warmup,
measured encode and socket round trip are separate records. Encode time excludes
transferring embeddings to the client. This small, shared-host comparison covers
the image encoder; it does not establish an equivalent improvement in complete
model TTFT or text generation throughput.

## Numerical checks

The original CPU and GPU configurations use different attention modes. The
initial per-token comparison exposed that mismatch; forcing cuBLAS FP32 did not
change the result. The final harness therefore retains the original CPU mode as
the latency baseline and adds one CPU standard-attention encode as a matching
numerical reference. Only that reference subprocess uses the test interposer.
The production CPU fallback retains its original memory-efficient attention.

| GPU comparison against | Global cosine | Global relative L2 | Minimum token cosine | Maximum token relative L2 |
|---|---:|---:|---:|---:|
| Original CPU attention | 0.9999651133 | 0.0083736331 | 0.9903172918 | 0.1396311655 |
| CPU standard attention | 0.9999460855 | 0.0103874508 | 0.9952528243 | 0.0995611796 |

All 1,541,120 output values are finite. Dimensions and M-RoPE positions match
exactly. Both comparisons require global cosine ≥ 0.99 and relative L2 ≤ 0.1.
The matching standard-attention comparison additionally applies the unchanged
cosine ≥ 0.99 and relative L2 ≤ 0.1 limits to every image token. This controls the
attention-mode difference; it does not establish either backend as mathematical
ground truth or promise identical downstream answers.

## Recovery and HTTP checks

| Check | Observed result |
|---|---|
| Invalid GPU name | Exit status 2 with a clear device error |
| No visible CUDA GPU | CPU response exactly matches the forced-CPU baseline |
| One reported GPU encode error | One CPU transition; complete response exactly matches CPU; next request stays on CPU |
| One injected NaN | No invalid success response; one CPU transition; exact CPU response; next request stays on CPU |
| Malformed image followed by valid image | Bad input rejected; valid image retains the selected backend |
| 2048 × 2048 image | Preflight detects a 16,384 MiB score tensor versus 6,560 MiB available; returns all 4,096 image tokens on CPU |
| Subsequent 64 × 64 image after large-image fallback | Finite CPU response; no second transition |
| HTTP image question with empty private cache | CUDA6 encode and correct description of the screenshot's CPU/GPU resource monitors |

The maximum-size CPU encode took 259.729 seconds; the complete retry took
261.808 seconds. Its output shape was `[4096, 64, 64, 64, 2560]`. The guard avoids
the previously observed 16,936.25 MiB CUDA graph allocation failure without
reducing image resolution or token count. Large CPU fallback is substantially
slower than the ordinary-image GPU path.

The final HTTP check used a private encoder/frontend/cache and the existing
shared text core. It returned 86 completion tokens from 628 prompt tokens, with
no cached prompt positions. Image encoding took **2.254 seconds** on CUDA6.
Text-engine prefill took **15.376 seconds (40.84 tokens/s)** and decode measured
**38.94 tokens/s**. These separate phases explain why removing CPU image
encoding does not remove the text engine's prefill cost. The serving encoder and
frontend binaries were not replaced by this check.

The interposer is test-only and is never loaded by the normal service. Its fault
modes inject one reported error or one NaN; its reference mode changes only CPU
attention. The maximum-size fixture is a deterministic synthetic BMP, not a
resized version of the issue attachment.

## Recorded jobs

Final encoder C source is `63a23450ade72531edbc6a0a36629f22eec864ec`.
The reference interposer and HTTP capture changes are in
`c84a0448ea3dd56ebdd3a9bf26c93f88587b5fbc`. Benchmark socket names are shortened
in `de99472` to fit the Unix-socket path limit. Both leave the encoder C source
unchanged; the final comparison runs on `de99472`.

| Job | Purpose | Result |
|---|---|---|
| `beb622a7-0454-4841-b509-dd31a5f3037f` | Three measured encodes per production backend plus matching-attention numerical reference | Pass |
| `c516cf32-1584-4740-b92a-ae2197e4a36f` | Invalid device, no GPU, reported error, NaN and subsequent CPU requests | Pass |
| `5d87781a-b336-409e-aa35-89c5284da0e1` | Real maximum-size image with memory preflight and all 4,096 tokens retained | Pass |
| `dc105c22-dc3d-499e-9255-31525ea63a55` | HTTP image request through the private frontend and existing core | Pass |
| `a7002f1f-7a49-4c8f-acd6-9d20420fb76b` | Diagnostic matching-attention CPU/GPU comparison | Pass |

Useful earlier diagnostics remain in the database: `67be441d-7e94-4511-9b8b-77e5fe5da58c`
exposed non-finite GPU Flash Attention output; `78e47aee-d8ca-41b9-924c-3acde3e547a2`
exposed the mixed-attention per-token difference; `f5151846-89bd-48b2-9458-4b411a67fdb7`
exposed the unguarded maximum-size allocation failure. The final benchmark's
first attempt, `56c31482-c27b-413d-a06a-f50353c582e7`, was interrupted by a recorder
SQLite lock and is not used for the performance result. The subsequent
`f33c7d38-dd68-4f67-919f-6cf60c13ac29` attempt exposed a too-long reference socket
path in the harness; the final run uses the corrected short path.

## Reproduce

From this checkout on Archy, export `LLAMA_CPP`, `MMPROJ`, `TEXT_GGUF` and `IMAGE`
to the matching CUDA tree and input paths above. Build only the C daemon and C
test interposer; the existing CUDA llama.cpp libraries are reused:

```sh
gpuq run --name issue96-compare -- bash -c '
  make -B -C kern flashvis &&
  cc -std=gnu99 -Wall -shared -fPIC \
    -I"$LLAMA_CPP/include" -I"$LLAMA_CPP/ggml/include" \
    -I"$LLAMA_CPP/tools/mtmd" validation/issue96/fault.c \
    -ldl -o kern/flashvis_fault.so &&
  ruby kern/flashvis_bench.rb kern/flashvis "$MMPROJ" "$TEXT_GGUF" "$IMAGE" 3
'

gpuq run --name issue96-recovery -- ruby validation/issue96/recovery.rb \
  kern/flashvis "$MMPROJ" "$TEXT_GGUF" "$IMAGE" kern/flashvis_fault.so

gpuq run --name issue96-large -- ruby validation/issue96/recovery.rb \
  kern/flashvis "$MMPROJ" "$TEXT_GGUF" "$IMAGE" kern/flashvis_fault.so large

gpuq run --name issue96-http -- bash validation/issue96/http.sh \
  "$MMPROJ" "$TEXT_GGUF" "$IMAGE"
```

`LLAMA_BUILD` can select another build directory from the same source tree.
Retrieve evidence with `gpuq logs JOB_ID`, `gpuq metrics JOB_ID` and `gpuq query`.
Collectors remain enabled. The resident engine has no enabled raw trace producer;
the recorder marks that trace unavailable, and shared-engine observations are
not attributed to the encoder workload. The daemon's timing records and the
harness's socket measurements provide the latency comparison.
