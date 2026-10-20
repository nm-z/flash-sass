# Issue #96: GPU image encoding

Issue: https://github.com/nm-z/flash-sass/issues/96

## Change

The old daemon forced `cp.use_gpu = false` and its default build linked the
CPU-only llama.cpp tree. The new default prefers the installed CUDA tree,
loads its backends explicitly, and selects the visible GPU with the most free
memory. `FLASHVIS_DEVICE=cpu` and explicit ggml GPU names remain available.

The M60 CUDA Flash Attention path returned non-finite embeddings in the first
GPU experiment. Standard attention passed the matched CPU/GPU comparison.
This result implicates that attention path; it does not isolate a particular
kernel instruction or prove FP16 overflow.

Every embedding is checked before the successful response header is sent.
Reported GPU allocation/encoding errors and non-finite output cause a single
retry from the original image bytes using a fresh CPU context with the original
CPU attention defaults. The daemon remains on CPU until restarted. Malformed
images do not cause this transition. Backend aborts are outside this in-process
recovery path.

## Workload and provenance

All commands run on Archy through Engi and `gpuq`. The queue database is the
primary record of source snapshots, executable hashes, stdout/stderr, workload
identity and hardware observations.

- Archy boot ID: `bfba2bdc-c2d5-4bff-af28-edf8f2bbbedf`.
- Tesla M60, CUDA device `CUDA6`; CUDA 11.8 runtime, driver 470.256.02.
- Matching llama.cpp CUDA source/build:
  `fcac19079f3f2985fe74ed845f0880e7e9a8030a`,
  `/home/nate/claude/llama.cpp-cuda`.
- Projector:
  `/home/nate/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf`.
- Text model opened vocab-only:
  `/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`.
- Image: the attachment in issue #96, 1387 × 452 pixels,
  SHA-256 `007bb34ed767f38fbd703883af7be07422a84d81db2297cb4489e5a65b8bc013`.
- Output: 602 image tokens, 43 × 14 grid, 43 sequence positions, width 2560.
- Both backends use the same binary, library build, projector, image bytes and
  ten CPU threads. The GPU uses standard attention; CPU retains its default.
  The harness talks directly to a private encoder socket, so no image cache
  can skip encoding.

## Preliminary result

Job `1a3afb14-acab-4580-973b-95b6c2e41154` passed on source `12ed5aa`.
This diagnostic used one warmup and one measured encode per backend.

| Host observation | CPU | CUDA6 |
|---|---:|---:|
| Warmup encode | 14.093 s | 2.262 s |
| Measured encode | 13.970 s | 1.879 s |
| Measured image round trip | 14.026980 s | 1.914520 s |

The measured round trip was 7.3266× faster. All 1,541,120 output values were
finite; dimensions and M-RoPE positions matched exactly. Global cosine
similarity was 0.9999651133 and relative L2 error was 0.0083736331.

This is image-encoder latency, not complete model TTFT or text decode throughput.
Startup, first encode, warmed encode and socket round trip are separate records.
Encode time excludes transferring the embeddings to the client.

## Final checks

The following jobs validate source `e47ff85b5f56bcf186eb77e5bbaa8674c36b0f30`,
which adds guarded output and request-time CPU recovery.

| Job | Purpose | Status |
|---|---|---|
| `78e47aee-d8ca-41b9-924c-3acde3e547a2` | Default CUDA build; one warmup and three measured encodes per backend; malformed image rejection; global and per-token numeric comparison | Queued |
| `6f761b7b-8c5a-4066-99f0-84cce5a4a8bc` | Invalid device, no visible GPU, reported encode failure, injected NaN, exact CPU recovery output, subsequent CPU request | Queued |
| `f5151846-89bd-48b2-9458-4b411a67fdb7` | 2048 × 2048 synthetic BMP: real allocation failure and CPU recovery retaining all 4096 image tokens | Queued |
| `9f9a8d51-2e68-49dd-8227-8a1620b65c0d` | Image question through a private HTTP frontend and image cache, using the existing shared text core | Queued |

The fault interposer is test-only. It is never linked into or loaded by the
normal daemon. The maximum-size fixture is generated deterministically by the
recovery script; the original issue image is not modified.

## Reproduce

Run from this checkout on Archy:

```sh
gpuq run --name issue96-gpu-image-final -- bash -c '
  make -B -C kern flashvis &&
  ruby kern/flashvis_bench.rb kern/flashvis "$MMPROJ" "$TEXT_GGUF" "$IMAGE" 3
'
```

Export the three input variables to the paths above (the image is downloaded from
the issue). The recorded job arguments contain the exact paths used.
`LLAMA_CPP` and `LLAMA_BUILD` can explicitly select another matching build.

Build `validation/issue96/fault.c` as a shared C library against the same
llama.cpp headers, then run `validation/issue96/recovery.rb` under `gpuq`
with arguments `BINARY MMPROJ TEXT_GGUF IMAGE FAULT_SO`. Add `large` to
exercise the maximum-size image. Exact build/run arguments are retained with
the recovery jobs.

`validation/issue96/http.sh MMPROJ TEXT_GGUF IMAGE` builds a private HTTP frontend
and starts a private encoder, with a new image cache and the existing shared
text core. It reuses `kern/vision_test.sh`, requires a nonempty image answer and
a CUDA encode, and bounds the request to 300 seconds. The script stops only its
own encoder/frontend processes on exit.

```sh
gpuq logs JOB_ID
gpuq metrics JOB_ID
gpuq query 'SELECT * FROM job_metric_status WHERE job_id="JOB_ID"'
```

Collectors remain enabled. The resident engine has no enabled raw trace
producer, which the recorder reports as unavailable; shared-engine observations
are not attributed to this encoder workload. The daemon's own timing records
and socket measurements provide the comparison.
