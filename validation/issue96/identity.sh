#!/bin/bash
set -euo pipefail
: "${GPUQ_JOB_ID:?run through gpuq}"
printf 'source_commit='
git rev-parse HEAD
git status --short
cc --version | head -1
/opt/cuda-11.4/bin/ptxas --version
sha256sum kern/flashvis kern/vis_fa.cubin kern/vis_fa.ptx kern/vis_ggml.c
for lib in libmtmd.so libllama.so libggml.so libggml-base.so libggml-cuda.so; do
  sha256sum "/home/nate/claude/llama.cpp-cuda/build/bin/$lib"
done
printf 'dependency_source_commit='
git -C /home/nate/claude/llama.cpp-cuda rev-parse HEAD
git -C /home/nate/claude/llama.cpp-cuda diff --stat
stat -c '%n bytes=%s modified=%y' /home/nate/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mmproj-BF16.gguf /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
