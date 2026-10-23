#!/usr/bin/env bash
set -euo pipefail
: "${GPUQ_JOB_ID:?run through gpuq}"
: "${GPUQ_RUN_DIR:?run through gpuq}"
cd /home/nate/work/flash-drafts-20261020/pr-43/kern
export CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b
export DIE_CUBIN="$PWD/die.cubin"
sha256sum qsa_gather_bench die.cubin | gpuq put pr43-binary-identity
nvidia-smi --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu --format=csv > "$GPUQ_RUN_DIR/gpu-before.csv"
for mode in 0 1; do
    ./qsa_gather_bench 2000 3 32 changing "$mode" "$GPUQ_RUN_DIR/host-$mode.csv" 4242 > "$GPUQ_RUN_DIR/host-$mode.json"
    gpuq put "pr43-host-$mode-summary" < "$GPUQ_RUN_DIR/host-$mode.json"
    gpuq put "pr43-host-$mode-csv" < "$GPUQ_RUN_DIR/host-$mode.csv"
done
gpuq emit pr43_gather_modes_pass count 2
