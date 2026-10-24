#!/bin/bash
set -euo pipefail
build=/home/nate/codex/flash-astra-build-20261018
out=/home/nate/codex/flash-astra-perf-20261018/component
mkdir -p "$out"
exec 9>>/tmp/flash-sass-engine-validation.lock
flock -w 3600 9
uuid=GPU-431371bd-b482-96b2-4b70-04f83c6d075b
exec 8>>"/home/nate/codex/asm-port/$uuid.lock"
flock -w 60 8
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$out/processes-before.csv"
if awk -F, -v uuid="$uuid" '$1==uuid{found=1} END{exit !found}' "$out/processes-before.csv"; then
  echo 'Selected GPU has an existing context'; exit 75
fi
nvidia-smi --query-gpu=index,uuid,clocks.sm,temperature.gpu,utilization.gpu --format=csv > "$out/gpu-before.csv"
export CUDA_VISIBLE_DEVICES="$uuid"
for mode in hc tree hc4; do
  timeout -k 5 180 "$build/hc/kern/hc_phase_test" --baseline "$build/base/kern/die.cubin" --candidate "$build/$mode/kern/die.cubin" > "$out/$mode-hc.log" 2>&1
  DIE_CUBIN="$build/$mode/kern/die.cubin" timeout -k 5 180 "$build/base/kern/qsa_direct_test" 65 8 4 > "$out/$mode-qsa.log" 2>&1
  printf '%s component checks passed\n' "$mode"
done
nvidia-smi --query-gpu=index,uuid,clocks.sm,temperature.gpu,utilization.gpu --format=csv > "$out/gpu-after.csv"
