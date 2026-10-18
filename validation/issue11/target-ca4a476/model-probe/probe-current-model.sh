#!/usr/bin/env bash
# Attempt a fresh 48-layer GPU capture using the pinned sibling capture adapter
# and PR31's exact current production image. Attach resident weight arenas
# read-only; do not reload, stop, or reconfigure a resident engine or holder.
set -euo pipefail
(($# == 1)) || exit 2
root=$1
mkdir "$root/model-probe"
out="$root/model-probe"
hostname > "$out/hostname.txt"
date -Is > "$out/start.txt"
exec 9>/tmp/flash-sass-engine-validation.lock
printf 'QUEUED: current-image full-model capture probe\n' | tee "$out/status.txt"
set +e
flock -w 600 -E 75 9
rc=$?
set -e
if ((rc)); then printf '%s\n' "$rc" > "$out/exit.txt"; exit "$rc"; fi
printf 'RUNNING\n' | tee "$out/status.txt"
date -Is > "$out/lock-acquired.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$out/gpu-before.csv"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader > "$out/processes-before.csv"
cd "$root/model-adapter/kern"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 CUDA_DEVICE_ORDER=PCI_BUS_ID
export MODEL_ATTN=qsa MODEL_NOTWIN=1 FLASH_NBLK=512 FLASH_KV_HOST=1
export MODEL_PLE=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf MODEL_PLE_OFF=192
export SPLIT_SHARES=112,97,81,116,97,97 ATTN_DIE=1 ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5
export MODEL_IDS="$root/model-ids.txt" MODEL_CAPTURE="$out/capture"
export MODEL_LAYER_CAPTURE="$out/layer3" MODEL_LAYER_INDEX=3 MODEL_LAYER_LENGTHS=1,4
# No scalar-twin substitution, model truncation, input injection or sampling.
unset MODEL_NPH MODEL_INJECT
printf '%s\n' ./model_test /unused-with-MODEL_NOTWIN 0 48 4 eh /tmp/hold-h{0,1,2,3,4,5}.manifest > "$out/command.argv"
set +e
timeout --signal=TERM --kill-after=15 120 ./model_test /unused-with-MODEL_NOTWIN 0 48 4 eh /tmp/hold-h{0,1,2,3,4,5}.manifest > "$out/run.stdout" 2> "$out/run.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/exit.txt"
date -Is > "$out/end.txt"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader > "$out/gpu-after.csv"
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader > "$out/processes-after.csv"
if ((rc == 0)); then printf 'CAPTURE_COMPLETED\n' | tee "$out/status.txt"; else printf 'CAPTURE_FAILED exit=%s\n' "$rc" | tee "$out/status.txt"; fi
cat "$out/run.stderr"
exit "$rc"
