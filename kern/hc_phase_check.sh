#!/bin/bash
# Run HC kernel checks on an idle device, without changing the resident engine.
# Usage: HC_GPU=6 ./hc_phase_check.sh BASELINE_DIE_CUBIN SHARD1
# Optional GPU_LOCKDIR uses the host's per-UUID reservation convention as well
# as the mandatory shared validation lock. All checks fail closed on bad data.
set -euo pipefail
if [[ $# != 2 ]]; then
    echo "usage: HC_GPU=6 $0 BASELINE_DIE_CUBIN SHARD1" >&2
    exit 64
fi
baseline=$(realpath "$1")
shard=$(realpath "$2")
cd -- "$(dirname -- "$0")"
gpu=${HC_GPU:-6}
wait_seconds=${HC_WAIT_SECONDS:-300}
[[ $gpu =~ ^[0-9]+$ && $wait_seconds =~ ^[1-9][0-9]*$ ]] || exit 64
[[ -r $baseline && -r $shard ]] || exit 66
for executable in hcops_test gemv_bf16_test hc_phase_test pair_bf16_test; do
    [[ -x ./$executable ]] || { echo "build $executable first" >&2; exit 66; }
done
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$gpu

run_locked() (
    deadline=$((SECONDS + wait_seconds))
    exec 9>/tmp/flash-sass-engine-validation.lock
    while (( SECONDS < deadline )); do
        if ! flock -w 5 9; then
            echo "WAIT: shared validation lock is held"
            continue
        fi
        # Both inspections happen under the shared lock before every process.
        timeout 10s nvidia-smi -i "$gpu"
        ps -eo pid,ppid,etime,args | awk 'NR == 1 || /[f]lashcore|[.]\/hold|[.]\/fence|[h]c_phase|[g]emv_bf16|[h]cops_test/'
        row=$(timeout 10s nvidia-smi -i "$gpu" --query-gpu=uuid,utilization.gpu,memory.free --format=csv,noheader,nounits)
        IFS=, read -r uuid utilization free_mib <<< "$row"
        uuid=${uuid//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        free_mib=${free_mib//[[:space:]]/}
        [[ $uuid == GPU-* && $utilization =~ ^[0-9]+$ && $free_mib =~ ^[0-9]+$ ]] || {
            echo "cannot verify device readiness" >&2; exit 1;
        }
        apps=$(timeout 10s nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
        occupied=$(awk -F, -v uuid="$uuid" '{gsub(/[[:space:]]/, "", $1); if ($1 == uuid) print $2}' <<< "$apps")
        reserved=0
        if [[ -n ${GPU_LOCKDIR:-} ]]; then
            exec 8>"$GPU_LOCKDIR/$uuid.lock"
            if ! flock -n 8; then reserved=1; fi
        fi
        if [[ -n $occupied ]] || (( utilization != 0 || free_mib < 512 || reserved )); then
            echo "WAIT: device $gpu is occupied, active, reserved, or has less than 512 MiB free"
            if [[ -n ${GPU_LOCKDIR:-} ]]; then exec 8>&-; fi
            flock -u 9
            sleep 5
            continue
        fi
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
        set +e
        timeout --kill-after=5s 180s "$@"
        result=$?
        set -e
        printf 'exit_code=%d\n' "$result"
        # Closing the subshell releases both locks, including on any error.
        exit "$result"
    done
    echo "validation waited ${wait_seconds}s; device or lock remains unavailable" >&2
    exit 75
)

run_locked ./hcops_test
run_locked ./pair_bf16_test
run_locked ./gemv_bf16_test "$shard" blk.4.ffn_gate_inp.weight 0
run_locked ./gemv_bf16_test "$shard" blk.4.hc_attn_down.weight 0
run_locked ./gemv_bf16_test "$shard" blk.4.hc_attn_up.weight 0
run_locked ./hc_phase_test --baseline "$baseline"
