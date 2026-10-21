#!/usr/bin/env bash
# Reserve an idle GPU, load an owned one-die holder, run a validation command,
# and release only the process and lock this script owns.
set -euo pipefail
if (( $# < 10 )) || [[ $9 != -- ]]; then
    printf 'usage: %s REPO NEW_OUT GPU_INDEX GPU_UUID CHECKPOINT B0 B1 MIN_FREE_MIB -- COMMAND...\n' "$0" >&2
    exit 2
fi
repo=$(realpath "$1")
out=$2
gpu_index=$3
gpu_uuid=$4
checkpoint=$(realpath "$5")
b0=$6
b1=$7
min_free_mib=$8
shift 9
[[ $gpu_index =~ ^[0-9]+$ && $b0 =~ ^[0-9]+$ && $b1 =~ ^[0-9]+$ && $min_free_mib =~ ^[0-9]+$ ]] || exit 2
[[ -x "$repo/kern/hold" && -x "$repo/kern/model_test" ]] || exit 2
mkdir "$out"
out=$(realpath "$out")
holder_pid=
cleanup() {
    local status=$?
    trap - EXIT
    set +e
    if [[ -n $holder_pid ]]; then
        kill -TERM "$holder_pid" 2>/dev/null
        wait "$holder_pid"
    fi
    nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu --format=csv > "$out/after-cleanup-gpu.csv" 2>&1
    flock -u 9
    exit "$status"
}
exec 9>>/tmp/flash-sass-engine-validation.lock
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
deadline=$((SECONDS + 1200))
while :; do
    if flock -w 30 9; then
        nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu --format=csv > "$out/preflight-gpu.csv"
        nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$out/preflight-compute.csv"
        ps -eo pid,ppid,etimes,stat,comm > "$out/preflight-ps.txt"
        got_uuid=$(awk -F, 'NR==2 {gsub(/^ +| +$/,"",$2); print $2}' "$out/preflight-gpu.csv")
        free_mib=$(awk -F, 'NR==2 {gsub(/[^0-9]/,"",$5); print $5}' "$out/preflight-gpu.csv")
        utilization=$(awk -F, 'NR==2 {gsub(/[^0-9]/,"",$7); print $7}' "$out/preflight-gpu.csv")
        [[ $got_uuid == "$gpu_uuid" ]] || { printf 'GPU index/UUID mismatch\n' >&2; exit 2; }
        if [[ ! -s "$out/preflight-compute.csv" && $utilization == 0 && $free_mib -ge $min_free_mib ]]; then
            break
        fi
        flock -u 9
    fi
    date -u '+%Y-%m-%dT%H:%M:%SZ waiting for lock/idle GPU' | tee -a "$out/wait.log"
    (( SECONDS < deadline )) || exit 75
    sleep 5
done
export CUDA_VISIBLE_DEVICES="$gpu_index"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export SPLIT_SHARES=1
export ATTN_DIE=0
export ATTN_MAP=0,0,0,0,0,0,0,0,0,0,0,0
export QSA_MODEL_MANIFEST="$out/holder.manifest"
sha256sum "$repo/kern/hold" "$repo/kern/model_test" "$repo/kern/die.cubin" > "$out/binaries.sha256"
jq -n --arg hostname "$(hostname)" --arg gpu "$gpu_uuid" --arg checkpoint "$checkpoint" --argjson gpu_index "$gpu_index" --argjson b0 "$b0" --argjson b1 "$b1" --args '{hostname:$hostname,gpu_uuid:$gpu,gpu_index:$gpu_index,checkpoint:$checkpoint,b0:$b0,b1:$b1,command:$ARGS.positional,scope:"one-die partial-model fixture; not the full production die layout"}' -- "$@" > "$out/command.json"
holder_command=("$repo/kern/hold" "$checkpoint" "$gpu_index" "$b0" "$b1" "$QSA_MODEL_MANIFEST" token_embd output split=0/1)
jq -n --args '$ARGS.positional' -- "${holder_command[@]}" > "$out/holder-command.json"
HOLD_VISIBLE=1 "${holder_command[@]}" 9>&- > "$out/holder.log" 2>&1 &
holder_pid=$!
printf '%s\n' "$holder_pid" > "$out/holder.pid"
ready_deadline=$((SECONDS + 900))
until [[ -f "$QSA_MODEL_MANIFEST.ready" ]]; do
    if ! kill -0 "$holder_pid" 2>/dev/null; then
        wait "$holder_pid" || true
        printf 'Owned holder exited before becoming ready\n' >&2
        exit 1
    fi
    (( SECONDS < ready_deadline )) || { printf 'Owned holder readiness timeout\n' >&2; exit 124; }
    sleep 1
done
export QSA_MODEL_HOLDER_PID="$holder_pid"
sha256sum "$QSA_MODEL_MANIFEST" > "$out/holder-manifest.sha256"
nvidia-smi -i "$gpu_index" --query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu --format=csv > "$out/before-model-gpu.csv"
nvidia-smi -i "$gpu_index" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$out/before-model-compute.csv"
ps -eo pid,ppid,etimes,stat,comm > "$out/before-model-ps.txt"
printf 'Owned holder %s ready; running validation under shared lock\n' "$holder_pid"
cd "$repo/kern"
set +e
"$@" > "$out/model.stdout" 2> "$out/model.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$out/exit-code.txt"
cat "$out/model.stdout" "$out/model.stderr"
exit "$status"
