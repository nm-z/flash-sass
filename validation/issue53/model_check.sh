#!/usr/bin/env bash
# Reproduce SPEC.md's whole-token model_test case with the C twins enabled.
# This covers blocks [0,4), embedding, PLE, attention and head, not all 48 layers.
# Usage: model_check.sh BASELINE_DIR CANDIDATE_DIR SHARD1 SHARD2 CACHE OUT
set -euo pipefail
[[ $# == 6 ]] || { echo "usage: $0 BASELINE_DIR CANDIDATE_DIR SHARD1 SHARD2 CACHE OUT" >&2; exit 64; }
baseline=$(realpath "$1")
candidate=$(realpath "$2")
shard1=$(realpath "$3")
shard2=$(realpath "$4")
cache=$(realpath "$5")
out=$(realpath -m "$6")
[[ ! -e $out ]] || { echo "use a fresh output directory" >&2; exit 64; }
mkdir -p "$out"
holder_pid=
cleanup() {
    result=$?
    trap - EXIT INT TERM HUP
    if [[ -n $holder_pid ]]; then
        # Only this wrapper's child holder is stopped. Resident holders/core are untouched.
        kill -TERM "$holder_pid" 2>/dev/null || true
        wait "$holder_pid" || true
    fi
    printf '%s\n' "$result" >"$out/exit-code.txt"
    date -u +%FT%TZ >"$out/finished-utc.txt"
    exit "$result"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP
date -u +%FT%TZ >"$out/started-utc.txt"
printf '%s\n' "$$" >"$out/wrapper.pid"
sha256sum "$0" >"$out/wrapper.sha256"
gpu=${HC_GPU:-7}
wait_seconds=${HC_WAIT_SECONDS:-2700}
[[ $gpu =~ ^[0-9]+$ && $wait_seconds =~ ^[1-9][0-9]*$ && $wait_seconds -le 2700 ]] || exit 64
[[ -x $candidate/hold && -x $candidate/model_test && -x $baseline/model_test ]] || exit 66
for name in "${!MODEL_@}"; do unset "$name"; done
unset ATTN_MAP SPLIT_SHARES
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu" HOLD_VISIBLE=1 ATTN_DIE=0
manifest="$out/holder.manifest"
HOLD_DRY=1 "$candidate/hold" "$shard1" "$gpu" 0 4 "$manifest" token_embd output split=0/1 >"$out/holder-size.stdout" 2>"$out/holder-size.stderr"
printf '%s\n' '248045' >"$out/free-run.ids"
printf '%s\n' 248045 872 198 3838 >"$out/teacher-forced.ids"
sha256sum "$baseline/model_test" "$baseline/die.cubin" "$candidate/model_test" "$candidate/die.cubin" "$candidate/hold" >"$out/artifacts.sha256"
for tensor in token_embd.weight output.weight blk.0.hc_attn_dj.weight blk.1.ple_key.weight blk.3.attn_q.weight; do
    [[ -r $cache/$tensor ]] || { echo "missing C-twin tensor $cache/$tensor" >&2; exit 66; }
done
exec 9>/tmp/flash-sass-engine-validation.lock
deadline=$((SECONDS + wait_seconds))
while (( SECONDS < deadline )); do
    if ! flock -w 5 9; then
        echo "WAIT shared engine-validation lock"
        continue
    fi
    timeout 10s nvidia-smi -i "$gpu" >"$out/gpu-readiness.txt"
    ps -eo pid,ppid,etime,args >"$out/process-readiness.txt"
    row=$(timeout 10s nvidia-smi -i "$gpu" --query-gpu=uuid,utilization.gpu,memory.free --format=csv,noheader,nounits)
    IFS=, read -r uuid utilization free_mib <<<"$row"
    uuid=${uuid//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    free_mib=${free_mib//[[:space:]]/}
    [[ $uuid == GPU-* && $utilization =~ ^[0-9]+$ && $free_mib =~ ^[0-9]+$ ]] || exit 1
    apps=$(timeout 10s nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
    occupied=$(awk -F, -v uuid="$uuid" '{gsub(/[[:space:]]/, "", $1); if ($1 == uuid) print $2}' <<<"$apps")
    reserved=0
    if [[ -n ${GPU_LOCKDIR:-} ]]; then
        exec 8>"$GPU_LOCKDIR/$uuid.lock"
        if ! flock -n 8; then reserved=1; fi
    fi
    host_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    [[ $host_kib =~ ^[0-9]+$ ]] || exit 1
    if [[ -n $occupied ]] || (( utilization != 0 || free_mib < 5000 || host_kib < 12582912 || reserved )); then
        echo "WAIT device $gpu occupied/active/reserved, GPU free <5000 MiB, or host available <12 GiB"
        if [[ -n ${GPU_LOCKDIR:-} ]]; then exec 8>&-; fi
        flock -u 9
        sleep 5
        continue
    fi
    break
done
(( SECONDS < deadline )) || { echo "resource wait exhausted after ${wait_seconds}s" >&2; exit 75; }
"$candidate/hold" "$shard1" "$gpu" 0 4 "$manifest" token_embd output split=0/1 >"$out/holder.stdout" 2>"$out/holder.stderr" &
holder_pid=$!
printf '%s\n' "$holder_pid" >"$out/holder.pid"
ready_deadline=$((SECONDS + 900))
until [[ -f $manifest.ready ]]; do
    kill -0 "$holder_pid" 2>/dev/null || { wait "$holder_pid"; exit 1; }
    (( SECONDS < ready_deadline )) || { echo "holder did not become ready within 900s" >&2; exit 75; }
    sleep 1
done
echo "PASS owned holder ready; GPU $gpu; blocks [0,4)"
for arm in baseline candidate; do
    directory=$baseline
    [[ $arm == baseline ]] || directory=$candidate
    for mode in token loop; do
        flags=eh
        ids="$out/free-run.ids"
        [[ $mode == token ]] || { flags=ehl; ids="$out/teacher-forced.ids"; }
        echo "START $arm model_test $mode"
        printf '%q ' env "MODEL_IDS=$ids" "MODEL_PLE=$shard2" MODEL_PLE_OFF=192 "$directory/model_test" "$cache" 0 4 4 "$flags" "$manifest" >"$out/$arm-$mode.command.txt"
        printf '\n' >>"$out/$arm-$mode.command.txt"
        set +e
        (cd "$directory" && MODEL_IDS="$ids" MODEL_PLE="$shard2" MODEL_PLE_OFF=192 timeout --kill-after=5s 900s ./model_test "$cache" 0 4 4 "$flags" "$manifest") >"$out/$arm-$mode.stdout" 2>"$out/$arm-$mode.stderr"
        result=$?
        set -e
        printf '%s\n' "$result" >"$out/$arm-$mode.exit"
        echo "RESULT $arm $mode exit=$result"
        (( result == 0 )) || exit "$result"
        if [[ $mode == token ]]; then
            rg -q '^OK: 4 of 4 tokens exact;' "$out/$arm-$mode.stdout"
        else
            rg -q 'die 0 tokens: all 4 exact' "$out/$arm-$mode.stdout"
            rg -q 'S.0 exact' "$out/$arm-$mode.stdout"
        fi
    done
done
echo "PASS documented whole-token model_test: baseline/candidate, token/loop, C twins enabled, blocks [0,4)"
