#!/usr/bin/env bash
# Run one real-model sequence matrix without disturbing another GPU owner.
set -euo pipefail
if (($# != 8)); then
    printf 'usage: %s OUTPUT_DIR PLE_SHARD MANIFEST0 MANIFEST1 MANIFEST2 MANIFEST3 MANIFEST4 MANIFEST5\n' "$0" >&2
    exit 2
fi
src=$(cd -- "$(dirname -- "$0")" && pwd)
out=$(realpath -m -- "$1")
ple=$(realpath -e -- "$2")
shift 2
manifests=()
for manifest in "$@"; do manifests+=("$(realpath -e -- "$manifest")"); done
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
revision=${SEQUENCE_REVISION:-$(git -C "$src" rev-parse HEAD)}
library=$(realpath -e -- "${SEQUENCE_LIBRARY:-$src/libengine.so}")
checker=$(realpath -e -- "${SEQUENCE_CHECKER:-$src/sequence_check.pl}")
case_name=${SEQUENCE_CASE:-all}
wait_seconds=${SEQUENCE_WAIT_SECONDS:-600}
min_free=${SEQUENCE_MIN_FREE_MIB:-200}
[[ $wait_seconds =~ ^[0-9]+$ && $min_free =~ ^[0-9]+$ ]] || exit 2
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to match the holder manifests}"
: "${SPLIT_SHARES:?Set SPLIT_SHARES to the resident model split}"
: "${ATTN_MAP:?Set ATTN_MAP to the resident attention-layer mapping}"
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export FLASH_ANCHOR_DIE=${FLASH_ANCHOR_DIE:-7}
export FLASH_MAXT=${FLASH_MAXT:-1024}
export FLASH_ENGINE_MAXT=${FLASH_ENGINE_MAXT:-$FLASH_MAXT}
export FLASH_MAXIMG=${FLASH_MAXIMG:-2}
export FLASH_PLE_DISK=1 FLASH_PLE_OFF=${FLASH_PLE_OFF:-192}
export FLASH_SEQUENCES=2 FLASH_QUANTUM=${FLASH_QUANTUM:-1}
export FLASH_NBLK=${FLASH_NBLK:-512} FLASH_KV_HOST=${FLASH_KV_HOST:-0}
export FLASH_TWO_POSITION=0 FLASH_TRACE=0
export FLASH_PREFILL_CHUNK=${FLASH_PREFILL_CHUNK:-0}
export FLASH_WATCHDOG_S=${FLASH_WATCHDOG_S:-60}
export FLASH_CUBIN=$(realpath -e -- "${FLASH_CUBIN:-$src/die.cubin}")
run_dir=$(mktemp -d "${TMPDIR:-/tmp}/flash-sequence-XXXXXXXX")
export FLASH_SOCK=$run_dir/core.sock
export FLASH_SEQUENCE_TRACE=$out/turns.jsonl
core_pid=''
cleanup() {
    local status=$?
    trap - EXIT
    # Only this shell's unreaped child can occupy core_pid.
    if [[ -n $core_pid ]]; then
        kill -TERM "$core_pid" 2>/dev/null || true
        wait "$core_pid" 2>/dev/null || true
    fi
    rm -f -- "$FLASH_SOCK"
    rmdir -- "$run_dir" 2>/dev/null || true
    printf '%s\n' "$status" > "$out/exit-code.txt"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
jq -n --arg hostname "$(hostname)" --arg boot_id "$(cat /proc/sys/kernel/random/boot_id)" \
   --arg utc "$(date -u +%FT%TZ)" --arg revision "$revision" --arg case_name "$case_name" \
   '{hostname:$hostname,boot_id:$boot_id,started_utc:$utc,revision:$revision,case:$case_name}' > "$out/identity.json"
sha256sum -- "$src/flashcore" "$library" "$FLASH_CUBIN" "$checker" "$src/sequence_validate.sh" "${manifests[@]}" > "$out/sha256.txt"
for manifest in "${manifests[@]}"; do head -1 -- "$manifest"; done > "$out/holder-arenas.txt"
env | LC_ALL=C sort | awk '/^(CUDA_VISIBLE_DEVICES|CUDA_DEVICE_ORDER|SPLIT_SHARES|ATTN_DIE|ATTN_MAP|FLASH_(ANCHOR_DIE|MAXT|ENGINE_MAXT|MAXIMG|PLE_DISK|PLE_OFF|SEQUENCES|QUANTUM|NBLK|KV_HOST|TWO_POSITION|PREFILL_CHUNK|TRACE|WATCHDOG_S|CUBIN|SOCK|SEQUENCE_TRACE))=/' > "$out/environment.txt"
# All sequence workers use this lease; retain resident read-only holders/fences.
# This path is shared with the other repository workers. TMPDIR must not
# silently give this worker a different GPU lease.
if [[ -n ${SEQUENCE_LOCK_FD:-} ]]; then
    # A queue controller can retain the same lease across several matrices and
    # restore a temporarily stopped resident core before finally releasing it.
    [[ $SEQUENCE_LOCK_FD == 9 && /proc/$BASHPID/fd/9 -ef /tmp/flash-sass-engine-validation.lock ]] || {
        printf 'SEQUENCE_LOCK_FD must be inherited fd 9 for the shared lease.\n' >&2
        exit 2
    }
else
    exec 9>/tmp/flash-sass-engine-validation.lock
fi
printf '%s pid=%s waiting\n' "$(date -u +%FT%TZ)" "$$" > "$out/lease.log"
if ! flock -w "$wait_seconds" 9; then
    printf 'GPU validation lease is busy; no core started.\n' | tee "$out/blocked.txt"
    exit 75
fi
printf '%s pid=%s acquired\n' "$(date -u +%FT%TZ)" "$$" >> "$out/lease.log"
started_wait=$SECONDS
attempt=0
while :; do
    attempt=$((attempt+1))
    nvidia-smi --query-gpu=index,name,uuid,memory.free,utilization.gpu,clocks.sm --format=csv > "$out/gpu-before.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader,nounits > "$out/compute-before.csv"
    ps -eo pid,ppid,etime,comm > "$out/ps-before.txt"
    busy=0
    while IFS=, read -r uuid pid process_name; do
        [[ -z $uuid ]] && continue
        pid=${pid//[[:space:]]/}
        process_name=${process_name#"${process_name%%[![:space:]]*}"}
        [[ $pid =~ ^[0-9]+$ ]] || { printf 'Malformed compute-app record\n' >&2; exit 1; }
        case ${process_name##*/} in hold|fence) ;; *) busy=1 ;; esac
    done < "$out/compute-before.csv"
    if ! awk -F, -v minimum="$min_free" 'NR>1 {gsub(/[^0-9]/,"",$4);gsub(/[^0-9]/,"",$5);if($5+0>0 || ($1+0<6 && $4+0<minimum))bad=1} END{exit bad}' "$out/gpu-before.csv"; then busy=1; fi
    if awk 'NR>1 && ($4=="flashcore" || $4=="flashd" || $4=="nvprof" || $4=="ncu" || $4 ~ /_test$/){found=1} END{exit !found}' "$out/ps-before.txt"; then busy=1; fi
    printf '%s attempt=%d busy=%d\n' "$(date -u +%FT%TZ)" "$attempt" "$busy" >> "$out/wait.log"
    if (( !busy )); then break; fi
    if (( SECONDS-started_wait >= wait_seconds )); then
        printf 'Existing GPU process, activity, or insufficient free memory; no core started.\n' | tee "$out/blocked.txt"
        exit 75
    fi
    sleep 10
done
core_command=("$src/flashcore" "$ple" "$library" "${manifests[@]}")
printf '%q ' "${core_command[@]}" > "$out/core-command.sh"
printf '\n' >> "$out/core-command.sh"
"${core_command[@]}" > "$out/core.stdout" 2> "$out/core.stderr" &
core_pid=$!
printf '%s\n' "$core_pid" > "$out/owned-core.pid"
for ((i=0;i<600;i++)); do
    [[ -S $FLASH_SOCK ]] && break
    if ! kill -0 "$core_pid" 2>/dev/null; then
        printf 'Owned core exited before creating its socket.\n' >&2
        exit 1
    fi
    sleep 0.1
done
[[ -S $FLASH_SOCK ]] || { printf 'Owned core startup timeout.\n' >&2; exit 1; }
check_command=(perl "$checker" --case "$case_name" --revision "$revision" "$FLASH_SOCK" "$FLASH_SEQUENCE_TRACE" "$out/sequence-results.json")
printf '%q ' "${check_command[@]}" > "$out/check-command.sh"
printf '\n' >> "$out/check-command.sh"
set +e
timeout --signal=TERM --kill-after=10s "${SEQUENCE_TEST_TIMEOUT:-300}s" "${check_command[@]}" > "$out/check.stdout" 2> "$out/check.stderr"
check_status=$?
set -e
printf '%s\n' "$check_status" > "$out/check-exit-code.txt"
nvidia-smi --query-gpu=index,name,uuid,memory.free,utilization.gpu,clocks.sm --format=csv > "$out/gpu-after.csv"
ps -eo pid,ppid,etime,comm > "$out/ps-after.txt"
cat -- "$out/check.stdout"
cat -- "$out/check.stderr" >&2
exit "$check_status"
