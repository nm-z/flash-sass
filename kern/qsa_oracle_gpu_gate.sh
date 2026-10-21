#!/usr/bin/env bash
# Queue behind the shared engine lock, then check the selected execution scope.
# A selected physical GPU is exposed to the child as CUDA ordinal 0. Without
# --device, preserve the conservative whole-host idle requirement.
# Exit 75 means no GPU command started before the wait deadline.
set -euo pipefail
out=
wait_seconds=0
run_seconds=900
check_only=0
device=
while (($#)); do
    case "$1" in
        --out) out=${2:?missing output directory}; shift 2 ;;
        --wait) wait_seconds=${2:?missing wait duration}; shift 2 ;;
        --run-timeout) run_seconds=${2:?missing run deadline}; shift 2 ;;
        --device) device=${2:?missing physical GPU ordinal}; shift 2 ;;
        --check) check_only=1; shift ;;
        --) shift; break ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
# Bound decimal strings before arithmetic, including leading-zero inputs.
[[ -n "$out" && "$wait_seconds" =~ ^[0-9]{1,5}$ && "$run_seconds" =~ ^[0-9]{1,5}$ ]] || {
    printf 'need --out NEW_DIR and bounded decimal --wait/--run-timeout\n' >&2; exit 2;
}
wait_seconds=$((10#$wait_seconds))
run_seconds=$((10#$run_seconds))
((wait_seconds <= 86400 && run_seconds >= 1 && run_seconds <= 86400)) || {
    printf 'wait must be 0..86400; run timeout must be 1..86400 seconds\n' >&2; exit 2;
}
if [[ -n "$device" ]]; then
    [[ "$device" =~ ^[0-9]{1,3}$ ]] || { printf 'device must be a physical ordinal\n' >&2; exit 2; }
    device=$((10#$device))
fi
if ((check_only)); then
    (($# == 0)) || { printf '--check does not accept a command\n' >&2; exit 2; }
else
    (($# > 0)) || { printf 'need -- COMMAND... or --check\n' >&2; exit 2; }
fi
for tool in nvidia-smi flock ps timeout; do command -v "$tool" >/dev/null; done
mkdir -- "$out"
printf '%s\n' "$(hostname)" > "$out/hostname.txt"
date -u +%FT%TZ > "$out/start.txt"
printf '%q ' "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
uuid=
if [[ -n "$device" ]]; then
    uuid=$(nvidia-smi -i "$device" --query-gpu=uuid --format=csv,noheader,nounits) || exit 2
    [[ "$uuid" =~ ^GPU-[0-9a-fA-F-]+$ ]] || { printf 'could not resolve one physical GPU UUID\n' >&2; exit 2; }
    printf 'physical_ordinal=%s\nuuid=%s\nchild_cuda_ordinal=0\n' "$device" "$uuid" > "$out/device.txt"
else
    printf 'whole_host\n' > "$out/device.txt"
fi
# The same inode is used by all validators. Never replace/remove the lock file,
# or release another process's lock. A blocking flock registers a real waiter
# instead of repeatedly racing other workers with nonblocking acquisition.
exec 9>/tmp/flash-sass-engine-validation.lock
start=$SECONDS
attempt=0
while :; do
    attempt=$((attempt + 1))
    elapsed=$((SECONDS - start))
    remaining=$((wait_seconds - elapsed))
    ((remaining >= 0)) || remaining=0
    printf 'QUEUED attempt=%s elapsed=%s wait_remaining=%s\n' "$attempt" "$elapsed" "$remaining" | tee -a "$out/wait.log"
    set +e
    flock -w "$remaining" -E 75 9
    lock_rc=$?
    set -e
    if ((lock_rc != 0)); then
        if ((lock_rc == 75)); then
            printf 'BLOCKED_LOCK: queued wait expired; no GPU command started\n' | tee "$out/status.txt"
        else
            printf 'LOCK_ERROR exit=%s: no GPU command started\n' "$lock_rc" | tee "$out/status.txt"
        fi
        printf '%s\n' "$lock_rc" > "$out/exit-code.txt"
        exit "$lock_rc"
    fi
    stem="$out/check-$attempt"
    date -u +%FT%TZ > "$stem.time.txt"
    nvidia-smi > "$stem.nvidia-smi.txt" 2>&1 || { printf 'nvidia-smi failed; idle state unknown\n' >&2; exit 2; }
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits > "$stem.compute.csv" 2>&1 || { printf 'compute query failed; idle state unknown\n' >&2; exit 2; }
    ps -eo pid=,ppid=,comm=,args= > "$stem.ps.txt"
    if [[ -n "$uuid" ]]; then
        # CUDA users on other devices do not occupy this isolated ordinal.
        # The global lock still excludes cooperating startup/test races.
        awk -F, -v uuid="$uuid" '{id=$3; gsub(/^[ \t]+|[ \t]+$/, "", id); if(id==uuid) print}' "$stem.compute.csv" > "$stem.scope-compute.csv"
        # Re-resolve under the lock to reject ordinal reordering/hot removal.
        current_uuid=$(nvidia-smi -i "$device" --query-gpu=uuid --format=csv,noheader,nounits) || exit 2
        [[ "$current_uuid" == "$uuid" ]] || { printf 'GPU ordinal identity changed\n' >&2; exit 2; }
        busy=0
        [[ ! -s "$stem.scope-compute.csv" ]] || busy=1
    else
        awk '$3 ~ /^(hold|fence|flashcore|flashd|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexity)$/ {print}' "$stem.ps.txt" > "$stem.gpu-processes.txt"
        busy=0
        [[ ! -s "$stem.compute.csv" && ! -s "$stem.gpu-processes.txt" ]] || busy=1
    fi
    if ((busy == 0)); then break; fi
    flock -u 9
    elapsed=$((SECONDS - start))
    if ((elapsed >= wait_seconds)); then
        printf 'BLOCKED_BUSY: execution scope occupied; no GPU command started\n' | tee "$out/status.txt"
        printf '75\n' > "$out/exit-code.txt"
        exit 75
    fi
    printf 'WAIT_BUSY attempt=%s elapsed=%s\n' "$attempt" "$elapsed" | tee -a "$out/wait.log"
    delay=$((wait_seconds - elapsed)); ((delay <= 10)) || delay=10
    sleep "$delay"
done
if ((check_only)); then
    printf 'IDLE_CHECK_ONLY: no GPU command started\n' | tee "$out/status.txt"
    printf '0\n' > "$out/exit-code.txt"
    exit 0
fi
printf 'RUNNING\n' | tee "$out/status.txt"
set +e
if [[ -n "$uuid" ]]; then
    CUDA_VISIBLE_DEVICES="$uuid" QSA_ORACLE_GPU_GUARDED=1 timeout --signal=TERM --kill-after=15 "$run_seconds" "$@" > "$out/run.stdout" 2> "$out/run.stderr"
else
    QSA_ORACLE_GPU_GUARDED=1 timeout --signal=TERM --kill-after=15 "$run_seconds" "$@" > "$out/run.stdout" 2> "$out/run.stderr"
fi
run_rc=$?
set -e
printf '%s\n' "$run_rc" > "$out/exit-code.txt"
date -u +%FT%TZ > "$out/end.txt"
if ((run_rc == 0)); then
    printf 'COMMAND_PASSED\n' | tee "$out/status.txt"
else
    printf 'COMMAND_FAILED exit=%s\n' "$run_rc" | tee "$out/status.txt"
fi
exit "$run_rc"
