#!/usr/bin/env bash
# Serialize a target-GPU run with other flash-sass validators and wait for an idle host.
# Exit 75 means no run was started because the GPU host remained busy.
# Usage: qsa_oracle_gpu_gate.sh --out NEW_DIR [--wait SECONDS] [--check | -- COMMAND...]
set -euo pipefail
out=
wait_seconds=0
check_only=0
while (($#)); do
    case "$1" in
        --out) out=${2:?missing output directory}; shift 2 ;;
        --wait) wait_seconds=${2:?missing wait duration}; shift 2 ;;
        --check) check_only=1; shift ;;
        --) shift; break ;;
        *) printf 'unknown option: %s\n' "$1" >&2; exit 2 ;;
    esac
done
[[ -n "$out" && "$wait_seconds" =~ ^[0-9]+$ ]] || { printf 'need --out NEW_DIR and nonnegative --wait\n' >&2; exit 2; }
((wait_seconds <= 86400)) || { printf 'wait is limited to 86400 seconds\n' >&2; exit 2; }
if ((check_only)); then
    (($# == 0)) || { printf '--check does not accept a command\n' >&2; exit 2; }
else
    (($# > 0)) || { printf 'need -- COMMAND... or --check\n' >&2; exit 2; }
fi
command -v nvidia-smi >/dev/null
command -v flock >/dev/null
command -v ps >/dev/null
mkdir -- "$out"
printf '%s\n' "$(hostname)" > "$out/hostname.txt"
date -u +%FT%TZ > "$out/start.txt"
printf '%q ' "$@" > "$out/command.txt"
printf '\n' >> "$out/command.txt"
# This lock is shared with the repository's other engine validators. It never
# authorizes stopping resident holders, servers, or another worker's process.
exec 9>/tmp/flash-sass-engine-validation.lock
start=$SECONDS
attempt=0
while :; do
    attempt=$((attempt + 1))
    stem="$out/check-$attempt"
    date -u +%FT%TZ > "$stem.time.txt"
    nvidia-smi > "$stem.nvidia-smi.txt" 2>&1 || { printf 'nvidia-smi failed; idle state unknown\n' >&2; exit 2; }
    nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits > "$stem.compute.csv" 2>&1 || { printf 'compute process query failed; idle state unknown\n' >&2; exit 2; }
    ps -eo pid=,ppid=,comm=,args= > "$stem.ps.txt"
    # The compute table catches all CUDA users. ps also catches the known
    # holders/tests during startup, before CUDA registers their context.
    awk '$3 ~ /^(hold|fence|flashcore|flashd|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexity)$/ {print}' "$stem.ps.txt" > "$stem.gpu-processes.txt"
    if [[ ! -s "$stem.compute.csv" && ! -s "$stem.gpu-processes.txt" ]] && flock -n 9; then
        # Recheck after acquiring the cooperative lock to close the normal
        # contention race between two validators using this wrapper.
        nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory --format=csv,noheader,nounits > "$stem.locked-compute.csv" 2>&1 || exit 2
        ps -eo pid=,ppid=,comm=,args= > "$stem.locked-ps.txt"
        awk '$3 ~ /^(hold|fence|flashcore|flashd|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexity)$/ {print}' "$stem.locked-ps.txt" > "$stem.locked-gpu-processes.txt"
        if [[ ! -s "$stem.locked-compute.csv" && ! -s "$stem.locked-gpu-processes.txt" ]]; then
            break
        fi
        flock -u 9
    fi
    if ((SECONDS - start >= wait_seconds)); then
        printf 'BLOCKED_BUSY: no GPU command started; see %s\n' "$out" | tee "$out/status.txt"
        printf '75\n' > "$out/exit-code.txt"
        exit 75
    fi
    printf 'WAIT_BUSY attempt=%s elapsed=%s\n' "$attempt" "$((SECONDS - start))" >> "$out/wait.log"
    sleep 10
done
if ((check_only)); then
    printf 'IDLE_CHECK_ONLY: no GPU command started\n' | tee "$out/status.txt"
    printf '0\n' > "$out/exit-code.txt"
    exit 0
fi
printf 'RUNNING\n' > "$out/status.txt"
set +e
QSA_ORACLE_GPU_GUARDED=1 "$@" > "$out/run.stdout" 2> "$out/run.stderr"
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
