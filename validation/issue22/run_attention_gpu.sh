#!/usr/bin/env bash
# Fixed-projection GPU fixtures; no original checkpoint or production indexer.
set -euo pipefail
if [[ $# != 5 ]]; then
    printf 'usage: %s GPU T BUDGET SNAPSHOT_LENGTHS NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
issue22_gpu=$1
issue22_total=$2
issue22_budget=$3
issue22_lengths=$4
issue22_out=$5
[[ $issue22_gpu =~ ^[0-9]+$ && $issue22_total =~ ^[0-9]+$ && $issue22_budget =~ ^[0-9]+$ ]] || exit 2
issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_revision=${ISSUE22_SOURCE_REVISION:-$(git -C "$issue22_repo" rev-parse HEAD)}
[[ $issue22_revision =~ ^[0-9a-f]{40}$ ]] || { printf 'missing full source revision\n' >&2; exit 2; }
[[ ! -e $issue22_out ]] || { printf 'output already exists: %s\n' "$issue22_out" >&2; exit 2; }
[[ -x $issue22_repo/kern/qsa_attention_gpu && -f $issue22_repo/kern/die.cubin ]] || {
    printf 'build qsa_attention_gpu and die.cubin first\n' >&2
    exit 2
}
mkdir -p -- "$(dirname -- "$issue22_out")"
mkdir -- "$issue22_out"
issue22_out=$(cd -- "$issue22_out" && pwd)
cd -- "$issue22_repo"
export LC_ALL=C CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$issue22_gpu
export DIE_CUBIN=$issue22_repo/kern/die.cubin
{
    date -u +%FT%TZ
    hostname
    uname -a
    printf 'source_revision=%s\nCUDA_VISIBLE_DEVICES=%s\nCUDA_DEVICE_ORDER=%s\n' \
        "$issue22_revision" "$CUDA_VISIBLE_DEVICES" "$CUDA_DEVICE_ORDER"
    printf 'command: '
    printf '%q ' "$issue22_repo/kern/qsa_attention_gpu" "$issue22_total" \
        "$issue22_budget" "$issue22_lengths" "$issue22_out/capture"
    printf '\n'
} > "$issue22_out/command.txt"
sha256sum kern/qsa_attention_gpu kern/qsa_attention_gpu.c kern/die.cubin kern/dieops.h \
    validation/issue22/attention-limits.json validation/issue22/run_attention_gpu.sh \
    > "$issue22_out/source-before.sha256"
exec 9>/tmp/flash-sass-engine-validation.lock
# Stay queued instead of repeatedly losing our position to other waiters.
# flock and this shell share the same open file description for descriptor 9.
flock -w 1800 -E 75 9 &
issue22_waiter=$!
trap 'if [[ -n $issue22_waiter ]]; then kill -TERM "$issue22_waiter" 2>/dev/null || true; fi; exec 9>&-' EXIT
issue22_waited=0
while kill -0 "$issue22_waiter" 2>/dev/null; do
    if (( issue22_waited % 180 == 0 )); then
        date -u '+%FT%TZ queued for shared validation lock' | tee -a "$issue22_out/lock.log"
        lslocks --noheadings --output PID,COMMAND,TYPE,MODE,PATH |
            rg 'flash-sass-engine-validation.lock' >> "$issue22_out/lock.log" || true
    fi
    sleep 5
    (( issue22_waited += 5 ))
done
set +e
wait "$issue22_waiter"
issue22_rc=$?
set -e
issue22_waiter=
if (( issue22_rc != 0 )); then
    printf 'shared validation lock acquisition ended with exit %s after up to 1800 seconds\n' "$issue22_rc" \
        | tee "$issue22_out/blocked.txt"
    exit "$issue22_rc"
fi
date -u '+%FT%TZ acquired shared validation lock' | tee -a "$issue22_out/lock.log"
nvidia-smi > "$issue22_out/gpu-before.txt"
nvidia-smi -i "$issue22_gpu" --query-gpu=index,uuid,pci.bus_id,name,memory.free,utilization.gpu \
    --format=csv,noheader,nounits > "$issue22_out/device-before.csv"
ps -eo pid,etime,args | rg '[f]lashcore|[h]old .*manifest|[f]ence|[q]sa_attention|flash-sass-engine-validation' \
    > "$issue22_out/processes-before.txt" || true
issue22_free=$(nvidia-smi -i "$issue22_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ $issue22_free =~ ^[0-9]+$ ]] || exit 2
if (( issue22_free < 3500 )); then
    printf 'GPU %s has %s MiB free; this fixture requires at least 3500 MiB; other processes left running\n' \
        "$issue22_gpu" "$issue22_free" | tee "$issue22_out/blocked.txt"
    exit 75
fi
set +e
timeout --signal=TERM --kill-after=15s 1200s \
    "$issue22_repo/kern/qsa_attention_gpu" "$issue22_total" "$issue22_budget" \
    "$issue22_lengths" "$issue22_out/capture" \
    > "$issue22_out/gpu.stdout" 2> "$issue22_out/gpu.stderr"
issue22_rc=$?
set -e
printf '%s\n' "$issue22_rc" > "$issue22_out/gpu.exit"
nvidia-smi > "$issue22_out/gpu-after.txt"
sha256sum kern/qsa_attention_gpu kern/qsa_attention_gpu.c kern/die.cubin kern/dieops.h \
    validation/issue22/attention-limits.json validation/issue22/run_attention_gpu.sh \
    > "$issue22_out/source-after.sha256"
cmp -- "$issue22_out/source-before.sha256" "$issue22_out/source-after.sha256"
(( issue22_rc == 0 )) || exit "$issue22_rc"
jq -e --arg revision "$issue22_revision" \
    '.schema == "flash-sass.qsa-attention-capture.v1" and .build_revision == $revision
     and .evidence_kind == "synthetic_gpu_fixture"
     and .production_sparse_selector_executed == false and .full_model == false' \
    "$issue22_out/capture/complete.json" > /dev/null
(
    cd -- "$issue22_out/capture"
    rg --files -0 | sort -z | xargs -0 sha256sum > "$issue22_out/capture.sha256"
)
jq -n --arg revision "$issue22_revision" --argjson gpu "$issue22_gpu" \
    --slurpfile capture "$issue22_out/capture/complete.json" \
    '{schema:"flash-sass.issue22-attention-gpu-run.v1",source_revision:$revision,
      gpu:$gpu,lock:"/tmp/flash-sass-engine-validation.lock",
      source_hashes_unchanged:true,capture_complete:true,comparison_performed:false,
      capture:$capture[0],evidence_kind:"synthetic_gpu_fixture",
      production_indexer_executed:false,full_model:false,
      original_checkpoint_quality:null,issue22_resolved:false}' > "$issue22_out/complete.json"
printf 'GPU attention fixtures complete; independent numerical comparison is separate.\n'
