#!/usr/bin/env bash
# Isolated component evidence; caller holds shared host lock on inherited FD 9.
set -u -o pipefail
[[ $# == 3 ]] || { echo "usage: $0 BUILD_ROOT NEW_RESULTS GPU_UUID" >&2; exit 2; }
build=$1 out=$2 gpu=$3
[[ $build = /* && $out = /* && $gpu =~ ^GPU-[0-9a-f-]+$ ]] || exit 2
[[ $(readlink /proc/$$/fd/9) == /tmp/flash-sass-engine-validation.lock ]] || exit 75
flock -n 9 || exit 75
mkdir "$out" || exit
mkdir -p /tmp/flash-sass-gpu-locks
exec 8>>"/tmp/flash-sass-gpu-locks/$gpu.lock"
flock -n 8 || { printf 'BLOCKED_GPU_LOCK\n' > "$out/status.txt"; exit 75; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
unset ATTN_TRACE_CHECK ATTN_TRACE_CTAS
harness="$build/matched-new/kern/attnpartn_regress"
keys=${KEYS:-1,1023,1024,1025,2047,2048,2049,2050,2051,2052,4095,4096,4097,8191,8192,8193}
seeds=${SEEDS:-0,77,4242}
repeats=${REPEATS:-32}
modes=${MODES:-historical,matched,traced}
[[ $keys =~ ^[0-9]+(,[0-9]+)*$ && $seeds =~ ^[0-9]+(,[0-9]+)*$ &&
   $repeats =~ ^[0-9]+$ && $modes =~ ^(historical|matched|traced)(,(historical|matched|traced))*$ ]] || exit 2
live_pid=
cleanup() {
    if [[ -n $live_pid ]] && kill -0 "$live_pid" 2>/dev/null; then
        kill -TERM "$live_pid" 2>/dev/null || :
        sleep 1
        kill -KILL "$live_pid" 2>/dev/null || :
        wait "$live_pid" 2>/dev/null || :
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf 'RUNNING\n' > "$out/status.txt"
jq -n --arg utc "$(date -u +%FT%TZ)" --arg host "$(hostname)" --arg gpu "$gpu" \
 --arg build "$build" --arg keys "$keys" --arg seeds "$seeds" --arg modes "$modes" --argjson repeats "$repeats" \
 --arg harness_source "$(cat "$build/matched-new/source-commit.txt")" \
 --arg host_lock /tmp/flash-sass-engine-validation.lock --argjson lock_shell_pid "$PPID" \
 '{utc:$utc,host:$host,gpu:$gpu,build:$build,keys:$keys,seeds:$seeds,modes:$modes,repeats:$repeats,
   harness_source:$harness_source,host_lock:$host_lock,lock_shell_pid:$lock_shell_pid,
   scope:"isolated single-GPU component; no resident-model IPC or reload",
   monitoring:"one-second samples; selected GPU permits only current harness PID; other GPUs must report zero utilization; samples may miss sub-second interference",
   whole_model:false}' > "$out/manifest.json" || exit
sha256sum "$0" "$harness" > "$out/runner-harness.sha256"
cp "$build/"*.sha256 "$out/"
seq=0
snapshot() {
    local tag=$1 allowed_pid=$2 initial=${3:-0}
    local prefix="$out/telemetry-$tag"
    date -u +%FT%TZ > "$prefix.utc"
    nvidia-smi --query-gpu=uuid,name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,power.draw --format=csv,noheader,nounits > "$prefix.gpu.csv" 2> "$prefix.stderr" || return 1
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$prefix.apps.csv" 2>> "$prefix.stderr" || return 1
    awk -F, -v gpu="$gpu" -v initial="$initial" '
      {for(i=1;i<=NF;i++)gsub(/^[ \t]+|[ \t]+$/,"",$i)}
      NF!=9 || $7!~/^[0-9]+$/ {bad=1}
      $1==gpu {seen++}
      ($1!=gpu || initial) && $7!=0 {bad=1}
      END {exit bad || seen!=1}' "$prefix.gpu.csv" || return 1
    awk -F, -v gpu="$gpu" -v allowed="$allowed_pid" '
      NF==0 {next}
      {for(i=1;i<=NF;i++)gsub(/^[ \t]+|[ \t]+$/,"",$i)}
      NF!=4 || $2!~/^[0-9]+$/ {bad=1}
      $1==gpu && ($2!=allowed || allowed==0) {bad=1}
      END {exit bad}' "$prefix.apps.csv"
}
snapshot initial 0 1 || { printf 'BLOCKED_ACTIVITY\n' > "$out/status.txt"; exit 75; }
IFS=, read -ra all_modes <<< "$modes"
IFS=, read -ra all_keys <<< "$keys"
IFS=, read -ra all_seeds <<< "$seeds"
for mode in "${all_modes[@]}"; do
    trace=0
    case "$mode" in
        historical) old=historical-old; new=historical-new;;
        matched) old=matched-old; new=matched-new;;
        traced) old=trace-old; new=trace-new; trace=1;;
    esac
    for n in "${all_keys[@]}"; do for seed in "${all_seeds[@]}"; do
        ((++seq))
        case_name="$mode-keys$n-seed$seed"
        snapshot "$case_name-before" 0 || { printf 'BLOCKED_ACTIVITY\n' > "$out/status.txt"; exit 75; }
        start=$SECONDS
        ATTN_TRACE_CHECK=$trace "$harness" "$build/$old/kern/die.cubin" "$build/$new/kern/die.cubin" "$n" "$seed" "$repeats" \
            > "$out/$case_name.json" 2> "$out/$case_name.stderr" &
        live_pid=$!
        printf '%s\t%s\n' "$case_name" "$live_pid" >> "$out/children.tsv"
        tick=0
        while kill -0 "$live_pid" 2>/dev/null; do
            ((++tick))
            if ! snapshot "$case_name-$tick" "$live_pid"; then
                printf 'INVALID_CONCURRENT_ACTIVITY\n' > "$out/status.txt"; exit 1
            fi
            if ((SECONDS-start>=180)); then
                printf 'HARNESS_TIMEOUT\n' > "$out/status.txt"; exit 124
            fi
            sleep 1
        done
        wait "$live_pid"; rc=$?
        live_pid=
        printf '%s\t%s\n' "$case_name" "$rc" >> "$out/exits.tsv"
        ((rc==0)) || { printf 'HARNESS_FAILED\n' > "$out/status.txt"; exit "$rc"; }
        snapshot "$case_name-after" 0 || { printf 'INVALID_CONCURRENT_ACTIVITY\n' > "$out/status.txt"; exit 1; }
        jq -e --argjson n "$n" --argjson seed "$seed" --argjson trace "$trace" '
          .keys==$n and .seed==$seed and .bit_exact==true and .boundary_canaries==true and
          .boundary_canary_words==64 and .partial_values==49536 and .final_values==6144 and
          .steps_per_sample==64 and .old.event.median_us>0 and .new.event.median_us>0 and
          .trace_verified==($trace==1) and .timing_trace_enabled==($trace==1) and
          (if $trace==1 then .trace_launches>0 and .trace_records>0 else .trace_launches==0 end)' \
          "$out/$case_name.json" > /dev/null || { printf 'BAD_COVERAGE\n' > "$out/status.txt"; exit 1; }
        printf 'PASS %s\n' "$case_name"
    done; done
done
# This must fail because the actual target kernels do not write trace records.
# Loader failures and arbitrary nonzero exits cannot satisfy the negative check.
snapshot rejection-before 0 || { printf 'BLOCKED_ACTIVITY\n' > "$out/status.txt"; exit 75; }
start=$SECONDS
ATTN_TRACE_CHECK=1 "$harness" "$build/matched-old/kern/die.cubin" "$build/matched-new/kern/die.cubin" 2051 77 4 \
    > "$out/uninstrumented-rejection.stdout" 2> "$out/uninstrumented-rejection.stderr" &
live_pid=$!
printf 'uninstrumented-rejection\t%s\n' "$live_pid" >> "$out/children.tsv"
tick=0
while kill -0 "$live_pid" 2>/dev/null; do
    ((++tick))
    snapshot "rejection-$tick" "$live_pid" || { printf 'INVALID_CONCURRENT_ACTIVITY\n' > "$out/status.txt"; exit 1; }
    if ((SECONDS-start>=180)); then printf 'HARNESS_TIMEOUT\n' > "$out/status.txt"; exit 124; fi
    sleep 1
done
wait "$live_pid"; rc=$?
live_pid=
printf 'uninstrumented-rejection\t%s\n' "$rc" >> "$out/exits.tsv"
[[ $rc == 1 ]] && rg -q '^trace rejected variant=' "$out/uninstrumented-rejection.stderr" &&
    ! rg -q 'undefined symbol|CUDA_ERROR|->' "$out/uninstrumented-rejection.stderr" || {
    printf 'BAD_REJECTION\n' > "$out/status.txt"; exit 1;
}
snapshot rejection-after 0 || { printf 'INVALID_CONCURRENT_ACTIVITY\n' > "$out/status.txt"; exit 1; }
printf 'PASS target-GPU uninstrumented-image rejection\n'
printf 'PASS\n' > "$out/status.txt"
printf 'PASS %s component cases; whole_model=false\n' "$seq"
