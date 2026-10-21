#!/usr/bin/env bash
# Private six-die issue #12 benchmark. Caller retains the global lock on FD 9.
set -euo pipefail
[[ $# == 2 ]] || { echo "usage: $0 BUILD_ROOT NEW_RUN_DIRECTORY" >&2; exit 2; }
build=$1 out=$2
[[ $build = /* && $out = /* ]] || exit 2
[[ $(readlink /proc/$$/fd/9) == /tmp/flash-sass-engine-validation.lock ]] || exit 75
flock -n 9 || exit 75
mkdir "$out"
core_pid=
cleanup() {
    rc=$?
    trap - EXIT
    if [[ -n $core_pid ]] && kill -0 "$core_pid" 2>/dev/null; then
        kill -TERM "$core_pid" 2>/dev/null || :
        sleep 1
        kill -KILL "$core_pid" 2>/dev/null || :
        wait "$core_pid" 2>/dev/null || :
    fi
    printf '%s\n' "$rc" > "$out/runner.exit"
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
printf 'PREFLIGHT\n' > "$out/runner.status"
date -u +%FT%TZ > "$out/started.utc"
nvidia-smi --query-gpu=index,uuid,memory.free,utilization.gpu,utilization.memory --format=csv,noheader,nounits > "$out/preflight.gpu.csv"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$out/preflight.apps.csv"
awk -F, '
    {for(i=1;i<=NF;i++)gsub(/^[ \t]+|[ \t]+$/,"",$i)}
    NF!=5 || $1!~/^[0-9]+$/ || $3!~/^[0-9]+$/ || $4!~/^[0-9]+$/ || $5!~/^[0-9]+$/ {bad=1}
    $4!=0 || $5!=0 || ($1<6 && $3<160) {bad=1}
    END {exit bad || NR!=8}' "$out/preflight.gpu.csv" || { printf 'BLOCKED_ACTIVITY_OR_MEMORY\n' > "$out/runner.status"; exit 75; }
awk -F, '
    {for(i=1;i<=NF;i++)gsub(/^[ \t]+|[ \t]+$/,"",$i)}
    NF!=4 || $2!~/^[0-9]+$/ || $3!~/(^|\/)(hold|fence)$/ {bad=1}
    END{exit bad || NR!=7}' "$out/preflight.apps.csv" || { printf 'BLOCKED_FOREIGN_COMPUTE\n' > "$out/runner.status"; exit 75; }
awk -F, '{gsub(/[ \t]/,"",$2);print $2}' "$out/preflight.apps.csv" | sort -nu > "$out/resident.pids"
while read -r pid; do
    awk '{print $1,$2,$22}' "/proc/$pid/stat"
done < "$out/resident.pids" > "$out/resident-before.identities"
while IFS=, read -r index uuid rest; do
    uuid=${uuid//[[:space:]]/}
    [[ $index == 6 ]] && continue
    exec {gpu_fd}>>"/tmp/flash-sass-gpu-locks/$uuid.lock"
    flock -n "$gpu_fd" || { printf 'BLOCKED_DEVICE_LOCK\n' > "$out/runner.status"; exit 75; }
done < "$out/preflight.gpu.csv"
manifests=(/tmp/hold-h0.manifest /tmp/hold-h1.manifest /tmp/hold-h2.manifest /tmp/hold-h3.manifest /tmp/hold-h4.manifest /tmp/hold-h5.manifest)
for m in "${manifests[@]}"; do test -f "$m.ready"; done
sha256sum "${manifests[@]}" > "$out/manifests-before.sha256"
sha256sum "$0" "$build/matched-new/kern/flashcore" "$build/matched-new/kern/libengine_cta_remap.so" "$build"/{matched-old,matched-new,trace-old,trace-new}/kern/die.cubin > "$out/runtime.sha256"
ple=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
stat -c '%n %s %y' "$ple" > "$out/checkpoint.stat"
printf 'ctabench %s/measurements %s/matched-old/kern/die.cubin %s/matched-new/kern/die.cubin %s/trace-old/kern/die.cubin %s/trace-new/kern/die.cubin 2048 32 8\n' "$out" "$build" "$build" "$build" "$build" > "$out/commands.txt"
env -i PATH=/usr/bin:/bin \
    LD_BIND_NOW=1 CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    FLASH_ANCHOR_DIE=7 FLASH_MAXT=2080 FLASH_ENGINE_MAXT=2080 FLASH_MAXIMG=0 \
    FLASH_KV_HOST=0 FLASH_PLE_DISK=1 FLASH_PLE_OFF=192 FLASH_HOLD_WAIT_S=0 \
    FLASH_NBLK=512 FLASH_LAYERS=48 FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=0 \
    FLASH_WATCHDOG_S=60 SPLIT_SHARES=112,97,81,116,97,97 ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5 \
    FLASH_CUBIN="$build/matched-new/kern/die.cubin" \
    "$build/matched-new/kern/flashcore" "$ple" "$build/matched-new/kern/libengine_cta_remap.so" "${manifests[@]}" \
    < "$out/commands.txt" > "$out/core.stdout" 2> "$out/core.stderr" &
core_pid=$!
printf '%s\n' "$core_pid" > "$out/core.pid"
printf 'RUNNING\n' > "$out/runner.status"
start=$SECONDS
tick=0
while kill -0 "$core_pid" 2>/dev/null; do
    ((++tick))
    date -u +%FT%TZ > "$out/telemetry-$tick.utc"
    nvidia-smi --query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem,memory.free,utilization.gpu,utilization.memory,power.draw --format=csv,noheader,nounits > "$out/telemetry-$tick.gpu.csv"
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$out/telemetry-$tick.apps.csv"
    awk -F, -v own="$core_pid" 'NR==FNR{known[$1]=1;next}
        {for(i=1;i<=NF;i++)gsub(/^[ \t]+|[ \t]+$/,"",$i)}
        NF!=4 || $2!~/^[0-9]+$/ || ($2!=own && !known[$2]) {bad=1}
        END{exit bad}' "$out/resident.pids" "$out/telemetry-$tick.apps.csv" || {
            printf 'INVALID_FOREIGN_COMPUTE\n' > "$out/runner.status"; exit 1;
        }
    nvidia-smi pmon -c 1 -s u > "$out/telemetry-$tick.pmon.txt"
    awk -v own="$core_pid" '
        /^#/ || NF==0 {next}
        NF!=8 {bad=1;next}
        $2~/^[0-9]+$/ && $2!=own && (($4~/^[0-9]+$/ && $4>0) || ($5~/^[0-9]+$/ && $5>0)) {bad=1}
        END{exit bad}' "$out/telemetry-$tick.pmon.txt" || {
            printf 'INVALID_FOREIGN_ACTIVITY\n' > "$out/runner.status"; exit 1;
        }
    if ((SECONDS-start>900)); then printf 'CORE_TIMEOUT\n' > "$out/runner.status"; exit 124; fi
done
set +e
wait "$core_pid"; rc=$?
set -e
printf '%s\n' "$rc" > "$out/core.exit"
core_pid=
[[ $rc == 0 && $(cat "$out/measurements/status.txt") == PASS ]] || {
    printf 'BENCHMARK_FAILED\n' > "$out/runner.status"; exit 1;
}
sha256sum -c "$out/manifests-before.sha256" > "$out/manifests-after.check"
while read -r pid; do
    awk '{print $1,$2,$22}' "/proc/$pid/stat"
done < "$out/resident.pids" > "$out/resident-after.identities"
cmp "$out/resident-before.identities" "$out/resident-after.identities"
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader,nounits > "$out/after.apps.csv"
date -u +%FT%TZ > "$out/finished.utc"
printf 'PASS\n' > "$out/runner.status"
printf 'PASS private six-die benchmark; resident holders unchanged\n'
