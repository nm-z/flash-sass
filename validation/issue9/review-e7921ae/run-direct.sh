#!/usr/bin/env bash
set -euo pipefail
source_root=$1
out=$2
gpu_uuid=$3
gpu_lock=$4
source_revision=$5
mkdir -p "$out"
exec 9>>/tmp/flash-sass-engine-validation.lock
exec 8>>"$gpu_lock"
deadline=$((SECONDS + 1200))
while :; do
    if flock -w 30 9; then
        if flock -n 8; then
            nvidia-smi -i "$gpu_uuid" --query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu --format=csv > "$out/preflight-gpu.csv"
            nvidia-smi -i "$gpu_uuid" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader > "$out/preflight-compute.csv"
            ps -eo pid,ppid,etimes,stat,comm > "$out/preflight-ps.txt"
            free_mib=$(awk -F, 'NR == 2 {gsub(/[^0-9]/,"",$5); print $5}' "$out/preflight-gpu.csv")
            utilization=$(awk -F, 'NR == 2 {gsub(/[^0-9]/,"",$7); print $7}' "$out/preflight-gpu.csv")
            if [[ ! -s "$out/preflight-compute.csv" && "$utilization" == 0 && "$free_mib" -ge 2500 ]]; then
                break
            fi
            flock -u 8
        fi
        flock -u 9
    fi
    printf 'Waiting for shared validation lock and idle %s\n' "$gpu_uuid"
    if (( SECONDS >= deadline )); then printf 'GPU unavailable within wait window\n' >&2; exit 75; fi
    sleep 5
 done
trap 'flock -u 8; flock -u 9' EXIT
cd "$source_root/kern"
printf '%s\n' "$source_revision" > "$out/source-revision.txt"
sha256sum qsa_direct_test die.cubin libengine.so model_test hold_cache > "$out/binaries.sha256"
sha256sum qsa_direct_test.c qsa_direct_projection.h qsa_direct_fixtures.h qsa_test.c prog.c prog.h dieops.h die.pl ../ref/qsa.c ../ref/qsa.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/attn.h ../ref/fmath.c > "$out/component-sources.sha256"
export CUDA_VISIBLE_DEVICES="$gpu_uuid"
export DIE_CUBIN="$source_root/kern/die.cubin"
set +e
perl run_qsa_direct_matrix.pl --binary "$source_root/kern/qsa_direct_test" --out "$out/matrix" --timeout 240 > "$out/matrix-console.log" 2>&1
status=$?
set -e
printf '%d\n' "$status" > "$out/exit-code.txt"
nvidia-smi -i "$gpu_uuid" --query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu --format=csv > "$out/after-gpu.csv"
ps -eo pid,ppid,etimes,stat,comm > "$out/after-ps.txt"
cat "$out/matrix-console.log"
exit "$status"
