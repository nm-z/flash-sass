#!/usr/bin/env bash
# Capture serial/parallel row-I/O evidence without disturbing another worker.
set -euo pipefail
if (( $# < 3 || $# > 7 )); then
    printf 'usage: %s NEW_OUT_DIR LOCAL_SHARD NFS_SHARD [OFFSET=192] [BATCHES=128] [SEED=20261017] [READERS=16]\n' "$0" >&2
    exit 2
fi
out=$1
local_shard=$2
nfs_shard=$3
offset=${4:-192}
batches=${5:-128}
seed=${6:-20261017}
readers=${7:-16}
bench_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
for needed in perl jq flock nvidia-smi ps stat sha256sum; do
    command -v "$needed" >/dev/null || { printf 'missing command: %s\n' "$needed" >&2; exit 2; }
done
[[ -x "$bench_dir/ple_read_bench" && -x "$bench_dir/ple_read_audit" ]] || {
    printf 'build ple_read_bench and ple_read_audit first\n' >&2
    exit 2
}
[[ ! -e "$out" ]] || { printf 'output already exists: %s\n' "$out" >&2; exit 2; }
mkdir -m 700 -- "$out"
out=$(CDPATH= cd -- "$out" && pwd)
status=preparing
finish() {
    local code=$?
    trap - EXIT
    jq -n --arg status "$status" --argjson exit "$code" \
        '{status:$status,exit:$exit}' > "$out/exit.json"
    if [[ -e "$out/lock-acquired" ]]; then flock -u 9 || :; fi
    exit "$code"
}
trap finish EXIT
jq -n --args '$ARGS.positional' "$0" "$@" > "$out/argv.json"
date -u +%Y-%m-%dT%H:%M:%SZ > "$out/started.utc"
hostname > "$out/hostname.txt"
exec 9>/tmp/flash-sass-engine-validation.lock
if ! flock -n 9; then
    status=busy
    printf 'shared validation lock is held; no benchmark started\n' >&2
    exit 75
fi
: > "$out/lock-acquired"
check_idle() {
    local stage=$1
nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used,memory.total \
    --format=csv,noheader,nounits > "$out/$stage-gpus.csv"
nvidia-smi --query-compute-apps=pid,process_name \
    --format=csv,noheader,nounits > "$out/$stage-compute.csv"
ps -eo pid=,ppid=,stat=,comm= > "$out/$stage-processes.txt"
if [[ -s "$out/$stage-compute.csv" ]] ||
    ! awk -F, 'BEGIN {good=1; rows=0} {rows++; gsub(/^[ \t]+|[ \t]+$/, "", $3); if ($3 !~ /^[0-9]+$/ || $3+0 != 0) good=0} END {exit !(good && rows>0)}' "$out/$stage-gpus.csv" ||
    awk '$4 ~ /^(dd|fio|ple_read_bench|ple_read_audit|ioping|bonnie\+\+|rsync)$/ {busy=1} END {exit !busy}' "$out/$stage-processes.txt"; then
    status=busy
    printf 'other GPU or disk benchmark work is present; no benchmark started\n' >&2
    exit 75
fi
}
check_idle initial
nvidia-smi -q -x > "$out/nvidia-before.xml"
git -C "$bench_dir/.." rev-parse HEAD > "$out/source-sha.txt"
git -C "$bench_dir/.." status --porcelain=v1 > "$out/source-status.txt"
git -C "$bench_dir/.." diff --binary HEAD > "$out/source.diff"
sha256sum "$bench_dir/ple_read_bench" "$bench_dir/ple_read_audit" > "$out/binaries.sha256"
[[ $(stat -Lc %s -- "$local_shard") == $(stat -Lc %s -- "$nfs_shard") ]] || {
    status=failed
    printf 'local and NFS shards must have the same length for the same seeded row sets\n' >&2
    exit 2
}
for placement in local nfs; do
    if [[ "$placement" == local ]]; then shard=$local_shard; else shard=$nfs_shard; fi
    [[ -f "$shard" && -r "$shard" ]] || { printf 'unreadable shard: %s\n' "$shard" >&2; exit 2; }
    stat -Lc 'device=%d inode=%i bytes=%s mtime=%y ctime=%z' -- "$shard" > "$out/$placement-file-before.txt"
    stat -f -c 'filesystem=%T' -- "$shard" > "$out/$placement-filesystem.txt"
    jq -n --args '$ARGS.positional' "$bench_dir/ple_read_bench" "$shard" "$offset" "$batches" "$seed" "$readers" > "$out/$placement-command.json"
    check_idle "$placement"
    status=benchmarking
    set +e
    "$bench_dir/ple_read_bench" "$shard" "$offset" "$batches" "$seed" "$readers" \
        > "$out/$placement.csv" 2> "$out/$placement.stderr"
    result=$?
    set -e
    jq -n --argjson exit "$result" '{exit:$exit}' > "$out/$placement.exit.json"
    (( result == 0 )) || { status=failed; exit "$result"; }
    stat -Lc 'device=%d inode=%i bytes=%s mtime=%y ctime=%z' -- "$shard" > "$out/$placement-file-after.txt"
    cmp -s "$out/$placement-file-before.txt" "$out/$placement-file-after.txt" || {
        status=failed
        printf 'file identity changed during %s measurement\n' "$placement" >&2
        exit 1
    }
done
# Require identical sampled inputs across storage, complete valid records, and
# exact-input warm pairs before computing any latency summary.
set +e
perl "$bench_dir/ple_io_validate.pl" "$out/local.csv" "$out/nfs.csv" "$batches" "$readers" \
    > "$out/summary.json" 2> "$out/validation.stderr"
result=$?
set -e
jq -n --argjson exit "$result" '{exit:$exit}' > "$out/validation.exit.json"
(( result == 0 )) || { status=failed; exit "$result"; }
# Audit is a separate instrumented experiment, never an input to throughput totals.
for placement in local nfs; do
    if [[ "$placement" == local ]]; then shard=$local_shard; else shard=$nfs_shard; fi
    check_idle "$placement-audit"
    jq -n --args '$ARGS.positional' "$bench_dir/ple_read_audit" "$shard" "$offset" "1" "$seed" "$readers" > "$out/$placement-audit-command.json"
    set +e
    "$bench_dir/ple_read_audit" "$shard" "$offset" 1 "$seed" "$readers" \
        > "$out/$placement-audit.csv" 2> "$out/$placement-audit.stderr"
    result=$?
    set -e
    jq -n --argjson exit "$result" '{exit:$exit}' > "$out/$placement-audit.exit.json"
    (( result == 0 )) || { status=failed; exit "$result"; }
done
nvidia-smi -q -x > "$out/nvidia-after.xml"
ps -eo pid=,ppid=,stat=,comm= > "$out/processes-after.txt"
status=complete
printf 'Raw row-I/O evidence: %s\n' "$out"
