#!/usr/bin/env bash
# Host-only guard regression: all GPU and benchmark commands are inert stubs.
set -euo pipefail
source_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
test_dir=$(mktemp -d)
trap 'rm -rf -- "$test_dir"' EXIT
mkdir "$test_dir/bin" "$test_dir/repo" "$test_dir/repo/kern"
cp "$source_dir/ple_io_run.sh" "$test_dir/repo/kern/"
export PLE_IO_TEST_LOG="$test_dir/calls"
: > "$PLE_IO_TEST_LOG"
cat > "$test_dir/bin/observations" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
case "$(basename -- "$0")" in
nvidia-smi)
    printf 'gpu_query\n' >> "$PLE_IO_TEST_LOG"
    if [[ " $* " == *" --query-compute-apps="* ]]; then
        [[ ${PLE_IO_TEST_BUSY:-0} == 0 ]] || printf '12345, another-worker\n'
    elif [[ " $* " == *" --query-gpu="* ]]; then
        printf '0, GPU-fixture, %s, 0, 1024\n' "${PLE_IO_TEST_UTIL:-0}"
    else
        printf '<nvidia_smi_log><gpu id="fixture"/></nvidia_smi_log>\n'
    fi ;;
ps)
    printf 'process_query\n' >> "$PLE_IO_TEST_LOG"
    if [[ ${PLE_IO_TEST_DISK:-0} != 0 ]]; then printf '12345 1 S fio\n'; else printf '12345 1 S fixture\n'; fi ;;
git)
    if [[ " $* " == *" rev-parse HEAD "* ]]; then printf '0000000000000000000000000000000000000001\n'; fi ;;
esac
STUB
chmod +x "$test_dir/bin/observations"
for tool in nvidia-smi ps git; do ln -s observations "$test_dir/bin/$tool"; done
cat > "$test_dir/repo/kern/ple_read_bench" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'benchmark %s %s\n' "$(basename -- "$0")" "$*" >> "$PLE_IO_TEST_LOG"
printf 'fixture-only\n'
exit "${PLE_IO_TEST_RESULT:-0}"
STUB
chmod +x "$test_dir/repo/kern/ple_read_bench"
cp "$test_dir/repo/kern/ple_read_bench" "$test_dir/repo/kern/ple_read_audit"
printf 'same fixture bytes\n' > "$test_dir/local shard"
cp "$test_dir/local shard" "$test_dir/nfs shard"
# The parser has its own malformed-record suite; this tests propagation only.
cat > "$test_dir/repo/kern/ple_io_validate.pl" <<'STUB'
print "{}\n";
exit($ENV{PLE_IO_TEST_VALIDATION_RESULT} || 0);
STUB

export PATH="$test_dir/bin:$PATH"
run_case() {
    local name=$1 expected=$2 actual
    : > "$PLE_IO_TEST_LOG"
    set +e
    bash "$test_dir/repo/kern/ple_io_run.sh" "$test_dir/$name" \
        "$test_dir/local shard" "$test_dir/nfs shard" 192 1 123 16 \
        > "$test_dir/$name.stdout" 2> "$test_dir/$name.stderr"
    actual=$?
    set -e
    [[ $actual == "$expected" ]] || {
        cat "$test_dir/$name.stderr" >&2
        printf '%s: expected exit %s, got %s\n' "$name" "$expected" "$actual" >&2
        exit 1
    }
    [[ $(jq -r .exit "$test_dir/$name/exit.json") == "$expected" ]]
}
export PLE_IO_TEST_BUSY=1
run_case busy_gpu 75
! rg -q '^benchmark ' "$PLE_IO_TEST_LOG"
[[ $(jq -r .status "$test_dir/busy_gpu/exit.json") == busy ]]
export PLE_IO_TEST_BUSY=0 PLE_IO_TEST_UTIL=7
run_case busy_util 75
! rg -q '^benchmark ' "$PLE_IO_TEST_LOG"
export PLE_IO_TEST_UTIL=0 PLE_IO_TEST_DISK=1
run_case busy_disk 75
! rg -q '^benchmark ' "$PLE_IO_TEST_LOG"
export PLE_IO_TEST_DISK=0 PLE_IO_TEST_RESULT=9
run_case failed_read 9
[[ $(rg -c '^benchmark ' "$PLE_IO_TEST_LOG") == 1 ]]
[[ $(jq -r .status "$test_dir/failed_read/exit.json") == failed ]]
export PLE_IO_TEST_RESULT=0
export PLE_IO_TEST_VALIDATION_RESULT=7
run_case failed_validation 7
[[ $(rg -c '^benchmark ' "$PLE_IO_TEST_LOG") == 2 ]]
[[ $(jq -r .status "$test_dir/failed_validation/exit.json") == failed ]]
[[ $(jq -r .exit "$test_dir/failed_validation/validation.exit.json") == 7 ]]
export PLE_IO_TEST_VALIDATION_RESULT=0
run_case complete 0
[[ $(rg -c '^benchmark ' "$PLE_IO_TEST_LOG") == 4 ]]
[[ $(rg -c '^process_query$' "$PLE_IO_TEST_LOG") == 6 ]]
[[ $(jq -r .status "$test_dir/complete/exit.json") == complete ]]
# Every earlier exit released the real coordination lock.
exec 8>/tmp/flash-sass-engine-validation.lock
flock -n 8
run_case lock_busy 75
[[ ! -s "$PLE_IO_TEST_LOG" ]]
flock -u 8
printf 'ple_io_run: busy GPU/utilization/disk/lock, early read or validation failure, per-run checks, raw exits and lock release passed\n'
