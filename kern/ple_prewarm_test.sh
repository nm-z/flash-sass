#!/usr/bin/env bash
# CPU-only regression for serve.sh prewarming. No services or GPU work.
set -euo pipefail
test_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_script=${1:-"$test_dir/serve.sh"}
[[ -f "$source_script" ]] || { printf 'missing serve.sh: %s\n' "$source_script" >&2; exit 2; }
real_dd=$(type -P dd)
real_bash=$(type -P bash)
real_env=$(type -P env)
real_timeout=$(type -P timeout)
system_path=${PATH:-/usr/bin:/bin}
work=$(mktemp -d "${TMPDIR:-/tmp}/flash-ple-prewarm-test.XXXXXX")
trap 'rm -rf -- "$work"' EXIT
mkdir -p "$work/bin" "$work/runtime" "$work/script with spaces"
# Relocate fixed logs/sockets in a copy; preserve production control flow.
escaped=$(printf '%s\n' "$work/runtime/" | sed 's/[&|\\]/\\&/g')
script="$work/script with spaces/serve.sh"
sed "s|/tmp/|$escaped|g" "$source_script" > "$script"
chmod +x "$script"
fixture="$work/model shard with spaces.gguf"
"$real_dd" if=/dev/zero of="$fixture" bs=12345 count=1 status=none
cat > "$work/bin/dd" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'warm:start\n' >> "$PLE_TEST_EVENTS"
if [[ ${PLE_TEST_DD_FAIL:-0} == 1 ]]; then
    printf 'injected read failure\n' >&2
    exit 74
fi
"$PLE_TEST_REAL_DD" "$@"
printf 'warm:done\n' >> "$PLE_TEST_EVENTS"
STUB
cat > "$work/bin/setsid" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
[[ ${1:-} != -f ]] || shift
printf 'spawn:%s\n' "${1##*/}" >> "$PLE_TEST_EVENTS"
# Never execute a service, holder, fence, or any supplied command.
exit 0
STUB
cat > "$work/bin/nvidia-smi" <<'STUB'
#!/usr/bin/env bash
set -euo pipefail
printf 'unexpected:nvidia-smi\n' >> "$PLE_TEST_EVENTS"
exit 97
STUB
chmod +x "$work/bin/dd" "$work/bin/setsid" "$work/bin/nvidia-smi"
checks=0
last_log=
fail() {
    printf 'not ok %d - %s\n' "$checks" "$1"
    [[ ! -f ${last_log:-} ]] || sed -n '1,60s/^/# /p' "$last_log"
    [[ ! -f "$work/events" ]] || sed 's/^/# event: /' "$work/events"
    exit 1
}
# label expected_exit expected_events command [ENV=value ...]
check() {
    local label=$1 expected_exit=$2 expected_events=$3 command=$4
    shift 4
    checks=$((checks + 1))
    last_log="$work/case-$checks.log"
    : > "$work/events"
    local rc=0 actual
    "$real_timeout" 5 "$real_env" -i PATH="$work/bin:$system_path" LC_ALL=C \
        GGUF2="$fixture" PLE_TEST_EVENTS="$work/events" PLE_TEST_REAL_DD="$real_dd" \
        "$@" "$real_bash" "$script" "$command" > "$last_log" 2>&1 || rc=$?
    [[ $rc == "$expected_exit" ]] || fail "$label: exit $rc, expected $expected_exit"
    actual=$(cat "$work/events")
    [[ $actual == "$expected_events" ]] || fail "$label: unexpected read/launch order"
    printf 'ok %d - %s\n' "$checks" "$label"
}
warm=$'warm:start\nwarm:done'
warm_core=$'warm:start\nwarm:done\nspawn:flashcore'
check 'standalone reads a path containing spaces without services' 0 "$warm" prewarm
checks=$((checks + 1))
rg -q '^12345 bytes' "$last_log" || fail 'standalone did not report all fixture bytes'
printf 'ok %d - real dd read all 12345 fixture bytes\n' "$checks"
check 'automatic warm completes before core launch' 0 "$warm_core" core \
    FLASH_PLE_PREWARM=1 FLASH_PLE_DISK=1
check 'unset prewarm does not read the file' 0 'spawn:flashcore' core PLE_TEST_DD_FAIL=1
check 'disabled prewarm does not read the file' 0 'spawn:flashcore' core \
    FLASH_PLE_PREWARM=0 FLASH_PLE_DISK=1 PLE_TEST_DD_FAIL=1
check 'automatic read failure prevents core launch' 1 'warm:start' core \
    FLASH_PLE_PREWARM=1 FLASH_PLE_DISK=1 PLE_TEST_DD_FAIL=1
check 'standalone read failure returns failure without services' 1 'warm:start' prewarm PLE_TEST_DD_FAIL=1
check 'missing file fails before reading or launching' 1 '' prewarm GGUF2="$work/missing shard.gguf"
check 'directory is rejected before reading or launching' 1 '' prewarm GGUF2="$work"
check 'automatic missing file prevents core launch' 1 '' core \
    FLASH_PLE_PREWARM=1 FLASH_PLE_DISK=1 GGUF2="$work/missing shard.gguf"
# flashcore enables disk PLE by presence: zero and empty both count.
check 'disk value zero retains presence semantics' 0 "$warm_core" core FLASH_PLE_PREWARM=1 FLASH_PLE_DISK=0
check 'empty disk value retains presence semantics' 0 "$warm_core" core FLASH_PLE_PREWARM=1 FLASH_PLE_DISK=
check 'automatic prewarm requires disk variable presence' 2 '' core FLASH_PLE_PREWARM=1
check 'all rejects absent disk mode before holders or core' 2 '' all FLASH_PLE_PREWARM=1
for bad in '' 2 yes -1; do
    check "all rejects prewarm flag '$bad' before holders or core" 2 '' all FLASH_PLE_PREWARM="$bad" FLASH_PLE_DISK=1
done
printf '1..%d\n' "$checks"
