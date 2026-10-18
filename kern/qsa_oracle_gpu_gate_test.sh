#!/usr/bin/env bash
# CPU-only gate regressions: mocked device/process queries, real flock/timeout.
# Only the fixed lock pathname is changed in a test copy; no GPU is accessed.
set -euo pipefail
(($# == 1)) || { printf 'usage: %s NEW_TEST_DIRECTORY\n' "$0" >&2; exit 2; }
mkdir -- "$1"
root=$(cd -- "$1" && pwd)
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
mkdir "$root/bin"
sed "s|/tmp/flash-sass-engine-validation.lock|$root/queue.lock|g" "$here/qsa_oracle_gpu_gate.sh" > "$root/gate.sh"
cat > "$root/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
set -eu
case " $* " in
    *--query-gpu=uuid*)
        if [[ ${MOCK_BAD_UUID:-0} == 1 ]]; then echo 'not-one-uuid'; else echo GPU-11111111-1111-1111-1111-111111111111; fi ;;
    *--query-compute-apps=*)
        [[ ${MOCK_QUERY_FAIL:-0} == 0 ]] || exit 9
        case ${MOCK_BUSY:-other} in
            selected) echo '123, ./hold, GPU-11111111-1111-1111-1111-111111111111, 512' ;;
            other) echo '234, ./hold, GPU-22222222-2222-2222-2222-222222222222, 4096' ;;
            none) : ;;
            *) exit 2 ;;
        esac ;;
    *) echo 'Mock NVIDIA inventory: CPU-only test' ;;
esac
SH
cat > "$root/bin/ps" <<'SH'
#!/usr/bin/env bash
printf '234 1 hold ./hold\n'
SH
chmod +x "$root/bin/"*
export PATH="$root/bin:$PATH"
checks=0
check() { checks=$((checks + 1)); if ! "$@"; then printf 'FAIL check=%s command=%q\n' "$checks" "$*" >&2; exit 1; fi; }
expect_exit() {
    local expected=$1; shift
    set +e
    "$@" > "$root/last.stdout" 2> "$root/last.stderr"
    local rc=$?
    set -e
    check test "$rc" -eq "$expected"
}
expect_exit 0 bash "$root/gate.sh" --out "$root/selected-idle" --device 6 --wait 0 -- \
    bash -c 'test "$CUDA_VISIBLE_DEVICES" = GPU-11111111-1111-1111-1111-111111111111 && test "$QSA_ORACLE_GPU_GUARDED" = 1 && echo child-ran'
check grep -q COMMAND_PASSED "$root/selected-idle/status.txt"
check grep -q child-ran "$root/selected-idle/run.stdout"
check grep -q child_cuda_ordinal=0 "$root/selected-idle/device.txt"
expect_exit 75 bash "$root/gate.sh" --out "$root/whole-host-busy" --wait 0 --check
check test ! -e "$root/whole-host-busy/run.stdout"
MOCK_BUSY=selected expect_exit 75 bash "$root/gate.sh" --out "$root/selected-busy" --device 6 --wait 0 --check
check grep -q BLOCKED_BUSY "$root/selected-busy/status.txt"
MOCK_BUSY=none expect_exit 0 bash "$root/gate.sh" --out "$root/check-only" --device 6 --check
check grep -q IDLE_CHECK_ONLY "$root/check-only/status.txt"
check test ! -e "$root/check-only/run.stdout"
MOCK_BAD_UUID=1 expect_exit 2 bash "$root/gate.sh" --out "$root/bad-uuid" --device 6 --check
MOCK_QUERY_FAIL=1 expect_exit 2 bash "$root/gate.sh" --out "$root/query-fail" --device 6 --check
check test ! -e "$root/query-fail/run.stdout"
expect_exit 2 bash "$root/gate.sh" --out "$root/invalid-wait" --wait 99999999999999 --check
expect_exit 2 bash "$root/gate.sh" --out "$root/invalid-timeout" --run-timeout 0 --check
expect_exit 2 bash "$root/gate.sh" --out "$root/check-with-command" --check -- true
expect_exit 2 bash "$root/gate.sh" --out "$root/invalid-device" --device -1 --check
expect_exit 2 bash "$root/gate.sh" --out "$root/no-command"
expect_exit 0 bash "$root/gate.sh" --out "$root/leading-zero" --wait 00000 --run-timeout 00001 --device 006 -- true
# A real holder blocks a nonblocking request without allowing the child to run.
flock "$root/queue.lock" bash -c 'touch "$1"; sleep 2' _ "$root/holder-ready" &
holder=$!
while [[ ! -f "$root/holder-ready" ]]; do sleep 0.02; done
expect_exit 75 bash "$root/gate.sh" --out "$root/lock-blocked" --device 6 --wait 0 -- touch "$root/forbidden-child"
check test ! -e "$root/forbidden-child"
check grep -q BLOCKED_LOCK "$root/lock-blocked/status.txt"
expect_exit 0 bash "$root/gate.sh" --out "$root/queued" --device 6 --wait 5 -- touch "$root/queued-child"
wait "$holder"
check test -f "$root/queued-child"
check grep -q QUEUED "$root/queued/wait.log"
# Bound our own child and retain its real exit code, including numerical errors.
expect_exit 3 bash "$root/gate.sh" --out "$root/numerical-failure" --device 6 -- bash -c 'exit 3'
check grep -q 'COMMAND_FAILED exit=3' "$root/numerical-failure/status.txt"
expect_exit 124 bash "$root/gate.sh" --out "$root/run-timeout" --device 6 --run-timeout 1 -- sleep 5
check grep -q 'COMMAND_FAILED exit=124' "$root/run-timeout/status.txt"
expect_exit 0 bash "$root/gate.sh" --out "$root/after-timeout" --device 6 --check
expect_exit 1 bash "$root/gate.sh" --out "$root/after-timeout" --device 6 --check
printf 'PASS: %s CPU-only GPU-gate checks, real lock queue and timeout; no GPU execution\n' "$checks"
