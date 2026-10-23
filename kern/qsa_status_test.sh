#!/usr/bin/env bash
# Exit-status regressions for issue #8. Run on one idle CUDA device.
set -eu
cd "$(dirname "$0")"
out=${QSA_STATUS_OUT:-qsa-status-results}
mkdir -p "$out"
{
  git rev-parse HEAD 2>/dev/null || true
  "${CC:-cc}" --version | head -n1
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
  sha256sum qsa_test "${DIE_CUBIN:-die.cubin}"
} > "$out/identity.txt"
"${CC:-cc}" -std=gnu99 -O2 -Wall -Wextra -o "$out/unitmap" attnpartn_unitmap.c
expect() {
  label=$1; wanted=$2; shift 2
  printf '%s:' "$label"; printf ' %s' "$@"; printf '\n'
  set +e
  "$@" > "$out/$label.log" 2>&1
  status=$?
  set -e
  cat "$out/$label.log"
  printf '%s exit=%s expected=%s\n' "$label" "$status" "$wanted" | tee -a "$out/status.txt"
  test "$status" -eq "$wanted"
  if test "$wanted" -eq 1 && rg -q 'BIT-EXACT' "$out/$label.log"; then
    printf 'false success in %s\n' "$label" >&2; return 1
  fi
}
: > "$out/status.txt"
expect unitmap-clean 0 "$out/unitmap"
expect unitmap-duplicate 1 "$out/unitmap" --inject-duplicate
expect unitmap-missing 1 "$out/unitmap" --inject-missing
for visible in 1 2052; do
  expect "kv-$visible-clean" 0 env QSA_KV_LAST_ONLY=1 ./qsa_test kv "$visible" 512
  expect "kv-$visible-query-byte" 1 env QSA_KV_LAST_ONLY=1 QSA_INJECT_QO_MISMATCH=1 ./qsa_test kv "$visible" 512
done
printf 'PASS: QSA query mismatches and non-bijective unit maps return failure\n'
