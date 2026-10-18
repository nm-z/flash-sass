#!/usr/bin/env bash
set -euo pipefail
repo=/home/nate/codex/flash-pr31-fullmodel-20261018
out=/home/nate/codex/flash-pr31-evidence-20261018/master99a6-validation
trap 'rc=$?; printf "%s\n" "$rc" > "$out/tests-exit.txt"' EXIT
cd "$repo"
mv "$out/capture-io.tap" "$out/capture-io-usage.log"
kern/qsa_capture_test "$out/capture-io" > "$out/capture-io.tap" 2>&1
kern/qsa_capture_prog_test > "$out/capture-program.tap" 2>&1
kern/qsa_reference_trace_test "$out/reference-trace" > "$out/reference-trace.tap" 2>&1
bash kern/qsa_oracle_gpu_gate_test.sh "$out/gpu-gate" > "$out/gpu-gate.tap" 2>&1
perl kern/qsa_oracle.pl --full --out "$out/native" > "$out/native.log" 2>&1
(cd "$repo" && sha256sum -c "$out/source-sha256.txt") > "$out/source-after.log"
date -u +%FT%TZ > "$out/end.txt"
printf 'MASTER_MERGE_TESTS_PASSED\n'
