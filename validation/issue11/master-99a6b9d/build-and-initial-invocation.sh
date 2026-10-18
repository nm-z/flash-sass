#!/usr/bin/env bash
set -euo pipefail
repo=/home/nate/codex/flash-pr31-fullmodel-20261018
out=/home/nate/codex/flash-pr31-evidence-20261018/master99a6-validation
mkdir "$out"
trap 'rc=$?; printf "%s\n" "$rc" > "$out/exit.txt"' EXIT
cd "$repo"
date -u +%FT%TZ > "$out/start.txt"
git rev-parse HEAD > "$out/source-commit.txt"
git status --porcelain=v1 --untracked-files=no > "$out/source-status.txt"
sha256sum kern/engine.c kern/Makefile kern/prog.c kern/prog.h kern/qsa_capture.c kern/qsa_capture.h kern/qsa_oracle.c kern/qsa_oracle_bridge.c kern/qsa_oracle_gpu.c kern/qsa_model_reference.c kern/qsa_model_selected_reference.c kern/qsa_model_compare.pl ref/qsa.c ref/attn.c > "$out/source-sha256.txt"
make -C kern dieops.h qsa_oracle qsa_logits_check qsa_capture_test qsa_capture_prog_test qsa_reference_trace_test qsa_model_reference qsa_model_selected_reference LLAMA_REF_SRC=/home/nate/codex/issue11-llama-fcac190 QSA_LLAMA_DIR=/home/nate/codex/issue11-llama-fcac190 > "$out/build.log" 2>&1
cc -std=gnu99 -O2 -ffp-contract=off -Wall -pthread -Ikern -Iref -I/opt/cuda/include -fsyntax-only kern/engine.c kern/qsa_capture.c > "$out/engine-syntax.log" 2>&1
make -C kern libengine.so libengine_reuse_test.so CC='cc -L/opt/cuda/lib64/stubs' > "$out/engine-link.log" 2>&1
kern/qsa_capture_test > "$out/capture-io.tap" 2>&1
kern/qsa_capture_prog_test > "$out/capture-program.tap" 2>&1
kern/qsa_reference_trace_test > "$out/reference-trace.tap" 2>&1
bash kern/qsa_oracle_gpu_gate_test.sh > "$out/gpu-gate.tap" 2>&1
perl kern/qsa_oracle.pl --full --out "$out/native" > "$out/native.log" 2>&1
(cd "$repo" && sha256sum -c "$out/source-sha256.txt") > "$out/source-after.log"
date -u +%FT%TZ > "$out/end.txt"
printf 'MASTER_MERGE_VALIDATION_PASSED\n'
