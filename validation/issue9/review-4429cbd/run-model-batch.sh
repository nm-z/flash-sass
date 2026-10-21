#!/usr/bin/env bash
set -euo pipefail
repo=$1
out=$2
cache=$3
build_log=$4
mkdir "$out"
common=(perl "$repo/kern/run_qsa_model_matrix.pl"
    --repo "$repo" --cache "$cache/tensors" --ids "$cache/ids9.txt"
    --blocks 0:8 --flags eh --cuda-visible 7 --lock-fd 9
    --weight-manifest "$cache/tensors.sha256"
    --weight-manifest "$cache/manifest.new" --build-log "$build_log"
    --min-free-mib 512 --timeout-seconds 600)
"${common[@]}" --out "$out/positive9" --tokens 9 --budgets 1,512 --kv 0,1 --modes token,loop --controls none
"${common[@]}" --out "$out/control1" --tokens 1 --budgets 1 --kv 0,1 --modes token --controls all
"${common[@]}" --out "$out/selection9" --tokens 9 --budgets 1 --kv 0,1 --modes token --controls selection
