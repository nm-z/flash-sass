#!/usr/bin/env bash
set -euo pipefail
repo=$1
out=$2
scripts=$3
ple=$4
mkdir "$out"
for budget in 1 512; do
    perl "$scripts/check_qsa_engine_requests.pl" \
        --repo "$repo" --out "$out/budget-$budget" --ple "$ple" \
        --revision 4429cbd0e40c9338cb9e5cf809afd52cb71e0a5e \
        --layers 8 --nblk "$budget" --cuda-visible 7 --lock-fd 9 \
        --modes scalar,chunk3 --kv 0,1 --timeout 600 --min-free-mib 512
done
