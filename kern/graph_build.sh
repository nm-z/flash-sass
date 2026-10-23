#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
perl die.pl graph
"${PTXAS11:-/opt/cuda-11.4/bin/ptxas}" -arch=sm_52 -o graph-ops/ops.cubin graph-ops/ops.ptx
for source in graph-ops/*.sass; do
    perl -I../maxas/lib ../maxas/bin/maxas.pl -i "$source" graph-ops/ops.cubin
done
