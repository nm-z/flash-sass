#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
perl die.pl graph
graph_directory=${GRAPH_OUTPUT:-graph-ops}
"${PTXAS11:-/opt/cuda-11.4/bin/ptxas}" -arch=sm_52 -o "$graph_directory/ops.cubin" "$graph_directory/ops.ptx"
for source in "$graph_directory"/*.sass; do
    perl -I../maxas/lib ../maxas/bin/maxas.pl -i "$source" "$graph_directory/ops.cubin"
done
