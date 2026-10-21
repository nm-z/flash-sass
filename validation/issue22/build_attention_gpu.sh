#!/usr/bin/env bash
set -euo pipefail

issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_cuda=${CUDA:-/opt/cuda}
issue22_ptxas=${PTXAS11:-/opt/cuda-11.4/bin/ptxas}
issue22_maxas=${MAXAS_DIR:-/home/nate/.cache/fs22-maxas-54eda7af}
issue22_revision=$(git -C "$issue22_repo" rev-parse HEAD)

[[ -x $issue22_ptxas ]] || { printf 'missing SM52 ptxas: %s\n' "$issue22_ptxas" >&2; exit 2; }
[[ -f $issue22_maxas/lib/MaxAs/MaxAs.pm && -x $issue22_maxas/bin/maxas.pl ]] || {
    printf 'prepare the pinned MaxAs checkout first: %s\n' "$issue22_maxas" >&2
    exit 2
}

issue22_maxas_command="PATH=$issue22_cuda/bin:/usr/bin perl -I$issue22_maxas/lib $issue22_maxas/bin/maxas.pl"
make -B -C "$issue22_repo/kern" CUDA="$issue22_cuda" PTXAS11="$issue22_ptxas" \
    FLASH_REV="$issue22_revision" MAXAS="$issue22_maxas_command" \
    qsa_attention_gpu die.cubin
