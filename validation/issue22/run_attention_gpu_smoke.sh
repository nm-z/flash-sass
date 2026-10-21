#!/usr/bin/env bash
set -euo pipefail

issue22_gpu=${ISSUE22_GPU:?ISSUE22_GPU is required}
issue22_total=${ISSUE22_TOTAL:?ISSUE22_TOTAL is required}
issue22_budget=${ISSUE22_BUDGET:?ISSUE22_BUDGET is required}
issue22_lengths=${ISSUE22_LENGTHS:?ISSUE22_LENGTHS is required}
issue22_runner=${ISSUE22_RUNNER:?ISSUE22_RUNNER is required}
issue22_out=${GPUQ_RUN_DIR:?GPUQ_RUN_DIR is required}/issue22-attention-smoke

exec bash "$issue22_runner" "$issue22_gpu" "$issue22_total" \
    "$issue22_budget" "$issue22_lengths" "$issue22_out"
