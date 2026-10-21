#!/usr/bin/env bash
set -euo pipefail

issue22_gpu=${ISSUE22_GPU:?ISSUE22_GPU is required}
issue22_total=${ISSUE22_TOTAL:?ISSUE22_TOTAL is required}
issue22_budget=${ISSUE22_BUDGET:?ISSUE22_BUDGET is required}
issue22_lengths=${ISSUE22_LENGTHS:?ISSUE22_LENGTHS is required}
issue22_runner=${ISSUE22_RUNNER:?ISSUE22_RUNNER is required}
issue22_out=${GPUQ_RUN_DIR:?GPUQ_RUN_DIR is required}/issue22-attention-smoke

# gpuq owns the shared validation lease and passes the locked descriptor
# through the job environment. Keep descriptor 9 open for the runner; opening
# the lock again here would deadlock against gpuq.
if [[ -n ${GPUQ_JOB_ID:-} || -n ${GPUQ_LOCK_FD:-} ]]; then
    [[ -n ${GPUQ_JOB_ID:-} && ${GPUQ_LOCK_FD:-} == 9 ]] || {
        printf 'incomplete inherited gpuq lock lease\n' >&2
        exit 2
    }
    [[ $(readlink -- "/proc/$$/fd/9" 2>/dev/null || true) == /tmp/flash-sass-engine-validation.lock ]] || {
        printf 'gpuq descriptor 9 does not reference the shared validation lock\n' >&2
        exit 2
    }
fi

exec bash "$issue22_runner" "$issue22_gpu" "$issue22_total" \
    "$issue22_budget" "$issue22_lengths" "$issue22_out"
