#!/usr/bin/env bash
set -euo pipefail
issue15_work=/home/nate/codex/flash-issue15-resident-20261018
issue15_revision=8ad9d6efded1be7aecb2398fcabd7faa201667c3
issue15_source="/home/nate/claude/flash-sass-resolve/timing-$issue15_revision"
issue15_out="/home/nate/claude/flash-sass-resolve/issue15-adaptive-${issue15_revision:0:7}-20261018"
ssh archy bash -s -- "$issue15_source" "$issue15_revision" > "$issue15_work/sse-cuda-build-resume.log" 2>&1 <<'REMOTE'
set -euo pipefail
issue15_source=$1
issue15_revision=$2
date -u +%Y-%m-%dT%H:%M:%SZ
make -C "$issue15_source/kern" -B CUDA=/opt/cuda PTXAS11=/opt/cuda-11.4/bin/ptxas \
 'MAXAS=perl -I/home/nate/claude/flash-sass/maxas/lib /home/nate/claude/flash-sass/maxas/bin/maxas.pl' \
 "FLASH_REV=$issue15_revision" libengine.so flashcore flashd timings_test \
 flashd_timing_test trace_session_test output_tokens_test prefill_plan_test
sha256sum "$issue15_source"/kern/{die.cubin,die_trace.cubin,libengine.so,flashcore,flashd,timings_test,flashd_timing_test,trace_session_test,output_tokens_test,prefill_plan_test} > "$issue15_source/artifact-sha256.txt"
perl -c "$issue15_source/validation/issue15/run_resident.pl"
date -u +%Y-%m-%dT%H:%M:%SZ
REMOTE
printf '%s\n' "$issue15_revision" > "$issue15_work/target-source-sha.txt"
printf '%s\n' "$issue15_source" > "$issue15_work/target-source-path.txt"
scp "$issue15_work/select-target.pl" archy:/tmp/flash-issue15-select-target.pl
scp "$issue15_work/queue-adaptive.sh" archy:/tmp/flash-issue15-queue-adaptive.sh
ssh archy bash -s -- "$issue15_source" "$issue15_out" "$issue15_revision" <<'REMOTE'
set -euo pipefail
issue15_source=$1
issue15_out=$2
issue15_revision=$3
mkdir "$issue15_out"
cp /tmp/flash-issue15-ipc-layout-20261018.json "$issue15_out/ipc-layout.json"
cp /tmp/flash-issue15-select-target.pl "$issue15_out/select-target.pl"
cp /tmp/flash-issue15-queue-adaptive.sh "$issue15_out/queue-adaptive.sh"
nohup setsid bash /tmp/flash-issue15-queue-adaptive.sh "$issue15_source" "$issue15_out" "$issue15_revision" "$issue15_out/ipc-layout.json" > "$issue15_out/queue.log" 2>&1 < /dev/null &
printf 'QUEUED_PID=%s\nOUTPUT=%s\nSOURCE_SHA=%s\n' "$!" "$issue15_out" "$issue15_revision"
REMOTE
printf '%s\n' "$issue15_out" > "$issue15_work/adaptive-output-path.txt"
