#!/usr/bin/env bash
set -euo pipefail
cd /home/nate/claude/flash-sass-resolve/issue-15
issue15_revision=$(git rev-parse HEAD)
[[ $issue15_revision =~ ^[0-9a-f]{40}$ ]]
issue15_target="/home/nate/claude/flash-sass-resolve/timing-$issue15_revision"
issue15_archive="/home/nate/codex/flash-issue15-resident-20261018/source-$issue15_revision.tar"
git archive --format=tar --output="$issue15_archive" "$issue15_revision"
scp "$issue15_archive" "archy:/tmp/flash-issue15-source-$issue15_revision.tar"
ssh archy bash -s -- "$issue15_revision" "$issue15_target" <<'REMOTE'
set -euo pipefail
issue15_revision=$1
issue15_target=$2
mkdir "$issue15_target"
tar -xf "/tmp/flash-issue15-source-$issue15_revision.tar" -C "$issue15_target"
date -u +%Y-%m-%dT%H:%M:%SZ
make -C "$issue15_target/kern" CUDA=/opt/cuda PTXAS11=/opt/cuda-11.4/bin/ptxas \
  'MAXAS=perl -I/home/nate/claude/flash-sass/maxas/lib /home/nate/claude/flash-sass/maxas/bin/maxas.pl' \
  "FLASH_REV=$issue15_revision" \
  die.cubin die_trace.cubin libengine.so flashcore flashd timings_test \
  flashd_timing_test trace_session_test output_tokens_test prefill_plan_test
sha256sum "$issue15_target"/kern/{die.cubin,die_trace.cubin,libengine.so,flashcore,flashd,timings_test,flashd_timing_test,trace_session_test,output_tokens_test,prefill_plan_test} > "$issue15_target/artifact-sha256.txt"
perl -c "$issue15_target/validation/issue15/run_resident.pl"
date -u +%Y-%m-%dT%H:%M:%SZ
REMOTE
printf '%s\n' "$issue15_revision" > /home/nate/codex/flash-issue15-resident-20261018/target-source-sha.txt
printf '%s\n' "$issue15_target" > /home/nate/codex/flash-issue15-resident-20261018/target-source-path.txt
