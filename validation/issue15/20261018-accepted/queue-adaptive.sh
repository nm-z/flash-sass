#!/usr/bin/env bash
set -u
issue15_source=$1
issue15_out=$2
issue15_revision=$3
issue15_layout=$4
date -u +%Y-%m-%dT%H:%M:%SZ
printf 'QUEUED pid=%s\n' "$$"
exec 9<> /tmp/flash-sass-engine-validation.lock
flock -w 2700 -E 75 9
issue15_lock_rc=$?
if [ "$issue15_lock_rc" -ne 0 ]; then
    printf 'QUEUE_EXIT=%s\n' "$issue15_lock_rc"
    printf '%s\n' "$issue15_lock_rc" > "$issue15_out/exit-code.txt"
    exit "$issue15_lock_rc"
fi
date -u +%Y-%m-%dT%H:%M:%SZ
printf 'LEASE_GRANTED pid=%s\n' "$$"
perl /tmp/flash-issue15-select-target.pl "$issue15_source" "$issue15_revision" "$issue15_out" "$issue15_layout" 9
issue15_run_rc=$?
printf 'RUN_EXIT=%s\n' "$issue15_run_rc"
printf '%s\n' "$issue15_run_rc" > "$issue15_out/exit-code.txt"
exit "$issue15_run_rc"
