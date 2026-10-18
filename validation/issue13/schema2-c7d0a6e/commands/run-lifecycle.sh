#!/bin/bash
set -euo pipefail
cd /home/nate/codex/flash-pr30-run
exec 9>/tmp/flash-sass-engine-validation.lock
flock -n 9 || exit 75
export FLASH_TRACE_VALIDATION_LOCK_HELD=1
pr30_control=evidence/schema2/control.json
pr30_output=evidence/schema2/lifecycle
pr30_finish() {
    pr30_status=$?
    trap - EXIT
    set +e
    ruby kern/trace_control.rb "$pr30_control" restore > evidence/schema2/restore-lifecycle.log 2>&1
    pr30_restore_status=$?
    cat evidence/schema2/restore-lifecycle.log
    if [ "$pr30_restore_status" -ne 0 ]; then
        printf 'Restoration failed with status %s\n' "$pr30_restore_status" >&2
        exit "$pr30_restore_status"
    fi
    exit "$pr30_status"
}
trap pr30_finish EXIT
ruby kern/trace_control.rb "$pr30_control" lifecycle > evidence/schema2/prepare-lifecycle.log 2>&1
cat evidence/schema2/prepare-lifecycle.log
ruby kern/trace_lifecycle_check.rb --socket /tmp/flash.sock --out "$pr30_output" --source-sha c7d0a6e14e99edd9e91ea16c010ae7938ac647e0 --pair-check --preflight /home/nate/codex/flash-pr30-run/evidence/schema2/preflight.sh
