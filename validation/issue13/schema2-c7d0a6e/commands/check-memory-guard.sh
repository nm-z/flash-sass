#!/bin/bash
set -euo pipefail
cd /home/nate/codex/flash-pr30-run
exec 9>/tmp/flash-sass-engine-validation.lock
flock -n 9 || exit 75
export FLASH_TRACE_VALIDATION_LOCK_HELD=1
pr30_out=evidence/schema2/memory-guard
mkdir "$pr30_out"
pr30_finish() {
    pr30_status=$?
    trap - EXIT
    set +e
    ruby kern/trace_control.rb evidence/schema2/control.json restore > "$pr30_out/restore.log" 2>&1
    pr30_restore_status=$?
    cat "$pr30_out/restore.log"
    if [ "$pr30_restore_status" -ne 0 ]; then exit "$pr30_restore_status"; fi
    exit "$pr30_status"
}
trap pr30_finish EXIT
ruby kern/trace_control.rb evidence/schema2/control.json ordinary16 > "$pr30_out/ordinary.log" 2>&1
set +e
ruby kern/trace_control.rb evidence/schema2/control.json traced16 > "$pr30_out/rejected.log" 2>&1
pr30_rejected_status=$?
set -e
cat "$pr30_out/rejected.log"
[ "$pr30_rejected_status" -eq 1 ]
rg -q 'insufficient memory headroom; reload was not sent' "$pr30_out/rejected.log"
if rg -q '^> reload ' "$pr30_out/rejected.log"; then exit 1; fi
ruby -r socket -e 's=UNIXSocket.new("/tmp/flash.sock"); s.puts "config"; abort "timeout" unless IO.select([s],nil,nil,10); puts s.gets; s.close' > "$pr30_out/after-rejection-config.txt"
rg -q 'trace_tokens=0 trace_ctas=16' "$pr30_out/after-rejection-config.txt"
printf 'PASS oversized trace reload rejected before mutation; ordinary core remains available\n'
