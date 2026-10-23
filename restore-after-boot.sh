#!/usr/bin/env bash
# restore-after-boot.sh: after a power cycle and `serve.sh holders` (flash-sass-next/kern), wait for the six holders, then start the core with
# the serving engine through gpuq (restore-service.rb checks the config against the serving one), then flashvis and flashd (flash-sass-live/kern).
set -u
B=/home/nate/codex/flash-goal-140-1200
export PATH=$B/gpuq-fixed/bin:/usr/local/bin:/usr/bin:/bin
log() { echo "$(date '+%-I:%M:%S %p') $*"; }
log "waiting for the six holders"
n=0
for i in $(seq 3600); do
  n=$(ls /tmp/hold-h[0-5].manifest.ready 2>/dev/null | wc -l)
  [ "$n" = 6 ] && break
  [ "$(pgrep -c -x hold)" -lt 6 ] && { log "a holder exited before it was ready"; exit 1; }
  sleep 1
done
[ "$n" = 6 ] || { log "holders not ready after 60 minutes"; exit 1; }
log "holders ready; core through gpuq"
cd $B && gpuq run --name restore-after-powercycle -- ruby $B/restore-service.rb || { log "core restore failed"; exit 1; }
cd /home/nate/claude/flash-sass-live/kern || exit 1
[ -x flashvis ] || make flashvis || log "flashvis did not build; images will answer 503"   # the binary is not kept in the tree
pgrep -x flashvis > /dev/null || ./serve.sh vis
pgrep -x flashd > /dev/null || ./serve.sh flashd
sleep 2
log "flashvis $(pgrep -x flashvis) flashd $(pgrep -x flashd)"
