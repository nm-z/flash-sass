#!/usr/bin/env bash
# reset-dies.sh: free memory stranded on GPUs 0-5 by a GPU reset, which needs every CUDA program on the host stopped.
# Stops the die-2 holder and the stop-hook LFM server (GPU 7), resets 0-5, recreates the stop-hook unit exactly as it was,
# re-applies the clock service, restarts the die-2 fence on the fresh die, and restarts the power logger appending to its file.
# The holders and the core are restored afterwards through gpuq (holders-units.rb with UNITS_RESTORE_ONLY).
set -u
log() { echo "$(date '+%-I:%M:%S %p') $*"; }
LFM_DESC='stop hook LFM2-1.2B at its trained 128000-token context, GPU 7, no host prompt cache (Nate 2026-10-18)'
LFM=(/home/nate/llama-cuda118-master-bin/llama-server -m /home/nate/models/persuader/LFM2-1.2B-Q8_0.gguf -ngl 99 -c 128000 --parallel 1
     --jinja --cache-ram 0 --host 0.0.0.0 --port 18621)
start_lfm() {
  systemd-run --user --unit=stop-hook-lfm --description="$LFM_DESC" -p Restart=no --working-directory=/home/nate \
    -E CUDA_DEVICE_ORDER=PCI_BUS_ID -E CUDA_VISIBLE_DEVICES=GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4 \
    -E LD_LIBRARY_PATH=/home/nate/llama-cuda118-master-bin "${LFM[@]}"
}
cuda_users() {
  sudo -n sh -c 'for p in /proc/[0-9]*; do ls -l $p/fd 2>/dev/null | grep -q "/dev/nvidia[0-9]" && echo "$(basename $p):$(cat $p/comm)"; done' | tr '\n' ' '
}

log "stopping the die-2 holder"
for p in $(pgrep -x hold); do grep -qz /tmp/hold-h2.manifest /proc/$p/cmdline && kill -TERM $p; done
log "stopping stop-hook-lfm"
systemctl --user stop stop-hook-lfm.service
for i in $(seq 1 40); do [ -z "$(cuda_users)" ] && break; sleep 0.5; done
left=$(cuda_users)
if [ -n "$left" ]; then log "still open: $left; restarting stop-hook-lfm and stopping"; start_lfm; exit 1; fi
ok=1
for i in 0 1 2 3 4 5; do
  out=$(sudo -n nvidia-smi -r -i $i 2>&1); log "reset $i: $(echo "$out" | grep -v '^$' | tail -1)"
  echo "$out" | grep -qi "success" || ok=0
done
log "recreating stop-hook-lfm"
start_lfm
sudo -n systemctl restart gpu-application-clocks.service && log "clock service applied"
log "fence on die 2"
rm -f /tmp/fence-d2.ready
(cd /home/nate/claude/flash-sass-next/kern && CUDA_DEVICE_ORDER=PCI_BUS_ID setsid nohup flock -s -n /home/nate/codex/asm-port/GPU-c767d250-4454-d181-d648-8706b5f2fb64.lock \
  ./fence 2 2 2750-3769 > /home/nate/codex/flash-goal-140-1200/remap/fence.log 2>&1 < /dev/null &)
for i in $(seq 1 120); do [ -e /tmp/fence-d2.ready ] && break; sleep 0.5; done
[ -e /tmp/fence-d2.ready ] && log "fence ready" || log "fence NOT ready (see remap/fence.log)"
log "power logger (appending)"
setsid nohup sh -c 'exec nvidia-smi --query-gpu=timestamp,index,power.draw,temperature.gpu,clocks.sm,memory.used,clocks_throttle_reasons.active --format=csv,noheader -l 2 >> /home/nate/flash-power2.csv' > /dev/null 2>&1 < /dev/null &
sleep 2
nvidia-smi --query-gpu=index,memory.used,clocks.applications.graphics --format=csv,noheader
systemctl --user is-active stop-hook-lfm.service
[ $ok = 1 ] || { log "a reset did not report success"; exit 2; }
