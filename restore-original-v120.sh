#!/bin/bash
set -euo pipefail
# Run the entire transition under GPUq. Only replace the six known task-owned
# idle balanced holders; preserve all other GPU processes and both disk caches.
loader=/home/nate/work/flash-per-phase-balance-v44/native_weights
cache=/home/nate/work/flash-balanced-native-20261022-v1/original-cache
out=/tmp/flash-unrestricted-v120
old=/tmp/flash-balanced-postboot-v51
pids=(3991 4448 2454 3021 3472 4928)
gpus=(0 1 6 3 4 5)
mkdir -p "$out"
for d in 0 1 2 3 4 5; do
 pid=${pids[$d]}
 args=$(tr '\0' ' ' < "/proc/$pid/cmdline")
 case "$args" in
  "$loader launch "*" $old/h$d.ipc.bin ") ;;
  *) printf 'Refusing changed holder pid=%s command=%s\n' "$pid" "$args"; exit 1;;
 esac
 test -f "$cache/die$d/arena.bin"
 test -f "$cache/die$d/metadata.bin"
 test ! -e "$out/h$d.ipc.bin"
done
for d in 0 1 2 3 4 5; do
 pid=${pids[$d]}
 kill -TERM "$pid"
 # The holder removes readiness after leaving its wait, before releasing CUDA.
 for attempt in {1..100}; do
  test ! -e "$old/h$d.ipc.bin.ready" && break
  sleep .1
 done
 test ! -e "$old/h$d.ipc.bin.ready"
 for attempt in {1..100}; do
  if ! kill -0 "$pid" 2>/dev/null; then break; fi
  if test "$(awk '{print $3}' "/proc/$pid/stat" 2>/dev/null)" = Z; then break; fi
  sleep .1
 done
 "$loader" launch "$cache/die$d" "${gpus[$d]}" "$out/h$d.ipc.bin"
done
printf 'original_layout_ready manifests=%s physical=0,1,6,3,4,5\n' "$out"
