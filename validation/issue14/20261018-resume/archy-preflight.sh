#!/bin/bash
set -u
record() {
  printf '\nCOMMAND'; printf ' %q' "$@"; printf '\n'
  "$@"
  code=$?
  printf 'EXIT %s\n' "$code"
  return "$code"
}
record hostname
record date -u +%Y-%m-%dT%H:%M:%SZ
exec 9>/tmp/flash-sass-engine-validation.lock
if ! flock -n 9; then printf 'BLOCKED: shared validation lock busy\n'; exit 73; fi
printf 'LOCK_ACQUIRED /tmp/flash-sass-engine-validation.lock\n'
trap 'flock -u 9; printf "LOCK_RELEASED /tmp/flash-sass-engine-validation.lock\n"' EXIT
record nvidia-smi --query-gpu=index,uuid,memory.free,memory.used,utilization.gpu,temperature.gpu,pcie.link.gen.current,pcie.link.width.current --format=csv
printf '\nCOMMAND ps -eo pid,ppid,stat,comm (filtered display to model processes)\n'
ps -eo pid,ppid,stat,comm | awk 'NR==1 || $4 ~ /^(flashcore|flashd|hold|fence)$/ {print}'
printf 'EXIT %s\n' "${PIPESTATUS[0]}"
printf '\nCOMMAND config over /tmp/flash.sock via timeout 10 socat -T 5\n'
printf 'config\n' | timeout 10 socat -T 5 - UNIX-CONNECT:/tmp/flash.sock
printf 'EXIT %s\n' "${PIPESTATUS[1]}"
printf '\nNo model request, reload, stop, sampling change, or GPU benchmark was issued.\n'
