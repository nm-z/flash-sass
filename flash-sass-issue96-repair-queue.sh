#!/usr/bin/env bash
set -euo pipefail
repair_dir=$(mktemp -d /tmp/issue96-queue-repair.XXXXXXXX)
trap 'rm -rf -- "$repair_dir"' EXIT
cat > "$repair_dir/check.sh" <<'CHECK'
#!/usr/bin/env bash
set -euo pipefail
pid=1980783
owner=1980224
read -r -a state < "/proc/$pid/stat"
[[ ${state[21]} == 2472411 ]]
[[ ${state[2]} == t || ${state[2]} == T ]]
[[ $(readlink "/proc/$pid/exe") == /home/nate/claude/flash-sass-live/kern/flashd ]]
[[ $(readlink "/proc/$pid/fd/2") == /tmp/flashd.err ]]
[[ $(readlink "/proc/$pid/fd/4") == 'pipe:[1324175]' ]]
[[ $(readlink "/proc/$pid/fd/5") == 'socket:[1311655]' ]]
[[ $(readlink "/proc/$pid/fd/9") == /tmp/flash-sass-engine-validation.lock ]]
[[ $(readlink "/proc/$owner/fd/8") == /tmp/flash-sass-engine-validation.lock ]]
[[ $(readlink "/proc/$owner/fd/9") == 'pipe:[1324175]' ]]
flags2=$(awk '$1 == "flags:" {print $2}' "/proc/$pid/fdinfo/2")
flags4=$(awk '$1 == "flags:" {print $2}' "/proc/$pid/fdinfo/4")
(( (8#$flags2 & 3) == 1 ))
(( (8#$flags4 & 3) == 1 ))
threads=(/proc/$pid/task/*)
[[ ${#threads[@]} == 1 ]]
for fd in /proc/$pid/fd/*; do
  target=$(readlink "$fd")
  if [[ $target == socket:* && ${fd##*/} != 5 ]]; then
    printf 'Unexpected request socket: %s -> %s\n' "$fd" "$target" >&2
    exit 1
  fi
done
sudo -n -u nate -H -- gpuq ls | jq -e 'any(.[]; .id == "81739785-52f0-4791-b3db-71a0ea4f89ce" and .status == "collecting" and .exit_code == 0)' >/dev/null
cat > "$1" <<'ACTIONS'
set $queue_close_result = (long) syscall((long)3, (long)9)
printf "Inherited queue lock close: %ld\n", $queue_close_result
if $queue_close_result != 0
  detach
  quit 1
end
set $queue_dup_result = (long) syscall((long)292, (long)2, (long)4, (long)524288)
printf "Stale pipe writer replacement: %ld\n", $queue_dup_result
if $queue_dup_result != 4
  detach
  quit 1
end
ACTIONS
printf 'Stopped process identity and stale queue handles verified.\n'
CHECK
cat > "$repair_dir/repair.gdb" <<GDB
set pagination off
set confirm off
set auto-load off
set debuginfod enabled off
attach 1980783
set scheduler-locking on
shell /bin/bash $repair_dir/check.sh $repair_dir/actions.gdb
source $repair_dir/actions.gdb
detach
quit
GDB
curl -fsS --max-time 3 http://127.0.0.1:18090/queue | jq -e '.running == 0 and .waiting == 0' >/dev/null
sudo -n gdb -nx -nh -batch -x "$repair_dir/repair.gdb"
readlink /proc/1980783/fd/2 /proc/1980783/fd/4 /proc/1980783/fd/5
if [[ -e /proc/1980783/fd/9 ]]; then
  printf 'Unexpected remaining FD9\n' >&2
  exit 1
fi
curl -fsS --max-time 3 http://127.0.0.1:18090/queue
