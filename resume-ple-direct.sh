#!/bin/bash
set -euo pipefail
source=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
target=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf
if test -f "$target"; then exit 0; fi
kill -TERM 2390184 2>/dev/null || true
if test -r /proc/2390187/cmdline; then
    tr '\0' '\n' < /proc/2390187/cmdline | rg -Fx "of=$target.tmp" >/dev/null
    kill -TERM 2390187
    while test -e /proc/2390187; do
        test "$(awk '{print $3}' /proc/2390187/stat)" != Z || break
        sleep 0.1
    done
fi
ple_bytes=$(stat -c %s "$source")
ple_block=8388608
ple_total=$((ple_bytes / ple_block))
ple_resume=$(($(stat -c %s "$target.tmp") / ple_block))
ple_count=$((ple_total - ple_resume))
ple_prefix=$((ple_total * ple_block))
ple_tail=$((ple_bytes - ple_prefix))
printf 'resume_bytes=%s remaining_direct_bytes=%s tail=%s\n' "$((ple_resume*ple_block))" "$((ple_count*ple_block))" "$ple_tail"
ionice -c 3 nice -n 10 dd if="$source" bs=8M skip="$ple_resume" count="$ple_count" iflag=direct,fullblock status=none |
    ionice -c 3 nice -n 10 dd of="$target.tmp" bs=8M seek="$ple_resume" iflag=fullblock oflag=direct conv=notrunc,fsync status=progress
if ((ple_tail)); then
    ionice -c 3 nice -n 10 dd if="$source" of="$target.tmp" bs=8M skip="$ple_prefix" count="$ple_tail" seek="$ple_prefix" \
        iflag=skip_bytes,count_bytes,fullblock oflag=seek_bytes conv=notrunc,fsync status=none
fi
test "$(stat -c %s "$target.tmp")" = "$ple_bytes"
mv "$target.tmp" "$target"
printf '\ncopy_complete=%s\n' "$ple_bytes"
