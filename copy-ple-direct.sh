#!/bin/bash
set -euo pipefail
source=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
target=/home/nate/codex/flash-astra-perf-20261018/ple-direct.gguf
test ! -e "$target"
test ! -e "$target.tmp"
ple_bytes=$(stat -c %s "$source")
ple_block=1048576
ple_blocks=$((ple_bytes / ple_block))
ple_prefix=$((ple_blocks * ple_block))
ple_tail=$((ple_bytes - ple_prefix))
printf 'copy_bytes=%s direct_prefix=%s tail=%s\n' "$ple_bytes" "$ple_prefix" "$ple_tail"
# Bound the dirty-page footprint: the large transfer bypasses both caches.
ionice -c 3 nice -n 10 dd if="$source" of="$target.tmp" bs=1M count="$ple_blocks" \
    iflag=direct,fullblock oflag=direct conv=fsync status=progress
if ((ple_tail)); then
    ionice -c 3 nice -n 10 dd if="$source" of="$target.tmp" bs=1M \
        skip="$ple_prefix" count="$ple_tail" seek="$ple_prefix" \
        iflag=skip_bytes,count_bytes,fullblock oflag=seek_bytes conv=notrunc,fsync status=none
fi
test "$(stat -c %s "$target.tmp")" = "$ple_bytes"
mv "$target.tmp" "$target"
printf '\ncopy_complete=%s\n' "$ple_bytes"
