#!/usr/bin/env bash
# CPU-only follow-up: do not acquire, release, or bypass the GPU-work lock.
# No CUDA execution, model load, real shard read, or disk benchmark is run.
set -euo pipefail
root=$1
revision=$2
cd "$root"
printf 'host=%s\nsource_commit=%s\nstarted_utc=%s\n' "$(hostname)" "$revision" "$(date -u +%FT%TZ)"
printf 'scope=CPU-only; shared GPU lock is left untouched\n'
lslocks -o COMMAND,PID,MODE,PATH | grep flash-sass || true
cc --version | head -1
awk '/^#define CUDA_VERSION / {print "installed_header_" $0}' /opt/cuda/include/cuda.h
# The explicitly named targets contain only C compilation, Perl header
# generation, and CPU fixtures. Serialize compilation at low CPU/I/O priority.
export CUDA_VISIBLE_DEVICES=''
ionice -c 3 nice -n 19 make -j1 -C kern CUDA=/opt/cuda FLASH_REV="$revision" libengine.so ple_read_test ple_read_bench ple_read_audit ple_read_bench_test ple_read_audit_test
ionice -c 3 nice -n 19 make -j1 -C kern CUDA=/opt/cuda FLASH_REV="$revision" test-ple
sha256sum kern/libengine.so kern/ple_read_test kern/ple_read_bench kern/ple_read_audit kern/ple_read_bench_test kern/ple_read_audit_test
printf 'target_cpu_validation=PASS\nfinished_utc=%s\n' "$(date -u +%FT%TZ)"
