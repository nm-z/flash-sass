#!/usr/bin/env bash
# CPU and CUDA-entry-trap validation only. No device initialization or launch.
set -euo pipefail
repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
out=${1:?usage: verify.sh NEW_OUTPUT_DIRECTORY}
mkdir -p "$(dirname "$out")"
mkdir "$out"
out=$(cd "$out" && pwd)
cd "$repo"
printf 'test\texit\texpected\tcommand\n' > "$out/status.tsv"
run() {
    local label=$1 expected=$2 rc command
    shift 2
    printf -v command '%q ' "$@"
    set +e
    "$@" > "$out/$label.log" 2>&1
    rc=$?
    set -e
    printf '%s\t%s\t%s\t%s\n' "$label" "$rc" "$expected" "$command" >> "$out/status.tsv"
    if [ "$rc" -ne "$expected" ]; then
        cat "$out/$label.log"
        return 1
    fi
}
{
    hostname
    date -u +%FT%TZ
    git rev-parse HEAD
    git status --porcelain
    cc --version | head -n 1
    perl -e 'print "$^V\n"'
} > "$out/identity.txt"
stub=${CUDA_STUB_DIR:-${CUDA:-/opt/cuda}/targets/x86_64-linux/lib/stubs}
test -f "$stub/libcuda.so"
mkdir "$out/cuda-stub"
ln -s "$stub/libcuda.so" "$out/cuda-stub/libcuda.so.1"
run cpu-build 0 make -C kern -B idxscore_position_test checked_test
run positions 0 ./kern/idxscore_position_test
run checked-api 0 ./kern/checked_test
run harness-build 0 make -C kern -B "CC=cc -L$stub" qsa_test attn_test
cd kern
run cli 0 env "LD_LIBRARY_PATH=$out/cuda-stub" perl checked_cli_test.pl --out "$out/cli"
cd "$repo"
sha256sum kern/qsa_test.c kern/qsa_test_bounds.h kern/idxscore_position_test.c \
    kern/checked_cli_test.pl kern/checked_test.c ref/checked.c ref/checked.h \
    kern/Makefile kern/engine.c > "$out/source-sha256.txt"
sha256sum kern/idxscore_position_test kern/checked_test kern/qsa_test \
    kern/attn_test > "$out/binary-sha256.txt"
cat "$out/status.tsv" "$out/positions.log" "$out/checked-api.log" "$out/cli.log"
