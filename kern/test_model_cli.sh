#!/usr/bin/env bash
# CPU-only boundary checks. A positive control proves cuInit is intercepted.
set -euo pipefail
binary=$(realpath "${1:-kern/model_test}")
scratch_dir=$(mktemp -d /tmp/flash-model-cli.XXXXXX)
trap 'rm -rf "$scratch_dir"' EXIT
cat > "$scratch_dir/no_cuda.c" <<'C'
#include <stdio.h>
#include <stdlib.h>
int cuInit(unsigned flags) {
    (void)flags;
    fputs("CUDA_BOUNDARY_REACHED\n", stderr);
    _Exit(99);
}
C
"${CC:-cc}" -std=c99 -Wall -Wextra -fPIC -shared -o "$scratch_dir/no_cuda.so" "$scratch_dir/no_cuda.c"
# On a machine without a driver, explicitly supply the toolkit's link-only stub.
# No CUDA function is called by this test: the preload exits at the boundary.
if [[ -n ${CUDA_STUB_LIBRARY:-} ]]; then
    ln -s "$(realpath "$CUDA_STUB_LIBRARY")" "$scratch_dir/libcuda.so.1"
fi
seq 10 18 > "$scratch_dir/ids"
printf '10\n11\n' > "$scratch_dir/short-ids"
printf '10\n248320\n' > "$scratch_dir/bad-ids"
printf '10\n11junk\n' > "$scratch_dir/junk-ids"
printf '10\n11\n12\n13\n14\n15\n16\n17\n18\nignored-tail\n' > "$scratch_dir/long-ids"
checks=0
check() {
    local name=$1 expected=$2 b0=$3 b1=$4 tokens=$5 flags=$6
    shift 6
    local status
    set +e
    env -u MODEL_NOTWIN -u MODEL_NPH -u MODEL_INJECT -u ATTN_DIE \
        -u MODEL_CAPTURE -u MODEL_LAYER_CAPTURE -u MODEL_LAYER_INDEX \
        -u MODEL_LAYER_LENGTHS -u MODEL_PLE \
        MODEL_ATTN=qsa FLASH_NBLK=1 FLASH_KV_HOST=1 \
        ATTN_MAP=0,0,0,0,0,0,0,0,0,0,0,0 SPLIT_SHARES=1 \
        MODEL_IDS="$scratch_dir/ids" LD_PRELOAD="$scratch_dir/no_cuda.so" \
        LD_LIBRARY_PATH="$scratch_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$@" "$binary" /unused-cache "$b0" "$b1" "$tokens" "$flags" /unused-manifest \
        > "$scratch_dir/$name.stdout" 2> "$scratch_dir/$name.stderr"
    status=$?
    set -e
    if [[ $status != "$expected" ]]; then
        printf 'FAIL %s: exit %d, expected %d\n' "$name" "$status" "$expected" >&2
        cat "$scratch_dir/$name.stdout" "$scratch_dir/$name.stderr" >&2
        exit 1
    fi
    if [[ $expected == 99 ]]; then
        rg -q '^CUDA_BOUNDARY_REACHED$' "$scratch_dir/$name.stderr"
    elif rg -q 'CUDA_BOUNDARY_REACHED' "$scratch_dir/$name.stderr"; then
        printf 'FAIL %s entered CUDA\n' "$name" >&2
        exit 1
    fi
    printf 'PASS %s exit=%d\n' "$name" "$status"
    checks=$((checks + 1))
}
check positive-control 99 0 8 9 eh
check zero-length 2 0 8 0 eh
check negative-length 2 0 8 -1 eh
check beyond-capacity 2 0 8 262145 eh
check malformed-length 2 0 8 9junk eh
check malformed-block 2 0junk 8 9 eh
check invalid-flags 2 0 8 9 ehx
check empty-flags 2 0 8 9 ''
check invalid-budget 2 0 8 9 eh FLASH_NBLK=1junk
check zero-budget 2 0 8 9 eh FLASH_NBLK=0
check invalid-placement 2 0 8 9 eh FLASH_KV_HOST=2
check missing-teacher-ids 2 0 8 9 eh MODEL_IDS="$scratch_dir/short-ids"
check out-of-vocabulary-id 2 0 8 2 eh MODEL_IDS="$scratch_dir/bad-ids"
check malformed-id 2 0 8 2 eh MODEL_IDS="$scratch_dir/junk-ids"
check valid-id-prefix 99 0 8 9 eh MODEL_IDS="$scratch_dir/long-ids"
check capture-positive-control 99 0 48 9 eh MODEL_CAPTURE="$scratch_dir/raw" MODEL_PLE=/unused-ple MODEL_NOTWIN=1
check capture-missing-ids 2 0 48 9 eh MODEL_CAPTURE="$scratch_dir/raw" MODEL_PLE=/unused-ple MODEL_NOTWIN=1 MODEL_IDS="$scratch_dir/short-ids"
check capture-extra-ids 2 0 48 9 eh MODEL_CAPTURE="$scratch_dir/raw" MODEL_PLE=/unused-ple MODEL_NOTWIN=1 MODEL_IDS="$scratch_dir/long-ids"
check layer-capture-positive-control 99 0 48 9 eh MODEL_LAYER_CAPTURE="$scratch_dir/layer" MODEL_LAYER_INDEX=3 MODEL_LAYER_LENGTHS=9 MODEL_PLE=/unused-ple MODEL_NOTWIN=1
check layer-capture-missing-ids 2 0 48 9 eh MODEL_LAYER_CAPTURE="$scratch_dir/layer" MODEL_LAYER_INDEX=3 MODEL_LAYER_LENGTHS=9 MODEL_PLE=/unused-ple MODEL_NOTWIN=1 MODEL_IDS="$scratch_dir/short-ids"
check layer-capture-extra-ids 2 0 48 9 eh MODEL_LAYER_CAPTURE="$scratch_dir/layer" MODEL_LAYER_INDEX=3 MODEL_LAYER_LENGTHS=9 MODEL_PLE=/unused-ple MODEL_NOTWIN=1 MODEL_IDS="$scratch_dir/long-ids"
check no-attention-block 2 0 3 9 eh
check unreachable-selection 2 0 8 4 eh MODEL_INJECT=selection
check reachable-selection 99 0 8 9 eh MODEL_INJECT=selection
check unreachable-logits 2 0 8 9 e MODEL_INJECT=logits
check unreachable-dn 2 3 4 9 eh MODEL_INJECT=dn
check unreachable-conv 2 3 4 9 eh MODEL_INJECT=conv
check unreachable-ple 2 3 4 9 eh MODEL_INJECT=ple
check unknown-injection 2 0 8 9 eh MODEL_INJECT=unknown
check timing-injection 2 0 8 9 eh MODEL_INJECT=layer MODEL_NOTWIN=1
check truncated-program 2 0 8 9 eh MODEL_NPH=1
check absent-attention-owner 2 0 8 9 eh ATTN_MAP=1,1,1,1,1,1,1,1,1,1,1,1
printf 'OK: %d CPU boundary checks; CUDA calls intercepted, no GPU validation\n' "$checks"
