#!/bin/bash
# bootstrap.sh [dir]: build and test everything from a fresh clone, the way a new machine would, and say which step broke.
#   clone the repo, clone MaxAs and lay maxas-patch/ over it, build the binaries and kernels the README names plus the ones serve.sh starts, then run the unit tests.
#   The tests that need a GPU run on one die (BOOT_DIE, default 6: the one the engine does not use); the model files are only read for the tests that open the GGUF.
#   It does not start the engine: ./serve.sh all does, from the directory this builds.
# Env: REPO, MAXAS_REPO, BOOT_DIE, JOBS, SHARD1 (the first GGUF shard).
set -u
REPO=${REPO:-https://github.com/nm-z/flash-sass}; MAXAS_REPO=${MAXAS_REPO:-https://github.com/NervanaSystems/maxas}
D=${1:-$HOME/claude/flash-sass-boot}; DIE=${BOOT_DIE:-6}; JOBS=${JOBS:-$(nproc)}
SHARD1=${SHARD1:-/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf}
fail=0
step() { printf '\n== %s  [%s]\n' "$1" "$(date +%T)"; }
run() { printf '+ %s\n' "$*"; "$@"; local rc=$?; [ $rc -eq 0 ] || { printf 'FAILED (exit %d): %s\n' $rc "$*"; fail=$((fail + 1)); }; return $rc; }
[ -e "$D" ] && { echo "bootstrap: $D exists; give a new directory" >&2; exit 2; }
step "clone $REPO"; run git clone -q "$REPO" "$D" || exit 1
step "MaxAs + patches"
run git clone -q "$MAXAS_REPO" "$D/maxas" || exit 1
run cp "$D"/maxas-patch/Cubin.pm "$D"/maxas-patch/MaxAs.pm "$D"/maxas-patch/MaxAsGrammar.pm "$D/maxas/lib/MaxAs/" || exit 1
cd "$D/kern" || exit 1
step "build: the binaries and kernels the README names, and what serve.sh starts"
run make -j"$JOBS" die.cubin libengine.so flashcore flashd hold fence memtest
step "build: the unit tests"
run make -j"$JOBS" all ple_eos_test dn_l2norm_test output_tokens_test plerows_test dnstep_test dnstep.cubin dnstep_tiled.cubin plerows.cubin qsa_test attnpartn_unitmap
step "unit tests without a GPU"
run ./ple_eos_test; run ./dn_l2norm_test; run ./attnpartn_unitmap >/dev/null; run ./output_tokens_test "$SHARD1"
step "unit tests on die $DIE"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DIE
run make test-gemv
S=$SHARD1; run ./gemv_k_test "$S" >/dev/null; run ./gemv_bf16_test "$S" >/dev/null
run ./plerows_test >/dev/null; run ./dnstep_test; run ./dnstep_test tiled
run ./qsa_status_test.sh >/dev/null
step "result"
if [ $fail -eq 0 ]; then echo "bootstrap: PASS ($D)"; else echo "bootstrap: $fail step(s) FAILED"; exit 1; fi
