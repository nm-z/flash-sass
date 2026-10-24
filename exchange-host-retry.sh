#!/bin/bash
set -euo pipefail
root=/home/nate/codex/flash-astra-build-20261018
out=/home/nate/codex/flash-astra-perf-20261018/component
exec 8>>/home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock
exec 7>>/home/nate/codex/asm-port/GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4.lock
flock 8
flock 7
export CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b,GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4
for mode in base reg single; do
  timeout -k 5 30 "$root/single/kern/exchange_test" --host "$root/$mode/kern/die.cubin" > "$out/$mode-exchange.log" 2>&1
  printf '%s two-die exchange checks passed\n' "$mode"
done
export CUDA_VISIBLE_DEVICES=GPU-431371bd-b482-96b2-4b70-04f83c6d075b
timeout -k 5 180 "$root/hc/kern/hc_phase_test" --baseline "$root/base/kern/die.cubin" --candidate "$root/single/kern/die.cubin" > "$out/single-hc.log" 2>&1
DIE_CUBIN="$root/single/kern/die.cubin" timeout -k 5 180 "$root/base/kern/qsa_direct_test" 65 8 4 > "$out/single-qsa.log" 2>&1
printf 'single-flag HC and QSA checks passed\n'
