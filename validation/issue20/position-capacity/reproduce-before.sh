#!/usr/bin/env bash
# Run from the repository root. Extract and execute the actual baseline formula.
set -eu
base=${1:-bf53cfd7c2b89972fa6adb2235531afa2da05a63}
out=${2:?usage: reproduce-before.sh COMMIT OUTPUT_DIRECTORY}
mkdir -p "$out"
git show "${base}:kern/qsa_test.c" > "$out/qsa_test_before.c"
{
  printf '#include <stdint.h>\n#include <stdio.h>\n'
  awk '/^static uint32_t rs =/ || /^static uint32_t urand\(void\)/ {print}' "$out/qsa_test_before.c"
  printf 'int main(void) { unsigned nv = 65536;\n'
  awk '/^static int test_idxscore\(/ {getline; print; getline; print; exit}' "$out/qsa_test_before.c"
  printf '(void)ngrp; printf("blocks=%%u position=%%u visible=%%u capacity=262144\\n", nv, t, t + 1); return t < 262144u ? 0 : 1; }\n'
} > "$out/idxscore_before.c"
cc -std=gnu99 -O2 -Wall -Wextra "$out/idxscore_before.c" -o "$out/idxscore_before"
"$out/idxscore_before"
