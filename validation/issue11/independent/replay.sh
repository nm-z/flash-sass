#!/usr/bin/env bash
# Replay the measured numerical comparison without a model, GPU or Python.
# Exit 3 is an actual numerical failure, not a successful checkpoint gate.
set -euo pipefail
if (($# != 1)); then
    printf 'usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
here=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
root=$(cd -- "$here/../../.." && pwd)
(cd -- "$here" && sha256sum --quiet -c SHA256SUMS)
mkdir -- "$1"
out=$(cd -- "$1" && pwd)
tar --no-same-owner -xzf "$here/captures.tar.gz" -C "$out"
(cd -- "$out" && sha256sum --quiet -c SHA256SUMS)
"${CC:-cc}" -std=gnu11 -O2 -fno-fast-math -ffp-contract=off -frounding-math \
    -Wall -Wextra -Werror "$root/kern/qsa_logits_check.c" -lm -o "$out/qsa_logits_check"
set +e
"$out/qsa_logits_check" \
    "$out/reference/ids.u32" "$out/historical-candidate/ids.u32" \
    "$out/reference/logits.f32" "$out/historical-candidate/logits.f32" \
    4 248320 "$out/numerical.json" > "$out/compare.stdout" 2> "$out/compare.stderr"
rc=$?
set -e
printf '%s\n' "$rc" > "$out/comparison-exit.txt"
cat "$out/compare.stderr"
printf 'Comparison exit=%s; candidate is historical; original-checkpoint provenance is unverified.\n' "$rc"
exit "$rc"
