#!/usr/bin/env bash
# Reproduce the recorded numerical failure and the seven rejection controls.
# This downloads a pinned historical target artifact; it launches no GPU work.
set -euo pipefail
(($# == 1)) || { printf 'usage: replay.sh NEW_OUTPUT_DIRECTORY\n' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
mkdir -- "$1"
out=$(cd -- "$1" && pwd)
(cd -- "$script_dir" && sha256sum -c SHA256SUMS)
(cd -- "$repo_root" && sha256sum -c "$script_dir/source-sha256.txt")
tar -xzf "$script_dir/reference-capture.tar.gz" -C "$out"
mkdir "$out/candidate"
source_base=https://raw.githubusercontent.com/nm-z/flash-sass/fcf435f728760b64de30214478524666093468ec/validation/issue22/archy-06968d1/gpu
for file in ids.u32 capture_info.json logits.f32.gz; do
    curl --fail --location --silent --show-error "$source_base/t4-budget512-kv1-raw/$file" -o "$out/candidate/$file"
done
curl --fail --location --silent --show-error "$source_base/t4-budget512-kv1-frozen-manifest.json" -o "$out/candidate/frozen-manifest.json"
gzip -dc "$out/candidate/logits.f32.gz" > "$out/candidate/logits.f32"
make -C "$repo_root/kern" qsa_logits_check
set +e
perl "$repo_root/kern/qsa_model_compare.pl" --historical-v1 "$out/capture" \
    "$out/candidate" "$out/candidate/frozen-manifest.json" "$out/comparison" \
    > "$out/compare.stdout" 2> "$out/compare.stderr"
comparison_rc=$?
set -e
printf '%s\n' "$comparison_rc" > "$out/comparison-exit-code.txt"
cat "$out/compare.stdout" "$out/compare.stderr"
((comparison_rc == 3)) || { printf 'expected the recorded numerical failure (exit 3), got %s\n' "$comparison_rc" >&2; exit 1; }
perl "$repo_root/kern/qsa_model_compare_test.pl" "$out/capture" "$out/candidate" \
    "$out/candidate/frozen-manifest.json" > "$out/negative-controls.tap"
cat "$out/negative-controls.tap"
printf 'REPLAY_CONFIRMED_NUMERICAL_FAILURE: issue #11 is not resolved by this diagnostic\n'
