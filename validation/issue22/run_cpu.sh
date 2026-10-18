#!/usr/bin/env bash
# Reproducible CPU checks only. These do not execute the engine or certify quality.
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_out=$1
if [[ -e $issue22_out ]]; then
    printf 'output already exists: %s\n' "$issue22_out" >&2
    exit 2
fi
mkdir -p -- "$(dirname -- "$issue22_out")"
mkdir -- "$issue22_out"
issue22_out=$(cd -- "$issue22_out" && pwd)
cd -- "$issue22_repo"
export LC_ALL=C
issue22_sources=(
    kern/model_quality.pl kern/model_quality_test.pl
    kern/qsa_quality.c kern/qsa_quality_replay.pl kern/qsa_quality_test.pl
    kern/model_layer_capture.h kern/model_layer_capture_test.c
    kern/model_layer_capture_test.pl kern/model_capture.h kern/model_test.c
    kern/model_reference.c kern/Makefile validation/issue22/replay_placement.sh
    validation/issue22/run_cpu.sh validation/issue22/replay_reference.sh
)
sha256sum -- "${issue22_sources[@]}" > "$issue22_out/source-before.sha256"
git rev-parse HEAD > "$issue22_out/source-commit.txt"
git status --porcelain=v1 > "$issue22_out/source-status.txt"
{
    date -u +%FT%TZ
    hostname
    uname -a
    cc --version
    perl -V:version -V:archname -V:nvtype
    make --version
} > "$issue22_out/runtime.txt"
: > "$issue22_out/commands.txt"
: > "$issue22_out/exits.tsv"
issue22_run() {
    local issue22_label=$1 issue22_rc
    shift
    printf '%s: ' "$issue22_label" >> "$issue22_out/commands.txt"
    printf '%q ' "$@" >> "$issue22_out/commands.txt"
    printf '\n' >> "$issue22_out/commands.txt"
    set +e
    "$@" > "$issue22_out/$issue22_label.log" 2>&1
    issue22_rc=$?
    set -e
    printf '%s\t%s\n' "$issue22_label" "$issue22_rc" >> "$issue22_out/exits.tsv"
    if (( issue22_rc != 0 )); then
        cat -- "$issue22_out/$issue22_label.log" >&2
        printf 'FAIL: %s (exit %s)\n' "$issue22_label" "$issue22_rc" >&2
        exit "$issue22_rc"
    fi
    printf 'PASS: %s\n' "$issue22_label"
}
issue22_run build make -B -C kern model_quality qsa_quality model_layer_capture_test
issue22_run comparator perl kern/model_quality_test.pl
issue22_run layer-export perl kern/model_layer_capture_test.pl
issue22_run analytic-indexer kern/qsa_quality --selftest
issue22_run frozen-indexer perl kern/qsa_quality_test.pl
issue22_run placement-replay bash validation/issue22/replay_placement.sh "$issue22_out/placement"
issue22_run recovered-reference bash validation/issue22/replay_reference.sh "$issue22_out/recovered-reference"
issue22_run synthetic-inputs kern/qsa_quality --synthetic "$issue22_out/synthetic-inputs"
mkdir -- "$issue22_out/synthetic-replay"
jq -er '.snapshot_lengths | if type == "array" and length > 0 then .[] else error("missing synthetic lengths") end' \
    "$issue22_out/synthetic-inputs/complete.json" > "$issue22_out/synthetic-lengths.txt"
while IFS= read -r issue22_n; do
    issue22_run "synthetic-n$issue22_n" perl kern/qsa_quality_replay.pl \
        "$issue22_out/synthetic-inputs" "$issue22_n" \
        "$issue22_out/synthetic-replay/n$issue22_n.json"
done < "$issue22_out/synthetic-lengths.txt"
sha256sum -- "${issue22_sources[@]}" > "$issue22_out/source-after.sha256"
if ! cmp -s -- "$issue22_out/source-before.sha256" "$issue22_out/source-after.sha256"; then
    printf 'source changed during checks; no completion report\n' >&2
    exit 2
fi
sha256sum kern/model_quality kern/qsa_quality kern/model_layer_capture_test \
    > "$issue22_out/executable.sha256"
jq -n --arg revision "$(cat "$issue22_out/source-commit.txt")" \
    --slurpfile fixture "$issue22_out/synthetic-inputs/complete.json" \
    '{schema:"flash-sass.issue22-cpu-checks.v1",source_revision:$revision,
      source_hashes_unchanged:true,cpu_checks_pass:true,
      synthetic_lengths:$fixture[0].snapshot_lengths,
      gpu_execution:false,independent_checkpoint_evidence:false,
      original_checkpoint_quality:null,issue22_resolved:false}' \
    > "$issue22_out/complete.json"
printf 'CPU checks complete; GPU and original-checkpoint quality remain separate.\n'
