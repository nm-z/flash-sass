#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'usage: %s COMPACT_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi

issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_compact=$1
issue22_full="${issue22_compact}-full"
[[ ! -e $issue22_compact && ! -e $issue22_full ]] || {
    printf 'output already exists: %s or %s\n' "$issue22_compact" "$issue22_full" >&2
    exit 2
}

bash "$issue22_repo/validation/issue22/run_cpu.sh" "$issue22_full"
mkdir -p -- "$issue22_compact/synthetic-inputs" "$issue22_compact/synthetic-replay"
cp -- "$issue22_full/complete.json" "$issue22_full/commands.txt" \
    "$issue22_full/executable.sha256" "$issue22_full/exits.tsv" \
    "$issue22_full/runtime.txt" "$issue22_full/source-after.sha256" \
    "$issue22_full/source-before.sha256" "$issue22_full/source-commit.txt" \
    "$issue22_full/source-status.txt" "$issue22_compact"/
cp -- "$issue22_full"/*.log "$issue22_compact"/
cp -a -- "$issue22_full/placement" "$issue22_full/recovered-reference" \
    "$issue22_compact"/
cp -- "$issue22_full/synthetic-inputs/complete.json" \
    "$issue22_full/synthetic-lengths.txt" "$issue22_compact/synthetic-inputs/"
for issue22_report in "$issue22_full"/synthetic-replay/n*.json; do
    case "$issue22_report" in *.pending.*) continue ;; esac
    cp -- "$issue22_report" "$issue22_compact/synthetic-replay/"
done

printf 'Compact CPU report written to %s; full raw fixtures remain in the DSH workspace.\n' \
    "$issue22_compact"
