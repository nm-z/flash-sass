#!/usr/bin/env bash
set -euo pipefail
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
archive=${1:-"$here/historical"}
fixture=$(mktemp -d)
trap 'rm -rf -- "$fixture"' EXIT
for storage in local nfs; do
    for suffix in csv log; do
        cp "$archive/flash-ple-7f37f6b-$storage.$suffix" "$fixture/"
    done
done
perl "$here/replay_historical.pl" "$fixture" > "$fixture/good.json"
[[ $(jq -r '.data.local.batches + .data.nfs.batches' "$fixture/good.json") == 4608 ]]
reject() {
    local reason=$1
    if perl "$here/replay_historical.pl" "$fixture" > "$fixture/result.json" 2> "$fixture/error"; then
        printf 'replay accepted %s\n' "$reason" >&2
        exit 1
    fi
}
csv="$fixture/flash-ple-7f37f6b-local.csv"
original="$archive/flash-ple-7f37f6b-local.csv"
perl -i -pe 'if($.==2){my @x=split /,/; $x[10]=1; $_=join(",",@x)}' "$csv"
reject read_error
cp "$original" "$csv"
sed -i '$d' "$csv"
reject truncated_run
cp "$original" "$csv"
perl -i -pe 'if(/^warm,0,16,/){my @x=split /,/; $x[8]="0000000000000000"; $_=join(",",@x)}' "$csv"
reject mismatched_warm_outputs
cp "$original" "$csv"
sed -i 's/^pass=1 /pass=0 /' "$fixture/flash-ple-7f37f6b-local.log"
reject failed_final_status
printf 'historical replay: complete raw files pass; read error, truncation, warm mismatch and failed final status are rejected\n'
