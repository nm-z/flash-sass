#!/usr/bin/env bash
# Build deliberately broken libraries for the same real-model acceptance checks.
set -euo pipefail
if (($# != 2)) || [[ $1 != serial && $1 != cancellation ]]; then
    printf 'usage: %s serial|cancellation NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
mode=$1
out=$(realpath -m -- "$2")
src=$(cd -- "$(dirname -- "$0")/.." && pwd)
revision=$(git -C "$src" rev-parse HEAD)
git -C "$src" diff --quiet HEAD -- kern ref || { printf 'Commit source changes before building a control.\n' >&2; exit 1; }
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
mkdir -- "$out/source"
git -C "$src" archive "$revision" | tar -x -C "$out/source"
perl -0777 - "$mode" "$src/kern/engine.c" > "$out/source/kern/engine.c" <<'PERL'
use strict;
use warnings;
my $mode = shift @ARGV;
my $source = <>;
if ($mode eq 'serial') {
    my $n = ($source =~ s/int engine_concurrent\(void\) \{ return 1; \}/int engine_concurrent(void) { return 0; }/g);
    die "serial mutation no longer matches exactly once\n" unless $n == 1;
} else {
    my $n = ($source =~ s/\.alive = client_gone,/.alive = NULL,/g);
    die "cancellation callback mutation no longer matches exactly once\n" unless $n == 1;
    $n = ($source =~ s/return fflush\(out\) != 0 \|\| ferror\(out\);/fflush(out); clearerr(out); return 0;/g);
    die "token cancellation mutation no longer matches exactly once\n" unless $n == 1;
}
print $source;
PERL
set +e
diff -u --label a/kern/engine.c --label b/kern/engine.c "$src/kern/engine.c" "$out/source/kern/engine.c" > "$out/mutation.patch"
diff_status=$?
set -e
[[ $diff_status == 1 ]] || exit 1
mutant_revision=$revision-mutant-$mode
build_command=(make -C "$out/source/kern" "CUDA=${CUDA:-/opt/cuda}" "FLASH_REV=$mutant_revision" libengine.so)
printf '%q ' "${build_command[@]}" > "$out/build-command.sh"
printf '\n' >> "$out/build-command.sh"
set +e
"${build_command[@]}" > "$out/build.stdout" 2> "$out/build.stderr"
status=$?
set -e
printf '%s\n' "$status" > "$out/build-exit-code.txt"
if ((status)); then cat "$out/build.stderr" >&2; exit "$status"; fi
sha256sum "$out/source/kern/engine.c" "$out/source/kern/libengine.so" > "$out/sha256.txt"
jq -n --arg mode "$mode" --arg revision "$revision" --arg mutant_revision "$mutant_revision" \
    --arg library "$out/source/kern/libengine.so" --arg hostname "$(hostname)" \
    '{control:$mode,source_revision:$revision,mutant_revision:$mutant_revision,library:$library,hostname:$hostname}' > "$out/control.json"
printf 'Built %s\n' "$out/source/kern/libengine.so"
printf 'Run sequence_validate.sh with SEQUENCE_LIBRARY set to this library, SEQUENCE_REVISION=%s, and SEQUENCE_CASE=%s. The checker must fail on the named assertion.\n' "$mutant_revision" "${mode/serial/progress}"
