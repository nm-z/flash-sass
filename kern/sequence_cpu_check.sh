#!/usr/bin/env bash
# CPU regressions for scheduling and acceptance logic. This never runs CUDA.
set -euo pipefail
if (($# != 1)); then
    printf 'usage: %s NEW_OUTPUT_DIR\n' "$0" >&2
    exit 2
fi
src=$(cd -- "$(dirname -- "$0")/.." && pwd)
out=$(realpath -m -- "$1")
git -C "$src" diff --quiet HEAD -- kern ref || {
    printf 'Commit source changes before recording CPU evidence.\n' >&2
    exit 1
}
mkdir -p -- "$(dirname -- "$out")"
mkdir -- "$out"
revision=$(git -C "$src" rev-parse HEAD)
printf '%s\n' "$revision" > "$out/source-revision.txt"
hostname > "$out/hostname.txt"
date -u +%FT%TZ > "$out/started-utc.txt"
make -C "$src/kern" seqsched_test prefill_plan_test sequence_http_test > "$out/build.log" 2>&1
"$src/kern/seqsched_test" > "$out/seqsched.log" 2>&1
"$src/kern/prefill_plan_test" > "$out/prefill-plan.log" 2>&1
perl "$src/kern/sequence_check.pl" --self-test > "$out/predicates.json"
"$src/kern/sequence_http_test" > "$out/http.log" 2>&1

# A build failure is never evidence that the regression test is sensitive.
mkdir "$out/http-mutant"
cp "$src/kern/sequence_http_test.c" "$out/http-mutant/sequence_http_test.c"
perl -0777 - "$src/kern/flashd.c" > "$out/http-mutant/flashd.c" <<'PERL'
use strict;
use warnings;
my $text = <>;
my $n = ($text =~ s/    if \(\*gone \|\| client_gone\(client_fd\)\) \{ \*gone = 1; return NULL; \}/    (void)client_fd; (void)gone;/g);
die "HTTP reader mutation did not match exactly once\n" unless $n == 1;
print $text;
PERL
set +e
diff -u --label a/kern/flashd.c --label b/kern/flashd.c "$src/kern/flashd.c" "$out/http-mutant/flashd.c" > "$out/http-mutation.patch"
diff_status=$?
set -e
[[ $diff_status == 1 ]]
compile=("${CC:-cc}" -std=gnu99 -O2 -Wall -Wextra -o "$out/http-mutant/sequence_http_test"
    "$out/http-mutant/sequence_http_test.c" "$src/kern/tok.c" "$src/kern/jinja.c"
    "$src/kern/json.c" "$src/ref/gguf.c" "-I$src/kern" "-I$src/ref" -lm -pthread)
printf '%q ' "${compile[@]}" > "$out/http-mutant-build-command.sh"
printf '\n' >> "$out/http-mutant-build-command.sh"
"${compile[@]}" > "$out/http-mutant-build.log" 2>&1
set +e
"$out/http-mutant/sequence_http_test" > "$out/http-mutant.log" 2>&1
mutant_status=$?
set -e
printf '%s\n' "$mutant_status" > "$out/http-mutant-exit-code.txt"
[[ $mutant_status == 1 ]]
rg -q 'failed: generation_line.*== NULL' "$out/http-mutant.log"
sha256sum "$src/kern/sequence_check.pl" "$src/kern/sequence_http_test.c" \
    "$src/kern/flashd.c" "$out/http-mutant/flashd.c" > "$out/source-sha256.txt"
jq -n --arg revision "$revision" --argjson mutant_status "$mutant_status" \
    '{kind:"CPU regressions; no GPU/model execution",source_revision:$revision,
      scheduler:"passed",prefill_planner:"passed",predicates:7,http_reader:"passed",
      http_no_disconnect_mutant_exit:$mutant_status,result:"passed"}' > "$out/result.json"
cat "$out/result.json"
