#!/bin/bash
# pr-audit.sh: check one pull request against the repository rules. Exit 0 = clean, 1 = a rule is broken.
#   PR=<number> [GH_REPO=owner/name] .github/scripts/pr-audit.sh      (needs gh and jq; GH_TOKEN in Actions, `gh auth login` locally)
# Rules (failures):
#   1. no files in a language this repository does not use: Python, C++, C#, JavaScript/TypeScript, Java, PHP, Kotlin, Swift
#      (by extension for every file the PR adds or changes, and by shebang for the lines it adds)
#   2. the description links an issue that exists: "Fixes|Closes|Resolves|Refs #N" (a pull request number does not count)
# Notes (printed, never fail): draft state, merge conflicts, python invoked from an added script or Makefile line.
set -euo pipefail
: "${PR:?set PR=<pull request number>}"
REPO=${GH_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}
BAD_EXT='\.(py|pyw|cpp|cc|cxx|c\+\+|hpp|hh|hxx|cs|js|mjs|cjs|ts|tsx|java|kt|swift|php)$'
WORK=$(mktemp -d); trap 'rm -rf "$WORK"' EXIT
fail=0
note() { printf '%s\n' "$*"; }

gh api "repos/$REPO/pulls/$PR" > "$WORK/pr.json"
# every file of the PR (the pulls/N/files endpoint pages through all of them; `gh pr view --json files` stops at 100)
gh api --paginate "repos/$REPO/pulls/$PR/files" --jq '.[] | [.status, .filename] | @tsv' > "$WORK/files.tsv"
note "PR #$PR: $(jq -r .title "$WORK/pr.json")"
note "  $(wc -l < "$WORK/files.tsv") files, draft=$(jq -r .draft "$WORK/pr.json"), mergeable=$(jq -r '.mergeable // "unknown"' "$WORK/pr.json")"

# 1. files in a language we do not use (a removed file is fine: that is how a port finishes)
bad=$(awk -F'\t' '$1 != "removed" {print $2}' "$WORK/files.tsv" | grep -E "$BAD_EXT" || true)
if [ -n "$bad" ]; then
	fail=1
	note "FAIL banned-language files (port to C, Perl or Bash, or drop the file and keep the results):"
	printf '%s\n' "$bad" | sed 's/^/    /'
fi
# the same by shebang and by python called from an added script line, on the PR's diff
have_diff=1
gh api "repos/$REPO/pulls/$PR" -H 'Accept: application/vnd.github.v3.diff' > "$WORK/pr.diff" 2> "$WORK/diff.err" || have_diff=0
if [ "$have_diff" = 0 ]; then
	# GitHub refuses the diff of a PR with more than 300 files: read the first line of every added script outside validation/ instead
	note "NOTE the PR diff is unavailable ($(head -c 80 "$WORK/diff.err")); checking script shebangs file by file"
	head_sha=$(jq -r .head.sha "$WORK/pr.json"); : > "$WORK/pr.diff"
	awk -F'\t' '$1 != "removed" && $2 !~ /^validation\// && ($2 ~ /\.(sh|pl|pm|rb|lua)$/ || $2 !~ /\.[A-Za-z0-9]+$/) {print $2}' "$WORK/files.tsv" | head -60 | while read -r f; do
		first=$(gh api "repos/$REPO/contents/$f?ref=$head_sha" -H 'Accept: application/vnd.github.raw' 2> /dev/null | head -1 || true)
		printf '+++ b/%s\n+%s\n' "$f" "$first" >> "$WORK/pr.diff"
	done
fi
shebang=$(awk '/^\+\+\+ b\//{f=substr($0,7)} /^\+#!.*(python|node|deno|bun|php)/{print f": "substr($0,2)}' "$WORK/pr.diff" | head -20)
if [ -n "$shebang" ]; then
	fail=1
	note "FAIL added shebang for a language we do not use:"
	printf '%s\n' "$shebang" | sed 's/^/    /'
fi
callpy=$(awk '/^\+\+\+ b\//{f=substr($0,7)} f !~ /pr-audit\.sh$/ && f ~ /(\.sh|\.pl|\.pm|Makefile|\.mk|\.ya?ml)$/ && /^\+[^#]*python[0-9.]*[ \t]/ && !/^\+\+\+/{print f": "substr($0,2,120)}' "$WORK/pr.diff" | head -10)
if [ -n "$callpy" ]; then
	note "NOTE python is invoked from an added line (check it is not a requirement of the PR):"
	printf '%s\n' "$callpy" | sed 's/^/    /'
fi

# 2. the description links an existing issue
refs=$(jq -r '.body // ""' "$WORK/pr.json" | grep -oiE '(fix(es|ed)?|close[sd]?|resolve[sd]?|refs?)[[:space:]]+(([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)?#[0-9]+)' | grep -oE '#[0-9]+' | tr -d '#' | sort -un || true)
issue_ok=0
for n in $refs; do
	[ "$n" = "$PR" ] && continue
	if gh api "repos/$REPO/issues/$n" --jq 'select(.pull_request == null) | .number' 2> /dev/null | grep -q .; then issue_ok=1; note "  issue #$n linked"; fi
done
if [ "$issue_ok" = 0 ]; then
	fail=1
	note "FAIL no linked issue: the description must contain \"Fixes #N\", \"Closes #N\" or \"Refs #N\" for an issue (not a pull request)"
fi

[ "$(jq -r .mergeable_state "$WORK/pr.json")" = dirty ] && note "NOTE merge conflicts with the base branch"
if [ "$fail" = 0 ]; then note "OK"; else note "FAILED"; fi
exit "$fail"
