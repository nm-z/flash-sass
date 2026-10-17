#!/bin/bash
# draft-conflicting.sh: convert every open, ready-for-review pull request that has merge conflicts with its base to a draft, with one comment.
#   [DRY_RUN=1] [GH_REPO=owner/name] .github/scripts/draft-conflicting.sh      (needs gh; GH_TOKEN in Actions)
# GitHub computes mergeability lazily after a push, so a PR that reports UNKNOWN is asked again (up to about a minute) and never guessed.
set -euo pipefail
REPO=${GH_REPO:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}
MARK='<!-- draft-on-conflict -->'
# a failed listing must fail the sweep (a failure inside the for-list below would be swallowed)
ready=$(gh pr list --repo "$REPO" --state open --json number,isDraft --jq '.[] | select(.isDraft | not) | .number')
for n in $ready; do
	state=UNKNOWN
	for try in 1 2 3 4 5 6 7 8 9 10; do
		state=$(gh pr view "$n" --repo "$REPO" --json mergeable --jq .mergeable)
		[ "$state" != UNKNOWN ] && break
		timeout 6 tail -f /dev/null || true
	done
	case "$state" in
	CONFLICTING)
		if [ "${DRY_RUN:-0}" = 1 ]; then echo "#$n conflicts: would convert to draft"; continue; fi
		gh pr ready "$n" --repo "$REPO" --undo > /dev/null
		# one comment per conflict episode: skip when the newest comment is already ours
		last=$(gh pr view "$n" --repo "$REPO" --json comments --jq '.comments | last | .body // ""')
		case "$last" in *"$MARK"*) ;; *)
			gh pr comment "$n" --repo "$REPO" --body "$MARK
Converted to draft: this pull request has merge conflicts with its base branch. Rebase it onto the base branch (or merge the base into it), resolve the conflicts, and mark it ready for review again; if it still conflicts it will be converted again." > /dev/null ;;
		esac
		echo "#$n conflicts: converted to draft" ;;
	UNKNOWN) echo "#$n mergeability still unknown after retries: left as is" ;;
	*) echo "#$n $state: left as is" ;;
	esac
done
