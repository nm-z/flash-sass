#!/usr/bin/env bash
# Feeds completed cloud trial runs (workflows "camber trial" and "azure trial" on nm-z/recipe-dev)
# to the Recipe Machine: each completed run's artifact is downloaded once and every
# evidence/<device>-run-<a>-<b>.txt in it is copied to the machine's inbox, whose ingester tags the
# device by the prefix before "-run". Runs already taken are listed in a state file.
set -uo pipefail
state_dir=$HOME/.local/state/recipe-machine
inbox=$state_dir/inbox
taken=$state_dir/ci-trials-taken
log=$state_dir/ci-trials.log
work=$state_dir/ci-trials
gh=$HOME/.local/bin/gh
mkdir -p "$inbox" "$work"
touch "$taken"
say() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >> "$log"; }
while :; do
	for workflow in camber-trial.yml azure-trial.yml; do
		runs=$("$gh" run list -R nm-z/recipe-dev --workflow "$workflow" --status completed --limit 20 --json databaseId --jq '.[].databaseId' 2>>"$log") || continue
		for run in $runs; do
			grep -qx "$run" "$taken" && continue
			dir=$work/$run
			rm -rf "$dir"
			if ! "$gh" run download "$run" -R nm-z/recipe-dev -D "$dir" 2>>"$log"; then
				say "run $run: no artifact yet or download failed"
				continue
			fi
			count=0
			while IFS= read -r file; do
				name=$(basename "$file")
				cp "$file" "$inbox/$name.part" && mv "$inbox/$name.part" "$inbox/$name"
				count=$((count + 1))
				say "run $run: $name packets=$(grep -c '^RECIPE FAILURE BEGIN$' "$file")"
			done < <(find "$dir" -type f -name '*-run-*.txt')
			say "run $run ($workflow): $count evidence files"
			echo "$run" >> "$taken"
			rm -rf "$dir"
		done
	done
	sleep 120
done
