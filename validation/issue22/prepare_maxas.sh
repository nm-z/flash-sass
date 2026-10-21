#!/usr/bin/env bash
set -euo pipefail

if [[ $# != 1 ]]; then
    printf 'usage: %s MAXAS_DIRECTORY\n' "$0" >&2
    exit 2
fi

issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_maxas=$1
issue22_revision=54eda7af086a46c9dae1688b691968235d560164
issue22_url=https://github.com/NervanaSystems/maxas.git

if [[ ! -d $issue22_maxas/.git ]]; then
    [[ ! -e $issue22_maxas ]] || { printf 'MaxAs path exists without Git metadata: %s\n' "$issue22_maxas" >&2; exit 2; }
    git clone --depth 1 "$issue22_url" "$issue22_maxas"
fi

issue22_remote=$(git -C "$issue22_maxas" remote get-url origin)
[[ $issue22_remote == "$issue22_url" ]] || { printf 'unexpected MaxAs origin: %s\n' "$issue22_remote" >&2; exit 2; }
issue22_current=$(git -C "$issue22_maxas" rev-parse HEAD)
if [[ $issue22_current != "$issue22_revision" ]]; then
    git -C "$issue22_maxas" fetch --depth 1 origin "$issue22_revision"
    git -C "$issue22_maxas" checkout --detach "$issue22_revision"
fi

mkdir -p -- "$issue22_maxas/lib/MaxAs"
install -m 0644 "$issue22_repo/maxas-patch/MaxAs.pm" "$issue22_maxas/lib/MaxAs/MaxAs.pm"
install -m 0644 "$issue22_repo/maxas-patch/MaxAsGrammar.pm" "$issue22_maxas/lib/MaxAs/MaxAsGrammar.pm"
install -m 0644 "$issue22_repo/maxas-patch/Cubin.pm" "$issue22_maxas/lib/MaxAs/Cubin.pm"

printf 'maxas_revision=%s\n' "$issue22_revision"
sha256sum "$issue22_repo/maxas-patch/MaxAs.pm" \
    "$issue22_repo/maxas-patch/MaxAsGrammar.pm" \
    "$issue22_repo/maxas-patch/Cubin.pm"
