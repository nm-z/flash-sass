#!/bin/bash
set -euo pipefail
source_dir=$(readlink -f "$1")
revision=$2
expected=$3
[[ $revision =~ ^[0-9a-f]{40}$ ]] || exit 2
actual=$(sha256sum /usr/local/bin/gpuq | cut -d' ' -f1)
[ "$actual" = "$expected" ] || { echo 'gpuq changed since inspection; installation aborted' >&2; exit 3; }
release=/usr/local/lib/gpuq/releases/$revision
install -d -m 755 "$release"
install -m 644 "$source_dir"/*.rb "$source_dir"/*.sql "$source_dir"/*.json "$release/"
install -m 644 "$source_dir/README.md" "$release/README.md"
if [ ! -e /usr/local/lib/gpuq/original-gpuq ]; then
  install -m 755 /usr/local/bin/gpuq /usr/local/lib/gpuq/original-gpuq
fi
launcher=$(mktemp /usr/local/bin/.gpuq.XXXXXX)
trap 'rm -f "$launcher"' EXIT
printf '#!/bin/bash\nexec /usr/bin/ruby %q "$@"\n' "$release/gpuq.rb" > "$launcher"
chmod 755 "$launcher"
mv -f "$launcher" /usr/local/bin/gpuq
ln -sfn "$release" /usr/local/lib/gpuq/current
