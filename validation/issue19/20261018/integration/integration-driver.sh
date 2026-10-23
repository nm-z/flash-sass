#!/usr/bin/env bash
set -euo pipefail
issue19_out=/home/nate/codex/flash-pr44-integration-evidence-20261018
trap 'printf "%s\n" "$?" > "$issue19_out/integration-driver.exit"' EXIT
cd /home/nate/codex/flash-pr44-integration-20261018
ruby validation/issue19/run.rb --mode gpu \
  --out "$issue19_out" \
  --gpu GPU-431371bd-b482-96b2-4b70-04f83c6d075b \
  --lock-dir /home/nate/codex/asm-port \
  --engine-lock /tmp/flash-sass-engine-validation.lock \
  --core-pid 3575268 --socket /tmp/flash.sock
ruby validation/issue19/run.rb --mode model \
  --out "$issue19_out" \
  --core-pid 3575268 --socket /tmp/flash.sock \
  --checkpoint /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --lock-dir /home/nate/codex/asm-port \
  --engine-lock /tmp/flash-sass-engine-validation.lock \
  --positions 128 --capacity 65552 \
  --corpus /home/nate/codex/flash-pr44-evidence-20261018/model-corpus.txt
