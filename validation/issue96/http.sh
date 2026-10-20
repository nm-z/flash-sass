#!/bin/bash
# Private HTTP integration using the existing image_url client and shared core.
set -euo pipefail
: "${GPUQ_JOB_ID:?run through gpuq}"
: "${GPUQ_RUN_DIR:?run through gpuq}"
projector=${1:?projector}
text_model=${2:?text GGUF}
image=${3:?image}
work="$GPUQ_RUN_DIR/vision-http"
mkdir -p "$work"
vis_pid= http_pid=
cleanup() {
  for pid in "$http_pid" "$vis_pid"; do
    if test -n "$pid"; then kill "$pid" 2>/dev/null || true; wait "$pid" 2>/dev/null || true; fi
  done
}
trap cleanup EXIT
make -C kern flashd
FLASHVIS_DEVICE=auto ./kern/flashvis "$projector" "$text_model" "$work/vis.sock" 10 >"$work/encoder.log" 2>&1 &
vis_pid=$!
for ((i=0; i<600; i++)); do
  test ! -S "$work/vis.sock" || break
  kill -0 "$vis_pid"
  sleep 0.1
done
test -S "$work/vis.sock"
FLASHVIS_SOCK="$work/vis.sock" FLASHD_IMGDIR="$work/image-cache" ./kern/flashd "$text_model" /tmp/flash.sock 127.0.0.1:18196 >"$work/http.log" 2>&1 &
http_pid=$!
for ((i=0; i<600; i++)); do
  if curl -fsS http://127.0.0.1:18196/queue >/dev/null 2>&1; then break; fi
  kill -0 "$http_pid"
  sleep 0.1
done
result=$(timeout 300 bash kern/vision_test.sh "$image" "Describe the resource monitors shown in this image in two sentences." http://127.0.0.1:18196)
printf '%s\n' "$result"
printf '%s\n' "$result" | rg '^answer: .+'
rg 'encode [0-9.]+ s, backend CUDA' "$work/encoder.log"
! rg 'switching to CPU|ready, backend CPU' "$work/encoder.log"
printf '%s\n' '{"event":"image_http_integration_passed","image_cache":"private-empty","encoder_backend":"CUDA"}'
