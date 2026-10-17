#!/bin/bash
# depth.sh <filler paragraphs> [needle at percent] [url]: decode speed at a context depth. Like needle.sh (numbered filler paragraphs with a code buried at a depth) but the model is also asked
# for a long, fixed continuation (the numbers 1 to 40), so the decode rate is taken over ~100 tokens instead of 6 intervals. Prints one line:
#   N paragraphs  prompt tokens  prefill tok/s  decode tok/s (decode_intervals / decode_ms)  PASS|FAIL (the code is in the answer)
set -eu
N=${1:-45}; PCT=${2:-50}; URL=${3:-http://127.0.0.1:18090}
[[ "$N" =~ ^[0-9]+$ && "$PCT" =~ ^[0-9]+$ ]] && (( N > 0 && PCT <= 100 )) || { echo "depth: invalid paragraph count or percentage" >&2; exit 2; }
if [[ -n "${DEPTH_OUT:-}" ]]; then OUT=$DEPTH_OUT; mkdir -p "$OUT"; else OUT=$(mktemp -d /tmp/flash-depth.XXXXXX); fi
echo "depth: raw request/response: $OUT" >&2
CODE=$(printf '%06x' $(( (RANDOM << 8) ^ RANDOM )))
AT=$(( N * PCT / 100 )); (( AT > 0 )) || AT=1
DOC=$(awk -v n="$N" -v at="$AT" -v code="$CODE" 'BEGIN {
	split("river market lantern orchard harbor meadow quarry bridge garden window kettle ladder pigeon violin button candle", w, " ");
	srand(7);
	for (i = 1; i <= n; i++) {
		printf "Paragraph %d. ", i;
		for (j = 0; j < 4; j++) printf "The %s near the %s was quiet while the %s waited by the %s. ", w[1+int(rand()*16)], w[1+int(rand()*16)], w[1+int(rand()*16)], w[1+int(rand()*16)];
		if (i == at) printf "The access code of the vault is %s. ", code;
		printf "\n";
	}
}')
REQ=$(jq -n --arg doc "$DOC" '{messages:[{role:"user",content:($doc + "\nFirst give the access code of the vault (six hex digits), then on a new line write the numbers 1 to 40 separated by single spaces and nothing else.")}],max_tokens:140,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
printf '%s\n' "$REQ" > "$OUT/request.json"
printf '%s\n' "$CODE" > "$OUT/expected-code.txt"
set +e
HTTP=$(curl -sS -m "${DEPTH_TIMEOUT:-7200}" -o "$OUT/response.json" -w '%{http_code}' "$URL/v1/chat/completions" -H 'Content-Type: application/json' --data-binary "@$OUT/request.json" 2> "$OUT/transport-error.txt")
STATUS=$?
set -e
printf '%s\n' "$HTTP" > "$OUT/http-status.txt"
if (( STATUS != 0 )); then
  cat "$OUT/transport-error.txt" >&2
  echo "depth: transport failed (curl exit $STATUS)" >&2; exit 2
fi
case "$HTTP" in 2[0-9][0-9]) ;; *) echo "depth: HTTP $HTTP; response saved in $OUT" >&2; exit 2 ;; esac
set +e
python3 "$(dirname "$0")/depth_check.py" "$OUT/response.json" "$CODE" "$N"
STATUS=$?
set -e
printf '%s\n' "$STATUS" > "$OUT/exit-status.txt"
exit "$STATUS"
