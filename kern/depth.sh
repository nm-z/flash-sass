#!/bin/bash
# depth.sh <filler paragraphs> [needle at percent] [url]: decode speed at a context depth. Like needle.sh (numbered filler paragraphs with a code buried at a depth) but the model is also asked
# for a long, fixed continuation (the numbers 1 to 40), so the decode rate is taken over ~100 tokens instead of 6 intervals. Prints one line:
#   N paragraphs  prompt tokens  prefill tok/s  decode tok/s (decode_intervals / decode_ms)  PASS|FAIL (the code is in the answer)
N=${1:-45}; PCT=${2:-50}; URL=${3:-http://127.0.0.1:18090}
CODE=$(printf '%06x' $(( (RANDOM << 8) ^ RANDOM )))
AT=$(( N * PCT / 100 ))
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
RESP=$(curl -sS -m 7200 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ") || { echo "depth: request failed" >&2; exit 2; }
ANS=$(printf '%s' "$RESP" | jq -r '.choices[0].message.content // empty') || { echo "depth: reply is not JSON" >&2; exit 2; }
V=PASS; case "$ANS" in *"$CODE"*) ;; *) V=FAIL ;; esac
printf '%s' "$RESP" | jq -r --arg n "$N" --arg v "$V" '"\($n) paragraphs  prompt \(.timings.prompt_n) tok  prefill \(.timings.prompt_per_second) tok/s  decode \(.timings.predicted_per_second) tok/s (\(.timings.decode_intervals // "?") intervals in \(.timings.decode_ms // .timings.predicted_ms) ms)  \($v)"'
