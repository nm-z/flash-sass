#!/bin/bash
# needle.sh <filler paragraphs> <needle at percent> [url]: a needle-in-a-haystack check of the long-context path through flashd.
# The prompt is that many numbered filler paragraphs with one sentence buried at the given depth ("The access code of the
# vault is <six hex digits>."); the model is asked for the code. Prints the prompt size, the answer and the timings.
N=${1:-200}; PCT=${2:-50}; URL=${3:-http://127.0.0.1:18090}
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
REQ=$(jq -n --arg doc "$DOC" '{messages:[{role:"user",content:($doc + "\nWhat is the access code of the vault? Answer with the six hex digits only.")}],max_tokens:60,temperature:0,chat_template_kwargs:{enable_thinking:false}}')
echo "needle $CODE at ${PCT}% of $N paragraphs"
curl -s -m 3600 "$URL/v1/chat/completions" -H 'Content-Type: application/json' -d "$REQ" | jq -r '"answer: \(.choices[0].message.content)", "usage: \(.usage)", "timings: \(.timings)"'
