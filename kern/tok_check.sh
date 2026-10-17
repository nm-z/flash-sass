#!/bin/bash
# tok_check.sh <gguf> <llama-tokenize>: tokenizes a set of awkward strings with tok_test and
# with llama.cpp's llama-tokenize (the reference) and reports every difference.
gguf=$1; ref=$2; dir=$(mktemp -d)
n=0
add() { n=$((n + 1)); printf '%b' "$1" > "$dir/case$n"; }
add 'Hello world'
add 'The capital of France is'
add "I'm can't we'll they've you'd She's IT'S O'Neil 'quoted' rock'n'roll"
add '12345 3.14159 1,000,000 2026-10-16 x2 v1.2.3 0x1f'
add '  leading\n\n\ttabs   and   spaces  \r\n trailing   '
add 'int main(void) { return x->y[0] + 3; } // comment\n#include <stdio.h>\n\tif (a && b || !c) {}'
add 'Café naïve résumé — “quotes” 日本語のテキスト 中文 한국어 Ελληνικά русский'
add '\U1F44D\U1F3FD \U1F1FA\U1F1F8 \U1F600!!'
add '<|im_start|>user\nHi there<|im_end|>\n<|im_start|>assistant\n<think>\n'
add '\xff\xfe abc \xc3 d\xe2\x82'
add 'é ä किन्दी'
add 'Hello!!!\n\nWorld???\r\n...and   \n  more?!'
add '$100 + 50% = ~$150 @user #tag ^_^ `code` |pipe| <b>bold</b>'
add 'Long line: The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs! How vexingly quick daft zebras jump?'
add '<think>\nreasoning here\n</think>\n\nThe answer is 42.<|im_end|>'
add ''
add '\n'
add '   '
fail=0
for ((i = 1; i <= n; i++)); do
	for mode in special raw; do
		if [ "$mode" = raw ]; then flag=--no-parse-special; arg=raw; else flag=; arg=; fi
		want=$("$ref" -m "$gguf" -f "$dir/case$i" --ids --log-disable --no-escape $flag); wrc=$?
		got=$(./tok_test "$gguf" "$dir/case$i" $arg); grc=$?
		if [ $wrc -ne 0 ] || [ $grc -ne 0 ]; then fail=$((fail + 1)); echo "case $i ($mode): a command failed (llama-tokenize rc $wrc, tok_test rc $grc); two failures with empty output must not count as agreement"; continue; fi
		if [ "$want" != "$got" ]; then fail=$((fail + 1)); echo "case $i ($mode) differs: $(head -c 60 "$dir/case$i" | od -c | head -2 | tr -s ' ')"; echo "  llama.cpp $want"; echo "  tok_test  $got"; fi
	done
done
echo "tok_check: $n cases x 2 modes, $fail differences or failures"
rm -r "$dir"
[ "$fail" -eq 0 ]   # the exit status is the verdict (it was the status of the cleanup)
