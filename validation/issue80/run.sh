#!/bin/bash
# Acceptance for issue #80 against fake cores: FLASHD_REQUEST_MAX_S=5 cancels streaming, non-streaming and queued requests; a quick request is unchanged.
# run.sh FLASHD GGUF1 OUTDIR
set -u
here=$(dirname "$(readlink -f "$0")") FLASHD=$(readlink -f "$1") GGUF=$2 OUT=$3 fail=0
mkdir -p "$OUT"; cd "$OUT" || exit 1
check() { if eval "$2"; then echo "PASS $1"; else echo "FAIL $1"; fail=1; fi; }
req() { curl -sS -m 60 -o "$1.body" -w '%{http_code} %{time_total}\n' http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":8,\"stream\":$2}" > "$1.status"; }
start() {   # start MODE LIMIT: fake core + flashd on a free port
	SOCK=$PWD/core-$1.sock PORT=$((18600 + RANDOM % 300)); : > core-$1.log
	perl "$here/fake_core.pl" "$SOCK" "$1" core-$1.log & CORE=$!
	for i in $(seq 600); do [ -S "$SOCK" ] && break; sleep 0.1; done; [ -S "$SOCK" ] || { echo "FAIL fake core did not start"; exit 1; }
	FLASHD_REQUEST_MAX_S=$2 "$FLASHD" "$GGUF" "$SOCK" 127.0.0.1:$PORT 2> flashd-$1.err & FD=$!
	for i in $(seq 600); do curl -s -m 1 http://127.0.0.1:$PORT/health > /dev/null && break; kill -0 $FD 2> /dev/null || { echo "FAIL flashd exited: $(cat flashd-$1.err)"; exit 1; }; sleep 0.5; done
}
stop() { kill $FD $CORE 2> /dev/null; wait $FD $CORE 2> /dev/null; }

start silent 5
req stream true & r1=$!; sleep 1; req plain false & r2=$!; req queued false & r3=$!; wait $r1 $r2 $r3
check "stream ends with request_timeout" 'grep -q "\"type\":\"request_timeout\"" stream.body && grep -q "data: \[DONE\]" stream.body'
check "stream cancelled after about 5 s" 'awk "{exit !(\$2 >= 4.5 && \$2 < 9)}" stream.status'
check "non-streaming request gets 504" 'grep -q "^504 " plain.status'
check "queued request times out on its own deadline" 'awk "{exit !(\$1 == 504 && \$2 >= 4.5 && \$2 < 6)}" queued.status'
check "core saw the running request hang up" 'grep -q "closed chat=1 after [4-8]\." core-silent.log'
check "a request whose deadline passed in the queue never reaches the core" '[ $(grep -c "closed chat=1" core-silent.log) -eq 1 ]'
check "flashd logged the cancellations" '[ $(grep -c "FLASHD_REQUEST_MAX_S" flashd-silent.err) -ge 3 ]'
stop

start answer 300
req quick false
check "a quick request still completes with 200" 'grep -q "^200 " quick.status && grep -q "\"finish_reason\":\"stop\"" quick.body'
stop
for f in *.status; do echo "$f: $(cat $f)"; done; cat core-*.log flashd-*.err | grep -v "^$"
exit $fail
