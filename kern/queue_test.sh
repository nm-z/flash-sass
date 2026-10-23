#!/bin/bash
# queue_test.sh <shard-1.gguf>: flashd's request queue against a stub core (no GPU, no model run). Builds flashd, serves it on a scratch port, and checks:
#   arrival order, a client that leaves while queued is never started, a full queue is a 503 with Retry-After, a client that leaves during a run
#   stops it, and GET /queue. Exit 0 only if every check passes.
cd "$(dirname "$0")" || exit 1
GGUF=${1:?usage: queue_test.sh <shard-1.gguf>}
PORT=${QUEUE_TEST_PORT:-18191}; D=$(mktemp -d /tmp/flashd-queue.XXXXXX); SOCK=$D/core.sock; export STUB_LOG=$D/stub.log
fail=0; ok() { echo "ok:   $*"; }; bad() { echo "FAIL: $*"; fail=1; }
make -s flashd >/dev/null || { echo "flashd does not build"; exit 1; }
cleanup() { kill $SOCAT $FLASHD 2>/dev/null; wait 2>/dev/null; rm -rf "$D"; }; trap cleanup EXIT
socat UNIX-LISTEN:"$SOCK",fork EXEC:"$PWD/queue_stub.sh" & SOCAT=$!
for i in $(seq 50); do [ -S "$SOCK" ] && break; timeout 0.1 tail -f /dev/null; done
FLASHD_QUEUE_MAX=4 ./flashd "$GGUF" "$SOCK" 127.0.0.1:$PORT 2> "$D/flashd.err" & FLASHD=$!
for i in $(seq 100); do curl -s -m 1 127.0.0.1:$PORT/health >/dev/null && break; timeout 0.1 tail -f /dev/null; done
req() {   # req <id> <hold seconds> [curl -m]: a non-stream chat; prints the HTTP status
	curl -s -m "${3:-60}" -o /dev/null -w '%{http_code}' 127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' \
		-d "{\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":$((100 * $2 + $1 + 1)),\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
}
idle() { for i in $(seq 200); do [ "$(curl -s -m 2 127.0.0.1:$PORT/queue)" = '{"running":0,"waiting":0,"max":4}' ] && return 0; timeout 0.1 tail -f /dev/null; done; return 1; }
started() { grep '^chat' "$STUB_LOG" | awk '{printf "%s ", $2}'; }

: > "$STUB_LOG"; P=()
for i in 1 2 3 4 5; do req $i 1 > /dev/null & P+=($!); timeout 0.15 tail -f /dev/null; done; wait "${P[@]}"; idle
[ "$(started)" = "1 2 3 4 5 " ] && ok "five requests ran in arrival order" || bad "order was: $(started)"

: > "$STUB_LOG"; P=()
req 10 3 > /dev/null & P+=($!); timeout 0.3 tail -f /dev/null
req 11 1 1 > /dev/null & P+=($!); timeout 0.3 tail -f /dev/null
req 12 1 > /dev/null & P+=($!); wait "${P[@]}"; idle
[ "$(started)" = "10 12 " ] && ok "a client that left while queued was never started" || bad "started: $(started) (11 must not appear)"
grep -q "client left the queue" "$D/flashd.err" && ok "the queue logged the departure" || bad "no departure line in the log"

: > "$STUB_LOG"; P=()
for i in 1 2 3 4 5; do req $((30 + i)) $([ $i = 1 ] && echo 4 || echo 0) > /dev/null & P+=($!); timeout 0.15 tail -f /dev/null; done
body=$(curl -s -i -m 5 127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":5}')
echo "$body" | head -1 | grep -q 503 && echo "$body" | grep -qi '^Retry-After:' && ok "the sixth request (queue max 4) got 503 with Retry-After" || bad "overflow answer: $(echo "$body" | head -1)"
wait "${P[@]}"; idle
: > "$STUB_LOG"
t0=$(date +%s.%N); req 20 8 1 > /dev/null
for i in $(seq 100); do grep -q '^abort 20' "$STUB_LOG" && break; timeout 0.1 tail -f /dev/null; done
t1=$(date +%s.%N); dt=$(echo "$t1 - $t0" | bc -l)
grep -q '^abort 20' "$STUB_LOG" && [ "$(echo "$dt < 6" | bc -l)" = 1 ] && ok "a client that left during a run stopped it after $(printf %.1f "$dt") s (the stub held it 8 s)" || bad "no abort within 6 s ($(cat "$STUB_LOG" | tr '\n' ' '))"
idle
curl -s -m 2 127.0.0.1:$PORT/queue | grep -q '"running":0' && ok "GET /queue answers" || bad "GET /queue"
[ $fail = 0 ] && echo "PASS: flashd request queue" || echo "FAILED"
exit $fail
