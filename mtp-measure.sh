#!/bin/bash
pkill -9 -x llama-server
until [ -z "$(pgrep -x llama-server)" ]; do sleep 1; done
DEVS=0,1,2,3,4,5,6,7 TS=0.7,1,1,1,1,1,0.3,0.3 GRAPHS=1 CTX=32768 UB=8 MTP=1 DRAFTMAX=2 \
  setsid -f ~/run-mtp-main.sh -v > ~/mtpmain.out 2> ~/mtpmain.log < /dev/null
t0=$(date +%s)
while :; do
  if curl -s -m 3 http://127.0.0.1:8092/health | rg -q '"status":"ok"'; then echo "UP in $(( $(date +%s)-t0 ))s"; break; fi
  if ! pgrep -x llama-server | rg -q .; then echo "DIED"; rg -N "error|ABORT|failed" ~/mtpmain.log | tail -3; exit 1; fi
  sleep 10
done
echo "n_rs_seq: $(rg -N 'n_rs_seq' ~/mtpmain.log | head -1)"
for p in "Write three sentences about rivers." "Count from 1 to 30."; do
  curl -s -m 130 http://127.0.0.1:8092/v1/chat/completions -H "Content-Type: application/json" \
    -d "{\"messages\":[{\"role\":\"user\",\"content\":\"$p\"}],\"max_tokens\":64,\"stream\":false}" \
    | jq -r '.timings | "tok/s=\(.predicted_per_second)"'
done
rg -N "draft acceptance" ~/mtpmain.log | tail -1
pgrep -x llama-server | rg -q . && echo "ALIVE" || echo "CRASHED"
rg -N "failed to remove sequence" ~/mtpmain.log | tail -1
