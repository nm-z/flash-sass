#!/usr/bin/env bash
# Recorded issue-22 model-capture command; synthetic token microfixture.
set -euo pipefail
[[ $# == 4 ]] || { printf 'usage: %s T BUDGET LENGTHS NEW_OUTPUT\n' "$0" >&2; exit 2; }
issue22_total=$1
issue22_budget=$2
issue22_lengths=$3
issue22_out=$4
[[ $issue22_total =~ ^[1-9][0-9]*$ && $issue22_budget =~ ^[1-9][0-9]*$ ]] || exit 2
[[ ! -e $issue22_out ]] || exit 2
issue22_source=/home/nate/codex/flash-pr46-attention-20261018.LnH4Z4/source
mkdir -- "$issue22_out"
issue22_out=$(cd "$issue22_out" && pwd)
cd "$issue22_source/kern"
perl -e 'my $n=shift; for(my $i=0;$i<$n;$i++){print 10+$i%200000,"\n"}' "$issue22_total" > "$issue22_out/ids.txt"
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export SPLIT_SHARES=112,97,81,116,97,97 ATTN_DIE=1 ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5
export MODEL_ATTN=qsa MODEL_NOTWIN=1 FLASH_NBLK=$issue22_budget FLASH_KV_HOST=1
export MODEL_PLE=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
export MODEL_PLE_OFF=192 MODEL_IDS=$issue22_out/ids.txt
export MODEL_CAPTURE=$issue22_out/logits MODEL_LAYER_CAPTURE=$issue22_out/layer
export MODEL_LAYER_INDEX=3 MODEL_LAYER_LENGTHS=$issue22_lengths
{
    date -u +%FT%TZ
    hostname
    printf 'source_revision=2284a214d3babea6df19510654203c65abfd6cce\n'
    env | LC_ALL=C sort | rg '^(MODEL_|FLASH_|CUDA_|SPLIT_SHARES=|ATTN_)'
    printf 'command: timeout --signal=TERM --kill-after=15s 1200s '
    printf '%q ' "$PWD/model_test" /unused-with-MODEL_NOTWIN 0 48 "$issue22_total" eh /tmp/hold-h{0,1,2,3,4,5}.manifest
    printf '\n'
} > "$issue22_out/command.txt"
sha256sum model_test model_test.c model_capture.h model_layer_capture.h model_qsa_test.h qsa_state.h prog.c prog.h die.cubin dieops.h > "$issue22_out/source-before.sha256"
exec 9>/tmp/flash-sass-engine-validation.lock
flock -w 1800 -E 75 9 &
issue22_waiter=$!
trap 'if [[ -n $issue22_waiter ]]; then kill -TERM "$issue22_waiter" 2>/dev/null || true; fi; exec 9>&-' EXIT
issue22_waited=0
while kill -0 "$issue22_waiter" 2>/dev/null; do
    if (( issue22_waited % 180 == 0 )); then
        date -u '+%FT%TZ queued for shared validation lock' | tee -a "$issue22_out/lock.log"
        lslocks --noheadings --output PID,COMMAND,TYPE,MODE,PATH |
            rg 'flash-sass-engine-validation.lock' >> "$issue22_out/lock.log" || true
    fi
    sleep 5
    (( issue22_waited += 5 ))
done
set +e
wait "$issue22_waiter"
issue22_rc=$?
set -e
issue22_waiter=
if (( issue22_rc != 0 )); then
    printf 'shared validation lock acquisition exit %s after up to 1800 seconds\n' "$issue22_rc" > "$issue22_out/blocked.txt"
    exit "$issue22_rc"
fi
date -u '+%FT%TZ acquired shared validation lock' | tee -a "$issue22_out/lock.log"
nvidia-smi > "$issue22_out/gpu-before.txt"
ps -eo pid,etime,args | rg '[f]lashcore|[/]hold |[f]ence|[m]odel_test|flash-sass-engine-validation' > "$issue22_out/processes-before.txt" || true
mkdir "$issue22_out/holders"
for d in 0 1 2 3 4 5; do
    cp "/tmp/hold-h${d}.manifest" "$issue22_out/holders/"
done
sha256sum /tmp/hold-h{0,1,2,3,4,5}.manifest > "$issue22_out/holder-before.sha256"
set +e
timeout --signal=TERM --kill-after=15s 1200s \
    ./model_test /unused-with-MODEL_NOTWIN 0 48 "$issue22_total" eh /tmp/hold-h{0,1,2,3,4,5}.manifest \
    > "$issue22_out/model.stdout" 2> "$issue22_out/model.stderr"
issue22_rc=$?
set -e
printf '%s\n' "$issue22_rc" > "$issue22_out/model.exit"
nvidia-smi > "$issue22_out/gpu-after.txt"
sha256sum /tmp/hold-h{0,1,2,3,4,5}.manifest > "$issue22_out/holder-after.sha256"
sha256sum model_test model_test.c model_capture.h model_layer_capture.h model_qsa_test.h qsa_state.h prog.c prog.h die.cubin dieops.h > "$issue22_out/source-after.sha256"
cmp "$issue22_out/source-before.sha256" "$issue22_out/source-after.sha256"
cmp "$issue22_out/holder-before.sha256" "$issue22_out/holder-after.sha256"
(( issue22_rc == 0 )) || { printf 'model harness exited %s; inspect model.stderr and gpu-before.txt\n' "$issue22_rc" > "$issue22_out/blocked.txt"; exit "$issue22_rc"; }
jq -e '.schema=="flash-sass.raw-model-capture.v2" and .complete==true' "$issue22_out/logits/capture_info.json" >/dev/null
test -f "$issue22_out/layer/complete.json"
(
    cd "$issue22_out"
    rg --files -0 logits layer | sort -z | xargs -0 sha256sum > capture.sha256
)
jq -n --argjson total "$issue22_total" --argjson budget "$issue22_budget" \
    '{schema:"flash-sass.issue22-model-run.v1",source_revision:"2284a214d3babea6df19510654203c65abfd6cce",
      positions:$total,budget:$budget,gpu_execution:true,full_model_capture:true,teacher_forced:true,
      input_kind:"consecutive_token_microfixture",held_out_quality:false,original_checkpoint_quality:null,
      issue22_resolved:false,lock:"/tmp/flash-sass-engine-validation.lock",source_hashes_unchanged:true,
      holder_manifests_unchanged:true}' > "$issue22_out/complete.json"
printf 'Full-model capture complete; independent replay and held-out quality remain separate.\n'
