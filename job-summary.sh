#!/usr/bin/env bash
# Wait for the named gpuq jobs (latest job per label) and print one summary line per job as each ends:
# status, same_output counts, decode/prefill rate stats, first divergence points.
# Usage: job-summary.sh LABEL...
export PATH=/home/nate/codex/flash-goal-140-1200/gpuq-fixed/bin:/usr/local/bin:/usr/bin:/bin
pending=("$@")
while ((${#pending[@]})); do
  next=()
  for n in "${pending[@]}"; do
    j=$(gpuq query "SELECT id FROM jobs WHERE label='$n' ORDER BY submitted_ns DESC LIMIT 1" | jq -r '.[0].id // empty')
    [ -z "$j" ] && { next+=("$n"); continue; }
    s=$(gpuq query "SELECT status FROM jobs WHERE id='$j'" | jq -r '.[0].status')
    case "$s" in
      queued|running) next+=("$n"); continue ;;
    esac
    same=$(gpuq query "SELECT value_json FROM samples WHERE job_id='$j' AND metric LIKE 'goal.%.same_output'" | jq -r '[.[].value_json] | group_by(.) | map("\(.[0])=\(length)") | join(" ")')
    rate=$(gpuq query "SELECT metric, value_number FROM samples WHERE job_id='$j' AND metric LIKE 'goal.%.tokens_s'" | jq -c 'group_by(.metric) | map({m: (.[0].metric | split(".")[1]), n: length, avg: ((map(.value_number) | add / length) * 100 | round / 100), lo: (map(.value_number) | min), hi: (map(.value_number) | max)})')
    diffs=$(gpuq query "SELECT d.data FROM documents d WHERE d.job_id='$j' AND d.source LIKE '%output-difference'" | jq -r '.[].data' 2>/dev/null | jq -rc '[.first_index, .expected, .actual] | tostring' 2>/dev/null | sort | uniq -c | tr '\n' ' ')
    echo "$n ${j:0:8} $s same:[$same] rate:$rate diffs:[$diffs]"
  done
  pending=("${next[@]}")
  ((${#pending[@]})) && sleep 30
done
