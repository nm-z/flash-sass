#!/usr/bin/env bash
# Run the production QSA indexer SASS on frozen original-layer projection inputs.
set -euo pipefail

issue22_gpu=${ISSUE22_GPU:?ISSUE22_GPU is required}
issue22_test=${ISSUE22_QSA_TEST:?ISSUE22_QSA_TEST is required}
issue22_test_revision=${ISSUE22_TEST_SOURCE_REVISION:?ISSUE22_TEST_SOURCE_REVISION is required}
issue22_weight_revision=${ISSUE22_WEIGHT_REVISION:?ISSUE22_WEIGHT_REVISION is required}
issue22_inputs=${QSA_INDEXER_INPUT_DIR:?QSA_INDEXER_INPUT_DIR is required}
issue22_positions=${QSA_INDEXER_POSITIONS:?QSA_INDEXER_POSITIONS is required}
issue22_budget=${ISSUE22_INDEXER_BUDGET:-1}
issue22_weights_sha=$(jq -er '.weights_file_sha256' "$issue22_inputs/complete.json")
issue22_out=${GPUQ_RUN_DIR:?GPUQ_RUN_DIR is required}/issue22-layer3-indexer
issue22_lock=/tmp/flash-sass-engine-validation.lock

[[ ${GPUQ_LOCK_FD:-} == 9 && -n ${GPUQ_JOB_ID:-} ]] || {
    printf 'indexer GPU test requires GPUQ inherited lock descriptor 9\n' >&2
    exit 2
}
[[ $(readlink -- "/proc/$$/fd/9" 2>/dev/null || true) == "$issue22_lock" ]] || {
    printf 'GPUQ descriptor 9 does not reference %s\n' "$issue22_lock" >&2
    exit 2
}
[[ $issue22_gpu =~ ^[0-9]+$ && $issue22_positions =~ ^[1-9][0-9]*$ && $issue22_budget =~ ^[1-9][0-9]*$ ]] || exit 2
(( issue22_budget <= 512 )) || exit 2
[[ $issue22_test_revision =~ ^[0-9a-f]{40}$ && $issue22_weight_revision =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ -x $issue22_test && -f ${DIE_CUBIN:?DIE_CUBIN is required} ]] || {
    printf 'qsa_test binary or die.cubin is missing\n' >&2
    exit 2
}
jq -e --arg revision "$issue22_weight_revision" --argjson positions "$issue22_positions" \
    --argjson budget "$issue22_budget" \
    '.schema == "flash-sass.qsa-layer-capture.v1" and .complete == true
     and .weights_origin == "original_checkpoint_safetensors_layer3"
     and .weights_revision == $revision and .positions == $positions and .budget == $budget
     and (.weights_file_sha256|type) == "string" and (.weights_file_sha256|length) == 64
     and .production_gpu_indexer_executed == false and .full_model == false' \
    "$issue22_inputs/complete.json" > /dev/null

issue22_nv=$((issue22_positions / 4))
[[ $issue22_nv -gt $issue22_budget ]] || { printf 'indexer test needs sparse candidate blocks\n' >&2; exit 2; }
issue22_selected=$((issue22_nv < issue22_budget ? issue22_nv : issue22_budget))
issue22_selected_bytes=$((issue22_selected * 4))
issue22_selected_ids=$(od -An -tu4 -N "$issue22_selected_bytes" \
    "$issue22_inputs/n$issue22_positions/selected_ids.u32" | tr -s ' ' '\n' | sed '/^$/d' | jq -R . | jq -s 'map(tonumber)')
mkdir -p -- "$(dirname -- "$issue22_out")"
mkdir -- "$issue22_out"
issue22_out=$(cd -- "$issue22_out" && pwd)
export LC_ALL=C CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$issue22_gpu
export QSA_INDEXER_INPUT_DIR=$issue22_inputs QSA_INDEXER_POSITIONS=$issue22_positions
export DIE_CUBIN
nvidia-smi > "$issue22_out/gpu-before.txt"
issue22_free=$(nvidia-smi -i "$issue22_gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
[[ $issue22_free =~ ^[0-9]+$ ]] || exit 2
if (( issue22_free < 1000 )); then
    printf 'GPU %s has %s MiB free; indexer component test requires 1000 MiB\n' \
        "$issue22_gpu" "$issue22_free" | tee "$issue22_out/blocked.txt"
    exit 75
fi
{
    printf 'weights_revision=%s\ntest_source_revision=%s\nGPU=%s\npositions=%s\nbudget=%s\n' \
        "$issue22_weight_revision" "$issue22_test_revision" "$issue22_gpu" \
        "$issue22_positions" "$issue22_budget"
    printf 'selected_ids=%s\n' "$issue22_selected_ids"
    sha256sum "$issue22_test" "$DIE_CUBIN" "$issue22_inputs/complete.json" \
        "${BASH_SOURCE[0]}"
    printf '\n'
} > "$issue22_out/inputs-and-source.txt"
sha256sum "$issue22_test" "$DIE_CUBIN" "${BASH_SOURCE[0]}" > "$issue22_out/source-before.sha256"
(
    cd -- "$issue22_inputs"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$issue22_out/inputs-before.sha256"
"$issue22_test" idxqk "$issue22_positions" > "$issue22_out/idxqk.stdout" 2> "$issue22_out/idxqk.stderr"
"$issue22_test" idxscore "$issue22_nv" "$issue22_budget" > "$issue22_out/idxscore.stdout" 2> "$issue22_out/idxscore.stderr"
"$issue22_test" idxsel "$issue22_nv" "$issue22_budget" 0 5 > "$issue22_out/idxsel.stdout" 2> "$issue22_out/idxsel.stderr"
rg -q "idxqk: $issue22_positions tokens: BIT-EXACT" "$issue22_out/idxqk.stdout"
rg -q "idxscore: nv $issue22_nv NBLK $issue22_budget: BIT-EXACT" "$issue22_out/idxscore.stdout"
rg -q "idxsel: nv $issue22_nv NBLK $issue22_budget mode 0: BIT-EXACT" "$issue22_out/idxsel.stdout"
nvidia-smi > "$issue22_out/gpu-after.txt"
sha256sum "$issue22_test" "$DIE_CUBIN" "${BASH_SOURCE[0]}" > "$issue22_out/source-after.sha256"
cmp -- "$issue22_out/source-before.sha256" "$issue22_out/source-after.sha256"
(
    cd -- "$issue22_inputs"
    find . -type f -print0 | sort -z | xargs -0 sha256sum
) > "$issue22_out/inputs-after.sha256"
cmp -- "$issue22_out/inputs-before.sha256" "$issue22_out/inputs-after.sha256"
issue22_selected_id=$(od -An -tu4 -N4 "$issue22_inputs/n$issue22_positions/selected_ids.u32" | tr -d ' ')
jq -n --arg weight_revision "$issue22_weight_revision" \
    --arg test_revision "$issue22_test_revision" \
    --arg weights_sha "$issue22_weights_sha" \
    --argjson gpu "$issue22_gpu" --argjson positions "$issue22_positions" \
    --argjson blocks "$issue22_nv" --argjson budget "$issue22_budget" \
    --argjson selected_ids "$issue22_selected_ids" \
    '{schema:"flash-sass.issue22-original-layer3-indexer-gpu.v1",
      checkpoint_revision:$weight_revision,test_source_revision:$test_revision,
      weights_sha256:$weights_sha,
      gpu:$gpu,positions:$positions,complete_blocks:$blocks,budget:$budget,
      selected_ids:$selected_ids,selected_blocks:($selected_ids|length),idxqk:"bit_exact",idxscore:"bit_exact",idxsel:"bit_exact",
      original_layer3_indexer_weights:true,synthetic_hidden_inputs:true,
      source_hashes_unchanged:true,input_hashes_unchanged:true,
      production_gpu_indexer_ops_executed:true,production_model_dispatch_executed:false,
      teacher_forced:false,full_model:false,held_out_quality:false,issue22_resolved:false}' \
    > "$issue22_out/complete.json"
printf 'QSA layer-3 indexer SASS: idxqk, idxscore, idxsel all BIT-EXACT.\n'
