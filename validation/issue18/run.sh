#!/usr/bin/env bash
# One parameterized runner for issue18. Run on the GPU host after building.
set -euo pipefail
usage() {
    echo 'run.sh --list --suite padding|direct|controls|timing|smoke'
    echo 'run.sh --run --suite SUITE --repo PATH --out NEW_DIR --gpu UUID --lock FILE [--parent PATH]'
}
mode=; suite=padding; repo=; parent=; out=; gpu=; lock=
while (($#)); do
    case $1 in
        --list|--run) mode=$1; shift ;;
        --suite|--repo|--parent|--out|--gpu|--lock)
            key=${1#--}; (($# >= 2)) || exit 2
            printf -v "$key" '%s' "$2"; shift 2 ;;
        *) usage >&2; exit 2 ;;
    esac
done
[[ $mode == --list || $mode == --run ]] || { usage >&2; exit 2; }
case $suite in padding|direct|controls|timing|smoke) ;; *) usage >&2; exit 2 ;; esac

# Fields: image kind placement K V visible budget iterations expectation.
row() { printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$@"; }
plan() {
    local placement part poison k v n b steps image pair kind expectation
    case $suite in
        padding)
            for placement in device host; do
                for part in K V; do
                    for poison in nan inf -inf finite; do
                        k=off; v=off
                        if [[ $part == K ]]; then k=$poison; else v=$poison; fi
                        for n in {1..33} 65 {255..257} {511..513} {767..769} {1023..1025} {2048..2055} {8191..8193}; do
                            row candidate dense "$placement" "$k" "$v" "$n" 0 0 pass
                        done
                        for b in 1 2 8; do
                            for n in 1 $((4*b+1)) $((4*b+2)) $((4*b+3)) $((4*b+4)) $((4*b+5)); do
                                row candidate kv "$placement" "$k" "$v" "$n" "$b" 0 pass
                            done
                        done
                        for n in {1023..1025} {2048..2055} {8191..8193} 262144; do
                            row candidate padding "$placement" "$k" "$v" "$n" 512 0 pass
                        done
                        for n in {16383..16387} 262144; do
                            row candidate padding "$placement" "$k" "$v" "$n" 4096 0 pass
                        done
                    done
                done
            done ;;
        direct)
            for placement in device host; do
                for n in 3100 8000 65000 262000 262144; do
                    steps=4; if ((n == 262144)); then steps=1; fi
                    row candidate direct "$placement" off off "$n" 512 "$steps" pass
                done
            done ;;
        controls)
            for placement in device host; do
                for kind in dense padding; do
                    n=1; b=0
                    if [[ $kind == padding ]]; then n=2051; b=512; fi
                    for part in K V; do
                        for poison in nan inf -inf; do
                            k=off; v=off; expectation=pass
                            if [[ $part == K ]]; then k=$poison; else v=$poison; expectation=padding-failure; fi
                            row parent "$kind" "$placement" "$k" "$v" "$n" "$b" 0 "$expectation"
                        done
                    done
                done
                row candidate dense "$placement" off off 1 0 0 guard-failure
            done ;;
        timing)
            for pair in {1..5}; do
                for placement in device host; do
                    for kind in dense padding; do
                        b=0; [[ $kind == padding ]] && b=512
                        # Alternate order to reduce a systematic warmup/order bias.
                        if ((pair % 2)); then
                            row parent "$kind" "$placement" off off 2051 "$b" 200 pass
                            row candidate "$kind" "$placement" off off 2051 "$b" 200 pass
                        else
                            row candidate "$kind" "$placement" off off 2051 "$b" 200 pass
                            row parent "$kind" "$placement" off off 2051 "$b" 200 pass
                        fi
                    done
                done
            done ;;
        smoke)
            row candidate dense device off nan 1 0 0 pass
            row candidate padding device inf off 262144 512 0 pass
            row candidate kv device off inf 13 2 0 pass
            row candidate padding host off -inf 16387 4096 0 pass ;;
    esac
}
if [[ $mode == --list ]]; then plan; exit; fi
[[ -n $repo && -n $out && $gpu == GPU-* && -n $lock ]] || { usage >&2; exit 2; }
for x in jq timeout sha256sum flock nvidia-smi ps git awk rg date; do command -v "$x" >/dev/null || exit 2; done
repo=$(cd "$repo" && pwd -P)
if [[ $suite == controls || $suite == timing ]]; then
    [[ -n $parent ]] || { echo '--parent is required for controls/timing' >&2; exit 2; }
    parent=$(cd "$parent" && pwd -P)
fi
for f in kern/attn_test kern/qsa_test kern/qsa_direct_test kern/die.cubin kern/attnpart.cubin kern/attnmerge.cubin; do
    [[ -f $repo/$f ]] || { echo "missing $repo/$f" >&2; exit 2; }
done
mkdir "$out"; out=$(cd "$out" && pwd -P)
exec 9>"$lock"
until flock -n 9; do echo 'waiting for the GPU coordination lock'; sleep 5; done
# QSA_HOSTKV is presence based: unset, rather than assign zero, for VRAM.
while IFS= read -r key; do
    case $key in ATTN_*|QSA_*|DIE_CUBIN|TEST_GUARD_INJECT) unset "$key" ;; esac
done < <(compgen -e)
export CUDA_VISIBLE_DEVICES=$gpu
plan > "$out/plan.tsv"
expected=$(wc -l < "$out/plan.tsv")
: > "$out/cases.jsonl"
provenance() {
    local tree=$1 label=$2
    git -C "$tree" status --porcelain=v1 > "$out/$label-source-status.txt"
    (
        cd "$tree"
        git rev-parse HEAD
        sha256sum ref/attn.c ref/attn.h kern/attn_test.c kern/qsa_test.c \
            kern/qsa_padding_test.h kern/test_guards.h kern/qsa_direct_test.c \
            kern/qsa_direct_projection.h kern/qsa_direct_fixtures.h \
            kern/attnpart.sass kern/attnpartn.sass kern/attn_test kern/qsa_test \
            kern/qsa_direct_test kern/attnpart.cubin kern/attnmerge.cubin kern/die.cubin
    ) > "$out/$label-provenance.txt"
}
provenance "$repo" candidate
[[ -z $parent ]] || provenance "$parent" parent
hostname > "$out/host.txt"
date -u +%FT%TZ > "$out/started.txt"
summary() {
    jq -s --arg suite "$suite" --arg gpu "$gpu" --argjson expected "$expected" \
        '{suite:$suite,gpu:$gpu,expected:$expected,completed:length,passed:(length==$expected and all(.[];.passed)),cases:.}' \
        "$out/cases.jsonl" > "$out/results.json"
}
trap summary EXIT

# A selected-device compute owner, any non-resident compute owner, an active
# flashcore request, another test, or nonzero sampled GPU activity makes us wait.
# Resident hold/fence and sleeping flashcore are recorded; no process is killed.
preflight() {
    local attempt=0 prefix busy
    while :; do
        attempt=$((attempt+1)); prefix=$out/$tag-idle-$attempt
        nvidia-smi --query-gpu=index,uuid,utilization.gpu,memory.used --format=csv,noheader > "$prefix.gpus"
        nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader > "$prefix.apps"
        ps -eo pid=,comm=,wchan:32=,args= > "$prefix.ps"
        busy=0
        if awk -F, '$3+0 != 0 {f=1} END{exit !f}' "$prefix.gpus"; then busy=1; fi
        if awk -F, -v uuid="$gpu" '$1==uuid || $3 !~ /\/(hold|fence|flashcore)[[:space:]]*$/ {f=1} END{exit !f}' "$prefix.apps"; then busy=1; fi
        if awk '$2 ~ /^(attn_test|qsa_test|qsa_direct_test|qsa_gather_benc)/ || ($2=="flashcore" && $3 !~ /skb_wait_for_more/) {f=1} END{exit !f}' "$prefix.ps"; then busy=1; fi
        ((busy)) || return 0
        if ((attempt == 1 || attempt % 30 == 0)); then echo "$tag waiting for idle GPUs/processes"; fi
        sleep 1
    done
}
positive_output() {
    local file=$1
    rg -q '^guards: PASS \([1-9][0-9]* allocations, 256 bytes per boundary\)$' "$file" || return 1
    case $kind in
        dense)
            local field
            for field in pm pd po out; do
                rg -q "^$field: BIT-EXACT \(0 of [0-9]+ differ\)$" "$file" || return 1
            done ;;
        padding)
            rg -q "^capped padding: visible=$n position=$((n-1)) NBLK=$budget keys=[0-9]+ K=$k V=$v compact_cache=$placement pm/pd/po/out=BIT-EXACT valid-data-seed=4242$" "$file" || return 1 ;;
        kv)
            rg -Fqx "kv: $n tokens, NBLK $budget: BIT-EXACT (0 tokens differ; query/append checks $n, gather/attention checks $n)" "$file" || return 1 ;;
    esac
}
direct_output() {
    awk -v start="$n" -v steps="$iters" '
        /^visible=/ {
            if (NF!=8 || $1!=("visible=" (start+rows)) || $2!~/^complete=[0-9]+$/ ||
                $3!~/^tail=[0-9]+$/ || $4!~/^selected=[0-9]+$/ ||
                $5!~/^keys=[0-9]+$/ || $6!~/^guarded_allocations=[0-9]+$/ ||
                $7!~/^exclusion=(yes|no)$/ || $8!="EXACT") bad=1;
            split($6,a,"="); if(a[2]+0<=0) bad=1; rows++;
        }
        END {exit (bad || rows!=steps)}
    ' "$1"
}
i=0
while IFS=$'\t' read -r image kind placement k v n budget iters expectation; do
    i=$((i+1))
    printf -v tag '%04d-%s-%s-%s-K%s-V%s-n%s-b%s' "$i" "$image" "$kind" "$placement" "$k" "$v" "$n" "$budget"
    preflight
    unset ATTN_HOSTKV QSA_HOSTKV ATTN_POISON_K ATTN_POISON_V QSA_POISON_K QSA_POISON_V TEST_GUARD_INJECT
    tree=$repo; [[ $image == parent ]] && tree=$parent
    # Driver/source stay matched to their recorded tree in both configurations.
    export DIE_CUBIN=$tree/kern/die.cubin
    if [[ $placement == host ]]; then export ATTN_HOSTKV=1 QSA_HOSTKV=1; fi
    export ATTN_POISON_K=$k ATTN_POISON_V=$v QSA_POISON_K=$k QSA_POISON_V=$v
    [[ $expectation == guard-failure ]] && export TEST_GUARD_INJECT=1
    case $kind in
        dense) cmd=("$tree/kern/attn_test" "$n" "$iters") ;;
        kv) cmd=("$tree/kern/qsa_test" kv "$n" "$budget") ;;
        padding) cmd=("$tree/kern/qsa_test" padding "$n" "$budget" "$iters") ;;
        direct)
            export QSA_PATTERN=hash QSA_DIRECT_SEED=4242
            cmd=("$tree/kern/qsa_direct_test" "$n" "$budget" "$iters") ;;
    esac
    printf '%q ' "${cmd[@]}" > "$out/$tag.command"; printf '\n' >> "$out/$tag.command"
    env | LC_ALL=C sort | rg '^(CUDA_VISIBLE_DEVICES|DIE_CUBIN|ATTN_|QSA_|TEST_GUARD_INJECT)' > "$out/$tag.env"
    start=$(date +%s%N)
    rc=0
    (cd "$tree/kern"; timeout 180 "${cmd[@]}") > "$out/$tag.stdout" 2> "$out/$tag.stderr" || rc=$?
    stop=$(date +%s%N); ok=true
    case $expectation in
        pass)
            if ((rc != 0)) || rg -qi 'MISMATCH|FAIL|NONFINITE' "$out/$tag.stdout" "$out/$tag.stderr"; then
                ok=false
            elif [[ $kind == direct ]]; then
                direct_output "$out/$tag.stdout" || ok=false
            else
                positive_output "$out/$tag.stdout" || ok=false
            fi ;;
        padding-failure)
            wanted=1; [[ $kind == dense ]] && wanted=3
            if ((rc != wanted)) || ! rg -q 'MISMATCH' "$out/$tag.stdout" ||
                ! rg -q '^guards: PASS ' "$out/$tag.stdout"; then ok=false; fi ;;
        guard-failure)
            if ((rc != 3)) || ! rg -q '^guards: FAIL ' "$out/$tag.stdout" ||
                ! rg -q '^guard mismatch: .*side=prefix' "$out/$tag.stderr"; then ok=false; fi ;;
    esac
    microseconds=null
    if ((iters > 0)) && [[ $kind != direct ]]; then
        microseconds=$(awk '/^timing:/{print $2} /^capped component timing:/{print $4}' "$out/$tag.stdout")
        if [[ ! $microseconds =~ ^[0-9]+([.][0-9]+)?$ ]]; then ok=false; microseconds=null; fi
    fi
    jq -n --arg name "$tag" --arg image "$image" --arg kind "$kind" --arg placement "$placement" \
        --arg k "$k" --arg v "$v" --arg expectation "$expectation" \
        --argjson n "$n" --argjson budget "$budget" --argjson iterations "$iters" \
        --argjson exit_code "$rc" --argjson elapsed_ns "$((stop-start))" \
        --argjson microseconds "$microseconds" --argjson passed "$ok" \
        '{name:$name,image:$image,kind:$kind,placement:$placement,K:$k,V:$v,n:$n,budget:$budget,iterations:$iterations,expectation:$expectation,exit_code:$exit_code,elapsed_ns:$elapsed_ns,microseconds:$microseconds,passed:$passed}' >> "$out/cases.jsonl"
    echo "$tag rc=$rc passed=$ok"
    [[ $ok == true ]] || exit 1
done < "$out/plan.tsv"
date -u +%FT%TZ > "$out/finished.txt"
