#!/usr/bin/env bash
# Reproducible orchestration for the validation-only C engine fixture.
set -Eeuo pipefail
export LC_ALL=C
repo=; control=; restorelib=; socket=; engine_lock=; component_lock=; out=
while (($#)); do
    (($# >= 2)) || exit 2
    case $1 in
        --repo) repo=$2 ;; --control-cubin) control=$2 ;;
        --restore-lib) restorelib=$2 ;; --socket) socket=$2 ;;
        --engine-lock) engine_lock=$2 ;; --component-lock) component_lock=$2 ;;
        --out) out=$2 ;; *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift 2
done
[[ -n $repo && -n $control && -n $restorelib && -n $socket &&
   -n $engine_lock && -n $component_lock && -n $out ]] || exit 2
for command in flock timeout socat nvidia-smi ps awk sort sha256sum realpath grep cmp git jq; do
    command -v "$command" >/dev/null || { echo "missing $command" >&2; exit 2; }
done
repo=$(realpath "$repo"); control=$(realpath "$control"); restorelib=$(realpath "$restorelib")
candidate=$repo/kern/die.cubin; testlib=$repo/kern/libengine_reuse_test.so
[[ -f $candidate && -f $testlib && -f $control && -f $restorelib && -S $socket ]] || exit 2
for path in "$restorelib" "$testlib" "$candidate" "$control"; do
    [[ $path != *[[:space:]]* ]] || { echo 'protocol paths cannot contain whitespace' >&2; exit 2; }
done
[[ ! -e $out ]] || { echo 'output directory must be new' >&2; exit 2; }
mkdir -p "$out"; out=$(realpath "$out")
exec 8>"$component_lock"
until flock -n 8; do echo 'waiting for component validation lock'; sleep 5; done
exec 9>"$engine_lock"
until flock -n 9; do echo 'waiting for engine validation lock'; sleep 5; done
printf '%s\n' "repo=$repo" "control=$control" "restorelib=$restorelib" "socket=$socket" > "$out/inputs.txt"
sha256sum "$restorelib" > "$out/restore-library.before.sha256"
sha256sum "$testlib" "$candidate" "$control" > "$out/test-inputs.sha256"
git -C "$repo" rev-parse HEAD > "$out/source-head.txt"
git -C "$repo" status --porcelain=v1 > "$out/source-status.txt"
request() {
    local command=$1 log=$2
    printf '%s\n' "$command" > "$log.command"
    printf '%s\n' "$command" | timeout 1800 socat -t1800 - "UNIX-CONNECT:$socket" > "$log" 2> "$log.stderr"
}
config_line() {
    local log=$1
    [[ $(awk '/^config /{n++}END{print n+0}' "$log") == 1 ]] || return 1
    awk '/^config /{print}' "$log"
}
get_key() {
    local line=$1 key=$2
    awk -v k="$key" '{for(i=1;i<=NF;i++)if(index($i,k"=")==1){print substr($i,length(k)+2);n++}}END{if(n!=1)exit 1}' <<< "$line"
}
preflight() {
    local label=$1 base="$out/preflight-$1" pid name bad=0
    nvidia-smi --query-gpu=index,uuid,utilization.gpu --format=csv,noheader,nounits > "$base.gpu" 2> "$base.gpu.stderr" || bad=1
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits > "$base.owners" 2> "$base.owners.stderr" || bad=1
    ps -eo pid=,ppid=,comm=,wchan:64= > "$base.all-ps" || bad=1
    awk -F, '{gsub(/^[ \t]+|[ \t]+$/,"",$3);if($3!="0")bad=1;n++}END{exit bad||n==0}' "$base.gpu" || bad=1
    awk -F, '{gsub(/[ \t]/,"",$1);if($1!~/^[0-9]+$/)exit 1;print $1}' "$base.owners" | sort -nu > "$base.pids" || bad=1
    : > "$base.owner-ps"
    while IFS= read -r pid; do
        [[ -n $pid ]] || continue
        name=$(awk -v p="$pid" '$1==p{print $3;n++}END{if(n!=1)exit 1}' "$base.all-ps") || { bad=1; continue; }
        case $name in hold|fence|flashcore) ;; *) echo "busy GPU owner $pid $name" >&2; bad=1 ;; esac
        awk -v p="$pid" '$1==p' "$base.all-ps" >> "$base.owner-ps"
    done < "$base.pids"
    awk '$3 ~ /test/ || $3 ~ /qsa_gather/ || $3 ~ /^compute-san/ || $3 ~ /^cuda-memcheck/ {print;bad=1}END{exit bad}' "$base.all-ps" > "$base.competing-tests" || bad=1
    local -a cores
    mapfile -t cores < <(awk '$3=="flashcore"{print $1}' "$base.all-ps")
    if ((${#cores[@]} != 1)); then
        echo 'requires exactly one flashcore' >&2; bad=1
    else
        awk -v p="${cores[0]}" '$1==p&&$4~/^skb_wait_for_more/{ok=1}END{exit !ok}' "$base.all-ps" || bad=1
    fi
    ((bad == 0))
}
wait_idle() {
    local label=$1 attempt=0
    while :; do
        attempt=$((attempt+1))
        if preflight "$label-$attempt"; then return 0; fi
        if ((attempt==1 || attempt%30==0)); then echo "$label waiting for idle GPU/process state"; fi
        sleep 2
    done
}
original=; restore_needed=0
declare -A O
restore_original() {
    local work_rc=$? restore_rc=0 line=
    trap - EXIT INT TERM
    set +e
    if ((restore_needed)); then
        wait_idle restore
        request "reload $restorelib FLASH_CUBIN=${O[cubin]} FLASH_ENGINE_MAXT=${O[maxt]} FLASH_KV_HOST=${O[kv_host]} FLASH_LAYERS=${O[layers]%/*} FLASH_NBLK=${O[nblk]} FLASH_TWO_POSITION=${O[two_position]} FLASH_PREFILL_CHUNK=${O[prefill_chunk]}" "$out/restore.log" || restore_rc=1
        grep -Fxq 'ok: engine reloaded' "$out/restore.log" || restore_rc=1
        if ((restore_rc == 0)); then
            request config "$out/restore-config.log" || restore_rc=1
            line=$(config_line "$out/restore-config.log") || restore_rc=1
            if [[ $line != "$original" ]]; then
                printf 'original: %s\nrestored: %s\n' "$original" "$line" > "$out/RESTORE_CONFIG_MISMATCH.txt"
                restore_rc=1
            fi
        fi
    fi
    sha256sum "$restorelib" > "$out/restore-library.after.sha256" || restore_rc=1
    cmp -s "$out/restore-library.before.sha256" "$out/restore-library.after.sha256" || restore_rc=1
    if ((restore_rc)); then echo 'RESTORE VERIFICATION FAILED; inspect output before further use' >&2; work_rc=1; fi
    jq -n --argjson workload_exit "$work_rc" --argjson restored "$((restore_rc==0))" \
        '{workload_exit:$workload_exit,restore_verified:($restored==1)}' > "$out/status.json"
    exit "$work_rc"
}
trap restore_original EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
wait_idle original
request config "$out/original-config.log"
original=$(config_line "$out/original-config.log")
for key in rev kv_host ple_disk maxt layers nblk dies attn_dies split attn_map img_rows two_position prefill_chunk cubin cubin_fnv; do
    O[$key]=$(get_key "$original" "$key")
done
[[ ${O[layers]} == 48/48 ]] || { echo 'original is not full model' >&2; exit 1; }
validate_fixture() {
    local log=$1 placement=$2
    ! grep -Eq 'FAIL|NONFINITE|device fault|error:|contexts rebuilt' "$log" || return 1
    [[ $(grep -Fxc 'reusecheck PASS' "$log") == 1 ]] || return 1
    awk -v host="$placement" '
        function val(k,i,a){for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]==k)return a[2]}return ""}
        /^reusecheck begin /{
            begin++;if(val("fresh_allocations")!="yes"||val("kv_host")!=host||val("layers")!=48||
                val("nblk")!=512||val("maxt")!=256||val("long")!=96||val("short")!=64||val("first_snapshot")!=16)bad=1
        }
        /^reusecheck run /{
            runs++;s=val("stage");if(seenrun[s]++)bad=1
            ms=val("run_ms");if(ms!~/^[0-9]+([.][0-9]+)?$/||ms+0<=0)bad=1
            r=substr(s,length(s),1);p=(s~/^(long|snapshot-long)-/)?96:64
            expected_snapshot=(s~/^snapshot-long-/)?16+r:0
            expected_resume=(s~/^rollback-short-/)?16+r:0
            if(val("prompt")!=p||val("positions")!=p+2||val("snapshot")!=expected_snapshot||
                val("resumed")!=expected_resume||val("tokens")!~/^[0-9]+,[0-9]+,[0-9]+$/)bad=1
        }
        /^reusecheck guards /{
            guards++;if(val("result")!="PASS"||val("allocations")+0<=0)bad=1
            s=val("stage");if(seenguard[s]++)bad=1
        }
        /^reusecheck compare /{
            comparisons++;s=val("stage")
            if(seencmp[s]++||val("result")!="PASS"||val("equality")!="bitwise"||val("tokens")!=3||val("logits")!=248320)bad=1
            if(s~/^short-after-long-r[0-3]$/){
                r=substr(s,length(s),1);if(val("expected_resume")!=0||val("actual_resume")!=0)bad=1;warm[r]++
            }else if(s~/^rollback-short-r[0-3]$/){
                r=substr(s,length(s),1);want=16+r
                if(val("expected_resume")!=want||val("actual_resume")!=want)bad=1;rollback[r]++
            }else bad=1
        }
        END{
            if(begin!=1||runs!=20||guards!=21||comparisons!=8||seenguard["allocation"]!=1)bad=1
            for(r=0;r<4;r++){
                if(warm[r]!=1||rollback[r]!=1)bad=1
                split("cold long short-after-long snapshot-long rollback-short",st," ")
                for(j=1;j<=5;j++){s=st[j]"-r"r;if(seenrun[s]!=1||seenguard[s]!=1)bad=1}
            }
            exit bad
        }' "$log"
}
check_test_config() {
    local log=$1 expected_host=$2 expected_cubin=$3 line key value rev
    line=$(config_line "$log") || return 1
    rev=$(git -C "$repo" rev-parse HEAD)
    [[ $(get_key "$line" rev) == "$rev" ]] || return 1
    for key in ple_disk dies attn_dies split attn_map img_rows; do
        value=$(get_key "$line" "$key") || return 1
        [[ $value == "${O[$key]}" ]] || return 1
    done
    [[ $(get_key "$line" kv_host) == "$expected_host" &&
       $(get_key "$line" maxt) == 256 && $(get_key "$line" layers) == 48/48 &&
       $(get_key "$line" nblk) == 512 && $(get_key "$line" two_position) == 0 &&
       $(get_key "$line" prefill_chunk) == 0 && $(get_key "$line" cubin) == "$expected_cubin" ]] || return 1
    value=$(get_key "$line" cubin_fnv)
    [[ $value =~ ^[0-9a-fA-F]{16}$ && $value != 0000000000000000 ]]
}
: > "$out/timings.jsonl"
for placement in 0 1; do
    # Reverse the two-image order for the second placement.
    variants=(control candidate)
    if ((placement)); then variants=(candidate control); fi
    for variant in "${variants[@]}"; do
        cubin=$candidate; [[ $variant == control ]] && cubin=$control
        label="h$placement-$variant"
        wait_idle "$label-before-reload"
        restore_needed=1
        request "reload $testlib FLASH_CUBIN=$cubin FLASH_ENGINE_MAXT=256 FLASH_KV_HOST=$placement FLASH_LAYERS=48 FLASH_NBLK=512 FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=0" "$out/$label-reload.log"
        grep -Fxq 'ok: engine reloaded' "$out/$label-reload.log"
        ! grep -Eq 'error:|device fault|contexts rebuilt' "$out/$label-reload.log"
        request config "$out/$label-config.log"
        check_test_config "$out/$label-config.log" "$placement" "$cubin"
        wait_idle "$label-before-fixture"
        request 'reusecheck 96 64 16' "$out/$label-reuse.log"
        validate_fixture "$out/$label-reuse.log" "$placement"
        awk -v placement="$placement" -v variant="$variant" '
            function val(k,i,a){for(i=1;i<=NF;i++){split($i,a,"=");if(a[1]==k)return a[2]}return ""}
            /^reusecheck run /{
                printf "{\"placement\":%d,\"variant\":\"%s\",\"stage\":\"%s\",\"prompt\":%d,\"snapshot\":%d,\"resumed\":%d,\"positions\":%d,\"run_ms\":%.9f,\"build_ms\":%.9f}\n",
                    placement,variant,val("stage"),val("prompt"),val("snapshot"),val("resumed"),val("positions"),val("run_ms"),val("build_ms")
            }' "$out/$label-reuse.log" >> "$out/timings.jsonl"
        echo "$label PASS"
    done
done
sha256sum -c "$out/test-inputs.sha256" > "$out/test-inputs-unchanged.log"
printf 'all four matched placement/variant fixtures passed\n' > "$out/acceptance.txt"
