#!/usr/bin/env nu

const SCRIPT_PATH = path self
const SHARED_LOCK = '/tmp/flash-sass-engine-validation.lock'

def unbound [line: int, position: int] {
    let message = 'validation/issue9/review-e7921ae/run-direct.sh: line ' + ($line | into string) + ': $' + ($position | into string) + ': unbound variable' + "\n"
    print -e -n $message
    exit 1
}

def epoch [] {
    let result = (^date '+%s' | complete)
    if $result.exit_code != 0 { error make {msg: 'read clock'} }
    $result.stdout | str trim | into int
}

def capture [argv: list<string>, path: string] {
    let program = $argv | get 0
    let params = $argv | skip 1
    let result = (^$program ...$params | complete)
    if ($result.stdout | is-not-empty) { save-text $path $result.stdout }
    if ($result.stderr | is-not-empty) { print -e -n $result.stderr }
    if $result.exit_code != 0 { exit $result.exit_code }
}

def preflight [gpu_uuid: string, out: string] {
    capture ['nvidia-smi', '-i', $gpu_uuid, '--query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu', '--format=csv'] ($out | path join 'preflight-gpu.csv')
    capture ['nvidia-smi', '-i', $gpu_uuid, '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'] ($out | path join 'preflight-compute.csv')
    capture ['ps', '-eo', 'pid,ppid,etimes,stat,comm'] ($out | path join 'preflight-ps.txt')
    let gpu_row = (open ($out | path join 'preflight-gpu.csv') | lines | get 1)
    let gpu_fields = $gpu_row | split row ','
    let free_text = $gpu_fields | get 4 | str replace --all --regex '[^0-9]' ''
    let utilization_text = $gpu_fields | get 6 | str replace --all --regex '[^0-9]' ''
    let free_mib = try { $free_text | into int } catch { 0 }
    let utilization = try { $utilization_text | into int } catch { -1 }
    let compute = try { open --raw ($out | path join 'preflight-compute.csv') } catch { '' }
    not (($compute | is-not-empty) or $utilization != 0 or $free_mib < 2500)
}

def execute-direct [source_root: string, out: string, gpu_uuid: string,
                    gpu_lock: string, source_revision: string] {
    if not (preflight $gpu_uuid $out) { exit 73 }
    'ready' | save --force --raw ($out | path join '.preflight-ready')
    save-text ($out | path join 'source-revision.txt') ($source_revision + "\n")
    let kernel_dir = $source_root | path join 'kern'
    let binaries = do { cd $kernel_dir; ^sha256sum qsa_direct_test die.cubin libengine.so model_test hold_cache | complete }
    let sources = do { cd $kernel_dir; ^sha256sum qsa_direct_test.c qsa_direct_projection.h qsa_direct_fixtures.h qsa_test.c prog.c prog.h dieops.h die.pl ../ref/qsa.c ../ref/qsa.h ../ref/block.c ../ref/block.h ../ref/attn.c ../ref/attn.h ../ref/fmath.c | complete }
    if $binaries.exit_code != 0 { print -e -n $binaries.stderr; exit $binaries.exit_code }
    if $sources.exit_code != 0 { print -e -n $sources.stderr; exit $sources.exit_code }
    $binaries.stdout | save --force --raw ($out | path join 'binaries.sha256')
    $sources.stdout | save --force --raw ($out | path join 'component-sources.sha256')
    let env_vars = {CUDA_VISIBLE_DEVICES: $gpu_uuid, DIE_CUBIN: ($source_root | path join 'kern/die.cubin')}
    let command = do --env {
        $env_vars | load-env
        cd ($source_root | path join 'kern')
        ^perl run_qsa_direct_matrix.pl --binary ($source_root | path join 'kern/qsa_direct_test') --out ($out | path join 'matrix') --timeout 240 out+err> ($out | path join 'matrix-console.log')
        $env.LAST_EXIT_CODE
    }
    let status = $command | into int
    (($status | into string) + "\n") | save --force --raw ($out | path join 'exit-code.txt')
    capture ['nvidia-smi', '-i', $gpu_uuid, '--query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu', '--format=csv'] ($out | path join 'after-gpu.csv')
    capture ['ps', '-eo', 'pid,ppid,etimes,stat,comm'] ($out | path join 'after-ps.txt')
    let log = try { open --raw ($out | path join 'matrix-console.log') } catch { '' }
    print -n $log
    exit $status
}

def main [...args: string] {
    let locked = (($args | length) > 0 and ($args | get 0) == '--_locked')
    let values = if $locked { $args | skip 1 } else { $args }
    if ($values | length) < 5 { unbound (3 + ($values | length)) (1 + ($values | length)) }
    let source_input = $values | get 0
    let out_input = $values | get 1
    let gpu_uuid = $values | get 2
    let gpu_lock_input = $values | get 3
    let source_revision = $values | get 4
    let source_result = (^realpath $source_input | complete)
    if $source_result.exit_code != 0 { print -e -n $source_result.stderr; exit $source_result.exit_code }
    let source_root = $source_result.stdout | str trim
    let out = $out_input | path expand
    ^mkdir -p $out
    let gpu_lock = $gpu_lock_input | path expand
    if $locked { execute-direct $source_root $out $gpu_uuid $gpu_lock $source_revision }
    let deadline = (epoch) + 1200
    while true {
        let child_args = ['--', '--_locked'] | append [$source_root $out $gpu_uuid $gpu_lock $source_revision]
        let result = (^flock --wait 30 --conflict-exit-code 73 $SHARED_LOCK -- ^flock --nonblock --conflict-exit-code 73 $gpu_lock -- $nu.current-exe $SCRIPT_PATH ...$child_args | complete)
        let ready = ($out | path join '.preflight-ready') | path exists
        if ($result.exit_code == 73 and not $ready) {
            print $'Waiting for shared validation lock and idle ($gpu_uuid)'
            if (epoch) >= $deadline { print -e 'GPU unavailable within wait window'; exit 75 }
            ^sleep 5
            continue
        }
        print -n $result.stdout
        print -e -n $result.stderr
        exit $result.exit_code
    }
}

def save-text [path: string, value: string] {
    $value | save --force --raw $path
}
