#!/usr/bin/env nu

const SCRIPT_PATH = path self
const LOCK = '/tmp/flash-sass-engine-validation.lock'
const USAGE = "usage: validation/issue9/with_private_holder.sh REPO NEW_OUT GPU_INDEX GPU_UUID CHECKPOINT B0 B1 MIN_FREE_MIB -- COMMAND...\n"

def save-text [path: string, value: string] {
    $value | save --force --raw $path
}

def capture [argv: list<string>, stdout_path: string, stderr_path: string] {
    let program = $argv | get 0
    let params = $argv | skip 1
    let result = (^$program ...$params | complete)
    save-text $stdout_path $result.stdout
    save-text $stderr_path $result.stderr
    $result
}

def find-holder [manifest: string] {
    let matches = (^pgrep -f $manifest | complete)
    if $matches.exit_code != 0 { return null }
    for pid_text in ($matches.stdout | lines) {
        let pid = try { $pid_text | into int } catch { continue }
        let exe = (^readlink $'/proc/($pid)/exe' | complete)
        if $exe.exit_code != 0 { continue }
        if (($exe.stdout | str trim | path basename) != 'hold') { continue }
        let cmdline = try { open --raw $'/proc/($pid)/cmdline' } catch { '' }
        if ($cmdline | str contains $manifest) { return $pid }
    }
    null
}

def stop-owned-holder [manifest: string] {
    let pid = find-holder $manifest
    if $pid == null { return }
    ^kill -TERM $pid
    while ((^kill -0 $pid | complete).exit_code == 0) { ^sleep 0.1 }
}

def run-command [argv: list<string>, cwd: string, stdout_path: string, stderr_path: string] {
    let program = $argv | get 0
    let params = $argv | skip 1
    do {
        cd $cwd
        ^$program ...$params out> $stdout_path err> $stderr_path
        {exit_code: $env.LAST_EXIT_CODE}
    }
}

def held-run [repo: string, out: string, gpu_index: int, gpu_uuid: string,
              checkpoint: string, b0: int, b1: int, min_free: int,
              command: list<string>] {
    let preflight_gpu = $out | path join 'preflight-gpu.csv'
    let preflight_compute = $out | path join 'preflight-compute.csv'
    let preflight_ps = $out | path join 'preflight-ps.txt'
    let query_gpu = ['nvidia-smi', '-i', ($gpu_index | into string), '--query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu', '--format=csv']
    let gpu = capture $query_gpu $preflight_gpu ($out | path join 'preflight-gpu.stderr')
    if $gpu.exit_code != 0 { print -e -n $gpu.stderr; exit $gpu.exit_code }
    let apps = capture ['nvidia-smi', '-i', ($gpu_index | into string), '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'] $preflight_compute ($out | path join 'preflight-compute.stderr')
    if $apps.exit_code != 0 { print -e -n $apps.stderr; exit $apps.exit_code }
    let ps = capture ['ps', '-eo', 'pid,ppid,etimes,stat,comm'] $preflight_ps ($out | path join 'preflight-ps.stderr')
    if $ps.exit_code != 0 { print -e -n $ps.stderr; exit $ps.exit_code }
    let row = $gpu.stdout | lines | get 1 | split row ',' | each {|field| $field | str trim }
    let got_uuid = $row | get 1
    if $got_uuid != $gpu_uuid { print -e 'GPU index/UUID mismatch'; exit 2 }
    let free_mib = ($row | get 4 | str replace --all --regex '[^0-9]' '' | into int)
    let utilization = ($row | get 6 | str replace --all --regex '[^0-9]' '' | into int)
    let compute_text = $apps.stdout
    if (($compute_text | is-not-empty) or $utilization != 0 or $free_mib < $min_free) { exit 73 }
    save-text ($out | path join '.holder-preflight-ready') 'ready'
    $env.CUDA_VISIBLE_DEVICES = ($gpu_index | into string)
    $env.CUDA_DEVICE_ORDER = 'PCI_BUS_ID'
    $env.SPLIT_SHARES = '1'
    $env.ATTN_DIE = '0'
    $env.ATTN_MAP = '0,0,0,0,0,0,0,0,0,0,0,0'
    $env.QSA_MODEL_MANIFEST = $out | path join 'holder.manifest'
    let binary_hash = (^sha256sum ($repo | path join 'kern/hold') ($repo | path join 'kern/model_test') ($repo | path join 'kern/die.cubin') | complete)
    if $binary_hash.exit_code != 0 { print -e -n $binary_hash.stderr; exit $binary_hash.exit_code }
    save-text ($out | path join 'binaries.sha256') $binary_hash.stdout
    let hostname = (^hostname | complete).stdout | str trim
    let command_json = (^jq -n --arg hostname $hostname --arg gpu $gpu_uuid --arg checkpoint $checkpoint --argjson gpu_index ($gpu_index | into string) --argjson b0 ($b0 | into string) --argjson b1 ($b1 | into string) --args '{hostname:$hostname,gpu_uuid:$gpu,gpu_index:$gpu_index,checkpoint:$checkpoint,b0:$b0,b1:$b1,command:$ARGS.positional,scope:"one-die partial-model fixture; not the full production die layout"}' -- ...$command | complete)
    if $command_json.exit_code != 0 { print -e -n $command_json.stderr; exit $command_json.exit_code }
    save-text ($out | path join 'command.json') $command_json.stdout
    let holder_command = [($repo | path join 'kern/hold') $checkpoint ($gpu_index | into string) ($b0 | into string) ($b1 | into string) $env.QSA_MODEL_MANIFEST 'token_embd' 'output' 'split=0/1']
    let holder_json = (^jq -n --args '$ARGS.positional' -- ...$holder_command | complete)
    if $holder_json.exit_code != 0 { print -e -n $holder_json.stderr; exit $holder_json.exit_code }
    save-text ($out | path join 'holder-command.json') $holder_json.stdout
    let holder_log = $out | path join 'holder.log'
    let holder_args = [($repo | path join 'kern/hold') $checkpoint ($gpu_index | into string) ($b0 | into string) ($b1 | into string) $env.QSA_MODEL_MANIFEST 'token_embd' 'output' 'split=0/1']
    let holder_job = job spawn --description 'owned QSA test holder' {
        ^env 'HOLD_VISIBLE=1' ...$holder_args out+err> $holder_log
    }
    let manifest = $env.QSA_MODEL_MANIFEST
    let holder_deadline = (date now | into int) + 900000000000
    mut holder_pid: any = null
    while not (($manifest + '.ready') | path exists) {
        let pid = find-holder $manifest
        if $pid != null { $holder_pid = $pid; save-text ($out | path join 'holder.pid') (($pid | into string) + "\n") }
        if $holder_pid == null and (job list | where id == $holder_job | is-empty) { print -e 'Owned holder exited before becoming ready'; stop-owned-holder $manifest; exit 1 }
        if ((date now | into int) > $holder_deadline) { print -e 'Owned holder readiness timeout'; stop-owned-holder $manifest; exit 124 }
        ^sleep 1
    }
    if $holder_pid == null {
        $holder_pid = find-holder $manifest
        if $holder_pid == null { print -e 'Owned holder PID unavailable after readiness'; stop-owned-holder $manifest; exit 1 }
        save-text ($out | path join 'holder.pid') (($holder_pid | into string) + "\n")
    }
    $env.QSA_MODEL_HOLDER_PID = ($holder_pid | into string)
    let manifest_hash = (^sha256sum $manifest | complete)
    if $manifest_hash.exit_code != 0 { print -e -n $manifest_hash.stderr; stop-owned-holder $manifest; exit $manifest_hash.exit_code }
    save-text ($out | path join 'holder-manifest.sha256') $manifest_hash.stdout
    capture ['nvidia-smi', '-i', ($gpu_index | into string), '--query-gpu=index,uuid,name,driver_version,memory.free,memory.used,utilization.gpu', '--format=csv'] ($out | path join 'before-model-gpu.csv') ($out | path join 'before-model-gpu.stderr') | ignore
    capture ['nvidia-smi', '-i', ($gpu_index | into string), '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader'] ($out | path join 'before-model-compute.csv') ($out | path join 'before-model-compute.stderr') | ignore
    capture ['ps', '-eo', 'pid,ppid,etimes,stat,comm'] ($out | path join 'before-model-ps.txt') ($out | path join 'before-model-ps.stderr') | ignore
    print $'Owned holder ($holder_pid) ready; running validation under shared lock'
    let result = run-command $command ($repo | path join 'kern') ($out | path join 'model.stdout') ($out | path join 'model.stderr')
    save-text ($out | path join 'exit-code.txt') (($result.exit_code | into string) + "\n")
    let stdout = try { open --raw ($out | path join 'model.stdout') } catch { '' }
    let stderr = try { open --raw ($out | path join 'model.stderr') } catch { '' }
    stop-owned-holder $manifest
    ^nvidia-smi -i ($gpu_index | into string) --query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu --format=csv out+err> ($out | path join 'after-cleanup-gpu.csv') | ignore
    print -n $stdout
    print -e -n $stderr
    exit $result.exit_code
}

def main [...args: string] {
    let locked = (($args | length) > 0 and ($args | get 0) == '--_locked')
    let values = if $locked { $args | skip 1 } else { $args }
    if ($values | length) < 10 { print -e -n $USAGE; exit 2 }
    if ($values | get 8) != '--' { print -e -n $USAGE; exit 2 }
    let repo_result = (^realpath ($values | get 0) | complete)
    if $repo_result.exit_code != 0 { print -e -n $repo_result.stderr; exit $repo_result.exit_code }
    let repo = $repo_result.stdout | str trim
    let out_input = $values | get 1
    let gpu_index_text = $values | get 2
    let gpu_uuid = $values | get 3
    let checkpoint_result = (^realpath ($values | get 4) | complete)
    if $checkpoint_result.exit_code != 0 { print -e -n $checkpoint_result.stderr; exit $checkpoint_result.exit_code }
    let checkpoint = $checkpoint_result.stdout | str trim
    let b0 = $values | get 5
    let b1 = $values | get 6
    let min_free = $values | get 7
    let command = $values | skip 9
    if (($gpu_index_text !~ '^\d+$') or ($b0 !~ '^\d+$') or ($b1 !~ '^\d+$') or ($min_free !~ '^\d+$')) { exit 2 }
    let gpu_index = $gpu_index_text | into int
    let block0 = $b0 | into int
    let block1 = $b1 | into int
    let min_free_mib = $min_free | into int
    let hold_check = (^test -x ($repo | path join 'kern/hold') | complete)
    let model_check = (^test -x ($repo | path join 'kern/model_test') | complete)
    if ($hold_check.exit_code != 0 or $model_check.exit_code != 0) { exit 2 }
    let out = $out_input | path expand
    let made = (^mkdir $out | complete)
    print -n $made.stdout
    print -e -n $made.stderr
    if $made.exit_code != 0 { exit $made.exit_code }
    let out_real = (^realpath $out | complete)
    if $out_real.exit_code != 0 { print -e -n $out_real.stderr; exit $out_real.exit_code }
    let output_dir = $out_real.stdout | str trim
    let marker = $output_dir | path join '.holder-preflight-ready'
    if not $locked {
        if ($marker | path exists) { unlink $marker }
        let started = date now | into int
        let deadline = $started + 1200000000000
        while true {
            let child_args = ['--', '--_locked'] | append $values
            let result = (^flock --timeout 30 --conflict-exit-code 73 --close $LOCK -- $nu.current-exe $SCRIPT_PATH ...$child_args | complete)
            let ready = $marker | path exists
            if $result.exit_code == 73 and not $ready {
                let timestamp = (^date -u '+%Y-%m-%dT%H:%M:%SZ waiting for lock/idle GPU' | complete).stdout
                $timestamp | save --append --raw ($output_dir | path join 'wait.log')
                print -n $timestamp
                if (date now | into int) >= $deadline {
                    print -e 'GPU unavailable within wait window'
                    capture ['nvidia-smi', '-i', ($gpu_index | into string), '--query-gpu=index,uuid,name,memory.free,memory.used,utilization.gpu', '--format=csv'] ($output_dir | path join 'after-cleanup-gpu.csv') ($output_dir | path join 'after-cleanup-gpu.stderr')
                    exit 75
                }
                ^sleep 5
                continue
            }
            if $ready { unlink $marker }
            print -n $result.stdout
            print -e -n $result.stderr
            exit $result.exit_code
        }
    }
    held-run $repo $output_dir $gpu_index $gpu_uuid $checkpoint $block0 $block1 $min_free_mib $command
}
