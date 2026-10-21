# Isolated component evidence runner; caller holds the shared host lock on FD 9.

def fail [message: string, code: int] {
    if $message != "" { print -n --stderr $message }
    exit $code
}

def path-join [...parts: string] {
    $parts | path join
}

def write-text [path: string, value: string, --append] {
    if $append { $value | save --append --raw $path } else { $value | save --force --raw $path }
}

def capture [name: string, args: list] {
    run-external $name ...$args | complete
}

def read-csv [path: string] {
    open --raw $path | from csv --noheaders
}

def result-or-fail [result: record, message: string] {
    if $result.exit_code != 0 {
        if $result.stderr != "" { print -n --stderr $result.stderr }
        if $result.stdout != "" { print -n --stderr $result.stdout }
        fail $"($message)\n" 1
    }
}

def snapshot [out: string, tag: string, gpu: string, allowed_pid: int, initial: bool] {
    let prefix = (path-join $out $"telemetry-($tag)")
    let timestamp = (capture "date" ["-u" "+%FT%TZ"])
    result-or-fail $timestamp "date failed"
    write-text $"($prefix).utc" $timestamp.stdout
    let gpu_query = (capture "nvidia-smi" ["--query-gpu=uuid,name,driver_version,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,power.draw" "--format=csv,noheader,nounits"])
    write-text $"($prefix).gpu.csv" $gpu_query.stdout
    let app_query = (capture "nvidia-smi" ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" "--format=csv,noheader,nounits"])
    write-text $"($prefix).apps.csv" $app_query.stdout
    let errors = $gpu_query.stderr + $app_query.stderr
    write-text $"($prefix).stderr" $errors
    if $gpu_query.exit_code != 0 or $app_query.exit_code != 0 { return false }

    mut seen = 0
    for row in (read-csv $"($prefix).gpu.csv") {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        if ($fields | length) != 9 or not (($fields | get 6) =~ '^[0-9]+$') { return false }
        let row_gpu = $fields | get 0
        let utilization = (($fields | get 6) | into int)
        if $row_gpu == $gpu { $seen = $seen + 1 }
        if ($row_gpu != $gpu or $initial) and $utilization != 0 { return false }
    }
    if $seen != 1 { return false }
    for row in (read-csv $"($prefix).apps.csv") {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        if ($fields | all { |field| $field == "" }) { continue }
        if ($fields | length) != 4 or not (($fields | get 1) =~ '^[0-9]+$') { return false }
        if ($fields | get 0) == $gpu {
            if $allowed_pid == 0 or (($fields | get 1) | into int) != $allowed_pid { return false }
        }
    }
    true
}

def process-exists [job_id: int] {
    (job list | where id == $job_id | length) > 0
}

def child-pids [job_id: int] {
    let found = (job list | where id == $job_id)
    if ($found | length) == 0 { return [] }
    $found | get 0 | get pids
}

def stop-child [job_id: int, pid: int] {
    run-external "kill" "-TERM" ($pid | into string) | complete | ignore
    sleep 1sec
    if (process-exists $job_id) {
        run-external "kill" "-KILL" ($pid | into string) | complete | ignore
        job kill $job_id
    }
}

def launch-case [harness: string, old_cubin: string, new_cubin: string, n: string, seed: string, repeats: string, trace: string, out_json: string, out_stderr: string, env_gpu: string] {
    job spawn {
        with-env { CUDA_DEVICE_ORDER: "PCI_BUS_ID", CUDA_VISIBLE_DEVICES: $env_gpu, ATTN_TRACE_CHECK: $trace, ATTN_TRACE_CTAS: null } {
            let result = (run-external $harness $old_cubin $new_cubin $n $seed $repeats | complete)
            write-text $out_json $result.stdout
            write-text $out_stderr $result.stderr
            write-text $"($out_json).exit" (($result.exit_code | into string) + "\n")
        }
    }
}

def main [...argv: string] {
    let locked = ($argv | length) == 4 and ($argv | get 0) == "locked"
    let args = if $locked { $argv | skip 1 } else { $argv }
    if ($args | length) != 3 {
        print -n --stderr "usage: validation/issue-12/gpu-20261018/run-components.sh BUILD_ROOT NEW_RESULTS GPU_UUID\n"
        exit 2
    }
    let build = $args | get 0
    let out = $args | get 1
    let gpu = $args | get 2
    if not ($build | str starts-with "/") or not ($out | str starts-with "/") or not ($gpu =~ '^GPU-[0-9a-f-]+$') { exit 2 }

    if not $locked {
        let host_link = (capture "readlink" ["/proc/self/fd/9"])
        if $host_link.exit_code != 0 or ($host_link.stdout | str trim) != "/tmp/flash-sass-engine-validation.lock" { exit 75 }
        let host_flock = (capture "flock" ["-n" "9"])
        if $host_flock.exit_code != 0 { exit 75 }
        mkdir $out
        run-external "mkdir" "-p" "/tmp/flash-sass-gpu-locks" | complete | ignore
        let lock_path = $"/tmp/flash-sass-gpu-locks/($gpu).lock"
        let script = (path-join $env.FILE_PWD "run-components.nu")
        let result = (capture "flock" ["-n" "-E" "75" $lock_path "nu" "--no-config-file" $script "locked" $build $out $gpu])
        if $result.exit_code == 75 {
            write-text (path-join $out "status.txt") "BLOCKED_GPU_LOCK\n"
            exit 75
        }
        if $result.stdout != "" { print -n $result.stdout }
        if $result.stderr != "" { print -n --stderr $result.stderr }
        exit $result.exit_code
    }

    let host_link = (capture "readlink" ["/proc/self/fd/9"])
    if $host_link.exit_code != 0 or ($host_link.stdout | str trim) != "/tmp/flash-sass-engine-validation.lock" { exit 75 }
    let host_flock = (capture "flock" ["-n" "9"])
    if $host_flock.exit_code != 0 { exit 75 }
    run-external "mkdir" "-p" "/tmp/flash-sass-gpu-locks" | complete | ignore

    let harness = (path-join $build "matched-new" "kern" "attnpartn_regress")
    let keys = if ($env.KEYS? | is-empty) { "1,1023,1024,1025,2047,2048,2049,2050,2051,2052,4095,4096,4097,8191,8192,8193" } else { $env.KEYS }
    let seeds = if ($env.SEEDS? | is-empty) { "0,77,4242" } else { $env.SEEDS }
    let repeats = if ($env.REPEATS? | is-empty) { "32" } else { $env.REPEATS }
    let modes = if ($env.MODES? | is-empty) { "historical,matched,traced" } else { $env.MODES }
    if not ($keys =~ '^[0-9]+(,[0-9]+)*$') or not ($seeds =~ '^[0-9]+(,[0-9]+)*$') or not ($repeats =~ '^[0-9]+$') or not ($modes =~ '^(historical|matched|traced)(,(historical|matched|traced))*$') { exit 2 }

    let harness_source_result = (capture "cat" [(path-join $build "matched-new" "source-commit.txt")])
    result-or-fail $harness_source_result "missing harness source identity"
    let parent_pid_result = (capture "ps" ["-o" "ppid=" "-p" ($nu.pid | into string)])
    let parent_pid = if $parent_pid_result.exit_code == 0 { (($parent_pid_result.stdout | str trim) | into int) } else { 0 }
    let now = (capture "date" ["-u" "+%FT%TZ"])
    let hostname = (capture "hostname" [])
    let manifest = {
        utc: ($now.stdout | str trim),
        host: ($hostname.stdout | str trim),
        gpu: $gpu,
        build: $build,
        keys: $keys,
        seeds: $seeds,
        modes: $modes,
        repeats: ($repeats | into int),
        harness_source: ($harness_source_result.stdout | str trim),
        host_lock: "/tmp/flash-sass-engine-validation.lock",
        lock_shell_pid: $parent_pid,
        scope: "isolated single-GPU component; no resident-model IPC or reload",
        monitoring: "one-second samples; selected GPU permits only current harness PID; other GPUs must report zero utilization; samples may miss sub-second interference",
        whole_model: false
    }
    (($manifest | to json --indent 2) + "\n") | save --force --raw (path-join $out "manifest.json")
    let source_hash = (capture "sha256sum" [(path-join $env.FILE_PWD "run-components.nu") (path-join $build "matched-new" "kern" "attnpartn_regress")])
    ($source_hash.stdout | save --force --raw (path-join $out "runner-harness.sha256"))
    for hash_file in (glob $"($build)/*.sha256") {
        let copy = (capture "cp" [$hash_file $out])
        result-or-fail $copy "could not copy build hashes"
    }

    if not (snapshot $out "initial" $gpu 0 true) {
        write-text (path-join $out "status.txt") "BLOCKED_ACTIVITY\n"
        exit 75
    }

    write-text (path-join $out "status.txt") "RUNNING\n"
    mut seq = 0
    let all_modes = ($modes | split row ",")
    let all_keys = ($keys | split row ",")
    let all_seeds = ($seeds | split row ",")
    for mode in $all_modes {
        let old_name = if $mode == "historical" { "historical-old" } else if $mode == "matched" { "matched-old" } else { "trace-old" }
        let new_name = if $mode == "historical" { "historical-new" } else if $mode == "matched" { "matched-new" } else { "trace-new" }
        let trace = if $mode == "traced" { "1" } else { "0" }
        for n in $all_keys {
            for seed in $all_seeds {
                $seq = $seq + 1
                let case = $"($mode)-keys($n)-seed($seed)"
                if not (snapshot $out $"($case)-before" $gpu 0 false) {
                    write-text (path-join $out "status.txt") "BLOCKED_ACTIVITY\n"
                    exit 75
                }
                let json_path = (path-join $out $"($case).json")
                let stderr_path = (path-join $out $"($case).stderr")
                let old_cubin = (path-join $build $old_name "kern" "die.cubin")
                let new_cubin = (path-join $build $new_name "kern" "die.cubin")
                let job = (launch-case $harness $old_cubin $new_cubin $n $seed $repeats $trace $json_path $stderr_path $gpu)
                mut pids = []
                for attempt in 0..100 {
                    $pids = (child-pids $job)
                    if ($pids | length) > 0 { break }
                    sleep 0.01sec
                }
                if ($pids | length) == 0 {
                    write-text (path-join $out "status.txt") "HARNESS_FAILED\n"
                    fail "could not observe harness PID\n" 1
                }
                let pid = ($pids | get 0)
                write-text (path-join $out "children.tsv") $"($case)\t($pid)\n" --append
                let start = ((capture "date" ["+%s"]).stdout | str trim | into int)
                mut tick = 0
                while (process-exists $job) {
                    $tick = $tick + 1
                    if not (snapshot $out $"($case)-($tick)" $gpu $pid false) {
                        write-text (path-join $out "status.txt") "INVALID_CONCURRENT_ACTIVITY\n"
                        stop-child $job $pid
                        exit 1
                    }
                    let current = ((capture "date" ["+%s"]).stdout | str trim | into int)
                    if ($current - $start) >= 180 {
                        write-text (path-join $out "status.txt") "HARNESS_TIMEOUT\n"
                        stop-child $job $pid
                        exit 124
                    }
                    sleep 1sec
                }
                let rc_path = $"($json_path).exit"
                let rc = (open --raw $rc_path | str trim | into int)
                run-external "rm" "-f" $rc_path | complete | ignore
                write-text (path-join $out "exits.tsv") $"($case)\t($rc)\n" --append
                if $rc != 0 {
                    write-text (path-join $out "status.txt") "HARNESS_FAILED\n"
                    exit $rc
                }
                if not (snapshot $out $"($case)-after" $gpu 0 false) {
                    write-text (path-join $out "status.txt") "INVALID_CONCURRENT_ACTIVITY\n"
                    exit 1
                }
                let sample = (open $json_path)
                if $sample.keys != ($n | into int) or $sample.seed != ($seed | into int) or not $sample.bit_exact or not $sample.boundary_canaries or $sample.boundary_canary_words != 64 or $sample.partial_values != 49536 or $sample.final_values != 6144 or $sample.steps_per_sample != 64 or $sample.old.event.median_us <= 0 or $sample.new.event.median_us <= 0 or $sample.trace_verified != ($trace == "1") or $sample.timing_trace_enabled != ($trace == "1") or (($trace == "1") and ($sample.trace_launches <= 0 or $sample.trace_records <= 0)) or (($trace == "0") and $sample.trace_launches != 0) {
                    write-text (path-join $out "status.txt") "BAD_COVERAGE\n"
                    exit 1
                }
                print $"PASS ($case)"
            }
        }
    }

    if not (snapshot $out "rejection-before" $gpu 0 false) {
        write-text (path-join $out "status.txt") "BLOCKED_ACTIVITY\n"
        exit 75
    }
    let rejection = (launch-case $harness (path-join $build "matched-old" "kern" "die.cubin") (path-join $build "matched-new" "kern" "die.cubin") "2051" "77" "4" "1" (path-join $out "uninstrumented-rejection.stdout") (path-join $out "uninstrumented-rejection.stderr") $gpu)
    mut rejection_pids = []
    for attempt in 0..100 {
        $rejection_pids = (child-pids $rejection)
        if ($rejection_pids | length) > 0 { break }
        sleep 0.01sec
    }
    let rejection_pid = ($rejection_pids | get 0)
    write-text (path-join $out "children.tsv") $"uninstrumented-rejection\t($rejection_pid)\n" --append
    mut rejection_tick = 0
    let rejection_start = ((capture "date" ["+%s"]).stdout | str trim | into int)
    while (process-exists $rejection) {
        $rejection_tick = $rejection_tick + 1
        if not (snapshot $out $"rejection-($rejection_tick)" $gpu $rejection_pid false) {
            write-text (path-join $out "status.txt") "INVALID_CONCURRENT_ACTIVITY\n"
            stop-child $rejection $rejection_pid
            exit 1
        }
        let current = ((capture "date" ["+%s"]).stdout | str trim | into int)
        if ($current - $rejection_start) >= 180 {
            write-text (path-join $out "status.txt") "HARNESS_TIMEOUT\n"
            stop-child $rejection $rejection_pid
            exit 124
        }
        sleep 1sec
    }
    let rejection_rc_path = $"(path-join $out "uninstrumented-rejection.stdout").exit"
    let rejection_rc = (open --raw $rejection_rc_path | str trim | into int)
    run-external "rm" "-f" $rejection_rc_path | complete | ignore
    write-text (path-join $out "exits.tsv") $"uninstrumented-rejection\t($rejection_rc)\n" --append
    let rejection_stderr = (open --raw (path-join $out "uninstrumented-rejection.stderr"))
    if $rejection_rc != 1 or not ($rejection_stderr =~ '(?m)^trace rejected variant=') or ($rejection_stderr =~ 'undefined symbol|CUDA_ERROR|->') {
        write-text (path-join $out "status.txt") "BAD_REJECTION\n"
        exit 1
    }
    if not (snapshot $out "rejection-after" $gpu 0 false) {
        write-text (path-join $out "status.txt") "INVALID_CONCURRENT_ACTIVITY\n"
        exit 1
    }
    print "PASS target-GPU uninstrumented-image rejection"
    write-text (path-join $out "status.txt") "PASS\n"
    print $"PASS ($seq) component cases; whole_model=false"
}
