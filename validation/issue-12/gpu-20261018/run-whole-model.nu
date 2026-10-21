# Private six-die issue #12 benchmark runner.

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

def exit-run [out: string, code: int] {
    write-text (path-join $out "runner.exit") (($code | into string) + "\n")
    exit $code
}

def process-exists [job_id: int] {
    (job list | where id == $job_id | length) > 0
}

def child-pids [job_id: int] {
    let found = (job list | where id == $job_id)
    if ($found | length) == 0 { return [] }
    $found | get 0 | get pids
}

def residency [out: string, pids: list, suffix: string] {
    mut lines = []
    for pid in $pids {
        let stat = (open --raw $"/proc/($pid)/stat" | str trim)
        let fields = ($stat | split row " ")
        if ($fields | length) < 22 { return false }
        $lines = ($lines | append $"($pid) (($fields | get 1)) (($fields | get 21))")
    }
    write-text (path-join $out $"resident-($suffix).identities") (($lines | str join "\n") + "\n")
    true
}

def validate-preflight [out: string] {
    let gpu_path = (path-join $out "preflight.gpu.csv")
    let apps_path = (path-join $out "preflight.apps.csv")
    let gpu = (capture "nvidia-smi" ["--query-gpu=index,uuid,memory.free,utilization.gpu,utilization.memory" "--format=csv,noheader,nounits"])
    write-text $gpu_path $gpu.stdout
    let apps = (capture "nvidia-smi" ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" "--format=csv,noheader,nounits"])
    write-text $apps_path $apps.stdout
    if $gpu.exit_code != 0 or $apps.exit_code != 0 { return false }
    let gpu_rows = (read-csv $gpu_path)
    if ($gpu_rows | length) != 8 { return false }
    for row in $gpu_rows {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        if ($fields | length) != 5 { return false }
        let index = try { ($fields | get 0) | into int } catch { return false }
        let free = try { ($fields | get 2) | into int } catch { return false }
        let compute = try { ($fields | get 3) | into int } catch { return false }
        let memory = try { ($fields | get 4) | into int } catch { return false }
        if $compute != 0 or $memory != 0 or ($index < 6 and $free < 160) { return false }
    }
    let app_rows = (read-csv $apps_path)
    if ($app_rows | length) != 7 { return false }
    mut pids = []
    for row in $app_rows {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        if ($fields | length) != 4 { return false }
        let pid = $fields | get 1
        let program = $fields | get 2
        if not ($pid =~ '^[0-9]+$') or not ($program =~ '(^|/)(hold|fence)$') { return false }
        $pids = ($pids | append ($pid | into int))
    }
    $pids = ($pids | uniq | sort)
    write-text (path-join $out "resident.pids") (($pids | each { |pid| $pid | into string } | str join "\n") + "\n")
    if not (residency $out $pids "before") { return false }
    $pids
}

def device-locks [out: string] {
    let rows = (read-csv (path-join $out "preflight.gpu.csv"))
    mut locks = []
    for row in $rows {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        let index = ($fields | get 0 | into int)
        let uuid = $fields | get 1
        if $index != 6 { $locks = ($locks | append $uuid) }
    }
    $locks
}

def lock-step [build: string, out: string, index: int, locks: list] {
    if $index >= ($locks | length) {
        run-core $build $out
    } else {
        let uuid = $locks | get $index
        let lock_path = $"/tmp/flash-sass-gpu-locks/($uuid).lock"
        let script = (path-join $env.FILE_PWD "run-whole-model.nu")
        let result = (capture "flock" ["-n" "-E" "75" $lock_path "nu" "--no-config-file" $script "locked" $build $out (($index + 1) | into string)])
        if $result.exit_code == 75 {
            let status_path = (path-join $out "runner.status")
            if (open --raw $status_path | str trim) == "PREFLIGHT" {
                write-text $status_path "BLOCKED_DEVICE_LOCK\n"
                exit-run $out 75
            }
        }
        if $result.stdout != "" { print -n $result.stdout }
        if $result.stderr != "" { print -n --stderr $result.stderr }
        if $result.exit_code != 0 { exit-run $out $result.exit_code }
    }
}

def monitor-apps [out: string, tag: string, own: int] {
    let resident = (open --raw (path-join $out "resident.pids") | lines | each { |line| $line | into int })
    let path = (path-join $out $"telemetry-($tag).apps.csv")
    let rows = (read-csv $path)
    for row in $rows {
        let fields = ($row | values | each { |field| $field | into string | str trim })
        if ($fields | length) == 0 or (($fields | all { |field| $field == "" })) { continue }
        if ($fields | length) != 4 or not (($fields | get 1) =~ '^[0-9]+$') { return false }
        let pid = (($fields | get 1) | into int)
        if $pid != $own and not ($resident | any { |allowed| $allowed == $pid }) { return false }
    }
    true
}

def monitor-pmon [path: string, own: int] {
    let text = open --raw $path
    for line in ($text | lines) {
        let stripped = ($line | str trim)
        if $stripped == "" or ($stripped | str starts-with "#") { continue }
        let fields = ($stripped | split row " " | where { |field| $field != "" })
        if ($fields | length) != 8 { return false }
        let pid_text = $fields | get 1
        if not ($pid_text =~ '^[0-9]+$') { continue }
        let pid = $pid_text | into int
        if $pid != $own {
            for column in [3 4] {
                let value = $fields | get $column
                if ($value =~ '^[0-9]+$') and ($value | into int) > 0 { return false }
            }
        }
    }
    true
}

def run-core [build: string, out: string] {
    let manifests = [
        "/tmp/hold-h0.manifest" "/tmp/hold-h1.manifest" "/tmp/hold-h2.manifest"
        "/tmp/hold-h3.manifest" "/tmp/hold-h4.manifest" "/tmp/hold-h5.manifest"
    ]
    for manifest in $manifests {
        if not (path-join $"($manifest).ready" | path exists) {
            write-text (path-join $out "runner.status") "MISSING_MANIFEST\n"
            exit-run $out 1
        }
    }
    let before_hashes = (capture "sha256sum" $manifests)
    if $before_hashes.exit_code != 0 { write-text (path-join $out "runner.status") "MANIFEST_HASH_FAILED\n"; exit-run $out 1 }
    write-text (path-join $out "manifests-before.sha256") $before_hashes.stdout

    let runtime = [
        (path-join $env.FILE_PWD "run-whole-model.nu"),
        (path-join $build "matched-new" "kern" "flashcore"),
        (path-join $build "matched-new" "kern" "libengine_cta_remap.so"),
        (path-join $build "matched-old" "kern" "die.cubin"),
        (path-join $build "matched-new" "kern" "die.cubin"),
        (path-join $build "trace-old" "kern" "die.cubin"),
        (path-join $build "trace-new" "kern" "die.cubin")
    ]
    let runtime_hashes = (capture "sha256sum" $runtime)
    if $runtime_hashes.exit_code != 0 { write-text (path-join $out "runner.status") "RUNTIME_HASH_FAILED\n"; exit-run $out 1 }
    write-text (path-join $out "runtime.sha256") $runtime_hashes.stdout
    let ple = "/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf"
    let checkpoint = (capture "stat" ["-c" "%n %s %y" $ple])
    if $checkpoint.exit_code != 0 { write-text (path-join $out "runner.status") "CHECKPOINT_MISSING\n"; exit-run $out 1 }
    write-text (path-join $out "checkpoint.stat") $checkpoint.stdout
    let commands = $"ctabench ($out)/measurements ($build)/matched-old/kern/die.cubin ($build)/matched-new/kern/die.cubin ($build)/trace-old/kern/die.cubin ($build)/trace-new/kern/die.cubin 2048 32 8\n"
    write-text (path-join $out "commands.txt") $commands

    let program = (path-join $build "matched-new" "kern" "flashcore")
    let engine = (path-join $build "matched-new" "kern" "libengine_cta_remap.so")
    let core_job = (job spawn {
        let result = (
            run-external "env" "-i" "PATH=/usr/bin:/bin" "LD_BIND_NOW=1"
                "CUDA_DEVICE_ORDER=PCI_BUS_ID" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7"
                "FLASH_ANCHOR_DIE=7" "FLASH_MAXT=2080" "FLASH_ENGINE_MAXT=2080" "FLASH_MAXIMG=0"
                "FLASH_KV_HOST=0" "FLASH_PLE_DISK=1" "FLASH_PLE_OFF=192" "FLASH_HOLD_WAIT_S=0"
                "FLASH_NBLK=512" "FLASH_LAYERS=48" "FLASH_TWO_POSITION=0" "FLASH_PREFILL_CHUNK=0"
                "FLASH_WATCHDOG_S=60" "SPLIT_SHARES=112,97,81,116,97,97" "ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5"
                $"FLASH_CUBIN=($build)/matched-new/kern/die.cubin"
                $program $ple $engine $manifests
            < (path-join $out "commands.txt") | complete
        )
        write-text (path-join $out "core.stdout") $result.stdout
        write-text (path-join $out "core.stderr") $result.stderr
        write-text (path-join $out "core.exit") (($result.exit_code | into string) + "\n")
    })
    let pids = (child-pids $core_job)
    if ($pids | length) == 0 { write-text (path-join $out "runner.status") "CORE_START_FAILED\n"; exit-run $out 1 }
    let core_pid = $pids | get 0
    write-text (path-join $out "core.pid") (($core_pid | into string) + "\n")
    write-text (path-join $out "runner.status") "RUNNING\n"
    let start = ((capture "date" ["+%s"]).stdout | str trim | into int)
    mut tick = 0
    while (process-exists $core_job) {
        $tick = $tick + 1
        let tag = $"($tick)"
        let timestamp = (capture "date" ["-u" "+%FT%TZ"])
        write-text (path-join $out $"telemetry-($tag).utc") $timestamp.stdout
        let gpu_query = (capture "nvidia-smi" ["--query-gpu=index,uuid,temperature.gpu,clocks.sm,clocks.mem,memory.free,utilization.gpu,utilization.memory,power.draw" "--format=csv,noheader,nounits"])
        write-text (path-join $out $"telemetry-($tag).gpu.csv") $gpu_query.stdout
        let apps_query = (capture "nvidia-smi" ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" "--format=csv,noheader,nounits"])
        write-text (path-join $out $"telemetry-($tag).apps.csv") $apps_query.stdout
        if $gpu_query.exit_code != 0 or $apps_query.exit_code != 0 or not (monitor-apps $out $tag $core_pid) {
            write-text (path-join $out "runner.status") "INVALID_FOREIGN_COMPUTE\n"
            run-external "kill" "-TERM" ($core_pid | into string) | complete | ignore
            sleep 1sec
            if (process-exists $core_job) { run-external "kill" "-KILL" ($core_pid | into string) | complete | ignore; job kill $core_job }
            exit-run $out 1
        }
        let pmon = (capture "nvidia-smi" ["pmon" "-c" "1" "-s" "u"])
        write-text (path-join $out $"telemetry-($tag).pmon.txt") $pmon.stdout
        if $pmon.exit_code != 0 or not (monitor-pmon (path-join $out $"telemetry-($tag).pmon.txt") $core_pid) {
            write-text (path-join $out "runner.status") "INVALID_FOREIGN_ACTIVITY\n"
            run-external "kill" "-TERM" ($core_pid | into string) | complete | ignore
            sleep 1sec
            if (process-exists $core_job) { run-external "kill" "-KILL" ($core_pid | into string) | complete | ignore; job kill $core_job }
            exit-run $out 1
        }
        let now = ((capture "date" ["+%s"]).stdout | str trim | into int)
        if ($now - $start) > 900 {
            write-text (path-join $out "runner.status") "CORE_TIMEOUT\n"
            run-external "kill" "-TERM" ($core_pid | into string) | complete | ignore
            sleep 1sec
            if (process-exists $core_job) { run-external "kill" "-KILL" ($core_pid | into string) | complete | ignore; job kill $core_job }
            exit-run $out 124
        }
        sleep 1sec
    }
    let core_rc = (open --raw (path-join $out "core.exit") | str trim | into int)
    write-text (path-join $out "core.exit") (($core_rc | into string) + "\n")
    let measurements_status = try { open --raw (path-join $out "measurements" "status.txt") | str trim } catch { "" }
    if $core_rc != 0 or $measurements_status != "PASS" {
        write-text (path-join $out "runner.status") "BENCHMARK_FAILED\n"
        exit-run $out 1
    }
    let checked_hashes = (capture "sha256sum" ["-c" (path-join $out "manifests-before.sha256")])
    if $checked_hashes.exit_code != 0 { write-text (path-join $out "runner.status") "MANIFEST_CHANGED\n"; exit-run $out 1 }
    write-text (path-join $out "manifests-after.check") $checked_hashes.stdout

    let pids = (open --raw (path-join $out "resident.pids") | lines | each { |line| $line | into int })
    if not (residency $out $pids "after") { write-text (path-join $out "runner.status") "RESIDENT_CHANGED\n"; exit-run $out 1 }
    let before = (open --raw (path-join $out "resident-before.identities"))
    let after = (open --raw (path-join $out "resident-after.identities"))
    if $before != $after { write-text (path-join $out "runner.status") "RESIDENT_CHANGED\n"; exit-run $out 1 }
    let apps = (capture "nvidia-smi" ["--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" "--format=csv,noheader,nounits"])
    if $apps.exit_code != 0 { write-text (path-join $out "runner.status") "AFTER_QUERY_FAILED\n"; exit-run $out 1 }
    write-text (path-join $out "after.apps.csv") $apps.stdout
    let finish = (capture "date" ["-u" "+%FT%TZ"])
    write-text (path-join $out "finished.utc") $finish.stdout
    write-text (path-join $out "runner.status") "PASS\n"
    print "PASS private six-die benchmark; resident holders unchanged"
    exit-run $out 0
}

def main [...argv: string] {
    let locked = ($argv | length) == 4 and ($argv | get 0) == "locked"
    if $locked {
        let build = $argv | get 1
        let out = $argv | get 2
        let index = (($argv | get 3) | into int)
        let links = (device-locks $out)
        lock-step $build $out $index $links
        exit-run $out 0
    }
    if ($argv | length) != 2 {
        print -n --stderr "usage: validation/issue-12/gpu-20261018/run-whole-model.sh BUILD_ROOT NEW_RUN_DIRECTORY\n"
        exit 2
    }
    let build = $argv | get 0
    let out = $argv | get 1
    if not ($build | str starts-with "/") or not ($out | str starts-with "/") { exit 2 }
    let host_link = (capture "readlink" ["/proc/self/fd/9"])
    if $host_link.exit_code != 0 or ($host_link.stdout | str trim) != "/tmp/flash-sass-engine-validation.lock" { exit 75 }
    let host_flock = (capture "flock" ["-n" "9"])
    if $host_flock.exit_code != 0 { exit 75 }
    mkdir $out
    write-text (path-join $out "runner.status") "PREFLIGHT\n"
    run-external "mkdir" "-p" "/tmp/flash-sass-gpu-locks" | complete | ignore
    let started = (capture "date" ["-u" "+%FT%TZ"])
    write-text (path-join $out "started.utc") $started.stdout
    let result = (validate-preflight $out)
    if not ($result | describe | str starts-with "list") {
        write-text (path-join $out "runner.status") "BLOCKED_ACTIVITY_OR_MEMORY\n"
        exit-run $out 75
    }
    let locks = (device-locks $out)
    let script = (path-join $env.FILE_PWD "run-whole-model.nu")
    let lock_result = (capture "nu" ["--no-config-file" $script "locked" $build $out "0"])
    if $lock_result.stdout != "" { print -n $lock_result.stdout }
    if $lock_result.stderr != "" { print -n --stderr $lock_result.stderr }
    if $lock_result.exit_code != 0 { exit-run $out $lock_result.exit_code }
    exit-run $out 0
}
