# Exact parent/remap evidence runner with an exclusive and freshly checked GPU host.

def m_fail [message: string, code: int = 1] {
    print -n --stderr $message
    if not ($message | str ends-with "\n") { print -n --stderr "\n" }
    exit $code
}

def m_path [...parts: string] { $parts | path join }

def m_capture [command: string, args: list, environment: record = {}] {
    if ($environment | columns | length) == 0 {
        run-external $command ...$args | complete
    } else {
        with-env $environment { run-external $command ...$args | complete }
    }
}

def m_process_exists [job_id: int] {
    (job list | where id == $job_id | length) > 0
}

def m_float [value] {
    if $value == ($value | math floor) { return (($value | into int) | to json) }
    let exponent = (($value | math abs | math log 10 | math floor) | into int)
    let decimals = 14 - $exponent
    ($value | into string --decimals $decimals | str replace --regex '0+$' "" | str replace --regex '\.$' "")
}

def m_json [value] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        let keys = ($value | columns | sort)
        "{" + ($keys | each { |key| (($key | to json) + ":" + (m_json ($value | get $key))) } | str join ",") + "}"
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        "[" + ($value | each { |item| m_json $item } | str join ",") + "]"
    } else if $kind == "float" {
        m_float $value
    } else {
        $value | to json
    }
}

def m_uint [value, label: string, max: int = 4294967295] {
    if ($value | describe) == "int" {
        if $value < 0 or $value > $max { return { ok: false, error: $"invalid unsigned ($label)" } }
        return { ok: true, value: $value }
    }
    let text = ($value | into string | str trim)
    if not ($text =~ '^(0|[1-9][0-9]*)$') { return { ok: false, error: $"invalid unsigned ($label)" } }
    let number = try { $text | into int } catch { return { ok: false, error: $"invalid unsigned ($label)" } }
    if $number > $max { return { ok: false, error: $"invalid unsigned ($label)" } }
    { ok: true, value: $number }
}

def m_csv-lines [text: string] {
    $text | lines | where { |line| ($line | str trim) != "" } | each { |line| $line | split row "," | each { |field| $field | str trim } }
}

def cpu-configured-llama [comm: string, args: string] {
    if $comm != "llama-server" or not ($args =~ '^(?:\S*/)?llama-server(?:\s|$)') { return false }
    let words = ($args | split row " " | where { |word| $word != "" })
    mut layers = []
    for index in 0..(($words | length) - 1) {
        let word = $words | get $index
        if $word == "--n-gpu-layers" or $word == "--gpu-layers" or $word == "-ngl" {
            if ($index + 1) < ($words | length) { $layers = ($layers | append ($words | get ($index + 1))) }
        } else if ($word | str starts-with "--n-gpu-layers=") or ($word | str starts-with "--gpu-layers=") or ($word | str starts-with "-ngl=") {
            $layers = ($layers | append ($word | split row "=" | get 1))
        }
    }
    ($layers | length) > 0 and ($layers | all { |value| $value == "0" })
}

def busy-reasons [s: record, policy: record = {}, ignore_target_utilization: bool = false] {
    let residents = (($policy | columns) | any { |key| $key == "idle_resident_queue" }) and $policy.idle_resident_queue != null
    let required = ["compute_apps" "ps" "gpu"]
    let names = if $residents { $required | append ["all_gpu_load" "queue"] } else { $required }
    for name in $names {
        if not (($s | columns) | any { |key| $key == $name }) { return { error: $"($name) query failed; GPU idle state is unknown\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        let query = $s | get $name
        if ($query | describe | str starts-with "record") == false or not (($query | columns) | any { |key| $key == "exit" }) or $query.exit != 0 {
            return { error: $"($name) query failed; GPU idle state is unknown\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] }
        }
        if not (($query | columns) | any { |key| $key == "stdout" }) {
            return { error: $"($name) query has no stdout\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] }
        }
    }
    if not ($s.ps.stdout =~ '\S') { return { error: "ps returned no process inventory\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
    if not ($s.gpu.stdout =~ 'GPU-[0-9a-f-]+' ) { return { error: "GPU metadata query returned no UUID\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }

    mut busy = []
    mut resident_pid = {}
    mut load_seen = {}
    mut compute_pid = {}
    mut cpu_workers = []
    mut queue_port = ""

    if $residents {
        let queue_match = ($policy.idle_resident_queue | parse --regex '^http://127\.0\.0\.1:(?<port>[1-9][0-9]{0,4})/queue$')
        if ($queue_match | length) != 1 { return { error: "idle-resident-queue must be http://127.0.0.1:PORT/queue\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        $queue_port = $queue_match | get 0 | get port
        if ($queue_port | into int) > 65535 { return { error: "idle-resident-queue must be http://127.0.0.1:PORT/queue\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        if not ($policy.gpu =~ '^GPU-[0-9a-f-]+$') { return { error: "idle resident mode requires a GPU UUID, not an index\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        let queue = try { $s.queue.stdout | from json } catch { null }
        if $queue == null or not (($queue | describe) | str starts-with "record") { return { error: "invalid queue JSON\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        for key in ["running" "waiting"] {
            if not (($queue | columns) | any { |column| $column == $key }) { return { error: $"invalid queue ($key) count\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            let count = m_uint ($queue | get $key) $"queue ($key)"
            if not $count.ok { return { error: $"invalid queue ($key) count\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            if $count.value > 0 { $busy = ($busy | append $"request queue ($key)=($count.value)") }
        }
        for row in (m_csv-lines $s.all_gpu_load.stdout) {
            if ($row | length) != 3 or not (($row | get 0) =~ '(?i)^GPU-[0-9a-f-]+$') { return { error: "invalid all-GPU utilization row\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            let compute = m_uint ($row | get 1) "all-GPU compute"
            let memory = m_uint ($row | get 2) "all-GPU memory"
            if not $compute.ok or not $memory.ok or $compute.value > 100 or $memory.value > 100 { return { error: "invalid all-GPU utilization row\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            let key = ($row | get 0 | str lowercase)
            if (($load_seen | columns) | any { |column| $column == $key }) { return { error: "duplicate GPU utilization row\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            $load_seen = ($load_seen | insert $key true)
            if ($compute.value > 0 or $memory.value > 0) and not ($ignore_target_utilization and $key == ($policy.gpu | str lowercase)) {
                $busy = ($busy | append $"GPU activity: ($row | get 0) compute=($compute.value) memory=($memory.value)")
            }
        }
        if not (($load_seen | columns) | any { |key| $key == ($policy.gpu | str lowercase) }) { return { error: "target GPU missing from utilization inventory\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
    }

    for line in ($s.compute_apps.stdout | lines) {
        if ($line | str trim) == "" { continue }
        let row = ($line | parse --regex '^\s*(?<id>GPU-[0-9a-f-]+)\s*,\s*(?<pid>[0-9]+)\s*,\s*(?<program>[^,]+),')
        if ($row | length) != 1 { return { error: "unrecognized nvidia-smi compute-app output\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        let record = $row | get 0
        let pid = $record.pid
        let id = $record.id
        let program = ($record.program | str trim)
        $compute_pid = ($compute_pid | upsert $pid true)
        if $residents and not (($load_seen | columns) | any { |key| $key == ($id | str lowercase) }) { return { error: "compute-app GPU missing from utilization inventory\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        let resident = ($program | parse --regex '(?:^|/)(?<name>hold|fence|flashcore)$')
        if $residents and ($id | str lowercase) != ($policy.gpu | str lowercase) and ($resident | length) == 1 {
            let name = $resident | get 0 | get name
            if (($resident_pid | columns) | any { |key| $key == $pid }) and ($resident_pid | get $pid) != $name {
                return { error: "conflicting resident process identity\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] }
            }
            $resident_pid = ($resident_pid | upsert $pid $name)
        } else {
            $busy = ($busy | append $"compute app: ($line)")
        }
    }

    mut verified = {}
    for line in ($s.ps.stdout | lines) {
        if ($line | str trim) == "" { continue }
        let row = ($line | parse --regex '^\s*(?<pid>[0-9]+)\s+(?<ppid>[0-9]+)\s+(?<stat>\S+)\s+(?<comm>\S+)\s*(?<args>.*)$')
        if ($row | length) != 1 { return { error: "unrecognized ps process row\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
        let p = $row | get 0
        if (($resident_pid | columns) | any { |key| $key == $p.pid }) {
            let name = $resident_pid | get $p.pid
            let first_arg = ($p.args | split row " " | get 0)
            if $p.comm != $name or ($first_arg | path basename) != $name or ($p.stat | str contains "Z") { return { error: $"resident process identity mismatch: pid=($p.pid)\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
            $verified = ($verified | insert $p.pid true)
            continue
        }
        let process_args = ($p.args | split row " " | where { |word| $word != "" })
        if $residents and $p.comm == "flashd" and not ($p.stat | str contains "Z") and ($process_args | length) == 4 and (($process_args | get 0 | path basename) == "flashd") and ($process_args | get 3) == $"127.0.0.1:($queue_port)" { continue }
        if $residents and not (($compute_pid | columns) | any { |key| $key == $p.pid }) and not ($p.stat | str contains "Z") and (cpu-configured-llama $p.comm $p.args) {
            $cpu_workers = ($cpu_workers | append { pid: ($p.pid | into int), comm: $p.comm, reason: "explicit zero GPU layers; no reported compute app" })
            continue
        }
        let worker = $p.comm =~ '^(?:flashcore|flashd|hold|fence|attnpartn_regre|qsa_oracle|qsa_test|model_test|attn_test|die_test|memtest|gemv.*test|llama-server|llama-cli|llama-perplexit)'
        let wrapped = $p.args =~ '^(?:(?:\S*/)?(?:perl|ruby|bash|sh)(?:\s+-[^ \t]+)*\s+)?(?:\S*/)?(?:flashcore[^\s/]*|flashd|hold(?:[-_.][^\s/]*)?|fence(?:[-_.][^\s/]*)?|attnpartn_regress|attnpartn_model_compare|qsa_test|model_test|attn_test|die_test|memtest|gemv[^\s/]*test|llama-server|llama-cli|llama-perplexity)(?:\s|$)'
        if $worker or $wrapped { $busy = ($busy | append $"worker process: pid=($p.pid) ppid=($p.ppid) stat=($p.stat) comm=($p.comm)") }
    }
    for pid in ($resident_pid | columns) {
        if not (($verified | columns) | any { |key| $key == $pid }) { return { error: $"resident process missing from ps inventory: pid=($pid)\n", busy: [], allowed_idle_residents: [], cpu_configured_workers: [] } }
    }
    mut allowed = []
    for pid in ($resident_pid | columns | sort) {
        $allowed = ($allowed | append { pid: ($pid | into int), comm: ($resident_pid | get $pid) })
    }
    { error: "", busy: $busy, allowed_idle_residents: $allowed, cpu_configured_workers: $cpu_workers }
}

def wait-idle [states: list, policy: record, wait_seconds: float, poll_seconds: float] {
    let limit = ($wait_seconds / $poll_seconds | math ceil | into int) + 1
    for index in 0..($limit - 1) {
        let s = if $index < ($states | length) { $states | get $index } else { $states | get (($states | length) - 1) }
        let result = busy-reasons $s $policy
        if $result.error != "" { return { status: "ERROR", error: $result.error, attempts: ($index + 1) } }
        if ($result.busy | length) == 0 { return { status: "IDLE", attempts: ($index + 1) } }
        if $index == ($limit - 1) { return { status: "BLOCKED_BUSY", error: "BLOCKED_BUSY: host still has other GPU workers\n", attempts: ($index + 1) } }
    }
    { status: "IDLE", attempts: 1 }
}

def post-idle [states: list, policy: record, settle: float] {
    mut snapshots = []
    let count = if $settle == 0 { 1 } else { 2 }
    for index in 0..($count - 1) {
        let s = if $index < ($states | length) { $states | get $index } else { $states | get (($states | length) - 1) }
        let busy = busy-reasons $s $policy
        if $busy.error != "" { return { status: "ERROR", error: $busy.error, snapshots: $snapshots } }
        let foreign = busy-reasons $s $policy true
        if $foreign.error != "" { return { status: "ERROR", error: $foreign.error, snapshots: $snapshots } }
        $snapshots = ($snapshots | append { busy_reasons: $busy.busy })
        if ($foreign.busy | length) > 0 { return { status: "CONTENTION", error: ("post-run contention: " + ($foreign.busy | str join "; ") + "\n"), snapshots: $snapshots } }
        if ($busy.busy | length) == 0 { return { status: "IDLE", snapshots: $snapshots } }
        if $index == ($count - 1) { return { status: "TIMEOUT", error: $"post-run target utilization did not settle within ($settle)s\n", snapshots: $snapshots } }
    }
    { status: "IDLE", snapshots: $snapshots }
}

def validate-preflight [result: record] {
    if $result.exit != 0 { return { ok: false, error: "eager driver-ABI self-test failed before GPU reservation\n" } }
    let data = try { $result.stdout | from json } catch { null }
    if $data == null or ($data | describe | str starts-with "record") == false or $data.self_test != "pass" or not $data.cpu_only or $data.negative_canary_cases != 16 or $data.negative_trace_cases != 14 or $data.mapping_cases != 64 {
        return { ok: false, error: "incomplete CPU-only driver-ABI self-test evidence\n" }
    }
    { ok: true, value: $data }
}

def validate-result [value: record, trace: bool] {
    if not $value.bit_exact or not $value.boundary_canaries or $value.boundary_canary_words != 64 or $value.partial_values != 49536 or $value.final_values != 6144 or $value.steps_per_sample != 64 {
        return { ok: false, error: "missing exact comparison, boundary canaries, or timing coverage\n" }
    }
    for arm in ["old" "new"] {
        let median = $value | get $arm | get event | get median_us
        if ($median | describe) == "nothing" or not (($median | into string) =~ '^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$') or $median <= 0 {
            return { ok: false, error: $"missing positive ($arm) event median\n" }
        }
    }
    if ($value.trace_verified | describe) == "nothing" or ($value.trace_launches | describe) == "nothing" {
        return { ok: false, error: "missing trace mode metadata\n" }
    }
    if $trace and (not $value.trace_verified or $value.trace_launches <= 0) { return { ok: false, error: "trace requested but harness did not verify fresh physical-CTA traces\n" } }
    if not $trace and ($value.trace_verified or $value.trace_launches != 0) { return { ok: false, error: "untraced run unexpectedly contains trace instrumentation\n" } }
    { ok: true, value: $value }
}

def m_add [checks: list, name: string, passed: bool] {
    $checks | append { name: $name, passed: $passed }
}

def inherited-host-lock [actual_path: string, expected_path: string, lock_info: string, same_open_file: bool] {
    if $actual_path != $expected_path { return { ok: false, error: "inherited host-lock descriptor names a different file\n" } }
    if not ($lock_info =~ '(?m)^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s') { return { ok: false, error: "inherited host-lock descriptor has no exclusive flock\n" } }
    if not $same_open_file { return { ok: false, error: "BLOCKED_BUSY: inherited host-lock verification failed\n" } }
    { ok: true, error: "" }
}

def m_run [command: string, args: list, timeout: float = 60.0] {
    let timeout_args = ["--signal=TERM" "--kill-after=1" $"($timeout)s" $command] ++ $args
    let result = (m_capture "timeout" $timeout_args)
    if $result.exit_code == 124 {
        return { stdout: $result.stdout, stderr: ($result.stderr + "RUNNER TIMEOUT\n"), exit: 124 }
    }
    { stdout: $result.stdout, stderr: $result.stderr, exit: $result.exit_code }
}

def self-test [] {
    mut checks = []
    let clean = {
        compute_apps: { exit: 0, stdout: "" },
        ps: { exit: 0, stdout: "1 0 S init /sbin/init\n" },
        gpu: { exit: 0, stdout: "uuid, name\nGPU-01234567-89ab-cdef-0123-456789abcdef, Tesla M60\n" }
    }
    let idle = busy-reasons $clean
    $checks = (m_add $checks "idle inventory accepted" ($idle.error == "" and ($idle.busy | length) == 0))
    let bad_smi = $clean | update compute_apps ($clean.compute_apps | update exit 9)
    $checks = (m_add $checks "failed smi query rejects" ((busy-reasons $bad_smi).error | str contains "query failed"))
    let bad_ps = $clean | update ps ($clean.ps | update exit 1)
    $checks = (m_add $checks "failed ps query rejects" ((busy-reasons $bad_ps).error | str contains "query failed"))
    let malformed_smi = $clean | update compute_apps ($clean.compute_apps | update stdout "unexpected\n")
    $checks = (m_add $checks "malformed smi query rejects" ((busy-reasons $malformed_smi).error | str contains "unrecognized"))
    let hold_ps = $clean | update ps ($clean.ps | update stdout ($clean.ps.stdout + "44 1 S perl perl /tmp/hold.pl\n"))
    $checks = (m_add $checks "resident hold process blocks" (((busy-reasons $hold_ps).busy | length) == 1))
    let self_ps = $clean | update ps ($clean.ps | update stdout ($clean.ps.stdout + "45 1 S perl perl kern/cta_remap_matrix.pl --harness ./attnpartn_regress\n"))
    $checks = (m_add $checks "runner harness argument does not block itself" (((busy-reasons $self_ps).busy | length) == 0))
    let busy = $clean | update compute_apps ($clean.compute_apps | update stdout "GPU-aaaa, 123, flashcore, 100\n")
    print -n --stderr "Waiting for idle GPU host: compute app: GPU-aaaa, 123, flashcore, 100\n"
    let waited = wait-idle [$busy $clean] {} 1.0 0.001
    $checks = (m_add $checks "busy host waits then releases" ($waited.status == "IDLE" and $waited.attempts == 2))
    print -n --stderr "Waiting for idle GPU host: compute app: GPU-aaaa, 123, flashcore, 100\n"
    let blocked = wait-idle [$busy] {} 0.0 0.001
    $checks = (m_add $checks "busy host ends BLOCKED_BUSY" ($blocked.status == "BLOCKED_BUSY"))

    let target = "GPU-00000000-0000-0000-0000-000000000007"
    let other = "GPU-00000000-0000-0000-0000-000000000000"
    let policy = { gpu: $target, idle_resident_queue: "http://127.0.0.1:18090/queue" }
    let residents = {
        compute_apps: { exit: 0, stdout: $"($other), 44, ./hold, 7000\n($other), 45, ./flashcore, 200\n($other), 46, ./fence, 100\n" },
        ps: { exit: 0, stdout: ("1 0 S init /sbin/init\n" + "44 1 S hold ./hold weights 0\n" + "45 1 S flashcore ./flashcore ple library manifest\n" + "46 1 S fence ./fence 0\n" + "47 1 S flashd ./flashd weights /tmp/flash.sock 127.0.0.1:18090\n") },
        gpu: { exit: 0, stdout: $"uuid, name\n($target), Tesla M60\n" },
        all_gpu_load: { exit: 0, stdout: $"($target), 0, 0\n($other), 0, 0\n" },
        queue: { exit: 0, stdout: "{\"running\":0,\"waiting\":0,\"max\":8}" }
    }
    let allowed = busy-reasons $residents $policy
    $checks = (m_add $checks "explicit idle off-device residents accepted" ($allowed.error == "" and ($allowed.busy | length) == 0 and ($allowed.allowed_idle_residents | length) == 3))
    $checks = (m_add $checks "default policy still blocks resident contexts" (((busy-reasons $residents).busy | length) > 0))
    let target_app = $residents | update compute_apps ($residents.compute_apps | update stdout ($residents.compute_apps.stdout + $"($target), 45, ./flashcore, 200\n"))
    $checks = (m_add $checks "target compute context still blocks" (((busy-reasons $target_app $policy).busy | length) > 0))
    let unknown_app = $residents | update compute_apps ($residents.compute_apps | update stdout ($residents.compute_apps.stdout + $"($other), 48, ./gemv_test, 100\n"))
    $checks = (m_add $checks "unrecognized off-device compute worker blocks" (((busy-reasons $unknown_app $policy).busy | length) > 0))
    let other_compute = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout ($residents.all_gpu_load.stdout | str replace $"($other), 0, 0" $"($other), 1, 0"))
    $checks = (m_add $checks "off-device compute activity blocks" (((busy-reasons $other_compute $policy).busy | length) > 0))
    let other_memory = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout ($residents.all_gpu_load.stdout | str replace $"($other), 0, 0" $"($other), 0, 1"))
    $checks = (m_add $checks "off-device memory activity blocks" (((busy-reasons $other_memory $policy).busy | length) > 0))
    for field in ["running" "waiting"] {
        let queue = $residents | update queue ($residents.queue | update stdout ($residents.queue.stdout | str replace $"\"($field)\":0" $"\"($field)\":1"))
        $checks = (m_add $checks $"queue ($field) blocks" (((busy-reasons $queue $policy).busy | length) > 0))
    }
    for field in ["queue" "all_gpu_load"] {
        let failed = $residents | update $field (($residents | get $field) | update exit 1)
        $checks = (m_add $checks $"failed ($field) query rejects" ((busy-reasons $failed $policy).error | str contains "query failed"))
    }
    let bad_queue = $residents | update queue ($residents.queue | update stdout "not JSON")
    $checks = (m_add $checks "malformed queue rejects" ((busy-reasons $bad_queue $policy).error | str contains "invalid queue JSON"))
    let fractional_queue = $residents | update queue ($residents.queue | update stdout "{\"running\":0.5,\"waiting\":0}")
    $checks = (m_add $checks "fractional queue count rejects" ((busy-reasons $fractional_queue $policy).error | str contains "invalid queue running"))
    let unknown_util = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout ($residents.all_gpu_load.stdout | str replace ", 0, 0" ", N/A, 0"))
    $checks = (m_add $checks "unknown utilization rejects" ((busy-reasons $unknown_util $policy).error | str contains "invalid all-GPU"))
    let missing_target = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout $"($other), 0, 0\n")
    $checks = (m_add $checks "missing target utilization rejects" ((busy-reasons $missing_target $policy).error | str contains "target GPU missing"))
    let duplicate_target = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout ($residents.all_gpu_load.stdout + $"($target), 0, 0\n"))
    $checks = (m_add $checks "duplicate utilization rejects" ((busy-reasons $duplicate_target $policy).error | str contains "duplicate GPU"))
    let unknown_gpu = $residents | update compute_apps ($residents.compute_apps | update stdout ($residents.compute_apps.stdout + "GPU-abcd, 44, ./hold, 100\n"))
    $checks = (m_add $checks "unknown compute-app GPU rejects" ((busy-reasons $unknown_gpu $policy).error | str contains "compute-app GPU missing"))
    let identity_bad = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout | str replace "44 1 S hold" "44 1 S evil"))
    $checks = (m_add $checks "resident identity mismatch rejects" ((busy-reasons $identity_bad $policy).error | str contains "identity mismatch"))
    let resident_gone = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout | str replace "44 1 S hold ./hold weights 0\n" ""))
    $checks = (m_add $checks "resident disappearing from ps rejects" ((busy-reasons $resident_gone $policy).error | str contains "missing from ps"))
    let other_port = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout | str replace ":18090" ":18091"))
    $checks = (m_add $checks "different daemon queue still blocks" (((busy-reasons $other_port $policy).busy | length) > 0))
    let model_worker = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout + "48 1 S model_test ./model_test weights\n"))
    $checks = (m_add $checks "worker without context still blocks" (((busy-reasons $model_worker $policy).busy | length) > 0))
    for flag in ["--n-gpu-layers 0" "-ngl 0" "--gpu-layers=0"] {
        let sample = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout + $"48 1 S llama-server /tmp/llama-server --model weights ($flag)\n"))
        let output = busy-reasons $sample $policy
        $checks = (m_add $checks $"CPU-configured llama accepted: ($flag)" (($output.busy | length) == 0 and ($output.cpu_configured_workers | length) == 1))
    }
    for flags in ["--n-gpu-layers 1" "--n-gpu-layers 0 -ngl 5" "--n-gpu-layers auto" "--model weights"] {
        let sample = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout + $"48 1 S llama-server /tmp/llama-server ($flags)\n"))
        $checks = (m_add $checks $"ambiguous/GPU llama still blocks: ($flags)" (((busy-reasons $sample $policy).busy | length) > 0))
    }
    let cpu_context = $residents | update ps ($residents.ps | update stdout ($residents.ps.stdout + "48 1 S llama-server /tmp/llama-server -ngl 0\n"))
        | update compute_apps ($residents.compute_apps | update stdout ($residents.compute_apps.stdout + $"($other), 48, /tmp/llama-server, 100\n"))
    let cpu_context_result = busy-reasons $cpu_context $policy
    $checks = (m_add $checks "CPU flag cannot hide a reported GPU context" (($cpu_context_result.busy | length) > 0 and ($cpu_context_result.cpu_configured_workers | length) == 0))
    let strict_cpu = $clean | update ps ($clean.ps | update stdout ($clean.ps.stdout + "48 1 S llama-server /tmp/llama-server -ngl 0\n"))
    $checks = (m_add $checks "strict policy still blocks CPU-configured llama" (((busy-reasons $strict_cpu).busy | length) > 0))

    let recent = $residents | update all_gpu_load ($residents.all_gpu_load | update stdout ($residents.all_gpu_load.stdout | str replace $"($target), 0, 0" $"($target), 70, 30"))
    $checks = (m_add $checks "initial target utilization still blocks" (((busy-reasons $recent $policy).busy | length) > 0))
    let settled = post-idle [$recent $residents] $policy 1.0
    $checks = (m_add $checks "owned target utilization settles without false rejection" ($settled.status == "IDLE" and ($settled.snapshots | length) == 2 and ($settled.snapshots | get 0 | get busy_reasons | length) > 0 and ($settled.snapshots | get 1 | get busy_reasons | length) == 0))
    $checks = (m_add $checks "target utilization settling is bounded" ((post-idle [$recent] $policy 0.0).status == "TIMEOUT"))
    let target_busy_after = $recent | update compute_apps ($recent.compute_apps | update stdout ($recent.compute_apps.stdout + $"($target), 45, ./flashcore, 200\n"))
    $checks = (m_add $checks "target context still rejects during settling" ((post-idle [$target_busy_after] $policy 1.0).status == "CONTENTION"))
    $checks = (m_add $checks "off-device activity still rejects during settling" ((post-idle [$other_compute] $policy 1.0).status == "CONTENTION"))
    let queue_after = $recent | update queue ($recent.queue | update stdout "{\"running\":1,\"waiting\":0}")
    $checks = (m_add $checks "new request rejects while target settles" ((post-idle [$recent $queue_after] $policy 1.0).status == "CONTENTION"))
    let failed_after = $recent | update all_gpu_load ($recent.all_gpu_load | update exit 1)
    $checks = (m_add $checks "failed settling query fails closed" ((post-idle [$failed_after] $policy 1.0).status == "ERROR"))

    let lock_root = if not ($env.PORT128_MATRIX_LOCK_ROOT? | is-empty) {
        $env.PORT128_MATRIX_LOCK_ROOT
    } else {
        ((m_capture "mktemp" ["-d" "--tmpdir" "flash-sass-lock-test-XXXXXX"]).stdout | str trim)
    }
    mkdir $lock_root | ignore
    let held_path = (m_path $lock_root "test-0.lock")
    let held = (job spawn { run-external "flock" "-n" $held_path "nu" "--no-config-file" "-c" "sleep 0.2sec" | complete | ignore })
    sleep 0.05sec
    print -n --stderr ("Queued for shared GPU lock: " + $held_path + " (timeout 0.05s)\n")
    let denied = (m_capture "flock" ["-n" "-E" "75" $held_path "true"])
    $checks = (m_add $checks "queued lock times out without stealing" ($denied.exit_code == 75))
    while (m_process_exists $held) { sleep 0.02sec }
    let release_path = (m_path $lock_root "test-1.lock")
    let holder = (job spawn { run-external "flock" "-n" $release_path "nu" "--no-config-file" "-c" "sleep 0.1sec" | complete | ignore })
    sleep 0.02sec
    print -n --stderr ("Queued for shared GPU lock: " + $release_path + " (timeout 2s)\n")
    let acquired = (m_capture "flock" ["-w" "2" $release_path "true"])
    $checks = (m_add $checks "queued lock acquires after release" ($acquired.exit_code == 0))
    while (m_process_exists $holder) { sleep 0.02sec }
    let info = "lock:\t1: FLOCK ADVISORY WRITE 1900 00:01:1234 0 EOF\n"
    $checks = (m_add $checks "inherited reservation is shared without nested locking" ((inherited-host-lock $held_path $held_path $info true).ok))
    $checks = (m_add $checks "different inherited file rejects" ((inherited-host-lock $held_path $release_path $info true).error | str contains "different file"))
    $checks = (m_add $checks "unlocked inherited descriptor rejects" ((inherited-host-lock $held_path $held_path "" true).error | str contains "no exclusive flock"))
    $checks = (m_add $checks "independent descriptor cannot impersonate caller lock" ((inherited-host-lock $held_path $held_path $info false).error | str contains "BLOCKED_BUSY"))

    let preflight = { exit: 0, stdout: "{\"self_test\":\"pass\",\"cpu_only\":true,\"negative_canary_cases\":16,\"negative_trace_cases\":14,\"mapping_cases\":64}" }
    $checks = (m_add $checks "eager CPU-only preflight accepted" ((validate-preflight $preflight).ok))
    $checks = (m_add $checks "driver ABI loader failure rejects before lock" ((validate-preflight ($preflight | update exit 127)).error | str contains "driver-ABI self-test failed"))
    $checks = (m_add $checks "invalid eager preflight JSON rejects" ((validate-preflight ($preflight | update stdout "")).error | str contains "incomplete CPU-only"))
    $checks = (m_add $checks "non-CPU-only preflight rejects" ((validate-preflight ($preflight | update stdout ($preflight.stdout | str replace "true" "false"))).error | str contains "incomplete CPU-only"))
    let valid = {
        bit_exact: true, boundary_canaries: true, boundary_canary_words: 64,
        partial_values: 49536, final_values: 6144, steps_per_sample: 64,
        old: { event: { median_us: 1 } }, new: { event: { median_us: 2 } },
        trace_verified: false, trace_launches: 0
    }
    $checks = (m_add $checks "untraced complete result accepted" ((validate-result $valid false).ok))
    $checks = (m_add $checks "missing canaries rejects" ((validate-result ($valid | update boundary_canaries false) false).error | str contains "boundary canaries"))
    $checks = (m_add $checks "missing requested trace rejects" ((validate-result $valid true).error | str contains "trace requested"))
    let traced = $valid | update trace_verified true | update trace_launches 1
    $checks = (m_add $checks "unexpected traced timing rejects" ((validate-result $traced false).error | str contains "untraced"))
    $checks = (m_add $checks "verified traced result accepted" ((validate-result $traced true).ok))

    let child = (m_run "nu" ["--no-config-file" "-c" "print -n 'out'; print -n --stderr 'err'; exit 7"] 2.0)
    $checks = (m_add $checks "child stdout and failure retained" ($child.stdout == "out" and $child.stderr == "err" and $child.exit == 7))
    let timeout = (m_run "nu" ["--no-config-file" "-c" "sleep 5sec"] 0.05)
    $checks = (m_add $checks "only own child is timed out" ($timeout.exit == 124 and ($timeout.stderr | str contains "RUNNER TIMEOUT")))

    let expected_names = [
        "idle inventory accepted" "failed smi query rejects" "failed ps query rejects" "malformed smi query rejects"
        "resident hold process blocks" "runner harness argument does not block itself" "busy host waits then releases"
        "busy host ends BLOCKED_BUSY" "explicit idle off-device residents accepted" "default policy still blocks resident contexts"
        "target compute context still blocks" "unrecognized off-device compute worker blocks" "off-device compute activity blocks"
        "off-device memory activity blocks" "queue running blocks" "queue waiting blocks" "failed queue query rejects"
        "failed all_gpu_load query rejects" "malformed queue rejects" "fractional queue count rejects" "unknown utilization rejects"
        "missing target utilization rejects" "duplicate utilization rejects" "unknown compute-app GPU rejects"
        "resident identity mismatch rejects" "resident disappearing from ps rejects" "different daemon queue still blocks"
        "worker without context still blocks"
        "CPU-configured llama accepted: --n-gpu-layers 0" "CPU-configured llama accepted: -ngl 0" "CPU-configured llama accepted: --gpu-layers=0"
        "ambiguous/GPU llama still blocks: --n-gpu-layers 1" "ambiguous/GPU llama still blocks: --n-gpu-layers 0 -ngl 5"
        "ambiguous/GPU llama still blocks: --n-gpu-layers auto" "ambiguous/GPU llama still blocks: --model weights"
        "CPU flag cannot hide a reported GPU context" "strict policy still blocks CPU-configured llama"
        "initial target utilization still blocks" "owned target utilization settles without false rejection"
        "target utilization settling is bounded" "target context still rejects during settling"
        "off-device activity still rejects during settling" "new request rejects while target settles"
        "failed settling query fails closed" "queued lock times out without stealing" "queued lock acquires after release"
        "inherited reservation is shared without nested locking" "different inherited file rejects"
        "unlocked inherited descriptor rejects" "independent descriptor cannot impersonate caller lock"
        "eager CPU-only preflight accepted" "driver ABI loader failure rejects before lock"
        "invalid eager preflight JSON rejects" "non-CPU-only preflight rejects" "untraced complete result accepted"
        "missing canaries rejects" "missing requested trace rejects" "unexpected traced timing rejects"
        "verified traced result accepted" "child stdout and failure retained" "only own child is timed out"
    ]
    if ($checks | length) != 61 { m_fail "self-test bookkeeping did not produce 61 checks\n" }
    for index in 0..60 {
        let check = $checks | get $index
        if not $check.passed { m_fail $"self-test ($check.name): failed\n" }
        if $check.name != ($expected_names | get $index) { m_fail "self-test output order mismatch\n" }
        print $"ok ($index + 1) - ($check.name)"
    }
    run-external "rm" "-rf" $lock_root | complete | ignore
    print (m_json { gpu_executed: false, status: "PASS", tests: 61 })
}

def m_indent [depth: int] {
    if $depth == 0 { "" } else { (1..($depth * 3) | each { " " } | str join "") }
}

def m_pretty [value, depth: int = 0] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        let keys = ($value | columns | sort)
        if ($keys | length) == 0 { return "{}" }
        mut entries = []
        for key in $keys {
            $entries = ($entries | append $"(m_indent ($depth + 1))($key | to json) : (m_pretty ($value | get $key) ($depth + 1))")
        }
        "{\n" + ($entries | str join ",\n") + $"\n(m_indent $depth)}"
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        let values = ($value | each { |item| m_pretty $item ($depth + 1) })
        if ($values | length) == 0 { return "[]" }
        let entries = ($values | each { |item| (m_indent ($depth + 1)) + $item })
        "[\n" + ($entries | str join ",\n") + $"\n(m_indent $depth)]"
    } else if $kind == "float" {
        m_float $value
    } else {
        $value | to json
    }
}

def m_put [path: string, value] {
    ((m_pretty $value) + "\n") | save --force --raw $path
}

def m_result [argv: list, environment: record = {}, timeout: float = 60.0] {
    let command = $argv | get 0
    let arguments = $argv | skip 1
    let timeout_args = ["--signal=KILL" $"($timeout)s" $command] ++ $arguments
    let run = if ($environment | columns | length) == 0 {
        m_capture "timeout" $timeout_args
    } else {
        with-env $environment { m_capture "timeout" $timeout_args }
    }
    let stderr = if $run.exit_code == 124 { $run.stderr + "\nRUNNER TIMEOUT\n" } else { $run.stderr }
    { argv: $argv, exit: $run.exit_code, stdout: $run.stdout, stderr: $stderr }
}

def m_absolute [path: string] {
    let expanded = ($path | path expand)
    if not ($expanded | path exists) { m_fail $"resolve ($path): no such file or directory\n" }
    $expanded
}

def m_image [path: string] {
    let absolute = m_absolute $path
    let hash = (m_capture "sha256sum" [$absolute])
    if $hash.exit_code != 0 { m_fail $"open ($path): no such file or directory\n" }
    let size = (m_capture "stat" ["-c" "%s" $absolute])
    if $size.exit_code != 0 { m_fail $"stat ($absolute): no such file or directory\n" }
    {
        path: $absolute,
        sha256: ($hash.stdout | split row " " | get 0),
        bytes: (($size.stdout | str trim) | into int)
    }
}

def m_gpu [uuid: string] {
    m_result ["nvidia-smi" "-i" $uuid "--query-gpu=uuid,name,driver_version,pci.bus_id,temperature.gpu,clocks.current.sm,clocks.current.memory,utilization.gpu,memory.used,ecc.mode.current,pcie.link.gen.current,pcie.link.width.current" "--format=csv"] {} 30.0
}

def m_snapshot [uuid: string, args: record] {
    let apps = (m_result ["nvidia-smi" "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory" "--format=csv,noheader,nounits"] {} 30.0)
    let ps = (m_result ["ps" "-eo" "pid=,ppid=,stat=,comm=,args="] {} 30.0)
    let gpu = (m_gpu $uuid)
    mut snapshot = { utc: ((m_capture "date" ["-u" "+%FT%TZ"]).stdout | str trim), compute_apps: $apps, ps: $ps, gpu: $gpu }
    if (($args | columns) | any { |key| $key == "idle_resident_queue" }) and $args.idle_resident_queue != null {
        $snapshot = ($snapshot | insert all_gpu_load (m_result ["nvidia-smi" "--query-gpu=uuid,utilization.gpu,utilization.memory" "--format=csv,noheader,nounits"] {} 30.0))
        $snapshot = ($snapshot | insert queue (m_result ["curl" "--fail" "--silent" "--show-error" "--noproxy" "*" "--max-time" "10" $args.idle_resident_queue] {} 15.0))
    }
    $snapshot
}

def m_redact [snapshot: record] {
    if ($snapshot.ps | describe | str starts-with "record") and $snapshot.ps.stdout != "" {
        let lines = ($snapshot.ps.stdout | lines | each {
            |line|
            let row = ($line | parse --regex '^(\s*\d+\s+\d+\s+\S+\s+\S+).*$')
            if ($row | length) == 1 { ($row | get 0 | get capture0) + " arguments inspected but not retained" } else { $line }
        })
        ($snapshot | update ps ($snapshot.ps | update stdout (($lines | str join "\n") + "\n") | upsert retained_fields "pid ppid stat comm; arguments inspected but not retained"))
    } else {
        $snapshot
    }
}

def m_now [] {
    ((m_capture "date" ["+%s"]).stdout | str trim | into int)
}

def m_wait_idle [args: record, out: string, case: string] {
    let start = m_now
    mut attempt = 0
    loop {
        let snapshot = m_snapshot $args.gpu $args
        $attempt = $attempt + 1
        let with_identity = $snapshot | insert case $case | insert attempt $attempt
        let validation = busy-reasons $with_identity $args
        if $validation.error != "" { return { status: "ERROR", error: $validation.error, attempts: $attempt } }
        let redacted = m_redact ($with_identity | insert busy_reasons $validation.busy)
        if not ($out | is-empty) { m_put (m_path $out $"gate-($case)-($attempt).json") $redacted }
        if ($validation.busy | length) == 0 { return $redacted }
        print -n --stderr ("Waiting for idle GPU host: " + ($validation.busy | str join "; ") + "\n")
        let remaining = $args.wait_seconds - ((m_now) - $start)
        if $remaining <= 0 { return { status: "BLOCKED_BUSY", error: "BLOCKED_BUSY: host still has other GPU workers\n", attempts: $attempt } }
        let delay = ($remaining | math min $args.poll_seconds) | into string
        sleep ($delay + "sec" | into duration)
    }
}

def m_post_idle [args: record, out: string, case: string] {
    let start = m_now
    mut samples = []
    loop {
        let snapshot = m_snapshot $args.gpu $args
        let normal = busy-reasons $snapshot $args
        if $normal.error != "" { return { status: "ERROR", error: $normal.error, snapshots: $samples } }
        let foreign = busy-reasons $snapshot $args true
        if $foreign.error != "" { return { status: "ERROR", error: $foreign.error, snapshots: $samples } }
        let sample = m_redact ($snapshot | insert case $case | insert post_run_sample (($samples | length) + 1) | insert busy_reasons $normal.busy)
        $samples = ($samples | append $sample)
        if not ($out | is-empty) { m_put (m_path $out $"gate-($case)-after-($samples | length).json") $sample }
        if ($foreign.busy | length) > 0 { return { status: "CONTENTION", error: ("post-run contention: " + ($foreign.busy | str join "; ") + "\n"), snapshots: $samples } }
        if ($normal.busy | length) == 0 { return { status: "IDLE", settle_limit_seconds: $args.settle_seconds, snapshots: $samples } }
        let remaining = $args.settle_seconds - ((m_now) - $start)
        if $remaining <= 0 { return { status: "TIMEOUT", error: $"post-run target utilization did not settle within ($args.settle_seconds)s\n", snapshots: $samples } }
        let delay = ($remaining | math min 0.25) | into string
        sleep ($delay + "sec" | into duration)
    }
}

def m_median [values: list] {
    let sorted = ($values | sort)
    let count = ($sorted | length)
    if $count == 0 { m_fail "no values for median\n" }
    let middle = ($count / 2) | into int
    if ($count mod 2) == 1 { $sorted | get $middle } else { (($sorted | get ($middle - 1)) + ($sorted | get $middle)) / 2 }
}

def m_prepare [args: record, out: string] {
    let script = (m_path $env.FILE_PWD "cta_remap_matrix.nu")
    let source_image = m_image $script
    let helper_image = m_image (m_path $env.FILE_PWD "cta_remap_matrix.nu")
    let env_run = (m_capture "env" [])
    let ptxas = if ($env.PTXAS11? | is-empty) { "/opt/cuda-11.4/bin/ptxas" } else { $env.PTXAS11 }
    let compiler = if ($env.CC? | is-empty) { "cc" } else { $env.CC }
    let cuda = if ($env.CUDA? | is-empty) { "/opt/cuda" } else { $env.CUDA }
    mut environment = { CUDA_DEVICE_ORDER: "PCI_BUS_ID", CUDA_VISIBLE_DEVICES: $args.gpu }
    mut trace_environment = {}
    for line in ($env_run.stdout | lines) {
        let pieces = ($line | split row "=")
        let name = $pieces | get 0
        if $name =~ '^(CUDA_|ATTN_|DIE_|FLASH_GPU_|LD_LIBRARY_PATH$|LD_PRELOAD$|LD_BIND_NOW$)' {
            let value = ($pieces | skip 1 | str join "=")
            $trace_environment = ($trace_environment | insert $name $value)
        }
    }
    $trace_environment = ($trace_environment | upsert CUDA_DEVICE_ORDER "PCI_BUS_ID" | upsert CUDA_VISIBLE_DEVICES $args.gpu)
    if (($trace_environment | columns) | any { |key| $key == "ATTN_TRACE_CTAS" }) {
        $trace_environment = ($trace_environment | reject ATTN_TRACE_CTAS)
    }
    if $args.trace {
        $trace_environment = ($trace_environment | upsert ATTN_TRACE_CHECK "1")
    } else if (($trace_environment | columns) | any { |key| $key == "ATTN_TRACE_CHECK" }) {
        $trace_environment = ($trace_environment | reject ATTN_TRACE_CHECK)
    }
    $environment = ($environment | upsert ATTN_TRACE_CTAS null)
    if $args.trace {
        $environment = ($environment | upsert ATTN_TRACE_CHECK "1")
    } else {
        $environment = ($environment | upsert ATTN_TRACE_CHECK null)
    }
    mut manifest = {
        arguments: $args,
        utc: ((m_capture "date" ["-u" "+%FT%TZ"]).stdout | str trim),
        runner: $source_image,
        helper: $helper_image,
        environment: $trace_environment,
        trace_requested: $args.trace,
        measurement_mode: (if $args.check_only { "check-only" } else if $args.trace { "physical-cta-trace" } else { "untraced-performance" }),
        locking: { host: $args.host_lock, gpu_dir: $args.lock_dir, inherited_host_fd: $args.host_lock_fd },
        resident_policy: (if not ($args.idle_resident_queue | is-empty) {
            "Explicit off-device idle residents: target has no reported compute app; all-device utilization and loopback queue counts must be zero before each case and after bounded settling of the owned target. Immediate post-run telemetry is retained; target contexts, other-device activity, and queued requests always reject. This does not authorize whole-model core reuse."
        } else {
            "Strict: any compute context or recognized GPU worker on the host blocks."
        })
    }
    if not $args.check_only {
        $manifest = ($manifest | insert old_cubin (m_image $args.old))
        $manifest = ($manifest | insert new_cubin (m_image $args.new))
        $manifest = ($manifest | insert old_library (m_image $args.old_library))
        $manifest = ($manifest | insert new_library (m_image $args.new_library))
        $manifest = ($manifest | insert harness (m_image $args.harness))
        $manifest = ($manifest | insert compiler (m_result [$compiler "--version"] {} 30.0))
        $manifest = ($manifest | insert ptxas (m_result [$ptxas "--version"] {} 30.0))
        $manifest = ($manifest | insert timing "64 positions per persistent launch, four warmup pairs, balanced alternating old/new order; one-phase partial and two-phase partial+merge programs; GPU events exclude allocation/upload and CPU checks")
        $manifest = ($manifest | insert input "finite fixed-seed FP32 values; seed0 has zero query; padding finite zero")
    }
    m_put (m_path $out "manifest.json") $manifest

    if not $args.check_only {
        let preflight_env = $environment | upsert LD_BIND_NOW "1"
        let preflight_run = m_result [(m_absolute $args.harness) "--self-test"] $preflight_env 30.0
        let preflight = ($preflight_run | insert LD_BIND_NOW 1)
        m_put (m_path $out "driver-abi-preflight.json") $preflight
        let valid = validate-preflight $preflight
        if not $valid.ok {
            m_put (m_path $out "status.json") { status: "FAIL", completed: 0, gpu_launches: 0, error: $valid.error, gpu_executed: false }
            m_fail $valid.error
        }
    }
    m_put (m_path $out "status.json") { status: "WAITING", cases: 0, gpu_executed: false }
    $environment
}

def m_execute [args: record, out: string, environment: record] {
    if $args.check_only {
        let idle = m_wait_idle $args $out "check-only"
        if $idle.status != "IDLE" {
            m_put (m_path $out "status.json") { status: (if $idle.status == "BLOCKED_BUSY" { "BLOCKED_BUSY" } else { "FAIL" }), error: $idle.error, check_only: true, gpu_executed: false }
            m_fail $idle.error (if $idle.status == "BLOCKED_BUSY" { 75 } else { 1 })
        }
        m_put (m_path $out "status.json") { status: "IDLE", check_only: true, gpu_executed: false }
        return 0
    }

    let counts = ($args.keys | split row "," | each { |value| $value | into int })
    let seeds = ($args.seeds | split row "," | each { |value| $value | into int })
    mut rows = []
    mut launches = 0
    for n in $counts {
        for seed in $seeds {
            let case = $"keys-($n)-seed-($seed)"
            let before = m_wait_idle $args $out $case
            if $before.status != "IDLE" {
                m_put (m_path $out "status.json") { status: (if $before.status == "BLOCKED_BUSY" { "BLOCKED_BUSY" } else { "FAIL" }), completed: ($rows | length), gpu_launches: $launches, error: $before.error, gpu_executed: ($launches > 0) }
                m_fail $before.error (if $before.status == "BLOCKED_BUSY" { 75 } else { 1 })
            }
            let harness_args = [(m_absolute $args.harness) (m_absolute $args.old) (m_absolute $args.new) ($n | into string) ($seed | into string) ($args.repeats | into string)]
            $launches = $launches + 1
            mut result = m_result $harness_args $environment 180.0
            let after_gpu = m_gpu $args.gpu
            $result = ($result | insert gpu_before $before | insert gpu_after $after_gpu)
            mut after_error = ""
            if not ($args.idle_resident_queue | is-empty) {
                let post = m_post_idle $args $out $case
                if $post.status != "IDLE" { $after_error = $post.error } else { $result = ($result | insert host_after $post) }
                if $after_error != "" { $result = ($result | insert post_run_error $after_error) }
            }
            m_put (m_path $out $"($case).json") $result
            if $result.exit != 0 {
                m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: $"GPU comparison failed: n=($n), seed=($seed), exit=($result.exit)\n", gpu_executed: true }
                m_fail $"GPU comparison failed: n=($n), seed=($seed), exit=($result.exit)\n"
            }
            if $after_error != "" {
                m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: $after_error, gpu_executed: true }
                m_fail $after_error
            }
            if $after_gpu.exit != 0 { m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: "post-run GPU metadata query failed\n", gpu_executed: true }; m_fail "post-run GPU metadata query failed\n" }
            let parsed = try { $result.stdout | from json } catch { null }
            if $parsed == null { m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: "harness result is not JSON\n", gpu_executed: true }; m_fail "harness result is not JSON\n" }
            let valid = validate-result $parsed $args.trace
            if not $valid.ok { m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: $valid.error, gpu_executed: true }; m_fail $valid.error }
            if $parsed.keys != $n or $parsed.seed != $seed { m_put (m_path $out "status.json") { status: "FAIL", completed: ($rows | length), gpu_launches: $launches, error: "harness reported wrong key count or seed\n", gpu_executed: true }; m_fail "harness reported wrong key count or seed\n" }
            $rows = ($rows | append $parsed)
            let progress = {
                keys: $n,
                seed: $seed,
                exact: true,
                boundary_canaries: true,
                trace_verified: $parsed.trace_verified,
                old_us: $parsed.old.event.median_us,
                new_us: $parsed.new.event.median_us
            }
            print (m_json $progress)
        }
    }
    mut summary = []
    for n in $counts {
        let samples = ($rows | where { |row| $row.keys == $n })
        let old = m_median ($samples | each { |row| $row.old.event.median_us })
        let new = m_median ($samples | each { |row| $row.new.event.median_us })
        $summary = ($summary | append { keys: $n, seeds: ($samples | length), old_partial_merge_us: $old, new_partial_merge_us: $new, delta_us: ($new - $old), speedup: ($old / $new) })
    }
    m_put (m_path $out "summary.json") $summary
    m_put (m_path $out "status.json") { status: "PASS", cases: ($rows | length), gpu_launches: $launches, trace_requested: $args.trace }
    0
}

def m_forward_args [args: record] {
    mut words = []
    for item in [
        ["--gpu" $args.gpu] ["--out" $args.out]
        ["--old-source" $args.old_source] ["--new-source" $args.new_source]
        ["--harness" $args.harness] ["--keys" $args.keys] ["--seeds" $args.seeds]
        ["--repeats" ($args.repeats | into string)] ["--instrumentation" $args.instrumentation]
        ["--lock-dir" $args.lock_dir] ["--host-lock" $args.host_lock]
        ["--settle-seconds" ($args.settle_seconds | into string)]
        ["--wait-seconds" ($args.wait_seconds | into string)]
        ["--poll-seconds" ($args.poll_seconds | into string)]
    ] {
        $words = ($words ++ $item)
    }
    for option in [
        ["--old" $args.old] ["--new" $args.new]
        ["--old-library" $args.old_library] ["--new-library" $args.new_library]
        ["--harness-source" $args.harness_source] ["--idle-resident-queue" $args.idle_resident_queue]
    ] {
        if not (($option | get 1) | is-empty) { $words = ($words ++ $option) }
    }
    if not ($args.host_lock_fd | is-empty) { $words = ($words ++ ["--host-lock-fd" ($args.host_lock_fd | into string)]) }
    if $args.trace { $words = ($words | append "--trace") }
    if $args.check_only { $words = ($words | append "--check-only") }
    $words
}

def m_verify_lock_file [path: string] {
    for candidate in (glob $"/proc/($nu.pid)/fd/*") {
        let link = (m_capture "readlink" [$candidate])
        if $link.exit_code != 0 or ($link.stdout | str trim) != $path { continue }
        let fd = $candidate | path basename
        let info_path = $"/proc/($nu.pid)/fdinfo/($fd)"
        let info = try { open --raw $info_path } catch { "" }
        if not ($info =~ '(?m)^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s') { continue }
        let held = (m_capture "flock" ["-n" $fd])
        if $held.exit_code == 0 { return true }
    }
    false
}

def m_environment [args: record] {
    let result = {
        CUDA_DEVICE_ORDER: "PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES: $args.gpu,
        ATTN_TRACE_CTAS: null,
        ATTN_TRACE_CHECK: (if $args.trace { "1" } else { null })
    }
    $result
}

def main [
    --old: string,
    --new: string,
    --old-library: string,
    --new-library: string,
    --old-source: string = "c6d3de8180d8b87caf0d2f61fa52edc9247daded",
    --new-source: string = "66725432f86cc193b75cd51230216aed44c60d3f",
    --harness: string = "./attnpartn_regress",
    --harness-source: string,
    --gpu: string,
    --out: string,
    --keys: string = "1,1023,1024,1025,2047,2048,2049,2050,2051,2052,4095,4096,4097,8191,8192,8193",
    --seeds: string = "0,77,4242",
    --repeats: int = 32,
    --instrumentation: string = "none",
    --trace,
    --notrace,
    --lock-dir: string,
    --host-lock: string,
    --idle-resident-queue: string,
    --settle-seconds: float = 5.0,
    --host-lock-fd: int,
    --wait-seconds: float = 60.0,
    --poll-seconds: float = 5.0,
    --self-test,
    --check-only,
    --locked,
    --locked-gpu,
    ...positional: string
] {
    if $self_test { self-test; exit 0 }
    if ($positional | length) > 0 { m_fail "unexpected positional arguments\n" }
    let trace = $trace and not $notrace
    let lock_dir = if ($lock_dir | is-empty) { if ($env.FLASH_GPU_LOCK_DIR? | is-empty) { "/tmp/flash-sass-gpu-locks" } else { $env.FLASH_GPU_LOCK_DIR } } else { $lock_dir }
    let host_lock = if ($host_lock | is-empty) { if ($env.FLASH_GPU_HOST_LOCK? | is-empty) { "/tmp/flash-sass-engine-validation.lock" } else { $env.FLASH_GPU_HOST_LOCK } } else { $host_lock }
    if ($gpu | is-empty) { m_fail "--gpu is required\n" }
    if ($out | is-empty) { m_fail "--out is required\n" }
    if not $check_only {
        for item in [$old $new $old_library $new_library $harness_source] {
            if ($item | is-empty) { m_fail "--old, --new, --old-library, --new-library, and --harness-source are required\n" }
        }
        for source in [$old_source $new_source $harness_source] {
            if not ($source =~ '^[0-9a-f]{40}$') { m_fail "source identities must be full 40-character git commit IDs\n" }
        }
    }
    if not ($gpu =~ '^(?i:GPU-[0-9a-f-]+|[0-9]+)$') { m_fail "--gpu must be a GPU UUID or numeric index\n" }
    if not ($idle_resident_queue | is-empty) {
        if not ($idle_resident_queue =~ '^http://127\.0\.0\.1:([1-9][0-9]{0,4})/queue$') { m_fail "--idle-resident-queue must name a loopback HTTP queue\n" }
        let port = (($idle_resident_queue | parse --regex '^http://127\.0\.0\.1:(?<port>[1-9][0-9]{0,4})/queue$') | get 0 | get port | into int)
        if $port > 65535 { m_fail "--idle-resident-queue must name a loopback HTTP queue\n" }
        if not ($gpu =~ '(?i)^GPU-') { m_fail "idle resident mode requires a GPU UUID, not an index\n" }
    }
    if $wait_seconds < 0 or $poll_seconds <= 0 or $poll_seconds > 60 { m_fail "wait-seconds >=0 and poll-seconds in (0,60] required\n" }
    if $settle_seconds < 0 or $settle_seconds > 30 { m_fail "settle-seconds in [0,30] required\n" }
    if not ($host_lock_fd | is-empty) and $host_lock_fd < 3 { m_fail "host-lock-fd must be >=3\n" }
    if $trace and $instrumentation == "none" or (not $trace and $instrumentation != "none") { m_fail "--trace requires --instrumentation; untraced timing requires instrumentation=none\n" }
    mut counts = ($keys | split row ",")
    mut seed_values = ($seeds | split row ",")
    for list in [$counts $seed_values] {
        if ($list | length) == 0 { m_fail "keys and seeds must be nonempty\n" }
        for value in $list {
            if not ($value =~ '^[0-9]+$') { m_fail "invalid unsigned integer list\n" }
        }
    }
    $counts = ($counts | each { |value| $value | into int })
    $seed_values = ($seed_values | each { |value| $value | into int })
    if ($counts | uniq | length) != ($counts | length) or ($seed_values | uniq | length) != ($seed_values | length) { m_fail "duplicate key/seed would overwrite evidence\n" }
    if ($counts | any { |value| $value < 1 or $value > 16384 }) { m_fail "keys 1..16384 required\n" }
    if ($seed_values | any { |value| $value > 4294967295 }) { m_fail "32-bit unsigned seed required\n" }
    if $repeats < 4 or $repeats > 10000 or ($repeats mod 2) != 0 { m_fail "even repetition count in 4..10000 required\n" }

    mut args = {
        old: $old,
        new: $new,
        old_library: $old_library,
        new_library: $new_library,
        old_source: $old_source,
        new_source: $new_source,
        harness: $harness,
        harness_source: $harness_source,
        gpu: $gpu,
        out: $out,
        keys: $keys,
        seeds: $seeds,
        repeats: $repeats,
        instrumentation: $instrumentation,
        trace: $trace,
        lock_dir: $lock_dir,
        host_lock: $host_lock,
        idle_resident_queue: $idle_resident_queue,
        settle_seconds: $settle_seconds,
        host_lock_fd: $host_lock_fd,
        wait_seconds: $wait_seconds,
        poll_seconds: $poll_seconds,
        check_only: $check_only
    }
    let output = if $locked or $locked_gpu {
        $out | path expand
    } else {
        let absolute = ($out | path expand)
        if ($absolute | path exists) { m_fail $"output already exists: ($out)\n" 255 }
        let parent = $absolute | path dirname
        run-external "mkdir" "-p" $parent | complete | ignore
        mkdir $absolute
        $absolute
    }
    $args = ($args | update out $output)

    if not $locked and not $locked_gpu {
        m_prepare $args $output
    }

    let script = (m_path $env.FILE_PWD "cta_remap_matrix.nu")
    if $locked and not (m_verify_lock_file $host_lock) {
        m_fail "BLOCKED_BUSY: inherited host-lock verification failed\n" 75
    }
    if $locked_gpu {
        if not (m_verify_lock_file $host_lock) { m_fail "inherited host-lock descriptor has no exclusive flock\n" 1 }
        if not (m_verify_lock_file (m_path ($args.lock_dir | path expand) $"($args.gpu).lock")) { m_fail "BLOCKED_BUSY: GPU reservation verification failed\n" 75 }
        let result = m_execute $args $output (m_environment $args)
        exit $result
    }

    if not $locked {
        if not ($args.host_lock_fd | is-empty) {
            let fd = $args.host_lock_fd | into string
            let actual = (m_capture "readlink" [$"/proc/self/fd/($fd)"])
            if $actual.exit_code != 0 or ($actual.stdout | str trim) != $args.host_lock {
                m_put (m_path $output "status.json") { status: "FAIL", completed: 0, gpu_launches: 0, error: "inherited host-lock descriptor names a different file\n", gpu_executed: false }
                m_fail "inherited host-lock descriptor names a different file\n" 1
            }
            let info = try { open --raw $"/proc/self/fdinfo/($fd)" } catch { "" }
            if not ($info =~ '(?m)^lock:\s+\d+:\s+FLOCK\s+ADVISORY\s+WRITE\s') {
                m_put (m_path $output "status.json") { status: "FAIL", completed: 0, gpu_launches: 0, error: "inherited host-lock descriptor has no exclusive flock\n", gpu_executed: false }
                m_fail "inherited host-lock descriptor has no exclusive flock\n" 1
            }
            let verify = (m_capture "flock" ["-n" $fd])
            if $verify.exit_code != 0 { m_fail "BLOCKED_BUSY: inherited host-lock verification failed\n" 75 }
        } else {
            run-external "mkdir" "-p" ($args.host_lock | path dirname) | complete | ignore
            let probe = (m_capture "flock" ["-n" $args.host_lock "true"])
            if $probe.exit_code != 0 and $args.wait_seconds <= 0 {
                m_put (m_path $output "status.json") { status: "BLOCKED_BUSY", completed: 0, gpu_launches: 0, error: $"BLOCKED_BUSY: shared lock held: ($args.host_lock)\n", gpu_executed: false }
                m_fail $"BLOCKED_BUSY: shared lock held: ($args.host_lock)\n" 75
            }
            if $probe.exit_code != 0 {
                print -n --stderr ("Queued for shared GPU lock: " + $args.host_lock + $" (timeout ($args.wait_seconds)s)\n")
            }
            let mode = if $probe.exit_code == 0 { "-n" } else { "-w" }
            let lock_args = [$mode ($args.wait_seconds | into string) "-E" "75" $args.host_lock "nu" "--no-config-file" $script "--locked"] ++ (m_forward_args $args)
            let command_args = if $mode == "-n" { ["-n" "-E" "75" $args.host_lock "nu" "--no-config-file" $script "--locked"] ++ (m_forward_args $args) } else { $lock_args }
            let host_result = (m_capture "flock" $command_args)
            if $host_result.stdout != "" { print -n $host_result.stdout }
            if $host_result.stderr != "" { print -n --stderr $host_result.stderr }
            if $host_result.exit_code != 0 {
                if $host_result.exit_code == 75 {
                    let current = try { open (m_path $output "status.json") } catch { {} }
                    if ($current.status? | default "WAITING") == "WAITING" {
                        m_put (m_path $output "status.json") { status: "BLOCKED_BUSY", completed: 0, gpu_launches: 0, error: $"BLOCKED_BUSY: shared lock held: ($args.host_lock)\n", gpu_executed: false }
                    }
                }
                exit $host_result.exit_code
            }
            exit 0
        }
    }

    run-external "mkdir" "-p" ($args.lock_dir | path expand) | complete | ignore
    let gpu_lock = (m_path ($args.lock_dir | path expand) $"($args.gpu).lock")
    let gpu_probe = (m_capture "flock" ["-n" $gpu_lock "true"])
    if $gpu_probe.exit_code != 0 and $args.wait_seconds <= 0 {
        m_put (m_path $output "status.json") { status: "BLOCKED_BUSY", completed: 0, gpu_launches: 0, error: $"BLOCKED_BUSY: shared lock held: ($gpu_lock)\n", gpu_executed: false }
        m_fail $"BLOCKED_BUSY: shared lock held: ($gpu_lock)\n" 75
    }
    if $gpu_probe.exit_code != 0 {
        print -n --stderr ("Queued for shared GPU lock: " + $gpu_lock + $" (timeout ($args.wait_seconds)s)\n")
    }
    let mode = if $gpu_probe.exit_code == 0 { "-n" } else { "-w" }
    let base = [$mode ($args.wait_seconds | into string) "-E" "75" $gpu_lock "nu" "--no-config-file" $script "--locked-gpu"]
    let command_args = if $mode == "-n" { ["-n" "-E" "75" $gpu_lock "nu" "--no-config-file" $script "--locked-gpu"] ++ (m_forward_args $args) } else { $base ++ (m_forward_args $args) }
    let gpu_result = (m_capture "flock" $command_args)
    if $gpu_result.stdout != "" { print -n $gpu_result.stdout }
    if $gpu_result.stderr != "" { print -n --stderr $gpu_result.stderr }
    if $gpu_result.exit_code != 0 {
        if $gpu_result.exit_code == 75 {
            let current = try { open (m_path $output "status.json") } catch { {} }
            if ($current.status? | default "WAITING") == "WAITING" {
                m_put (m_path $output "status.json") { status: "BLOCKED_BUSY", completed: 0, gpu_launches: 0, error: $"BLOCKED_BUSY: shared lock held: ($gpu_lock)\n", gpu_executed: false }
            }
        }
        exit $gpu_result.exit_code
    }
    exit 0
}
