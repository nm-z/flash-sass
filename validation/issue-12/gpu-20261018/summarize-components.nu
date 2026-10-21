# Recompute component distributions and physical CTA evidence from retained files.

def fail [message: string] {
    print -n --stderr $"($message)\n"
    exit 255
}

def ensure [condition: bool, message: string] {
    if not $condition { fail $message }
}

def integer [value, label: string] {
    if ($value | describe) == "int" { return $value }
    let text = ($value | into string | str trim)
    if not ($text =~ '^[+-]?[0-9]+$') { fail $"invalid integer ($label)" }
    try { $text | into int } catch { fail $"invalid integer ($label)" }
}

def positive [value] {
    let text = ($value | into string)
    if not ($text =~ '^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$') { fail "invalid positive sample" }
    let number = try { $text | into float } catch { fail "invalid positive sample" }
    ensure ($number > 0 and $number < 1e100) "invalid positive sample"
    $number
}

def stats [values: list] {
    let sorted = ($values | sort)
    let n = ($sorted | length)
    ensure ($n > 0) "empty distribution"
    mut total = 0.0
    for value in $sorted { $total = $total + $value }
    let middle = (($n / 2) | into int)
    let median = if ($n mod 2) == 1 { $sorted | get $middle } else { (($sorted | get ($middle - 1)) + ($sorted | get $middle)) / 2.0 }
    let p95_index = (((($n * 0.95) | math ceil) | into int) - 1)
    {
        n: $n,
        min: ($sorted | get 0),
        max: ($sorted | get ($n - 1)),
        mean: ($total / $n),
        median: $median,
        p95: ($sorted | get $p95_index)
    }
}

def canonical [value] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        mut result = {}
        for key in ($value | columns | sort) {
            $result = ($result | insert $key (canonical ($value | get $key)))
        }
        $result
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        $value | each { |item| canonical $item }
    } else {
        $value
    }
}

def indent [depth: int] {
    if $depth == 0 { "" } else { (1..($depth * 3) | each { " " } | str join "") }
}

def pretty-json [value, depth: int = 0] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        let keys = ($value | columns | sort)
        if ($keys | length) == 0 { return "{}" }
        mut entries = []
        for key in $keys {
            let item = pretty-json ($value | get $key) ($depth + 1)
            $entries = ($entries | append $"(indent ($depth + 1))($key | to json) : ($item)")
        }
        "{\n" + ($entries | str join ",\n") + $"\n(indent $depth)}"
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        let values = ($value | each { |item| pretty-json $item ($depth + 1) })
        if ($values | length) == 0 { return "[]" }
        let entries = ($values | each { |item| (indent ($depth + 1)) + $item })
        "[\n" + ($entries | str join ",\n") + $"\n(indent $depth)]"
    } else if $kind == "float" and $value == ($value | math floor) {
        ($value | into int | to json)
    } else if $kind == "float" {
        let exponent = (($value | math abs | math log 10 | math floor) | into int)
        let decimals = 14 - $exponent
        (($value | into string --decimals $decimals | str replace --regex '0+$' "" | str replace --regex '\.$' "") )
    } else {
        $value | to json
    }
}

def add-value [data: record, key: string, value] {
    let previous = if (($data | columns) | any { |column| $column == $key }) { $data | get $key } else { [] }
    $data | upsert $key ($previous | append $value)
}

def nested3 [root: record, first: string, second: string, third: string, value] {
    let first_record = if (($root | columns) | any { |key| $key == $first }) { $root | get $first } else { {} }
    let second_record = if (($first_record | columns) | any { |key| $key == $second }) { $first_record | get $second } else { {} }
    let next_second = ($second_record | upsert $third $value)
    let next_first = ($first_record | upsert $second $next_second)
    $root | upsert $first $next_first
}

def csv-rows [path: string] {
    open --raw $path
    | from csv --noheaders
}

def main [...argv: string] {
    if ($argv | length) != 1 {
        print -n --stderr "usage: summarize-components.pl COMPONENT_DIRECTORY\n"
        exit 255
    }
    let dir = ($argv | get 0)
    let script_dir = $env.FILE_PWD
    let die_script = ([$script_dir "../../../kern/die.pl"] | path join)
    let header = (run-external "perl" $die_script "header" | complete)
    ensure ($header.exit_code == 0) "opcode header failed"
    mut opcode = {}
    for line in ($header.stdout | lines) {
        let row = ($line | parse --regex '^#define OP_(?<name>ATTNPARTN|ATTNMERGE) (?<number>[0-9]+)$')
        if ($row | length) == 1 {
            let item = ($row | get 0)
            $opcode = ($opcode | insert $item.name (integer $item.number "opcode"))
        }
    }
    ensure (($opcode | columns | length) == 2) "opcode header failed"

    let manifest = (open $"($dir)/manifest.json")
    let status = try { open --raw $"($dir)/status.txt" } catch { fail "missing status" }
    ensure (($status | lines | get 0) == "PASS") "component matrix did not pass"
    let keys = ($manifest.keys | split row ",")
    let seeds = ($manifest.seeds | split row ",")
    let modes = ($manifest.modes | split row ",")
    let repeats = $manifest.repeats
    mut expected = []
    mut timing_result = {}
    for mode in $modes {
        for n in $keys {
            for seed in $seeds {
                $expected = ($expected | append $"($mode)-keys($n)-seed($seed)")
            }
        }
    }
    let files = (glob $"($dir)/*-keys*-seed*.json" | each { |path| $path | path basename | str replace ".json" "" } | sort)
    ensure ($files == ($expected | sort)) "missing, extra, or duplicate component cases"

    mut timing = {}
    mut work = {}
    mut body = {}
    mut wait = {}
    mut cases = 0
    mut guard_launches = 0
    mut verified_records = 0
    mut raw_records = 0

    for mode in $modes {
        for n_text in $keys {
            let n = (integer $n_text "key count")
            for seed_text in $seeds {
                let seed = (integer $seed_text "seed")
                let name = $"($mode)-keys($n)-seed($seed)"
                let path = $"($dir)/($name).json"
                let sample = (open $path)
                $cases = $cases + 1
                ensure ($sample.keys == $n and $sample.seed == $seed and $sample.bit_exact and $sample.boundary_canaries and $sample.boundary_canary_words == 64 and $sample.partial_values == 49536 and $sample.final_values == 6144 and $sample.steps_per_sample == 64 and $sample.guard_checked_launches == (4 + 4 * ($repeats + 4))) $"bad coverage ($name)"
                let traced = $mode == "traced"
                ensure (($sample.trace_verified == $traced) and ($sample.timing_trace_enabled == $traced) and ($sample.trace_launches == (if $traced { $sample.guard_checked_launches } else { 0 })) and ($sample.trace_records == (if $traced { 256 + 192 * ($repeats + 4) } else { 0 }))) $"bad trace coverage ($name)"
                $guard_launches = $guard_launches + $sample.guard_checked_launches
                $verified_records = $verified_records + $sample.trace_records
                for metric in ["event" "partial_program" "merge"] {
                    mut arm_samples = {}
                    for arm in ["old" "new"] {
                        let measurement = ($sample | get $arm | get $metric)
                        ensure (($measurement.samples_us | describe | str starts-with "list") or ($measurement.samples_us | describe | str starts-with "table")) $"sample count ($name) ($arm) ($metric)"
                        ensure (($measurement.samples_us | length) == $repeats) $"sample count ($name) ($arm) ($metric)"
                        mut values = []
                        for value in $measurement.samples_us { $values = ($values | append (positive $value)) }
                        let derived = (stats $values)
                        ensure ((($derived.median - $measurement.median_us) | math abs) < 0.002 and (($derived.p95 - $measurement.p95_us) | math abs) < 0.002) $"reported statistic ($name)"
                        let base = $"($mode):($n):($metric):($arm)"
                        $timing = (add-value $timing $base $values)
                        $timing = (add-value $timing $"($base):case-median" $measurement.median_us)
                        $arm_samples = ($arm_samples | insert $arm $values)
                    }
                    for index in 0..($repeats - 1) {
                        let delta = (($arm_samples.new | get $index) - ($arm_samples.old | get $index))
                        $timing = (add-value $timing $"($mode):($n):($metric):paired_delta" $delta)
                    }
                }

                if $traced {
                    let raw_path = $"($dir)/($name).stderr"
                    let lines = (open --raw $raw_path | lines)
                    mut records = {}
                    mut previous = {}
                    for line in $lines {
                        let matched = ($line | parse --regex '^TRACE (?<fields>.*)$')
                        ensure (($matched | length) == 1) $"unexpected trace stderr ($name)"
                        let parts = (($matched | get 0 | get fields | split row " ") | each { |part| $part | split row "=" })
                        mut record = {}
                        for part in $parts {
                            ensure (($part | length) == 2 and (($part | get 1) =~ '^[0-9]+$')) $"bad raw field ($name)"
                            let key = ($part | get 0)
                            ensure (not (($record | columns) | any { |column| $column == $key })) $"duplicate trace field ($name)"
                            $record = ($record | insert $key (integer ($part | get 1) "raw field"))
                        }
                        for field in ["variant" "launch" "token" "phase" "cta" "opcode" "nonempty_units" "nonce" "wait_start" "wait_end" "body_start" "body_end" "end" "exchange_wait_ns" "dispatch_ns" "body_ns" "grid_wait_ns"] {
                            ensure (($record | columns | any { |column| $column == $field })) $"bad raw field ($name) ($field)"
                        }
                        let launch = $record.launch
                        let token = $record.token
                        let phase = $record.phase
                        let cta = $record.cta
                        let identity = $"($launch):($token):($phase):($cta)"
                        ensure (not (($records | columns) | any { |column| $column == $identity })) "duplicate raw record"
                        ensure ($launch >= 1 and $launch <= 4 and $record.variant == ([0 1 1 0] | get ($launch - 1)) and $record.nonce == $launch and $token < 2 and $phase < 2 and $cta < 16 and $record.opcode == (if $phase == 0 { $opcode.ATTNPARTN } else { $opcode.ATTNMERGE })) $"raw trace identity ($name)"
                        let chunk_size = (((($n + 8191) / 8192) | math floor) * 1024) | into int
                        mut units = 0
                        for unit in [$cta ($cta + 16)] {
                            let chunk = if $record.variant == 0 { $unit mod 8 } else { (($unit / 4) | math floor) | into int }
                            if $phase == 0 and ($chunk * $chunk_size) < $n { $units = $units + 1 }
                        }
                        ensure ($record.nonempty_units == $units) $"raw work mismatch ($name)"
                        let times = [$record.wait_start $record.wait_end $record.body_start $record.body_end $record.end]
                        ensure (($times | get 0) > 0 and ($times | get 4) > ($times | get 0) and ($times | window 2 | all { |pair| ($pair | get 1) >= ($pair | get 0) })) $"empty or stale trace ($name)"
                        let cta_key = $cta | into string
                        if (($previous | columns) | any { |column| $column == $cta_key }) {
                            ensure (($times | get 0) >= ($previous | get $cta_key)) $"empty or stale trace ($name)"
                        }
                        $previous = ($previous | upsert $cta_key ($times | get 4))
                        ensure ($record.exchange_wait_ns == ($times.1 - $times.0) and $record.dispatch_ns == ($times.2 - $times.1) and $record.body_ns == ($times.3 - $times.2) and $record.grid_wait_ns == ($times.4 - $times.3)) $"raw interval arithmetic ($name)"
                        $records = ($records | insert $identity $record)
                        $raw_records = $raw_records + 1
                        if $phase == 0 {
                            let arm = if $record.variant == 0 { "old" } else { "new" }
                            let work_key = $"($n):($arm):($cta)"
                            if (($work | columns) | any { |column| $column == $work_key }) {
                                ensure (($work | get $work_key) == $units) "work varies"
                            } else {
                                $work = ($work | insert $work_key $units)
                            }
                            $body = (add-value $body $work_key ($record.body_ns * 0.001))
                            $wait = (add-value $wait $work_key ($record.grid_wait_ns * 0.001))
                        }
                    }
                    ensure (($records | columns | length) == 256) $"incomplete ABBA trace ($name)"
                    for launch in 1..4 {
                        for cta in 0..15 {
                            mut last = 0
                            for token in 0..1 {
                                for phase in 0..1 {
                                    let record = ($records | get $"($launch):($token):($phase):($cta)")
                                    ensure ($record.wait_start >= $last) $"backwards phase/token transition ($name)"
                                    $last = $record.end
                                }
                            }
                        }
                        if $launch > 1 {
                            let old_end = ($records | transpose key value | where { |row| $row.key | str starts-with $"($launch - 1):" } | each { |row| $row.value.end } | math max)
                            let new_start = ($records | transpose key value | where { |row| $row.key | str starts-with $"($launch):" } | each { |row| $row.value.wait_start } | math min)
                            ensure ($new_start > $old_end) $"stale timestamps across launches ($name)"
                        }
                    }
                }
            }
        }
    }

    mut result = {
        harness_source: $manifest.harness_source,
        gpu: $manifest.gpu,
        whole_model: false,
        cases: $cases,
        guard_checked_launches: $guard_launches,
        verified_trace_records: $verified_records,
        retained_raw_trace_records: $raw_records,
        keys: ($keys | each { |value| integer $value "key count" }),
        seeds: ($seeds | each { |value| integer $value "seed" }),
        repeats: $repeats,
        component_units: "microseconds per fixed-input persistent-dispatch position",
        event_definition: "CUDA event interval for 64 attention-plus-merge positions divided by 64",
        partial_program_definition: "separate 64-position attention-only program; includes dispatcher overhead",
        merge_definition: "last-position CTA0 phase-end difference; includes dispatch and synchronization",
        physical_trace_scope: "initial ABBA correctness launches; body/wait distributions are not warmed production latency",
        across_modes: "sequential mode batches; traced/untraced differences are descriptive, not balanced instrumentation-overhead estimates"
    }
    for mode in $modes {
        for n_text in $keys {
            let n = (integer $n_text "key count")
            for metric in ["event" "partial_program" "merge"] {
                let node = $"($mode):($n):($metric)"
                mut metric_result = {}
                for arm in ["old" "new"] {
                    $metric_result = ($metric_result | insert $arm (stats ($timing | get $"($node):($arm)" | flatten)))
                    $metric_result = ($metric_result | insert $"($arm)-case-medians" (stats ($timing | get $"($node):($arm):case-median" | flatten)))
                }
                let paired = (stats ($timing | get $"($node):paired_delta" | flatten))
                $metric_result = ($metric_result | insert paired_delta $paired)
                let faster = ($timing | get $"($node):paired_delta" | flatten | where { |value| $value < 0 } | length)
                $metric_result = ($metric_result | insert new_faster_pairs $faster)
                $timing_result = (nested3 $timing_result $mode ($n | into string) $metric $metric_result)
            }
        }
    }

    mut physical = {}
    for key in ($work | columns) {
        let parts = ($key | split row ":")
        let n = $parts | get 0
        let arm = $parts | get 1
        let cta = ($parts | get 2 | into int)
        let units = ($work | get $key)
        let body_values = ($body | get $key)
        let wait_values = ($wait | get $key)
        let entry = {
            cta: $cta,
            nonempty_units: $units,
            body_us: (stats $body_values),
            grid_wait_us: (stats $wait_values)
        }
        let n_record = if (($physical | columns) | any { |column| $column == $n }) { $physical | get $n } else { {} }
        let arm_values = if (($n_record | columns) | any { |column| $column == $arm }) { $n_record | get $arm } else { [] }
        let next_arm = ($arm_values | append $entry)
        $physical = ($physical | upsert $n ($n_record | upsert $arm $next_arm))
    }
    mut physical_result = {}
    for n in ($physical | columns) {
        let arms = ($physical | get $n)
        mut next_arms = {}
        for arm in ($arms | columns) {
            $next_arms = ($next_arms | insert $arm (($arms | get $arm) | sort-by cta))
        }
        $physical_result = ($physical_result | insert $n $next_arms)
    }
    $result = ($result | insert timing $timing_result | insert physical_cta $physical_result)

    mut telemetry = {}
    mut utc_values = []
    mut samples = 0
    mut other_max = 0
    for path in (glob $"($dir)/telemetry-*.gpu.csv" | sort) {
        let rows = (csv-rows $path)
        let tag = (($path | path basename | str replace "telemetry-" "" | str replace ".gpu.csv" ""))
        mut selected = 0
        for row in $rows {
            let fields = ($row | values | each { |field| $field | into string | str trim })
            ensure (($fields | length) == 9) $"telemetry columns ($path)"
            let utilization = integer ($fields | get 6) "GPU utilization"
            if ($fields | get 0) != $manifest.gpu {
                if $utilization > $other_max { $other_max = $utilization }
                continue
            }
            $selected = $selected + 1
            for item in [[3 "temperature_c"] [4 "sm_mhz"] [5 "memory_mhz"] [6 "gpu_utilization_percent"] [7 "memory_used_mib"] [8 "power_w"]] {
                let index = $item | get 0
                let label = $item | get 1
                let text = $fields | get $index
                ensure ($text =~ '^[0-9]+(?:\.[0-9]+)?$') $"unknown telemetry ($label) ($path)"
                $telemetry = (add-value $telemetry $label ($text | into float))
            }
            if $utilization > 0 {
                $telemetry = (add-value $telemetry "active_sm_mhz" (($fields | get 4) | into float))
            }
        }
        ensure ($selected == 1) $"target GPU missing ($path)"
        $samples = $samples + 1
        let utc_path = ($path | str replace ".gpu.csv" ".utc")
        $utc_values = ($utc_values | append ((open --raw $utc_path | str trim)))
    }
    ensure ($samples > 0 and $other_max == 0) "missing telemetry or competing GPU activity"
    mut telemetry_result = {
        samples: $samples,
        first_utc: ($utc_values | sort | get 0),
        last_utc: ($utc_values | sort | get (($utc_values | length) - 1)),
        maximum_other_gpu_utilization_percent: $other_max
    }
    for label in ($telemetry | columns) {
        $telemetry_result = ($telemetry_result | insert $label (stats ($telemetry | get $label)))
    }
    $result = ($result | insert telemetry $telemetry_result)

    let rejection_path = $"($dir)/uninstrumented-rejection.stderr"
    if ($rejection_path | path exists) {
        let rejection = (open --raw $rejection_path)
        ensure ($rejection =~ '(?m)^trace rejected variant=0 launch=1 record=0: launch nonce ' and not ($rejection =~ 'undefined symbol|CUDA_ERROR|->')) "incorrect negative result"
        let exits = (open --raw $"($dir)/exits.tsv" | lines)
        let negative = ($exits | where { |line| $line | str starts-with "uninstrumented-rejection\t" })
        ensure (($negative | length) == 1 and ($negative | get 0) == "uninstrumented-rejection\t1") "incorrect negative exit"
        $result = ($result | insert uninstrumented_image_rejection { passed: true, exit_code: 1, reason: "missing launch nonce" })
    }
    print -n ((pretty-json (canonical $result)) + "\n")
}
