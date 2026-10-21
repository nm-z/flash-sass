# CPU-only audit of the preserved issue-12 component evidence.

def fail [message: string] {
    print -n --stderr $"($message)\n"
    exit 1
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

def finite-positive [value] {
    let text = ($value | into string)
    if not ($text =~ '^(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$') {
        return false
    }
    let number = try { $text | into float } catch { return false }
    $number > 0 and $number < 1e100
}

def median [values: list] {
    let sorted = ($values | sort)
    let count = ($sorted | length)
    ensure ($count > 0) "empty distribution"
    if ($count mod 2) == 1 {
        $sorted | get (($count / 2) | into int)
    } else {
        let middle = (($count / 2) | into int)
        (($sorted | get ($middle - 1)) + ($sorted | get $middle)) / 2.0
    }
}

def read-table [path: string] {
    let rows = (open --raw $path | lines | each { |line| $line | split row "\t" })
    ensure (($rows | all { |row| ($row | length) == 2 })) $"invalid table ($path)"
    let names = ($rows | each { |row| $row | get 0 })
    ensure (($names | uniq | length) == ($names | length)) $"duplicate table row ($path)"
    mut out = {}
    for row in $rows {
        $out = ($out | insert ($row | get 0) (integer ($row | get 1) "table value"))
    }
    $out
}

def csv-rows [path: string] {
    open --raw $path
    | from csv --noheaders
}

def main [...argv: string] {
    if ($argv | length) != 2 {
        print -n --stderr "usage: audit-component.rb RAW_COMPONENT_DIRECTORY OUTPUT_JSON\n"
        exit 2
    }
    let root = ($argv | get 0 | path expand)
    let out = ($argv | get 1 | path expand)
    let manifest = (open $"($root)/manifest.json")
    ensure ((open --raw $"($root)/status.txt" | str trim) == "PASS") "run did not finish PASS"
    ensure ($manifest.whole_model == false) "component evidence mislabeled whole-model"
    ensure ($manifest.host_lock == "/tmp/flash-sass-engine-validation.lock") "wrong host reservation"
    ensure ($manifest.harness_source == "6ae0a092f2a5a0f6a472066152bcdd12daef8303") "unexpected harness source"

    let counts = [1 1023 1024 1025 2047 2048 2049 2050 2051 2052 4095 4096 4097 8191 8192 8193]
    let seeds = [0 77 4242]
    let modes = ["historical" "matched" "traced"]
    ensure (($manifest.keys | split row "," | each { |x| integer $x "key count" }) == $counts) "wrong key coverage"
    ensure (($manifest.seeds | split row "," | each { |x| integer $x "seed" }) == $seeds) "wrong seed coverage"
    ensure (($manifest.modes | split row ",") == $modes) "wrong mapping/trace coverage"
    ensure ($manifest.repeats == 32) "wrong repetition count"

    mut expected = []
    for mode in $modes {
        for n in $counts {
            for seed in $seeds {
                $expected = ($expected | append $"($mode)-keys($n)-seed($seed)")
            }
        }
    }
    let files = (glob $"($root)/*-keys*-seed*.json" | each { |path| $path | path basename | str replace ".json" "" } | sort)
    ensure ($files == ($expected | sort)) "missing, extra, or duplicate component cases"

    let exits = (read-table $"($root)/exits.tsv")
    let children = (read-table $"($root)/children.tsv")
    let expected_exits = ($expected | append "uninstrumented-rejection" | sort)
    ensure (($exits | columns | sort) == $expected_exits) "incomplete exit coverage"
    ensure (($children | columns | sort) == ($exits | columns | sort)) "incomplete child identity coverage"
    ensure (($children | columns | each { |key| ($children | get $key) > 0 } | all { |v| $v })) "invalid child PID"
    for key in $expected {
        ensure (($exits | get $key) == 0) "positive case failed"
    }
    ensure (($exits | get "uninstrumented-rejection") == 1) "negative case did not exit 1"
    let rejection = (open --raw $"($root)/uninstrumented-rejection.stderr")
    ensure ($rejection =~ '(?m)^trace rejected variant=') "negative test did not reject a trace"
    ensure (not ($rejection =~ 'undefined symbol|CUDA_ERROR|->')) "loader or CUDA failure mistaken for negative trace test"

    mut results = {}
    mut raw_records = 0
    mut work = {}
    for mode in $modes {
        for n in $counts {
            for seed in $seeds {
                let name = $"($mode)-keys($n)-seed($seed)"
                let sample = (open $"($root)/($name).json")
                let wanted_trace = $mode == "traced"
                ensure ($sample.keys == $n) $"($name): wrong keys"
                ensure ($sample.seed == $seed) $"($name): wrong seed"
                ensure ($sample.device == "Tesla M60") $"($name): wrong device"
                ensure ($sample.bit_exact == true) $"($name): wrong bit_exact"
                ensure ($sample.boundary_canaries == true) $"($name): wrong boundary_canaries"
                ensure ($sample.boundary_canary_words == 64) $"($name): wrong boundary_canary_words"
                ensure ($sample.guard_checked_launches == 148) $"($name): wrong guard_checked_launches"
                ensure ($sample.partial_values == 49536) $"($name): wrong partial_values"
                ensure ($sample.final_values == 6144) $"($name): wrong final_values"
                ensure ($sample.steps_per_sample == 64) $"($name): wrong steps_per_sample"
                ensure ($sample.trace_verified == $wanted_trace) $"($name): wrong trace_verified"
                ensure ($sample.timing_trace_enabled == $wanted_trace) $"($name): wrong timing_trace_enabled"
                ensure ($sample.trace_launches == (if $wanted_trace { 148 } else { 0 })) $"($name): wrong trace_launches"
                ensure ($sample.trace_records == (if $wanted_trace { 7168 } else { 0 })) $"($name): wrong trace_records"

                for arm in ["old" "new"] {
                    for phase in ["event" "partial_program" "merge"] {
                        let distribution = ($sample | get $arm | get $phase)
                        let samples = $distribution.samples_us
                        ensure (($samples | length) == 32 and ($samples | all { |x| finite-positive $x })) $"($name): invalid ($arm)/($phase) samples"
                        let med = median $samples
                        let sorted = ($samples | sort)
                        let p95_index = (((($samples | length) * 0.95) | math ceil) | into int) - 1
                        ensure ((($med - $distribution.median_us) | math abs) <= 0.002) $"($name): inconsistent median"
                        ensure (((($sorted | get $p95_index) - $distribution.p95_us) | math abs) <= 0.002) $"($name): inconsistent p95"
                    }
                }

                $results = ($results | insert $name $sample)
                if $wanted_trace {
                    let trace_path = $"($root)/($name).stderr"
                    let lines = (open --raw $trace_path | lines)
                    ensure (($lines | length) == 256 and ($lines | all { |line| $line | str starts-with "TRACE " })) $"($name): incomplete raw trace"
                    mut records = {}
                    for line in $lines {
                        let parts = ($line | str replace "TRACE " "" | split row " " | each { |part| $part | split row "=" })
                        ensure (($parts | all { |part| ($part | length) == 2 and (($part | get 1) =~ '^[0-9]+$') })) $"($name): malformed trace field"
                        let keys = ($parts | each { |part| $part | get 0 })
                        ensure (($keys | uniq | length) == ($keys | length)) $"($name): duplicate trace field"
                        mut record = {}
                        for part in $parts {
                            $record = ($record | insert ($part | get 0) (integer ($part | get 1) "trace field"))
                        }
                        let launch = $record.launch
                        let token = $record.token
                        let phase = $record.phase
                        let cta = $record.cta
                        let identity = $"($launch):($token):($phase):($cta)"
                        ensure (not ($records | columns | any { |key| $key == $identity })) $"($name): duplicate CTA record"
                        ensure (($launch >= 1 and $launch <= 4) and ($token >= 0 and $token <= 1) and ($phase >= 0 and $phase <= 1) and ($cta >= 0 and $cta <= 15)) $"($name): invalid CTA identity"
                        let variant = ([0 1 1 0] | get ($launch - 1))
                        ensure ($record.variant == $variant and $record.nonce == $launch) $"($name): stale or wrong launch identity"
                        ensure ($record.opcode == (if $phase == 0 { 67 } else { 22 })) $"($name): wrong operation code"
                        let chunk_size = (((($n + 8191) / 8192) | math floor) * 1024) | into int
                        mut units = 0
                        for unit in [$cta ($cta + 16)] {
                            let chunk = if $variant == 0 { $unit mod 8 } else { (($unit / 4) | math floor) | into int }
                            if $phase == 0 and ($chunk * $chunk_size) < $n { $units = $units + 1 }
                        }
                        ensure ($record.nonempty_units == $units) $"($name): wrong physical CTA work count"
                        let times = [$record.wait_start $record.wait_end $record.body_start $record.body_end $record.end]
                        ensure (($times | get 0) > 0 and ($times | get 4) > ($times | get 0) and ($times | window 2 | all { |pair| ($pair | get 0) <= ($pair | get 1) })) $"($name): invalid timestamp order"
                        ensure ($record.exchange_wait_ns == ($times.1 - $times.0)) $"($name): inconsistent exchange_wait_ns"
                        ensure ($record.dispatch_ns == ($times.2 - $times.1)) $"($name): inconsistent dispatch_ns"
                        ensure ($record.body_ns == ($times.3 - $times.2)) $"($name): inconsistent body_ns"
                        ensure ($record.grid_wait_ns == ($times.4 - $times.3)) $"($name): inconsistent grid_wait_ns"
                        $records = ($records | insert $identity $record)
                        $raw_records = $raw_records + 1
                    }

                    for launch in 1..4 {
                        for cta in 0..15 {
                            mut previous = 0
                            for token in 0..1 {
                                for phase in 0..1 {
                                    let record = ($records | get $"($launch):($token):($phase):($cta)")
                                    if $previous > 0 {
                                        ensure ($previous <= $record.wait_start) $"($name): backwards phase/token transition"
                                    }
                                    $previous = $record.end
                                }
                            }
                        }
                        if $launch != 1 {
                            let prior_ends = ($records | transpose key value | where { |row| $row.key | str starts-with $"($launch - 1):" } | each { |row| $row.value.end })
                            let now_starts = ($records | transpose key value | where { |row| $row.key | str starts-with $"($launch):" } | each { |row| $row.value.wait_start })
                            ensure (($now_starts | math min) > ($prior_ends | math max)) $"($name): stale timestamps across launches"
                        }
                    }
                    if $seed == 77 and ($n == 2048 or $n == 2051) {
                        mut arms = {}
                        for arm in ["old" "new"] {
                            let launch = if $arm == "old" { 1 } else { 2 }
                            mut values = []
                            for cta in 0..15 {
                                $values = ($values | append ($records | get $"($launch):0:0:($cta)" | get nonempty_units))
                            }
                            let row = {
                                nonempty_units_by_physical_cta: $values,
                                active_ctas: ($values | where { |value| $value > 0 } | length),
                                total_nonempty_units: ($values | math sum)
                            }
                            $arms = ($arms | insert $arm $row)
                        }
                        $work = ($work | insert ($n | into string) $arms)
                    }
                }
            }
        }
    }

    let target = $manifest.gpu
    mut telemetry_count = 0
    mut target_rows = []
    mut max_other_util = 0
    let telemetry_files = (glob $"($root)/telemetry-*.gpu.csv" | sort)
    for path in $telemetry_files {
        let prefix = ($path | str replace ".gpu.csv" "")
        let tag = ($prefix | path basename | str replace "telemetry-" "")
        let rows = (csv-rows $path)
        ensure (($rows | length) == 8 and ($rows | all { |row| ($row | columns | length) == 9 })) $"($tag): incomplete GPU telemetry"
        let uuids = ($rows | each { |row| $row.column0 | str trim })
        ensure (($uuids | uniq | length) == 8) $"($tag): duplicate GPU UUID"
        ensure (($rows | where { |row| ($row.column0 | str trim) == $target } | length) == 1) $"($tag): target missing"
        for row in $rows {
            let utilization = (integer $row.column6 "GPU utilization")
            ensure ($utilization >= 0 and $utilization <= 100) $"($tag): invalid utilization"
            if ($row.column0 | str trim) != $target {
                if $utilization > $max_other_util { $max_other_util = $utilization }
                ensure ($utilization == 0) $"($tag): competing off-device activity"
            } else {
                if $tag == "initial" { ensure ($utilization == 0) "initial target not idle" }
                $target_rows = ($target_rows | append $row)
            }
        }
        let allowed = if ($tag == "initial" or ($tag | str ends-with "-before") or ($tag | str ends-with "-after")) {
            0
        } else if ($tag =~ '^rejection-[0-9]+$') {
            $children | get "uninstrumented-rejection"
        } else {
            let matches = ($tag | parse --regex '^(?<case>.+-keys[0-9]+-seed[0-9]+)-[0-9]+$')
            ensure (($matches | length) == 1) $"unrecognized telemetry sample ($tag)"
            $children | get ($matches | get 0 | get case)
        }
        let app_path = $"($prefix).apps.csv"
        for row in (csv-rows $app_path) {
            if (($row | values | all { |field| ($field | into string | str trim) == "" })) { continue }
            ensure (($row | columns | length) == 4) $"($tag): malformed compute-app row"
            let pid = (integer $row.column1 "compute-app PID")
            if ($row.column0 | str trim) == $target {
                ensure ($allowed > 0 and $pid == $allowed) $"($tag): another target compute process"
            }
        }
        ensure ((open --raw $"($prefix).stderr") == "") $"($tag): telemetry query stderr"
        $telemetry_count = $telemetry_count + 1
    }
    ensure ($telemetry_count > 0) "no telemetry"

    mut summaries = {}
    for mode in $modes {
        mut rows = []
        for n in $counts {
            mut old_values = []
            mut new_values = []
            for seed in $seeds {
                let sample = ($results | get $"($mode)-keys($n)-seed($seed)")
                $old_values = ($old_values | append $sample.old.event.median_us)
                $new_values = ($new_values | append $sample.new.event.median_us)
            }
            let old = (median $old_values)
            let new = (median $new_values)
            let row = {
                keys: $n,
                seeds: 3,
                old_partial_merge_us: $old,
                new_partial_merge_us: $new,
                delta_us: ($new - $old),
                reduction_percent: (100.0 * ($old - $new) / $old)
            }
            $rows = ($rows | append $row)
        }
        $summaries = ($summaries | insert $mode $rows)
    }

    let result = {
        status: "PASS",
        cpu_only_audit: true,
        gpu_executed_by_this_audit: false,
        source: $manifest,
        component_cases: ($results | columns | length),
        retained_event_samples_checked: (($results | columns | length) * 2 * 3 * 32),
        raw_cta_records_checked: $raw_records,
        negative_trace_rejection: "PASS",
        telemetry_samples_checked: $telemetry_count,
        maximum_other_gpu_utilization_percent: $max_other_util,
        telemetry_limitation: "One-second sampled evidence cannot exclude sub-second interference.",
        target: {
            uuid: $target,
            drivers: ($target_rows | each { |row| $row.column2 | str trim } | uniq),
            temperature_c_range: [($target_rows | each { |row| ($row.column3 | str trim | into int) } | math min) ($target_rows | each { |row| ($row.column3 | str trim | into int) } | math max)],
            sm_clock_mhz_range: [($target_rows | each { |row| ($row.column4 | str trim | into int) } | math min) ($target_rows | each { |row| ($row.column4 | str trim | into int) } | math max)],
            memory_clock_mhz_range: [($target_rows | each { |row| ($row.column5 | str trim | into int) } | math min) ($target_rows | each { |row| ($row.column5 | str trim | into int) } | math max)]
        },
        physical_cta_work: $work,
        summaries: $summaries,
        whole_model_validated: false
    }
    let json = (($result | to json --indent 2) + "\n")
    $json | save --raw $out
    print "PASS: 144 component cases, 12288 raw CTA records, 656 telemetry snapshots; whole-model not validated."
    for mode in $modes {
        for row in ($summaries | get $mode | where { |item| [2048 2051 8192] | any { |n| $n == $item.keys } }) {
            ^printf "%s keys=%d: %.3f -> %.3f us; reduction %.3f%%\n" $mode $row.keys $row.old_partial_merge_us $row.new_partial_merge_us $row.reduction_percent
        }
    }
}
