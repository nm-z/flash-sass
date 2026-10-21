# CPU-only malformed-evidence tests for cta_remap_summary.nu.

def fail [message: string] {
    print -n --stderr $"($message)\n"
    exit 1
}

def ensure [condition: bool, message: string] {
    if not $condition { fail $message }
}

def write-text [path: string, value: string] {
    $value | save --force --raw $path
}

def write-json [path: string, value] {
    (($value | to json --indent 4) + "\n") | save --force --raw $path
}

def edit-sample [dir: string, stage: string, arm: string, pair: int, edit: closure] {
    let path = $"($dir)/($stage)-($arm)-pair($pair).json"
    let sample = (open $path)
    let updated = (do $edit $sample)
    write-json $path $updated
}

def edit-config [dir: string, transform: closure] {
    let path = $"($dir)/config.txt"
    let input = (open --raw $path)
    write-text $path (do $transform $input)
}

def edit-trace-field [dir: string, row: int, column: int, value] {
    let path = $"($dir)/traced-new-pair1.trace.csv"
    let lines = (open --raw $path | lines)
    let fields = (($lines | get $row) | split row ",")
    let updated = ($fields | update $column ($value | into string))
    let next_lines = ($lines | update $row ($updated | str join ","))
    write-text $path (($next_lines | str join "\n") + "\n")
}

def create-fixture [dir: string] {
    mkdir $dir
    write-text $"($dir)/status.txt" "PASS\n"
    mut config = "pairs=2\ndecode_intervals=2\ndies=6\nlayers=48\nnblk=512\nprompt=2048\n"
    for die in 0..5 { $config = $config + $"die($die)_phases=4\n" }
    write-text $"($dir)/config.txt" $config
    let columns = ["die" "token" "phase" "cta" "opcode" "op" "wait_start_ns" "wait_end_ns" "body_start_ns" "body_end_ns" "phase_end_ns" "nonempty_units" "nonce"]
    let opcodes = [67 66 70 0]
    let names = ["attnpartn" "kvgather" "idxscore" "embrow"]

    for stage in ["untraced" "traced"] {
        for arm in ["old" "new"] {
            for pair in -2..1 {
                let us = 1000 + (if $arm == "new" { 20 } else { 0 }) + (if $stage == "traced" { 40 } else { 0 }) + (10 * $pair)
                let order = if ($pair mod 2) != 0 { if $arm == "new" { 1 } else { 0 } } else { if $arm == "old" { 1 } else { 0 } }
                let stage_order = if ($pair mod 2) != 0 { if $stage == "traced" { 1 } else { 0 } } else { if $stage == "untraced" { 1 } else { 0 } }
                let nonce = (($pair + 2) * 4) + ($stage_order * 2) + $order + 1
                mut sample = {
                    stage: $stage,
                    arm: $arm,
                    pair: $pair,
                    warmup: ($pair < 0),
                    order: $order,
                    stage_order: $stage_order,
                    token_parity: true,
                    resumed: 2047,
                    decode_intervals: 2,
                    token_ids: [11 22 33],
                    arrival_since_first_us: [0 $us (2 * $us)],
                    host_decode_us: $us,
                    dies: []
                }
                if $stage == "traced" {
                    mut csv = ($columns | str join ",") + "\n"
                    for die in 0..5 {
                        let base = ((6 - $die) * 1000000000000) + ($nonce * 10000000)
                        let metrics = {
                            die: $die,
                            device_decode_us: 800,
                            attention_body_us: 50,
                            gather_body_us: 50,
                            indexer_body_us: 50,
                            other_body_us: 50,
                            exchange_wait_us: 100,
                            dispatch_us: 100,
                            grid_wait_us: 200,
                            transition_us: 200
                        }
                        $sample = ($sample | update dies ($sample.dies | append $metrics))
                        for token_offset in 0..1 {
                            for phase in 0..3 {
                                for cta in 0..15 {
                                    let start = $base + ($token_offset * 800000) + ($phase * 200000) + $cta
                                    let work = if $phase != 0 { 0 } else if $arm == "old" { if ($cta mod 8) < 3 { 2 } else { 0 } } else { if $cta < 12 { 1 } else { 0 } }
                                    let fields = [$die (2048 + $token_offset) $phase $cta ($opcodes | get $phase) ($names | get $phase) $start ($start + 25000) ($start + 50000) ($start + 100000) ($start + 150000) $work $nonce]
                                    $csv = $csv + (($fields | each { |field| $field | into string } | str join ",") + "\n")
                                }
                            }
                        }
                    }
                    write-text $"($dir)/($stage)-($arm)-pair($pair).trace.csv" $csv
                }
                write-json $"($dir)/($stage)-($arm)-pair($pair).json" $sample
            }
        }
    }
}

def main [...argv: string] {
    let made = (^mktemp -d --tmpdir cta-remap-summary-test-XXXXXX | complete)
    ensure ($made.exit_code == 0) "could not create temporary fixture directory"
    let root = ($made.stdout | str trim)
    let base = $"($root)/complete"
    create-fixture $base

    let cases = [
        { name: "complete", want_fail: false },
        { name: "missing_sample", want_fail: true },
        { name: "missing_warmup", want_fail: true },
        { name: "missing_trace", want_fail: true },
        { name: "changed_tokens", want_fail: true },
        { name: "fractional_token", want_fail: true },
        { name: "first_token_bias", want_fail: true },
        { name: "nonfinite_callback", want_fail: true },
        { name: "bad_order", want_fail: true },
        { name: "bad_reconciliation", want_fail: true },
        { name: "coherent_json_tampering", want_fail: true },
        { name: "missing_die", want_fail: true },
        { name: "failed_status", want_fail: true },
        { name: "arbitrary_nonempty_trace", want_fail: true },
        { name: "missing_raw_record", want_fail: true },
        { name: "duplicate_raw_cta", want_fail: true },
        { name: "wrong_nonce", want_fail: true },
        { name: "wrong_cta15_work", want_fail: true },
        { name: "wrong_opcode_name", want_fail: true },
        { name: "backwards_raw_time", want_fail: true },
        { name: "nonfinite_raw_time", want_fail: true },
        { name: "stale_raw_launch", want_fail: true },
        { name: "wrong_phase_count", want_fail: true },
        { name: "duplicate_configuration", want_fail: true }
    ]
    let summary_script = [$env.FILE_PWD "cta_remap_summary.nu"] | path join
    mut passed = 0

    for case in $cases {
        let dir = if $case.name == "complete" {
            $base
        } else {
            let copy = $"($root)/($case.name)"
            ^cp -a $base $copy
            $copy
        }
        match $case.name {
            "missing_sample" => { rm $"($dir)/untraced-old-pair1.json" }
            "missing_warmup" => { rm $"($dir)/untraced-old-pair-1.json" }
            "missing_trace" => { rm $"($dir)/traced-new-pair1.trace.csv" }
            "changed_tokens" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update token_ids ($sample.token_ids | update 2 99) }
            }
            "fractional_token" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update token_ids ($sample.token_ids | update 2 33.5) }
            }
            "first_token_bias" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update host_decode_us ($sample.host_decode_us + 10) }
            }
            "nonfinite_callback" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update arrival_since_first_us ($sample.arrival_since_first_us | update 1 "Infinity") }
            }
            "bad_order" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update order 0 }
            }
            "bad_reconciliation" => {
                edit-sample $dir "traced" "new" 1 {
                    |sample|
                    let die = ($sample.dies | get 4 | update device_decode_us (($sample.dies | get 4).device_decode_us + 1))
                    $sample | update dies ($sample.dies | update 4 $die)
                }
            }
            "coherent_json_tampering" => {
                edit-sample $dir "traced" "new" 1 {
                    |sample|
                    let die = ($sample.dies | get 4)
                    let changed = ($die | update attention_body_us ($die.attention_body_us + 1) | update transition_us ($die.transition_us - 1))
                    $sample | update dies ($sample.dies | update 4 $changed)
                }
            }
            "missing_die" => {
                edit-sample $dir "traced" "new" 1 { |sample| $sample | update dies ($sample.dies | take 5) }
            }
            "failed_status" => { write-text $"($dir)/status.txt" "FAIL\n" }
            "arbitrary_nonempty_trace" => { write-text $"($dir)/traced-new-pair1.trace.csv" "synthetic_fixture\n" }
            "missing_raw_record" => {
                let path = $"($dir)/traced-new-pair1.trace.csv"
                let lines = (open --raw $path | lines | drop)
                write-text $path (($lines | str join "\n") + "\n")
            }
            "duplicate_raw_cta" => {
                let path = $"($dir)/traced-new-pair1.trace.csv"
                let lines = (open --raw $path | lines)
                write-text $path (($lines | update 2 ($lines | get 1) | str join "\n") + "\n")
            }
            "wrong_nonce" => { edit-trace-field $dir 1 12 9 }
            "wrong_cta15_work" => { edit-trace-field $dir 16 11 3 }
            "wrong_opcode_name" => { edit-trace-field $dir 1 5 "attnmerge" }
            "backwards_raw_time" => {
                let lines = (open --raw $"($dir)/traced-new-pair1.trace.csv" | lines)
                let field = (($lines | get 1 | split row "," | get 8) | into int)
                edit-trace-field $dir 1 8 ($field - 100000)
            }
            "nonfinite_raw_time" => { edit-trace-field $dir 1 6 "NaN" }
            "stale_raw_launch" => {
                let path = $"($dir)/traced-new-pair1.trace.csv"
                mut lines = (open --raw $path | lines)
                for index in 1..(($lines | length) - 1) {
                    let fields = ($lines | get $index | split row ",")
                    mut updated = $fields
                    for column in 6..10 { $updated = ($updated | update $column ((($fields | get $column | into int) - 50000000) | into string)) }
                    $lines = ($lines | update $index ($updated | str join ","))
                }
                write-text $path (($lines | str join "\n") + "\n")
            }
            "wrong_phase_count" => { edit-config $dir { |input| $input | str replace "die0_phases=4" "die0_phases=5" } }
            "duplicate_configuration" => { edit-config $dir { |input| $input + "prompt=2048\n" } }
            _ => {}
        }

        let result = (^nu --no-config-file $summary_script $dir | complete)
        ensure (($result.exit_code != 0) == $case.want_fail) $"($case.name) unexpected rc=($result.exit_code)\n($result.stdout)\n($result.stderr)"
        if not $case.want_fail {
            let summary = ($result.stdout | from json)
            ensure ($summary.raw_samples == 8 and $summary.warmup_samples == 8 and $summary.raw_trace_records == 6144 and $summary.raw_traces_verified and $summary.untraced.paired_host_delta_us.median == 20 and $summary.instrumentation_overhead.old.paired_host_delta_us.median == 40 and $summary.traced.new.dies."0".attention_body_us.median == 50 and $summary.traced.new.dies."0".device_decode_us.median == 800) "synthetic distribution or raw-trace arithmetic incorrect"
        }
        let outcome = if $case.want_fail { "rejected malformed synthetic evidence" } else { "accepted complete synthetic evidence" }
        print ($case.name + ": PASS (" + $outcome + ")")
        $passed = $passed + 1
    }
    print $"($passed) CPU-only summary checks passed; no GPU results generated"
    ^rm -rf $root
}
