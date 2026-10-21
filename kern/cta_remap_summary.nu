# Recompute device timing from complete raw cta_remap_bench.csv traces.

def s_fail [message: string, code: int = 255] {
    print -n --stderr $message
    if not ($message | str ends-with "\n") { print -n --stderr "\n" }
    exit $code
}

def s_ensure [condition: bool, message: string] {
    if not $condition { s_fail $"($message)\n" }
}

def s_uint [value, name: string, max: int = 4294967295] {
    if ($value | describe) == "int" {
        if $value < 0 or $value > $max { s_fail $"invalid unsigned ($name)\n" }
        return $value
    }
    let text = ($value | into string)
    if not ($text =~ '^(0|[1-9][0-9]*)$') { s_fail $"invalid unsigned ($name)\n" }
    let number = try { $text | into int } catch { s_fail $"invalid unsigned ($name)\n" }
    if $number > $max { s_fail $"invalid unsigned ($name)\n" }
    $number
}

def s_number [value, name: string] {
    let text = ($value | into string)
    if not ($text =~ '^-?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$') {
        s_fail $"invalid nonnegative finite ($name)\n"
    }
    let number = try { $text | into float } catch { s_fail $"invalid nonnegative finite ($name)\n" }
    if $number < 0 or $number >= 1e300 { s_fail $"invalid nonnegative finite ($name)\n" }
    $number
}

def s_stats [values: list] {
    let sorted = ($values | sort)
    let count = ($sorted | length)
    s_ensure ($count > 0) "empty distribution"
    let middle = (($count / 2) | into int)
    let median_value = if ($count mod 2) == 1 {
        $sorted | get $middle
    } else {
        (($sorted | get ($middle - 1)) + ($sorted | get $middle)) / 2.0
    }
    let p95_index = (((($count * 0.95) | math ceil) | into int) - 1)
    {
        n: $count,
        min: ($sorted | get 0),
        max: ($sorted | get ($count - 1)),
        median: $median_value,
        p95: ($sorted | get $p95_index)
    }
}

def s_canonical [value] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        mut result = {}
        for key in ($value | columns | sort) {
            $result = ($result | insert $key (s_canonical ($value | get $key)))
        }
        $result
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        $value | each { |item| s_canonical $item }
    } else {
        $value
    }
}

def s_indent [depth: int] {
    if $depth == 0 { "" } else { (1..($depth * 3) | each { " " } | str join "") }
}

def s_pretty_json [value, depth: int = 0] {
    let kind = ($value | describe)
    if $kind == "record" or ($kind | str starts-with "record<") {
        let keys = ($value | columns | sort)
        if ($keys | length) == 0 { return "{}" }
        mut entries = []
        for key in $keys {
            let json_key = ($key | to json)
            let item = s_pretty_json ($value | get $key) ($depth + 1)
            $entries = ($entries | append $"(s_indent ($depth + 1))($json_key) : ($item)")
        }
        "{\n" + ($entries | str join ",\n") + $"\n(s_indent $depth)}"
    } else if $kind == "list" or ($kind | str starts-with "list<") or ($kind | str starts-with "table<") {
        let items = ($value | each { |item| s_pretty_json $item ($depth + 1) })
        if ($items | length) == 0 { return "[]" }
        let entries = ($items | each { |item| (s_indent ($depth + 1)) + $item })
        "[\n" + ($entries | str join ",\n") + $"\n(s_indent $depth)]"
    } else if $kind == "float" and $value == ($value | math floor) {
        (($value | into int) | to json)
    } else {
        $value | to json
    }
}

def s_csv [path: string] {
    open --raw $path | from csv
}

def s_nonempty [token: int, cta: int, arm: string, nblk: int] {
    mut n = $token + 1
    if (($n / 4) | math floor) > $nblk { $n = (4 * $nblk) + ($n mod 4) }
    let chunk_size = (((($n + 8191) / 8192) | math floor) * 1024) | into int
    mut count = 0
    for unit in [$cta ($cta + 16)] {
        let chunk = if $arm == "old" { $unit mod 8 } else { (($unit / 4) | math floor) | into int }
        if ($chunk * $chunk_size) < $n { $count = $count + 1 }
    }
    $count
}

def s_raw_trace [path: string, sample: record, config: record, opname: record, phase_plan: record, ranges: record, metrics: list, columns: list] {
    let rows = (s_csv $path)
    mut expected = 0
    for die in 0..5 {
        $expected = $expected + (32 * ($config | get $"die($die)_phases"))
    }
    s_ensure (($rows | length) == $expected) $"raw trace record count ($path)"
    let first_token = $config.prompt + $config.decode_intervals - 2
    let nonce = (($sample.pair + 2) * 4) + ($sample.stage_order * 2) + $sample.order + 1
    mut index = 0
    mut derived = []
    mut next_plan = $phase_plan
    mut next_ranges = $ranges

    for die in 0..5 {
        let phase_count = ($config | get $"die($die)_phases")
        mut previous = {}
        mut cta0 = [ [] [] ]
        mut minimum = 0
        mut maximum = 0
        for token_offset in 0..1 {
            for phase in 0..($phase_count - 1) {
                for cta in 0..15 {
                    let row = ($rows | get $index)
                    $index = $index + 1
                    s_ensure ((($row | columns | sort) == ($columns | sort)) and (($row | columns | length) == ($columns | length))) $"raw trace columns ($path)"
                    s_ensure ((s_uint $row.die "die") == $die and (s_uint $row.token "token") == ($first_token + $token_offset) and (s_uint $row.phase "phase") == $phase and (s_uint $row.cta "CTA") == $cta and (s_uint $row.nonce "nonce") == $nonce) $"raw trace identity/nonce ($path)"
                    let opcode = (s_uint $row.opcode "opcode")
                    s_ensure (($opname | columns | any { |key| $key == ($opcode | into string) }) and $row.op == ($opname | get ($opcode | into string))) $"raw trace opcode ($path)"
                    let plan_key = $"($die):($phase)"
                    if not (($next_plan | columns | any { |key| $key == $plan_key })) {
                        $next_plan = ($next_plan | insert $plan_key $opcode)
                    }
                    s_ensure (($next_plan | get $plan_key) == $opcode) $"raw trace phase plan changed ($path)"
                    let work = if $row.op == "attnpartn" { s_nonempty ($first_token + $token_offset) $cta $sample.arm $config.nblk } else { 0 }
                    s_ensure ((s_uint $row.nonempty_units "work") == $work) $"raw trace work count ($path)"
                    let timestamp_names = ["wait_start_ns" "wait_end_ns" "body_start_ns" "body_end_ns" "phase_end_ns"]
                    mut timestamps = []
                    for name in $timestamp_names {
                        $timestamps = ($timestamps | append (s_uint ($row | get $name) $"timestamp ($name)" 9223372036854775807))
                    }
                    s_ensure (($timestamps | get 0) > 0 and ($timestamps | get 4) > ($timestamps | get 0)) $"raw trace empty timestamps ($path)"
                    s_ensure (($timestamps | window 2 | all { |pair| ($pair | get 1) >= ($pair | get 0) })) $"raw trace timestamp order ($path)"
                    let cta_key = $cta | into string
                    if ($previous | columns | any { |key| $key == $cta_key }) {
                        s_ensure (($timestamps | get 0) >= ($previous | get $cta_key)) $"raw trace phase/token order ($path)"
                    }
                    $previous = ($previous | upsert $cta_key ($timestamps | get 4))
                    if $minimum == 0 or ($timestamps | get 0) < $minimum { $minimum = ($timestamps | get 0) }
                    if ($timestamps | get 4) > $maximum { $maximum = ($timestamps | get 4) }
                    if $cta == 0 {
                        let old = ($cta0 | get $token_offset)
                        let entry = { op: $row.op, ts: $timestamps }
                        $cta0 = ($cta0 | update $token_offset ($old | append $entry))
                    }
                }
            }
        }
        let range_key = $"($nonce):($die)"
        s_ensure (not ($next_ranges | columns | any { |key| $key == $range_key })) $"duplicate launch nonce ($path)"
        $next_ranges = ($next_ranges | insert $range_key [$minimum $maximum])
        let token0 = ($cta0 | get 0)
        let token1 = ($cta0 | get 1)
        let begin = (($token0 | get (($token0 | length) - 1)).ts | get 4)
        let end = (($token1 | get (($token1 | length) - 1)).ts | get 4)
        s_ensure ($end > $begin) $"raw trace decode interval ($path)"
        mut value = { die: $die, device_decode_us: (($end - $begin) * 0.001) }
        for metric in $metrics { $value = ($value | insert $metric 0) }
        mut covered = 0
        for phase in 0..($phase_count - 1) {
            let row = ($token1 | get $phase)
            let ts = $row.ts
            let category = if $row.op == "attnpartn" or $row.op == "attnmerge" {
                0
            } else if $row.op == "kvgather" {
                1
            } else if $row.op == "idxqk" or $row.op == "idxscore" or $row.op == "idxsel" {
                2
            } else {
                3
            }
            let body_metric = ($metrics | get $category)
            $value = ($value | update $body_metric (($value | get $body_metric) + (($ts | get 3) - ($ts | get 2)) * 0.001))
            $value = ($value | update exchange_wait_us ($value.exchange_wait_us + (($ts | get 1) - ($ts | get 0)) * 0.001))
            $value = ($value | update dispatch_us ($value.dispatch_us + (($ts | get 2) - ($ts | get 1)) * 0.001))
            $value = ($value | update grid_wait_us ($value.grid_wait_us + (($ts | get 4) - ($ts | get 3)) * 0.001))
            $covered = $covered + (($ts | get 4) - ($ts | get 0))
        }
        s_ensure ($covered <= ($end - $begin)) $"raw trace interval reconciliation ($path)"
        $value = ($value | update transition_us ((($end - $begin - $covered) * 0.001)))
        $derived = ($derived | append $value)
    }
    { derived: $derived, phase_plan: $next_plan, ranges: $next_ranges, count: ($rows | length) }
}

def main [...argv: string] {
    if ($argv | length) != 1 { s_fail "usage: kern/cta_remap_summary.pl WHOLE_MODEL_OUTPUT_DIRECTORY\n" }
    let dir = ($argv | get 0)
    let state = try { open --raw $"($dir)/status.txt" } catch { s_fail $"missing status: no such file or directory\n" }
    s_ensure ($state == "PASS\n") "run is not PASS"
    let config_text = try { open --raw $"($dir)/config.txt" } catch { s_fail $"missing configuration: no such file or directory\n" }
    mut config = {}
    for line in ($config_text | lines) {
        let parts = ($line | split row "=")
        s_ensure (($parts | length) >= 2 and (($parts | get 0) | str length) > 0) "malformed/duplicate configuration"
        let key = ($parts | get 0)
        let value = ($parts | skip 1 | str join "=")
        s_ensure (not (($config | columns) | any { |seen| $seen == $key })) "malformed/duplicate configuration"
        $config = ($config | insert $key $value)
    }
    for key in ["pairs" "decode_intervals" "dies" "layers" "nblk" "prompt"] {
        $config = ($config | update $key (s_uint ($config | get $key) $"configuration ($key)"))
    }
    s_ensure ($config.pairs >= 2 and $config.pairs <= 100 and ($config.pairs mod 2) == 0 and $config.dies == 6 and $config.layers == 48 and $config.nblk == 512 and $config.prompt >= 2 and $config.decode_intervals >= 2 and $config.decode_intervals <= 1024) "insufficient whole-model coverage"
    for die in 0..5 {
        let key = $"die($die)_phases"
        let count = s_uint ($config | get $key) $"die phase count" 65535
        s_ensure ($count > 0) "empty phase plan"
        $config = ($config | update $key $count)
    }

    let die_script = [$env.FILE_PWD "die.pl"] | path join
    let header_run = (^perl $die_script header | complete)
    s_ensure ($header_run.exit_code == 0) "opcode header generation failed"
    mut opname = {}
    for line in ($header_run.stdout | lines) {
        let matches = ($line | parse --regex '^#define OP_(?<name>\w+) (?<number>[0-9]+)$')
        if ($matches | length) == 1 {
            let name = ($matches | get 0 | get name)
            if $name != "COUNT" {
                $opname = ($opname | insert (($matches | get 0 | get number) | into int | into string) ($name | str lowercase))
            }
        }
    }
    s_ensure (($opname | columns | length) > 0) "empty opcode header"

    let metrics = ["attention_body_us" "gather_body_us" "indexer_body_us" "other_body_us" "exchange_wait_us" "dispatch_us" "grid_wait_us" "transition_us"]
    let columns = ["die" "token" "phase" "cta" "opcode" "op" "wait_start_ns" "wait_end_ns" "body_start_ns" "body_end_ns" "phase_end_ns" "nonempty_units" "nonce"]
    mut runs = []
    mut by = { untraced: { old: {}, new: {} }, traced: { old: {}, new: {} } }
    mut phase_plan = {}
    mut ranges = {}
    mut raw_records = 0
    mut canonical_tokens = ""

    for stage in ["untraced" "traced"] {
        for pair in (-2..($config.pairs - 1)) {
            for arm in ["old" "new"] {
                let path = $"($dir)/($stage)-($arm)-pair($pair).json"
                mut sample = try { open $path } catch { s_fail $"missing sample ($path): no such file or directory\n" }
                s_ensure (($sample.token_ids | describe | str starts-with "list") and ($sample.arrival_since_first_us | describe | str starts-with "list") and (($sample.dies | describe | str starts-with "list") or ($sample.dies | describe | str starts-with "table"))) $"bad sample structure ($path)"
                let warm = $pair < 0
                let expected_order = if ($pair mod 2) != 0 { if $arm == "new" { 1 } else { 0 } } else { if $arm == "old" { 1 } else { 0 } }
                let expected_stage_order = if ($pair mod 2) != 0 { if $stage == "traced" { 1 } else { 0 } } else { if $stage == "untraced" { 1 } else { 0 } }
                s_ensure ($sample.stage == $stage and $sample.arm == $arm and $sample.pair == $pair and $sample.warmup == $warm and $sample.token_parity == true and $sample.resumed == ($config.prompt - 1) and $sample.order == $expected_order and $sample.stage_order == $expected_stage_order and $sample.decode_intervals == $config.decode_intervals and ($sample.token_ids | length) == ($config.decode_intervals + 1) and ($sample.arrival_since_first_us | length) == ($config.decode_intervals + 1)) $"bad coverage ($path)"
                for token in $sample.token_ids { s_uint $token "generated token ID" }
                let token_key = ($sample.token_ids | each { |token| $token | into string } | str join ",")
                if $canonical_tokens == "" { $canonical_tokens = $token_key }
                s_ensure ($canonical_tokens == $token_key) $"token mismatch ($path)"
                mut times = []
                for time in $sample.arrival_since_first_us { $times = ($times | append (s_number $time "callback timestamp")) }
                s_ensure (($times | get 0) == 0) $"bad first-token origin ($path)"
                s_ensure (($times | window 2 | all { |pair| ($pair | get 1) >= ($pair | get 0) })) $"callback order ($path)"
                let mean = ($times | get (($times | length) - 1)) / $config.decode_intervals
                s_ensure ($mean > 0 and (($mean - (s_number $sample.host_decode_us "host decode time") | math abs) < 0.001)) $"host decode timing mismatch ($path)"
                if $stage == "traced" {
                    s_ensure (($sample.dies | length) == 6) $"missing per-die trace ($path)"
                    let trace = (s_raw_trace $"($dir)/($stage)-($arm)-pair($pair).trace.csv" $sample $config $opname $phase_plan $ranges $metrics $columns)
                    $phase_plan = $trace.phase_plan
                    $ranges = $trace.ranges
                    $raw_records = $raw_records + $trace.count
                    mut seen = {}
                    for die_record in $sample.dies {
                        s_ensure (($die_record | describe | str starts-with "record")) $"bad die structure ($path)"
                        let die_id = (s_uint $die_record.die "die" 5)
                        s_ensure (not (($seen | columns) | any { |key| $key == ($die_id | into string) })) $"bad die identity ($path)"
                        $seen = ($seen | insert ($die_id | into string) true)
                        for metric in ($metrics | prepend "device_decode_us") {
                            let reported = s_number ($die_record | get $metric) $metric
                            let computed = ($trace.derived | get $die_id | get $metric)
                            s_ensure ((($reported - $computed) | math abs) < 0.001) $"raw trace metric mismatch ($path) die=($die_id) metric=($metric)"
                        }
                    }
                    $sample = ($sample | update dies $trace.derived)
                } else {
                    s_ensure (($sample.dies | length) == 0) "untraced sample contains trace data"
                }
                if $pair >= 0 {
                    $runs = ($runs | append $sample)
                    let stage_record = ($by | get $stage)
                    let arm_record = ($stage_record | get $arm | insert ($pair | into string) $sample)
                    $by = ($by | update $stage ($stage_record | update $arm $arm_record))
                }
            }
        }
    }

    for die in 0..5 {
        mut last = 0
        let nonces = ($ranges | columns | each { |key| ($key | split row ":" | get 0 | into int) } | uniq | sort)
        for nonce in $nonces {
            let range = ($ranges | get $"($nonce):($die)")
            s_ensure (($range | get 0) > $last) $"raw trace stale/replayed launch die=($die) nonce=($nonce)"
            $last = ($range | get 1)
        }
    }

    mut summary = {
        configuration: $config,
        host_units: "microseconds per decode interval",
        host_definition: "last minus first generated-token callback, divided by decode intervals; first token and prefill excluded",
        device_definition: "CTA0 final phase end of last decoded position minus preceding position on each die independently",
        phase_definition: "last decoded position; bodies include internal opcode waits and the local CTA drain barrier",
        raw_samples: ($runs | length),
        warmup_samples: 8,
        raw_trace_records: $raw_records,
        raw_traces_verified: true,
        untraced: {},
        traced: {}
    }
    for stage in ["untraced" "traced"] {
        for arm in ["old" "new"] {
            mut host_values = []
            for pair in 0..($config.pairs - 1) {
                $host_values = ($host_values | append (($by | get $stage | get $arm | get ($pair | into string)).host_decode_us))
            }
            let stage_record = ($summary | get $stage)
            $summary = ($summary | update $stage ($stage_record | upsert $arm { host_decode_us: (s_stats $host_values) }))
        }
        mut deltas = []
        for pair in 0..($config.pairs - 1) {
            let old = (($by | get $stage | get old | get ($pair | into string)).host_decode_us)
            let new = (($by | get $stage | get new | get ($pair | into string)).host_decode_us)
            $deltas = ($deltas | append ($new - $old))
        }
        $summary = ($summary | update $stage (($summary | get $stage) | upsert paired_host_delta_us (s_stats $deltas)))
    }
    for arm in ["old" "new"] {
        mut deltas = []
        for pair in 0..($config.pairs - 1) {
            let traced = (($by | get traced | get $arm | get ($pair | into string)).host_decode_us)
            let untraced = (($by | get untraced | get $arm | get ($pair | into string)).host_decode_us)
            $deltas = ($deltas | append ($traced - $untraced))
        }
        mut die_results = {}
        for die in 0..5 {
            mut die_values = []
            for pair in 0..($config.pairs - 1) {
                let sample = ($by | get traced | get $arm | get ($pair | into string))
                $die_values = ($die_values | append ($sample.dies | get $die))
            }
            mut die_summary = {}
            for metric in ($metrics | prepend "device_decode_us") {
                mut values = []
                for item in $die_values { $values = ($values | append ($item | get $metric)) }
                $die_summary = ($die_summary | insert $metric (s_stats $values))
            }
            $die_results = ($die_results | insert ($die | into string) $die_summary)
        }
        let traced_record = ($summary | get traced)
        let traced_arm = ($traced_record | get $arm | insert dies $die_results)
        $summary = ($summary | update traced ($traced_record | update $arm $traced_arm))
        let overhead = if ($summary | columns | any { |key| $key == "instrumentation_overhead" }) { $summary.instrumentation_overhead } else { {} }
        $summary = ($summary | upsert instrumentation_overhead ($overhead | insert $arm { paired_host_delta_us: (s_stats $deltas) }))
    }
    let json = ((s_pretty_json $summary) + "\n")
    $json | save --force --raw $"($dir)/summary.json"
    print -n $json
}
