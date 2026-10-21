#!/usr/bin/env nu

def fail [message: string] {
    error make { msg: $message }
}

def write-json [path: string value: any] {
    $value | to json --indent 2 | save --raw --force $path
}

def sha256-file [path: string] {
    let result = (run-external "sha256sum" $path | complete)
    if $result.exit_code != 0 {
        fail $"sha256sum failed for ($path): ($result.stderr)"
    }
    $result.stdout | str trim | split row " " | where {|field| $field != "" } | first
}

def run-command [program: string arguments timeout: int environment: record] {
    with-env $environment {
        run-external "/usr/bin/timeout" "--signal=TERM" "--kill-after=2" ($timeout | into string) $program ...$arguments | complete
    }
}

def make-cases [nv: int matrix: bool] {
    if $matrix {
        mut cases = []
        let values = ([$nv 128] | uniq | sort)
        for $value in $values {
            for $tail in 0..3 {
                for $pattern in ["clustered" "even" "random" "changing" "captured"] {
                    if (4 * $value + $tail) <= 262144 {
                        $cases = ($cases | append { nv: $value, tail: $tail, pattern: $pattern })
                    }
                }
            }
        }
        $cases
    } else {
        let tail = if (4 * $nv + 1) <= 262144 { 1 } else { 0 }
        [{ nv: $nv, tail: $tail, pattern: "changing" }]
    }
}

def main [
    --binary: string = "./qsa_gather_bench"
    --out: string
    --nv: int = 5000
    --iterations: int = 32
    --seed: int = 4242
    --matrix
] {
    if ($nv < 1) or ($nv > 65536) or ($iterations < 1) or ($iterations > 10000) or ($seed < 0) or ($seed > 4294967295) {
        fail "invalid nv/iterations/seed"
    }

    let binary = ($binary | path expand)
    let out = ($out | path expand)
    let cubin_input = ($env.DIE_CUBIN? | default "die.cubin")
    let cubin = ($cubin_input | path expand)
    if not ($binary | path exists) { fail $"benchmark binary does not exist: ($binary)" }
    if not ($cubin | path exists) { fail $"DIE_CUBIN does not exist: ($cubin)" }
    if ($out | path exists) { fail $"output already exists: ($out)" }
    mkdir ($out | path dirname) | ignore
    mkdir $out | ignore

    let here = $env.FILE_PWD
    let nu_bin = ($env.NU_BIN? | default ($env.HOME | path join ".local" "bin" "nu"))
    let capture_helper = ($here | path join "write_qsa_gather_ids.nu" | path expand)
    let files = [
        $binary
        $cubin
        $capture_helper
        ($here | path join "qsa_gather_bench.c" | path expand)
        ($here | path join "qsa_test.c" | path expand)
        ($here | path join ".." "ref" "qsa.c" | path expand)
        ($here | path join ".." "ref" "attn.c" | path expand)
        ($here | path join ".." "ref" "fmath.c" | path expand)
    ]
    mut identities = {}
    for $file in $files {
        $identities = ($identities | upsert $file (sha256-file $file))
    }

    let smi = (run-command "nvidia-smi" [
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.free"
        "--format=csv"
    ] 30 {})
    ($smi.stdout + $smi.stderr) | save --raw --force ($out | path join "nvidia-smi.txt")

    let visible_devices = ($env.CUDA_VISIBLE_DEVICES? | default null)
    mut report = {
        identities_sha256: $identities
        CUDA_VISIBLE_DEVICES: $visible_devices
        seed: $seed
        iterations: $iterations
        hardware_counter_gate: "not_run"
        true_dram_bandwidth_validated: false
        pairs: []
    }
    write-json ($out | path join "provenance.json") $report

    let cases = (make-cases $nv $matrix)
    for $case in $cases {
        let value = $case.nv
        let tail = $case.tail
        let pattern = $case.pattern
        let count = if $value < 512 { $value } else { 512 }
        let tag = $"nv($value)-tail($tail)-($pattern)"
        let capture = if $pattern == "captured" {
            let path = ($out | path join $"($tag).ids.u32")
            let generated = (run-command $nu_bin [
                "--no-config-file"
                $capture_helper
                ($value | into string)
                ($count | into string)
                ($iterations | into string)
                ($seed | into string)
                $path
            ] 180 {})
            if $generated.exit_code != 0 {
                fail $"captured ID generator failed exit=($generated.exit_code): ($generated.stderr)"
            }
            $path
        } else {
            ""
        }

        mut arms = {}
        mut rows_by_host = {}
        for $host in [0 1] {
            let host_text = ($host | into string)
            let raw = ($out | path join $"($tag)-host($host).csv")
            let command = [
                $binary
                ($value | into string)
                ($tail | into string)
                ($iterations | into string)
                $pattern
                $host_text
                $raw
                ($seed | into string)
            ]
            let child_environment = { DIE_CUBIN: $cubin, GATHER_IDS: $capture }
            let result = (run-command $binary ($command | skip 1) 180 $child_environment)
            ($result.stdout + $result.stderr) | save --raw --force ($out | path join $"($tag)-host($host).log")
            if $result.exit_code != 0 {
                fail $"benchmark failed exit=($result.exit_code): ($command | to json); see log"
            }

            let metrics = ($result.stdout | from json)
            if ($metrics.preflight_verified_all? | default false) != true {
                fail "missing preflight verification"
            }
            let csv = (open --raw $raw | from csv)
            if ($csv | length) != $iterations {
                fail "missing raw timing rows"
            }

            let arm = {
                command: $command
                result: $metrics
                raw_csv_sha256: (sha256-file $raw)
            }
            $arms = ($arms | upsert $host_text $arm)
            $rows_by_host = ($rows_by_host | upsert $host_text $csv)
        }

        let vram = ($arms | get "0" | get result)
        let host_memory = ($arms | get "1" | get result)
        for $field in [
            "cache_fnv1a64"
            "selections_fnv1a64"
            "logical_keys"
            "physical_gathered_slots"
            "selection_changes"
            "verified_output_signature_changes"
        ] {
            if ($vram | get $field) != ($host_memory | get $field) {
                fail $"host/VRAM input or output identity differs: ($field)"
            }
        }

        let vram_rows = ($rows_by_host | get "0")
        let host_rows = ($rows_by_host | get "1")
        for $index in 0..<$iterations {
            let left = ($vram_rows | get $index)
            let right = ($host_rows | get $index)
            for $field in ["iteration" "selection_fnv1a64" "verified_output_fnv1a64"] {
                if ($left | get $field) != ($right | get $field) {
                    fail $"raw pair mismatch: ($field)"
                }
            }
        }

        let denominator = $vram.gpu_us_p50
        let ratio = if $denominator == 0 { null } else { $host_memory.gpu_us_p50 / $denominator }
        let pair = {
            case: $tag
            arms: $arms
            host_to_vram_p50_ratio: $ratio
        }
        let final_pair = if $pattern == "captured" {
            $pair
            | upsert captured_ids_sha256 (sha256-file $capture)
            | upsert captured_ids_origin "generated fixture, not model capture"
            | upsert captured_ids_generator "lcg32-1664525-1013904223-partial-fisher-yates-v1"
        } else {
            $pair
        }
        $report = ($report | update pairs ($report.pairs | append $final_pair))
        write-json ($out | path join "report.json") $report
        print $"($tag) verified ($ratio)"
    }
}
