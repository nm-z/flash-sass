#!/usr/bin/env nu
const USAGE = "Usage:
  perl kern/run_qsa_model_matrix.pl --repo ROOT --out NEW_DIRECTORY
    --cache TENSOR_CACHE --ids IDS --tokens N --blocks FIRST:END
    --flags eh --manifest MANIFEST [--manifest MANIFEST ...]
    --cuda-visible 0,1,2,3,4,5
    [--budgets 1,512] [--kv 0,1] [--modes token,loop]
    [--controls all|none|layer,logits,dn,conv,ple,selection]
    [--holder-pid PID ...] [--lock-fd 9]
    [--weight-manifest FILE ...] [--build-log FILE]
    [--min-free-mib 1024] [--min-host-mib 1024]
    [--wait-seconds 300] [--timeout-seconds 600]
    [--plan]

  perl kern/run_qsa_model_matrix.pl --self-test

Flags contain e/h, or '-' for neither. Blocks use an exclusive end.
The first N IDs are used; extra IDs are permitted and ignored.
Inherited ATTN_MAP, ATTN_DIE, SPLIT_SHARES, MODEL_PLE and MODEL_PLE_OFF are preserved.
QSA_MODEL_MANIFEST and QSA_MODEL_HOLDER_PID supply defaults for one holder.

--plan prints the requested cases without CUDA or nvidia-smi access.
An inherited --lock-fd must refer to the fixed lock file and admit LOCK_EX.
The runner never starts, stops or reconfigures weight holders or an engine.
Minimum-memory options are admission thresholds, not allocation guarantees.
Successful results mean only that every selected case passed.
"

const LOCK = '/tmp/flash-sass-engine-validation.lock'
const SCRIPT_PATH = path self

def fail [message: string, code: int] {
    print -e -n $message
    exit $code
}

def absolutize [path: string] {
    $path | path expand
}

def as-int [value: string, label: string] {
    if ($value !~ '^\d+$') { error make {msg: $"invalid ($label)"} }
    $value | into int
}

def numbers [text: string, low: int, high: int, label: string] {
    let raw = $text | split row ','
    if ($raw | is-empty) { error make {msg: $"invalid ($label)"} }
    mut out = []
    for value in $raw {
        if ($value !~ '^\d+$') { error make {msg: $"invalid ($label): ($value)"} }
        let number = $value | into int
        if ($number < $low or $number > $high) { error make {msg: $"invalid ($label): ($value)"} }
        if ($out | any {|seen| $seen == $number }) { error make {msg: $"duplicate ($label): ($number)"} }
        $out ++= [$number]
    }
    $out
}

def n-new [request: record] {
    try { $request.n_new } catch { 0 }
}

def make-cases [o: record] {
    mut injections = ['layer']
    if ($o.flags | str contains 'h') { $injections ++= ['logits'] }
    let blocks = 0..($o.first - 1)
    let actual_blocks = $o.first..($o.end - 1)
    let has_dn = ($actual_blocks | where {|block| $block mod 4 != 3 } | length) > 0
    if $has_dn { $injections ++= ['dn', 'conv'] }
    if ($o.first <= 1 and $o.end > 1) { $injections ++= ['ple'] }
    mut cases = []
    for budget in $o.budget_list {
        mut controls = $injections
        if ($o.tokens >= 4 * ($budget + 1)) { $controls ++= ['selection'] }
        if $o.controls == 'none' {
            $controls = []
        } else if $o.controls != 'all' {
            $controls = $o.controls | split row ','
            for control in $controls {
                if not ($injections | append (if ($o.tokens >= 4 * ($budget + 1)) { 'selection' } else { '' }) | any {|reachable| $reachable == $control }) {
                    error make {msg: $"unreachable control ($control) at budget ($budget)"}
                }
            }
        }
        for kv in $o.kv_list {
            for mode in $o.mode_list {
                for injection in ([''] | append $controls) {
                    let flags0 = if $o.flags == '-' { '' } else { $o.flags }
                    let flags1 = if $mode == 'loop' { $"($flags0)l" } else { $flags0 }
                    let flags = if ($flags1 | is-empty) { '-' } else { $flags1 }
                    let env0 = $o.base_env
                    let env1 = $env0 | upsert MODEL_ATTN 'qsa' | upsert MODEL_IDS $o.ids
                    let env2 = $env1 | upsert FLASH_NBLK ($budget | into string)
                    let env3 = $env2 | upsert FLASH_KV_HOST ($kv | into string)
                    let environment = $env3 | upsert CUDA_VISIBLE_DEVICES $o.visible
                    let final_env = if ($injection | is-empty) { $environment } else { $environment | upsert MODEL_INJECT $injection }
                    let attention_layers = ($actual_blocks | where {|block| $block mod 4 == 3 } | length)
                    let expected_exit = if ($injection | is-empty) { 0 } else { 3 }
                    let name = [$"b($budget)" $"kv($kv)" $mode (if ($injection | is-empty) { 'positive' } else { $injection })] | str join '-'
                    let argv = [$"($o.repo)/kern/model_test" $o.cache ($o.first | into string) ($o.end | into string) ($o.tokens | into string) $flags] | append $o.manifests
                    $cases ++= [{
                        name: $name,
                        budget: $budget,
                        kv_host: $kv,
                        mode: $mode,
                        attention_layers: $attention_layers,
                        dies: ($o.manifests | length),
                        injection: $injection,
                        expected_exit: $expected_exit,
                        argv: $argv,
                        environment: $final_env,
                        cwd: $"($o.repo)/kern",
                    }]
                }
            }
        }
    }
    $cases
}

def has-regex [text: string, pattern: string] {
    $text =~ $pattern
}

def check-output [case: record, result: record, output: string, tokens: int] {
    if $result.timed_out { return [false 'child timed out'] }
    if $result.signal != 0 { return [false 'child terminated by signal'] }
    if $result.exit_code != $case.expected_exit { return [false $"exit ($result.exit_code); expected ($case.expected_exit)"] }
    if not (has-regex $output '(?m)^attention mode: qsa \(production INT8 QSA;') { return [false 'production QSA mode banner is absent'] }
    let effective_pattern = '(?m)^FLASH_NBLK=' + ($case.budget | into string) + ' FLASH_KV_HOST=' + ($case.kv_host | into string) + ';'
    if not (has-regex $output $effective_pattern) { return [false 'effective budget or KV placement differs from the case'] }
    let assertion_lines = $output | lines | where {|line| $line =~ '^\s+die \d+ asserted QSA opcodes and FLASH_NBLK=\d+ for \d+ layer\(s\)$' }
    let counts = $assertion_lines | each {|line| (($line | parse '  die {die} asserted QSA opcodes and FLASH_NBLK={budget} for {layers} layer(s)') | get 0 | get layers | into int) }
    let total_layers = $counts | math sum
    if (($assertion_lines | length) != $case.dies or $total_layers != $case.attention_layers) { return [false 'QSA opcode assertions do not cover the requested attention layers/dies'] }
    if ($case.injection | is-not-empty) {
        let injection_pattern = 'MODEL_INJECT=' + $case.injection + '[^\n]*\breached\b'
        if not (has-regex $output $injection_pattern) { return [false 'injection was not explicitly reached'] }
        if not ((has-regex $output '(?m)\bMISMATCH\b') and (has-regex $output '(?m)^FAIL\b')) { return [false 'expected comparison failure is absent'] }
        return [true 'injection reached and mismatch rejected']
    }
    if (has-regex $output '(?m)\bMISMATCH\b|^FAIL\b') { return [false 'positive run contains a mismatch'] }
    if $case.mode == 'token' {
        let token_pattern = '(?m)^OK:\s+' + ($tokens | into string) + ' of ' + ($tokens | into string) + ' tokens exact\b'
        if not (has-regex $output $token_pattern) { return [false 'complete per-token comparison summary is absent'] }
    } else if not (has-regex $output '(?m)^OK:\s+one launch\b') { return [false 'loop comparison summary is absent'] }
    [true 'requested positive comparison completed']
}

def json-text [value: any] {
    let formatted = ($value | to json | ^jq -S --indent 3 | complete)
    if $formatted.exit_code != 0 { fail $formatted.stderr 1 }
    let text = $formatted.stdout | lines | each {|line| $line | str replace --regex '^(\s*".*"):' '$1 :' } | str join "\n"
    $text + "\n"
}

def json-output [value: any] {
    print -n (json-text $value)
}

def self-test [] {
    let scope = {budget: 1, kv_host: 1, attention_layers: 2, dies: 1}
    let positive = $scope | merge {mode: 'token', injection: '', expected_exit: 0}
    let negative = $scope | merge {mode: 'token', injection: 'layer', expected_exit: 3}
    let banner = "attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=1 FLASH_KV_HOST=1; cache positions=128 compact positions=2080\n  die 0 asserted QSA opcodes and FLASH_NBLK=1 for 2 layer(s)\n"
    let tests = [
        {case: $positive, output: "OK: 9 of 9 tokens exact\n", exit_code: 0, expected: true},
        {case: $positive, output: "OK: 8 of 9 tokens exact\n", exit_code: 0, expected: false},
        {case: $positive, output: "OK: 9 of 9 tokens run\n", exit_code: 0, expected: false},
        {case: $positive, output: "OK: 9 of 9 tokens exact\n", exit_code: 3, expected: false},
        {case: $negative, output: "MODEL_INJECT=layer reached\nMISMATCH\nFAIL\n", exit_code: 3, expected: true},
        {case: $negative, output: "MISMATCH\nFAIL\n", exit_code: 3, expected: false},
        {case: $negative, output: "MODEL_INJECT=layer reached\nMISMATCH\nFAIL\n", exit_code: 0, expected: false},
        {case: $negative, output: "MODEL_INJECT=selection reached\nMISMATCH\nFAIL\n", exit_code: 3, expected: false},
    ]
    mut failed = 0
    for index in 0..7 {
        let test = $tests | get $index
        let output = $banner + $test.output
        let observed = (check-output $test.case {exit_code: $test.exit_code, signal: 0, timed_out: false} $output 9 | get 0)
        if $observed != $test.expected {
            print -e $"self-test ($index) failed"
            $failed += 1
        }
    }
    let stale = [
        'attention mode: dense\nOK: 9 of 9 tokens exact\n',
        'attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=512 FLASH_KV_HOST=1;\n  die 0 asserted QSA opcodes and FLASH_NBLK=512 for 2 layer(s)\nOK: 9 of 9 tokens exact\n',
        'attention mode: qsa (production INT8 QSA; instruction-order twin)\nFLASH_NBLK=1 FLASH_KV_HOST=1;\n  die 0 asserted QSA opcodes and FLASH_NBLK=1 for 1 layer(s)\nOK: 9 of 9 tokens exact\n',
    ]
    for stale_output in $stale {
        if (check-output $positive {exit_code: 0, signal: 0, timed_out: false} $stale_output 9 | get 0) { $failed += 1 }
    }
    let o = {
        first: 0, end: 8, flags: 'eh', tokens: 9, controls: 'all',
        budget_list: [1, 512], kv_list: [0, 1], mode_list: ['token', 'loop'], base_env: {},
        ids: 'ids', visible: '7', repo: '/repo', cache: '/cache', manifests: ['/manifest'],
    }
    let cases = make-cases $o
    let selections = $cases | where {|case| $case.injection == 'selection' }
    if (($selections | length) != 4 or ($selections | any {|case| $case.budget != 1 })) { $failed += 1 }
    let no_controls = make-cases ($o | upsert controls 'none')
    if ($no_controls | length) != 8 { $failed += 1 }
    let unreachable = try { make-cases ($o | upsert controls 'selection'); false } catch { true }
    if not $unreachable { $failed += 1 }
    if $failed == 0 {
        print 'OK: 8 child-result checks, 3 stale/scope checks and selection/control planning'
        return 0
    }
    print -e $"FAIL: ($failed) CPU runner checks"
    1
}

def save-text [path: string, value: string] {
    $value | save --force --raw $path
}

def epoch-ns [] {
    let r = (^date '+%s%N' | complete)
    if $r.exit_code != 0 { error make {msg: 'read clock'} }
    $r.stdout | str trim | into int
}

def identity [path: string] {
    let full = $path | path expand
    let sum = (^sha256sum $full | complete)
    if $sum.exit_code != 0 { error make {msg: $'hash ($full): ($sum.stderr)'} }
    let stat = (^stat -c '%s %Y' $full | complete)
    if $stat.exit_code != 0 { error make {msg: $'stat ($full): ($stat.stderr)'} }
    let size_mtime = $stat.stdout | str trim | split row ' '
    {
        path: $full,
        bytes: ($size_mtime | get 0 | into int),
        mtime: ($size_mtime | get 1 | into int),
        sha256: ($sum.stdout | split row ' ' | get 0),
    }
}

def checked-capture [argv: list<string>, cwd: string, prefix: string] {
    let program = $argv | get 0
    let child_args = $argv | skip 1
    let result = do {
        cd $cwd
        ^$program ...$child_args | complete
    }
    save-text ($prefix + '.stdout') $result.stdout
    save-text ($prefix + '.stderr') $result.stderr
    if ($result.exit_code != 0) { error make {msg: $'inventory command failed; see ($prefix).stderr'} }
    $result.stdout
}

def holder-allowed [pid: int, o: record] {
    if $pid not-in $o.holders { return false }
    let exe_result = (^readlink $'/proc/($pid)/exe' | complete)
    if ($exe_result.exit_code != 0) { return false }
    let exe = $exe_result.stdout | str trim | path basename
    if $exe != 'hold' { return false }
    let args = try { (open --raw $'/proc/($pid)/cmdline') | split row "\u{0000}" } catch { [] }
    ($o.manifests | any {|manifest| $manifest in $args })
}

def preflight [o: record, prefix: string] {
    checked-capture ['ps', '-eo', 'pid=,ppid=,etimes=,stat=,comm='] $o.repo ($prefix + '.ps') | ignore
    let gpu = checked-capture [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ] $o.repo ($prefix + '.gpu')
    let apps = checked-capture [
        'nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ] $o.repo ($prefix + '.apps')
    mut seen = []
    mut uuids = []
    for line in ($gpu | lines) {
        let fields = $line | split row ',' | each {|field| $field | str trim }
        if ($fields | length) != 6 { continue }
        let index = try { $fields | get 0 | into int } catch { continue }
        if $index not-in $o.visible_list { continue }
        $seen ++= [$index]
        $uuids ++= [$fields.1]
        if (($fields.4 !~ '^\d+$') or ($fields.5 !~ '^\d+$')) { return $'GPU ($index) memory/utilization unavailable' }
        let free = $fields.4 | into int
        let utilization = $fields.5 | into int
        if $free < $o.min_free_mib { return $'GPU ($index) has ($free) MiB free; need ($o.min_free_mib)' }
        if $utilization != 0 { return $'GPU ($index) utilization is ($utilization)%' }
    }
    for index in $o.visible_list {
        if $index not-in $seen { return $'GPU index ($index) was not reported' }
    }
    for line in ($apps | lines) {
        let fields = $line | split row ',' | each {|field| $field | str trim }
        if ($fields | length) < 4 { continue }
        let uuid = $fields | get 0
        if $uuid not-in $uuids { continue }
        let pid = try { $fields | get 1 | into int } catch { return $'unrecognized GPU process row: ($line)' }
        if not (holder-allowed $pid $o) { return $'GPU process ($pid) is not an allowed holder' }
    }
    let mem = try { open --raw '/proc/meminfo' } catch { '' }
    let available_line = $mem | lines | where {|line| $line =~ '^MemAvailable:\s+\d+\s+kB$' } | first
    if $available_line == null { return 'host MemAvailable unavailable' }
    let available = ($available_line | parse 'MemAvailable: {kb} kB' | get 0 | get kb | into int)
    if $available < 1024 * $o.min_host_mib { return $'host memory below ($o.min_host_mib) MiB' }
    ''
}

def wait-admission [o: record, name: string] {
    let started = epoch-ns
    mut attempt = 0
    while true {
        let reason = preflight $o ($o.out | path join $'($name).preflight.($attempt)')
        $attempt += 1
        if ($reason | is-empty) { return }
        let elapsed = (epoch-ns) - $started
        if $elapsed >= ($o.wait_seconds * 1000000000) { error make {msg: $'BLOCKED: ($reason)'} }
        print -e $'waiting: ($reason)'
        ^sleep 1
    }
}

def run-child [case: record, o: record, prefix: string] {
    let started = epoch-ns
    let keys = $env | columns | where {|key| ($key !~ '^(MODEL_|FLASH_)') and ($key not-in ['ATTN_MAP', 'ATTN_DIE', 'SPLIT_SHARES', 'CUDA_VISIBLE_DEVICES']) }
    let inherited = $keys | each {|key| $'($key)=(($env | get $key) | into string)' }
    let case_keys = $case.environment | columns
    let overrides = $case_keys | each {|key| $'($key)=(($case.environment | get $key) | into string)' }
    let binary = $case.argv | get 0
    let child_args = $case.argv | skip 1
    let duration = $'($o.timeout_seconds)s'
    let result = do {
        cd $case.cwd
        ^timeout --signal=TERM --kill-after=2 $duration ^env -i ...$inherited ...$overrides -- $binary ...$child_args | complete
    }
    save-text ($prefix + '.stdout') $result.stdout
    save-text ($prefix + '.stderr') $result.stderr
    let elapsed_ns = (epoch-ns) - $started
    {
        argv: $case.argv,
        cwd: $case.cwd,
        stdout: ($prefix + '.stdout'),
        stderr: ($prefix + '.stderr'),
        exit_code: $result.exit_code,
        signal: (if $result.exit_code >= 128 { $result.exit_code - 128 } else { 0 }),
        timed_out: ($result.exit_code == 124),
        elapsed_seconds: ($elapsed_ns / 1000000000),
        stdout_text: $result.stdout,
        stderr_text: $result.stderr,
    }
}

def run-matrix [o: record, raw: record, args: list<string>] {
    if ($raw.out | is-empty) { fail "--out NEW_DIRECTORY is required\n" 255 }
    if ((not $raw.locked) and ($raw.lock_fd | is-empty)) {
        mut waiting = true
        while $waiting {
            let child_args = ['--', '--_locked'] | append $args
            let result = (^flock --nonblock --conflict-exit-code 73 $LOCK -- $nu.current-exe $SCRIPT_PATH ...$child_args | complete)
            if $result.exit_code == 73 {
                print -e 'waiting: shared validation lock is held'
                ^sleep 1
            } else {
                print -n $result.stdout
                print -e -n $result.stderr
                exit $result.exit_code
            }
        }
        return
    }
    execute-matrix $o $raw
}

def execute-matrix [o: record, raw: record] {
    let out = $raw.out | path expand
    if ($out | path exists) { fail "create fresh output directory: File exists\n" 255 }
    mkdir $out | ignore
    let result_file = $out | path join 'results.json'
    let input = try { open --raw $o.ids } catch { fail $'open ($o.ids) failed\n' 255 }
    let input_ids = $input | str replace --regex '\s+' ' ' | split row ' ' | where {|id| $id | is-not-empty }
    if ($input_ids | length) < $o.tokens { fail $'need at least ($o.tokens) token IDs\n' 1 }
    let effective = $input_ids | take $o.tokens
    if ($effective | any {|id| ($id !~ '^\d+$') or ($id | into int) >= 248320 }) { fail "invalid token ID\n" 1 }
    let ids_text = ($effective | str join "\n") + "\n"
    let ids_path = $out | path join 'ids.txt'
    save-text $ids_path $ids_text
    let o = $o | upsert ids $ids_path
    let cases = make-cases $o
    let input_identity = identity $raw.ids
    let effective_identity = identity $ids_path
    mut files = [
        ($o.repo | path join 'kern/model_test'),
        ($o.repo | path join 'kern/die.cubin'),
    ] | append $o.manifests | append $o.weight_manifests
    let engine = $o.repo | path join 'kern/libengine.so'
    if ($engine | path exists) { $files ++= [$engine] }
    if ($o.build_log | is-not-empty) { $files ++= [$o.build_log] }
    $files ++= [$SCRIPT_PATH]
    let file_identities = $files | each {|path| identity $path }
    let head = checked-capture ['git', 'rev-parse', 'HEAD'] $o.repo ($out | path join 'source-head')
    let status = checked-capture ['git', 'status', '--porcelain'] $o.repo ($out | path join 'source-status')
    checked-capture ['git', 'diff', '--binary', 'HEAD'] $o.repo ($out | path join 'source-diff') | ignore
    let source_diff = identity ($out | path join 'source-diff.stdout')
    mut report = {
        schema: 'flash-sass.qsa-model-matrix.v1',
        status: 'RUNNING', acceptance_complete: false,
        scope: {
            blocks: [$o.first, $o.end], tokens: $o.tokens, flags: $raw.flags,
            cuda_visible_devices: $o.visible, minimum_gpu_free_mib: $o.min_free_mib,
            minimum_host_available_mib: $o.min_host_mib,
        },
        remaining_scope: $o.remaining_scope,
        plan: $cases, cases: [],
        input_ids: $input_identity, effective_ids: $effective_identity,
        file_identities: $file_identities, cache_directory: $o.cache,
        git_head: $head, git_status: $status, source_diff: $source_diff,
        source_identity_note: 'Repository snapshot plus exact binary hashes; use the build log to link them. The model_test executable does not load libengine.so.',
    }
    save-text $result_file (json-text $report)
    mut failure: any = null
    for case in $cases {
        print -e $'case ($case.name)'
        let attempt = try {
            wait-admission $o $case.name
            let prefix = $out | path join $case.name
            let result = run-child $case $o $prefix
            let combined = $result.stdout_text + $result.stderr_text
            let checked = check-output $case {
                exit_code: $result.exit_code, signal: $result.signal,
                timed_out: $result.timed_out,
            } $combined $o.tokens
            let passed = $checked | get 0
            let reason = $checked | get 1
            let case_report = $result
                | reject stdout_text stderr_text
                | upsert name $case.name
                | upsert environment $case.environment
                | upsert expected_exit $case.expected_exit
                | upsert passed $passed
                | upsert reason $reason
                | upsert stdout_identity (identity $result.stdout)
                | upsert stderr_identity (identity $result.stderr)
            $report = $report | update cases ($report.cases | append $case_report)
            save-text $result_file (json-text $report)
            if not $passed { error make {msg: $'case ($case.name): ($reason)'} }
            {error: null}
        } catch {|err| {error: $err.msg}}
        if $attempt.error != null { $failure = $attempt.error; break }
    }
    if $failure == null {
        $report = $report | upsert status 'SELECTED_CASES_PASSED'
    } else {
        let status = if ($failure | str starts-with 'BLOCKED:') { 'BLOCKED' } else { 'FAILED' }
        $report = $report | upsert status $status | upsert error ($failure + "\n")
    }
    $report = $report | upsert finished_epoch ((epoch-ns) / 1000000000)
    save-text $result_file (json-text $report)
    print $'($report.status): ($result_file)'
    if $report.error? != null { print -e -n $report.error }
    if $report.status == 'SELECTED_CASES_PASSED' { return }
    exit 1
}

def parse-args [args: list<string>] {
    mut o = {
        repo: '', out: '', cache: '', ids: '',
        tokens: '9', blocks: '0:48', flags: 'eh', controls: 'all',
        budgets: '1,512', kv: '0,1', modes: 'token,loop', visible: (try { $env.CUDA_VISIBLE_DEVICES } catch { '' }),
        min_free_mib: '1024', min_host_mib: '1024', wait_seconds: '300', timeout_seconds: '600',
        manifests: [], holders: [], weight_manifests: [], build_log: '',
        lock_fd: '', plan: false, self_test: false, help: false, locked: false,
    }
    mut manifests = []
    mut holders = []
    mut weight_manifests = []
    mut i = 0
    while $i < ($args | length) {
        let arg = $args | get $i
        if $arg == '--plan' { $o = $o | upsert plan true; $i += 1; continue }
        if $arg == '--self-test' { $o = $o | upsert self_test true; $i += 1; continue }
        if $arg == '--help' { $o = $o | upsert help true; $i += 1; continue }
        if $arg == '--_locked' { $o = $o | upsert locked true; $i += 1; continue }
        let split = $arg | split row '='
        let key = $split | get 0
        let inline = ($split | length) > 1
        let value = if $inline { $split | skip 1 | str join '=' } else {
            $i += 1
            if $i >= ($args | length) { return {error: 'missing option value'} }
            $args | get $i
        }
        if not ($arg | str starts-with '--') { return {error: $"unexpected arguments: ($arg)"} }
        let name = $key | str substring 2..
        if $name == 'manifest' { $manifests ++= [$value]; $i += 1; continue }
        if $name == 'holder-pid' { $holders ++= [$value]; $i += 1; continue }
        if $name == 'weight-manifest' { $weight_manifests ++= [$value]; $i += 1; continue }
        if $name in ['repo', 'out', 'cache', 'ids', 'tokens', 'blocks', 'flags', 'controls', 'budgets', 'kv', 'modes', 'cuda-visible', 'min-free-mib', 'min-host-mib', 'wait-seconds', 'timeout-seconds', 'build-log', 'lock-fd'] {
            let field = if $name == 'cuda-visible' { 'visible' } else { $name | str replace --all '-' '_' }
            $o = $o | upsert $field $value
            $i += 1
            continue
        }
        return {error: $"unknown option ($key)"}
        $i += 1
    }
    let finished = $o | upsert manifests $manifests | upsert holders $holders | upsert weight_manifests $weight_manifests
    {value: $finished}
}

def main [...args: string] {
    let parsed = parse-args $args
    if 'error' in $parsed {
        if $parsed.error == 'missing option value' { fail ("invalid option\n" + $USAGE) 255 }
        fail ($parsed.error + "\n" + $USAGE) 255
    }
    let raw0 = $parsed.value
    let manifests = if (($raw0.manifests | is-empty) and (try { $env.QSA_MODEL_MANIFEST } catch { null }) != null) { [$env.QSA_MODEL_MANIFEST] } else { $raw0.manifests }
    let holders_raw = if (($raw0.holders | is-empty) and (try { $env.QSA_MODEL_HOLDER_PID } catch { null }) != null) { [$env.QSA_MODEL_HOLDER_PID] } else { $raw0.holders }
    let raw = $raw0 | upsert manifests $manifests | upsert holders $holders_raw
    if $raw.help { print -n $USAGE; return }
    if $raw.self_test {
        let rc = self-test
        exit $rc
    }
    if (($raw.cache | is-empty) or ($raw.ids | is-empty) or ($raw.manifests | is-empty)) { fail ($USAGE) 255 }
    let block_parts = $raw.blocks | split row ':'
    if (($block_parts | length) != 2 or ($block_parts | any {|part| $part !~ '^\d+$' })) { fail "invalid block range\n" 255 }
    let first = ($block_parts | get 0 | into int)
    let end = ($block_parts | get 1 | into int)
    if ($first >= $end or $end > 48) { fail "invalid block range\n" 255 }
    let block_range = $first..($end - 1)
    if (($block_range | where {|block| $block mod 4 == 3 } | length) == 0) { fail "range contains no attention block\n" 255 }
    if $raw.flags !~ '^(?:-|e|h|eh|he)$' { fail "invalid flags\n" 255 }
    let tokens = try { as-int $raw.tokens 'token count' } catch { fail "invalid token count\n" 255 }
    if ($tokens < 1 or $tokens > 262144) { fail "invalid token count\n" 255 }
    let wait_seconds = try { as-int $raw.wait_seconds 'wait/timeout/memory threshold' } catch { fail "invalid wait/timeout/memory threshold\n" 255 }
    let timeout_seconds = try { as-int $raw.timeout_seconds 'wait/timeout/memory threshold' } catch { fail "invalid wait/timeout/memory threshold\n" 255 }
    let min_free_mib = try { as-int $raw.min_free_mib 'wait/timeout/memory threshold' } catch { fail "invalid wait/timeout/memory threshold\n" 255 }
    let min_host_mib = try { as-int $raw.min_host_mib 'wait/timeout/memory threshold' } catch { fail "invalid wait/timeout/memory threshold\n" 255 }
    if ($wait_seconds < 0 or $wait_seconds > 3600 or $timeout_seconds < 1 or $timeout_seconds > 86400 or $min_free_mib < 1 or $min_host_mib < 1) { fail "invalid wait/timeout/memory threshold\n" 255 }
    for holder in $raw.holders {
        let holder_string = $holder | into string
        if (($holder_string !~ '^\d+$') or ($holder_string | into int) < 1) { fail "invalid holder PID\n" 255 }
    }
    if (($raw.lock_fd | is-not-empty) and (try { as-int $raw.lock_fd 'inherited lock descriptor' } catch { -1 }) < 3) { fail "invalid inherited lock descriptor\n" 255 }
    if ($raw.controls not-in ['all', 'none']) {
        let controls = $raw.controls | split row ','
        if (($controls | uniq | length) != ($controls | length) or ($controls | any {|control| $control not-in ['layer', 'logits', 'dn', 'conv', 'ple', 'selection'] })) { fail "invalid or duplicate control\n" 255 }
    }
    let budget_list = try { numbers $raw.budgets 1 4096 'budget' } catch {|err| fail ($err.msg + "\n") 255 }
    let kv_list = try { numbers $raw.kv 0 1 'KV placement' } catch {|err| fail ($err.msg + "\n") 255 }
    let visible_list = try { numbers $raw.visible 0 255 'numeric CUDA visibility' } catch {|err| fail ($err.msg + "\n") 255 }
    let visible = $visible_list | each {|item| $item | into string } | str join ','
    let modes = $raw.modes | split row ','
    if (($modes | is-empty) or ($modes | any {|mode| $mode not-in ['token', 'loop'] })) { fail "invalid launch mode\n" 255 }
    if (($modes | length) > 2 or (($modes | length) == 2 and ($modes | get 0) == ($modes | get 1))) { fail "duplicate launch mode\n" 255 }
    if (($raw.manifests | length) != ($visible_list | length) or ($raw.manifests | length) > 6) { fail "manifest/device counts differ or exceed six dies\n" 255 }
    let manifests = $raw.manifests | each {|path| absolutize $path }
    let holders = $raw.holders | each {|pid| $pid | into int }
    let repo = if ($raw.repo | is-empty) { pwd | path expand } else { absolutize $raw.repo }
    let cache = absolutize $raw.cache
    let ids = absolutize $raw.ids
    mut base_env = {}
    for key in ['ATTN_MAP', 'ATTN_DIE', 'SPLIT_SHARES', 'MODEL_PLE', 'MODEL_PLE_OFF', 'CUDA_DEVICE_ORDER', 'LD_LIBRARY_PATH'] {
        let val = try { $env | get $key } catch { null }
        if $val != null { $base_env = $base_env | upsert $key ($val | into string) }
    }
    let o0 = {
        repo: $repo, out: $raw.out, cache: $cache, ids: $ids,
        tokens: $tokens, first: $first, end: $end, flags: $raw.flags,
        controls: $raw.controls, budget_list: $budget_list, kv_list: $kv_list,
        mode_list: $modes, visible: $visible, visible_list: $visible_list,
        min_free_mib: $min_free_mib, min_host_mib: $min_host_mib,
        wait_seconds: $wait_seconds, timeout_seconds: $timeout_seconds,
        manifests: $manifests, holders: $holders, weight_manifests: $raw.weight_manifests,
        build_log: $raw.build_log, lock_fd: $raw.lock_fd, base_env: $base_env,
        locked: $raw.locked,
    }
    let gaps0 = [
        'This runner does not exercise request reset or snapshot/resume.',
        'Huge-cache direct-position component validation is a separate matrix.',
        'Loop mode retains final-token layer buffers; earlier predictions have weaker coverage.',
        'Manifest fingerprints alone do not establish checkpoint conversion provenance.',
    ]
    let gaps1 = if ($first == 0 and $end == 48 and ($raw.flags | str contains 'e') and ($raw.flags | str contains 'h')) { $gaps0 } else { $gaps0 | append 'The selected block/embedding/head scope is not a complete model.' }
    let gaps = if ($tokens >= 2052 and ($budget_list | any {|budget| $budget == 512 })) { $gaps1 } else { $gaps1 | append 'Production-budget exclusion is not reached by this token count.' }
    let o = $o0 | upsert remaining_scope $gaps
    let report = {
        schema: 'flash-sass.qsa-model-matrix.v1',
        status: 'PLAN_ONLY',
        acceptance_complete: false,
        scope: {
            blocks: [$first, $end], tokens: $tokens, flags: $raw.flags,
            cuda_visible_devices: $visible, minimum_gpu_free_mib: $min_free_mib,
            minimum_host_available_mib: $min_host_mib,
        },
        remaining_scope: $gaps,
        plan: (make-cases $o),
        cases: [],
    }
    if $raw.plan { json-output $report; return }
    if ($raw.out | is-empty) { fail "--out NEW_DIRECTORY is required\n" 255 }
    run-matrix $o $raw $args
}
