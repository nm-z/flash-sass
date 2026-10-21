#!/usr/bin/env nu

const LOCK = '/tmp/flash-sass-engine-validation.lock'
const SCRIPT_DIR = path self '.'
const SCRIPT_PATH = path self
const SCRIPT_PATH = path self
const REQUEST_TEST_FILE = path self 'test_qsa_engine_requests.nu'

def fail [message: string, code: int] {
    print -e -n $message
    exit $code
}

def numbers [text: string, low: int, high: int, label: string] {
    let raw = $text | split row ','
    if ($raw | is-empty) { error make {msg: $"invalid ($label)"} }
    mut out = []
    for value in $raw {
        if ($value !~ '^\d+$') { error make {msg: $"invalid ($label): ($value)"} }
        let number = $value | into int
        if ($number < $low or $number > $high) { error make {msg: $"invalid ($label): ($value)"} }
        let previous = $out
        if ($previous | any {|seen| $seen == $number }) { error make {msg: $"duplicate ($label): ($number)"} }
        $out ++= [$number]
    }
    $out
}

def env-default [name: string, fallback: string] {
    try { $env | get $name | into string } catch { $fallback }
}

def request-scope [raw: record] {
    let visible_list = numbers $raw.visible 0 255 'numeric CUDA visibility'
    let dies = $visible_list | length
    if ($dies > 6 or ($raw.manifests | length) != $dies or ($raw.holder_pids | length) != $dies) {
        error make {msg: 'require one manifest and holder PID per visible die (at most six)'}
    }
    let manifests = $raw.manifests | each {|path| $path | path expand }
    if (($manifests | uniq | length) != ($manifests | length)) { error make {msg: 'duplicate manifest path'} }
    if ($manifests | any {|path| $path | is-empty }) { error make {msg: 'empty manifest path'} }
    let holder_pids = $raw.holder_pids | each {|pid|
        if (($pid | into string) !~ '^\d+$' or ($pid | into int) < 1) { error make {msg: 'invalid or duplicate holder PID'} }
        $pid | into int
    }
    if (($holder_pids | uniq | length) != ($holder_pids | length)) { error make {msg: 'invalid or duplicate holder PID'} }
    let split_raw = if ($raw.split_shares | is-empty) { if $dies == 1 { '1' } else { try { $env.SPLIT_SHARES } catch { null } } } else { $raw.split_shares }
    let map_raw = if ($raw.attn_map | is-empty) { if $dies == 1 { (0..11 | each { '0' } | str join ',') } else { try { $env.ATTN_MAP } catch { null } } } else { $raw.attn_map }
    if ($split_raw == null or $map_raw == null) { error make {msg: 'multi-die requests require --split-shares and --attn-map (or their environment settings)'} }
    let shares = $split_raw | split row ','
    if (($shares | length) != $dies or ($shares | any {|share| ($share !~ '^\d+$') or ($share | into int) < 1 or ($share | into int) > 1000000 })) { error make {msg: 'invalid split shares'} }
    let owners_raw = $map_raw | split row ','
    if (($owners_raw | length) != 12 or ($owners_raw | any {|owner| ($owner !~ '^\d+$') or ($owner | into int) >= $dies })) { error make {msg: 'invalid attention map'} }
    let owners = $owners_raw | each {|owner| $owner | into int }
    let shares_normal = $shares | each {|share| ($share | into int) | into string } | str join ','
    let owners_normal = $owners | each {|owner| $owner | into string } | str join ','
    let visible = $visible_list | each {|item| $item | into string } | str join ','
    let groups = ($raw.layers / 4) | math floor | into int
    let attn_dies = 0..($groups - 1) | each {|index| $owners | get $index } | uniq | sort | each {|owner| $owner | into string } | str join ','
    if ($raw.tokens < 6 or $raw.tokens > 262142) { error make {msg: '--tokens must be 6..262142'} }
    let maxt = (128 * ((($raw.tokens + 2 + 127) / 128) | math floor)) | into int
    let snapshot_list = numbers $raw.snapshots 1 ($raw.tokens - 1) 'snapshot position'
    $raw
    | upsert visible_list $visible_list
    | upsert dies $dies
    | upsert visible $visible
    | upsert manifests $manifests
    | upsert holder_pids $holder_pids
    | upsert split_shares $shares_normal
    | upsert attn_map $owners_normal
    | upsert attn_owners $owners
    | upsert attn_dies $attn_dies
    | upsert maxt $maxt
    | upsert snapshot_list $snapshot_list
}

def request-plan [snapshots: list<int>] {
    mut plan = [
        {name: 'fresh-A', prompt: 'A', resume: 0, snap: 0},
        {name: 'fresh-B', prompt: 'B', resume: 0, snap: 0},
    ]
    for snapshot in $snapshots {
        $plan ++= [
            {name: $"save-($snapshot)-A", prompt: 'A', resume: 0, snap: $snapshot},
            {name: $"resume-($snapshot)-A", prompt: 'A', resume: $snapshot, snap: 0},
            {name: $"reset-($snapshot)-B", prompt: 'B', resume: 0, snap: 0},
        ]
    }
    $plan ++= [
        {name: 'final-A', prompt: 'A', resume: 0, snap: 0},
        {name: 'generation-A', prompt: 'G', resume: 0, snap: 0, n_new: 1},
    ]
    $plan
}

def commands-for [plan: list<record>, tokens: int] {
    mut commands = ['config']
    for request in $plan {
        if $request.snap > 0 { $commands ++= [$"snap ($request.snap)"] }
        let first = if $request.prompt == 'B' { 30 } else { 10 }
        let ids = 0..($tokens - 1) | each {|index| (($first + $index) mod 248320) | into string } | str join ' '
        let n_new = try { $request.n_new } catch { 0 }
        $commands ++= [$"gen ($n_new) ($ids)" 'statehash']
    }
    $commands ++= ['quit']
    $commands
}

def expected-pairs [request: record, mode: string, tokens: int] {
    if $mode == 'scalar' { return 0 }
    let prefix = $tokens - 1
    let n_new = try { $request.n_new } catch { 0 }
    if $n_new > 0 {
        let triples = (($prefix / 3) | math floor | into int)
        let doubles = ((($prefix mod 3) / 2) | math floor | into int)
        return (2 * ($triples + $doubles))
    }
    let resume = $request.resume
    let snapshot = $request.snap
    if $snapshot > 0 {
        let before = ((($snapshot - $resume) / 2) | math floor | into int)
        let after = ((($prefix - $snapshot) / 2) | math floor | into int)
        return ((2 * $before) + (2 * $after))
    }
    let paired = ((($prefix - $resume) / 2) | math floor | into int)
    2 * $paired
}

def owned-layers [o: record, die: int] {
    0..($o.layers - 1) | where {|layer| ($layer mod 4 != 3) or ($o.attn_owners | get (($layer / 4) | math floor | into int)) == $die }
}

def parse-capture [text: string, o: record, mode: string, kv: int,
                   plan: list<record>, refs: record] {
    mut records = []
    mut config: any = null
    mut tokens: any = null
    mut current: any = null
    mut header: any = null
    mut references = $refs
    for line in ($text | lines) {
        if ($line =~ '^error:|\bMISMATCH\b') { error make {msg: $"engine reported an error: ($line)"} }
        if ($line =~ '^config ') {
            if $config != null { error make {msg: 'duplicate config response'} }
            mut fields = {}
            for item in ($line | split row ' ' | skip 1 | where {|word| $word | is-not-empty }) {
                let pair = $item | split row '='
                if ($pair | length) < 2 { error make {msg: $"malformed config field ($item)"} }
                let key = $pair | get 0
                let value = $pair | skip 1 | str join '='
                if $key in ($fields | columns) { error make {msg: $"duplicate config field ($key)"} }
                $fields = $fields | upsert $key $value
            }
            let expected = {
                rev: $o.revision, kv_host: ($kv | into string), ple_disk: '1',
                maxt: ($o.maxt | into string), layers: $"($o.layers)/48",
                nblk: ($o.nblk | into string), dies: ($o.dies | into string),
                split: $o.split_shares, attn_map: $o.attn_map, attn_dies: $o.attn_dies,
                two_position: (if $mode == 'scalar' { '0' } else { '1' }),
                prefill_chunk: (if $mode == 'scalar' { '0' } else { '3' }),
            }
            for key in ($expected | columns | sort) {
                let actual = try { $fields | get $key } catch { '<missing>' }
                if $actual != ($expected | get $key) { error make {msg: $"config ($key) mismatch: ($actual)"} }
            }
            let sha = ($line + "\n") | hash sha256
            $config = {raw: $line, fields: $fields, sha256: $sha}
            continue
        }
        if ($line =~ '^tokens:\s*') {
            if $tokens != null { error make {msg: 'unexpected duplicate tokens response'} }
            let values = $line | str replace --regex '^tokens:\s*' '' | split row ' ' | where {|value| $value | is-not-empty }
            let request = try { $plan | get ($records | length) } catch { error make {msg: 'extra token response'} }
            let n_new = try { $request.n_new } catch { 0 }
            let expected_count = if $n_new > 0 { $n_new + 1 } else { $o.tokens }
            let actual_count = $values | length
            if (($values | length) != $expected_count or ($values | any {|value| ($value !~ '^\d+$') or ($value | into int) >= 248320 })) {
                error make {msg: $"wrong number of numeric token outputs: got ($actual_count), expected ($expected_count), line=($line)"}
            }
            $tokens = $values | each {|value| $value | into int }
            continue
        }
        if ($line =~ '^time:') {
            if ($tokens == null or $current != null) { error make {msg: 'time response without tokens or before preceding statehash'} }
            let parsed = $line | parse 'time: {ms} ms for {ran} positions run ({resumed} from the snapshot; {pairs} paired positions; {chunks} prefill chunks)'
            if ($parsed | is-empty) { error make {msg: 'incomplete generation statistics'} }
            let stats = $parsed | get 0
            let request = try { $plan | get ($records | length) } catch { error make {msg: 'extra generation response'} }
            let ran = $stats.ran | into int
            let resumed = $stats.resumed | into int
            let pairs = $stats.pairs | into int
            let chunks = $stats.chunks | into int
            let n_new = try { $request.n_new } catch { 0 }
            let positions = $o.tokens + $n_new
            if ($resumed != $request.resume or $ran != ($positions - $request.resume)) { error make {msg: $"($request.name): wrong resumed/executed counts"} }
            let expected = expected-pairs $request $mode $o.tokens
            if $pairs != $expected { error make {msg: $"($request.name): paired positions ($pairs), expected ($expected)"} }
            let expected_chunks = if ($n_new > 0 and $mode == 'chunk3') { (($o.tokens - 1 + 2) / 3) | math floor | into int } else { 0 }
            if $chunks != $expected_chunks { error make {msg: $"($request.name): headless chunk count mismatch"} }
            $current = $request
            | merge {
                tokens: $tokens, time_line: $line, positions_run: $ran,
                paired_positions: $pairs, prefill_chunks: $chunks,
                layer_states: {}, ple: {}, logits: {}, statehash_header: null,
            }
            $tokens = null
            continue
        }
        if ($line =~ '^statehash positions=') {
            if ($current == null or $header != null) { error make {msg: 'unexpected/duplicate statehash header'} }
            let parsed = $line | parse 'statehash positions={positions} paired_positions={paired} prefill_chunks={chunks}'
            if ($parsed | is-empty) { error make {msg: 'unexpected/duplicate statehash header'} }
            let value = $parsed | get 0
            $header = {positions: ($value.positions | into int), paired: ($value.paired | into int), chunks: ($value.chunks | into int)}
            continue
        }
        if ($line =~ '^statehash die=\d+\s+layer=\d+\s+state=[0-9a-fA-F]{16}\s*$') {
            if ($current == null or $header == null) { error make {msg: 'state outside current statehash'} }
            let item = $line | parse 'statehash die={die} layer={layer} state={hash}' | get 0
            let die = $item.die | into int
            let layer = $item.layer | into int
            let hash = $item.hash | str lowercase
            let owner = ($o.attn_owners | get (($layer / 4) | math floor | into int))
            let layer_allowed = $layer < $o.layers and (($layer mod 4 != 3) or $owner == $die)
            let die_allowed = $die < $o.dies
            let die_key = $die | into string
            let layer_key = $layer | into string
            let layer_states = $current.layer_states
            let existing = try { $layer_states | get $die_key | get $layer_key } catch { null }
            if (not $die_allowed or not $layer_allowed or $existing != null) { error make {msg: 'unexpected or duplicate layer state'} }
            let state_for_die = try { $layer_states | get $die_key } catch { {} }
            let updated_die = $state_for_die | upsert $layer_key $hash
            let updated_states = $layer_states | upsert $die_key $updated_die
            $current = $current | upsert layer_states $updated_states
            continue
        }
        if ($line =~ '^statehash die=\d+\s+ple=[0-9a-fA-F]{16}\s+logits=[0-9a-fA-F]{16}\s*$') {
            let item = $line | parse 'statehash die={die} ple={ple} logits={logits}' | get 0
            let die = $item.die | into int
            let die_key = $die | into string
            let ple = $item.ple | str lowercase
            let logits = $item.logits | str lowercase
            if ($current == null or $header == null or $die >= $o.dies or $die_key in ($current.ple | columns)) { error make {msg: 'unexpected or duplicate PLE/logits state'} }
            $current = $current
            | upsert ple ($current.ple | upsert $die_key $ple)
            | upsert logits ($current.logits | upsert $die_key $logits)
            continue
        }
        if ($line =~ '^statehash done\s*$') {
            if ($current == null or $header == null) { error make {msg: 'incomplete statehash'} }
            for die in 0..($o.dies - 1) {
                let die_key = $die | into string
                let layers = owned-layers $o $die
                let states = try { $current.layer_states | get $die_key } catch { {} }
                let actual_layer_count = $states | columns | length
                if ($die_key not-in ($current.ple | columns) or $die_key not-in ($current.logits | columns) or $actual_layer_count != ($layers | length)) {
                    error make {msg: $"incomplete statehash for die ($die)"}
                }
            }
            let n_new = try { $current.n_new } catch { 0 }
            if ($header.positions != ($o.tokens + $n_new) or $header.paired != $current.paired_positions or $header.chunks != $current.prefill_chunks) {
                error make {msg: 'statehash position/execution counts disagree'}
            }
            let fingerprint = {layer_states: $current.layer_states, ple: $current.ple, logits: $current.logits}
            let prompt = $current.prompt
            if $prompt not-in ($references | columns) {
                if ($current.resume != 0 or $current.snap != 0) { error make {msg: 'first reference must be a fresh request'} }
                $references = $references | upsert $prompt {fingerprint: $fingerprint, tokens: $current.tokens, kv_host: $kv, mode: $mode}
            }
            let reference = $references | get $prompt
            if $fingerprint != $reference.fingerprint { error make {msg: $"($current.name): final state/logits fingerprint mismatch"} }
            if $n_new > 0 {
                if $current.tokens != $reference.tokens { error make {msg: $"($current.name): generated token mismatch"} }
            } else {
                for index in $current.resume..($o.tokens - 1) {
                    if (($current.tokens | get $index) != ($reference.tokens | get $index)) { error make {msg: $"($current.name): token mismatch at position ($index)"} }
                }
            }
            $current = $current | upsert statehash_header $header | upsert passed true
            $records ++= [$current]
            $current = null
            $header = null
            continue
        }
        if ($line =~ '^statehash ') { error make {msg: $"unrecognized statehash response: ($line)"} }
    }
    if ($config == null or $tokens != null or $current != null or ($records | length) != ($plan | length)) {
        error make {msg: 'missing config or incomplete response sequence'}
    }
    {config: $config, requests: $records, references: $references}
}

def save-text [path: string, value: string] {
    $value | save --force --raw $path
}

def request-now [] {
    let result = (^date '+%s%N' | complete)
    if $result.exit_code != 0 { error make {msg: 'read clock'} }
    ($result.stdout | str trim | into int) / 1000000000
}

def request-identity [path: string] {
    let full = $path | path expand
    let sum = (^sha256sum $full | complete)
    if $sum.exit_code != 0 { error make {msg: $'hash ($full): ($sum.stderr)'} }
    let stat = (^stat -c '%s %Y' $full | complete)
    if $stat.exit_code != 0 { error make {msg: $'stat ($full): ($stat.stderr)'} }
    let fields = $stat.stdout | str trim | split row ' '
    {
        path: $full, bytes: ($fields | get 0 | into int),
        mtime: ($fields | get 1 | into int),
        sha256: ($sum.stdout | split row ' ' | get 0),
    }
}

def request-capture [argv: list<string>, cwd: string, prefix: string] {
    let program = $argv | get 0
    let params = $argv | skip 1
    let result = do {
        cd $cwd
        ^$program ...$params | complete
    }
    save-text ($prefix + '.stdout') $result.stdout
    save-text ($prefix + '.stderr') $result.stderr
    if $result.exit_code != 0 { error make {msg: $'inventory command failed; see ($prefix).stderr'} }
    $result.stdout
}

def request-holder-allowed [pid: int, o: record] {
    if $pid not-in $o.holder_pids { return false }
    let exe = (^readlink $'/proc/($pid)/exe' | complete)
    if $exe.exit_code != 0 { return false }
    if (($exe.stdout | str trim | path basename) != 'hold') { return false }
    let args = try { (open --raw $'/proc/($pid)/cmdline') | split row "\u{0000}" } catch { [] }
    ($o.manifests | any {|manifest| $manifest in $args })
}

def request-preflight [o: record, prefix: string] {
    request-capture ['ps', '-eo', 'pid=,ppid=,etimes=,stat=,comm='] $o.repo ($prefix + '.ps') | ignore
    let gpu = request-capture [
        'nvidia-smi',
        '--query-gpu=index,uuid,name,driver_version,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits',
    ] $o.repo ($prefix + '.gpu')
    let apps = request-capture [
        'nvidia-smi',
        '--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
        '--format=csv,noheader,nounits',
    ] $o.repo ($prefix + '.apps')
    mut found = []
    mut uuids = []
    for line in ($gpu | lines) {
        let fields = $line | split row ',' | each {|field| $field | str trim }
        if ($fields | length) != 6 { continue }
        let index = try { $fields | get 0 | into int } catch { continue }
        if $index not-in $o.visible_list { continue }
        $found ++= [$index]
        $uuids ++= [$fields.1]
        if (($fields.4 !~ '^\d+$') or ($fields.5 !~ '^\d+$')) { return $'GPU ($index) memory/utilization unavailable' }
        let free = $fields.4 | into int
        let utilization = $fields.5 | into int
        if $free < $o.min_free_mib { return $'GPU ($index) has ($free) MiB free; need ($o.min_free_mib)' }
        if $utilization != 0 { return $'GPU ($index) utilization is ($utilization)%' }
    }
    for index in $o.visible_list {
        if $index not-in $found { return $'GPU index ($index) was not reported' }
    }
    for line in ($apps | lines) {
        let fields = $line | split row ',' | each {|field| $field | str trim }
        if ($fields | length) < 4 { continue }
        if ($fields | get 0) not-in $uuids { continue }
        let pid = try { $fields | get 1 | into int } catch { return $'unrecognized GPU process row: ($line)' }
        if not (request-holder-allowed $pid $o) { return $'GPU process ($pid) is not an allowed holder' }
    }
    let mem = try { open --raw '/proc/meminfo' } catch { '' }
    let line = $mem | lines | where {|row| $row =~ '^MemAvailable:\s+\d+\s+kB$' } | first
    if $line == null { return 'host MemAvailable unavailable' }
    let available = $line | parse 'MemAvailable: {kb} kB' | get 0 | get kb | into int
    if $available < 1024 * $o.min_host_mib { return $'host memory below ($o.min_host_mib) MiB' }
    ''
}

def request-wait [o: record, name: string] {
    let started = request-now
    mut attempt = 0
    while true {
        let reason = request-preflight $o ($o.out | path join $'($name).preflight.($attempt)')
        $attempt += 1
        if ($reason | is-empty) { return }
        if (request-now) - $started >= 30 { error make {msg: $'BLOCKED: ($reason)'} }
        print -e $'waiting: ($reason)'
        ^sleep 1
    }
}

def request-run-core [o: record, env_values: record, commands_path: string,
                      prefix: string, deadline: number] {
    let started = request-now
    let remaining = [1, (($deadline - (request-now)) | math ceil | into int)] | math max
    let inherited_keys = $env | columns | where {|key| ($key !~ '^(MODEL_|FLASH_)') }
    let inherited = $inherited_keys | each {|key| $'($key)=(($env | get $key) | into string)' }
    let explicit = $env_values | columns | each {|key| $'($key)=(($env_values | get $key) | into string)' }
    let binary = $o.repo | path join 'kern/flashcore'
    let library = $o.repo | path join 'kern/libengine.so'
    let argv = [$binary $o.ple $library] | append $o.manifests
    let params = $argv | skip 1
    let result = do {
        cd ($o.repo | path join 'kern')
        open --raw $commands_path | ^timeout --signal=TERM --kill-after=2 $'($remaining)s' ^env -i ...$inherited ...$explicit -- $binary ...$params | complete
    }
    save-text ($prefix + '.stdout') $result.stdout
    save-text ($prefix + '.stderr') $result.stderr
    let elapsed = (request-now) - $started
    {
        argv: $argv, environment: $env_values, stdin: $commands_path,
        stdout: ($prefix + '.stdout'), stderr: ($prefix + '.stderr'),
        exit_code: $result.exit_code,
        signal: (if $result.exit_code >= 128 { $result.exit_code - 128 } else { 0 }),
        timed_out: ($result.exit_code == 124),
        interrupted: false, elapsed_seconds: $elapsed,
        stdout_text: $result.stdout, stderr_text: $result.stderr,
    }
}

def execute-request-check [o: record, plan: list<record>, commands: list<string>, report0: record] {
    let out = $o.out | path expand
    if ($out | path exists) { fail 'create fresh output directory: File exists\n' 255 }
    mkdir $out | ignore
    let result_file = $out | path join 'results.json'
    let deadline = (request-now) + $o.timeout
    let inherited_lock = (^readlink $'/proc/self/fd/($o.lock_fd)' | complete)
    if $inherited_lock.exit_code != 0 { fail "open inherited lock fd failed\n" 255 }
    let lock_path = $inherited_lock.stdout | str trim
    if (($lock_path | path expand) != ($LOCK | path expand)) { fail "wrong inherited lock file\n" 255 }
    let lock_probe = (^flock --exclusive --nonblock $o.lock_fd | complete)
    if $lock_probe.exit_code != 0 { fail "inherited lock fd does not admit LOCK_EX\n" 255 }
    let files = [
        ($o.repo | path join 'kern/flashcore'),
        ($o.repo | path join 'kern/libengine.so'),
        ($o.repo | path join 'kern/die.cubin'),
    ] | append $o.manifests | append $o.weight_manifests | append [
        ($o.repo | path join 'kern/engine.c'),
        ($o.repo | path join 'kern/flashcore.c'),
        ($o.repo | path join 'kern/prog.c'),
        $SCRIPT_PATH,
        ($SCRIPT_DIR | path join 'run_qsa_model_matrix.nu'),
    ]
    let file_identities = $files | each {|path| request-identity $path }
    let head = request-capture ['git', 'rev-parse', 'HEAD'] $o.repo ($out | path join 'source-head')
    let status = request-capture ['git', 'status', '--porcelain'] $o.repo ($out | path join 'source-status')
    request-capture ['git', 'diff', '--binary', 'HEAD'] $o.repo ($out | path join 'source-diff') | ignore
    let source_diff = request-identity ($out | path join 'source-diff.stdout')
    let commands_path = $out | path join 'commands.txt'
    save-text $commands_path (($commands | str join "\n") + "\n")
    let report_path = $out | path join 'results.json'
    mut report = $report0
        | upsert file_identities $file_identities
        | upsert git_head $head
        | upsert git_status $status
        | upsert source_diff $source_diff
        | upsert commands_identity (request-identity $commands_path)
        | upsert status 'RUNNING'
    save-text $report_path (json-output-text $report)
    for kv in $o.kv {
        for mode in ($o.modes | split row ',') {
            if (request-now) >= $deadline { $report = $report | upsert status 'FAILED' | upsert error "whole-run deadline reached\n"; break }
            let name = $'kv($kv)-($mode)'
            let attempt = try {
                request-wait $o $name
                let paired = if $mode == 'scalar' { '0' } else { '1' }
                let chunk = if $mode == 'scalar' { '0' } else { '3' }
                mut env_values = {
                    CUDA_VISIBLE_DEVICES: $o.visible,
                    ATTN_MAP: $o.attn_map,
                    SPLIT_SHARES: $o.split_shares,
                    FLASH_MAXT: ($o.maxt | into string),
                    FLASH_ENGINE_MAXT: ($o.maxt | into string),
                    FLASH_LAYERS: ($o.layers | into string),
                    FLASH_NBLK: ($o.nblk | into string),
                    FLASH_MAXIMG: '0',
                    FLASH_PLE_DISK: '1',
                    FLASH_PLE_OFF: ($o.ple_off | into string),
                    FLASH_KV_HOST: ($kv | into string),
                    FLASH_TWO_POSITION: $paired,
                    FLASH_PREFILL_CHUNK: $chunk,
                }
                for key in ['CUDA_DEVICE_ORDER', 'LD_LIBRARY_PATH'] {
                    let value = try { $env | get $key } catch { null }
                    if $value != null { $env_values = $env_values | upsert $key ($value | into string) }
                }
                let prefix = $out | path join $name
                let result = request-run-core $o $env_values $commands_path $prefix $deadline
                let core = $result | reject stdout_text stderr_text
                    | upsert name $name
                    | upsert stdout_identity (request-identity $result.stdout)
                    | upsert stderr_identity (request-identity $result.stderr)
                $report = $report | update cores ($report.cores | append $core)
                save-text $result_file (json-output-text $report)
                if ($result.exit_code != 0 or $result.signal != 0 or $result.timed_out or $result.interrupted) {
                    error make {msg: $'($name): private core did not exit successfully'}
                }
                let checked = parse-capture $result.stdout_text $o $mode $kv $plan $report.references
                let merged = $core | merge $checked | upsert passed true
                $report = $report | update references $checked.references
                    | update cores ($report.cores | update (($report.cores | length) - 1) $merged)
                save-text $result_file (json-output-text $report)
                {error: null}
            } catch {|err| {error: $err.msg}}
            if $attempt.error != null {
                let status = if ($attempt.error | str starts-with 'BLOCKED:') { 'BLOCKED' } else { 'FAILED' }
                $report = $report | upsert status $status | upsert error ($attempt.error + "\n")
                break
            }
        }
        if $report.status in ['FAILED', 'BLOCKED'] { break }
    }
    if $report.status == 'RUNNING' { $report = $report | upsert status 'SELECTED_REQUEST_CHECKS_PASSED' }
    $report = $report | upsert finished_epoch (request-now)
    save-text $result_file (json-output-text $report)
    print $'($report.status): ($result_file)'
    if $report.error? != null { print -e -n $report.error }
    if $report.status != 'SELECTED_REQUEST_CHECKS_PASSED' { exit 1 }
}

def json-output-text [value: any] {
    let formatted = ($value | to json | ^jq -S --indent 3 | complete)
    if $formatted.exit_code != 0 { fail $formatted.stderr 1 }
    ($formatted.stdout | lines | each {|line| $line | str replace --regex '^(\s*".*"):' '$1 :' } | str join "\n") + "\n"
}

def json-output [value: any] {
    let formatted = ($value | to json | ^jq -S --indent 3 | complete)
    if $formatted.exit_code != 0 { fail $formatted.stderr 1 }
    let text = $formatted.stdout | lines | each {|line| $line | str replace --regex '^(\s*".*"):' '$1 :' } | str join "\n"
    print $text
}

def parse-args [args: list<string>] {
    mut o = {
        repo: '', out: '', ple: '', manifests: [], holder_pids: [], revision: '',
        weight_manifests: [], build_log: '', layers: '8', nblk: '1', tokens: '9',
        snapshots: '1,2,3,5', split_shares: '', attn_map: '', modes: 'scalar,chunk3', kv: '0,1',
        visible: (env-default 'CUDA_VISIBLE_DEVICES' '7'), lock_fd: '9', timeout: '600',
        min_free_mib: '1024', min_host_mib: '1024', ple_off: (env-default 'FLASH_PLE_OFF' '192'),
        plan: false, self_test: false,
    }
    mut manifests = []
    mut holders = []
    mut weights = []
    mut i = 0
    while $i < ($args | length) {
        let arg = $args | get $i
        if $arg == '--plan' { $o = $o | upsert plan true; $i += 1; continue }
        if $arg == '--self-test' { $o = $o | upsert self_test true; $i += 1; continue }
        let parts = $arg | split row '='
        let key = $parts | get 0
        let inline = ($parts | length) > 1
        let value = if $inline { $parts | skip 1 | str join '=' } else {
            $i += 1
            if $i >= ($args | length) { return {error: 'invalid options'} }
            $args | get $i
        }
        if not ($arg | str starts-with '--') { return {error: 'invalid options'} }
        let name = $key | str substring 2..
        if $name == 'manifest' { $manifests ++= [$value]; $i += 1; continue }
        if $name == 'holder-pid' { $holders ++= [$value]; $i += 1; continue }
        if $name == 'weight-manifest' { $weights ++= [$value]; $i += 1; continue }
        if $name in ['repo', 'out', 'ple', 'revision', 'layers', 'nblk', 'tokens', 'snapshots', 'split-shares', 'attn-map', 'modes', 'kv', 'cuda-visible', 'lock-fd', 'timeout', 'min-free-mib', 'min-host-mib', 'ple-off', 'build-log'] {
            let field = if $name == 'split-shares' { 'split_shares' } else if $name == 'attn-map' { 'attn_map' } else if $name == 'cuda-visible' { 'visible' } else if $name == 'lock-fd' { 'lock_fd' } else if $name == 'min-free-mib' { 'min_free_mib' } else if $name == 'min-host-mib' { 'min_host_mib' } else if $name == 'ple-off' { 'ple_off' } else { $name | str replace --all '-' '_' }
            $o = $o | upsert $field $value
            $i += 1
            continue
        }
        return {error: 'invalid options'}
    }
    let finished = $o | upsert manifests $manifests | upsert holder_pids $holders | upsert weight_manifests $weights
    {value: $finished}
}

def main [...args: string] {
    if (try { $env.QSA_REQUEST_TEST_SOURCE } catch { '' }) == '1' { return }
    let parsed = parse-args $args
    if 'error' in $parsed { fail "invalid options\n" 255 }
    mut raw = $parsed.value
    if $raw.self_test {
        let test = (^$nu.current-exe $REQUEST_TEST_FILE | complete)
        print -n $test.stdout
        print -e -n $test.stderr
        exit $test.exit_code
    }
    if ($raw.repo | is-empty) { $raw = $raw | upsert repo (($SCRIPT_DIR | path join '..') | path expand) }
    if (($raw.ple | is-empty) or ($raw.manifests | is-empty) or ($raw.holder_pids | is-empty) or ($raw.revision | is-empty)) {
        fail "require --ple, --manifest, --holder-pid and --revision\n" 255
    }
    let layers = try { $raw.layers | into int } catch { fail "invalid scope/options\n" 255 }
    let nblk = try { $raw.nblk | into int } catch { fail "invalid scope/options\n" 255 }
    let tokens = try { $raw.tokens | into int } catch { fail "invalid scope/options\n" 255 }
    let lock_fd = try { $raw.lock_fd | into int } catch { fail "invalid scope/options\n" 255 }
    let timeout = try { $raw.timeout | into int } catch { fail "invalid scope/options\n" 255 }
    let ple_off = try { $raw.ple_off | into int } catch { fail "invalid scope/options\n" 255 }
    let min_free_mib = try { $raw.min_free_mib | into int } catch { fail "invalid scope/options\n" 255 }
    let min_host_mib = try { $raw.min_host_mib | into int } catch { fail "invalid scope/options\n" 255 }
    if ($layers < 4 or $layers > 48 or $nblk < 1 or $nblk > 4096 or $lock_fd < 3 or $timeout < 1 or $timeout > 86400 or $ple_off < 0 or $min_free_mib < 1 or $min_host_mib < 1 or $raw.revision !~ '^[0-9a-f]{7,40}(-dirty)?$') { fail "invalid scope/options\n" 255 }
    if $raw.modes not-in ['scalar', 'scalar,chunk3'] { fail "--modes must be scalar or scalar,chunk3\n" 255 }
    let kv = try { numbers $raw.kv 0 1 'KV placement' } catch {|err| fail ($err.msg + "\n") 255 }
    let manifests = if (($raw.manifests | is-empty) and (try { $env.QSA_MODEL_MANIFEST } catch { null }) != null) { [$env.QSA_MODEL_MANIFEST] } else { $raw.manifests }
    let holders_raw = if (($raw.holder_pids | is-empty) and (try { $env.QSA_MODEL_HOLDER_PID } catch { null }) != null) { [$env.QSA_MODEL_HOLDER_PID] } else { $raw.holder_pids }
    let holders = $holders_raw | each {|pid| try { $pid | into int } catch { fail "invalid scope/options\n" 255 } }
    let weight_manifests = $raw.weight_manifests | each {|path| $path | path expand }
    let o0 = {
        repo: ($raw.repo | path expand), out: $raw.out, ple: ($raw.ple | path expand), manifests: $manifests,
        holder_pids: $holders, revision: $raw.revision, layers: $layers, nblk: $nblk,
        tokens: $tokens, snapshots: $raw.snapshots, split_shares: $raw.split_shares,
        attn_map: $raw.attn_map, modes: $raw.modes, kv: $raw.kv, visible: $raw.visible,
        lock_fd: $lock_fd, timeout: $timeout, min_free_mib: $min_free_mib,
        min_host_mib: $min_host_mib, ple_off: $ple_off,
        weight_manifests: $weight_manifests, build_log: $raw.build_log,
    }
    let scoped = try { request-scope $o0 } catch {|err| fail ($err.msg + "\n") 255 }
    let plan = request-plan $scoped.snapshot_list
    let commands = commands-for $plan $scoped.tokens
    let scope_name = if $scoped.layers == 48 { 'Full-model' } else { 'Partial-model' }
    let scope = $scope_name + ' GPU request/snapshot and paired-launch equivalence.'
    let report = {
        schema: 'flash-sass.private-qsa-request-check.v2',
        status: 'PLAN_ONLY', acceptance_complete: false,
        scope: $scope,
        layers: $scoped.layers, block_budget: $scoped.nblk, tokens: $scoped.tokens,
        capacity: $scoped.maxt, dies: $scoped.dies, cuda_visible_devices: $scoped.visible,
        split_shares: $scoped.split_shares, attn_map: $scoped.attn_map, manifests: $scoped.manifests,
        holder_pids: $scoped.holder_pids, snapshots: $scoped.snapshot_list,
        requested_selection_exclusion: ($scoped.tokens >= (4 * ($scoped.nblk + 1))),
        expected_revision: $scoped.revision, commands: $commands, request_plan: $plan,
        cores: [], references: {},
        limitations: [
            'No scalar model or independent checkpoint comparison.',
            'State comparisons use engine FNV fingerprints.',
            'The explicit die layout must match the supplied holder arenas; a planned layout is not measured production-placement coverage.',
            'Manifest hashes alone do not establish checkpoint conversion lineage; attach weight identities with --weight-manifest.',
            'gen 0 is teacher-forced. The separate gen 1 response is compared with the full-head scalar-launch engine mode, not with a CPU reference for generated IDs.',
            'Resumed token comparisons exclude the cleared output prefix.',
        ],
    }
    if $raw.plan { json-output $report; return }
    if ($raw.out | is-empty) { fail "--out NEW_DIRECTORY required\n" 255 }
    let out_path = $raw.out | path expand
    let with_out = $scoped | upsert out $out_path | upsert weight_manifests $weight_manifests
    execute-request-check $with_out $plan $commands $report
}
