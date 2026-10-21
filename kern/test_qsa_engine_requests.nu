#!/usr/bin/env nu
$env.QSA_REQUEST_TEST_SOURCE = '1'
const CHECK_FILE = path self 'check_qsa_engine_requests.nu'
source $CHECK_FILE

def raw-scope [dies: int, tokens: int, layers: int] {
    let visible = (0..($dies - 1) | each {|die| $die | into string } | reverse) | str join ','
    let manifests = 0..($dies - 1) | each {|die| $"/unused/die-($die).manifest" }
    let holders = 0..($dies - 1) | each {|die| 1000 + $die }
    let shares = 0..($dies - 1) | each {|die| ($die + 1) | into string } | str join ','
    let owners = 0..11 | each {|index| (($index + 1) mod $dies) | into string } | str join ','
    {
        revision: 'abcdef0', layers: $layers, nblk: 512,
        tokens: $tokens, snapshots: '1,2,3,5', visible: $visible,
        manifests: $manifests, holder_pids: $holders,
        split_shares: $shares, attn_map: $owners,
    }
}

def hex16 [number: int] {
    let raw = $number | format number | get lowerhex | str substring 2..
    let zeros = (1..(16 - ($raw | str length)) | each { '0' } | str join '')
    $zeros + $raw
}

def fixture [o: record, mode: string, kv: int] {
    let plan = request-plan $o.snapshot_list
    let paired = if $mode == 'scalar' { 0 } else { 1 }
    let chunk = if $paired == 1 { 3 } else { 0 }
    mut lines = [$"config rev=($o.revision) kv_host=($kv) ple_disk=1 maxt=($o.maxt) layers=($o.layers)/48 nblk=($o.nblk) dies=($o.dies) split=($o.split_shares) attn_map=($o.attn_map) attn_dies=($o.attn_dies) two_position=($paired) prefill_chunk=($chunk)"]
    for request in $plan {
        let base = if $request.prompt == 'A' { 100 } else if $request.prompt == 'B' { 200 } else { 300 }
        let n_new = try { $request.n_new } catch { 0 }
        let ids = if $n_new > 0 {
            ($o.tokens - 1)..$o.tokens | each {|index| (($base + $index) mod 248320) | into string }
        } else {
            0..($o.tokens - 1) | each {|index|
                if $index < $request.resume { '0' } else { (($base + $index) mod 248320) | into string }
            }
        }
        let positions = $o.tokens + $n_new
        let pairs = expected-pairs $request $mode $o.tokens
        let chunks = if ($n_new > 0 and $paired == 1) { ((($o.tokens + 1) / 3) | math floor | into int) } else { 0 }
        let token_line = 'tokens: ' + ($ids | str join ' ')
        $lines ++= [$token_line]
        let run_count = $positions - $request.resume
        let resumed = $request.resume
        let time_line = 'time: 0 ms for ' + ($run_count | into string) + ' positions run (' + ($resumed | into string) + ' from the snapshot; ' + ($pairs | into string) + ' paired positions; ' + ($chunks | into string) + ' prefill chunks)'
        $lines ++= [$time_line]
        $lines ++= [$"statehash positions=($positions) paired_positions=($pairs) prefill_chunks=($chunks)"]
        for die in 0..($o.dies - 1) {
            for layer in (owned-layers $o $die) {
                let hash = hex16 ($base + 1000 * $die + $layer)
                $lines ++= [$"statehash die=($die) layer=($layer) state=($hash)"]
            }
            let ple = hex16 ($base + 1000 * $die + 60)
            let logits = hex16 ($base + 1000 * $die + 70)
            $lines ++= [$"statehash die=($die) ple=($ple) logits=($logits)"]
        }
        $lines ++= ['statehash done']
    }
    ($lines | str join "\n") + "\n"
}

def run-tests [] {
    hide-env --ignore-errors SPLIT_SHARES ATTN_MAP
    mut checks = []
    mut positive = 0
    mut negative = 0
    let shapes = [[1, 9, 8], [2, 6, 8], [2, 7, 8], [2, 8, 8], [2, 10, 8], [6, 9, 48], [6, 2052, 48]]
    for shape in $shapes {
        let o0 = raw-scope ($shape | get 0) ($shape | get 1) ($shape | get 2)
        let o1 = if ($shape | get 1) == 2052 { $o0 | upsert snapshots ($o0.snapshots + ',2048,2051') } else { $o0 }
        let o = request-scope $o1
        let plan = request-plan $o.snapshot_list
        mut references = {}
        for kv in [0, 1] {
            for mode in ['scalar', 'chunk3'] {
                let capture = fixture $o $mode $kv
                let result = try { parse-capture $capture $o $mode $kv $plan $references } catch {|err| print -e $err.msg; null }
                let passed = ($result != null and ($result.requests | length) == ($plan | length))
                $checks ++= [{passed: $passed, label: $"synthetic ($o.dies)-die/($o.layers)-layer/($o.tokens)-token kv($kv) ($mode) capture"}]
                $positive += 1
                if $result != null { $references = $result.references }
            }
        }
        $checks ++= [{passed: (($references.A.fingerprint.ple | columns | length) == $o.dies), label: 'reference contains every PLE/logits shard'}]
    }

    let o = request-scope (raw-scope 6 9 48)
    let plan = request-plan $o.snapshot_list
    let capture = fixture $o 'chunk3' 0
    let parsed = parse-capture $capture $o 'chunk3' 0 $plan {}
    let references = $parsed.references
    let mutations = [
        {label: 'die-5 recurrent state', pattern: '(statehash die=5 layer=0 state=)[0-9a-f]+', replacement: '$1ffffffffffffffff'},
        {label: 'die-5 attention state', pattern: '(statehash die=5 layer=19 state=)[0-9a-f]+', replacement: '$1ffffffffffffffff'},
        {label: 'die-5 PLE', pattern: '(statehash die=5 ple=)[0-9a-f]+', replacement: '$1ffffffffffffffff'},
        {label: 'die-5 logits', pattern: '(statehash die=5 ple=[0-9a-f]+ logits=)[0-9a-f]+', replacement: '$1ffffffffffffffff'},
        {label: 'missing remote layer', pattern: '(?m)^statehash die=5 layer=0 .*\n', replacement: ''},
        {label: 'missing remote attention', pattern: '(?m)^statehash die=5 layer=19 .*\n', replacement: ''},
        {label: 'missing remote PLE/logits', pattern: '(?m)^statehash die=5 ple=.*\n', replacement: ''},
        {label: 'duplicate remote layer', pattern: '(?m)^(statehash die=5 layer=0 .*\n)', replacement: '$1$1'},
        {label: 'duplicate remote PLE/logits', pattern: '(?m)^(statehash die=5 ple=.*\n)', replacement: '$1$1'},
        {label: 'unexpected die', pattern: 'die=5 layer=0 ', replacement: 'die=6 layer=0 '},
        {label: 'attention on wrong owner', pattern: 'die=1 layer=3 ', replacement: 'die=0 layer=3 '},
        {label: 'wrong revision', pattern: 'rev=abcdef0', replacement: 'rev=abcdef1'},
        {label: 'duplicate config field', pattern: 'rev=abcdef0', replacement: 'rev=abcdef1 rev=abcdef0'},
        {label: 'wrong die count', pattern: 'dies=6 ', replacement: 'dies=1 '},
        {label: 'wrong split', pattern: 'split=1,2,3,4,5,6', replacement: 'split=1,1,1,1,1,1'},
        {label: 'wrong attention map', pattern: 'attn_map=1,2', replacement: 'attn_map=0,2'},
        {label: 'wrong active dies', pattern: 'attn_dies=0,1', replacement: 'attn_dies=1'},
        {label: 'wrong budget', pattern: 'nblk=512', replacement: 'nblk=1'},
        {label: 'wrong capacity', pattern: 'maxt=128', replacement: 'maxt=256'},
        {label: 'wrong paired count', pattern: '8 paired positions', replacement: '6 paired positions'},
        {label: 'out-of-vocabulary token', pattern: 'tokens: 100 ', replacement: 'tokens: 248320 '},
        {label: 'mismatched generation token', pattern: 'tokens: 308 309', replacement: 'tokens: 308 310'},
        {label: 'wrong chunk count', pattern: '6 paired positions; 3 prefill chunks', replacement: '6 paired positions; 4 prefill chunks'},
        {label: 'incomplete final response', pattern: 'statehash done\n$', replacement: ''},
        {label: 'engine error', pattern: '^', replacement: 'error: failed generation\n'},
    ]
    for mutation in $mutations {
        let bad = $capture | str replace --regex $mutation.pattern $mutation.replacement
        let rejected = try { parse-capture $bad $o 'chunk3' 0 $plan $references; false } catch { true }
        $checks ++= [{passed: $rejected, label: $"reject ($mutation.label)"}]
        $negative += 1
    }

    for case in [[6, 4], [7, 4], [8, 4], [9, 6], [10, 6], [2052, 1368]] {
        $checks ++= [{passed: ((expected-pairs {n_new: 1} 'chunk3' ($case | get 0)) == ($case | get 1)), label: $"paired generation-prefix positions at length ($case | get 0)"}]
    }
    $checks ++= [
        {passed: ((expected-pairs {resume: 2051, snap: 0} 'chunk3' 2052) == 0), label: 'resume at final prompt position leaves no paired prefix'},
        {passed: ((expected-pairs {resume: 0, snap: 2051} 'chunk3' 2052) == 2050), label: 'snapshot immediately before final position splits paired prefix'},
        {passed: ((expected-pairs {resume: 0, snap: 0} 'scalar' 2052) == 0), label: 'scalar-launch reference never claims paired work'},
    ]

    let bad_scopes = [
        {label: 'duplicate device', key: 'visible', value: '0,0'},
        {label: 'too many devices', key: 'visible', value: '0,1,2,3,4,5,6'},
        {label: 'missing manifest', key: 'manifests', action: 'shorten'},
        {label: 'duplicate manifest', key: 'manifests', action: 'duplicate'},
        {label: 'missing holder', key: 'holder_pids', action: 'shorten'},
        {label: 'duplicate holder', key: 'holder_pids', action: 'duplicate'},
        {label: 'bad holder PID', key: 'holder_pids', value: [-1, 1001, 1002, 1003, 1004, 1005]},
        {label: 'missing shares', key: 'split_shares', value: ''},
        {label: 'missing map', key: 'attn_map', value: ''},
        {label: 'zero share', key: 'split_shares', value: '1,0,1,1,1,1'},
        {label: 'malformed share', key: 'split_shares', value: '1,1x,1,1,1,1'},
        {label: 'short map', key: 'attn_map', value: '0,1'},
        {label: 'out-of-range owner', key: 'attn_map', value: (0..11 | each { '6' } | str join ',')},
        {label: 'negative owner', key: 'attn_map', value: (0..11 | each { '-1' } | str join ',')},
        {label: 'unreachable default snapshot', key: 'tokens', value: 5},
        {label: 'capacity overflow', key: 'tokens', value: 262143},
        {label: 'duplicate snapshot', key: 'snapshots', value: '1,1'},
        {label: 'snapshot past prompt', key: 'snapshots', value: '9'},
    ]
    for bad_case in $bad_scopes {
        mut bad = raw-scope 6 9 48
        if $bad_case.action? == 'shorten' {
            let shorter = if $bad_case.key == 'manifests' { $bad.manifests | take 5 } else { $bad.holder_pids | take 5 }
            $bad = $bad | upsert $bad_case.key $shorter
        } else if $bad_case.action? == 'duplicate' {
            let first = if $bad_case.key == 'manifests' { $bad.manifests | get 0 } else { $bad.holder_pids | get 0 }
            $bad = $bad | upsert $bad_case.key [$first, $first]
        } else {
            $bad = $bad | upsert $bad_case.key $bad_case.value
        }
        let rejected = try { request-scope $bad; false } catch { true }
        $checks ++= [{passed: $rejected, label: $"reject scope: ($bad_case.label)"}]
    }

    let deep0 = raw-scope 6 2052 48
    let deep = request-scope ($deep0 | upsert snapshots '1,2,3,5,2048,2051')
    $checks ++= [{passed: ($deep.maxt == 2176), label: '2052 positions plus parser/generation headroom'}]
    let deep_plan = request-plan $deep.snapshot_list
    let commands = commands-for $deep_plan $deep.tokens
    let gen = $commands | where {|command| $command | str starts-with 'gen ' }
    $checks ++= [{passed: (($gen | length) == 22), label: 'deep snapshot positions add explicit save/resume/reset requests'}]
    $checks ++= [{passed: (($gen | get 0 | split row ' ' | length) == 2054), label: 'teacher command has exactly 2052 IDs'}]
    $checks ++= [{passed: ($commands | any {|command| $command == 'snap 2051' }), label: 'production-boundary snapshot is not dropped'}]
    let last = request-scope (raw-scope 1 262142 8)
    $checks ++= [{passed: ($last.maxt == 262144), label: 'maximum prompt respects capacity and generation headroom'}]
    let long_gen = commands-for (request-plan [1]) $last.tokens | where {|command| $command | str starts-with 'gen ' }
    let long_ids = $long_gen | get 0 | split row ' ' | skip 2
    $checks ++= [{passed: (($long_ids | length) == 262142 and not ($long_ids | any {|id| ($id | into int) >= 248320 })), label: 'long synthetic prompts wrap at the vocabulary boundary'}]

    for index in 0..(($checks | length) - 1) {
        let check = $checks | get $index
        let outcome = if $check.passed { 'ok' } else { 'not ok' }
        print $"($outcome) ($index + 1) - ($check.label)"
    }
    print $"1..($checks | length)"
    print -e $"# ($positive) complete captures and ($negative) corrupt captures tested; CPU fixtures only, no GPU validation"
}

run-tests
