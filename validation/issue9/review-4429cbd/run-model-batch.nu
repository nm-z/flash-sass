#!/usr/bin/env nu

def unbound [line: int, position: int] {
    let message = 'validation/issue9/review-4429cbd/run-model-batch.sh: line ' + ($line | into string) + ': $' + ($position | into string) + ': unbound variable' + "\n"
    print -e -n $message
    exit 1
}

def run-nu [script: string, options: list<string>] {
    let result = (^nu $script -- ...$options | complete)
    print -n $result.stdout
    print -e -n $result.stderr
    if $result.exit_code != 0 { exit $result.exit_code }
}

def main [...args: string] {
    let count = $args | length
    if $count < 4 { unbound (3 + $count) (1 + $count) }
    let repo = $args | get 0
    let out = $args | get 1
    let cache = $args | get 2
    let build_log = $args | get 3
    let made = (^mkdir $out | complete)
    print -n $made.stdout
    print -e -n $made.stderr
    if $made.exit_code != 0 { exit $made.exit_code }
    let script = $repo + '/kern/run_qsa_model_matrix.nu'
    let common = [
        '--repo' $repo '--cache' ($cache + '/tensors') '--ids' ($cache + '/ids9.txt')
        '--blocks' '0:8' '--flags' 'eh' '--cuda-visible' '7' '--lock-fd' '9'
        '--weight-manifest' ($cache + '/tensors.sha256')
        '--weight-manifest' ($cache + '/manifest.new') '--build-log' $build_log
        '--min-free-mib' '512' '--timeout-seconds' '600'
    ]
    run-nu $script ($common | append ['--out' ($out + '/positive9') '--tokens' '9' '--budgets' '1,512' '--kv' '0,1' '--modes' 'token,loop' '--controls' 'none'])
    run-nu $script ($common | append ['--out' ($out + '/control1') '--tokens' '1' '--budgets' '1' '--kv' '0,1' '--modes' 'token' '--controls' 'all'])
    run-nu $script ($common | append ['--out' ($out + '/selection9') '--tokens' '9' '--budgets' '1' '--kv' '0,1' '--modes' 'token' '--controls' 'selection'])
}
