#!/usr/bin/env nu

def unbound [line: int, position: int] {
    let message = 'validation/issue9/review-4429cbd/run-engine-batch.sh: line ' + ($line | into string) + ': $' + ($position | into string) + ': unbound variable' + "\n"
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
    let scripts = $args | get 2
    let ple = $args | get 3
    let made = (^mkdir $out | complete)
    print -n $made.stdout
    print -e -n $made.stderr
    if $made.exit_code != 0 { exit $made.exit_code }
    for budget in ['1', '512'] {
        run-nu ($scripts + '/check_qsa_engine_requests.nu') [
            '--repo' $repo '--out' ($out + '/budget-' + $budget) '--ple' $ple
            '--revision' '4429cbd0e40c9338cb9e5cf809afd52cb71e0a5e'
            '--layers' '8' '--nblk' $budget '--cuda-visible' '7' '--lock-fd' '9'
            '--modes' 'scalar,chunk3' '--kv' '0,1' '--timeout' '600' '--min-free-mib' '512'
        ]
    }
}
