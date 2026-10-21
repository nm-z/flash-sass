#!/usr/bin/env nu
# CPU-only boundary checks. A positive control proves cuInit is intercepted.

def save-text [path: string, value: string] {
    $value | save --force --raw $path
}

def check [binary: string, scratch: string, name: string, expected: int,
           b0: string, b1: string, tokens: string, flags: string,
           ...overrides: string] {
    let stdout_path = ($scratch | path join $"($name).stdout")
    let stderr_path = ($scratch | path join $"($name).stderr")
    let old_ld = (try { $env.LD_LIBRARY_PATH } catch { '' })
    let ld = if ($old_ld | is-empty) { $scratch } else { $"($scratch):($old_ld)" }
    let ids = ($scratch | path join 'ids')
    let preload = ($scratch | path join 'no_cuda.so')
    let env_result = (^env -u MODEL_NOTWIN -u MODEL_NPH -u MODEL_INJECT -u ATTN_DIE
        'MODEL_ATTN=qsa' 'FLASH_NBLK=1' 'FLASH_KV_HOST=1'
        'ATTN_MAP=0,0,0,0,0,0,0,0,0,0,0,0' 'SPLIT_SHARES=1'
        $"MODEL_IDS=($ids)" $"LD_PRELOAD=($preload)" $"LD_LIBRARY_PATH=($ld)"
        ...$overrides $binary '/unused-cache' $b0 $b1 $tokens $flags '/unused-manifest'
        | complete)
    save-text $stdout_path $env_result.stdout
    save-text $stderr_path $env_result.stderr
    if $env_result.exit_code != $expected {
        print -e $"FAIL ($name): exit ($env_result.exit_code), expected ($expected)"
        let captured = (^cat $stdout_path $stderr_path | complete)
        print -e -n $captured.stdout
        ^rm -rf $scratch
        exit 1
    }
    let reached = ($env_result.stderr | lines | any {|line| $line == 'CUDA_BOUNDARY_REACHED' })
    if ($expected == 99 and not $reached) {
        print -e $"FAIL ($name): missing CUDA boundary marker"
        let captured = (^cat $stdout_path $stderr_path | complete)
        print -e -n $captured.stdout
        ^rm -rf $scratch
        exit 1
    }
    if ($expected != 99 and ($env_result.stderr | str contains 'CUDA_BOUNDARY_REACHED') ) {
        print -e $"FAIL ($name) entered CUDA"
        let captured = (^cat $stdout_path $stderr_path | complete)
        print -e -n $captured.stdout
        ^rm -rf $scratch
        exit 1
    }
    print $"PASS ($name) exit=($env_result.exit_code)"
}

def main [...args: string] {
    let input_binary = if ($args | is-empty) { 'kern/model_test' } else { $args | get 0 }
    let real = (^realpath $input_binary | complete)
    if $real.exit_code != 0 {
        print -e $real.stderr
        exit $real.exit_code
    }
    let binary = $real.stdout | str trim
    let temp = (^mktemp -d '/tmp/flash-model-cli.XXXXXX' | complete)
    if $temp.exit_code != 0 {
        print -e $temp.stderr
        exit $temp.exit_code
    }
    let scratch = $temp.stdout | str trim
    let no_cuda = "#include <stdio.h>\n#include <stdlib.h>\nint cuInit(unsigned flags) {\n    (void)flags;\n    fputs(\"CUDA_BOUNDARY_REACHED\\n\", stderr);\n    _Exit(99);\n}\n"
    save-text ($scratch | path join 'no_cuda.c') $no_cuda
    let cc = try { $env.CC } catch { 'cc' }
    let compile = (^$cc -std=c99 -Wall -Wextra -fPIC -shared -o ($scratch | path join 'no_cuda.so') ($scratch | path join 'no_cuda.c') | complete)
    if $compile.exit_code != 0 {
        print -e $compile.stderr
        ^rm -rf $scratch
        exit $compile.exit_code
    }
    if (try { $env.CUDA_STUB_LIBRARY } catch { '' }) != '' {
        let stub = (^realpath $env.CUDA_STUB_LIBRARY | complete)
        if $stub.exit_code != 0 {
            print -e $stub.stderr
            exit 1
        }
        ^ln -s ($stub.stdout | str trim) ($scratch | path join 'libcuda.so.1')
    }
    save-text ($scratch | path join 'ids') "10\n11\n12\n13\n14\n15\n16\n17\n18\n"
    save-text ($scratch | path join 'short-ids') "10\n11\n"
    save-text ($scratch | path join 'bad-ids') "10\n248320\n"
    save-text ($scratch | path join 'junk-ids') "10\n11junk\n"
    save-text ($scratch | path join 'long-ids') "10\n11\n12\n13\n14\n15\n16\n17\n18\nignored-tail\n"
    check $binary $scratch 'positive-control' 99 '0' '8' '9' 'eh'
    check $binary $scratch 'zero-length' 2 '0' '8' '0' 'eh'
    check $binary $scratch 'negative-length' 2 '0' '8' '-1' 'eh'
    check $binary $scratch 'beyond-capacity' 2 '0' '8' '262145' 'eh'
    check $binary $scratch 'malformed-length' 2 '0' '8' '9junk' 'eh'
    check $binary $scratch 'malformed-block' 2 '0junk' '8' '9' 'eh'
    check $binary $scratch 'invalid-flags' 2 '0' '8' '9' 'ehx'
    check $binary $scratch 'empty-flags' 2 '0' '8' '9' ''
    check $binary $scratch 'invalid-budget' 2 '0' '8' '9' 'eh' 'FLASH_NBLK=1junk'
    check $binary $scratch 'zero-budget' 2 '0' '8' '9' 'eh' 'FLASH_NBLK=0'
    check $binary $scratch 'invalid-placement' 2 '0' '8' '9' 'eh' 'FLASH_KV_HOST=2'
    check $binary $scratch 'missing-teacher-ids' 2 '0' '8' '9' 'eh' $"MODEL_IDS=($scratch | path join 'short-ids')"
    check $binary $scratch 'out-of-vocabulary-id' 2 '0' '8' '2' 'eh' $"MODEL_IDS=($scratch | path join 'bad-ids')"
    check $binary $scratch 'malformed-id' 2 '0' '8' '2' 'eh' $"MODEL_IDS=($scratch | path join 'junk-ids')"
    check $binary $scratch 'valid-id-prefix' 99 '0' '8' '9' 'eh' $"MODEL_IDS=($scratch | path join 'long-ids')"
    check $binary $scratch 'no-attention-block' 2 '0' '3' '9' 'eh'
    check $binary $scratch 'unreachable-selection' 2 '0' '8' '4' 'eh' 'MODEL_INJECT=selection'
    check $binary $scratch 'reachable-selection' 99 '0' '8' '9' 'eh' 'MODEL_INJECT=selection'
    check $binary $scratch 'unreachable-logits' 2 '0' '8' '9' 'e' 'MODEL_INJECT=logits'
    check $binary $scratch 'unreachable-dn' 2 '3' '4' '9' 'eh' 'MODEL_INJECT=dn'
    check $binary $scratch 'unreachable-conv' 2 '3' '4' '9' 'eh' 'MODEL_INJECT=conv'
    check $binary $scratch 'unreachable-ple' 2 '3' '4' '9' 'eh' 'MODEL_INJECT=ple'
    check $binary $scratch 'unknown-injection' 2 '0' '8' '9' 'eh' 'MODEL_INJECT=unknown'
    check $binary $scratch 'timing-injection' 2 '0' '8' '9' 'eh' 'MODEL_INJECT=layer' 'MODEL_NOTWIN=1'
    check $binary $scratch 'truncated-program' 2 '0' '8' '9' 'eh' 'MODEL_NPH=1'
    check $binary $scratch 'absent-attention-owner' 2 '0' '8' '9' 'eh' 'ATTN_MAP=1,1,1,1,1,1,1,1,1,1,1,1'
    print 'OK: 26 CPU boundary checks; CUDA calls intercepted, no GPU validation'
    ^rm -rf $scratch
}
