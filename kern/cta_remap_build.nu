# Reproduce issue #12's exact component build matrix. This builds images only.

def fail [message: string, code: int] {
    print -n --stderr $message
    exit $code
}

def checked [result: record, context: string] {
    if $result.exit_code != 0 {
        if $result.stderr != "" { print -n --stderr $result.stderr }
        if $result.stdout != "" { print -n --stderr $result.stdout }
        fail $"($context)\n" $result.exit_code
    }
}

def s_path [...parts: string] {
    $parts | path join
}

def main [...argv: string] {
    if ($argv | length) < 1 or ($argv | length) > 2 {
        print -n --stderr "usage: kern/cta_remap_build.sh NEW_OUTPUT_DIRECTORY [FULL_HARNESS_COMMIT]\n"
        exit 2
    }
    let script_dir = $env.FILE_PWD
    let root_result = (run-external "git" "-C" $script_dir "rev-parse" "--show-toplevel" | complete)
    checked $root_result "could not locate repository root"
    let root = ($root_result.stdout | str trim)
    let revision_name = if ($argv | length) == 2 { $"($argv | get 1)^{commit}" } else { "HEAD^{commit}" }
    let revision_result = (run-external "git" "-C" $root "rev-parse" $revision_name | complete)
    checked $revision_result "could not resolve source commit"
    let revision = ($revision_result.stdout | str trim)
    let parent = "c6d3de8180d8b87caf0d2f61fa52edc9247daded"
    let remap = "66725432f86cc193b75cd51230216aed44c60d3f"

    let out_arg = ($argv | get 0)
    mkdir $out_arg
    let out = ($out_arg | path expand)
    let ptxas = if ($env.PTXAS11? | is-empty) { "/opt/cuda-11.4/bin/ptxas" } else { $env.PTXAS11 }
    let cuda = if ($env.CUDA? | is-empty) { "/opt/cuda" } else { $env.CUDA }
    let compiler = if ($env.CC? | is-empty) { "cc" } else { $env.CC }
    let maxas_root = if ($env.MAXAS_ROOT? | is-empty) { [$root "maxas"] | path join } else { $env.MAXAS_ROOT }
    let maxas_root = ($maxas_root | path expand)
    let maxas_program = [$maxas_root "bin" "maxas.pl"] | path join
    let maxas_library = [$maxas_root "lib" "MaxAs"] | path join
    if not ($maxas_program | path exists) or not ($maxas_library | path exists) {
        print -n --stderr "set MAXAS_ROOT to the patched MaxAs checkout (bin/maxas.pl and lib/MaxAs)\n"
        exit 2
    }

    let toolchain = (s_path $out "maxas-toolchain")
    mkdir $toolchain | ignore
    let copy_result = (run-external "cp" "-a" (s_path $maxas_root "bin") (s_path $maxas_root "lib") $toolchain | complete)
    checked $copy_result "could not snapshot MaxAs toolchain"
    let hash_result = (do {
        cd $toolchain
        ^rg --files -0 bin lib | ^sort -z | ^xargs -0 sha256sum | complete
    })
    checked $hash_result "could not hash MaxAs toolchain"
    ($hash_result.stdout | save --force --raw (s_path $out "maxas.sha256"))
    let maxas_commit = (run-external "git" "-C" $maxas_root "rev-parse" "HEAD" | complete)
    checked $maxas_commit "could not record MaxAs commit"
    ($maxas_commit.stdout | save --force --raw (s_path $out "maxas-upstream.txt"))
    let maxas_diff = (run-external "git" "-C" $maxas_root "diff" "HEAD" "--" "bin" "lib" | complete)
    checked $maxas_diff "could not record MaxAs local patch"
    ($maxas_diff.stdout | save --force --raw (s_path $out "maxas-local.patch"))

    let host = (run-external "hostname" | complete)
    let compiler_version = (run-external $compiler "--version" | complete)
    let ptxas_version = (run-external $ptxas "--version" | complete)
    let perl_version = (run-external "perl" "-V:version" | complete)
    let trace_env = (run-external "env" | complete)
    checked $host "hostname query failed"
    checked $compiler_version "compiler version query failed"
    checked $ptxas_version "ptxas version query failed"
    checked $perl_version "Perl version query failed"
    checked $trace_env "environment query failed"
    let selected_env = ($trace_env.stdout | lines | where { |line| $line =~ '^(DIE_|ATTN_|GEMV_|CUDA_|FLASH_|SPLIT_|PTXAS|CC=)' } | sort | str join "\n")
    let toolchain_text = $"host=($host.stdout | str trim)\nrevision=($revision)\nparent=($parent)\nremap=($remap)\nCUDA=($cuda)\nPTXAS11=($ptxas)\nCC=($compiler)\n($compiler_version.stdout)($ptxas_version.stdout)($perl_version.stdout)MAXAS_ROOT=($maxas_root)\n($selected_env)\n"
    $toolchain_text | save --force --raw (s_path $out "toolchain.txt")

    let mapping = (run-external "git" "-C" $root "diff" $parent $remap "--" "kern/attnpartn.sass" | complete)
    checked $mapping "could not create remap patch"
    $mapping.stdout | save --force --raw (s_path $out "remap.patch")

    let variants = ["historical-old" "historical-new" "matched-old" "matched-new" "trace-old" "trace-new"]
    for variant in $variants {
        let source = if $variant == "historical-old" { $parent } else if $variant == "historical-new" { $remap } else { $revision }
        let build = (s_path $out $variant)
        mkdir $build | ignore
        let extract = (^git "-C" $root "archive" $source | ^tar "-xf" "-" "-C" $build | complete)
        checked $extract $"could not extract source for ($variant)"
        let link_result = (run-external "ln" "-s" $toolchain (s_path $build "maxas") | complete)
        checked $link_result $"could not link MaxAs for ($variant)"
        ($source + "\n") | save --force --raw (s_path $build "source-commit.txt")

        if $variant == "matched-old" or $variant == "trace-old" {
            let patch_result = (^patch $"--directory=($build)" "-p1" "--reverse" "--batch" "--fuzz=0" < (s_path $out "remap.patch") | complete)
            checked $patch_result $"could not reverse mapping for ($variant)"
            $patch_result.stdout | save --force --raw (s_path $out $"($variant).mapping.log")
        }

        let trace = if ($variant | str starts-with "trace-") { "1" } else { "0" }
        let log = [$out $"($variant).build.log"] | path join
        let make_args = ["-C" (s_path $build "kern") $"CUDA=($cuda)" $"PTXAS11=($ptxas)" $"CC=($compiler)" $"FLASH_REV=($source)" "die.cubin" "libengine.so"]
        let build_result = if $trace == "1" {
            with-env { DIE_CTA_TRACE: "1" } { ^make ...$make_args | complete }
        } else {
            with-env { DIE_CTA_TRACE: null } { ^make ...$make_args | complete }
        }
        ($build_result.stdout + $build_result.stderr) | save --force --raw $log
        if $build_result.exit_code != 0 { fail ($build_result.stderr + $build_result.stdout) $build_result.exit_code }

        if $variant == "matched-new" {
            let test_args = ["-C" (s_path $build "kern") $"CUDA=($cuda)" $"PTXAS11=($ptxas)" $"CC=($compiler)" $"FLASH_REV=($source)" "attnpartn_regress" "libengine_cta_remap.so" "flashcore" "hold"]
            let test_result = (run-external "make" ...$test_args | complete)
            ($test_result.stdout + $test_result.stderr) | save --append --raw $log
            if $test_result.exit_code != 0 { fail ($test_result.stderr + $test_result.stdout) $test_result.exit_code }
        }

        let hashes = (do {
            cd $build
            let base = (^sha256sum "kern/die.pl" "kern/attnpartn.sass" "kern/die.sass" "kern/die.cubin" "kern/engine.c" "kern/libengine.so" | complete)
            if not ((s_path $build "kern/attnpartn_regress") | path exists) {
                $base.stdout
            } else {
                let extra = (^sha256sum "kern/attnpartn_regress.c" "kern/cta_trace.h" "kern/cta_remap_bench.c" "kern/attnpartn_regress" "kern/libengine_cta_remap.so" "kern/flashcore.c" "kern/flashcore" "kern/hold.c" "kern/hold" | complete)
                $base.stdout + $extra.stdout
            }
        })
        $hashes | save --force --raw (s_path $out $"($variant).sha256")
        let status_line = $"($variant) source=($source) DIE_CTA_TRACE=($trace) build=PASS\n"
        $status_line | save --append --raw (s_path $out "status.txt")
    }
    let matched_cmp = (run-external "cmp" (s_path $out "matched-old" "kern/libengine.so") (s_path $out "matched-new" "kern/libengine.so") | complete)
    checked $matched_cmp "matched engine libraries differ"
    let trace_cmp = (run-external "cmp" (s_path $out "trace-old" "kern/libengine.so") (s_path $out "trace-new" "kern/libengine.so") | complete)
    checked $trace_cmp "traced engine libraries differ"
    "PASS: six isolated builds; matched library pairs are byte-identical\n" | save --append --raw (s_path $out "status.txt")
}
