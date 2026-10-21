#!/usr/bin/env bash
# Reproduce issue #12's exact historical images and the matched current-engine
# pair. Build only: this script never initializes CUDA or launches a GPU kernel.
# Usage: bash kern/cta_remap_build.sh NEW_OUTPUT_DIRECTORY [FULL_HARNESS_COMMIT]
set -euo pipefail
[[ $# == 1 || $# == 2 ]] || { echo "usage: $0 NEW_OUTPUT_DIRECTORY [FULL_HARNESS_COMMIT]" >&2; exit 2; }
root=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
revision=$(git -C "$root" rev-parse "${2:-HEAD}^{commit}")
parent=c6d3de8180d8b87caf0d2f61fa52edc9247daded
remap=66725432f86cc193b75cd51230216aed44c60d3f
mkdir -- "$1"
out=$(cd "$1" && pwd)
ptxas=${PTXAS11:-/opt/cuda-11.4/bin/ptxas}
cuda=${CUDA:-/opt/cuda}
compiler=${CC:-cc}
maxas_root=${MAXAS_ROOT:-"$root/maxas"}
[[ -f "$maxas_root/bin/maxas.pl" && -d "$maxas_root/lib/MaxAs" ]] || {
    echo "set MAXAS_ROOT to the patched MaxAs checkout (bin/maxas.pl and lib/MaxAs)" >&2
    exit 2
}
maxas_root=$(cd "$maxas_root" && pwd)
# Freeze the exact assembler implementation once for all six variants.
mkdir "$out/maxas-toolchain"
cp -a "$maxas_root/bin" "$maxas_root/lib" "$out/maxas-toolchain/"
(
    cd "$out/maxas-toolchain"
    rg --files -0 bin lib | LC_ALL=C sort -z | xargs -0 sha256sum
) > "$out/maxas.sha256"
git -C "$maxas_root" rev-parse HEAD > "$out/maxas-upstream.txt"
git -C "$maxas_root" diff HEAD -- bin lib > "$out/maxas-local.patch"
{
    printf 'host=%s\nrevision=%s\nparent=%s\nremap=%s\nCUDA=%s\nPTXAS11=%s\nCC=%s\n' "$(hostname)" "$revision" "$parent" "$remap" "$cuda" "$ptxas" "$compiler"
    "$compiler" --version
    "$ptxas" --version
    perl -V:version
    printf 'MAXAS_ROOT=%s\n' "$maxas_root"
    env | LC_ALL=C sort | sed -n '/^\(DIE_\|ATTN_\|GEMV_\|CUDA_\|FLASH_\|SPLIT_\|PTXAS\|CC=\)/p'
} > "$out/toolchain.txt" 2>&1
git -C "$root" diff "$parent" "$remap" -- kern/attnpartn.sass > "$out/remap.patch"
for variant in historical-old historical-new matched-old matched-new trace-old trace-new; do
    source=$revision
    [[ $variant != historical-old ]] || source=$parent
    [[ $variant != historical-new ]] || source=$remap
    build="$out/$variant"
    mkdir "$build"
    git -C "$root" archive "$source" | tar -xf - -C "$build"
    ln -s "$out/maxas-toolchain" "$build/maxas"
    printf '%s\n' "$source" > "$build/source-commit.txt"
    if [[ $variant == matched-old || $variant == trace-old ]]; then
        patch --directory="$build" -p1 --reverse --batch --fuzz=0 < "$out/remap.patch" > "$out/$variant.mapping.log"
    fi
    trace=0
    [[ $variant != trace-* ]] || trace=1
    # The source archive has no .git directory: pass the full source identity.
    if [[ $trace == 1 ]]; then
        DIE_CTA_TRACE=1 make -C "$build/kern" CUDA="$cuda" PTXAS11="$ptxas" CC="$compiler" FLASH_REV="$source" die.cubin libengine.so > "$out/$variant.build.log" 2>&1
    else
        env -u DIE_CTA_TRACE make -C "$build/kern" CUDA="$cuda" PTXAS11="$ptxas" CC="$compiler" FLASH_REV="$source" die.cubin libengine.so > "$out/$variant.build.log" 2>&1
    fi
    if [[ $variant == matched-new ]]; then
        make -C "$build/kern" CUDA="$cuda" PTXAS11="$ptxas" CC="$compiler" FLASH_REV="$source" attnpartn_regress libengine_cta_remap.so flashcore hold >> "$out/$variant.build.log" 2>&1
    fi
    (
        cd "$build"
        sha256sum kern/die.pl kern/attnpartn.sass kern/die.sass kern/die.cubin kern/engine.c kern/libengine.so
        if [[ -f kern/attnpartn_regress ]]; then
            sha256sum kern/attnpartn_regress.c kern/cta_trace.h kern/cta_remap_bench.c kern/attnpartn_regress kern/libengine_cta_remap.so kern/flashcore.c kern/flashcore kern/hold.c kern/hold
        fi
    ) > "$out/$variant.sha256"
    printf '%s source=%s DIE_CTA_TRACE=%s build=PASS\n' "$variant" "$source" "$trace" >> "$out/status.txt"
done
# The host code is identical between arms. Retain the exact kernel-only mapping
# patch as well as each generated SASS/cubin identity.
cmp "$out/matched-old/kern/libengine.so" "$out/matched-new/kern/libengine.so"
cmp "$out/trace-old/kern/libengine.so" "$out/trace-new/kern/libengine.so"
printf 'PASS: six isolated builds; matched library pairs are byte-identical\n' >> "$out/status.txt"
