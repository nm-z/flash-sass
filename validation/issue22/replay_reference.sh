#!/usr/bin/env bash
# Recompare the recovered CPU capture with the historical T=4 GPU capture.
# This executes no model and declares no quality thresholds.
set -euo pipefail
if [[ $# != 1 ]]; then
    printf 'usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
issue22_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
issue22_out=$1
[[ ! -e $issue22_out ]] || { printf 'output exists: %s\n' "$issue22_out" >&2; exit 2; }
mkdir -p -- "$(dirname -- "$issue22_out")"
mkdir -- "$issue22_out"
issue22_out=$(cd -- "$issue22_out" && pwd)
issue22_tmp=$(mktemp -d)
trap 'rm -rf -- "$issue22_tmp"' EXIT
issue22_cpu="$issue22_repo/validation/issue22/archy-llama-7182ef7"
issue22_gpu="$issue22_repo/validation/issue22/archy-06968d1"
(
    cd -- "$issue22_cpu"
    sha256sum --check --status SHA256SUMS
)
perl -MDigest::SHA -MJSON::PP -e '
    my($root)=@ARGV;
    open my $f,"<","$root/IDENTITIES.json" or die $!;
    local $/; my $m=decode_json(<$f>); close $f;
    for my $p(qw(gpu/t4-budget512-kv0-raw/logits.f32.gz
                 gpu/t4-budget512-kv0-raw/ids.u32
                 gpu/t4-budget512-kv0-raw/capture_info.json
                 gpu/t4-budget512-kv0-metadata.json)) {
        open my $in,"<:raw","$root/$p" or die "$p: $!";
        my $h=Digest::SHA->new(256); $h->addfile($in); close $in;
        $h->hexdigest eq $m->{$p}{sha256} or die "changed input: $p\n";
    }
' "$issue22_gpu"
make -C "$issue22_repo/kern" model_quality
gzip -dc -- "$issue22_gpu/gpu/t4-budget512-kv0-raw/logits.f32.gz" > "$issue22_tmp/gpu.f32"
gzip -dc -- "$issue22_cpu/logits.f32.gz" > "$issue22_tmp/cpu.f32"
gzip -dc -- "$issue22_cpu/vocab_ids.u32.gz" > "$issue22_tmp/vocab.u32"
"$issue22_repo/kern/model_quality" freeze --kind candidate \
    --ids "$issue22_gpu/gpu/t4-budget512-kv0-raw/ids.u32" \
    --logits "$issue22_tmp/gpu.f32" --vocab-size 248320 \
    --raw-info "$issue22_gpu/gpu/t4-budget512-kv0-raw/capture_info.json" \
    --metadata "$issue22_gpu/gpu/t4-budget512-kv0-metadata.json" \
    --out "$issue22_tmp/candidate" > "$issue22_out/candidate-freeze.json"
"$issue22_repo/kern/model_quality" freeze --kind reference \
    --ids "$issue22_cpu/ids.u32" --logits "$issue22_tmp/cpu.f32" \
    --vocab-ids "$issue22_tmp/vocab.u32" --vocab-size 248320 \
    --raw-info "$issue22_cpu/capture_info.json" --metadata "$issue22_cpu/metadata.json" \
    --out "$issue22_tmp/reference" > "$issue22_out/reference-freeze.json"
"$issue22_repo/kern/model_quality" compare --candidate "$issue22_tmp/candidate" \
    --reference "$issue22_tmp/reference" --diagnostic --top-k 10 \
    --out "$issue22_out/comparison.json"
cp -- "$issue22_tmp/candidate/manifest.json" "$issue22_out/candidate-manifest.json"
cp -- "$issue22_tmp/reference/manifest.json" "$issue22_out/reference-manifest.json"
sha256sum "$issue22_repo/kern/model_quality" "$issue22_repo/kern/model_quality.pl" \
    "$issue22_repo/validation/issue22/replay_reference.sh" > "$issue22_out/tool.sha256"
printf 'Recomputed four-position diagnostic; no model execution or quality acceptance.\n'
