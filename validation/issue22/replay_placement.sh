#!/usr/bin/env bash
set -euo pipefail
# Replay committed placement captures through the native comparator.
# No model is loaded; this does not create an independent reference.
if [[ $# != 1 ]]; then
    printf 'usage: %s NEW_OUTPUT_DIRECTORY\n' "$0" >&2
    exit 2
fi
quality_repo=$(cd -- "$(dirname -- "$0")/../.." && pwd)
quality_out=$1
if [[ -e $quality_out ]]; then
    printf 'output already exists: %s\n' "$quality_out" >&2
    exit 2
fi
mkdir -p -- "$quality_out"
quality_out=$(cd -- "$quality_out" && pwd)
quality_tmp=$(mktemp -d)
trap 'rm -rf -- "$quality_tmp"' EXIT
quality_evidence="$quality_repo/validation/issue22/archy-06968d1"
make -C "$quality_repo/kern" model_quality
for quality_tag in t4-budget512 t9-budget1; do
    for quality_host in 0 1; do
        quality_raw="$quality_evidence/gpu/$quality_tag-kv$quality_host-raw"
        quality_logits="$quality_tmp/$quality_tag-kv$quality_host.f32"
        # Verify only the inputs used here against their recorded identities.
        perl -MDigest::SHA -MJSON::PP -e '
          my($root,@paths)=@ARGV;
          open my $f,"<","$root/IDENTITIES.json" or die $!;
          local $/; my $m=decode_json(<$f>); close $f;
          for my $p(@paths) {
            open my $in,"<:raw","$root/$p" or die "$p: $!";
            my $h=Digest::SHA->new(256); $h->addfile($in); close $in;
            $h->hexdigest eq $m->{$p}{sha256} or die "changed input: $p\n";
          }
        ' "$quality_evidence" \
          "gpu/$quality_tag-kv$quality_host-raw/logits.f32.gz" \
          "gpu/$quality_tag-kv$quality_host-raw/ids.u32" \
          "gpu/$quality_tag-kv$quality_host-raw/capture_info.json" \
          "gpu/$quality_tag-kv$quality_host-metadata.json"
        gzip -dc -- "$quality_raw/logits.f32.gz" > "$quality_logits"
        quality_kind=candidate
        [[ $quality_host == 1 ]] && quality_kind=reference
        "$quality_repo/kern/model_quality" freeze --kind "$quality_kind" \
          --ids "$quality_raw/ids.u32" --logits "$quality_logits" \
          --vocab-size 248320 --raw-info "$quality_raw/capture_info.json" \
          --metadata "$quality_evidence/gpu/$quality_tag-kv$quality_host-metadata.json" \
          --out "$quality_tmp/$quality_tag-kv$quality_host-frozen" \
          > "$quality_out/$quality_tag-kv$quality_host-freeze.stdout"
        cp -- "$quality_tmp/$quality_tag-kv$quality_host-frozen/manifest.json" \
          "$quality_out/$quality_tag-kv$quality_host-manifest.json"
    done
    "$quality_repo/kern/model_quality" compare \
      --candidate "$quality_tmp/$quality_tag-kv0-frozen" \
      --reference "$quality_tmp/$quality_tag-kv1-frozen" --diagnostic \
      --max-rmse 0 --max-mean-kl 0 --min-top1-agreement 1 \
      --out "$quality_out/$quality_tag-comparison.json"
done
sha256sum "$quality_repo/kern/model_quality" "$quality_repo/kern/model_quality.cc" \
  "$quality_repo/kern/json.c" > "$quality_out/native-identities.txt"
printf 'PASS: both recorded placement pairs compared across the full vocabulary\n'
