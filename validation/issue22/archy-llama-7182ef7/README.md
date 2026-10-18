# Recovered CPU llama.cpp capture

This is a recovered historical four-position capture for an exploratory
same-quantized-GGUF backend comparison. It is not a new model run or an accepted
original-checkpoint quality evaluation.

## Recorded process

The files were recovered from archy at
`/home/nate/claude/flash-sass-prs/issue22-reference-review-20261017`.
`reference.exit` records exit 0. `capture_info.json` records completion of four
teacher-forced input IDs `10,11,12,13`, each followed by 248,320 FP32 logits.
The settings are CPU-only llama.cpp, no GPU/op/KQ/V offload, FP32 K/V caches,
two threads, context capacity 256, no inserted tokens and 535.857938 seconds
elapsed. The recovered stderr records the 48-layer architecture and native QSA
nodes at its attention layers. It does not measure production-budget sparse
exclusion at this length. The consecutive token IDs are a microfixture, not a
held-out language sample. The stored chat template was not applied to them.

The raw logits SHA-256 is
`75acec6af6dd43c3a30fe720293a7b405e9718e55b03facdb627db17ade4915e`.
The deterministic gzip files preserve the logits, complete vocabulary IDs,
ordered vocabulary text and stderr. No binary or model weights are added to Git.

## Identity and historical limits

The recovered `model_reference.c` SHA-256 is
`f3b352250c81482f1a8bb774237c6f11c6d996509094c4fc079339472793a284`,
matching this branch's C adapter source. The recovered executable file SHA-256 is
`54eba33c757c5d692e524464fdbf2c0d6d2225371d54e416a076af5babfdd920`.
The executable was not rebuilt or rerun in this recovery.

`inspection-after-run.txt` records an inspection on 2026-10-18. At that time,
`ldd` resolved the executable's libraries to the `llama.cpp-up` build and that
source worktree was clean at
`7182ef71e6d5dd0c8ea76486fac134fcd25025d4`. The inspection includes source,
header and library hashes, build settings, and hardware information. These
identify files present during recovery. There is no contemporaneous loaded
library manifest or source-to-binary build attestation for the historical
process, so its exact loaded revision remains unverified.

The recovered post-run hashes of both GGUF shards match the hashes retained
with the historical flash-sass captures in `../archy-06968d1/gpu/weights-sha256.json`.
`weights-hash.started` and `weights-hash.finished` retain their recorded timing.
This match does not prove that model files stayed unchanged during the earlier
CPU process. Original checkpoint revision, tokenizer and conversion provenance
remain unverified. `metadata.json` records these limits instead of substituting
GGUF vocabulary hashes for an original tokenizer identity.

## Recompute the diagnostic

From the repository root:

```bash
bash validation/issue22/replay_reference.sh /absolute/path/new-reference-report
```

The script verifies the committed inputs, decompresses both sets of logits,
freezes their manifests, aligns the exact input and vocabulary IDs, and compares
all 993,280 values with the current Perl comparator. Its candidate is the
historical GPU source `06968d1a5d6d7fdaf12106236cc39a1cb5cdf389` at T=4 and
budget 512; its reference is this recovered CPU capture. It runs in diagnostic
mode without acceptance thresholds. Its results apply to these files and four
positions only. They cannot isolate QSA, INT8 KV, activation quantization,
conversion effects, or independent model changes tracked in #1/#2.

`SHA256SUMS` binds the archived inputs and provenance files. A checksum proves
artifact identity, not independent execution or checkpoint quality.
