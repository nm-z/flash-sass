# Direct-position QSA validation

`qsa_direct_test visible NBLK [steps=4]` uploads a synthetic historical cache once and runs the current position through activation quantization, original-input Q/K/V and indexer projections, INT8 KV append, indexer norm/RoPE/pooling, score scan, all five selection passes, gather, capped attention partials and merge. Descriptor assertions reject a dense-attention substitution, a missing stage or a mismatched budget. CUDA executes the normal persistent dispatcher.

The default inputs are fixed FP32[2560] activations and deterministic BF16 projection matrices. Q/K/V norm vectors are FP32[256], indexer norm vectors FP32[128], and the indexer output is four query heads plus one raw key of 128 values. Historical KV is derived from deterministic raw activations through the same norm/RoPE/INT8 contract; pooled keys and the four-key ring are seeded consistently. Future raw-cache slots contain distinguishable finite sentinels. Historical state is synthetic, not a real model prefix.

The patched MaxAs checkout described in the repository README must already exist at `maxas/`. The build receipt preserves its upstream commit, exact Perl source hashes and local patch, together with the repository's `maxas-patch/` inputs.

```sh
QSA_EVIDENCE=$(mktemp -d)
perl kern/build_qsa_direct.pl --out "$QSA_EVIDENCE/build"
CUDA_VISIBLE_DEVICES=GPU-... perl kern/run_qsa_direct_matrix.pl \
  --build-manifest "$QSA_EVIDENCE/build/build.json" \
  --gpu-lock /path/to/shared/GPU-....lock --out "$QSA_EVIDENCE/matrix"
```

Use a full M60 GPU UUID and the same per-UUID lock file as the other workers. The runner waits for the exclusive reservation, then checks `nvidia-smi` and `ps` before every GPU case. It waits while the selected GPU has a compute process or nonzero utilization; it never signals another worker. The resident model on other GPUs can keep its allocations.

The build wrapper requires committed source, forces a rebuild of both the executable and cubin, and records the full source revision, source hashes, toolchain, build command, output and exit status. The matrix requires that build receipt, verifies the source and target hashes, and checks the compiled source banner in every case. An old executable or cubin cannot silently supply evidence for changed source. Evidence-only commits may follow a run when all recorded source hashes remain identical.

The full matrix covers N=1, 2047..2058, 3100..3103, 8000..8003, 65000..65003, 262000..262003 and 262141..262144 with production block budget 512 and both pinned-host and VRAM KV. It also spans the 128-position packed-pool boundary, every 256-key attention tile boundary through 2048 keys, and N=16383..16391, where complete blocks cross 4096 and the indexer needs its next scoring/selection chunk. These are correctness checks; recorded wall durations are not benchmark results.

The remaining cases check sequential seeding for a 64-token prefix, additional seeds, equal/adjacent cutoff scores and oldest/newest/clustered/scattered selections. Pattern cases begin at N=3101 so current append does not replace a pooled key before the first check. The adjacent fixture puts exactly the block budget above the lower adjacent score, then independently ranks actual GPU scores and checks that the last selected and first excluded scores differ by one FP32 bit. The equal fixture checks equal boundary scores and ascending-ID tie order. Both boundary records are required by the matrix. `--quick` is explicitly a subset. Negative controls inject selected-ID, gather and final-output mismatches and require exit 3.

For each position the test compares projection results, normalized attention and indexer queries, the ring, current INT8 code bytes and FP16 scale bits, the newly completed packed pool group, score bits, selected IDs, every physically gathered K/V row (including the final partially visible block), partial maxima/denominators/weighted sums, and final output. Selection uses an independent score-descending/ID-ascending sort in the scalar reference; selected IDs are then sorted into source order. Prefix and suffix canaries surround every fixture allocation. Failure of a comparison or guard returns nonzero. The matrix independently verifies reported key counts and requires actual exclusion whenever complete blocks exceed the budget.

`QSA_REPLAY_PREFIX=1` validates cache construction by replaying up to 64 historical tokens through the append/indexer kernels. Before the current chain runs, it compares all historical INT8 codes and scale bits, the complete four-key ring, and the packed pooled-key state with direct seeding. It then checks the current connected chain against that same direct-seeded expected state. Pool-pattern overrides are explicitly synthetic and are disallowed with replay. `QSA_DIRECT_SEED` is uint32; `QSA_HOSTKV` accepts only 0 or 1. Patterns are `hash`, `oldest`, `newest`, `clustered`, `scattered`, `equal` and `adjacent`.

## External inputs and exact numerical scope

`QSA_INPUTS` supplies `steps` little-endian FP32[2560] inputs. `QSA_MODEL_CACHE` plus `QSA_BLOCK` and `QSA_TYPES=q,k,v` supplies the holder's repacked projection weights. File sizes and K=2560-supported formats are checked. Indexer weights remain BF16. These file fingerprints identify the supplied cache; they do not establish original-checkpoint/conversion provenance.

`QSA_PROJECTED` supplies `steps` complete `direct_frame` records and disables the projection stage. This mode is labeled `POSTPROJECTION_DIAGNOSTIC`; it must not be used as evidence that original-input projections were validated. It is mutually exclusive with original-input files/cached weights. Norm weights must remain unchanged across frames.

Byte comparisons apply to the repository's explicitly ordered FP32 instruction twins, INT8 storage and FP16 scales/pool, built with `-ffp-contract=off` and explicit `fmaf` where specified. They do not assert bit equality with PyTorch/BF16 checkpoint arithmetic, dense attention, or different tile/reduction orders. The runner preserves raw logs, commands, inputs/settings, exits, GPU identity and binary/cubin hashes. Whole-model sparse-path parity, checkpoint quality, physical profiler counters and performance remain separate validations.
