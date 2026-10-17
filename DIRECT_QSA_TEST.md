# Direct-position QSA validation

`qsa_direct_test visible NBLK [steps=4]` uploads a synthetic historical cache once and runs the current position through activation quantization, original-input Q/K/V and indexer projections, INT8 KV append, indexer norm/RoPE/pooling, score scan, all five selection passes, gather, capped attention partials and merge. Descriptor assertions reject a dense-attention substitution, a missing stage or a mismatched budget. CUDA executes the normal persistent dispatcher.

The default inputs are fixed FP32[2560] activations and deterministic BF16 projection matrices. Q/K/V norm vectors are FP32[256], indexer norm vectors FP32[128], and the indexer output is four query heads plus one raw key of 128 values. Historical KV is derived from deterministic raw activations through the same norm/RoPE/INT8 contract; pooled keys and the four-key ring are seeded consistently. Future raw-cache slots contain distinguishable finite sentinels. Historical state is synthetic, not a real model prefix.

```sh
make -C kern qsa_direct_test die.cubin
cd kern
CUDA_VISIBLE_DEVICES=GPU-... python3 run_qsa_direct_matrix.py --out /tmp/qsa-direct-results
```

Reserve one idle GPU first. The matrix covers N=1, 2047..2058, 3100..3103, 8000..8003, 65000..65003, 262000..262003 and the last supported position 262144, with production block budget 512 and both pinned-host and VRAM KV. It also checks sequential seeding for a manageable 64-token prefix, additional seeds, equal/adjacent cutoff scores and oldest/newest/clustered/scattered selections. `--quick` is explicitly a subset. Negative controls inject selected-ID, gather and final-output mismatches and require exit 3.

For each position the test compares projection results, normalized attention and indexer queries, the ring, current INT8 code bytes and FP16 scale bits, the newly completed packed pool group, score bits, selected IDs, every physically gathered K/V row (including the final partially visible block), partial maxima/denominators/weighted sums, and final output. Selection uses an independent score-descending/ID-ascending sort in the scalar reference; selected IDs are then sorted into source order. Prefix and suffix canaries surround every fixture allocation. Failure of a comparison or guard returns nonzero. The matrix independently verifies reported key counts and requires actual exclusion whenever complete blocks exceed the budget.

`QSA_REPLAY_PREFIX=1` validates cache construction by replaying up to 64 historical tokens through the append/indexer kernels, then checks the current connected chain against the same direct-seeded expected state. Pool-pattern overrides are explicitly synthetic and are disallowed with replay. `QSA_DIRECT_SEED` is uint32; `QSA_HOSTKV` accepts only 0 or 1. Patterns are `hash`, `oldest`, `newest`, `clustered`, `scattered`, `equal` and `adjacent`.

## External inputs and exact numerical scope

`QSA_INPUTS` supplies `steps` little-endian FP32[2560] inputs. `QSA_MODEL_CACHE` plus `QSA_BLOCK` and `QSA_TYPES=q,k,v` supplies the holder's repacked projection weights. File sizes and K=2560-supported formats are checked. Indexer weights remain BF16. These file fingerprints identify the supplied cache; they do not establish original-checkpoint/conversion provenance.

`QSA_PROJECTED` supplies `steps` complete `direct_frame` records and disables the projection stage. This mode is labeled `POSTPROJECTION_DIAGNOSTIC`; it must not be used as evidence that original-input projections were validated. It is mutually exclusive with original-input files/cached weights. Norm weights must remain unchanged across frames.

Byte comparisons apply to the repository's explicitly ordered FP32 instruction twins, INT8 storage and FP16 scales/pool, built with `-ffp-contract=off` and explicit `fmaf` where specified. They do not assert bit equality with PyTorch/BF16 checkpoint arithmetic, dense attention, or different tile/reduction orders. The runner preserves raw logs, commands, inputs/settings, exits, GPU identity and binary/cubin hashes. Whole-model sparse-path parity, checkpoint quality, physical profiler counters and performance remain separate validations.
