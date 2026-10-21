# Original layer-3 QSA projection fixture

This is a partial same-weight component fixture made from the pinned original
BF16 checkpoint. It is not a model forward, teacher-forced continuation, or
quality run.

## Source weights and projection

- Checkpoint: `Qwen/Qwen3.8-Flash-Next`, revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540`.
- The Safetensors index maps all layer-3 self-attention Q/K/V and norm tensors
  to `model-00082-of-00131.safetensors`. Its verified SHA-256 is recorded in
  [projection-complete.json](projected-inputs/projection-complete.json). The
  complete 360 GB model remains unavailable.
- The corrected C extractor was built and run on Benji through DSH build
  `build-3238978-1792570309168879849`, source revision
  `647f0eccc19a03cb69e552a0a7f447ee61ab4434`. It converted the pinned BF16
  `q_proj`, `k_proj`, `v_proj`, `q_norm`, and `k_norm` tensors to frozen FP32
  projection inputs.
- The input is nine deterministic synthetic hidden vectors, seed 4242. The
  last vector supplies `qg.f32`; all nine supply `raw_k.f32` and `raw_v.f32`.
  `hidden_input.f32` retains the exact inputs. Projection input hashes are in
  [projection-sha256.txt](projection-sha256.txt).

The manifest records exact tensor shapes, file offsets, weights revision, and
weight hash. The projection artifact explicitly marks `teacher_forced`,
`model_forward`, `production_indexer_executed`, and `full_model` as false.

The first wired GPUQ attempt used a pre-fix extractor and exited on non-finite
projection input before CUDA initialization. It exposed a little-endian BF16
decode ordering error; commit `647f0eccc19a03cb69e552a0a7f447ee61ab4434` fixes
the conversion. The invalid projection files were replaced. That attempt is
not a GPU measurement.

## GPU capture and independent replay

- GPUQ job `eaf63d75-09a1-4b5a-9fb1-66d2bd0fc470` ran on device 6 with N=9,
  budget 1, and supplied selected ID `[0]`. The capture completed with tail 1
  and 5 attended keys. The GPUQ log confirms the runner reused inherited
  `GPUQ_LOCK_FD=9`; there was no nested lock acquisition.
- The GPU fixture exit code is 0; `capture.sha256` and
  `frozen-inputs.sha256` verify. GPUQ marked the overall job failed with exit
  70 after its artifact importer raised `NoMethodError: undefined method
  'include?' for nil`. That importer error is tracked in [gpuq #51](https://github.com/nm-z/gpuq/issues/51).
  The completed output was recovered from the GPUQ staging area and database.
- The independent CPU FP64 replay ran on Benji through DSH build
  `build-3304830-1792570891765182913` and reports `arithmetic_limits_met` with
  no failed fields. For N=9, arithmetic max absolute error is `3.8879e-9`,
  RMSE is `6.4200e-10`, relative L2 is `9.6767e-8`, and query max absolute
  error is `1.3782e-7`.
- The profile limits are max absolute `0.005`, RMSE `0.001`, relative L2
  `0.00005`, and query max absolute `0.00001`. The exact profile hash is
  `e6e6c337f222bfdef8eba689f307d6a39d6041464ac662d073a69687ba5b2f8e`; Git
  history shows it was committed in `affe73f3b62c94639875a1d5b2c5d6958752f7be`
  before this run. The replay report's own
  `historical_predeclaration_attested` field remains false, so the report does
  not claim that attestation.

Machine-readable capture and job state are in `gpuq-run-r2/`; the compact
metrics are in [replay-summary.json](replay-summary.json), and the full replay
is retained as [n9-replay.json.gz](replays/n9-replay.json.gz).

## Scope

This fixture supplies original layer-3 attention weights to a deterministic
component comparison with fixed selected IDs. It measures same-weight
attention arithmetic and K/V storage behavior for this synthetic activation
sequence. The production sparse selector is not executed, teacher-forced model
inputs are absent, and held-out model quality remains unmeasured.
