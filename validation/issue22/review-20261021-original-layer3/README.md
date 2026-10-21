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

## Original indexer score and cutoff component

The C extractor also projects the original layer-3 `index_qk_proj` and norm
tensors from shard 82. The same nine synthetic hidden vectors feed the
indexer query and keys; the C QSA twin produces two pooled keys, candidate
scores, and selected IDs. The independent FP64 indexer replay ran through DSH
build `build-3487257-1792572458629072170` and passed all six checks in the
predeclared `model-indexer-limits.json` profile. Top-1 overlap is 1.0, exact
tie compatibility is true, and the FP64 cutoff is unambiguous with margin
`0.03406928`.

Git history shows the numeric profile was committed in
`04e77e181fa6195f199ab384ad19e36fb18484d5` before this run. The replay
manifest still reports `predeclaration_attested: false`, so that field is not
claimed as an attested fact by the replay tool.

The GPUQ job `55818b8d-1b5a-4297-b0fe-28aee82465cb` then ran the production
SASS `idxqk`, `idxscore`, and `idxsel` tests against these same original-weight
projections. All three report `BIT-EXACT`; source and input hashes stayed
unchanged. GPUQ again marked the job failed with exit 70 after its importer
crashed, as tracked in [gpuq #51](https://github.com/nm-z/gpuq/issues/51).
The test binary was built on Benji through DSH build
`build-3499579-1792572563183695791` from `qsa_test.c` source revision
`5440dbbf31513fc160ee3e813c1fefe2aa7bb4d2`. Recovered logs and machine-readable
state are under `indexer-gpuq-r2/`;
[indexer-summary.json](indexer-summary.json) and
[n9-indexer-replay.json](indexer-replay-final/n9-indexer-replay.json) retain
the scores, margins, metrics, and gate details.

This does not run the model-level sparse dispatcher: the fixture has only two
complete blocks and uses a budget-one override, with synthetic hidden inputs.
The SASS operators are exercised on original layer weights, but long-context
selector coverage and model quality remain unmeasured.

## Scope

This fixture supplies original layer-3 attention weights to a deterministic
component comparison with fixed selected IDs. It measures same-weight
attention arithmetic and K/V storage behavior for this synthetic activation
sequence. The production sparse selector is not executed, teacher-forced model
inputs are absent, and held-out model quality remains unmeasured.
