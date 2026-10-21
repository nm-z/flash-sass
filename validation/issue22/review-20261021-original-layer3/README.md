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

## Scope

This fixture supplies original layer-3 attention weights to a deterministic
component comparison with fixed selected IDs. It can measure same-weight
attention arithmetic and K/V storage behavior once the GPU capture and
independent replay are added. The generated activations are synthetic, the
production sparse selector is not executed, and held-out model quality remains
unmeasured.
