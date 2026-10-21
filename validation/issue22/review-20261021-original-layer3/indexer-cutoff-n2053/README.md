# Original layer-3 indexer at the 512-block cutoff

This is a layer-only cutoff witness using original BF16 indexer weights and
deterministic synthetic hidden inputs. It is not a model forward or a quality
measurement.

## Fixture and reference replay

- Safetensors revision: `de4b8e4d43b917e7706784d8bb445c9af86a3540`; layer 3
  `index_qk_proj`, Q norm, and K norm tensors come from shard 82. Their hash
  and exact ranges are in [the tensor manifest](../../original-layer3-attention-tensors.json).
- The C fixture streamed 2,053 generated hidden vectors through the original
  BF16 indexer projection on Benji via DSH build
  `build-3635578-1792573848926883475`. It produced 513 complete four-token
  blocks plus tail token 2052.
- The independent FP64 replay ran on Benji via DSH build
  `build-3640384-1792573895949705691` with production budget 512. It selected
  512 blocks and excluded one. Canonical overlap against the FP64 ranking is
  512/512, exact-tie compatibility is true, and the cutoff is unambiguous with
  margin `0.0008201705`.
- All six checks in `model-indexer-limits.json` pass. Git history shows the
  limit profile was committed in
  `04e77e181fa6195f199ab384ad19e36fb18484d5` before this run; the replay
  manifest still records `predeclaration_attested: false`.

Machine-readable results are in [cutoff-summary.json](cutoff-summary.json),
the complete replay is in `replay/n2053-indexer-replay.json`, and all frozen
projection inputs and selection outputs are under `indexer-capture/`.

## GPU queue status and limits

GPUQ job `3a717105-29dd-4f61-be19-a9d882bf8385` (`fs22-layer3-indexer-cutoff-n2053-r1`)
is queued behind the existing shared-queue tickets. When admitted, its short
SASS test will run `idxqk`, `idxscore`, and `idxsel` on these frozen inputs.
This checks the kernel path near the cutoff; it still does not run the
production model dispatcher, original-checkpoint teacher forcing, or held-out
quality evaluation.
