# GPUQ lock handoff and one-die replay

This run verifies the corrected GPUQ lock handoff and captures a short synthetic
attention fixture. It is component evidence; it does not execute the production
indexer, a full model, or the original checkpoint.

## Execution

- GPUQ job: `e2da7237-e204-4174-ad24-775c1d3f83a5` (`fs22-attn-smoke-r3`), device 6.
- The fixture executable was built from source revision
  `6b52888d5fd354184a2af80671be2309c92f7f3f`. The corrected wrapper and runner
  were copied from PR source `4d577742662a0a94b68f88d713331e9e4758b318`.
- The GPUQ log records `using inherited GPUQ_LOCK_FD=9 lease`; there is no
  nested `flock` in this path. `gpu.exit` is 0, `complete.json` marks the
  capture complete, and `capture.sha256` verifies every archived capture file.
- The queued command used the committed `run_attention_gpu_smoke.sh` wrapper
  and `run_attention_gpu.sh` runner with `ISSUE22_GPU=6`, `ISSUE22_TOTAL=9`,
  `ISSUE22_BUDGET=1`, and `ISSUE22_LENGTHS=1,4,8,9`. GPUQ retains the exact
  argument vector; its temporary `/tmp/fs22-qsa-attn-6b52888-build-2644923`
  inputs and failed-job staging directory were removed after copying the
  outputs.
- GPUQ itself reports failed / exit 70 because its artifact importer raised
  `NoMethodError: undefined method 'include?' for nil` after storing the
  capture completion document. I retrieved that document with `gpuq document`
  and restored it before verifying the capture hashes. The importer failure is
  reported in [gpuq #51](https://github.com/nm-z/gpuq/issues/51); see
  [gpuq-job.json](gpuq-job.json) for machine-readable execution state.

## Independent numerical replay

The CPU replay binary was compiled and all four replay profiles were run on
Benji through DSH build
`build-2894068-1792567318732828928` (exit 0). The predeclared arithmetic limits
were max absolute error `0.005`, RMSE `0.001`, relative L2 `0.00005`, and query
max absolute error `0.00001`. All four replays report
`arithmetic_limits_met`:

| N | Supplied block IDs | Tail | Attended keys | Max absolute | RMSE | Relative L2 | Query max absolute |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `[]` | 1 | 1 | 9.4132e-8 | 1.5123e-8 | 4.8081e-8 | 1.8917e-7 |
| 4 | `[0]` | 0 | 4 | 1.5174e-7 | 1.6697e-8 | 1.1072e-7 | 1.9598e-7 |
| 8 | `[0]` | 0 | 4 | 1.1710e-7 | 1.7063e-8 | 1.1265e-7 | 2.3266e-7 |
| 9 | `[0]` | 1 | 5 | 1.0924e-7 | 1.5154e-8 | 1.1264e-7 | 1.9904e-7 |

The compact results are in [replay-summary.json](replay-summary.json); the
complete replay JSON is retained as gzip files under `replays/`, with their
hashes in [replay-sha256.txt](replay-sha256.txt). Original GPU captures,
metadata, the full capture hash list, and before/after source hashes are also
retained here.

## Limits of this evidence

The fixture uses generated projections and externally supplied IDs with a
budget-one override. It verifies GPU attention arithmetic against an independent
CPU FP64 replay for these four short lengths. It does not establish independent
selection, production sparse dispatch, teacher-forced logits, original-checkpoint
agreement, or held-out long-context quality. Issue #22 remains unresolved.
