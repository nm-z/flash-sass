# Issue #22 review evidence — f6ddcd5

**The CPU checks pass. Issue #22 is not resolved and PR #46 remains draft.**
The original-checkpoint quality evaluation, real frozen-layer GPU comparisons,
and production-budget sparse length grid remain unmeasured.

## Tested source and commands

The tested source revision is
`f6ddcd5d95f0a9219809b9361d7137a97eb35b3b`, including parent
`8ab8cc5daa8e6a9a7d90b28c71f815036ba9c6ea`. `source-before.sha256` and
`source-after.sha256` agree. `source-status.txt` records only the pre-existing,
untracked `validation/issue22/revision-review/` directory; those earlier files
were preserved and were not inputs to these checks.

The complete CPU runner command was:

```bash
bash validation/issue22/run_cpu.sh /home/nate/codex/flash-issue22-20261018.P8ncez/cpu-final
```

`commands.txt` gives every constituent command and `exits.tsv` records exit 0
for all of them. The checks ran on engi, without a model or GPU execution.

| Check | Result |
| --- | --- |
| Perl teacher-forced comparator | 74 assertions passed |
| C layer exporter with CPU readback mock | 48 assertions passed |
| Frozen indexer replay and numerical gates | 36 assertions passed |
| C analytic indexer selftest | 16 assertions passed |
| Historical GPU placement comparisons | Both T=4/budget512 and T=9/budget1 pairs replayed successfully |
| Recovered CPU/GPU logit comparison | Four positions, 248,320 vocabulary entries per position |
| Synthetic indexer matrix | All 14 lengths completed |
| Model CLI before CUDA initialization | 32 checks passed with CUDA intercepted |

The synthetic lengths are 1, 2047, 2048, 2049, 2050, 2051, 2052, 2053, 2054,
2055, 3100, 8000, 65000 and 262000. Their JSON reports retain
`input_origin:synthetic_cpu_fixture` and `gpu_execution_performed_by_replay:false`.
The dense-to-sparse boundary and sparse tail metadata are exercised, but these
CPU fixtures do not prove a GPU production dispatch. The scalar generator's
large inputs and numeric sidecars remain outside Git; the committed generator,
commands, input and output hashes reproduce them. Only compact reports are
retained here.

The model integration commands were:

```bash
LIBRARY_PATH=/opt/cuda/lib64/stubs make -B -C kern model_test FLASH_REV=f6ddcd5d95f0a9219809b9361d7137a97eb35b3b
CUDA_STUB_LIBRARY=/opt/cuda/lib64/stubs/libcuda.so bash kern/test_model_cli.sh
```

The second command intercepts `cuInit`; it does not initialize a GPU. See
`model-final-build.log`, `model-final-cli.log`, their exit records and
`model-final.sha256`. The capture paths reject missing and extra IDs before CUDA.

`reference-build-current.log` records a successful C adapter build against the
separate engi llama.cpp checkout at
`fcac19079f3f2985fe74ed845f0880e7e9a8030a`. This is build compatibility evidence
only, and must not be attributed to the recovered archy process or its library
revision. The adapter source is unchanged between that build and the tested
source revision.

## Measured short backend diagnostic

`recovered-reference/comparison.json` compares the historical GPU capture from
source `06968d1a5d6d7fdaf12106236cc39a1cb5cdf389` with the recovered CPU llama.cpp
capture in `../archy-llama-7182ef7`. Both use the four IDs `10,11,12,13`, and the
recorded GGUF hashes match. The same-implementation host/VRAM GPU captures are
byte-identical; this cross-backend comparison measures a different question.

| Metric | Measured value |
| --- | ---: |
| Logit values compared | 993,280 |
| Logit RMSE | 0.186845505716465 |
| Centered logit RMSE | 0.183368965086050 |
| Maximum absolute logit error | 1.18950200080872 |
| Mean reference-to-candidate KL | 0.0210943973410854 |
| Top-1 agreement | 1.0 (4/4 positions) |
| Mean top-10 overlap | 0.85 |
| Mean next-token NLL increase | 0.155291599652249 (3 labels) |

No numerical acceptance threshold was applied to this exploratory comparison.
It reports `diagnostic_missing_provenance`, `quality_pass:null`, and
`independent_checkpoint_evidence:false`. The input microfixture is not held-out
language. Original checkpoint/tokenizer/conversion provenance and the historical
process's exact loaded backend identity are unverified; see the recovered
capture README for the source and library inspection timing. These numbers do
not isolate QSA, KV precision, activation quantization, or model changes #1/#2.

## GPU blocker and remaining acceptance

`gpu-preflight-final.log` records the final attempt beginning at
2026-10-18T03:17:08Z. Waiting 45 seconds for
`/tmp/flash-sass-engine-validation.lock` returned exit 75. The lock remained held
by the issue #19 validation process, PID 4010221. Its process and the resident
engine were left running. No new GPU workload was launched for this PR review.

`acceptance.json` records the remaining measurements explicitly: the real GPU
layer export and length grid; same-input/same-selection attention comparisons;
separate activation/projection and INT8-KV effects; an original-checkpoint
conversion comparison; predeclared held-out loss/perplexity and task accuracy;
and separate free-running generation with controls for other model changes.
The discovered public original-checkpoint revision in
`../reference-discovery.json` was not executed. Tooling and synthetic passes do
not substitute for those measurements.
