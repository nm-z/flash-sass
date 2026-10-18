# Issue #11 follow-up evidence

Refs #11. PR #31 remains draft because current target execution and independent
checkpoint acceptance have not been measured. This directory records new work
on top of the earlier `native/` evidence; it does not replace historical results.

## Tested code and changes

The tested source commit is `705f16640e499c1ffc3a7f3fd1015471b64e221f`.
Merge commit `b2002a713d1ae74ff75765604e7e60706ffd8482` incorporated master
`3d57d734d63aa74f167a6b5a02c616ddc018d2ff`, resolving the Makefile conflict
while preserving both the oracle and checked-boundary targets.

The follow-up adds an explicit local byte-parity gate alongside the independent
numerical limits. In target mode it compares the observed index query, FP16
pool, scores, selected IDs, attention partials/output, and quantizer codes/scales
with the instruction-order scalar twins on identical inputs. Mismatches remain
failures even when the external numerical tolerance passes.

Quantizer fixtures now route through the production ATTNQKQ V quantizer in
target mode. Its q8 instruction sequence is also used for K. This isolates the
quantization boundary; it does not measure nonzero K normalization/RoPE or the
complete append/gather path. Dequantization of observed codes/scales occurs on
the host. The adapter uses 512-row batches and checks unwritten sentinels.

## Measured results

| Gate | Result | Evidence |
|---|---|---|
| Portable full native suite | 58 invocations matched their expected outcomes; exit 0 | [CPU report](cpu/report.json), [run log](cpu/run.log) |
| Source/build/host stability during CPU run | Verified; no changed sources | [CPU report](cpu/report.json) |
| Actual score-NRMS negative control | Required exit 3; NRMS 0.00692820323027551 exceeds 0.005 | [CPU report](cpu/report.json) |
| One-ULP and signed-zero negative controls | Required exit 3; both exact comparisons fail while the numerical limit passes | [Parity report](cpu/negative-parity.json), [diagnostics](cpu/negative-parity.stderr) |
| Full quantizer boundary/batch fixtures | CPU pass | [Quantizer report](cpu/quantizer.json), [retained arrays](cpu/quantizer-fixtures.tar.gz) |
| SM52 production image and target C adapter | Build only; exit 0 | [Build identity](build/identity.json), [adapter build log](build/oracle-build.log), [image build log](build/cubin-build.log.gz) |
| Shared GPU availability | Blocked; exit 75 after a 30-second wait; no GPU command started | [Gate record](gpu-gate/check.json) |
| Installed GGUF shards | Both full-file SHA-256 hashes match the pinned release | [GGUF identity](gguf-identity.json) |
| Independent checkpoint conformance | Not measured | [Summary and outstanding acceptance](summary.json) |

The native run took 61.36 seconds on engi. Its largest enforced score NRMS
among positive cases was 0.000157283398165829 at generated length 2048,
below the predeclared 0.005 limit. The matrix includes 13 pinned external
fixtures, 23 index/sparse lengths through 262003, 18 same-ID attention lengths
through 8193, host arithmetic/domain checks, the quantizer matrix, and two
required failing controls.

The quantizer capture includes adjacent FP32 input amplitudes whose full
FP32 scale multiplication rounds to FP16 zero and the smallest positive
subnormal. It also covers signed zero, the largest admitted amplitude,
invalid input, and group counts 1, 15, 16, 17, 8191, 8192 and 8193.
The archive contains the actual arrays and `fixture.json`; their original
per-file identities appear in [the artifact manifest](cpu/artifacts.json).
Large index/attention buffers remain outside git and are represented by
measured hashes. This does not claim that all generated buffers are checked in.

## Target and checkpoint limitations

The SM52 image was assembled at merge commit `b2002a7`; its relevant SASS,
PTX and Makefile inputs are unchanged at tested commit `705f166`. The image
SHA-256 is `116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14`.
The target adapter linked against CUDA stubs for compilation only. No target
fixture pass or device byte-parity result is claimed.

The gate checked archy from 2026-10-18T03:36:01Z through 03:36:32Z.
Resident holders and flashcore were active; a separate observation at 03:37:26Z
found another validator holding `/tmp/flash-sass-engine-validation.lock`.
No resident process was stopped and no GPU command started. Published process
evidence contains only PID, parent PID and command names, plus nvidia-smi output.

Matching the two installed GGUF shards to release revision
`ed59f92082b1e93c0e96d60a8b11aab089b52f09` establishes their artifact identity.
It does not verify the original checkpoint/tokenizer revision, exact conversion
source/configuration, or gamma conversion convention. Issue #11 still needs
those identities and independent full-checkpoint teacher-forced logits plus
per-layer QSA selection/output diagnostics at the required sparse lengths.
The existing logit-checker synthetic evidence remains under `../native/`; its
checker was unchanged in this follow-up and no new model-logit run is claimed.

## Reproduce

From the repository root, using C and Perl:

```sh
perl kern/qsa_oracle.pl --full --out /path/to/new-run
```

Use a disk-backed destination with room for generated buffers. The driver
rebuilds the default executable and records source, binary, compiler, host and
artifact identities. Follow [NUMERICAL_VALIDATION.md](../../../NUMERICAL_VALIDATION.md)
for the production-image build, shared-lock target gate, pinned reference
identity and independent checkpoint acceptance criteria.
