# M60 target validation at ca4a476

The current numerical adapter was executed on an actual Tesla M60 after
joining the shared GPU-validation lock queue. This is a measured target pass,
not the earlier build-only result. It does not establish full-model or
original-checkpoint conformance; a subsequent full-model capture probe failed
to allocate its GPU buffers.

## Executed identity and scope

| Item | Recorded value |
|---|---|
| Tested source | `ca4a47699b87e5214e3b20aa4f84ec0754da8ccf` |
| Controller | `engi` |
| GPU host | `archy` |
| Physical device | M60 ordinal 6, `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` |
| Child CUDA ordinal | 0, with visibility restricted to that device UUID |
| Global lock | `/tmp/flash-sass-engine-validation.lock` |
| Queued | 2026-10-18 06:12:01 UTC |
| Lock acquired / suite started | 2026-10-18 06:14:42 UTC |
| Suite finished | 2026-10-18 06:15:49 UTC |
| Queued wait | 161 seconds from recorded timestamps |
| Measured suite duration | 66.2644050121307 seconds |
| Exit | 0 |
| Result | `PASS_TARGET_SYNTHETIC` |

The production image SHA-256 is
`116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14`.
The complete build log, CUDA 11.4 assembler/compiler identification, generated
image/header/source hashes, patched MaxAs hashes and target executable hash
are retained under `build/`. The runner confirmed unchanged sources, executable,
image, compiler, kernel, host and boot identity throughout the suite. Its
tracked worktree was clean at the recorded start and end.

## Measured outcomes

| Check | Result |
|---|---|
| Entire suite | 58 of 58 expected outcomes passed |
| Target-bearing invocations | 54 |
| Actual CUDA kernel launches | 1898 |
| Target GPU/scalar exact byte comparisons | 707 |
| Target byte-parity failures | 0 |
| Maximum scaled-score normalized RMS | 0.000157283398165829 |
| Predeclared scaled-score normalized RMS limit | 0.005 |
| CPU-only gate tests | 32 passed, including real flock queue and timeout tests |
| Native CPU suite after the gate change | 58 expected outcomes passed |

The 58 outcomes include four CPU-only invocations: host arithmetic/domain
checks, the required failing score control, the required failing one-ULP and
signed-zero parity control, and the frozen BF16-boundary check. The two
negative controls return their required exit 3. They are not target failures
and are not counted in the 707 target byte comparisons.

Target coverage includes production query normalization/RoPE and pooled keys,
scores and selected IDs, histogram handoff, quantizer codes and FP16 scales,
and attention partials/merge output. All fixed external numerical gates and
local exact-parity checks remain unchanged. The INT8 quantizer boundary and
batch fixtures execute the production ATTNQKQ V sequence; the test still does
not claim the complete normalized-K append/gather path from that isolated
quantizer measurement.

Generated indexer/sparse lengths are:
`1`, `2047..2052`, `3100..3103`, `8000..8003`, `65000..65003`, and
`262000..262003`. This covers all modulo-four tails at the long positions.
Same-ID attention lengths are `1..4`, `1023..1027`, `2047..2052`, and
`8191..8193`. Thirteen pinned external fixtures are also replayed. These are
synthetic or frozen numerical fixtures, not newly measured long-context
teacher-forced checkpoint examples.

## Queue and device-scoping fix

The old gate checked whether the entire GPU host was empty before attempting
a nonblocking lock. Resident weights on unrelated dies therefore prevented
this single-device test from ever registering a useful wait or running.

The revised gate uses blocking `flock` on the same shared lock, resolves an
explicit physical device to its UUID, rechecks its identity and compute users
under the lock, and exposes only that GPU to the child. The whole-host idle
rule remains the default when no device is specified. Device 6 had no CUDA
compute user at the checked start. Resident processes on other devices were
not stopped or reconfigured. The gate also bounds its own child with a
recorded timeout and preserves the actual exit code.

Reproduce from the tested source with CUDA 11.4, the repository's patched
MaxAs checkout, Perl and a C compiler:

```sh
make -C kern die.cubin dieops.h qsa_oracle_gpu CUDA_LIBDIR=/usr/lib
bash kern/qsa_oracle_gpu_gate.sh --device 6 --wait 600 --run-timeout 240 \
  --out /absolute/new-gate-output -- \
  perl kern/qsa_oracle.pl --full --gpu kern/die.cubin \
  --binary kern/qsa_oracle_gpu --device 0 --out /absolute/new-target-output
```

The physical ordinal is specific to this measured host. It must identify an
appropriate idle M60 on another system. Never bypass the shared lock because
an individual GPU has free memory.

## Preserved artifacts

`gpu/report.json` retains all commands, expected/actual exits, metrics,
provenance and source/image hashes. `gpu/checks/` contains every child JSON
report and stderr. `gpu/artifacts.json` identifies all 1347 run artifacts.
`gpu/pinned-and-quantizer-captures.tar.gz` preserves the full frozen replays and
quantizer boundary/batch arrays, including observed target outputs.

The entire approximately 1.2-GiB run, including the generated long-case inputs
and device outputs, was copied to persistent storage on `engi` at:

`/home/nate/codex/issue11-continue-XDHJ8Fal/target-evidence/gpu-1`

All 1347 files were verified there against the run's size and SHA-256 catalog.
The large generated arrays are not claimed to be committed to Git; their
identities are preserved in the catalog and they can be regenerated using
the fixed generator and command above. `SHA256SUMS` identifies the committed
evidence files. No executable, model weight or driver library is committed.

## Current full-model blocker: measured allocation failure

After the target pass, a separate fresh full-model capture probe also joined
and acquired the global lock. It used the pinned PR #46 capture adapter at
`d02d05738deedca561a7ff84e41b9cd195f471d0` with this PR's exact production
`die.cubin`. `prog.c`, `prog.h`, generated opcode header and image were checked
byte-identical between those two private build trees. The adapter identity is
recorded separately rather than falsely attributing its source to PR #31.

The requested run was full blocks 0..48 with actual embedding/output head and
PLE table, exact teacher-forced IDs `10,11,12,13`, six existing resident weight
arenas, production block budget 512 and host-resident KV. It requested full
logits and layer-3 snapshots at visible lengths 1 and 4. It did not replace
GPU values with scalar reference outputs or reconfigure the resident model.

**Observed failure, exit 1:**

```text
model_test.c:73: cuMemAlloc(&p, bytes) -> out of memory
```

No completed logit or layer capture was produced. `model-probe/` preserves
the exact command, environment script, adapter build and hashes, timestamps,
GPU/process inventories and unaltered stdout/stderr. This is an allocation
blocker in the six-device resident configuration, not a numerical pass and
not a newly measured checkpoint mismatch. No resident process was stopped.

The PR therefore remains draft. Required next acceptance still includes a
successful current-image full-model capture with sufficient GPU memory,
verified original checkpoint/tokenizer and exact conversion/gamma lineage,
and the paired per-layer long-sparse model evaluation under the predeclared
limits. The earlier failing comparison against historical GPU arrays remains
historical; this target-kernel pass does not replace that missing model gate.
