# Hyper-connection phase work for issue 53

This is a partial implementation for [issue #53](https://github.com/nm-z/flash-sass/issues/53),
based on default-branch commit `b97b6b94bc0aba994830396fadebe5f8acd082f7`.
The full issue body, evidence, and acceptance criteria were read; the comments
endpoint returned no comments. [PR #76](https://github.com/nm-z/flash-sass/pull/76)
is open as a draft. Default-branch commit `3d57d734d63aa74f167a6b5a02c616ddc018d2ff`
was merged into this branch as `be5df804165f44e5fe74df98a6030445244837c4`;
that merge changed CI and the QSA test's position bound, not the HC runtime code.
The candidate **fails both measured production performance gates**: total HC
time reaches **8.496 ms/token** against the 4.0 ms limit, and short-context
decode averages **36.3167 tok/s** against the 38 tok/s requirement. The four
fixed-prompt output token streams match exactly. The documented `model_test`
regression passes for both baseline and candidate in free-run and loop modes;
its scope is blocks [0,4), not an all-48-block C-twin comparison. PR #76 remains
a draft because the two performance requirements are unmet.

## Changes

`hcwsum` assigns 160 columns to each of 16 work units, replacing 256 columns on
10 units. Each CTA has five active warps. The 16-CTA persistent kernel can use
all 16 M60 SMs for the 2,560-column write. The fixed-order sum of the dies' MoE
partials and the four stream-update FFMAs retain their original order.
Each warp evaluates the four independent write gates in parallel: lane `i`
evaluates gate `i mod 4`, then `SHFL.IDX` broadcasts gates 0 through 3. Each
gate uses the same sigmoid implementation and floating-point operations as
before. Whole-warp exits preserve all shuffle source lanes even for short tails.

The four-slice HC-down peer-output variants (`bf16hcx`, `bf16hcrx`, and
`bf16hcrwx`) collect eight unchanged row reductions in CTA-local shared memory,
then write eight contiguous floats per peer. CTA-uniform loop bounds and two
barriers protect partial row groups and scratch reuse. This changes the store
pattern without adding registers. The candidate persistent kernel uses 249
registers and 49,152 bytes of shared memory.

The number of global phases and the profiling names are unchanged: the
production trace still contains 242 HC phases per token. Build the
host phase builder and `die.cubin` together: an old host still dispatching ten
`hcwsum` units cannot drive the new 160-column layout correctly.

## Results

The isolated numerical checks ran on archy GPU 7, a Tesla M60 (`sm_52`, 16 SMs), on
2026-10-18 UTC. Every test acquired the shared validation lock and the existing
per-UUID reservation, checked `nvidia-smi` and `ps`, and waited when the device
or lock was busy. The resident engine and holders remained running.

| Check | Result |
| --- | --- |
| Existing `hcops_test` | SiLU, mix, and write all bit-exact |
| Existing `pair_bf16_test` | Zero failed cases, missing writes, or guard errors; unchanged paired kernels compared with the candidate scalar kernels |
| Existing `gemv_bf16_test`, `blk.4.ffn_gate_inp.weight` | All 512 rows bit-exact, K=2,560 |
| Existing `gemv_bf16_test`, `blk.4.hc_attn_down.weight` | All 320 rows bit-exact, K=10,240 |
| Existing `gemv_bf16_test`, `blk.4.hc_attn_up.weight` | All 10,240 rows bit-exact, K=320 |
| New persistent `hc_phase_test` | 102 cases, zero failures, each run against baseline and candidate |
| `model_test` | Baseline and candidate pass free-run and loop modes, four tokens each, embedding through blocks [0,4) and head, with the C twins enabled |

The new C test compares HCWSUM with an independent fixed-order peer sum and
the existing C stream-write twin, including logical die counts 1 through 6,
in-place and separate stream inputs, cancellation-sensitive sums, and tails
around 160- and 256-column boundaries. It compares the persistent HC-down
variants' complete output buffers with the pinned baseline, including XH,
SOUT, six distinct output slots, canaries, missing-write checks, repeated
launches, and varied row counts and work-unit counts. HCRWX's stream write is
also checked against the C twin.

These peer buffers are on one physical GPU. Descriptor exchange flags are
zero, so these checks cover opcode bodies, dispatch, and grid barriers, but
not physical peer transfers or the publication/wait protocol.

Evidence:

- [validation.log](validation.log): full locked suite output with trailing display spaces removed; all six test processes exited 0.
- [readiness.log](readiness.log): locked inspection after validation and resident-process state.
- [build.txt](build.txt): baseline revision, toolchain identities, source/binary SHA-256 values, and candidate resource use.
- [test-build.log](test-build.log): new C test compiler command and clean build.
- [gates/kernel-check-2700.log](gates/kernel-check-2700.log): the fresh rerun with a 2,700-second resource wait budget; all six processes exited 0 and all 102 persistent HC cases passed again.

## Measured live-engine gates — 2026-10-18

The live comparison ran from 05:32:22 to 05:33:04 UTC on archy's six-die,
48-layer resident engine. It acquired `/tmp/flash-sass-engine-validation.lock`
immediately, after issue #19's validation had finished. Both arms ran under
one lock, with the same core process, holder manifests, split, attention map,
KV/PLE mode, context capacity, image capacity, and prefill settings. Each arm
loaded its matching host library and cubin; the validator checked the reported
revision, cubin fingerprint, and actual mapped library bytes.

| Gate | Baseline | Candidate | Acceptance |
| --- | --- | --- | --- |
| HC sum, largest of six dies across three captures | 9.0819 ms | **8.4960 ms** | **FAIL:** at most 4.0 ms |
| HC sum, range across all 18 die/capture pairs | 8.2329–9.0819 ms | **7.7630–8.4960 ms** | Every candidate value exceeds 4.0 ms |
| Three measured decode requests | 35.77, 35.75, 35.74 tok/s | **36.31, 36.29, 36.35 tok/s** | Mean must reach 38 tok/s |
| Arithmetic mean decode speed | 35.7533 tok/s | **36.3167 tok/s** | **FAIL** |
| Four fixed-prompt output token streams | 90 generated IDs | Same 90 IDs, in the same four streams | **PASS** for these fixtures |
| Original engine restoration | Revision `edc66d9` | Same revision, cubin fingerprint, mapped library SHA-256 and core identity after restoration | **PASS** |

The maximum HC sum fell by approximately 6.45%, and mean decode throughput
rose by approximately 1.58% in this matched experiment. Those improvements
are insufficient for issue #53. Each capture contains `bf16hcup` ×97,
`hcwsum` ×48, `bf16hcrwx` ×48, and `bf16hcrx` ×49. The validator sums all HC
kinds independently on each die; it does not add concurrent dies' times
together. Even the fastest candidate observation exceeds the time limit.

Each arm used one HTTP warmup followed by three serial measured requests
with the exact prompt, token limit, and temperature from
`kern/linkmatrix_test.sh`. Every response generated 38 completion tokens.
`PROF_ALL=1 prof.sh` ran immediately after each measured response, with no
intervening generation. No PCIe link settings were changed.

The separate token-ID comparison used [prompts.json](prompts.json), greedy
sampling, identical raw input and stop IDs, disabled snapshot reuse, and
matching runtime settings. The C-code, capital-city, multi-turn arithmetic,
and number-sequence cases generated 21, 2, 3, and 64 IDs respectively. The
validator requires a complete terminal response and checks token counts,
stop status, prompt length, and absence of prefix reuse before comparing IDs.
This is a four-fixture production comparison, not a full-model numerical
oracle for every input.

The wrapper exited **1** because the two performance gates failed, with
restoration recorded separately as **PASS**. The original engine library and
cubin were saved before either reload and restored before releasing the lock.
The resident core and weight holders remained running. The restored engine
references the saved cubin path in the summary; those original images must
remain on archy while it is configured to use them.

Evidence and validator:

- [gates/live-summary.json](gates/live-summary.json): gate booleans, all per-die phase totals, requests, raw token arrays, runtime settings, source revisions, holder identities, and restoration result.
- [gates/live-raw.json](gates/live-raw.json): all 140 original text artifacts, with exact contents, byte counts, and SHA-256 values, including command arguments, responses, profiles, and readiness checks.
- [gates/profiles.txt](gates/profiles.txt): readable copies of all six complete profiling captures.
- [gates/live-driver.log](gates/live-driver.log), [gates/live-exit.txt](gates/live-exit.txt), and [gates/metadata.json](gates/metadata.json): execution timeline, process result, and artifact provenance.
- [engine_check.rb](engine_check.rb): locked baseline/candidate comparison and verified restoration.
- [engine_check_test.rb](engine_check_test.rb) and [gates/engine-check-cpu.log](gates/engine-check-cpu.log): CPU-only malformed-protocol checks and failure-injection checks for restoration after either arm fails and rejection of a restoration mismatch.

## Existing model-test regression — PASS

The model attempt completed from 06:13:53 to 06:18:06 UTC, including its lock
wait, holder loading, and cleanup. [model_check.sh](model_check.sh) acquired the
same shared engine-validation lock, checked GPU 7 under that lock, acquired
its existing UUID reservation, and loaded an owned holder for blocks [0,4).
It ran the existing `model_test` executable for both baseline and candidate,
with embedding, all four selected blocks, real PLE data, dense attention,
and the output head. Every run carried state across four token positions.
`MODEL_NOTWIN` and `MODEL_NPH` were unset, so the C comparisons ran and the
program was not truncated.

| Arm | Mode | Numerical result | Output IDs |
| --- | --- | --- | --- |
| Baseline | Free-running, `eh` | Four of four tokens exact; complete stream and token-result comparisons pass | 8678, 0, 44480, 1132 |
| Candidate | Free-running, `eh` | Four of four tokens exact; complete stream and token-result comparisons pass | 8678, 0, 44480, 1132 |
| Baseline | One-launch teacher-forcing, `ehl` | All four tokens and final `S.0` stream exact | 8678, 101308, 77620, 13051 |
| Candidate | One-launch teacher-forcing, `ehl` | All four tokens and final `S.0` stream exact | 8678, 101308, 77620, 13051 |

All four test processes, the wrapper, and its supervisor exited **0**. The
teacher-forcing inputs are arbitrary fixed IDs; the log's count of predictions
matching those inputs is not the numerical oracle. The GPU-to-C comparisons
are the exact comparisons reported above.

This reproduces the existing whole-token case documented in `SPEC.md`. It
covers one physical GPU and blocks [0,4); it does not establish an all-48-block
C-twin result or a six-die numerical comparison. The separate four-fixture
token-ID comparison above ran on the complete six-die, 48-layer production
engine. The documented `model_test` gate is satisfied for its executed scope;
no additional all-48-block numerical acceptance threshold is assumed.

The owned holder was stopped after the tests; its PID, the wrapper, supervisor,
and timeout process were all absent in the cleanup check. Resident `flashcore`
PID 3575268 remained running. No resident engine reload was needed for this
isolated regression.

- [model/results.txt](model/results.txt): complete stdout from all four successful model-test processes.
- [model/attempt-3-status.json](model/attempt-3-status.json): source revisions, wrapper identity, modes, IDs, results, coverage, and cleanup.
- [model/attempt-3-raw.json](model/attempt-3-raw.json): exact commands and text results with hashes; the ephemeral IPC manifest and unrelated host process inventory are represented by identity only.
- [model/attempt-1-status.json](model/attempt-1-status.json) and [model/attempt-2-status.json](model/attempt-2-status.json): earlier attempts that ran no GPU tests. The first command session ended without an exit result before acquiring the lock. The second was stopped with exit 143 before starting a holder so its repeated five-second lock waits could be replaced by one continuous wait. Their raw logs are retained beside the status files.

## Reproduction

Both baseline and candidate were built on engi with GCC 16.2.1, the same
CUDA 11.4.152 `ptxas`, MaxAs revision
`54eda7af086a46c9dae1688b691968235d560164`, and the repository's MaxAs
patches. The C build used engi's CUDA headers and driver link stubs;
`CUDA_VERSION` was 13040. Runtime validation used archy's driver 470.256.02.
The assembler emitted the repository's existing register-bank conflict warnings;
both complete builds exited 0.

From the candidate checkout, the exact build commands were:

```bash
make -C kern -j2 \
  PTXAS11=/home/nate/claude/flash-sass-resolve/issue-53-evidence/toolchain/ptxas \
  'CC=gcc -L/opt/cuda/targets/x86_64-linux/lib/stubs' \
  die.cubin hcops pair_bf16_test gemv_bf16_test \
  gemv_bf16_k2560.cubin gemv_bf16_k320.cubin gemv_bf16_hc.cubin model_test
make -C kern 'CC=gcc -L/opt/cuda/targets/x86_64-linux/lib/stubs' hc_phase_test
```

The pinned baseline was extracted with `git archive` and built with the same
first command, omitting `pair_bf16_test`. No generated cubins are committed.
Use equivalent local toolchain paths when reproducing elsewhere.

The candidate binaries and cubins were copied into the isolated directory
`/tmp/flash-sass-issue53.4IEoK6/candidate` on archy. From that directory:

```bash
HC_GPU=7 HC_WAIT_SECONDS=300 GPU_LOCKDIR=/home/nate/codex/asm-port \
  ./hc_phase_check.sh \
  /tmp/flash-sass-issue53.4IEoK6/baseline/die.cubin \
  /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
```

From a source checkout, use `kern/hc_phase_check.sh` with the corresponding
baseline cubin and checkpoint paths. `GPU_LOCKDIR` is optional; when present,
use the reservation directory already used by the host's other GPU consumers.
The shared lock `/tmp/flash-sass-engine-validation.lock` is always required.
Each test retains its locks until it exits. Unavailable resources produce
exit 75 after the wait budget, never a passing result. Failed device queries
also fail the run.

## Acceptance status

| Requirement | Current evidence |
| --- | --- |
| HC phase total at most 4.0 ms/token | **FAIL:** candidate maximum 8.496 ms; all 18 observations exceed 4.0 ms |
| Existing `model_test` with numerical twins | **PASS:** baseline/candidate, free-run/loop, four tokens each, documented blocks [0,4) with embedding, PLE, attention and head |
| Identical output tokens on fixed prompts | **PASS:** all four baseline/candidate streams match, 90 generated IDs per arm |
| Short-context decode at least 38 tok/s | **FAIL:** arithmetic mean 36.3167 tok/s using one warmup and three measured requests |

Do not execute `kern/linkmatrix_test.sh` itself on the shared host: it runs
`sudo setpci` to retrain links. Reproduce only its HTTP protocol: prompt
`What is the capital of France? Answer in one word.`, `max_tokens: 40`,
`temperature: 0`; use the arithmetic mean of the three measured requests'
`timings.predicted_per_second`. Capture raw token IDs as well as HTTP text;
equal decoded text alone does not establish equal token IDs.

`prof.sh` reports the last token's phase times, including peer waits attributed
to the phase following an exchange. Querying the resident engine measures its
loaded code. Baseline and candidate need the same benchmark configuration;
a different KV or PLE mode would not establish the requested speedup.
The core's `reload` command can replace the library and its `FLASH_CUBIN`
together while retaining the core, weight holders, and PLE allocation. Save
the actual loaded library and cubin first, preserve the runtime settings,
and restore both images after the comparison. A second six-die model is not
required for that validation path.

## Earlier blocked attempt

The matching baseline and candidate libraries built successfully on engi at
the revisions above. The candidate cubin's SHA-256 is unchanged from the
earlier numerical run. [resume/status.json](resume/status.json) and
[resume/sha256.txt](resume/sha256.txt) record the source and build identities;
the two host build logs retain the existing `ref/quant.c` compiler warnings.

Four fixed prompts were rendered with the checkpoint's own chat template and
tokenizer: a capital-city question, a number sequence, a multi-turn arithmetic
question, and a C function. [prompts.json](prompts.json) retains their messages,
raw prompt IDs, stop IDs, and 64-token limits for baseline/candidate comparison.

The earlier attempt to rerun `hc_phase_check.sh` on archy GPU 7 waited 300 seconds
for `/tmp/flash-sass-engine-validation.lock`, then exited **75**, with **zero
GPU tests executed**. The lock was held by issue #19's model validation
(`run.rb --mode model --positions 128,32768,65536`). The complete wait output is
in [resume/kernel-check.log](resume/kernel-check.log). The earlier passing
numerical logs remain earlier evidence; this blocked attempt adds no GPU or
production performance result. It is retained as historical evidence; the
2,700-second rerun and live measurements above supersede its lock blocker.
PR #76 must remain a draft until the acceptance table above is satisfied.

## Reproduce the live comparison

Run the validator on the GPU host with matching baseline/candidate
`libengine.so`, `die.cubin`, and `prof.sh` artifacts. The source revisions below
identify the compiled runtime; later branch commits contain evidence and
validation tooling. Use a fresh output directory and a current, verified
resident core PID.

```bash
ruby validation/issue53/engine_check.rb \
  --core-pid 3575268 --socket /tmp/flash.sock \
  --url http://127.0.0.1:18090 \
  --baseline-dir /tmp/flash-sass-issue53-resume.IwwJba/baseline \
  --candidate-dir /tmp/flash-sass-issue53-resume.IwwJba/candidate \
  --baseline-rev b97b6b94bc0aba994830396fadebe5f8acd082f7 \
  --candidate-rev be5df804165f44e5fe74df98a6030445244837c4 \
  --fixtures validation/issue53/prompts.json \
  --out /tmp/issue53-live-new-attempt --wait-seconds 2700
```

The validator waits up to 45 minutes for the shared lock and for each idle
admission check. A measured gate failure exits 1; it is not a lock timeout or
a passing result. A lock timeout does not cause a reload. The output directory
contains the original engine images needed for restoration; preserve it
after the run while the restored engine references its cubin.

The CPU checks require no GPU or resident engine:

```bash
FLASH_SOCK=/tmp/issue53-cpu-mock ruby validation/issue53/engine_check_test.rb
```

## Reproduce the model-test regression

Build `hold` alongside the candidate `model_test` and cubin, using the same
compiler and CUDA driver stubs as the other host binaries. The existing
unsplit C-twin cache used by this run contains blocks 0 through 7 and the
embedding/head tensors; the wrapper reads it without changing it. It creates
an independent GPU holder manifest for the selected four blocks.

From a persistent session on the GPU host:

```bash
HC_GPU=7 HC_WAIT_SECONDS=2700 GPU_LOCKDIR=/home/nate/codex/asm-port \
  bash validation/issue53/model_check.sh \
  /tmp/flash-sass-issue53-resume.IwwJba/baseline \
  /tmp/flash-sass-issue53-resume.IwwJba/candidate \
  /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
  /home/nate/codex/flash-pr35-cache-20261017/tensors \
  /tmp/issue53-model-new-attempt
```

The lock wait uses the entire remaining budget in one `flock` call. GPU or
host-memory admission failures release the lock and retry within the same
budget. The wrapper records start, finish, and exit status, and its exit trap
cleans up only its owned holder. The recorded host-side supervisor allows the
wait to survive a command-session disconnect and imposes a separate upper
bound on the complete holder/test sequence. Its exact script is included in
the raw attempt bundle.
