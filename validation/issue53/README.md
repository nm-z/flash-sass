# Hyper-connection phase work for issue 53

This is a partial implementation for [issue #53](https://github.com/nm-z/flash-sass/issues/53),
based on default-branch commit `b97b6b94bc0aba994830396fadebe5f8acd082f7`.
The full issue body, evidence, and acceptance criteria were read; the comments
endpoint returned no comments. [PR #76](https://github.com/nm-z/flash-sass/pull/76)
is open as a draft. Default-branch commit `3d57d734d63aa74f167a6b5a02c616ddc018d2ff`
was merged into this branch as `be5df804165f44e5fe74df98a6030445244837c4`;
that merge changed CI and the QSA test's position bound, not the HC runtime code.
The requested production performance and whole-model acceptance are not established.

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

The number of global phases and the profiling names are unchanged. This PR
does not establish how much production time these changes save. Build the
host phase builder and `die.cubin` together: an old host still dispatching ten
`hcwsum` units cannot drive the new 160-column layout correctly.

## Results

All GPU checks ran on archy GPU 7, a Tesla M60 (`sm_52`, 16 SMs), on
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
| `model_test` | Built successfully; whole-model execution remains outstanding |

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

## Full acceptance still required

| Requirement | Remaining evidence |
| --- | --- |
| HC phase total at most 4.0 ms/token | Matched candidate `PROF_ALL=1 kern/prof.sh` capture on all six production dies; count all HC operations and retain raw output |
| Existing whole-model bit-exact checks | Execute `model_test` against the same checkpoint with correctness checks enabled |
| Identical output tokens on fixed prompts | Compare raw token IDs from baseline and candidate using the same prompt IDs, stop IDs, sampling, KV, and PLE configuration |
| Short-context decode at least 38 tok/s | One warmup and three serial requests using the protocol in `kern/linkmatrix_test.sh`; retain each request and result |

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

## Resumed validation

The matching baseline and candidate libraries built successfully on engi at
the revisions above. The candidate cubin's SHA-256 is unchanged from the
earlier numerical run. [resume/status.json](resume/status.json) and
[resume/sha256.txt](resume/sha256.txt) record the source and build identities;
the two host build logs retain the existing `ref/quant.c` compiler warnings.

Four fixed prompts were rendered with the checkpoint's own chat template and
tokenizer: a capital-city question, a number sequence, a multi-turn arithmetic
question, and a C function. [prompts.json](prompts.json) retains their messages,
raw prompt IDs, stop IDs, and 64-token limits for baseline/candidate comparison.

The new attempt to rerun `hc_phase_check.sh` on archy GPU 7 waited 300 seconds
for `/tmp/flash-sass-engine-validation.lock`, then exited **75**, with **zero
GPU tests executed**. The lock was held by issue #19's model validation
(`run.rb --mode model --positions 128,32768,65536`). The complete wait output is
in [resume/kernel-check.log](resume/kernel-check.log). The earlier passing
numerical logs remain earlier evidence; this blocked attempt adds no GPU or
production performance result. PR #76 must remain a draft until the full
acceptance table above is satisfied.
