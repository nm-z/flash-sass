# Issue 22: attention and model-layer validation at 2284a21

PR #46 remains **open and draft**. The new numerical tools and completed
experiments below close several tooling and measurement gaps, but they do not
complete every acceptance criterion in issue #22.

Executed code: **2284a214d3babea6df19510654203c65abfd6cce**. The later
**04e77e181fa6195f199ab384ad19e36fb18484d5** commit adds only the numerical
profiles for a separate actual-model length experiment. The evidence and
documentation commits do not change the tested executable source.

## Changes and completed checks

The independent C attention replay consumes frozen postprojection inputs and
the same ordered selected block IDs as the candidate. It reconstructs stable
softmax, GQA, output gating and query/key normalization with supplied gamma and
RoPE tables in FP64, without linking the repository's instruction twins.
Separate arms measure attention arithmetic, Q normalization/RoPE, V storage,
combined K normalization/storage, and an optional separate GPU FP32 K append.

The Perl wrapper validates dimensions, selected-token order, completion
metadata, epsilon, finite values, scales, limits and output shape. It hashes the
inputs, executable, wrapper and limits before and after replay, preserves
numerical failures, and refuses to overwrite an existing result. A passing
numerical comparison never certifies original-checkpoint quality.

| CPU check | Result |
| --- | --- |
| Comparator regressions | PASS, 76 assertions |
| Logit-export readback mock | PASS, 28 assertions |
| Layer-export readback mock | PASS, 48 assertions |
| Frozen indexer replay | PASS, 36 assertions |
| Analytic indexer | PASS, 16 checks |
| Frozen attention replay and malformed-input cases | PASS, 46 assertions |
| Analytic attention | PASS, six analytic groups |
| Synthetic indexer length matrix | PASS, all 14 lengths |
| Historical placement and recovered-reference comparisons | Recomputed successfully |

The first six rows total **250 analytic/regression checks**. The six attention
analytic groups, 14 synthetic length replays and archived comparisons are
separate. Commands, exit codes, runtime information and matching source hashes
are in [cpu](cpu). Source-status records retain the pre-existing untracked
revision-review directory; no tracked executable source changed during the run.

## Fixed-input GPU attention experiment

The standalone fixture ran the production attention SASS operations on GPU 6
of archy, a Tesla M60 at PCI address 0000:12:00.0. The CUDA driver API reports
11040. The compiled cubin SHA-256 is:

    116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14

This execution uses model_test-independent C fixture code and die.cubin; it
does not load libengine.so. Run manifests preserve the actual binary, source,
cubin and input hashes, device UUID, hardware state, seed 4242 and opcodes.

The numerical profile was committed in
**affe73f3b62c94639875a1d5b2c5d6958752f7be**, before either capture. Its SHA-256
is e6e6c337f222bfdef8eba689f307d6a39d6041464ac662d073a69687ba5b2f8e.
The limits remained unchanged: maximum absolute attention error 0.005, RMSE
0.001, relative L2 error 0.00005, and maximum query error 0.00001.

The budget-one smoke acquired the shared lock at 06:15:53 UTC on 2026-10-18 and
captured N=1,4,8,9. All four numerical comparisons passed. The budget-512 run
acquired it at 06:21:42 UTC and captured all 14 rows below. Every comparison
passed all four predeclared gates.

| N | Selected / complete blocks | Tail | Attention max abs | Attention RMSE | Attention relative L2 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0 / 0 | 1 | 9.41321e-8 | 1.51230e-8 | 4.80808e-8 |
| 2047 | 511 / 511 | 3 | 2.19292e-5 | 1.66610e-6 | 2.90114e-7 |
| 2048 | 512 / 512 | 0 | 1.39499e-5 | 1.63646e-6 | 2.84957e-7 |
| 2049 | 512 / 512 | 1 | 1.41596e-5 | 1.67350e-6 | 2.91614e-7 |
| 2050 | 512 / 512 | 2 | 1.76780e-5 | 1.66608e-6 | 2.90643e-7 |
| 2051 | 512 / 512 | 3 | 1.71771e-5 | 1.63159e-6 | 2.84958e-7 |
| 2052 | 512 / 513 | 0 | 2.12194e-5 | 1.69912e-6 | 2.96483e-7 |
| 2053 | 512 / 513 | 1 | 2.27174e-5 | 1.75148e-6 | 3.05812e-7 |
| 2054 | 512 / 513 | 2 | 1.85426e-5 | 1.67823e-6 | 2.93119e-7 |
| 2055 | 512 / 513 | 3 | 2.15948e-5 | 1.70677e-6 | 2.98136e-7 |
| 3100 | 512 / 775 | 0 | 1.62941e-5 | 1.67316e-6 | 2.92436e-7 |
| 8000 | 512 / 2000 | 0 | 1.38637e-5 | 1.59769e-6 | 3.18064e-7 |
| 65000 | 512 / 16250 | 0 | 1.87895e-5 | 1.82121e-6 | 3.20261e-7 |
| 262000 | 512 / 65500 | 0 | 1.52557e-5 | 1.62136e-6 | 2.86792e-7 |

The largest query error in this grid is 2.37891e-7. The full precision values,
mode comparisons, selected IDs and expanded token order are in
[gpu/production-budget-2284a21](gpu/production-budget-2284a21). The compressed
full JSON reports retain every output vector. The smaller summary omits only
those output vectors, whose reports are separately hashed.

These fixtures supply the block IDs externally. They establish execution of
production attention kernels, selected-block exclusion and tail handling on
the supplied synthetic inputs. They do not execute the production indexer or
full-model sparse dispatcher, and their length coverage cannot substitute for
the requested teacher-forced model-length matrix.

The optional normalized_k.f32 artifact comes from a separate FP32 K-append
operation, not the exact prequantization registers of the INT8 append.
Differences between those normalization/RoPE paths remain possible.
Consequently pure_k_storage_isolated and k_storage_only_measured remain false.
The V-storage arm holds the query and effective K fixed; combined arms are
labeled explicitly, and no additivity of their errors is assumed.

## Actual model capture and independent component replays

[model/model-smoke-guarded-2284a21](model/model-smoke-guarded-2284a21)
records a new successful full-48-layer run with embedding, output head, real
disk-PLE weights, production QSA, budget one and pinned-host KV. It acquired the
lock at **06:36:32 UTC** and completed all nine teacher-forced IDs 10 through 18.
Those consecutive IDs are a microfixture, not held-out language.

The v2 logit completion marker and layer completion marker were published.
Layer 3 snapshots at N=1,4,8,9 include actual projections, pools, scores when
dispatched, selected IDs, attention outputs and layer outputs. Both independent
component replays completed for all four snapshots.

At N=9, the candidate selects one of two complete blocks and includes the
one-token tail. Its selection matches both the exported candidate scores and
the independent FP64 selected set, with canonical overlap 1 and exact-tie
compatibility. The maximum scaled-indexer-score difference from the full FP64
control is 0.000254709208537207. Fixed-ID attention differs from the independent
replay by maximum absolute error 4.32428643448723e-7, RMSE
3.69038940697303e-8 and relative L2 4.90433969366284e-7.

These model smoke comparisons are **diagnostic**. No numerical profile was
applied to them and the later profiles do not retroactively make them accepted
experiments. They do not establish independent original-checkpoint logits.

All 2,234,880 exported vocabulary values match the archived nine-token,
budget-one capture byte for byte. Their raw SHA-256 is:

    894ef61a63680dd3f00496dc001c26d574c9d0700e19a23653357cd55bfb35d9

The matching historical capture was already checked for finite values. This
byte comparison is evidence about these exact inputs and two recorded builds;
it is not whole-engine parity, representative quality or independent
checkpoint agreement.

## Interrupted attempt and production-budget preflights

The earlier [model/model-smoke-2284a21](model/model-smoke-2284a21) acquired the
lock at 06:26:30 UTC but did not finish. It was explicitly interrupted, reports
exit 143, and produced no completion marker. A resident flashcore process was
also using the same devices. The cause of the stall has not been established;
it is not a measured numerical or quality failure.

TERM left an owned CUDA event-handler thread alive; KILL was subsequently
sent only to the same owned process. Cleanup completed and the lock was
released by the 06:34:09 UTC observation. No other worker or holder was
signaled. The successful retry above used line-buffered output, a bounded
watchdog and a preflight that refuses a resident engine on the same devices.

The production-budget experiment was prepared for T=8000 with snapshots at
N=1,2047..2055,3100,8000. Its numerical profiles were committed in 04e77e1 at
06:44:40 UTC, before any model kernel in that experiment could launch:

| Attempt | Lock acquired | Outcome |
| --- | --- | --- |
| model-production-8000-2284a21 | 06:45:27 UTC | Exit 75 before launch: resident flashcore PID 1249426 used the six holder devices |
| model-production-8000-retry-2284a21 | 06:52:52 UTC | Exit 75 before launch: resident flashcore PID 1321713 used the same devices |

The first resident process was observed to have exited before the retry.
Each preflight's process identity and hardware observation belongs to that
attempt's timestamp, not an assertion that the PID will remain live. No model
kernel or model-length comparison ran in either blocked attempt.

The operational blocker is obtaining a quiescent six-device window for the
full-model validation configuration. Acquiring the cooperative lock alone
does not stop an already resident service. The runner leaves the service and
holders running. The published attempts identify the exact service commands,
holder manifests and source configuration involved.

The later [resource observation](model/resource-observation-final.txt) at
**07:05:59 UTC** found the lock free and a new resident flashcore, PID **1369637**,
using the same six holder manifests. GPUs 0–5 had respectively 52, 16, 77, 7, 3
and 65 MiB free. This is the latest recorded resource observation, not a claim
that the earlier refused PIDs remained alive.

## Checkpoint and conversion provenance preflight

The original repository is public and ungated at
[de4b8e4d43b917e7706784d8bb445c9af86a3540](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/de4b8e4d43b917e7706784d8bb445c9af86a3540).
The retrieved configuration, tokenizer configuration, tensor index and
published shard identities are recorded in [provenance](provenance). Its
tensor index has 1,658 entries across 131 shards and declares 359,999,963,128
weight bytes. No original weight tensors were downloaded or executed.

The configuration declares BF16 text weights, 48 layers, 512 routed experts
with ten selected per token, 24 query heads, two KV heads, indexer budget 2048
tokens with compression ratio four, and maximum position 262144. The PLE layer
number is converted from the checkpoint's one-based convention.

The two measured GGUF shard hashes match the publisher's LFS identities at
[ed59f92082b1e93c0e96d60a8b11aab089b52f09](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/tree/ed59f92082b1e93c0e96d60a8b11aab089b52f09).
The publisher identifies the base model and the GSQ/RCO methods. This verifies
the identity of the downloaded quantized artifacts, not the original revision,
calibration data, tool commits and exact conversion/search execution that
produced them. That lineage remains unverified.

A bounded source inspection found a relevant path in the clean existing
llama.cpp source at 7182ef71e6d5dd0c8ea76486fac134fcd25025d4:
[Qwen4Exp conversion](https://github.com/ggml-org/llama.cpp/blob/7182ef71e6d5dd0c8ea76486fac134fcd25025d4/conversion/qwen4exp.py),
[native model graph](https://github.com/ggml-org/llama.cpp/blob/7182ef71e6d5dd0c8ea76486fac134fcd25025d4/src/models/qwen4exp.cpp)
and [mapped loader](https://github.com/ggml-org/llama.cpp/blob/7182ef71e6d5dd0c8ea76486fac134fcd25025d4/src/llama-model-loader.cpp).
The converter recognizes this architecture, preserves integer PLE constants,
records the QSA metadata, converts zero-centered gamma and handles the large
PLE table in chunks. The loader recognizes BF16 and memory-mapped input.
Reading that existing source did not author or execute Python or C++ code.

Those source observations are not a successful original-checkpoint conversion
or bounded-memory model run. Available RAM/VRAM cannot hold all BF16 parameters
resident at once, but mapped or streamed execution has not been ruled out.
No claim of inaccessible weights or hardware impossibility is made. A verified
model pair, an executed independent reference, frozen held-out inputs, quality
thresholds and long-context measurements remain required.

## Acceptance state

[acceptance.json](acceptance.json) records these states machine-readably.

| #22 criterion | State | Remaining work |
| --- | --- | --- |
| 1. Reproducible tooling and complete provenance | Unmet | Tooling and new fixture identities are recorded; original-checkpoint execution and exact GGUF conversion/search lineage are not verified |
| 2. Same-input controls followed by independent selection | Unmet | New fixed-input attention controls and a real small-budget selector replay are measured; projection/activation quantization, pure K-storage isolation and the complete independent-selection experiment remain |
| 3. Native pooling, masks, positions and explicit precision conversions | Unmet | Contiguous-prefix block/tail and dtype controls are implemented and partly measured; the complete native-QSA precision/mask comparison remains |
| 4. Exports, divergence metrics and predeclared thresholds | Unmet | New attention outputs and real model-layer snapshots are compared; the production-length and independent teacher-forced logit experiments with their applicable thresholds remain |
| 5. Production sparse length matrix and fixed continuation | Unmet | Synthetic attention reaches 262000; the new actual model capture is only T9/budget1, and both T8000/budget512 preflights stopped before launch |
| 6. Held-out long-context quality and model-difference controls | Unmet | Loss/perplexity, retrieval/task accuracy, separate free generation and controlled original-checkpoint comparisons remain unmeasured |
| 7. Machine-readable reporting and qualified support statement | Met | Completed, failed, blocked and unmeasured scopes are distinguished; overall acceptance remains false |

The candidate source includes the PLE EOS and DeltaNet epsilon fixes tracked
in issues #1/#2. Their presence alone does not control the corresponding
independent-backend differences in a quality experiment.

## Reproduce

Build instructions and replay profiles are documented in
[MODEL_QUALITY.md](../../../MODEL_QUALITY.md). The CPU runner is:

    bash validation/issue22/run_cpu.sh /absolute/path/new-cpu-evidence

The GPU fixture runner acquires /tmp/flash-sass-engine-validation.lock and
records its queue observations every three minutes. Each model attempt
preserves its exact run-script.sh, command, environment, hardware state and
source identities. Model runs require the recorded six-holder split and
attention map; a standalone fixture needs only its selected device.

Large regenerated synthetic inputs and raw long attention captures remain on
archy under /home/nate/codex/flash-pr46-attention-20261018.LnH4Z4. The CPU
replay inputs remain under
/home/nate/codex/flash-issue22-20261018.P8ncez/review-2284a21-cpu.
The committed hashes bind those artifacts. This directory preserves compact
metrics, complete compressed attention reports, the completed small actual-model
capture and its raw layer export, commands and failure/preflight records.

PR #46 is not Ready for review while criteria 1–6 remain unmet.
