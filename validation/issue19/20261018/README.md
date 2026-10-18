# Issue #19 validation — 2026-10-18

The numerical, real-checkpoint and overhead acceptance checks are complete. Measurements are attributed to the revision that actually executed them:

| Measured revision | Completed work |
|---|---|
| `6da421bccb68e613ecbdc8142104935deecccb29` | CPU boundaries/rejection tests, three builds, M60 synthetic matrix and operation benchmark, and checkpoint prefills at 128, 32,768 and 65,536 positions. |
| `d46d188a435c5d959993f16dc1f78c837cc9f70d` | Merged-master CPU/build/M60 checks, repeated operation benchmark, and a 128-position model comparison at capacity 65,552. |
| `5b3ea525005fa5f8573ab544bfbeee6783c62e25` | Successful-call overhead of the checked scalar API on engi. This revision adds only the C benchmark and leaves the merged runtime sources unchanged. |

All corresponding manifests report `PASS`. Both model runs restored the original resident engine. Long contexts were executed at `6da421b`; they were **not repeated at `d46d188`**. The evidence supporting their applicability after the merge is described below as an inference. The older [archy-final](../archy-final/README.md) results remain historical.

## Real-checkpoint maxima and event counts

The checkpoint is the two-shard `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` GGUF, with 37,623,740,192 and 28,800,138,432 bytes respectively. Both checkpoint files and all six holder manifests were hashed in each model run; the recorded hashes match. Requests use all 48 blocks, all 12 attention layers, the 512-block sparse cutoff, mapped-host KV, no snapshot/resume, no paired execution, and capacity 65,552. The merged run explicitly disables chunked prefill.

Inputs are actual checkpoint-tokenizer output from recorded repository text. The 32k and 64k requests use distinct slices of that corpus. The corpus SHA-256 is `b6fde5d541106553ed422b1b40c422eb93b8550ac8a9f8a44148ef7035c901ea`. Complete corpus text, tokenizer output, input arrays and slice offsets are retained.

| Positions | K/V rows | Groups per row | Total groups | Largest K magnitude | Largest V magnitude | Underflow groups | Clipped groups | Would-overflow groups |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 24 | 2,048 | 49,152 | 15.7117481 | 34.5206795 | 0 | 0 | 0 |
| 32,768 | 24 | 524,288 | 12,582,912 | 15.6265059 | 37.5961761 | 0 | 0 | 0 |
| 65,536 | 24 | 1,048,576 | 25,165,824 | 15.6141577 | 37.5779762 | 0 | 0 | 0 |

These three rows were measured at `6da421b`. Every layer/kind row also reports zero zero-input groups, subnormal scales, nonfinite inputs, nonfinite stored scales, missing trace slots and scale mismatches. The merged 128-position run has the same numerical rows and totals.

The [per-layer tables](model-layer-tables.md) give K and V maxima and event counts for all 12 attention layers at every context length. [model.json](model.json) retains exact decimal, hexadecimal and binary32-bit representations, maximum coordinates, per-layer counts, commands and raw-output hashes. [model-audit.json](model-audit.json) records independently checked coverage and totals.

Maxima are captured **before clipping and before the reciprocal floor**. Counts refer to groups of 32 components. Underflow includes only nonzero groups whose actual stored GPU scale is zero. Would-overflow applies the former unbounded scale expression to observed raw maxima; it is a counterfactual count, not observed infinite scales in the bounded kernel.

These measurements establish the event counts of the recorded checkpoint and repository-text workload. They do not establish model-quality effects or event rates for other checkpoints, text distributions, or longer contexts.

## Numerical acceptance

The target GPU is a Tesla M60, sm_52. Both measured GPU revisions pass the same 12-outcome matrix: three variants (bounded, unbounded control, bounded trace), normal/extreme fixtures, and VRAM/mapped-host KV. Unbounded extreme cases intentionally exit 1 with the numerical overflow witness in both placements. The other ten cases exit 0.

[CPU output](cpu-boundaries.stdout) covers 44 signed fixtures, expected half-scale classes, an ordered clipped-input reference, finite reconstruction, nonfinite inputs at each of 32 component positions, rejected rounding modes/null pointers and unchanged outputs on rejection.

The actual adjacent binary32 zero-to-positive stored-scale pair is `0x367e0001` / `0x367e0002`. Rounded FP32 scale products are `0x33000000` / `0x33000001`, and half conversion produces `0x0000` / `0x0001`. Both signs pass the GPU tests in both placements. The products are normal FP32, so FP32 flush-to-zero does not move this crossing; half subnormals are retained.

Successful KV tests compare independent expected scale bits and extreme codes, C/GPU code bytes and scale bits, finite reconstruction, query/cache append, gather and attention. Each reports `BIT-EXACT (0 tokens differ; query/append checks 16, gather/attention checks 16)`. Trace cases also verify 512 raw group maxima. The unbounded control reaches an infinite GPU half scale (`7c00`); its expected failure is numerical, not a CUDA startup failure.

The [range contract](../../../KV_SCALE_RANGE.md) specifies component saturation to ±8,319,008 before both codes and scales, preservation of ordinary in-range arithmetic, finite-input/rounding requirements and the deliberately lossy underflow policy.

## Merge integration and measured scope

The [merged evidence](integration/) records completed CPU/build/GPU/model checks at `d46d188`. Additional checked-input and prefill-planner CPU tests pass; the new direct QSA harness compiles. No GPU execution of that separate direct harness is claimed here.

[Artifact checks](integration/artifact-equivalence.json) confirm byte-identical cubins, generated SASS and descriptor headers between `6da421b` and `d46d188` for all three variants (nine checks), with twelve copied-assembler-source checks. [Source checks](integration/source-equivalence.json) cover 85 existing source files and nine relevant host functions. The whole engine source changed in the merge; these checks do not claim otherwise.

The [128-position comparison](integration/parity-128.json) confirms:

- Identical checkpoint/holder hashes, corpus hash and input-token hash.
- Matching effective computation settings and capacity 65,552; merged chunked prefill is disabled.
- Exact equality of all 24 per-layer metric rows, output tokens, and 234 state-hash payload records.
- Successful restoration of the original engine in each run.

The unchanged numerical artifacts and relevant computation/dispatch functions, together with the actual merged-runtime comparison, support applying the earlier long-context observations to the unchanged computation path. **That is an evidence-backed inference, not execution of 32k/64k at the merged revision.**

Model logs retain elapsed times: 8,348.5 ms for the original 128-position run, 1,498,564.5 ms for 32,768, 3,419,972.0 ms for 65,536, and 15,900.9 ms for the merged 128-position run. These were not controlled paired throughput measurements; the differing 128-position timings are not interpreted as a merge-performance result.

## Measured clamp and trace cost

The complete `OP_ATTNQKQ` operation includes Q/K normalization, RoPE, INT8 KV append, dispatcher and synchronization. Each launch repeats 10,000 operations at fixed position 0, with 16 CTAs, 256 threads/CTA and 26 work units. Two warmup rounds of 1,000 operations per variant precede 12 timed samples, using all six variant orders twice.

CUDA events cover kernel execution; setup, transfers and readback are outside the interval. Each placement passes independent ordinary K/V and raw-maximum checks, then checks operation completion and bitwise agreement after all 36 timed launches.

| Revision | KV placement | Bounded median (µs/op) | Unbounded median (µs/op) | Trace median (µs/op) | Median paired clamp delta (µs/op; %) | Median paired trace delta (µs/op; %) |
|---|---|---:|---:|---:|---:|---:|
| `6da421b` | VRAM | 8.40736313 | 8.41913567 | 8.85977592 | −0.0260368347; −0.308680727% | +0.438263702; +5.20228809% |
| `6da421b` | Mapped host | 8.39322395 | 8.41701927 | 8.83148804 | −0.0233360291; −0.277288941% | +0.411387253; +4.88273114% |
| `d46d188` | VRAM | 8.38506851 | 8.4136734 | 8.81619835 | −0.0291278839; −0.34608381% | +0.431027222; +5.14017963% |
| `d46d188` | Mapped host | 8.39864311 | 8.41753426 | 8.83247681 | −0.0201488495; −0.239330786% | +0.411764908; +4.88644673% |

Raw samples: original [VRAM](quant-benchmark-vram.stdout) / [mapped host](quant-benchmark-host.stdout), and merged [VRAM](integration/quant-benchmark-vram.stdout) / [mapped host](integration/quant-benchmark-host.stdout).

Paired differences are calculated within each sample before taking the median. They are not differences between variant medians. Negative observations are retained. No statistically established speedup, overhead at other context positions or whole-model throughput result is claimed.

### Checked scalar API cost

The [scalar evidence](scalar/README.md) measures the real checked and unchecked C entry points on engi at `5b3ea52`, using 64 finite in-range groups with positive, negative and zero components. Across 12 alternating pairs, there are 3,145,728 timed calls per variant and 24 timed output checks, all passing.

Thread CPU medians are 94.2586555 ns/group unchecked and 113.424664 ns/group checked. Median paired added cost is **16.9233723 ns/group (18.0249748%)**. Monotonic wall-time samples and host/process inventory are also retained. This measures successful scalar calls, including argument/rounding/finite checks and temporary-output copying, on that shared CPU host. It is outside the production GPU path and does not measure rejected-input latency or whole-model overhead.

## Artifacts and integrity

| Record | Contents |
|---|---|
| [cpu.json](cpu.json), [build.json](build.json), [gpu.json](gpu.json) | Original revision, commands, exits, compiler/ptxas versions, flags, assembler hashes, binary/cubin hashes and output hashes. |
| [cpu-build-gpu-raw.tar.gz](cpu-build-gpu-raw.tar.gz) | All 182 original CPU/build/GPU files; 180 stdout/stderr hashes checked before packaging. |
| [model-raw.tar.gz](model-raw.tar.gz) | All 95 model records/inputs, including full corpus/token arrays, logs, commands, inventories and restoration records; 82 command-record file hashes checked. |
| [model-raw-package.json](model-raw-package.json) | Model archive hash and per-member hashes/sizes. |
| [integration/integration-raw.tar.gz](integration/integration-raw.tar.gz) | All 272 merged CPU/build/GPU/model and integration records; 256 command-record file hashes checked. |
| [integration/integration-raw-package.json](integration/integration-raw-package.json) | Merged archive hash and per-member hashes/sizes. |
| [scalar/](scalar/) | Scalar source/build identities and all 14 checked raw stdout/stderr files. |
| [audit-completed.rb](audit-completed.rb) | Offline hash, token-slice, row/counter/maximum, restoration and merged-parity checks. |
| [source-note.json](source-note.json) | Measured revisions and evidence scope. |

Archives contain recorded inputs, commands and results. Checkpoint weights, engine binaries and cubins are excluded; exact binary identities remain in the build manifests. Archive hashes were checked again after transfer:

| Archive | SHA-256 |
|---|---|
| CPU/build/GPU | `3a4dce497e202bf41e9b379b04449de31cc8ef77640d9b811aa0aaae25d14c63` |
| Model | `bad37f72b8ea0e4230646cc2d803a3729dc1f152f1d8ed8e4b3cc8e0c96f141e` |
| Merged integration | `f28c4cf405c7007724376fb44af376b03c88cfbb75310054c79e2a87777dede5` |

### Cubin SHA-256, identical at both GPU revisions

| Variant | SHA-256 |
|---|---|
| Bounded production | `6d5bbe4c5eedf0d47cf31804e1e0befe2a953db0d5acd0e2ab9d585c067f1367` |
| Unbounded validation control | `116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14` |
| Bounded diagnostic trace | `a2b5b013f3e99fed1e29f415c4805a68c49ac02cd5e658a09020303c7c5b62b8` |

## Commands and shared-host cleanup

Original measurements used this checkout and evidence directory on archy:

```bash
cd /home/nate/codex/flash-pr44-acceptance-20261018
ruby validation/issue19/run.rb --mode cpu \
  --out /home/nate/codex/flash-pr44-evidence-20261018
ruby validation/issue19/run.rb --mode build \
  --out /home/nate/codex/flash-pr44-evidence-20261018 \
  --cuda /opt/cuda --ptxas /opt/cuda-11.4/bin/ptxas \
  --maxas-dir /home/nate/claude/flash-sass-live/maxas
ruby validation/issue19/run.rb --mode gpu \
  --out /home/nate/codex/flash-pr44-evidence-20261018 \
  --gpu GPU-431371bd-b482-96b2-4b70-04f83c6d075b \
  --lock-dir /home/nate/codex/asm-port \
  --engine-lock /tmp/flash-sass-engine-validation.lock \
  --core-pid 3575268 --socket /tmp/flash.sock
ruby validation/issue19/run.rb --mode model \
  --out /home/nate/codex/flash-pr44-evidence-20261018 \
  --core-pid 3575268 --socket /tmp/flash.sock \
  --checkpoint /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --lock-dir /home/nate/codex/asm-port \
  --engine-lock /tmp/flash-sass-engine-validation.lock \
  --positions 128,32768,65536
```

The [merged driver](integration/integration-driver.sh) records its exact GPU and model commands, including `--positions 128 --capacity 65552` and the same explicit corpus. Its [exit status](integration/integration-driver.exit) is zero. CPU/build commands and additional CPU targets are in the corresponding manifests.

Every GPU mode acquired `/tmp/flash-sass-engine-validation.lock`; isolated tests also acquired the device UUID lock. The runner checked `nvidia-smi`, `ps` and socket occupancy before GPU work and repeated these checks before each test/request. It waited for occupied resources, restored the original engine from saved byte-identical library/cubin copies, verified original settings and cubin identity, and released its locks. Saved restoration copies remain on archy because the resident engine refers to the saved cubin path. No other process was killed.

The [resident-log audit](integration/resident-log-audit.json) preserves the [full log through post-restoration capture](integration/resident-stderr-through-restoration.log), including earlier activity, and finds no matching error/failure/short-read/watchdog/fault/timeout lines.

To repeat the offline audit, extract the original CPU/build/GPU and model archives into one directory, extract the merged archive into another, then run:

```bash
ruby validation/issue19/20261018/audit-completed.rb \
  /path/to/extracted-original /path/to/extracted-merged
```
