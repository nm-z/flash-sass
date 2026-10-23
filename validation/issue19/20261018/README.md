# Issue #19 validation — 2026-10-18

The CPU and M60 results in this directory were measured from committed source **`6da421bccb68e613ecbdc8142104935deecccb29`** on archy. CPU, build and GPU manifests report `PASS`. Full-checkpoint long-context measurements are still running and are not claimed by these synthetic results.

The implementation and reproducible Ruby runner are at that exact commit. This evidence addition does not modify runtime sources. The older [archy-final](../archy-final/README.md) results describe `807e91c...` and remain historical.

## Artifacts and raw records

| Record | Contents |
|---|---|
| [cpu.json](cpu.json) | Compiler command, exact source revision, executable hash, exits and stdout/stderr hashes. |
| [build.json](build.json) | Three isolated source archives, compiler/ptxas versions, generation flags, assembler-source hashes, and hashes/sizes for every executable, engine library, generated SASS and cubin. |
| [gpu.json](gpu.json) | Every preflight, test/benchmark command, environment, expected and actual exit status, duration and raw-output hash. |
| [assembler-copies.json](assembler-copies.json) | Twelve checks confirming the copied MaxAs sources match across the three build trees. |
| [cpu-build-gpu-raw.tar.gz](cpu-build-gpu-raw.tar.gz) | All 182 files referenced by the completed CPU/build/GPU records, including raw build logs, NVIDIA inventory, process/socket preflights, and zero-length stderr files. Selected numerical outputs are also directly readable below. |
| [source-note.json](source-note.json) | Tested source/tree identity and evidence-package integrity checks. |

Before packaging, all **180 recorded stdout/stderr hashes** were recomputed successfully. The archive SHA-256 is `3a4dce497e202bf41e9b379b04449de31cc8ef77640d9b811aa0aaae25d14c63`; the transferred copy has the same hash. The archive contains logs and manifests, not checkpoint weights or build binaries. Exact binary and cubin identities are retained in `build.json`.

### Cubin SHA-256

| Variant | SHA-256 |
|---|---|
| Bounded production | `6d5bbe4c5eedf0d47cf31804e1e0befe2a953db0d5acd0e2ab9d585c067f1367` |
| Unbounded validation control | `116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14` |
| Bounded diagnostic trace | `a2b5b013f3e99fed1e29f415c4805a68c49ac02cd5e658a09020303c7c5b62b8` |

## Numerical acceptance

The target is a Tesla M60, compute capability 5.2, CUDA driver API 11040. The synthetic matrix uses physical GPU 6, UUID `GPU-431371bd-b482-96b2-4b70-04f83c6d075b`.

| Case | VRAM KV | Mapped-host KV |
|---|---|---|
| Bounded, ordinary inputs | [PASS](bounded-normal-vram.stdout) | [PASS](bounded-normal-host.stdout) |
| Bounded, 44 signed boundary fixtures | [PASS](bounded-boundaries-vram.stdout) | [PASS](bounded-boundaries-host.stdout) |
| Trace, ordinary inputs | [PASS](trace-normal-vram.stdout) | [PASS](trace-normal-host.stdout) |
| Trace, 44 signed boundary fixtures | [PASS](trace-boundaries-vram.stdout) | [PASS](trace-boundaries-host.stdout) |
| Unbounded control, ordinary inputs | [PASS](unbounded-normal-vram.stdout) | [PASS](unbounded-normal-host.stdout) |
| Unbounded control, extreme inputs | [Expected numerical failure](unbounded-boundaries-vram.stderr), exit 1 | [Expected numerical failure](unbounded-boundaries-host.stderr), exit 1 |

[CPU output](cpu-boundaries.stdout) covers all 44 signed fixtures, explicit expected half-scale classes, the ordered clipped-input reference, finite reconstruction, nonfinite inputs at each of 32 component positions, rejected rounding modes/null pointers and unchanged outputs on rejection.

The true zero-to-positive stored-scale pair is binary32 `0x367e0001` / `0x367e0002`. The C test asserts that they are adjacent, that the rounded FP32 products are `0x33000000` / `0x33000001`, and that half conversion gives `0x0000` / `0x0001`. Both signs pass the GPU test in both placements. The product values are normal FP32, so FP32 flush-to-zero does not shift this boundary. Half subnormals are retained by the conversion.

The GPU matrix compares independent expected scale bits and extreme codes, C/GPU code bytes and scale bits, finite reconstruction, queries and cache appends, then gather and attention. Each successful KV test reports `BIT-EXACT (0 tokens differ; query/append checks 16, gather/attention checks 16)`. Trace runs additionally verify **512 raw group maxima**. The unbounded control reaches an infinite GPU half scale (`7c00`) for the original overflow witness; its expected failure is a numerical result, not a CUDA launch failure.

These tests establish the documented bounded quantizer for the tested cases. They do not establish checkpoint event counts or model-quality effects.

## Measured clamp and trace cost

The [VRAM samples](quant-benchmark-vram.stdout) and [mapped-host samples](quant-benchmark-host.stdout) measure the complete `OP_ATTNQKQ` operation, including Q/K normalization, RoPE, INT8 KV append, dispatcher and synchronization. Each launch repeats **10,000 operations at fixed position 0**, with 16 CTAs, 256 threads/CTA and 26 work units. Two warmup rounds of 1,000 operations per variant precede 12 timed samples. All six variant orders are used twice.

CUDA events cover kernel execution. Setup, transfers and result readback are outside the timed interval. Each placement passes independent ordinary K/V and raw-maximum checks, then verifies the completed operation count and bitwise agreement after all **36 timed launches**.

| KV placement | Bounded median (µs/op) | Unbounded median (µs/op) | Trace median (µs/op) | Median paired clamp delta (µs/op; %) | Median paired trace delta (µs/op; %) |
|---|---:|---:|---:|---:|---:|
| VRAM | 8.40736313 | 8.41913567 | 8.85977592 | −0.0260368347; −0.308680727% | +0.438263702; +5.20228809% |
| Mapped host | 8.39322395 | 8.41701927 | 8.83148804 | −0.0233360291; −0.277288941% | +0.411387253; +4.88273114% |

The paired delta is computed within each sample before taking the median; it is not the difference of the variant medians. Negative measurements are retained. These results do not establish a statistically significant speedup, whole-model tokens/second, or overhead at other context positions. The optional checked scalar C API is outside the production GPU path and was not benchmarked.

## Commands actually used

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
```

GPU mode acquired the shared engine-validation lock and the device UUID lock before checking `nvidia-smi`, `ps` and socket occupancy. It repeated those checks before every GPU process, including waits for observed utilization to return to idle, and released its locks on completion. It used the otherwise unoccupied test GPU while preserving the sleeping resident core and paused weight holders. No other process was killed.

## Pending checkpoint acceptance

The full-checkpoint validator is collecting complete prefills at 128, 32,768 and 65,536 positions with the same measured source and trace cubin. The 128-position preflight has completed, but the overall model run is not yet complete. Long-context maxima, underflow and clipping counts, complete raw model records and verified engine restoration will be added after the corresponding measurements finish.
