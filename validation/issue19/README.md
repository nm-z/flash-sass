# Reproducing issue #19 validation

`run.rb` is the single validation runner. It uses Ruby's standard library, a C compiler, Make, the repository's patched MaxAs dependency, CUDA 11.4 `ptxas`, and the NVIDIA driver. There is no Python dependency in this validation workflow.

The runner requires committed sources and records the exact revision, runner hash, tool versions, assembler source hashes, build flags, binary/cubin hashes, commands, exit statuses, raw stdout/stderr, and elapsed times. Build outputs are isolated `git archive` copies under the chosen output directory. A build from a different revision or an altered artifact is rejected.

The [2026-10-18 evidence](20261018/README.md) records the completed CPU/build/M60 measurements at `6da421bccb68e613ecbdc8142104935deecccb29` and identifies the remaining checkpoint work. The [older evidence](archy-final/README.md) is retained with its original measured revision.

## Commands

Run from any directory. Choose a new output directory for a fresh build. All checkout, dependency, evidence, device and socket paths are arguments; the script contains no temporary-worktree paths.

```bash
ruby validation/issue19/run.rb --mode cpu --out "$evidence"
ruby validation/issue19/run.rb --mode build --out "$evidence" \
  --cuda "$cuda" --ptxas "$ptxas" --maxas-dir "$maxas"
ruby validation/issue19/run.rb --mode gpu --out "$evidence" \
  --gpu "$gpu_uuid" --lock-dir "$gpu_lock_dir" \
  --core-pid "$resident_core_pid" --socket "$resident_socket"
ruby validation/issue19/run.rb --mode model --out "$evidence" \
  --core-pid "$resident_core_pid" --socket "$resident_socket" \
  --checkpoint "$checkpoint_shard1" --lock-dir "$gpu_lock_dir" \
  --positions 32768,65536
```

The optional `--core-pid` and `--socket` arguments in GPU mode identify the existing idle model service. The test GPU itself must have no compute process. The runner checks `nvidia-smi`, `ps`, and existing socket connections before every test process or model request, and waits while another worker is active. The only resident processes it tolerates on other dies are the named sleeping core with no client and paused `hold`/`fence` weight holders. Both GPU modes hold the shared `flash-sass-engine-validation.lock` in the system temporary directory (`--engine-lock` or `FLASH_ENGINE_VALIDATION_LOCK` can override it); isolated tests also hold their GPU UUID lock under `--lock-dir`. It never sends `quit`, kills another process, unloads a weight holder, or starts a competing core.

## Three builds from the same source

| Variant | Generation flag | Purpose |
|---|---|---|
| Bounded | none | The production range policy. |
| Unbounded control | `QSA_KV_UNBOUNDED=1` | Validation-only removal of component clamps; isolates the original numerical failure and the incremental cost of clamps. |
| Bounded with trace | `QSA_KV_TRACE=1` | Validation-only capture of pre-clamp group maxima. |

These are separate build trees, so an environment flag cannot accidentally reuse a previously generated kernel. The unbounded control is not a supported production quantizer. Ordinary inputs must agree with the bounded contract in every build. The unbounded extreme control must fail with a numerical witness; a missing cubin or CUDA startup error is not accepted as that witness.

`quant_bench` reports repeated CUDA-event measurements for the complete Q/K normalization, RoPE and INT8 KV append operation. It uses 12 samples by default, all six variant orders twice, and 10,000 operations in each persistent launch. It verifies the completed operation count and unchanged outputs after every launch. The difference between the bounded and unbounded variants measures the incremental clamp cost within that operation. The difference between trace and bounded measures diagnostic overhead. Raw negative timing deltas are retained. These measurements are not whole-model tokens/second and do not include the optional checked scalar API.

## Checkpoint measurements

Model mode uses the already loaded real checkpoint through the production QSA engine, all 48 blocks, all 12 attention layers, a 512-block sparse cutoff, mapped host KV, and complete known-input prefill from position zero. It disables snapshot creation and verifies the reported number of processed positions. Each request is serialized on one core connection, and its source revision, cubin path and configuration are checked before any token is submitted.

Before loading the trace engine, the runner saves the exact loaded engine image and verifies that the original kernel file matches the core's recorded FNV. It restores the original engine, kernel and settings after the measurements, including on a validation failure, and checks the restored configuration. Checkpoint and holder-manifest identities are recorded.

Without `--corpus`, the input is a deterministic concatenation of this revision's README, SPEC, C sources, Perl generators and SASS sources. This is a repository-text workload, not a repeated synthetic KV cache or a claim about every natural-language workload. The two context lengths use distinct slices of its actual tokenizer output. The corpus/source hashes, complete token arrays, offsets, commands and outputs make the workload reproducible. An explicit UTF-8 corpus can be supplied with `--corpus`.

The trace captures each group's absolute maximum before component clipping and before the reciprocal floor. K is observed after normalization/RoPE; V is observed as projected. Unsigned magnitude-bit reductions preserve evidence of nonfinite inputs. Unwritten slots retain an impossible sentinel. The host combines these observations with the actual stored GPU scale bits after the run.

Per-layer K and V rows report maxima, zero groups, nonzero groups whose scale underflowed to zero, subnormal scales, clipped groups, and nonfinite stored scales. A separate `would_overflow` count is computed from the observed raw maxima under the former unbounded scale expression; it is a counterfactual, not an observed infinite scale in the bounded kernel. Counts are groups of 32 components, not individual clipped components. Missing slots, nonfinite input groups and scale-contract mismatches invalidate the trace.

Synthetic boundaries establish numerical behavior. Only completed checkpoint traces establish the maxima and event counts of their stated workloads. Do not infer measurements at a longer context or on a different checkpoint from these runs.
