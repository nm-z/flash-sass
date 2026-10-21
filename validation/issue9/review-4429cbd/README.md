# Eight-block review validation

## Identity and scope

The C executables were built from `4429cbd0e40c9338cb9e5cf809afd52cb71e0a5e` on archy. [build.log](build.log), [binaries.sha256](binaries.sha256), [source-files.sha256](source-files.sha256) and [toolchain.txt](toolchain.txt) record the build, binary and source identities. The cubin was copied from the `e7921ae` direct-position build. [direct-source-equivalence.log](direct-source-equivalence.log) verifies all recorded direct-component sources against this checkout; the C harness/engine changes do not change that cubin's sources.

The fixture uses actual packed checkpoint tensors for blocks `[0,8)`: six DeltaNet blocks, the block-1 PLE prefix, and attention blocks 3 and 7. It includes the real token embedding and complete output head. Each positive model case supplies a prefix of IDs 10..18. Real PLE rows come from the second shard at offset 192. This is a one-die partial-model fixture, not the production six-die placement or the complete 48-block model. Its predictions are not a model-quality measurement.

The CPU export has 189 packed tensors. Their fresh SHA-256 identities, the non-IPC packing manifest and export output are in [the cache-preparation record](../review-e7921ae/cache-preparation/). The private GPU holder repacks the checkpoint with the same holder code. Its exact executable, invocation, runtime manifest identity and readiness log are recorded in each wrapper directory. No new whole-checkpoint file hash is claimed; the packed-tensor identities are the weight identities measured for this fixture.

## Model matrix

| Record | Cases and measured result | Comparison scope |
|---|---|---|
| [first-model](first-model/matrix/results.json) | One nine-ID positive case, budget 1, host KV; exit 0. | Every position's layer/QSA/recurrent buffers and full vocabulary logits compared byte for byte. |
| [positive9](model-batch/matrices/positive9/results.json) | Eight positive cases: budgets 1/512 × host/VRAM KV × token/loop launches; all exited 0. | Token launches compare every position in detail. Loop launches compare every prediction and only the final detailed buffers. |
| [control1](model-batch/matrices/control1/results.json) | Two one-ID positives and ten injected mismatches, in both KV placements; positive exits 0 and all five control kinds per placement exit 3. | Layer stream, full logits, DeltaNet, convolution and PLE readback controls each emit their reachability marker and one-bit mismatch. |
| [selection9](model-batch/matrices/selection9/results.json) | Two nine-ID positives and two selection controls, budget 1, both placements; positives exit 0 and controls exit 3. | Selection controls reach actual exclusion at the second completed block. |

The main batch therefore has **24 passing expected outcomes: 12 positive cases and 12 deliberately rejected mismatches**. [injection-results.txt](injection-results.txt) indexes the raw control evidence. The separate initial positive is not included in that count.

Budget 512 at nine IDs tests allocation/program integration below exclusion. It does not test production-budget exclusion at 2,052 visible positions. The long direct-position component matrix was repeated after the final integration and is recorded separately under [review-523110f](../review-523110f/README.md), using synthetic history and fixture projection weights. The earlier [review-e7921ae](../review-e7921ae/direct/matrix/results.json) capture is also retained.

## CPU checks

[cpu/model-cli.log](cpu/model-cli.log) records 26 passing boundary checks, including an explicit `T=0` rejection with exit 2. `cuInit` is intercepted; positive interception controls exit 99, while invalid model arguments/IDs and unreachable controls exit 2 before reaching CUDA. These are CPU admission checks, not GPU evidence. The model executable was linked against the CUDA toolkit's stub on engi for that test; its archy GPU build is recorded separately above.

The Perl model-runner checks reject wrong exits, missing reachability/mismatch evidence, timing-only summaries, stale dense output and incorrect scope/budget assertions. The private-engine parser checks reject corrupt/incomplete output. [cpu](cpu/) also preserves the cache-only build and symbol inspection.

## Private-engine request checks

The first `engine-batch` capture is deliberately retained as a **failed checker run**. The engine completed its command stream, but the checker incorrectly assumed `gen 1` would stop at nine processed positions and emit one token. The actual API runs `nprompt + n_new` positions and emits from `nprompt - 1`; the captured nine-ID `gen 1` request therefore processed ten positions and emitted two predictions. The checker now treats that request as a separate generation reference. Do not count the first report as a passed matrix.

The corrected [engine-batch-v2](engine-batch-v2/matrices/) passed **128 request checks**: 16 requests per private core × two launch modes × two KV placements × two budgets. Each budget has its own full-head, scalar-launch engine reference. [Budget 1](engine-batch-v2/matrices/budget-1/results.json) and [budget 512](engine-batch-v2/matrices/budget-512/results.json) each record four successful cores, their `config` responses, exact binary/helper identities and raw outputs.

For each core, two nine-ID prompts exercise fresh requests, alternating-prompt reset, snapshots at positions 1/2/3/5, reuse of each snapshot, and final fresh replay. The measured resume counts must match the requested boundaries; all live layer-state, PLE and final-logit fingerprints must match the corresponding fresh reference. These checks cover partial-block indexer-ring restoration. Full-head runs report the expected paired-position counts and zero headless chunks.

A separate `gen 1` request uses the nine-ID A prefix, processes ten positions and returns two predictions under the existing API. It has its own generation reference, rather than comparing its generated IDs with the teacher-forced A sequence. In chunk-3 mode it records six paired positions and three headless prefill chunks; its generated outputs and final state/logit fingerprints match the full-head engine reference.

Engine readbacks are FNV fingerprints of live state and final logits; they are not direct byte comparisons against the scalar model. The active eight-block count stays fixed for each private core and its snapshots. No production-budget exclusion is reached by these nine-/ten-position requests. The checker scripts were copied separately from the committed C build; their exact hashes are in the reports. The model runner's only subsequent change makes its helper functions importable, and its CPU checks were rerun.

## Shared-host handling and remaining acceptance

The wrapper acquired `/tmp/flash-sass-engine-validation.lock`, checked `nvidia-smi` and `ps`, and waited for a free GPU 7 before loading its own holder. The matrix runner checked admission again before each child. Cleanup stopped only the wrapper's owned holder and released its lock. Each wrapper preserves preflight, before-model and after-cleanup inventories; a utilization sample immediately after cleanup can still reflect the completed run.

The 366 per-case `.preflight.` inventory files are preserved without content changes in [preflight-inventories.tar.gz](preflight-inventories.tar.gz), so repeated process/GPU inventories do not obscure the case logs in the changed-file list. Paths in the archive are relative to the repository root. To restore the exact recorded paths, run `tar -xzf validation/issue9/review-4429cbd/preflight-inventories.tar.gz` from that root. Raw model/engine outputs, result JSON and wrapper inventories remain directly browsable.

The remaining issue #9 gates require actual-weight 48-block scalar/logit agreement, the production die placement, production-budget full-model exclusion, and the corresponding full-scope request/snapshot matrix. The six production GPUs remained occupied during this work; the resident engine was not stopped, parked, restarted or reconfigured. Keep `Refs #9` and leave the PR in draft until these gates have measured results.
