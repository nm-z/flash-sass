# Direct-position validation after master integration

## Measured result

The direct component executable was rebuilt at `523110f37d2aa27e4af29ea8c831f084fd6c8a36`, after merging master `3d57d734d63aa74f167a6b5a02c616ddc018d2ff`. The merged fixture correction clamps the legacy `idxscore` fixture to its last valid position; `qsa_direct_test.c` includes that source, so the direct matrix was repeated.

[results.json](direct/matrix/results.json) records **41 passing cases**: 38 positive cases checking **146 positions**, plus three negative controls. [exit-code.txt](direct/exit-code.txt) records runner exit 0; [matrix-console.log](direct/matrix-console.log) indexes the raw per-case logs.

The matrix covers host-pinned and VRAM KV, budget 512, the required length families and their modulo-four tails through 262,144, budget-8 short-history replay, diverse selection patterns and seeds, and selection/gather/output injections with expected exit 3. These are synthetic-history component fixtures with fixture projection weights. They do not establish actual-weight full-model logits, production six-die execution, or full-model request/snapshot behavior.

## Build and execution identity

[build.log](build.log) and [build.exit](build.exit) record the rebuilt direct executable. [source-revision.txt](direct/source-revision.txt), [component-sources.sha256](direct/component-sources.sha256) and [binaries.sha256](direct/binaries.sha256) identify the source, executable and cubin. The existing cubin was copied from the `4429cbd` build, which reused the `e7921ae` cubin; its generator inputs did not change. Other copied executables were not represented as rebuilt or rerun at this revision.

The [source comparison](model-engine-source-equivalence.txt) records that the only committed `kern/` or `ref/` change between measured model/engine revision `4429cbd` and integration revision `523110f` is the included legacy fixture correction. The [eight-block model and private-engine results](../review-4429cbd/README.md) retain their actual `4429cbd` C executable identity. Subsequent runner and documentation changes do not relabel those binaries.

Execution used GPU 7, UUID `GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4`, on archy. The wrapper acquired the shared `/tmp/flash-sass-engine-validation.lock` and the existing per-GPU reservation, then checked `nvidia-smi` and `ps` before running. [Preflight GPU state](direct/preflight-gpu.csv), [compute processes](direct/preflight-compute.csv), [process inventory](direct/preflight-ps.txt), [final GPU state](direct/after-gpu.csv) and [final process inventory](direct/after-ps.txt) are retained. Both locks were released when the wrapper ended. The resident engine and its holders were not changed.

## Acceptance state

This repeat adds current integrated-source component evidence. Issue #9 still needs measured 48-block scalar/GPU agreement, the production six-die layout, actual-model budget-512 exclusion and the corresponding full-model request/snapshot matrix. The PR remains linked with `Refs #9` and stays draft.
