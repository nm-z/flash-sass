# Issue #9 review map

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9) and
[PR #105](https://github.com/nm-z/flash-sass/pull/105), stacked on
[PR #35](https://github.com/nm-z/flash-sass/pull/35).
The PR remains draft while the production-budget reference and comparisons
finish. This page maps each requirement to its implementation and measured
scope; the run records retain exact identities and unsuccessful attempts.

| Requirement | Implementation and evidence | Current scope |
| --- | --- | --- |
| Explicit dense baseline | `MODEL_ATTN=dense` in `kern/model_test.c`; this is also the default. `MODEL_ATTN=qsa` selects the production chain. | Preserved as a separately named source path. |
| Actual full model and production placement | Complete packed export, real checkpoint/indexer weights and norms, all 48 layers, embedding, PLE, full head and six production dies. Live holder geometry checks reject incompatible layouts. | The [current source record](whole-model-e2ff58f/README.md) reports nine-position actual-weight results with production shares/map. |
| Instruction-order scalar comparison | CUDA-free C producer, typed per-layer/QSA/recurrent/PLE/logit records and exact byte comparisons. Raw selected IDs and prediction tuples are retained. | Independent serial and four-worker nine-position records and checkpoints match exactly; each reader verifies 9 positions / 8280 items. |
| Host/VRAM KV and launch modes | GPU consumer reads the sealed scalar records before CUDA initialization, asserts full scope and checks executed QSA opcodes/budget. | All four current-source modes pass: per-token modes compare every complete frame; single-launch modes compare every prediction and the final complete frame. |
| Actual selection under a small budget | Budget 1, explicit teacher IDs, successful capture requires full scope and scalar agreement. | Twelve actual selected-ID captures per current-source run, one for every attention layer at visible length 9. Selection exclusion is active at visible lengths 8 and 9. |
| Requested cache depths and modulo-four tails | Direct-at-position harness constructs deterministic inputs, synthetic history and projection fixtures. | [GPUQ `ef56e0c2`](whole-model-3902787/README.md) passes 14 cases / 56 positions, both KV placements, budget 512, starts 1, 2048, 2052, 3100, 8000, 65000 and 262000, each with four tails. The 2048 case includes 2051. This is component coverage. |
| Actual-weight production-budget exclusion | Uniform four-worker scalar replay from position zero, 2055 teacher-forced positions, budget 512, durable chunks of 32. | [The initial batch](whole-model-e2ff58f/cpu-deep-first256.excerpt) committed 256 positions. The directly read checkpoint later reached position 1248 at record offset 131918388 bytes; the record file has 1691256 bytes beyond that durable offset. The DSH result is unconfirmed after the Sentry NFS timeout, and no full-stream validation ran. The 78.9 GB model shard set does not fit on Archy's 1.8 GB root or 3.9 GB HDD free space, so no replacement run has been started. Full completion and subsequent GPU comparisons remain pending. |
| Deliberate mismatch detection | Direct selection/gather/output controls; complete-model controls flip one bit in a host copy of a GPU readback. The C matrix requires the intended label, exact comparison count, mismatch position and program exit. | Six direct controls reject their intended errors. Current-source GPUQ `1a7a5f7c` passes all seven whole-model expected-rejection cases; the runner also passes twelve CPU fixtures. Complete per-case evidence is in the current source record. |
| API boundary and input integrity | Full-model/capture/ID/geometry guards, sealed source and model identities, locked records, layout checks before hashing and on reading. | Current consumer passes 46 pre-CUDA boundary checks; current oracle passes 17 API/CLI cases. Three real input-rejection cases leave empty record/receipt files. |
| Consecutive requests, snapshot/resume and free-running generation | Private C runner starts four independent cores and checks complete state/logit fingerprints, predictions and resume/pair/chunk counters. Teacher-forced comparison and generation requests are separate. | One complete nine-position matrix passes all 64 requests. Other paired runs expose an intermittent mismatch, tracked in [issue #109](https://github.com/nm-z/flash-sass/issues/109). The canonical 2055-position/B512 matrix fails in VRAM scalar mode with a CUDA illegal memory access; [issue #110](https://github.com/nm-z/flash-sass/issues/110) and the current source record retain the failed gate. |
| Source, library, cubin, weights, flags and device/driver identities | GPUQ validates the kernel receipt and records binary/source/checkpoint/holder identities and arguments. CPU excerpts retain compiler flags and binary hashes. | Durable compressed GPUQ exports and DSH excerpts are linked from the source-epoch records. Optional resident traces were unavailable; they are not recorded as zero. |

The canonical numerical source epoch is
`e2ff58ff1b0e956d61a1c93830c5f66210d27788`. Later validation/Make/documentation
commits retain those numerical file contents; the consumer still checks their
content identity against its scalar input seal. The serial/parallel contract
and version-2 input-receipt derivation are described in
[PARALLEL_ORACLE.md](PARALLEL_ORACLE.md).

The separate GEN-between-pair-bodies experiment for #109 uses isolated
checkouts and its own cubin. PR #105 retains the canonical cubin whose direct
and full-model results are listed above. Matched stock and transformed request
controls both fail a fingerprint check; the experiment does not supply a
runtime fix. Its separate
[run record](https://github.com/nm-z/flash-sass/tree/test/issue109-pair-grid/validation/issue109)
retains both outcomes.

The separate last-phase recorder for #110 is reviewed in
[draft PR #112](https://github.com/nm-z/flash-sass/pull/112). Its instrumentation
and short GPU checks are diagnostic evidence; this acceptance map retains
the canonical numerical source and cubin.
