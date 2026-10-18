# Issue #11: independent forward capture and candidate QSA observations

**PR remains draft.** The new adapters close a missing data-collection path;
these measurements do not close the numerical/quality acceptance contract.

Tested source: `438d2b6806de4baad627eb88c57c5ca3b9085d43c270661db2fd578a2401c1ba1fc`.
CPU work ran on `engi`; guarded target checks ran on `archy`.
The source tree was clean during the recorded source-specific runs.

## Measured results

| Check | Observed result | Scope |
|---|---|---|
| Full native oracle suite | 58 expected outcomes, exit 0 | Synthetic/frozen arithmetic, sparse selection, attention, and quantizer fixtures; not a new full-checkpoint comparison |
| Existing logit checker | 32 synthetic cases, zero failures | Comparator regression, not candidate/reference model agreement |
| Candidate capture serialization | 92 assertions, exit 0 | Mocked readback; dense/no-score handling, sparse selections, pool/K detiling, full vocabulary, per-die prefix identity, invalid inputs, failed reads and overwrite refusal |
| Address/undefined-behavior sanitizers | Same 92 assertions passed; leak detection enabled | Host serializer only |
| Production descriptor preservation | 90 cases, 2,250 descriptors, 1,320 expected pointer substitutions | Operations, masks, barriers and exchange order unchanged after normalizing the isolated buffer pointers; not GPU output parity |
| Reference capture serializer | Synthetic I/O test passed | Strides, shapes, exclusive files, cleanup, bounds and name/argument validation |
| Production library | Build completed against CUDA stubs on `engi` | **BUILD_ONLY**; existing warnings in unchanged `ref/quant.c` are retained in the log |
| Independent GGUF forward, visible 1 | Exit 0; 248,320 logits; 252 tensors across all 12 QSA layers | Real 48-layer model evaluation on smoke prefix `[1]`, **CAPTURED_NOT_VALIDATED** |
| Independent GGUF forward, visible 4 | Exit 0; 248,320 logits; 252 tensors across all 12 QSA layers | Real 48-layer model evaluation on smoke prefix `[1,2,3,4]`, **CAPTURED_NOT_VALIDATED** |
| Fresh complete GGUF hashes | Both shards match the pinned release | Original-checkpoint/tokenizer/conversion lineage is still not established |
| Guarded GPU attempts | Both exited 75, **BLOCKED_BUSY** | No GPU validation command started |

The independent runs were CPU-only, with two threads, a 4 GiB memory limit,
no swap allowance, and a 240-second timeout per attempt. Both finished within
the limit. Their logs include total service/runtime statistics, including
model loading and artifact writes; these are **not matched performance
benchmarks**. The captured short prefixes do not fulfill the required real
long sparse fixture matrix.

The final recorded GPU check was `2026-10-18T05:56:39Z`: a `bash` process,
PID `970242`, held `/tmp/flash-sass-engine-validation.lock`. `hold`, `fence`
and `flashcore` contexts occupied the M60s; the resident core PID was
`3575268`. No resident process was stopped, reloaded, or reused by this worker.
The GPU gate and occupancy records are under [gpu](gpu).
The staged production cubin and existing oracle adapter are unchanged from
the prior branch build; their actual hashes appear in the gate log. No fresh
GPU execution or newly assembled kernel is claimed.

## What is preserved

[summary.json](summary.json) separates measured observations from the unmet
acceptance gates. [source-sha256.txt](source-sha256.txt),
[binary-sha256.txt](binary-sha256.txt), build logs, package metadata, fresh
model hashes, and the local host proof identify the tested inputs and code.
The model files' size, inode and modification time were unchanged after the
independent runs.

The two compressed [reference archives](reference) contain **every actual
captured tensor**, the complete final logit row, copied consumed prefix IDs,
typed tensor catalog and capture manifest. They are not synthetic stand-ins
for a missing candidate execution. Each archive expands into `reference-1/`
or `reference-4/`; tensor catalogs/manifests are also available uncompressed.
`SHA256SUMS` covers all committed evidence files except itself.

Native and comparator reports, artifact hashes and logs are retained here.
Their large, generated synthetic input/output buffers remain local and are
reproducible from the commands below; they are not represented as files in
this evidence directory.

## Independent reference boundary

The installed package is `llama.cpp-hip b10705-2`. Its binary hashes are
recorded separately from the public source pin:
`2578138397d7b422bb0e160efdd429976c55fb55`,
`src/models/qwen4exp.cpp`, blob
`abf6a0502fbfbcad3232228dc0f45c18fba2091d`.
The corresponding fetched source SHA-256 is
`1fe800a11543d2af7aef898f75e6ee6ffb1f27f1486b4810cb926d94fc77e148`.
This does not attest a reproducible build of the local package.

This runtime is an independent implementation, but is **not** the pinned
Transformers arithmetic contract: its pooling boundary, unscaled score sum,
and cache-cell top-k/masking behavior differ. Those differences are documented
in [NUMERICAL_VALIDATION.md](../../../NUMERICAL_VALIDATION.md). The tool makes
no original-checkpoint provenance assertion and no numerical/quality pass
claim. Full-model differences must not be conflated with same-selection
kernel reduction or KV-quantization error.

## Reproduce

```sh
make -C kern qsa_reference_capture qsa_capture_test qsa_capture_prog_test qsa_logits_check
kern/qsa_reference_capture --self-test /tmp/new-reference-io
kern/qsa_capture_test /tmp/new-candidate-io
kern/qsa_capture_prog_test
perl kern/qsa_logits_check_test.pl --checker kern/qsa_logits_check --out /tmp/new-logit-regressions
perl kern/qsa_oracle.pl --full --out /tmp/new-native-suite
# Use a resource limit appropriate for the shared host.
kern/qsa_reference_capture "$MODEL_GGUF" validation/issue11/forward/reference/smoke-prefix.u32 1 /tmp/new-reference-1 2
kern/qsa_reference_capture "$MODEL_GGUF" validation/issue11/forward/reference/smoke-prefix.u32 4 /tmp/new-reference-4 2
```

See the numerical-validation document for the production `FLASH_QSA_CAPTURE`
mode and the required whole-prefix comparison before logit checking.

## Remaining gates before Ready for review

1. Execute the current component oracle and the new production capture on the
   target under the shared GPU lock. Establish capture-enabled/disabled output
   byte parity; descriptor preservation alone is insufficient.
2. Establish the original checkpoint/tokenizer/conversion lineage, including
   normalization-weight transformation, and connect the independent full-model
   reference to the intended numerical contract.
3. Run matched candidate/reference full-vocabulary teacher-forced comparisons
   and per-layer QSA diagnostics at the required real sparse lengths and tails.
   Apply the unchanged predeclared numerical limits; short captures cannot
   substitute for these measurements.
