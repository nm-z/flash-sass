# Independent full-model capture — issue #11

**The fresh native reference capture completed. The diagnostic comparison with
the archived GPU capture fails both predeclared logit limits. Issue #11 remains
unresolved and PR #31 must remain draft.**

This workstream owns independent full-model logits and per-layer QSA
diagnostics. A separate worker owns verified original-checkpoint and conversion
lineage. These records do not assert that lineage or replace that worker's work.

## Fresh execution

The C adapter executed the complete 48-block model on engi using a clean,
separate CPU-only build of llama.cpp at
`fcac19079f3f2985fe74ed845f0880e7e9a8030a`. It decoded the four input IDs
`10,11,12,13`, without inserted tokens, and retained full-vocabulary rows at
visible lengths **1 and 4**. These are consecutive-token microfixtures, not
held-out natural-language quality data.

- Start: `2026-10-18T05:40:42Z`; elapsed capture time: **70.061360 seconds**.
- Complete prefix: 4 tokens; selected zero-based positions: `[0,3]`.
- Logits: 2 rows × 248320 vocabulary columns = **496640 finite FP32 values**.
- Native snapshots: **408 tensors** across the two selected lengths.
- Each selected length contains all twelve QSA blocks (3, 7, …, 47), every
  required indexer/selection/attention tensor, and all 48 `l_last` block outputs.
- Selected sets contain the one visible token at length 1 and all four tokens
  at length 4, at every QSA block.
- The reference uses FP32 K/V with CPU operation execution, disabled flash
  attention and MTP, and no GPU backend.

The adapter source SHA-256 is
`119011fb745b3b01bb504b901b012d6d98ce7fd5c0344169612d09f3ec782e73`.
Its executed binary SHA-256 is
`4108832d8a608e8259d14f509d43aebdaa3b7e62b1a515e8888372a3f20e130d`.
The identity record's repository HEAD is the pre-edit base `7e52c66`; the
measured source hash identifies the new code that was actually compiled.

`reference-runtime.log.gz` preserves the actual dynamic loader trace. It loads
the four private llama/ggml libraries whose before/after hashes match in the
identity files. The reference source worktree was clean. The build logs and
CMake cache preserve the build configuration; no existing shared reference
build was overwritten.

The existing Q2_0 GGUF shard paths were used. `gguf-sha256-after.txt` records a
fresh full-file hash after this capture, and
[the earlier shard identity record](../resume/gguf-identity.json) records the
matching pre-capture hashes and pinned published release. These observations
establish the quantized artifact identities; they do not establish the original
checkpoint revision or conversion process.

## Historical target comparison

The comparison downloads the pinned four-token, production-budget, pinned-host
KV capture from
[PR #46 at fcf435f728760b64de30214478524666093468ec](https://github.com/nm-z/flash-sass/tree/fcf435f728760b64de30214478524666093468ec/validation/issue22/archy-06968d1/gpu/t4-budget512-kv1-raw).
That GPU run used flash-sass executable source
`06968d1a5d6d7fdaf12106236cc39a1cb5cdf389`, all 48 blocks, actual embedding, PLE
and output-head weights, production QSA budget 512, and IDs `10,11,12,13`.
It is **historical evidence, not a new GPU run or a run of the current PR head**.

The complete input prefix matches exactly. The candidate IDs and all four
logit rows match the frozen artifact manifest, including logit SHA-256
`ff487223d8ca1c397cfc01403c3a28f5412e39341fc29d8ecb62bfda251e6e06`.
The canonical vocabulary-column hash matches the fresh reference. The wrapper
then extracts target rows at the same two selected positions. It records the
archived raw metadata's own hash separately; that normalized file does not have
the original freezer metadata byte hash.

The issue #11 limits were already fixed at maximum absolute error **0.10** and
normalized RMS error **0.01**, enforced for every row and the aggregate. They
were not changed after observing this result.

| Visible length | Max absolute logit error | Normalized RMS error | Top-1 IDs, reference / target | Limits |
| --- | ---: | ---: | --- | --- |
| 1 | 0.7673995495 | 0.05300301267 | 18 / 18 | Fail |
| 4 | 0.9910707474 | 0.07493433639 | 271 / 271 | Fail |
| Aggregate | 0.9910707474 | 0.06226231368 | 2 of 2 agree | Fail |

Aggregate logit RMSE is **0.1642817181**. Both rows fail. The numerical checker
returns **3**, and the wrapper preserves the completed failure report.

The C checker's row indices 0 and 1 correspond to complete-prefix positions 0
and 3. Top-1 agreement does not override either error limit. At these lengths
all visible tokens are selected, so this result cannot attribute the discrepancy
to sparse selection. Candidate/reference weight precision, other model math,
and the historical target revision remain possible causes. The reference
snapshots provide diagnostics for subsequent localization; candidate per-layer
snapshots for a matched current run are still missing.

## Controls and reproduction

`negative-controls.tap` records **7/7 passing rejection controls** against the
actual retained capture. In particular, modifying an unselected earlier token
is rejected even though the selected current-token IDs still match. A missing
QSA output is rejected even when the total tensor count is changed consistently.
Incomplete captures, future selection positions, truncated attention tensors,
changed target logits and implicit use of historical v1 data are also rejected.

From this PR checkout:

```sh
bash validation/issue11/full-model/replay.sh /absolute/new-replay-directory
```

The replay verifies the local evidence checksums, extracts the complete native
reference capture, downloads the immutable archived target, verifies its IDs,
logits and column mapping, reproduces the numerical failure, and runs all seven
negative controls. It uses Bash, C and Perl and starts no GPU work. The expected
successful replay result is `REPLAY_CONFIRMED_NUMERICAL_FAILURE`.

See [QSA_MODEL_REFERENCE.md](../../../QSA_MODEL_REFERENCE.md) for a new native
capture, the tensor schema and the complete-prefix comparison contract. The
raw reference archive includes the logits, full IDs, selected positions,
vocabulary records, every indexed tensor and all derived selection sets.
Per-layer activation norms and hashes are also available as reviewable JSON.

## Remaining blockers

1. **Fresh target execution:** the shared GPU lock was repeatedly busy during
   this workstream. Attempts and holder PIDs are recorded in the
   `flash-sass-issue11-gpu-check-*.txt` files. At check 12
   (`2026-10-18T06:04:11Z`), Bash PID **1002367** held the lock and five waiters
   were queued. No new GPU capture was launched. The same check
   showed only 3–77 MiB free on resident model GPUs 0–5. The lock does not
   authorize stopping those workloads or reloading their engine.
2. **Matched per-layer localization:** a current full-model target capture with
   the same complete prefix and all QSA-layer diagnostics is needed to locate
   the observed dense-prefix discrepancy. The supplied target archive has full
   logits but no corresponding layer snapshots.
3. **Production sparse-length coverage:** full-model paired captures at the
   requested 2047–2052, 3100, 8000, 65000, 262000 and modulo-four tail lengths
   have not been measured here. The new adapter supports selected lengths but
   this short run does not exercise sparse selection.
4. **Original-checkpoint and conversion lineage:** this remains the other
   worker's assigned gate. A same-GGUF independent backend capture cannot
   certify it, nor can it establish checkpoint quality.

`comparison.json` explicitly keeps `issue11_resolved:false`,
`independent_checkpoint_evidence:false`, and `quality_validation:false`.
