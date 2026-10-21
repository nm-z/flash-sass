# Issue #22 continuation review

Observed on October 20, 2026 at 7:01 PM PDT, from the isolated `~/work/fs22`
checkout. The PR branch now contains current `master`
(`270aa639dd9dc86eb16db698e318aaa7e7d3fd2c`) through merge commit
`ad2a7532c64bc22a323f1e0276518b8a7f991c4a`. Conflicts were resolved in
`.gitignore`, `QSA_MODEL_TEST.md`, `kern/Makefile`,
`kern/check_qsa_engine_requests.pl`, `kern/engine.c`, `kern/model_test.c`, and
`kern/test_model_cli.sh`. The newer master request checker and tracing changes
were retained alongside the PR's capture tooling.

## Checks completed

- `perl kern/check_qsa_engine_requests.pl --self-test`: **PASS**, 93 CPU-only
  checks. The suite reported 28 complete synthetic captures and 25 corrupt
  captures; it explicitly does not validate GPU execution.
- `make -C kern -n model_test model_capture_test model_layer_capture_test
  qsa_quality qsa_attention_quality model_quality`: **PASS**, Makefile recipes
  and dependencies expanded. `-n` did not compile anything.
- `bash -n` on the issue #22 CPU, placement-replay, and GPU-runner scripts:
  **PASS**.
- `perl -c` on the merged request checker, model-quality tool, and QSA replay
  wrapper: **PASS**.
- The Hugging Face CLI confirmed `Qwen/Qwen3.8-Flash-Next` revision
  `de4b8e4d43b917e7706784d8bb445c9af86a3540` is public and ungated, with 131
  Safetensors shards and `Qwen4ExpForConditionalGeneration` architecture. No
  checkpoint weights were downloaded or executed.

## Checks blocked by host access

The required full CPU build/test commands were not run locally on engi. The
default `dsh build check-hosts benji` failed before host probing because the
active DSH config has no `[build]` field. An isolated build-only config parsed,
but `dsh --config /tmp/fs22-build.toml build check-hosts benji` then timed out
during the SSH banner exchange. No build or test was submitted.

The required read-only Archy queue query,
`ssh archy gpuq search --help`, failed before reaching `gpuq` with SSH public
key/password authentication denied. The available SSH agent has no identities.
No GPU job was started, and no process or model on Archy was changed.

The pinned original checkpoint metadata was confirmed, but its weights were not
downloaded or executed. Existing reports under `review-2284a21` remain the only
model-level evidence: short same-implementation fixtures and fixed-ID attention
controls. Original-checkpoint comparison, exact conversion lineage, complete
precision/mask controls, production-budget model-length coverage, and held-out
long-context quality remain unmeasured.

## Acceptance status

Criteria 1–6 remain **unmet** for the gaps above. Criterion 7 remains **met**:
reports distinguish completed, diagnostic, blocked, and unmeasured results.
`issue22_resolved` and `ready_for_review` remain false. This integration and
CPU parser evidence is not sufficient to mark PR #46 ready or to claim issue
#22 is fixed.
