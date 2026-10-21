# Issue #22 continuation review

Observed on October 20, 2026 at 10:28 PM PDT, from the isolated `~/work/fs22`
checkout. The PR branch now contains current `master`
(`c09927306f5170843116cc49fe85088ae42e02b3`) through merge commits
`ad2a7532c64bc22a323f1e0276518b8a7f991c4a` and
`c7f90f797ee634266129033d33b6a041a31949f`. Conflicts were resolved in
`.gitignore`, `QSA_MODEL_TEST.md`, `kern/Makefile`,
`kern/check_qsa_engine_requests.pl`, `kern/engine.c`, `kern/model_test.c`, and
`kern/test_model_cli.sh`. The newer master request checker and tracing changes
were retained alongside the PR's capture tooling.

## Checks completed

- `perl kern/check_qsa_engine_requests.pl --self-test`: **PASS**, 93 CPU-only
  checks. The suite reported 28 complete synthetic captures and 25 corrupt
  captures; it explicitly does not validate GPU execution.
- `perl kern/model_quality_test.pl /tmp/fs22-model-quality`: **PASS**, 76
  CPU-only comparator and capture-validation checks. The executable was a
  temporary copy of the Perl implementation; no C code was compiled.
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
but `dsh --config /tmp/fs22-build.toml build check-hosts benji` was rejected
with `Permission denied (publickey,password)`. No build or test was submitted.
I checked every socket in the documented SSH-agent directory; none authenticated
to Benji or Archy.

The required read-only Archy queue query,
`ssh archy gpuq search --help`, failed before reaching `gpuq` with SSH public
key/password authentication denied.
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

## Next step

I am posting the exact SSH blocker and completed CPU-only work on issue #22. I
will continue CPU-only artifact and metadata checks; full CPU tests must run
through `dsh build` on Benji, and GPU experiments must run through
`ssh archy gpuq run` after an authorized SSH identity is available.
