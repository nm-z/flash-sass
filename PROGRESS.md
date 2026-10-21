# Production phase-count result

The requested full-model production path has been implemented and runtime
verified at **499 phases/token on every one of six dies**. All48 layers and the
output head execute; MTP and FP8 are disabled. Implementation and evidence are
published in draft PR #135; implementation commit `03c14eb`.

Authoritative report: `validation/astra-2026-10-21/PHASE499.md`, with raw logs and
job metadata under `validation/astra-2026-10-21/phase499/`. All runs remain in
archy's single canonical GPUQ history database.

Validation complete:
- First full end-to-end request: job1564d463-115c-4fe9-a648-6b4304260573.
- Matched64-prompt/128-output greedy comparison: job4a49c988-40f2-4074-af34-823fe636f752; zero mismatches,499phases/all six dies.
- Matched4096-prompt/16-output comparison: job09a0a0d6-97d1-4f67-9957-5fe22b29c1fb; zero mismatches,499phases; nontrivial QSA selection exercised.
- Matched seeded sampling: joba3747079-01be-4b69-a1f8-99a791c1a28f; zero mismatches,499phases.
- Final configuration restored identically after all three matched comparisons.

Performance is a regression:32.9848TPS vs40.1152TPS matched short-context baseline;
29.7441TPS vs34.8074TPS at4096prompt tokens. Do not claim a speedup or60/140TPS.
The phase-count objective is met; the earlier phase-cost/prefetch optimization
ideas have not been completed. Those are separate remaining performance work.

Live artifacts: archy `/home/nate/work/flash-phase499-20261021-v8/`;
libengine_exact.so and die_phase.cubin. Runtime map0,1,2,3,4,5,0,1,2,3,4,5;
context32768, B512, scalar prompt path. Producer quantization, matrix grouping
and fused GR enabled. Residual storageFP32. Startup recovery unit retains its
original baseline command; no permanent boot-service migration is claimed.

Failures retained in report: firstv6 illegal access; v7 first-QSA deadlock.
Corrections: warp-uniform publication loops and dependent-predicate spacing in
selection. Existing weight holders were retained across faulted-core restarts.

A local edit accidentally truncated engine.c during implementation. It was
reconstructed from the captured DFlash-v2 source plus the recorded phase, GR,
FP8-disabled, and MTP/prefill edits. The recovered planner first reproduced the
previous496/508 counts; the subsequent full-model comparisons above validate
the active path. Recovery material and source backups live under
`~/.cache/flash-engine-recovery/`. Use apply_patch or evaluate edits fully before
opening a destination, then rename atomically. Do not repeat the truncation bug.

No more tests are needed for the phase-count goal absent new changes/failures.
Do not silently resume MTP/FP8 work. Prior experimental sources are retained but
are not enabled or validated by this phase-count result.
