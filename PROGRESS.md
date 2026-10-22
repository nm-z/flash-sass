# Current: wait audit and revised production kernel

Reports: `validation/astra-2026-10-21/WAIT-DEPENDENCIES.md` contains the DeltaNet/QSA/common-MoE/PLE/output dependency tables. `phase499/wait-boundaries.csv` lists all 499 original candidate boundaries on every die. `WAITJOIN-RESULTS.md` records revisions and failures; `waitjoin-v6/operations.md` and `.csv` contain the final matched PROF_ALL tables.

Full candidate: all 48 layers and head, **499 phases/token on every die**, **144 local grid joins**, **354 peer acquires**, no MTP/FP8. The under-500 guard remains. It removes 354 duplicate producer grid polls plus one independent PLE RMS join, transposes publication copying into one owned row per lane, caches each CTA's fixed-order FP32 six-peer GR sum (160 KiB/die), and uses eight local warps for PLE quantization. Data-bearing local and peer handoffs remain.

AMAX had five equality → integer tie-break predicate dependencies with six clocks instead of thirteen. Canonical `kern/amax.sass` now uses thirteen clocks with yield. `phase_schedule_patch --amax-only` corrects exactly those five scheduling fields in the original baseline binary, with unchanged arithmetic opcodes. Both final measured kernels include it. Canonical-source rebuilds have identical library and phase-image SHA-256 values to the measured artifacts.

Passing short-context job **84dff40d-5cc1-44a7-a99f-7c26cd02a28c**: four 64-prompt/128-output requests per arm, all 512 candidate tokens match. Warm medians: **40.274955 TPS / 24.829326 ms/token candidate**, **40.092022 TPS / 24.942618 ms/token baseline**. Seeded sampling job **ba72bfdf-f20e-48a8-b436-c0f3181b324c**: all 16 tokens match. This recovers the original 32.99 TPS phase-kernel regression; the small final margin is not a large baseline speedup or a 60/140 TPS result.

**Long-context correctness is NOT resolved.** Final 4,096-prompt/16-output job **f608e18c-5665-4681-b41b-acdb1317d9e0** completes but has 10 mismatches beginning at token 6. Its timing is diagnostic only. Earlier baseline/candidate discrepancies, a client timeout and frozen-reference failures are retained in the report. AMAX's hazard is real but does not resolve every discrepancy. Do not promote the candidate or claim full correctness.

**Live serving: baseline with AMAX scheduling correction**, archy `/home/nate/work/flash-phase499-20261021-waitjoin-v6/baseline-amax13.cubin`, FNV **d215dc737080be6d**. Deployment job **233d4a55-d90d-4cfe-b902-1bf63586116d** succeeded; a following config read confirms fusion/grouping/balancing disabled, FP32 residuals, no MTP. The original baseline binary remains intact. Recovery startup retains its original command; no permanent boot-service migration is claimed.

Retained candidate artifacts: `/home/nate/work/flash-phase499-20261021-waitjoin-v6/`; library SHA-256 `04117f10a371f6d61ed78b4e06aa84d00a1246dba7d72c90bc591c6366d06c76`, phase image SHA-256 `da67831240e7200f8a5d8ea156df29ec9c0a64b9d4a88a52680393aae192a836`, FNV **14f13ad89b1e2445**. Immutable source/binary snapshots are retained. Every run uses the single canonical GPUQ DB; JSON files are exports, not additional databases.

No more optional repeats: the concrete remaining concern is the long-context output discrepancy. Diagnose it before further performance changes. Router/QSA/shared-expert producer joins remain where data crosses blocks. Launch occupancy is evaluated statically (249 registers, 256 threads/block, at most eight resident warps/SM), not redesigned. Prompt processing remains scalar. Do not silently resume MTP/FP8. Use safe atomic edits; the earlier engine.c recovery warning below still applies.

# Historical: original phase-count result

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

User subsequently authorized testing. A targeted warm-repeatability comparison,
job03bff6fb-db1f-4844-9d91-0e2e381ea17f, passed four128-token requests per arm:
all512 candidate output tokens match,499phases/all six dies, configuration restored
identically. Discarding warm-up, medians are32.9853TPS candidate vs40.1113baseline.
GR sum/write/read inclusive phase time increased2.16–2.32ms/token across dies.
This is profiling evidence, not a causal ablation. Kernel code was not modified.
Report and raw output are in PHASE499.md and its phase499 directory.
