# Production wait and publication revision

The full production kernel retains **499 logical phases per token on all six dies** and reduces end-of-phase local grid joins from **499 to 144**. It retains all **354 peer acquires** and the sub-500-phase admission guard. MTP and FP8 are disabled; all 48 layers and the output head execute.

The final short-context comparison measured **40.2750 decode TPS (24.8293 ms/token)** against **40.0920 TPS (24.9426 ms/token)** for baseline. The 0.46% margin over baseline is small. Both final kernels include the AMAX predicate scheduling correction described below. This repairs the large regression in the original 499-phase implementation; it does not establish a large speedup over baseline or meet the 60/140 TPS targets.

**Long-context correctness is still open.** The final matched 4,096-prompt/16-output comparison has **10 output mismatches**, starting at token 6. Its timings are diagnostic only. The candidate is not promoted to serving. Short-context and seeded-sampling comparisons pass; that does not override the long-context failure.

[DeltaNet/QSA dependency tables](WAIT-DEPENDENCIES.md) · [Every original candidate phase boundary](phase499/wait-boundaries.csv) · [Complete final per-operation tables](waitjoin-v6/operations.md) · [Final per-die operation CSV](waitjoin-v6/operations.csv)

## Complete-kernel measurements

Each stage below was built as a complete production kernel before being run. Each job ran four matched requests per arm, with 64 prompt and 128 generated tokens. Request 0 is warm-up; requests 1–3 determine the median. Times are host-observed post-first-token elapsed time divided by 127 decode intervals, not GPU-only isolated operator timings. The measured baseline used the same resident weights and scalar prompt-processing configuration in every job.

| Kernel revision | Baseline TPS | Candidate TPS | Candidate ms/token | Candidate token comparison |
|---|---:|---:|---:|---|
| Remove duplicate local grid joins | 40.1187 | 32.9457 | 30.3530 | All 512 match |
| Also transpose producer publication into one row per copying lane | 40.1054 | 37.1923 | 26.8873 | All 512 match |
| Also cache each CTA's fixed-order six-peer GR sum | 40.1069 | 40.2043 | 24.8729 | All 512 match |
| Also distribute PLE quantization across eight local warps; passing historical repeat | 40.0925 | 40.2278 | 24.8584 | All 512 match; subsequent discrepancies required investigation |
| Also explicit cached-memory acquires and AMAX predicate scheduling correction in both kernels | **40.0920** | **40.2750** | **24.8293** | **All 512 match; all eight requests complete** |

The first stage shows that eliminating the redundant grid polling alone did not improve throughput. The following consumer already waited for the same local producers through its peer acquire. The publication and GR changes reduce work in the merged bodies rather than moving a dependency elsewhere.

The final matched prompt rates were **41.5234 tokens/s baseline** and **41.7223 candidate**. Prompt processing remains scalar; this work does not claim chunked-prefill acceleration.

| Final workload | Baseline TPS | Candidate TPS | Candidate output comparison | Status |
|---|---:|---:|---|---|
| 64 prompt, four 128-token greedy requests; warm median | 40.0920 | 40.2750 | All 512 match | Pass |
| 64 prompt, 16 sampled tokens; seed 7, temperature 1, top-k 20, top-p 0.95 | 40.0447 | 40.1767 | All 16 match | Pass; single request per arm |
| 4,096 prompt, 16 greedy tokens | 35.5142 | 35.5589 | 10 mismatches; first at index 6 | **Fail; both timing values are diagnostic** |

All final requests complete without a device error. Every candidate die's uploaded runtime program reports 499 phases, 144 local grid joins and 354 peer acquires. All benchmark restorations report identical baseline configuration. The final long-context comparison is job `f608e18c-5665-4681-b41b-acdb1317d9e0`.

## What changed

- The planner marks each sending phase followed by an exchange acquire. The footer retains output fences, every block's peer completion flags, generation bookkeeping and local descriptor handoff, but bypasses its redundant local sixteen-block poll. The consumer still acquires six dies × sixteen flags, including local producers. This removes 354 grid joins. The PLE key/query normalization group also skips one join before the independent value GEMV, leaving 144.
- Publication copying now assigns different owned rows to different lanes. It does not repeat the GEMV's mostly inactive subgroup-leader layout. It preserves ownership across all quantization formats, exits only whole inactive warps, fences the producing threads and keeps the real handoff before copying another warp's rows. Nonpublishing matrix bodies skip that extra handoff. No additional system fence is placed in the row/peer copy loop.
- Each GR CTA now computes the six-peer sum once, with the original FP32 addition order, then reuses its private cache for residual branch normalization. Sixteen 10 KiB caches cost **160 KiB per die**. The data-bearing CTA handoff remains; there is no cross-CTA cache reader or new cross-CTA spin. The original standalone GR image is unchanged. ABI markers bind the phase library to the matching wait/cache image.
- PLE gather keeps its original arithmetic in warp 0. All eight warps of the same block now quantize independent vectors after a local gather-to-quantizer handoff. There is one PLE gather block and no new grid phase.
- New same-block global-memory producer/consumer handoffs explicitly invalidate cached loads after the producing warps have fenced and joined. This preserves the publication and GR cache memory-visibility contract; those acquires are not removed as empty waits.
- AMAX's five FP equality predicates now have a 13-clock stall with yield enabled before the dependent integer tie-break. The canonical `amax.sass` assembly source is corrected. An identity-bound C patch applies exactly the same five scheduling-field changes to the existing baseline binary without changing arithmetic opcodes.

GDN conv/state/norm/gate/quantization, routed/shared gate/up/activation, and QSA score/selection remain fused. The router, shared expert and QSA producer-ready joins remain where the audit identifies actual cross-block data. This revision does not parallelize router selection, rewrite the PLE branch RMS arithmetic, or change launch occupancy. The two first-row timing terms in the original router comparison were an incomplete subtotal: the baseline router matrix projection was in a shared BF16 opcode bucket. That subtotal alone did not establish a router arithmetic regression.

## Per-operation evidence

The linked table contains every recorded opcode and each die separately, equivalent to `PROF_ALL=1 kern/prof.sh`. Timings include synchronization and changed attribution. All values below are final-token µs/token averaged over six dies, not request medians or isolated fusion ablations.

| Candidate opcode | Wait-removal-only revision | Publication revision | GR-cache revision | Final AMAX-corrected comparison |
|---|---:|---:|---:|---:|
| `grwrite_read` | 5,633.0 | 5,656.1 | 3,959.1 | 3,941.0 |
| `q2moedn` | 1,742.1 | 1,464.2 | 1,470.1 | 1,482.4 |
| `q4kk6144` | 1,725.8 | 1,273.3 | 1,272.0 | 1,291.8 |
| `iq4k6144` | 1,321.3 | 807.9 | 807.9 | 802.8 |
| `q6kk1536` | 1,086.5 | 994.5 | 1,001.1 | 1,004.0 |

The complete residual sum/read family is still **4,085.0 µs versus 3,552.2 baseline**, an inclusive excess of 532.8 µs/token/die. The cached sum reduces the original excess substantially but does not eliminate it. Allocation of baseline ACTQ, XSEND and BF16 matrix times to individual merges remains unavailable from opcode totals; the reports do not invent those allocations.

## Discrepancies and unsuccessful attempts

Job `a3dd062b-59eb-4ddd-bfc8-6dd2e9e69511` returned failure because baseline run 3 emitted token 303 where the first baseline reference emitted 1083 at printed index 102. All 512 candidate tokens matched that first reference. Its timing is diagnostic only. The unchanged-kernel repeat `bbcf3f39-3404-455f-a010-83eda2c9ed17` passed all requests and comparisons, but did not establish that the underlying nondeterminism was fixed.

Long-context job `64073d15-a87a-4975-af52-f71669a3c3e9` completed both arms but had 13 output mismatches. The candidate's full 16-token output matched **both** arms of the earlier passing DB job `09a0a0d6-97d1-4f67-9957-5fe22b29c1fb`; the new baseline chose a different continuation starting at token 3. A subsequent candidate build with explicit cached-memory acquires, job `a8a1591e-1959-42bd-b5c2-52d367d7bae7`, diverged on its first short request at token 50 (including a choice of token 0), then matched on its next three requests. Its frozen-reference long-context job `f4cf08da-0287-4294-abd2-b8c86a3979a5` had 10 mismatches. These are failed comparisons, not passing performance results.

Inspection then found a concrete scheduling defect common to both kernels: each of AMAX's five shuffle-reduction steps issued `FSETP.EQ` with a six-clock stall and immediately consumed P2 in `ISETP.LT ... P2`. Maxwell predicate consumption requires 13 clocks, with the yield hint for the high stall count. The corrected control is `--:-:-:Y:d`. This changes scheduling only. The observed erroneous choices of smaller token IDs are consistent with a stale equality predicate; the full causality of every historical mismatch has not been independently isolated. [MaxAs predicate latency and yield documentation](https://github.com/NervanaSystems/maxas/wiki/Control-Codes)

The existing baseline correction is bound to FNV `9abf8aafb9134d59` and changes five control fields at instruction PCs `0x573b0`, `0x57448`, `0x574d8`, `0x57570`, `0x57608`. Its resulting FNV is `d215dc737080be6d`. No instruction opcode changes. The final matched short-context job `84dff40d-5cc1-44a7-a99f-7c26cd02a28c` uses corrected AMAX in **both** arms and passes all eight requests. Final seeded sampling job `ba72bfdf-f20e-48a8-b436-c0f3181b324c` also passes all 16 candidate tokens. These workload checks are not an exhaustive correctness proof.

The final corrected-AMAX long-context comparison still fails at token 6, with 10 mismatches. The predicate correction therefore does **not** resolve the complete output discrepancy. The full candidate design is retained for further diagnosis; no fallback or partial-model result is presented as success. The unresolved difference needs isolation before the candidate can replace serving baseline.

An earlier long-context attempt on the wait-removal-only version, `01a340c1-3067-4474-8ddd-e7f7bb0cc32d`, used a 90-second client receive timeout even though the previous baseline 4,096-token prompt took about 108 seconds. The client timed out in baseline prefill before receiving a token and before starting the candidate arm. Serving was restored identically. It supplies no long-context performance or output comparison. The final version uses a 300-second client timeout for that workload.

## Resource and fence evaluation

The actual baseline binary reports `REG:249`, `SHARED:49152`, `STACK:0`, `LOCAL:0`. The final phase binary also uses 249 registers/thread and 49,152 shared bytes per block. At 256 threads/block, two blocks exceed Maxwell's 65,536-register budget, so both kernels have at most one such resident block per SM: eight resident warps against a 64-warp architectural maximum. Actual eligible/issued warp activity was not measured. The static resource result supports a register/layout redesign as a separate lever; it does not justify increasing a software-barrier grid beyond guaranteed residency.

Disassembly of the actual baseline binary confirms the predicated footer `MEMBAR.SYS`. Both schedules publish 354 exchanges with one fencing thread per block: **5,664 footer fence-thread executions per token per die**. Fusion did not change one system fence per phase into sixteen. Source details and the distinction between resident blocks and useful work are in the dependency report. [NVIDIA Maxwell resource limits](https://docs.nvidia.com/cuda/archive/11.6.1/maxwell-tuning-guide/index.html)

## Artifacts and canonical records

Final measured artifacts on archy: `/home/nate/work/flash-phase499-20261021-waitjoin-v6/`.

- `libengine_exact.so`: SHA-256 `04117f10a371f6d61ed78b4e06aa84d00a1246dba7d72c90bc591c6366d06c76`
- `die_phase.cubin`: SHA-256 `da67831240e7200f8a5d8ea156df29ec9c0a64b9d4a88a52680393aae192a836`; runtime FNV `14f13ad89b1e2445`.
- `baseline-amax13.cubin`: SHA-256 `9e239d8658765496f4396e1751f8fd469bfdbcce2b4b05c7be80bd92bed3a684`; FNV `d215dc737080be6d`.
- Full phase kernel text: 984,384 bytes; 249 registers; phase ABI v6 plus exchange-join/cache/AMAX-schedule ABI markers.
- Library revision string: `df6bb69-dirty-exact`; source and patch snapshots are retained next to each immutable remote artifact set.
- Rebuilding from the final canonical AMAX source produces exactly the same library and phase-kernel SHA-256 values as the measured artifacts. The canonical-source archive is retained separately from the initial source archive.

Every job remains in `/home/nate/.local/state/gpuq/history.sqlite3`. The linked JSON files are exports of those records, not additional databases.

| Job | Purpose / result |
|---|---|
| `0ae67de8-e4b3-47db-b329-aaca5463c685` | Return serving to baseline |
| `83e2e806-d830-4d6d-a4d2-d60921fbf171` | Full wait-removal-only comparison; passes |
| `9c00680c-b88e-4583-baf8-4d0127c6c379` | Full publication revision comparison; passes |
| `d0b1773c-1e9d-4680-b88d-bef51d2b1ca5` | Full cached-GR comparison; passes |
| `a3dd062b-59eb-4ddd-bfc8-6dd2e9e69511` | First PLE-warp revision comparison; baseline discrepancy, job fails |
| `bbcf3f39-3404-455f-a010-83eda2c9ed17` | Unchanged final-kernel repeat; passes |
| `01a340c1-3067-4474-8ddd-e7f7bb0cc32d` | Earlier long-context client timeout; no candidate result |
| `64073d15-a87a-4975-af52-f71669a3c3e9` | PLE-warp version at long context; comparison fails |
| `a8a1591e-1959-42bd-b5c2-52d367d7bae7` | Explicit-acquire revision; first-request discrepancy, job fails |
| `f4cf08da-0287-4294-abd2-b8c86a3979a5` | Explicit-acquire revision against frozen long reference; fails |
| `84dff40d-5cc1-44a7-a99f-7c26cd02a28c` | Final AMAX-corrected matched warm comparison; passes |
| `ba72bfdf-f20e-48a8-b436-c0f3181b324c` | Final matched seeded sampling; passes |
| `f608e18c-5665-4681-b41b-acdb1317d9e0` | Final matched long context; completes but fails 10 token comparisons |
| `233d4a55-d90d-4cfe-b902-1bf63586116d` | Serve baseline with AMAX scheduling correction; succeeds |

## Final serving state

After the final comparison, serving was switched through GPUQ to the baseline with only the five AMAX scheduling controls corrected. A following `config` read confirms cubin `/home/nate/work/flash-phase499-20261021-waitjoin-v6/baseline-amax13.cubin`, FNV **d215dc737080be6d**, fusion/grouping/balancing disabled, FP32 residuals and no MTP. The original baseline binary remains intact at its original path. The failing phase candidate is retained for diagnosis and is not serving. The transient service's original recovery startup command is unchanged; this reports the live loaded state, not a permanent boot-service migration.
