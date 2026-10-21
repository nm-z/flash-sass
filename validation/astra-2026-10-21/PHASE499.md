# Production decode: 499 phases on every die

[Detailed merged-operation timing and block participation](phase499/MERGED-OPERATIONS.md)
and [complete side-by-side PROF_ALL tables](phase499/operations.md) were extracted
from the existing canonical GPUQ records without another GPU run. The report
distinguishes measured opcode totals, unresolved timing subsets, and source-derived
block participation; the saved records do not contain per-operation SM occupancy.

The complete production decode kernel executes **499 logical phases per token on
all six dies**. This includes all 48 layers, the output head and ordinary token
selection. MTP and FP8 residual storage are disabled. The result is verified by
runtime profiles, not just the offline schedule compiler.

**This is not a throughput improvement.** In the matched 64-prompt/128-output
comparison, decode regressed from **40.12 TPS to 32.98 TPS**. Phase count fell
from 1,178–1,230 to 499, but the current fused operator implementations cost more
per phase. The throughput regression has not been isolated or optimized away.

## Matched results

Both arms use the same resident weights, six SM52 dies, split
`112,97,81,116,97,97`, a 32,768-position context allocation, B512 attention,
scalar prompt processing, and the same inputs and sampling seed. Each row is a
single matched request, not a warmed median. Decode rates use post-first-token
elapsed time and the number of decode intervals.

| Workload | Baseline decode TPS | Candidate decode TPS | Output comparison | Candidate phases, dies 0–5 |
|---|---:|---:|---|---|
| 64 prompt, 128 greedy output tokens | 40.1152 | 32.9848 | All 128 tokens match | 499, 499, 499, 499, 499, 499 |
| 4,096 prompt, 16 greedy output tokens | 34.8074 | 29.7441 | All 16 tokens match | 499, 499, 499, 499, 499, 499 |
| 64 prompt, 16 sampled output tokens | 40.0118 | 32.9288 | All 16 tokens match | 499, 499, 499, 499, 499, 499 |

The long prompt crosses the QSA selection threshold and exercises radix selection
and compaction. Sampling uses seed 7, temperature 1, top-k 20, top-p 0.95. The
first successful unreferenced 16-token request also matched the previously
stored production baseline when compared afterward.

The matched prompt rates were 41.55 versus 33.99 tokens/s at 64 tokens, and
37.84 versus 31.45 tokens/s at 4,096 tokens. Prompt-processing acceleration is not
claimed. These checks cover the specified greedy and seeded-sampling workloads;
they are not exhaustive validation of every context length or cancellation mode.

## Follow-up: warmed repeatability

After testing was explicitly authorized, job
`03bff6fb-db1f-4844-9d91-0e2e381ea17f` ran four matched requests per arm with
64 prompt tokens and 128 output tokens. Run 0 was warm-up; runs 1–3 determine
the medians below. The kernel and its configuration were unchanged.

| Measurement | Baseline | 499-phase candidate |
|---|---:|---:|
| Warm decode rates, TPS | 40.1136, 40.1113, 40.1059 | 32.9841, 32.9853, 32.9863 |
| Median decode TPS | **40.1113** | **32.9853** |
| Median prompt tokens/s | 41.5298 | 34.0033 |
| Final runtime phases per die | 1,178–1,230 | 499 on all six |
| Output mismatches | 0 | 0 across all 512 candidate tokens |

The candidate is about 17.8% slower in this warm comparison. All eight requests
completed without a device error, and restoration reported an identical
configuration. This follow-up addresses warm-state repeatability and the
steady-throughput question; the previously passing long-context and sampling
cases were reused rather than run again without a new concern.

The final-token profiles attribute the following time to `hcwsum` plus
`bf16hcrx` in baseline, and their fused `grwrite_read` plus remaining unfused
operations in the candidate:

| Die | Baseline GR group, ms/token | Candidate GR group, ms/token | Increase, ms/token |
|---|---:|---:|---:|
| 0 | 3.4365 | 5.5961 | 2.1596 |
| 1 | 3.5102 | 5.6658 | 2.1556 |
| 2 | 3.8359 | 6.0273 | 2.1914 |
| 3 | 3.4990 | 5.7088 | 2.2098 |
| 4 | 3.5922 | 5.9137 | 2.3215 |
| 5 | 3.3741 | 5.6944 | 2.3203 |

These are inclusive phase times: synchronization waits and ownership changes
affect attribution. They identify a substantial cost increase, not a causal
isolation of the arithmetic itself. The fused GR emitter expands six-peer sums
at 30 RMS-write load sites, making repeated loads/reductions a concrete next
optimization target. No optimization or ablation is claimed from this test.

[Raw warmed-run output](phase499/03bff6fb-db1f-4844-9d91-0e2e381ea17f.json)
and [job metadata](phase499/warmed-job.json) are retained alongside the original
evidence. All data remains in the same canonical GPUQ database.

## Production changes

- Quantize GR and GDN activations in their producing operations and publish
  codes, sums and scales directly to receiving dies.
- Combine residual write/sum with the following normalized GR down read.
- Group independent matrix projections, including mixed quantization formats
  and the independent BF16 alpha/beta projection.
- Keep GDN convolution, recurrent-state update, normalization, gating and
  quantized publication in one head-owning CTA operation.
- Combine router-logit projection, routing and expert selection. Only the
  consuming warp joins the producer-ready signals.
- Fuse routed gate/up/activation work with independent shared-expert projections
  and the shared activation consumer. Routed/shared down accumulations preserve
  their original CTA row ownership and addition order.
- Keep four Q6 column slices in one CTA-local dependent chain.
- Fuse QSA score production with selection, and attention partitions with gated
  merge and quantized publication. Merge heads join their four producer CTAs.
- Run radix selection in one CTA with the original threshold and tie-order
  arithmetic; eliminate the now-unneeded inter-CTA counting pass.
- Fuse PLE gather/quantization and gate/normalization; combine independent PLE
  normalizations and publish the value projection directly.
- Publish token embeddings directly into each die's residual buffer and publish
  argmax/top-k results from the final reducer.
- Distribute the twelve QSA layers as two per die. Original holder placement
  remains unchanged; needed resident K/V/indexer/norm tensors are copied to
  their runtime owners. Query shards still read from the original holder source.

The original holder attention map remains
`1,1,1,1,4,4,4,4,5,5,5,5`. The runtime map is
`0,1,2,3,4,5,0,1,2,3,4,5`. `FLASH_ATTN_BALANCE=0` disables redistribution;
otherwise the phase-fusion candidate enables it by default.

A phase is a logical operation bounded by the persistent framework's phase
boundary. Auxiliary descriptors do not count as independent phases. Dependent
Q6/shared-down chains use CTA-local fences, not hidden grid barriers. The offline
compiler separately counts inner grid-barrier requests; the candidate has zero.
Producer/consumer readiness joins and CTA-local barriers still have a real cost.

## Evidence and artifacts

All jobs are in the single canonical database on archy:
`/home/nate/.local/state/gpuq/history.sqlite3`. No isolated database was created.
Raw job output is exported in [phase499/](phase499/).

| Job | Purpose | Outcome |
|---|---|---|
| `dc62bc91-b446-462e-a7dc-4f76c29c8229` | First complete v6 run | Failed before completing a prompt position; illegal access reported on die 3 |
| `c76417cc-0f30-4a9a-b5d4-b726f907f0f0` | Uniform-publication v7 | Reached first QSA layer; watchdog located stall in score/selection |
| `fd0aa78e-6532-434a-8d57-ad708b45569b` | Faulted-core restart | Completed through GPUQ; persistent weight holders retained |
| `1564d463-115c-4fe9-a648-6b4304260573` | First successful v8 full-model run | 16 tokens, six runtime profiles at 499 phases |
| `4a49c988-40f2-4074-af34-823fe636f752` | Matched 128-token greedy comparison | Zero mismatches; configuration restored identically |
| `09a0a0d6-97d1-4f67-9957-5fe22b29c1fb` | Matched 4,096-token prompt comparison | Zero mismatches; configuration restored identically |
| `a3747079-01be-4b69-a1f8-99a791c1a28f` | Matched seeded-sampling comparison | Zero mismatches; configuration restored identically |
| `76243b50-8ed5-4746-9095-b678dee86824` | Service diagnostic journal | Captures the failed runs and restart diagnostics |

The two execution fixes preserved the complete 499-phase design: publication
loops now remain warp-uniform with predicated memory operations, and the QSA
selection exit path includes the required dependent-predicate latency gap.
No kernel execution occurred between the user's prohibition and construction of
the complete sub-500 schedule. Subsequent runs followed specific changes or
covered previously unexercised output/selection paths.

Measured live artifacts are under
`/home/nate/work/flash-phase499-20261021-v8/` on archy:

- `libengine_exact.so`, SHA-256
  `ff204888d1a339ffe28d8e04c142c60f74ff7ea80fbecb25cd956b978ebbb636`
- `die_phase.cubin`, SHA-256
  `a6481dc005498f1d162853b1d6d1504e32151b06a97ba86c0c61a21ff8605e49`
- Runtime cubin FNV `9bb60fd39ab64f28`; phase ABI v6; 249 registers.

The final compared configuration remains loaded. The systemd recovery unit's
original startup command is retained as its existing recovery configuration;
this work does not claim a permanent boot-time service migration.

## Reproduce the schedule without a GPU

`make -C kern phase_schedule` builds a descriptor compiler with no CUDA linkage
(CUDA headers are needed). Run it with a directory containing the six holder
metadata manifests. It calls the production `model_program()` with symbolic
addresses and real weight formats. It does not execute model arithmetic.

`FLASH_SCHEDULE_DESCRIPTORS=1` includes every logical phase's opcode, descriptor
span, flags and layer. GPU execution is additionally guarded against a
phase-fusion candidate with 500 or more phases on any die.

The branch also retains earlier MTP, DFlash and FP8 experimental source. None is
enabled in these measurements. Earlier unchecked MTP results do not establish
correctness or throughput for this no-MTP path. Refs #54, #53 and #82 remain open;
the measured throughput regression prevents claiming their speed goals met.
