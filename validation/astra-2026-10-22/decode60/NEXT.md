# Current work toward 60 TPS

The retained default is v97:25.146432488TPS,492 phases, maximum pairwise
assigned-work ratio1.076923077. Its128 short-reference outputs,288 retained
state hashes, full head audit and4096-token baseline fixture pass.60TPS has
not been achieved. Physical GPUs2 and7 remain reserved for Nate.

Two larger matrix-execution alternatives are now complete and preserved in
normal build targets. `die_balanced_split.cubin` uses eight lanes per routed
down row (v98,24.992749323TPS). `die_balanced_lut.cubin` uses exact activation
pair lookup tables (v100,24.804969555TPS). The first LUT version, v99, retained
unused register activation loads and measured24.464268428TPS. Removing those
loads recovered some time, but neither alternative beat the retained default.
All three match128 reference outputs and288 state hashes. See README.md for
canonical DB jobs and implementation details. These are individual observations,
not randomized estimates of small performance differences.

The distributed-quantization experiment remains complete at commit29c92ec on
`perf/distributed-quant-20261022`. Its best run was23.635491116TPS, slower than
local preparation; it is not an unimplemented next step.

Objective weights: throughput1.0, fidelity1.0, phase/work constraints1.0,
engineering time0.8, cost0.8. The lower-effort lane split was expected to shorten
per-lane integer work and use the idle half of each routed-down CTA. It did
so structurally, but did not improve the request time. The larger lookup-table
alternative removed the multiplies from the row body; table preparation,
shared reads and added joins did not earn back their cost in these runs.

A remaining implementation direction is a genuinely looped Q2 body with
per-half-block activation loads, instead of the earlier compact-Q2 experiment's
large register-copy dispatch. The original five FP32 accumulation steps must
remain in order. Any implementation must include producer/shared-memory
lifetime, all tail warps, weight addressing and the real full-model path before
running. The lane and LUT results warn against treating a smaller instruction
body or more active warps as proof of better throughput.
