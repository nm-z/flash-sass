# Current work toward 60 TPS

The complete distributed-quantization experiment is implemented and preserved
at commit29c92ec on `perf/distributed-quant-20261022`. Its best full-model run
was23.635491116TPS, slower than the ordinary local preparation path. See the
linked report in README.md; this is no longer pending implementation.

The production build now includes routed-expert weight prefetch. v96 measured
25.023809159TPS; v97 removes redundant pointer construction after the first
expert and tightens integer-address scheduling, measuring25.146432488TPS.
Both match128 reference outputs and all288 retained state hashes. Both keep
492 phases, with maximum pairwise assigned-work ratio1.076923077. The v97
4096-token fixture also passes with both output tokens matching the retained
baseline; its two output tokens are not a decode-throughput benchmark.

Objective weights remain throughput1.0, fidelity1.0, phase/work constraints1.0,
engineering time0.8 and cost0.8. The lower-effort approach tightens and overlaps
work within the existing matrix geometry: it is complete here, and its measured
benefit is small. The higher-effort alternative changes the Q2 row/lane
geometry and activation/weight representation together, reducing the serial
integer dot-product work each active lane performs. This requires complete
producer/consumer coverage and preservation of half-block accumulation order;
weight padding must not be confused with useful balanced work.

The next substantial implementation should address matrix execution itself.
Current routed-down rows use four lanes and five serial half-block bodies per
lane. An eight-lane geometry can divide each original32-value half-block
between a pair of lanes, summing their integer dot products before conversion
to FP32. Keeping the same five half-blocks and the same four-way FP32 reduction
would preserve the original accumulation association. Each thread would hold
40 activation words instead of80; one shuffle/add per half-block replaces
half its integer multiplication work. Activation and weight addressing,
prefetch, tail guards and retained-work accounting must change together.
The weight representation itself can stay unchanged. Gate/up needs its own
geometry analysis. This is an unimplemented
candidate, not a claimed speedup, and must be completed for the full kernel
before a hardware run. Do not substitute a partial-layer benchmark.
