# Current work toward60 TPS

The retained default is v104:25.721831473 and25.792883107TPS on two requests,
492 phases and maximum pairwise assigned-work ratio1.076923077. Both requests
match128 reference outputs and288 retained state hashes each; full head and
work audits pass. The4096-token fixture also passes.60TPS is not achieved.
Physical GPUs2 and7 remain reserved for Nate.

The latest retained change batches independent predicated loads and stores,
then tightens issue spacing inside long bursts. It preserves all final waits,
original wait bits, memory fences and instruction order. The compiled image
has125 bursts, including36 predicated and29 store bursts;856 intermediate
waits and5,165 static scheduled cycles are removed. Those are compile-time
counts, not measured cycles saved per token. v103 and v104 have essentially
similar measured request times; no isolated store-batching gain is established.

The true streamed Q2 down body is now complete in `die_balanced_stream.cubin`.
It uses two alternating register banks and two arithmetic bodies, pipelines
all five half-blocks without register-copy dispatch, and keeps the original
FP32 association. v101 matched all outputs/state hashes but measured24.664605312
TPS. It remains opt-in. The lane-split and LUT alternatives are also complete,
with historical results in README.md. They are not pending implementation.

The distributed-quantization experiment remains at commit29c92ec on
`perf/distributed-quant-20261022`; its best full-model run was23.635491116TPS.

Objective weights remain throughput1.0, fidelity1.0, phase/work constraints1.0,
engineering time0.8 and cost0.8. The larger streamed-body alternative was
expected to improve overlap and instruction locality; it was correct but
slower. The lower-effort shared memory-scheduling pass produced a measurable
request-time improvement and was retained after repeated-request and longer
context checks. Measurements are separate observations, not randomized A/B
estimates of small differences.

Current per-operation evidence is in `memory-bursts/operations.md` and its
per-die CSV, extracted directly from DB jobs. GR read/update intervals fell
by about1ms on the final token; hcupq_x remains the largest inclusive family
at about6ms, followed by routed gate/up and down. These phase intervals include
synchronization and are not isolated body timings. Remaining diagnostic waits
are concentrated in GDN, GR and QSA bodies, often around non-adjacent accesses.
Any further scheduling transformation must account for physical aliases,
result dependencies, register-read lifetimes and scoreboard reuse across
pipelines. Do not infer safety merely from different symbolic register names,
or count elapsed wait time as pure link latency.
