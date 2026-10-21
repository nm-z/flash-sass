The prompt is the 64-token fixture embedded in `kern/resident_decode_bench.c`.
`fixture64-greedy128.ids` freezes the 128 output IDs from scalar arm run 0 of
GPUQ job `52e7b787-5332-474f-bf25-7aa1003d2d2b`, task database
`/home/nate/work/flash-140-1200/gpuq.sqlite3` on archy. All four scalar streams
matched in that run. The overall A/B job failed because the batch candidate
differed; this is a reference for that specific workload, not a passing report
for the entire job or the model's other request modes.

The run used all 48 layers, six M60 dies, B512, 16384 capacity, no prefix reuse,
greedy sampling (`sample 7 0 20 0.95`), the resident whole-expert holders and
local PLE table. The scalar kernel was `die-base-v4.cubin`, with native chunk,
paired execution, fused MoE and fast exchange disabled. The benchmark log
retains the complete configuration and input identities.

The file is only an output oracle. The benchmark must never supply these IDs
to generation, pre-stage their PLE rows, or count them as computed tokens.
Comparing a new run to this file does not constitute a matched throughput A/B.
