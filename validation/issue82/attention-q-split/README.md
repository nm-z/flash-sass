# Distributed attention query projection

Refs #82. This improves the existing resident six-die model; the 120–144 tokens/s goal remains unmet.

## Implementation

The original attention owner computes all 12,288 Q/gate rows before attention. This change copies row slices of those resident weights into the other workers' available memory during engine initialization. It uses the original scalar/paired GEMV bodies and the original six-destination exchange. The attention owner computes K/V/indexer concurrently with the other dies' Q GEMVs. The gathered Q/gates occupy the opposite inbox slot from Z and are consumed before the next reuse of that slot.

The memory plan includes full context capacity, image capacity and paired activation scratch. Workers retain an 8 MiB reserve; a worker that cannot hold all its required query slices is excluded and the plan is recomputed. Weight slices are owned by the engine's normal allocation records and freed on reload. No additional full model is loaded. No weights are requantized.

## Measurements

The short-prompt reference/candidate/reference comparison used the same IQ4 kernel from PR89 and the same host library, with the query split disabled/enabled. The experimental selector 3 was the original-broadcast path now exposed as FLASH_ATTN_Q_SPLIT=1; its code is preserved in e0618e9. The release removes the targeted-send experiment entirely.

- Fixed 64-token prompt: candidate 40.074/40.061 tokens/s; references approximately 37.95–37.97 tokens/s. Approximately 5.6% on this workload.
- First-token time 1,193.1 ms versus approximately 1,275 ms.
- Both repetitions of every short-prompt arm produced identical output IDs.
- All 48 layers, 512 experts, context 230000, image capacity 4096, paired passes/chunk 64, local durable PLE file, and the same six resident dies. Worker mask 0x37 selects dies 0, 1, 2, 4, 5.

These fixed-prompt results do not establish120–144tok/s or a corresponding gain for every chat. A long-prompt comparison and live HTTP results are recorded separately below.

## Rejected experiment

A targeted-send variant that sent Q only to the owner produced an inconsistent response. Adding a per-writer system fence then produced a device fault. Its isolated copy fixture passed 70 cases, which was insufficient to establish whole-model correctness. Neither its SASS changes nor its capability marker is included in this release. The original broadcast kernel remains in use.

## Release comparison

The exact release source e037162 was built as attention-q-release. With a 2,048-token prompt and 129 generated outputs, the disabled/enabled comparison produced identical output IDs. First-token time fell from 43,278.4 to 41,060.4 ms (about 5.4% higher prompt throughput); post-first decode rose from 33.823 to 34.715 tokens/s. This was one complete run per arm, used to check the longer sequence and QSA cutoff, not a repeated statistical estimate. Both arms kept full serving capacity and paired prompt processing. Raw results are in long-prompt.jsonl.
