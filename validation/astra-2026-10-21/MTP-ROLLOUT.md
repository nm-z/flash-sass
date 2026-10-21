# Unchecked recursive MTP rollout

At Nate's explicit request, this run performs the MTP operation loop without
target verification, output comparison, acceptance, rejection recovery, or a
generation fallback. It is a different experiment from verified speculative
decoding and from the earlier shadow run.

GPUQ job: `7dae18b0-9d5b-4962-91ee-9b9b2a55e506`, recorded in the canonical
`/home/nate/.local/state/gpuq/history.sqlite3` on archy. Status `succeeded`, exit
0. [Captured output](mtp-rollout-v1.json) includes tokens and timing records.
No output-quality or token-equivalence check was performed.

## Operation

`FLASH_MTP_AUDIT=2` selects the unchecked rollout. Mode 1 retains the earlier
shadow implementation, but it is not a fallback for mode 2.

1. Reset the draft state. For the first 63 positions of the 64-token prompt,
   advance the MTP block headlessly and execute the full target's headless
   position. Carry that target hidden state into the next prompt position.
2. At the last prompt token, run the MTP block and its output head. Emit its
   predicted token directly.
3. Copy the MTP block's hidden output into its next hidden input. Gather the
   embedding of its own predicted token and run the next MTP step.
4. Repeat until the requested output length or an explicit stop/cancellation.

The recurrent loop uses the actual checkpoint's embedding/hidden normalization,
Q8 EH projection, HC, attention/indexer/cache, routed/shared MoE, and output
head. It executes on the six model dies. No target forward pass runs during
generated-token rollout. The target remains resident for prompt initialization.
The inference path does not consult a reference token stream or compute an
acceptance decision. Missing draft capacity fails initialization rather than
silently switching to target generation.

## Measured result

Four runs, 64 prompt tokens and 128 directly emitted MTP tokens each. The first
run is warmup; the reported median uses the other three. Post-first-token TPS
uses 127 intervals and host `CLOCK_MONOTONIC`, including the loop's host work.

| Run | MTP rollout TPS | Post-first-token duration |
| --- | ---: | ---: |
| Warmup | 270.8334 | 468.9229 ms |
| Measured 1 | 269.1010 | 471.9418 ms |
| Measured 2 | 269.2430 | 471.6929 ms |
| Measured 3 | 269.3708 | 471.4690 ms |

**Median: 269.24 MTP rollout tokens/s.** Warmed time to first token was about
1.740 seconds; the prompt/first-token metric was approximately 36.77 positions/s.
Each request reported 191 total draft steps: 63 prompt steps plus 128 rollout
steps. The legacy `one_step_matches=0` field is not a measured agreement rate in
this mode: comparison was disabled. `correctness_checked=false` and
`target_verified=0` explicitly label these results.

This measures recursive MTP generation, not equivalent 48-layer target decode.
No conclusion about output quality, agreement, or the original verified
140-decode/1,200-PP objective follows from this rate. The shared service was
restored after the measurement; no algorithmic fallback occurred in the loop.

## Reproduction

The build uses the existing `libengine_mtp.so` and `resident_decode_bench`
targets. This test reused `die_mtp-v5.cubin` and the production schedule-v3
target module from the earlier implementation. No new kernel assembly or
correctness test was run as a prerequisite for this measurement.

```sh
ssh archy gpuq run --name mtp-unchecked-rollout-128-v1 -- env \
  BENCH_MTP_AUDIT=2 BENCH_CANDIDATE_ONLY=1 BENCH_PROMPT_COPIES=1 \
  BENCH_GENERATE=128 BENCH_SOCKET_TIMEOUT=360 \
  BENCH_EXPECT_RESTORE_REV=attention-q-release \
  /home/nate/work/flash-mtp-rollout-20261021-v1/resident_decode_bench \
  --resident /tmp/flash.sock \
  /home/nate/work/flash-mtp-rollout-20261021-v1/libengine_mtp.so \
  /dev/shm/flash-sass-kernel-artifacts/die-production-schedule-v3.cubin \
  /dev/shm/flash-sass-kernel-artifacts/die_mtp-v5.cubin \
  /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so \
  /home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin
```

Measured library SHA256:
`a2601e0a4da16f331d71fce7fe40007a1709e3f13cf57dd4a484ea77c43177ab`.
Measured runner SHA256:
`d6072217b90f76bcfd64ca1f644d04cfd7d75be13d15a4dc2d2b277db5368fa0`.

The benchmark intentionally does not compare output IDs, run the target during
generation, check completion-generation invariants, or fall back to shadow or
target decoding in this mode. CUDA API failures still abort execution.
