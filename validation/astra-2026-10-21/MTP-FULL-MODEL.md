# MTP plus the full production model, unchecked

This replaces the draft-only rollout experiment as the response to Nate's
request for **MTP + the full production kernel**. The target is the original
production `iq4-bias/kern/die.cubin`, FNV `9abf8aafb9134d59`. Every generated
position executes all 48 target layers and the target output head. The MTP
block supplies speculative inputs to two-position target batches.

GPUQ job `f4238ff2-e414-406b-9c2d-bec5bd817a55` completed successfully, exit 0,
in the canonical database. [Captured output](mtp-full-model-v1.json) records
the runtime configuration, emitted IDs, timing, and operation counts. No
output comparison or acceptance decision was performed.

## Architecture and execution

- The target retains three GDN layers followed by one QSA layer per four-layer
  group, across all 48 layers. Its existing widened residual / gated read and
  write operations are retained.
- Layer 2's n-gram embedding and host-prefetched PLE rows remain in the target
  program. During generation the host stages rows for the actual proposed
  token sequence consumed by the two-position batch.
- MTP consumes the prior target hidden state and the next input token, then
  predicts the second input to a two-position target batch.
- The production target executes both positions, with separate activation and
  exchange buffers. Both target-output tokens are streamed unconditionally.
  There is no proposal/target comparison, acceptance gate, rejection rewind,
  target-only fallback, or frozen-output reference.
- The draft cache is advanced for the second consumed input using the first
  position's target hidden state. This step **reuses the MTP QSA selected
  indices**: IDXSCORE/IDXSEL are omitted, while IDXQK still updates pooled keys
  and the ring. The next batch starts from the second target hidden state.

`FLASH_MTP_AUDIT=3` selects this full-model mode. Mode 2 is the separately
labelled draft-only experiment; mode 1 is shadow. Neither is a fallback for
mode 3. A final single-position tail uses the same full target, without a
second proposal. No new target SASS module was substituted for production.

The new work is host scheduling and MTP descriptor integration, not a claim
that target and draft have been fused into one persistent GPU launch. Target
and MTP retain their own state/controllers and kernel launches.

## Result

64 prompt tokens, 128 generated tokens, six M60 dies, all 48 target layers,
B512. One warmup plus three measured requests:

| Run | Combined MTP + target TPS | Post-first-token duration |
| --- | ---: | ---: |
| Warmup | 43.2148 | 2938.8070 ms |
| Measured 1 | 43.4009 | 2926.2088 ms |
| Measured 2 | 43.6589 | 2908.9163 ms |
| Measured 3 | 43.8437 | 2896.6566 ms |

**Median combined throughput: 43.66 TPS.** Warmed time to first token was
approximately 1.777 seconds; prompt/first-token throughput was about 36.01
positions/s. These are host-clock end-to-end timings including drafting,
target computation, staging, and hidden-state transfers, not draft-head timing.

Each request recorded **128 full-target generated positions in 64 batches**,
**64 MTP proposals**, **191 total draft steps**, and **64 QSA-index reuse steps**.
The execution statistics report comparisons, acceptance, recovery, and fallback
as zero. The benchmark did not inspect token agreement or output quality.
This is not a correctness-qualified 60/140-TPS result.

The earlier 269.24 TPS number measured MTP-only rollout and does not describe
this combined full-model path.

## Live state and reproduction

The runner used `BENCH_KEEP_LOADED=1`; the combined mode **remained loaded** in
the live engine after all four requests. A subsequent control read showed
`mtp_mode=3`, `mtp_rollout=0`, `two_position=1`, `layers=48/48`, `dies=6`,
the original production target cubin, 16,384 context capacity, and one image
row. This experimental mode currently supports greedy text without snapshots.

```sh
ssh archy gpuq run --name mtp-full48-production-pair-unchecked-v1 -- env \
  BENCH_MTP_AUDIT=3 BENCH_CANDIDATE_ONLY=1 BENCH_KEEP_LOADED=1 \
  BENCH_PROMPT_COPIES=1 BENCH_GENERATE=128 BENCH_SOCKET_TIMEOUT=360 \
  BENCH_EXPECT_RESTORE_REV=attention-q-release \
  /home/nate/work/flash-mtp-full-20261021-v1/resident_decode_bench \
  --resident /tmp/flash.sock \
  /home/nate/work/flash-mtp-full-20261021-v1/libengine_mtp.so \
  /home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin \
  /dev/shm/flash-sass-kernel-artifacts/die_mtp-v5.cubin \
  /home/nate/codex/flash-astra-build-20261018/qsplit-release/kern/libengine.so \
  /home/nate/codex/flash-astra-build-20261018/iq4-bias/kern/die.cubin
```

That exact command describes the admission configuration used for this run.
Because mode 3 is now loaded, its original-revision admission guard will reject
an immediate identical rerun. Do not interpret that guard as an inference
correctness check or add a new isolated database to run it again.
