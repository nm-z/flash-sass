# Issue #7 validation at a140799

Source revision: `a1407995f4d273e1d2527c0f5fb3998d09252dc2`.

## Status

CPU regressions and the native build pass. Whole-model sequence acceptance is **blocked**, not passed. The model launcher waited 30 seconds for `/tmp/flash-sass-engine-validation.lock` and exited **75** before starting a private core.

At **2026-10-18 03:08:15 UTC**, the lock was held by **Ruby PID 4010221**. Dies 0–5 each reported **100% utilization**; die 3 had **28 MiB free**. The current holder/core inventory and raw measurements are in `archy-a140799/`.

## Completed checks

| Host | Check | Result |
| --- | --- | --- |
| engi | FIFO GPU-turn scheduler, eligible admission, bounded slots, cancellation, reuse and fault propagation | Passed |
| engi | Prefill planner boundaries, snapshot, cancellation and overflow | Passed |
| engi | Seven acceptance-predicate fixtures, including rejected serialization and ignored/unbounded cancellation | Passed |
| engi | HTTP client disconnect with unread engine tokens still buffered | Passed |
| engi | Same HTTP test with disconnect detection deliberately removed | Compiled successfully; expected assertion failure, exit 1 |
| archy | Fresh native build of libengine.so, flashcore, flashd and die.cubin | Exit 0 |
| archy | Scheduler test | Passed |
| archy | GGUF output-token checks, stream=0 and stream=1 | Passed |
| archy | Six-die model matrix, VRAM KV / quantum 1 / QSA budget 1 | Blocked at shared lease, exit 75; no private core started |

The build used the exact committed source from a git archive, system CUDA headers at `/opt/cuda/include`, CUDA 11.4 `ptxas`, and a private copy of the existing patched MaxAs checkout. Its revision, local patch and source hashes are retained. The build log is gzip-compressed without changing its contents.

The `engi-a140799/` directory contains the CPU runner output, predicate results, HTTP mutation patch, compiler command and expected failing assertion. The `archy-a140799/` directory contains the native build command/log, compiler and assembler versions, binary identities, model-launch environment, holder manifest hashes, lease failure and GPU/process inventory.

## Exact binary identities

| Artifact | SHA-256 |
| --- | --- |
| libengine.so | `78939bc2f58ad9dabbb45bcd86381b0744980f42c4a69a349e5452c1e6c9ac8e` |
| flashcore | `5ee9d367833843581d88ea8a928dbb7d92a3768823f6376d08f87a12ed1f4bfa` |
| flashd | `21ae4e76fed85c4bd8eca994485119bd3ce64bd1654f86e54d285a0ed9910cfd` |
| die.cubin | `116d828f20139d144d46d7d8115a28e35313dac1f6c24ba81c1b5c0984f50e14` |

## Reproduction and remaining gate

From a checkout of the source revision above, the CPU command is:

```sh
kern/sequence_cpu_check.sh NEW_OUTPUT_DIR
```

The real-model entry points and full placement example are in `SEQUENCES.md`. They require the same six weight holders and a free shared validation lease. The private source/build used for this attempt remains on archy at:

```text
/tmp/flash-issue7-a140799-9zQ4QV24
```

Before marking the PR ready, run the complete real-model checker in both documented configurations:

1. VRAM KV, quantum 1, QSA budget 1.
2. Host KV, quantum 8, QSA budget 512, headless prefill chunks of 4.

The checker must verify serial/interleaved greedy and seeded output parity, swapped slots, independent snapshots, active and queued cancellation, slot reuse, nonzero image/M-RoPE isolation, and a short request finishing after an observed long-prefill turn but before that long request's prefill boundary. The serial and cancellation mutants must fail their named model assertions; a resource, build or connection error does not count as a successful negative control.

The original `archy-c67d391/` records remain historical evidence at their recorded source revision. Their prior pass labels neither validate this revision nor establish the stronger causal progress/cancellation assertions. No new whole-model correctness or throughput result is claimed here.
