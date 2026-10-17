# Gather benchmark contract

`kern/qsa_gather_bench` measures the production `OP_KVGATHER` through the persistent dispatcher. It uploads a deterministic, finite, nonzero INT8 cache and FP16 scales, independently constructs the expected compact FP32 K/V buffers with the C reference, and compares every physically gathered value before accepting timings. The K/V patterns depend on source position and channel, so changing selected blocks changes verified output signatures.

```sh
make -C kern qsa_gather_bench die.cubin
cd kern
CUDA_VISIBLE_DEVICES=GPU-... perl run_qsa_gather_pair.pl --out /tmp/gather-pair --nv 2000 --iterations 32 --matrix
```

Run on an exclusively reserved idle GPU. The binary accepts `nv tail iterations pattern host csv [seed]`, where `nv` is complete blocks, tail is 0..3, host is exactly 0 or 1, and budget is 512. Logical visible length is `4*nv+tail`. The position is fixed within a run; `changing` means a different selection on each launch at that position, not model-generated selections or sequential decoding. The matrix runs all tails independently. `clustered`, `even`, `random` and `changing` are synthetic. `captured` accepts one or one-per-iteration sorted, unique little-endian uint32 ID row through `GATHER_IDS`; the paired runner's captured-format case explicitly labels its IDs as generated fixtures.

Both placements receive the same cache, selected IDs, position, launch configuration and seed. The runner compares input hashes and every per-iteration gathered-output signature. It records binary/source/cubin hashes, GPU inventory, commands, raw CUDA event timings and nearest-rank percentiles. Fixture allocation, upload, selection construction, verification/readback and the selection upload before each timed launch are outside the CUDA event interval. Ten warm-up launches precede measurement. The interval includes the persistent kernel framework and mapped host-memory accesses made by gather.

The Perl runner uses core modules and the shared `QsaEvidence.pm` process helper. Its captured-format fixtures use the explicitly named `lcg32-1664525-1013904223-partial-fisher-yates-v1` generator. These bytes differ from earlier runner fixtures; both arms still consume identical sorted, unique IDs. The new runner was exercised on all 40 matrix pairs with eight timed launches per arm. Raw results and identities are in `validation/issue16/perl-matrix/`.

The reported source/destination byte counts are **unique-address footprints**, not measured PCIe or DRAM traffic. Transactions, cache reuse, coalescing, replay and stalls require a separate Maxwell-compatible profiler run; this gate remains `not_run` until attached. Do not add a second transfer duration to gather time for the same mapped-memory reads. Running host then VRAM in one order is a correctness pairing, not a fully controlled causal performance experiment; randomize/reverse arm order and repeat on thermally stable, idle hardware before making a placement-speed claim.

## Legacy compatibility

`QSA_HOSTKV=0` now means VRAM, `1` means pinned host memory, and other values are errors. Previously any set value selected host memory. Legacy `qsa_test gathertime` remains available but labels its zero-filled, unverified repeated-layout fixture explicitly; it reports the actual 51-position key-count range instead of the false fixed 2,051-key label. It rejects `nv<512`, preventing the duplicate-zero selection produced by its historical `nv/512` construction when crossing the cap. Use the verified benchmark for new results.

This benchmark characterizes gather. It does not establish the cause of historical whole-model decode throughput changes or measure prefill, the indexer, attention, INT8 conversion, host polling, or HTTP overhead.
