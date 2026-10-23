# Implementation checkpoint: quantized DFlash2 and retained B4 target states

Branch: `perf/decode120`. Main target GGUF is unchanged. This checkpoint is
source work, not a measured throughput result. No new GPU job, benchmark,
compilation or test was run while writing this implementation. The complete
120-TPS architecture is still in progress; this branch is not deployment-ready.

The architecture guide is `NEXT.md`, with the conditional B4 cost ledger in
`validation/astra-2026-10-23/throughput-model/dflash-b4.lua`. Its 24 ms/round and
mean committed length of three imply 125 TPS arithmetically. Neither input is
established for this implementation.

## Selected approach

Objective weights: same target checkpoint 1.0; committed decode throughput 1.0;
lossless greedy acceptance/state continuation 1.0; engineering time 0.8; extra
resource cost 0.5. Selected: resident quantized DFlash2, true B4 target matrix
reuse, retained prefix banks and unique/topological target placement.

Lower effort alternative: quantized proposer with scalar target replays and
snapshot/recovery. Likely outcome: a functioning speculative loop whose target
weight reads and replay costs do not earn the guide's budget. Throughput
importance 1.0, time importance 0.8, cost importance 0.5.

Higher effort alternative: finish the entire native B4 operator family, paired
MoE columns and board-hierarchical payload graph before measuring. Likely outcome:
the guide's bandwidth savings become implementable, while realized throughput
still depends on packed-kernel efficiency and acceptance. Throughput importance
1.0, time importance 0.8, cost importance 0.5. This is the selected path; no more
intermediate benchmark loop.

## Written, not built or run

- Pinned `incoai/Qwen3.8-Flash-Next-DFlash2-GGUF` Q4_K_M loader: 45 Q4_K,
  four Q6_K and 32 F32 tensors. Main matrix storage remains quantized. Metadata
  and published file identity are recorded in `kern/dflash/model-q4km.json`.
  Full files were downloaded on Engi and Archy; the recorded hash is published
  upstream, not claimed to be locally checked.
- Resident GPU draft graph: five layers, two-tap grouped convolutions,
  normalization, RoPE, noncausal B4 attention, 2048-position conditioning cache,
  MLPs and greedy conditional candidate selection. Candidate logits use all
  three positions in one pass over each resident target-head shard. Ordinary
  activations and cache state stay on the GPU.
- Full-target feature capture after the five completed layers named by the
  checkpoint (5, 15, 25, 35, 45 for this file). Each feature is the mean of four
  residual branches. A 2048-position feature ring resides on the draft owner.
- `FLASH_BALANCED_SPECULATE=dflash2` controller with one anchor plus three
  proposals, target acceptance, replacement/bonus token, stop handling, and
  append of only retained consumed-input features. Draft self-attention KV is
  never appended as target conditioning. The existing prompt-lookup mode remains.
- Four banks for GDN, convolution, PLE and indexer-ring state. Position zero
  reads the selected bank through a device table; subsequent positions read
  the previous prefix. GDN writes directly to its output bank. Ring operators
  copy/update their own channels inside their existing operation. Rejection
  selects a bank with one control-word update per die and does not replay the
  accepted prefix. Older recurrent-only cross-request snapshots do not contain
  draft features, so this mode rebuilds prompt conditioning.
- Native four-position GR-up using one weight stream and four accumulation
  chains. The five-position mode is retained through a generator parameter.
- Native B4 GDN precomputes all four q/k/value/gate records before loading
  state, avoiding the scalar phase-A register aliases. It loads each state tile
  once, performs four ordered updates, and stores all four prefix states.
- Immutable target programs are cached on device per request/width. Ordinary
  rounds no longer transfer every phase descriptor again. Width changes still
  update the control word, and profile metadata follows the selected program.
- Complete-round timing and retained-length histogram accounting, including
  draft head, target execution, PLE, acceptance, cache append and controller.
  `spec_round` records go to the job's captured stderr; no second result DB.
- Unique scalar placement: router rows and GDN Q/K rows have unique owners;
  global top-ten routing is formed from six small candidate packets. QSA K/V
  projection rows are also read once. These new layouts have distinct ABIs.
  The current collective change aggregates completion flags only.

## Work still required to earn the budget

1. Native B4 target matrix coverage beyond GR-up. The remaining generic phase
   wrappers still execute scalar matrix bodies separately. The target is not
   yet entitled to the 8.243 ms modeled B4 matrix service.
2. Pair MoE gate/up row shards with compact down-column shards; eliminate the
   exchanged expert intermediate. Current unique placement still uses the
   older row-sharded MoE intermediate exchange. Implement the ten 64-channel
   blocks and count capacity/imbalance, including the fenced die.
3. Board-pair tiled payload transport and MoE reduce/scatter. Aggregating flags
   is not the guide's hierarchical payload implementation. Current physical
   placement remains 0,1,6,3,4,5; it is not automatically the proposed board-pair
   topology. Preserve unrelated allocations and the existing memory fence.
4. Fold target-feature capture into suitable residual producers and complete
   native B4 convolution work. Separate feature capture currently interrupts
   five write/read fusions. Ring-bank copying is extra state traffic, although
   recovery no longer transfers or recomputes a complete recurrent prefix.
5. Finish bulk proposed-token PLE staging/overlap and the full scalar/B4
   arithmetic/transport integration. Then assemble and run the complete engine
   end to end through Archy's GPU queue and canonical database.

The older dependency-scheduler prototypes are retained as research. The
memory-scheduling prototype produced incorrect output and is not used by this
image. None of its diagnostic timings is a valid speed claim for this work.
