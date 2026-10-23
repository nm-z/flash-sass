# Implementation checkpoint: quantized DFlash2 and retained B4 target states

Branch: `perf/decode120`. Main target GGUF is unchanged. The combined host
library and scalar/B4 target images now build. The quantized draft device module
also builds. These are build results, not a measured decode speed. No model
inference has run with this combined implementation yet; 120 TPS remains unproven.

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

## Implemented in the combined build

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
- Native four-position GR-down now prepares all four residual inputs with the
  existing residual-write/RMS arithmetic, then runs one BF16 weight stream
  through four independent chains. A producer CTA releases the prepared
  vectors to the matrix CTAs. This removes repeated normalization across all
  matrix CTAs, adds one local readiness handoff, and preserves the original
  four-branch addition and warp reduction order.
- Native four-position dense Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS and ordinary
  BF16 projections, including the complete Q5_K output head and its existing
  winner reduction. The packed formats and per-result FP accumulation order
  are retained. Integer scaled dots stay within their exact integer range.
  These bodies loop over K groups instead of expanding every group inline.
- The B4 builder transposes independent mixed matrix bundles from position
  order into matrix order. QKV, gate and alpha/beta projections can each share
  their weights across four positions. Only the final member publishes the
  original peer epoch, so this does not add peer exchanges. Four dependent Q6
  output slices retain their original slice order and explicit local joins.
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
  The payload path now also uses board-pair forwarding, described below.
- Paired MoE cache format and production path: each die owns one or two
  contiguous 64-channel blocks, with rank two always light and the other light
  rank rotating per layer. Gate/up rows and down columns use the same selection.
  `native_weights paired SOURCE_CACHE NEW_CACHE` copies the original packed
  codes/scales. Q2 down columns use separate scale/code planes at exactly 18
  bytes per output row per 64-channel block. Shared Q4_0, Q5_0, Q8_0 and IQ4_NL
  columns retain their existing codes/scales; Q5 uses a separate high-bit plane
  so the low-code vector loads remain aligned without per-row padding.
- Scalar and B4 compact-column down operators now quantize only the local
  intermediate and evaluate the routed/shared down columns. B4 builds a private
  union of the four expert lists; overlapping experts reuse weights. Per-rank
  partials preserve each position's router-rank addition order. Completed
  64-row output tiles publish while later tiles compute. Gate/up intermediates
  have no peer publication in this path.
- MoE transport now gathers each odd die's partial onto its even board mate,
  reduces the three board sums by tile owner, broadcasts to three board
  leaders, then forwards within each board. Both scalar and B4 use the same
  FP32 order: `(die0+die1) + (die2+die3) + (die4+die5)`, with the board sums
  accumulated left to right. This is a precision/association choice relative
  to the older six-way left fold; lossless speculation must be established
  against the scalar implementation of this same target path.
- Ordinary allgathers write one remote copy per destination board. The two
  redundant peer-table entries alias the sender's own local destination, so
  their identical-value stores consume local traffic instead of link traffic.
  A host ledger records each source CTA's exact published byte ranges. The
  matching CTA on each receiving board leader waits for those producers and
  forwards the remote ranges to its board mate while other producer CTAs may
  still be computing. Final exchange completion follows forwarding; readiness
  is not announced before the data arrives.
- Native B4 router projection preserves the scalar 32-lane accumulation order
  and uses the existing local-top10 and packet publication tail for each
  position. Ordinary native BF16 projections now likewise select the actual
  scalar geometry: 32 lanes for K2560, eight for K320.
- Proposed-token PLE rows are submitted to the existing reader pool as one
  block. The full target starts first, so reads and DMA can overlap embedding
  and the first layer. The whole block remains in throughput accounting.

## Build and deployment state

- `kern/libengine_dflash2.so`, `kern/die_dflash2_scalar.cubin` and
  `kern/die_dflash2_b4.cubin` build on Engi. MaxAs reports 249 registers and
  roughly 1.09/1.25 MB of instruction image for scalar/B4. This does not prove
  arithmetic correctness or instruction-cache performance.
- The complete draft module compiled on Archy with CUDA 11.8 and its existing
  GCC 11 toolchain: canonical job `3791cac1-0487-4c41-9062-9cd108591f4a`.
  An earlier build picked up unsupported system GCC and failed before code
  generation (`05323364-5803-49d7-a2c6-4b36036318a4`).
- The cache builder compiled in job `0a4acb80-1c23-40c1-8610-bacd0eed0ce3`.
  An initial network-backed conversion
  (`aa45dc7a-0865-4df7-bb08-1f12a58a34f8`) was stopped explicitly after its
  observed transfer rate was about 10 MB/s. Archy's unused old balanced arenas
  in `/home/nate/work/flash-native-catalog-v41/balanced-cache` were replaced to
  make room locally. Their catalogs, original exporter and rebuild instructions
  are retained, including an Engi backup at
  `/home/nate/models-cache/flash-sass/decode120-v128/previous-balanced-catalog`.
  The original whole-expert source cache is retained. The active conversion
  writes `/home/nate/work/flash-decode120-v128/paired-cache` on Archy; the partial
  Engi cache is not used for inference.
- Current live holders still use physical 0,1,6,3,4,5. The new cache nominates
  0,1,2,3,4,5, with the light logical rank on the fenced die. The engine checks
  CUDA's board-group IDs before using the pair transport. Draft rank two is the
  initial deployment choice, subject to the completed cache's actual capacity.
- `run.lua` describes the concrete transition of only the six identified
  task-owned holders and the first full-model DFlash2 execution. It does not
  automatically roll back this implementation on an inference failure.
  `DECODE120_COMPARE=1` selects the existing benchmark's scalar/B4 comparison
  after the first complete execution. Both arms use the same packed target.

Build integration fixed a generated-SASS dependency cycle, Lua match helpers
that lost the second `find` result, required vector-register aliases, and a
MaxAs FFMA encoding restriction: the Q4/Q5 subtraction now negates the supported
second multiplicand instead of emitting an ignored first-source negation.

## Work still required to establish 120 TPS

Finish the active cache conversion, load the new placement, execute the complete
engine, and repair any runtime issues in this design. Establish scalar/B4 output
agreement and actual acceptance, then report full committed throughput from the
canonical database. The modeled 8.243 ms target matrix service and 24 ms round
budget are still unmeasured. Separate feature captures interrupt five residual
write/read fusions, and ring-bank copying remains extra state traffic; these
are performance work if the complete-round result requires it.

GPUq exact-label search returned no match for a recorded named build; feedback
was filed at https://github.com/nm-z/gpuq/issues/74. Exact SQL label lookup and
document IDs are being used to read the same canonical database.

The older dependency-scheduler prototypes are retained as research. The
memory-scheduling prototype produced incorrect output and is not used by this
image. None of its diagnostic timings is a valid speed claim for this work.
