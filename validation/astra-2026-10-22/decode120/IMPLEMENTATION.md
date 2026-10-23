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
  The current collective change aggregates completion flags only.
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
  have no peer publication in this path. Final partials currently retain the
  six-die sum; board-pair reduction/transport is still outstanding.

## Work still required to earn the budget

1. Complete native B4 router and integration of the new target families. The
   routed gate/up weights still use the existing operators over their local
   channel rows; native compact-column down includes routed and block-32
   shared-down work. The dense and MoE native bodies have been written but
   their instruction scheduling, assembly and execution are not yet verified.
   The target is not yet entitled to the 8.243 ms modeled B4 matrix service.
2. Generate the new paired cache and account for its actual capacity alongside
   target state, DFlash and existing allocations. The row/column layout and
   local-intermediate path are written but no cache was generated or loaded.
   The old unique mode remains separately identified and still uses its old
   exchanged expert intermediate. The new paired mode has a separate ABI.
3. Board-pair tiled payload transport and MoE reduce/scatter. Aggregating flags
   is not the guide's hierarchical payload implementation. Current physical
   placement remains 0,1,6,3,4,5; it is not automatically the proposed board-pair
   topology. New paired-cache catalogs nominate 0,1,2,3,4,5, but loading must
   match the actual board topology and capacity. Preserve unrelated allocations
   and the existing memory fence.
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
