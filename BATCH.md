# Two positions per phase (MTP verify + draft, prefill chunks)

The split's per-phase fixed costs (barrier + dispatch ≈ 3 µs, cold instruction fetch
5–12 µs for a GEMV body, one or two memory latencies per unit, the exchange fence and
flag flight) dominate a token on six dies; the data a phase moves is small. Every one
of those costs is paid once per phase whatever the number of positions a phase
handles, so the lever is **more positions per phase**: MTP needs two (the accepted
token and the draft), prefill wants a chunk. This note fixes the design for B = 2
first (prefill chunks generalise the same way later).

## Implemented known-position pass

`FLASH_TWO_POSITION=1` enables a causal forward over pairs of known prompt IDs.
With both prefill settings unset, the default remains B1. Pairs stop before the
final prompt position; an odd tail,
the final prompt position and generated-token decoding run through the B1 program.
Snapshots split the known prefix at their exact position. This is the execution
foundation for prefill; draft proposals, speculative acceptance and rollback are
still future work in the design below.

Each logical paired phase contains two adjacent 128-byte scalar descriptors.
Word 30 bit 3 marks the primary descriptor. The dispatcher fixes up each token mask
independently, executes both positions and then performs one grid barrier and one
peer publication. `ctl[9]` selects one or two positions per iteration; zero retains
the scalar ABI. Secondary activations, router lists, reduction counters and both
exchange slots have separate allocations. Layer states and positional KV storage
are shared. Stateful units retain their CTA ownership and run position 0 before 1.
Every warp reads the replay decision before any warp clears it.

The five appended BF16 variants (`bf16biasp`, `bf16k2560p`, `bf16k320p`, `bf16hcp`,
`bf16hcxp`) reuse each weight load across two accumulation chains. Bit 4 marks these
fused bodies. Fusion requires equal weights, dimensions and unit counts, no token
mask, and independent output storage. XOUT additionally requires the caller's
allocation guarantee for the peer destinations. The engine supplies that guarantee
through its separate inbox allocations. Other BF16 variants, quantized GEMVs and
stateful kernels replay their existing arithmetic for both positions. They do not
yet amortize every weight or recurrent-state load. Distinct MoE expert lists remain
independent; no expert-union optimization is claimed.

## Implemented chunked prefill

`FLASH_PREFILL_CHUNK=N` bounds each host submission to at most N known prompt
positions. An absent setting or `0` disables it; decimal sizes `1..4096` are
accepted and malformed or out-of-range settings fail engine initialization.
Sizes above one enable the paired pass by default. An explicit
`FLASH_TWO_POSITION=0` selects scalar microbatches within the same headless chunks;
`FLASH_TWO_POSITION=1` retains pairing even when chunking is disabled.

For requests that generate tokens, the prefix program omits the vocabulary head
and sampler. It consumes only `[resume, nprompt-1)`, splitting chunks at exact
snapshot boundaries. Within each chunk, pairs run first and an odd tail runs B1.
The final prompt position and generated positions use the full B1 program. The
existing `gen 0` teacher-forced diagnostic keeps the full heads at all positions.
Sampling continues to use absolute position indices, so skipping unused prefix
heads does not shift the seeded sampler.

Client liveness is polled before every chunk, including when the preceding launch
finished before the in-flight poll. A disconnected request stops submitting chunks;
an active launch retains the existing cooperative abort protocol. Only a fully
completed snapshot boundary is saved. `config`, `statehash` and `gen` statistics
report the configured size and completed chunks; the streaming chat format stays
the same.

The chunk size controls scheduling, not the number of activation vectors allocated:
the device programs use B1 or B2 and reuse their buffers throughout a chunk.
General B=N matrix kernels, concurrent request scheduling and speculative MTP
acceptance remain separate work.

## Opt-in eight-position causal replay

`FLASH_PREFILL_BATCH=8` selects eight known positions per device program iteration.
Most scalar operations replay positions 0 through 7 in causal order, followed by
one grid barrier and one peer publication for the complete group. Five BF16
variants share weight loads across eight positions: ordinary K2560 and K320,
K2560 with bias, the four-slice HC projection, and its XOUT form. Other operations
retain causal replay. The existing shared-weight BF16 B2 kernels remain available
for tails.

Eight position programs have separate activation scratch, router/selection
scratch and two peer inbox slots each. The contiguous KVGATHER, ATTNPARTN,
ATTNMERGE region is strictly validated and runs all three operations for one
position before starting the next, with the existing grid barrier after each op.
Positions 2 through 7 share position 0's compact K/V and attention partial scratch;
position 1 retains its independent scratch for the unchanged B2 tail. Each
position's merged attention output remains separate. Positions 2 through 7 also
reuse unused head scratch, since they only run headless programs. Each of those
six positions packs its owned activation fields and inboxes into one arena with
256-byte field alignment, avoiding many small device allocations.
Weights, IDs, positional KV and recurrent state stay shared. Stateful virtual
units retain their CTA/warp owner and synchronize between positions. The four-slot
convolution/indexer rings and ten-slot PLE ring therefore retain their scalar
causal semantics, including when a group crosses their wrap boundaries.

If no host chunk is configured, explicit B8 selects chunks of eight. A configured
`FLASH_PREFILL_CHUNK` still bounds each host submission; each chunk uses B8 first,
then B2 and B1 for its remainder. Snapshots split at their exact requested
position, and the final prompt position and generated positions remain full-head
B1. Explicit B8 selects the B2 tail even if `FLASH_TWO_POSITION=0` is also set.
`gen 0` retains the existing full-head B1/B2 diagnostic. Absent or zero
`FLASH_PREFILL_BATCH` preserves the existing B1/B2 defaults and allocation sizes;
other batch settings are rejected.

At 230,000-position capacity and the default 512-block sparse-attention budget,
each new headless arena requests 960,512 or 961,024 bytes under the six-die split,
so six additional positions request 5.496 to 5.499 MiB per die. This includes field
alignment and inboxes, and excludes descriptor storage and device allocator
rounding. The larger control/timestamp array adds 112 KiB over B2. Program storage
uses 128 bytes per actual physical descriptor, and retains the old allocation
until growth succeeds; the legacy program allocation was 512 KiB. Sharing the
compact attention region avoids 96.75 MiB of extra K/V gathers per attention die.
Explicit B8 fails if allocations do not fit; it never shrinks the context or
silently changes the selected batch. A nearly full serving configuration still
needs its actual allocation footprint checked before deployment.

The primary descriptor's word 30 retains bit 3 for grouping and bit 4 for fused
B2; bits 8 through 15 encode an explicit group width. A zero width retains the
legacy two-position descriptor ABI. B8 programs support 32,768 physical
descriptors. Their device program buffers grow to the actual descriptor count;
control/timestamp storage has the fixed maximum capacity and keeps stable
addresses. Profiling compacts timestamps using each phase's physical width,
including the scalar attention triples. Generated-token traces retain scalar strides.
`config` reports `prefill_batch`, and `statehash` reports `batched_positions` in
addition to the existing B2 and host-chunk counters.

This path requires a cubin generated from the updated `die.pl`. CPU descriptor,
planner and code-generation checks do not establish GPU state parity or a
throughput improvement; those need measurement through fresh chat requests.

### Shared-weight B8 BF16 bodies

`gemv_bf16_batch.sass` loads one packed BF16 weight fragment and streams eight
floats from each position into small fragment banks. Eight independent FFMA
chains interleave with at least eight issue slots between dependent instructions.
Each position retains the scalar FFMA sequence, lane butterfly and slice-add
order. The largest new variant reaches register 195 including descriptor
parameters and framework registers, below the existing kernel maximum of 248.
It adds no device activation allocations beyond the B8 arenas above.

Five opcodes are appended, retaining all earlier scalar and B2 opcode values.
Only eligible B8 phases select them: weights, dimensions and unit counts must
match, token masks must be clear, and output regions must be disjoint from every
position's input, bias and output. XOUT requires the caller's guarantee that peer
inboxes are independent; the engine supplies it through the position arenas.
RMS, RMS with stream writes, HCUP and quantized projections retain their existing
generic paths.

Fused B8 phases preload seven secondary descriptors into shared banks from
`0xbe80` through `0xbb80`. Virtual constant bank 2 retains the existing position-1
ABI, and banks 3 through 8 select positions 2 through 7. The fused bit skips
generic replay while the descriptor and token strides remain eight. Generic B8
does not preload these extra banks.

The existing BF16 fixture accepts `BF16_POSITIONS=8` and `FLASH_CUBIN`, comparing
all five variants with eight scalar bodies, including non-multiple rows, virtual
CTA loops, separate biases, 48 XOUT destinations and output guards. Run GPU
fixtures on archy through `gpuq run -- env BF16_POSITIONS=8 ./pair_bf16_test`.

See [PAIR_TESTING.md](PAIR_TESTING.md) for the component and model comparisons.
The remaining sections describe the broader optimization and speculative-decoding
plan, including work beyond this implemented pass.

## What changes per op class

- **GEMVs (every weight format):** two activation vectors, one pass over the weights,
  two results per row. Register budget is the problem: the kernels keep a lane's
  activations in registers (Q2_0: 160 int16 in 80 registers; K-quant: 5 sub-blocks;
  BF16: 8C fp32). B = 2 doubles that and does not fit beside the 8-way XMAD
  accumulators. Two ways out, measured per kernel:
  1. activations in shared memory, read with `LDS.128` broadcast at use (the SPEC's
     option from Phase 1): registers drop to the accumulators + a few staging words;
     one more LDS per 8 weights, no extra global traffic;
  2. fewer sub-blocks per lane (more lanes per row, SHFL reduce wider): halves the
     resident activations per lane; costs reduction steps.
  Either way the second accumulator set is 8 more registers per position for the XMAD
  kernels, one FFMA chain for BF16. Outputs: `y[pos][row]` (position-major, so the
  consumers' row slices stay contiguous per position); XOUT stores both rows.
- **Elementwise (rms, hcsilu, hcmix, hcwrite, actq, ffnact, moesel, sumn, dnnorm,
  conv1d prologue):** a position loop inside the unit (unit count unchanged; the
  second position's addresses are the first's + a stride), or 2× the units.
- **dnstep:** the recurrence is sequential in position: process position 0 then
  position 1 inside the unit with the state in registers/shared memory between them
  (today's kernel already loads and stores the 128×128 state once per head; two
  positions amortise that too).
- **attention:** two queries against the cache (the second sees the first's K/V:
  write K/V for both positions first, then attnqk/attnpart/attnmerge per position
  with the causal limit n+1 and n+2).
- **router / top-10 / moesel:** per position; the expert sets differ per position, so
  the gather GEMVs take two expert lists (up to 20 experts, each row range tagged
  with the position whose activation it multiplies).
- **PLE rows:** two row sets (positions t and t+1; the second's context includes the
  first's token, known at step start for the draft).
- **head:** two argmaxes; `amax` reduces two pairs per die.
- **exchanges:** the parts carry both positions (2× bytes, same count).
- **framework:** the token loop advances by the number of positions accepted (1 or 2):
  the head decides acceptance (draft == the main model's token at position t+1) and
  writes the count the token-end code adds to TOK; positions, cache slots and PLE
  contexts take the counter as today (`+tok` mask).

## Order of work

1. `actq` / `rms` / `hcwrite` with a position loop (cheap; builds the B-aware
   descriptor conventions: a `nb` word and per-position strides).
2. BF16 GEMV with B = 2 (register room exists; the hc down/up GEMVs are 4 of the
   ~17 GEMV phases per block) — bit-exact against a twin that runs the two positions
   one after the other.
3. The Q2_0 expert GEMVs with activations from shared memory (the MoE is half the
   block's GEMV time); then the K-quant / IQ4 / b32 kernels the same way.
4. dnstep, conv1d, dnnorm with two positions; attention with two queries.
5. The MTP block on die 7 (Q8_0 everywhere: the b32q80 path), its head row-split,
   acceptance in the head, the token-loop advance by 1 or 2.
6. Prefill: the same ops with B = chunk (register/smem budgets decide the chunk per
   op; the program takes B from the descriptor so prefill and decode share ops).

Every step keeps today's B = 1 path intact (the descriptor's `nb` word defaults to 1)
and is checked with the existing twins before the next.
