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
positions. `flashcore` defaults it to `64`; set `0` to disable chunking or use
a decimal size from `1..4096`. Malformed or out-of-range settings fail engine
initialization. For direct engine loads with the setting unset, requests with
more than 4096 positions still to run after the prefix-cache resume point
automatically use chunks of at most 4096; shorter requests keep the single-launch
path. Explicit sizes above one enable the paired pass by default. An explicit
`FLASH_TWO_POSITION=0` selects scalar microbatches within chunks;
`FLASH_TWO_POSITION=1` retains pairing even when chunking is disabled. Automatic
direct-engine chunks keep the configured pairing setting, which is scalar by
default. For chunked requests with `n_new == 0`, the headless prefix pass uses
scalar B1 launches even when pairing is enabled; the fused headless pair faults
on this path.

The prefix program omits the vocabulary head and sampler. It consumes only
`[resume, nprompt-1)`, splitting chunks at exact snapshot boundaries. Within
each chunk, pairs run first and an odd tail runs B1. The final prompt position
and generated positions use the full B1 program. This prefix path also runs when
`n_new == 0`, so a one-token completion request still uses the bounded chunks.
The existing `gen 0` teacher-forced diagnostic keeps the full heads at all
positions. Sampling continues to use absolute position indices, so skipping
unused prefix heads does not shift the seeded sampler.

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
