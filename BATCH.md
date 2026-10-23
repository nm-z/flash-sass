# Two positions per phase (MTP verify + draft, prefill chunks)

The split's per-phase fixed costs (barrier + dispatch ≈ 3 µs, cold instruction fetch
5–12 µs for a GEMV body, one or two memory latencies per unit, the exchange fence and
flag flight) dominate a token on six dies; the data a phase moves is small. Every one
of those costs is paid once per phase whatever the number of positions a phase
handles, so the lever is **more positions per phase**: MTP needs two (the accepted
token and the draft), prefill wants a chunk. This note fixes the design for B = 2
first (prefill chunks generalise the same way later).

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
