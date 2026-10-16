# Spec: Flash-Next persistent SASS kernel on archy's 4× M60

Status (2026-10-15 23:30): Phase 0 done, numbers below. Phases 1+ need the model
files, which are currently unreachable (see "Blockers").

## Rules

- One `cuLaunchKernel` per die per response. Prefill, decode, MTP draft/verify,
  sampling and the token feedback all happen inside that one persistent kernel.
  No per-token launches, memcpys or host round trips.
- Weights are used exactly as stored in the published GGUF. Integer formats are
  decoded with integer ops (shift/mask/PRMT) and dotted with packed XMAD; the
  block scale is applied once per block in fp32.
- MTP head drives 2-token speculative steps.
- The n-gram (PLE) table lives in archy RAM and is read by die 0 over PCIe.
- Written in Maxwell SASS through MaxAs. Host side in C on the driver API.

## Target hardware (measured)

8 dies (Tesla M60, GM204, sm_52), 16 SMs each, 1177 MHz, 8129 MiB (dies 2,3: 7618
MiB). Driver 470.256.02. Archy: 48 GB RAM, 12 cores, 8 GB swap, root SSD 461 GB
(6 GB free), HDD 285 GB (41 GB free).

| Probe | Result |
|---|---|
| Peer access | every pair, both directions, including across the host bridge |
| Flag round trip, P2P, same board (0↔1, 2↔3) | 1.5 µs |
| Flag round trip, P2P, across boards | 1.9 µs (0↔2, 1↔2), 2.8 µs (0↔7), 3.9 µs (3↔4) |
| Flag round trip through host-pinned memory | 2.8–3.7 µs |
| Host-RAM row read from die 0 (dependent, 92 B) | 1.04 µs; 16 rows in parallel: 1.1 µs total |
| Host-RAM row read from die 7 | 1.85 µs |
| Host-RAM random-row throughput | ~5 GB/s (4096 rows in flight) |
| Device-memory row read (same kernel) | 0.23 µs dependent, 80 GB/s with 4096 rows |
| Write-combined vs cached pinned vs cuMemHostRegister | identical |
| cuMemHostAlloc 1 GiB / cuMemHostRegister 1 GiB | 0.43 s / 0.14 s |
| XMAD issue rate | 123.7 per clock per SM (97% of 128); same as FFMA |
| Software grid barrier (16/32/64 CTAs) | 1.51 / 1.59 / 1.68 µs |
| Die 7 sustained clock under load | 811 MHz effective vs 1078 on die 0 (power or heat) |

Probe sources: `probes/probe.c`, `probes/probes.ptx`. Build with ptxas from CUDA
11.4 (`/opt/cuda-11.4/bin/ptxas`): cubins from the 12.9 ptxas are rejected by the
470 driver ("device kernel image is invalid").

## The model file, as it actually is

`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, directory `Q2_0/`:

| Shard | Bytes | Content |
|---|---|---|
| `...-00001-of-00002.gguf` | 37,623,740,192 | 1223 tensors: 48 blocks (~770 MB each), token_embd, output, output_hc_* |
| `...-00002-of-00002.gguf` | 28,800,138,432 | 1 tensor: `per_layer_token_embd.weight` IQ4_NL [160 × 320,001,536] |

Header facts (parsed with `ref/ggufhdr.c` from the first 64 MB of each shard):
arch `qwen4exp`, 48 blocks, hidden 2560, context 262,144, 24 Q heads, 2 KV heads,
head dim 256, rope dims 64 (neox, base 1e7, `dimension_sections` [11,11,10,0]),
full attention every 4th block (blocks 3,7,…,47; `compress_ratios` = 4 there),
indexer 4 heads × 128, `indexer.top_k` 2048, DeltaNet: conv 4, state 128, 16
key groups, 48 value heads, inner 6144; hyper-connections 4 × rank 320; MoE 512
experts, top-10, expert FFN 640, shared FFN 640; PLE on block 1, 3-gram, 8 heads
per order, 16 head ranges of ~20M rows each, multipliers
[23703573157769, 20109073645365, 8052911324071], `ple.eos_token_id` 248044;
vocab 248,320; `general.file_type` 41; alignment 32.

### Weight formats in shard 1 (by bytes)

| Type (ggml id) | Tensors | Bytes | Share | Used for |
|---|---|---|---|---|
| Q2_0 (42) | 202 | 34.05 GB | 90.5% | all `ffn_{gate,up,down}_exps`, most shexp, some attn_q/qkv, `ple_key` |
| BF16 (30) | 483 | 1.47 GB | 3.9% | `hc_*_{down,up,inject}`, `ffn_gate_inp(_shexp)`, `indexer.{q,k}_proj`, `ssm_{alpha,beta}`, `ple_value`, output_hc |
| Q3_K (11) | 91 | 772 MB | 2.1% | `token_embd`, most `attn_gate`, many `attn_qkv`, some shexp |
| Q5_K (13) | 16 | 521 MB | 1.4% | `output.weight`, some attn_v/output/ssm_out |
| IQ4_XS (23) | 56 | 438 MB | 1.2% | `ssm_out` (17/36), `attn_qkv` (13/36), some attn_q/k/v/gate, shexp |
| Q4_K (12) | 38 | 232 MB | 0.6% | attn_output, ssm_out, shexp |
| Q6_K (14) | 12 | 84 MB | 0.2% | attn_output/k/v, ssm_out |
| Q4_0 (2), Q5_0 (6), Q8_0 (8), IQ4_NL (20) | 32 | 32 MB | 0.1% | `ffn_down_shexp` |
| F32 (0), F16 (1) | 293 | 10 MB | | norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ple_conv1d` |

So the kernel needs these decode paths: Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0,
Q5_0, Q8_0, IQ4_NL, IQ4_XS, BF16, F16, F32. Everything except the three float
types is integer-with-block-scales. IQ4_NL/IQ4_XS use a fixed 16-entry int8 level
table (`-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113`), which is
selected with PRMT from four registers — still integer, no float dequant.

### Block layouts (from upstream ggml-common.h / ggml-quants.c)

- **Q2_0**: 64 values, 18 B: `fp16 d; u8 qs[16]`. Element j is
  `(qs[j/4] >> 2*(j%4)) & 3`, value `d*(q-1)` (levels −d, 0, +d, +2d; the
  reference quantizer sets d = amax and never emits code 3). Integer dot:
  `Σ q·a − Σ a`, then × d.
- **IQ4_NL**: 32 values, 18 B: `fp16 d; u8 qs[16]`; j<16 low nibble, j+16 high
  nibble; value `d*k[nib]`.
- **IQ4_XS**: 256 values, 136 B: `fp16 d; u16 scales_h; u8 scales_l[4]; u8 qs[128]`;
  sub-block ib (32 values): `ls = (scales_l[ib/2] >> 4*(ib%2) & 0xf) | ((scales_h >> 2*ib) & 3) << 4`,
  scale `d*(ls-32)`; nibble split as IQ4_NL within each 32.
- **Q3_K**: 256 values, 110 B: `u8 hmask[32]; u8 qs[64]; u8 scales[12]; fp16 d`;
  16 sub-blocks of 16; 6-bit scales (−32) unpacked with masks 0x0f0f0f0f/0x03030303;
  `q = (qs >> shift & 3) − (hmask bit ? 0 : 4)`.
- **Q4_0** 32/18 B `(q−8)·d`; **Q5_0** 32/22 B with `qh[4]` high bits, `(q−16)·d`;
  **Q8_0** 32/34 B int8·d.
- **Q4_K/Q5_K** 256 values, 144/176 B: `fp16 d, dmin; u8 scales[12]` (6-bit scale
  and min per 32), `qs[128]` (+`qh[32]` for Q5_K); `d·sc·q − dmin·m`.
- **Q6_K** 256 values, 210 B: `ql[128], qh[64], i8 scales[16], fp16 d`;
  `d·sc·(q6−32)`.
- **BF16**: upper half of fp32; unpack with one PRMT/SHL, then FFMA.

### Per-token bytes (decode, one token)

| Source | Bytes/token | Note |
|---|---|---|
| Expert FFN, 10 of 512 × 48 blocks | ~0.66 GB | Q2_0, the part XMAD dominates |
| Hyper-connection BF16 mats, 48 × 4 × 10240×320 | ~1.26 GB | BF16, the largest single cost |
| Attention/DeltaNet projections, routers, indexer, shared experts | ~1.0 GB | mixed |
| Output head Q5_K 2560 × 248,320 | 437 MB | die 7 only |
| N-gram rows 16 × 90 B | 1.4 KB | host RAM |
| **Total** | **~3.4 GB** | |

Each die moves its ~1/8 at ≤ 160 GB/s (M60 peak), pipelined die 0→7, so a single
stream is bounded near 25–30 ms/token (≈ 35–40 tok/s) before MTP; a 2-token MTP step
roughly doubles that. The HF card's 94 tok/s was on faster memory. If Nate wants
more, the lever is requantizing the BF16 hyper-connection mats to Q8_0 at load
(halves 1.26 GB) — not bit-exact to the published file, so it is an option, not
the default.

## Memory placement

| Where | Contents | Size |
|---|---|---|
| Archy RAM, pinned (`cuMemHostAlloc` or `cuMemHostRegister` of the mmap'd shard) | `per_layer_token_embd` IQ4_NL | 28.8 GB (48 GB box → nothing else big may run) |
| Each die | 6 blocks, contiguous as in the file | ~4.6 GB |
| Die 0 extra | `token_embd` Q3_K 273 MB, PLE conv/norms | |
| Die 7 extra | `output.weight` Q5_K 437 MB, `output_hc_*` 13 MB, MTP block + head | ~0.5 GB + MTP |
| Per die | DeltaNet states (3 MB/layer fp32 + one rollback snapshot), KV cache 2 KB/token/attention layer, indexer keys 128 × fp16/token/attention layer, mailboxes | small |

MTP file: the Q8_0 MTP GGUF was on sentry. Archy has
`~/models-hdd-backup/Qwen3.8-Flash-Next-GGUF/mtp-Qwen3.8-Flash-Next-Q4_K_M.gguf`
(2.79 GB) locally. Die 7 has room for either.

Layer split is a Phase-1 output: balance bytes **and** cycles; die 7 carries the
head and runs slower (811 MHz measured), so it gets fewer blocks.

## Execution model

- **Launch**: host loads tensors, pins the n-gram shard, zeroes mailboxes, then
  launches 8 persistent kernels (one per die). Afterwards it only reads a token
  ring buffer in pinned memory until the stop flag is set.
- **Grid**: SMs × resident CTAs per SM (all co-resident, checked with the
  occupancy API). Grid barrier = atomic counter + generation word, 1.5–1.7 µs
  measured. Budget: ≤ 8 barriers per block → ≤ 48 per die per token ≈ 80 µs,
  well under the ~3 ms of memory traffic per die.
- **Pipeline**: token *t* flows die 0 → 7. Mailbox = 40 KB hyper-stream state
  (4 × 2560 fp32), plus token id, position, step kind, and a sequence word. The
  sender writes the payload with P2P stores, `membar.sys`, then the sequence
  word; the receiver spins on the sequence word in its own memory. P2P is used for
  every hop (works across the bridge, 1–2 µs one way); host-pinned mailboxes are
  the fallback only.
- **Feedback**: die 7 samples (greedy or temperature; device RNG), writes the
  token to die 0's inbox and to the host ring buffer with a timestamp.
- **N-gram**: die 0 computes, for orders n∈{2,3} and heads h∈0..7:
  `mixed_n = XOR_{j<n}(c_j · m_j)` (u64 wrapping, c_0 = current token, c_1, c_2 its
  predecessors; a predecessor before the start or at/behind `ple.eos_token_id`
  reads as 248044), row = `head_offsets[h'] + mixed_n mod head_vocab_sizes[h']`
  with h' = h for order 2 and 8+h for order 3. 16 rows × 90 B are read directly
  from host RAM (1.1 µs for all 16, measured) and prefetched as soon as the token
  is known.
- **MTP speculation**: die 7 runs the MTP block on the final state to draft
  *t+2*; the pipeline then carries 2 tokens per step (verify *t+1*, speculate
  *t+2*). On rejection the speculative DeltaNet update, KV write and indexer key are
  rolled back from a one-step snapshot.
- **Prefill**: same kernel, prompt tokens in chunks through the same pipeline.
- **Termination**: EOS/max tokens → stop flag → all loops exit → 8 completions.

## Block math (cx-flash, `results/cx-flash-blockmath.md`, plus the GGUF metadata)

Metadata (`ggufhdr … kv`): eps = 1e-6 everywhere (RMS, L2 floor, indexer, PLE);
`rope.dimension_count` = 64 (partial), `rope.freq_base` = 1e7,
`rope.dimension_sections` = [11, 11, 10, 0] (M-RoPE sections — for text all
position components are equal, so this is plain NeoX RoPE on the first 64 dims);
`indexer.top_k` = 2048, `compress_ratios` = 4 on attention blocks (0 elsewhere);
`expert_gating_func` absent → softmax; `ple.ngram_size` 3, `heads_per_ngram` 8,
eos 248044 (= BOS id; tokenizer eos 248046), head offsets/vocab sizes as in the
SPEC n-gram section; no logit soft-capping key. All norms are RMSNorm with the
plain weight (no 1+w); no block has a pre- or post-norm — the only norms are
the hyper-connection norms, `ssm_norm`, q/k norms, indexer norms and PLE norms.

- **Stream and hyper-connections.** Stream S ∈ R^{4×2560} (4 lanes). Embedding:
  all 4 lanes = `token_embd[id]` (Q3_K row). Each block runs two hyper-blocks
  (mixer with `hc_attn_*`, MoE with `hc_ffn_*`), each:
  `x̂_l = RMS_2560(S_l) ⊙ hc_norm[l]` (10240 weights = per-lane slices);
  read gate `g = σ(W_up · SiLU(0.25 · W_down · x̂))` with `W_down` 10240→320,
  `W_up` 320→10240; branch input `z = 0.25 · Σ_l g_l ⊙ x̂_l` (2560);
  `y = Branch(z)`; write gate `w_l = 2σ(0.25 · (W_inj x̂)_l)` (`hc_inject` 10240→4);
  `S_l += w_l · y`. Head: the same read gate with `output_hc_{norm,down,up}`
  gives h (2560, no further norm); `logits = output.weight · h`.
- **Gated DeltaNet** (36 blocks): `qkv = W_qkv z` (10240 = q 16×128 | k 16×128 |
  v 48×128); depthwise causal conv k=4 over all 10240 channels (`ssm_conv1d`
  [4×10240], oldest tap first, state = last 3 inputs per channel); SiLU on all
  10240; `q̂ = q/max(‖q‖₂, ε)`, `k̂` likewise per 128-vector; gates
  `a = W_α z + ssm_dt.bias`, `b = W_β z` (48 each); `g_h = −exp(A_h)·softplus(a_h)`
  where the file stores `ssm_a = −exp(A)` (so `−exp(A_h) = ssm_a_h`), decay
  `α_h = exp(g_h)`, write `β_h = σ(b_h)`; per value head h with key head
  `kh(h)` and S ∈ R^{128×128}: `S ← α_h S; r = k̂ᵀS; e = v − r;
  S += β_h · k̂ eᵀ; o = q̂ᵀS` (readout after the write); `o ← o/√128` **once**
  (cx-flash's B path applies it twice — a bug there, ours applies it once;
  verify against llama.cpp); `ô = RMS_128(o) ⊙ ssm_norm` (128 weights shared by
  heads); `ô ⊙ σ(W_g z)` (`attn_gate` 2560→6144, **sigmoid**); `ssm_out` 6144→2560.
  **Open: key-head mapping** — cx-flash B tiles `kh = h mod 16`, A's inference
  path groups `kh = h / 3`; llama.cpp's Qwen3-Next uses grouped (3 consecutive
  value heads per key head). We implement grouped and verify numerically.
- **Full attention** (blocks 3, 7, …, 47): `attn_q` 2560→12288 = 24 × [q (256) |
  gate (256)], `attn_k`/`attn_v` 2560→512 (2 KV heads × 256), no biases; per-head
  RMS_256 on q and k (shared weights), v unnormalised; NeoX RoPE on dims 0..63
  (pairs (p, p+32), θ_p = 1e7^(−2p/64)); `score = q·k/√256`, query head i uses
  KV head i/12; softmax over admitted causal keys; `out_i ⊙ σ(gate_i)`;
  `attn_output` 6144→2560.
  Indexer: `iq = W_iq z` (4×128), `ik = W_ik z` (128); `iq_h ← RoPE(RMS_128(iq_h)
  ⊙ q_norm)` at position t; keys pooled per block of B = 4: representative
  `r_b = RoPE_{pos=4b}(RMS_128(mean_j ik_j) ⊙ k_norm)` (the block containing t
  uses its prefix mean); `score_b = Σ_h ReLU(iq_h · r_b)`; admit the
  `ceil(2048/4) = 512` highest blocks (ties → lower index); attention runs over
  every key in admitted blocks. (Below 2048 tokens nothing is dropped.)
- **MoE** (every block): `p = softmax(W_r z)` (`ffn_gate_inp` BF16 2560→512),
  top-10 (ties → lower id), `w_e = p_e / Σ_top p`; `y_e = W_down,e(SiLU(W_gate,e z)
  ⊙ W_up,e z)`; `routed = Σ w_e y_e`; shared: `s = σ(ffn_gate_inp_shexp · z)`
  (scalar), `y_s = W_down,s(SiLU(W_gate,s z) ⊙ W_up,s z)`; `out = routed + s·y_s`.
- **PLE** (top of block 1, on the 4-lane stream): rows from the hash in the
  execution-model section, `g` = 16 rows × 160 = 2560; `K = W_k g` (10240 =
  per-lane), `k̂ = RMS_2560/lane(K) ⊙ ple_norm_key`, `Q = RMS_2560/lane(S) ⊙
  ple_norm_query`, `s_l = (1/√2560) Σ_{c∈lane l} k̂_c Q_c`, `γ_l = σ(sgn(s_l)·
  √max(|s_l|, 1e-6))`, `V = W_v g` (2560), `G_l = γ_l V`, `out = G +
  SiLU(DConv_{k=4, dilation=3}(RMS_2560/lane(G) ⊙ ple_norm_conv))` (`ple_conv1d`
  F16 [4×10240], a 4-tap causal conv over tokens per channel), `S += out`.
- **MTP** (block 48 file): `e = RMS(token_embd'[next]) ⊙ nextn.enorm`; per lane
  `u_l = RMS(S_l) ⊙ nextn.hnorm[l]` where S is the head-normalised stream (x̂ of
  the output hyper-block); `lane_l = W_eh [e ; u_l]` (`eh_proj` 5120→2560, same
  for all lanes); then one ordinary hyper-block pair (gated attention + MoE)
  and the head `nextn.hc_head_*` → its own `output.weight`. Drafts verified by
  equality with the main model's argmax; greedy output is unchanged by MTP.
- **Numerics in cx-flash** (for tolerance expectations): fp32 accumulation,
  DeltaNet state fp64 in its A path, experts forced fp32, activations quantised
  per 32-element block (`d = max|x|/127`, round half away) — we quantise per 32
  as well (see Numerics) but with round-to-nearest-even and an IEEE reciprocal.

## Numerics

- Activations quantised **per 32-element group** (as cx-flash and llama.cpp's
  Q8_1 do; a per-vector scale lost most of the vector to one outlier) to int8
  codes (carried as int16) + one fp32 scale per group, by `kern/actq.sass` (one
  warp per vector) with the bit-exact twin `quant_act_gpu32`: gmax = max(max |x|
  over the 32, 2^-100); r = rcp_rn(gmax) (the ptxas fast path: MUFU.RCP, FFMA
  e = gmax·r − 1, FFMA r += r·(−e), IEEE 1/gmax for any normal gmax below 2^125);
  inv = r·127; q = F2I.RN(x·inv) = lrintf(x·inv); per-16 and per-32 sums of the
  codes; as[g] = gmax·(1/127). The GEMV kernels multiply each sub-block's
  contribution by as[g] (one extra FMUL per 32 elements). Verified bit-exact for
  n = 640, 2560, 6144, 10240 on six magnitudes including zero and an outlier.
  Rules found on the way: SASS
  float immediates are 20-bit (1/127 is not representable; constants like it
  come from MOV32I); any predicate consumer — EXIT and MOV included — needs the
  ~18-cycle gap; a counter incremented at stall 1 is read stale by the next
  instruction (the loop ran once too often and its stray stores zeroed the
  neighbouring buffers).
- Block inputs (2560 wide) go through `kern/normq.sass`: RMSNorm fused with the
  quantiser, one warp: ss by one FFMA chain per lane over its 4-element chunks
  plus the SHFL butterfly, m = ss·(1/2560), v = m + eps, s = sqrt_rn(v) (ptxas
  MUFU.RSQ fast path, IEEE), r = rcp_rn(s), y = (x·r)·w stored fp32 for the fp32
  consumers (router), then the quantiser on y. Twin `rmsnorm_quant_gpu`,
  bit-exact on six magnitudes. The norm weight convention (w vs 1 + w) is a
  load-time transform of the weights, not a kernel property.
- Q2_0 × int8 in XMAD SWAR: each 16-bit lane accumulates up to 8 products
  (|q·a| ≤ 3·127) before extraction; other widths pack fewer terms. Accumulate in
  int32, multiply by the block scale once per block in fp32.
- fp32 for norms, DeltaNet state and update, softmax, RoPE, router, hyper-connection
  gates, sampling. BF16 weights unpacked by shift and used with FFMA.

## Toolchain

- `/home/nate/claude/flash-sass/` (engi mirror and archy copy via rsync). Not in
  `~/work/cx-flash` (its AGENTS.md forbids new files there).
- MaxAs: `maxas/` with the grammar patched for CUDA 12.9 disassembly
  (`ISCADD R.CC`); round trip of a ptxas kernel is bit-identical and a changed
  immediate re-encodes correctly (`t3.cubin`). Use `cuobjdump` from 12.9 and
  `ptxas` from 11.4.
- Host: C, driver API, prototype `cuCtxCreate_v2` (12.9 headers map
  `cuCtxCreate` to a v4 the 470 driver lacks).
- Kernel source: one `.sass` per die role (die 0, middle, die 7) built from shared
  macro files: `q2_0_gemv`, `q3_k_gemv`, `q4_k_gemv`, `q5_k_gemv`, `q6_k_gemv`,
  `q4_0_gemv`, `q5_0_gemv`, `q8_0_gemv`, `iq4_gemv` (NL and XS), `bf16_gemv`,
  `deltanet_step`, `gqa_attn_step`, `indexer_topk`, `moe_route`, `hc_gates`,
  `ple_lookup`, `barrier_grid`, `mailbox_wait`, `mailbox_publish`, `sample`.

## Phase 1 status (2026-10-16)

- `kern/gemv_q2_0.sass` (MaxAs, 713 instructions, 128 registers, 2 CTAs/SM):
  **bit-exact** against `ref/quant.c` on `blk.4.ffn_gate_exps.weight` for experts
  0, 7, 511 and for all 327,680 rows at once (236 MB), with random, all-ones and
  alternating activations. Verified by `kern/gemv_test`.
- Speed: 57 GB/s of tile on the full tensor (4.1 ms), 36 GB/s on one expert
  (launch-bound at 13 µs). Measured device roofline (`probe bw`, LDG.128
  streaming, plain or read-only path): **145 GB/s**. Decomposition on the full
  tensor: loads only 3.5 ms (67 GB/s), compute only 3.1 ms, both 4.1 ms; the
  10 small scale loads cost nothing, plain LDG is slower than LDG.CI (6.1 ms).
  Ruled out by microbenchmarks (`kern/xmadbench.sass`): the instruction forms
  (XMAD.U16.S16 with H1 selects 106/clk/SM, LOP32I 117, the GEMV mix 76) and the
  instruction cache (no cliff up to a 25 KB loop body). Per-warp iteration time
  in the load-only build is ~5.5 µs for 10 loads, which no latency explains;
  clock-stamp instrumentation (`GEMV_TIMING`) is measuring where it goes.
- **Found (2026-10-16 02:00)**: the 57 GB/s was an artefact of launching one
  16-row iteration per CTA (20,480 CTAs each reloading the 5 KB activation
  vector); with a persistent-style grid (32–96 CTAs looping over rows) the loop
  body runs at memory speed. Two Maxwell rules surfaced while making the loop
  correct, both measured with build variants:
  1. **A predicate written by ISETP must not be consumed for ~18 cycles.** A
     `@P BRA` 6–13 cycles after its ISETP reads the *previous* predicate value
     (the loop ran once); so did `@P2 STG` 7 cycles after its ISETP (the flaky
     K=640 rows: whether a lane stored depended on leftover predicate state).
     Fix: compute P1/P2 at the loop top, far from their consumers. ptxas keeps
     ≥13 cycles plus a yield before predicated branches.
  2. MaxAs's automatic operand-reuse flags are suspect around the pointer bumps
     (`SHL t, rowstep.reuse` followed by an instruction with a different slot-a
     register); builds with `-n` (no reuse flags) behaved differently. Pending the
     research notes, kernels are built with `-n` unless measured otherwise.
- **Die plan (Nate, 02:10)**: the model fits on dies 0–5 (47.7 GB): 8 transformer
  blocks per die would put head + MTP over 8 GB on die 5, so use 9/9/9/9/8/4
  blocks with `output.weight` (0.44 GB) + MTP on die 5 (Q4_K_M 2.8 GB is on archy;
  Q8_0 4.1 GB fits only with ≤4 blocks). Dies 6/7 sit behind a gen2 x2 slot and
  stay off the token path; the middle board is x8, fine for the 40 KB hop.
- **Result (02:40)**: with early predicates and a persistent-style grid (32–96
  CTAs), both Q2_0 kernels are bit-exact on every test (K=2560 expert 0 ×3, all
  327,680 rows at grids 32/48/64/96; K=640 expert 0 ×3 and all 1,310,720 rows)
  and run at **139–140 GB/s** (236 MB in 1.69 ms; 252 MB in 1.80 ms), 96% of the
  145 GB/s streaming ceiling, with or without MaxAs reuse flags. The v1
  structure (lanes split K, activation slice in registers, SHFL reduction) stays:
  in the persistent kernel the 80 activation registers are live only inside the
  matvec phase.
- Register-pressure alternative kept in reserve for shapes where the per-lane
  activation slice does not fit (BF16 K=10240): **lane = row** with activations
  broadcast from shared memory, K split across warps, fixed-order partial sums. A warp
  owns 32 rows, all lanes walk the same k, activations are broadcast from shared
  memory (LDS.128 = 8 int16, conflict-free broadcast), the Σa offsets become
  per-block constants, no SHFL reduction, no 80-register activation footprint,
  one LDG.128 per block per lane, ~1.8 issue slots per MAC, ~75 registers. The
  same structure serves the persistent kernel (a warp takes a 32-row chunk of
  any matvec) and the other formats.
- Toolchain fixes needed on the way, all in `maxas-patch/` and applied on archy:
  `ISCADD R.CC` in the grammar; **EIATTR_REGCOUNT** in the global `.nv.info` is what
  the 470 driver reads for the register count (MaxAs only rewrote the section
  header, so kernels launched with 2 registers and faulted); unknown `.nv.info`
  attributes are now kept instead of dropped.
- Maxwell scheduling facts measured with `kern/semtest*.sass`: XMAD.U16.S16
  half-selects, LOP32I, SHR, I2F, F2F, IADD3 negation, LDG.64/128/U16 and
  SHFL.BFLY all behave as the SASS reads. A variable-latency instruction's write
  barrier is not honoured by an instruction issued the next cycle: give SHFL (and
  I2F/F2F/LDG when the consumer is adjacent) a stall of at least 2. Stall 1 gave a
  reduction that silently used stale shuffle results.
- Reference decoders for all 13 formats decode the real tensors without NaN and
  with plausible statistics (`ref/ref_test`).
- **K-quants (2026-10-16 evening)**: `kern/gemv_q4_k.sass` is one generator for
  Q4_K and Q5_K (`GEMV_FMT=q5_k`, assembled into the `gemv_q5_k` skeleton). The
  tile is sub-block-major (`repack_q4_k`/`repack_q5_k`): per block a 16 B header
  (d, dmin, 12 scale bytes), 8 × 16 B of nibbles (sub-block j, element 2i low /
  2i+1 high), and for Q5_K 8 × 4 B words of high bits (bit e = element e). A
  lane owns sub-block `lane & 7` of blocks `lane >> 3 + (L/8) i`; per sub-block it
  decodes the 6-bit scale/min with predicated ops (`P3 = j >= 4`), streams the
  nibbles as masked 16-bit fields through XMAD.U16.S16 into 8 accumulators, and
  for Q5_K streams `qh >> t` (t = 0..3) with 1-bit masks into 8 more, then
  `dot += dh << 4`. Bit-exact on every tensor tried: Q4_K `blk.0.attn_gate`
  (6144 rows), `blk.3.attn_output` and `blk.1.ssm_out` (K = 6144), `attn_k`;
  Q5_K `blk.3.attn_v`, `blk.0.ssm_out` (K = 6144) and **`output.weight`, all
  248,320 rows, at 137.9 GB/s** (437 MB in 3.17 ms, grid 64 or 96). Registers:
  168/192 (Q4_K K = 2560/6144), 186/212 (Q5_K); 1 CTA/SM in every case, which
  is enough: the K=2560 kernels sit at the 140 GB/s ceiling. Small tensors
  (≤ 11 MB) report 70–75 GB/s because a 120 µs launch never fills the device;
  that is a launch artefact, irrelevant to the persistent kernel.
- MaxAs research notes (haiku agent, 2026-10-16), kept here so they are not
  re-derived: control word = 21 bits per instruction (stall 4 | yield 1 | wbar 3 |
  rbar 3 | wait 6 | reuse 4); yield is raw bit 4, printed `Y`; "barriers take one
  extra clock to become active" is the documented form of our measured stall ≥ 2
  rule; predicate producers are modelled with latency 13 (we measured ~18 to a
  branch). Operand reuse: the flag goes on the *earlier* reader, the cache is
  2-way per operand slot (8 bytes), source `.reuse` tags are ignored unless `-n`,
  and MaxAs's pass ignores predicates and does not treat labels as boundaries;
  nothing documents whether a write invalidates a cached operand on Maxwell.
  Our loops are bit-exact with and without automatic reuse over 1.9 M rows, so
  the default builds keep it; a kernel that misbehaves gets a `-n` build as the
  first experiment. Known MaxAs bugs: TODO opcodes (IMAD, IMUL, DEPBAR with
  non-LE comparators) assemble silently with zero opcode bits; the register count
  ignores vector widths (a 128-bit destination at the top of the file would be
  undercounted — our maps put scalars last); the stall-table entries 12–15 are
  unreliable (two sources disagree) — we never use stalls above 6 except `f` on
  EXIT; `SCHEDULE_BLOCK` reorders only by register dependencies (no barriers, no
  memory order), so we do not use it.
- **Q3_K (2026-10-16 night)**: `kern/gemv_q3_k.sass`, bit-exact on
  `blk.3.attn_k`, `blk.0.ffn_up_shexp`, `blk.5.attn_qkv` (10,240 rows) and
  `blk.6.ssm_out` (K = 6144). Tile (`repack_q3_k`, 1152 B per K=2560 row vs 1100
  raw, 4.7%): a per-row prefix of fp16 d per lane group, then per block 16 code
  words (16 × 2-bit fields), 16 half-words of *inverted* high bits and 16 int8
  scales (sc − 32). A lane owns two 16-element groups (32 consecutive elements)
  of block Q i + q. The 2-bit codes stream through 8 masks `0x00030003 << 2s`
  into 8 accumulators per group; the inverted high bits stream as in Q5_K
  (`hw >> u`, masks `0x00010001 << 4v`) and are subtracted ×4, so no activation
  sums are needed (value = code − 4·!h). Scales are sign-extended in pairs with
  one `PRMT 0x9180` and applied as exact integers: `sumi = sc0·dot0 + sc1·dot1`
  with XMAD.S16.S16 (|dot| ≤ 8128 fits), one I2F + FFMA per block. Both PRMT and
  XMAD.S16.S16 behaved as written on first use. 172/192 registers. Speed: 107
  GB/s on `attn_qkv` (11.8 MB, grid 32): more compute per byte than Q4_K, and
  the kernels do not overlap a row's loads with the previous row's compute
  (one barrier for all of a row's loads, issued after the row's SHFL). The
  fix — issue the next row's loads for step i right after step i's XMADs, with
  loads spread over barriers 1/2/6 and the STG read on barrier 5 — is noted for
  the tuning pass; it applies to every K-quant kernel.
- **Q6_K**: `kern/gemv_q6_k.sass`, bit-exact on `blk.5.ffn_up_shexp`,
  `blk.31.attn_v` and `blk.11.attn_output` (K = 6144, 111.6 GB/s). 208-byte
  block records (zero overhead) plus the d prefix; low nibbles through the Q4_K
  stream with −32·sum16 folded into each group's first XMAD (the harness passes
  per-16 sums in the sums slot), 2-bit high fields through 8 shift accumulators
  per group, int8 scales applied as exact 32-bit products (two XMADs: low half
  unsigned, high half signed, then ISCADD by 16 — |dot| ≤ 65k does not fit
  int16). 204/230 registers.
- **IQ4_XS**: `kern/gemv_iq4_xs.sass`, bit-exact on six tensors (attn_qkv,
  attn_q 12,288 rows, attn_gate, ssm_out K = 6144, attn_k, ffn_gate_shexp) at
  118–129 GB/s. The non-linear table is applied with PRMT: a nibble word is its
  own selector (upper four nibbles after SHR 16); PRMT replicates the sign of
  the selected byte when selector bit 3 is set, so `PRMT(kv[0..7], w)` gives
  kv[q] for q < 8 and 0xFF otherwise, `PRMT(kv[8..15], w ^ 0x8888)` gives kv[q]
  for q ≥ 8 and 0x00 otherwise, `PRMT(0x7F.., w)` gives 0x7F/0x00, `m | m << 1`
  is the byte mask and `LOP3 0xE2` selects; two more PRMTs (0x9180, 0xB3A2)
  sign-extend the bytes into int16 pairs for XMAD.S16.S16 into 8 round-robin
  accumulators. ~108 issue slots per 32 elements (18 B), all stall-free by
  construction (every dependency ≥ 6 issues apart). Scales pre-decoded to int8
  (ls − 32); 184/206 registers.
- **Block-32 formats (Q4_0, Q5_0, Q8_0, IQ4_NL at K = 640, the shared-expert
  down projections)**: one generator `kern/gemv_b32.sass`, assembled four times
  (`GEMV_FMT`, `-k gemv_q4_0` …) into one shared PTX skeleton. 4 lanes per row,
  lane l owns blocks l + 4i; the tile is a 48 B fp16-d prefix plus 16 B (32 B
  for Q8_0) per block, Q5_0 adds 20 high-bit words. Q4_0/Q5_0 reuse the Q4_K
  nibble stream with −8/−16·sum32 folded into the first XMAD (Q5_0 adds the
  Q5_K high-bit stream); Q8_0 sign-extends byte pairs with PRMT and uses
  XMAD.S16.S16 into 8 accumulators; IQ4_NL is the IQ4_XS lookup without the
  scale. All four bit-exact on `ffn_down_shexp` tensors (2560 rows). 144–180
  registers. These matrices are ~1.5% of per-token bytes.
- **BF16 K = 10240 (hc_*_down 320 rows, hc_*_inject 4 rows)**: no new kernel —
  the K=2560 BF16 kernel assembled with `GEMV_ROWSTRIDE=20480 GEMV_ACC=1`
  (`gemv_bf16_hc.cubin`) runs the four column slices in order, each adding its
  partial into y: y = (((0 + p0) + p1) + p2) + p3, p_s = the K=2560 kernel's
  value on slice s. In the persistent kernel a warp keeps its rows and loops
  s = 0..3, reloading the 10 KB activation slice per s from L2 — no grid
  barrier between slices. The reference composes `gemv_bf16_ref` on the slices
  the same way.
- **Grid barrier primitive (Phase 2 start)**: `kern/gridbar.sass` — after
  BAR.SYNC, warp 0 reads the generation word (LDG.E.CV), MEMBAR.GL.IVALLD, lane
  0 does ATOM.E.ADD on the counter and SHFL.IDX broadcasts the count so the
  last-arriver test and the spin loop are warp-uniform (no divergence, no SSY);
  the last arriver STG.E.WT's the counter to 0, fences, bumps the generation;
  the others spin with LDG.E.CV; a fence and BAR.SYNC release the CTA. These
  are exactly the forms ptxas emits for `membar.gl` / `ld.volatile` /
  `st.volatile` / `atom.add` on sm_52 (`MEMBAR.GL.IVALLD` needed a grammar
  flag). Visibility test (each CTA publishes the iteration before, reads its
  neighbour's slot after; double-buffered by parity): **0 mismatches over
  200,000 barriers** at 64 and 128 CTAs; 1.64 / 1.71 / 1.85 / 2.50 µs per
  barrier for 16 / 32 / 64 / 128 CTAs. Lesson re-learned: a predicated MOV six
  cycles after its ISETP read the stale predicate (the wrap-around of the
  neighbour index) — every predicate consumer, not just branches and stores,
  needs the ~18-cycle gap.
- **Mailbox primitive**: `kern/mailbox.sass` — one CTA stores a 40 KB payload
  into the peer die's memory with STG.128 (P2P), BAR.SYNC, MEMBAR.SYS, lane 0
  STG.E.WT's the sequence word; the receiver spins with LDG.E.CV, MEMBAR.SYS,
  reads the payload with LDG.E.CV.128 and acknowledges the same way. **0 payload
  mismatches over 10,000 round trips** on every pair tried; round trip (payload
  one way + ack back): 0↔1 17 µs (same board), 0→2 28 µs (across the middle
  board), 5→0 37 µs, 0→5 48 µs, 3→4 43 µs. Five hops per token ≈ 0.1–0.2 ms
  against ~25 ms of memory time: fine as is; more CTAs issuing the stores would
  cut it if ever needed.
- **Per-die bytes (from the header dump, raw GGUF sizes)**: blocks are 0.76–0.79
  GB each; with 9/9/9/9/8/4 blocks dies 0–3 hold 6.9 GB, die 4 6.16 GB, die 5
  3.09 GB + `output.weight` 0.44 GB + MTP (Q4_K_M 2.8 GB on archy's HDD) ≈ 6.3
  GB, plus `token_embd` 0.27 GB on die 0. Tiles add 1–5%; states, KV, mailboxes
  and activations are small. Everything fits under the 7.6 GiB usable per die
  with ~0.5 GB to spare on dies 0–3.
- **MTP file (Q4_K_M, 2.8 GB, archy HDD) contents** (`ggufhdr`): blk.48 is a full
  attention + MoE block (attn_q/k Q4_K, attn_v Q6_K, attn_output Q4_K, indexer
  BF16, 512 experts: gate/up Q4_K, down Q8_0 (891 MB), shared expert Q4_K/Q8_0,
  router `ffn_gate_inp` **F32**, hc_*_down Q4_K/Q6_K K=10240, hc_*_up **Q5_0
  K=320**, hc_*_inject Q4_K) plus `nextn.eh_proj` Q4_K **K=5120**, `nextn.enorm`
  (2560), `nextn.hnorm` (10240), `nextn.hc_head_down/up/norm`, and its own
  `output.weight` (Q6_K, 521 MB) and `token_embd` (Q4_K, 358 MB). New kernel
  shapes for Phase 5: F32 GEMV K=2560, Q4_K K=5120 (32 lanes × 5), Q4_K/Q6_K
  K=10240 (as accumulating slices, as BF16), Q5_0 K=320. Die 5 budget with
  everything: ≈ 3.1 + 0.44 + 2.8 = 6.3 GB.
- **fp32 RMSNorm** `kern/rmsnorm.sass` (n = 2560 / 10240, one warp, two
  streaming passes, twin `rmsnorm_gpu`) — bit-exact once a hazard was fixed that
  the GEMV kernels had been handling by design: **a memory op reads its address
  and data registers late, so a pointer bump (or any overwrite) after a burst of
  stores must wait on the stores' read barrier.** 20 back-to-back STG.128s
  queue up in the LSU; the last six read the address register after the bump,
  stored to the wrong place, and the unwritten slots showed the harness's NaN
  fill. (Checked on the way with `kern/bartest.sass`: one write barrier tracks at
  least 24 outstanding LDG.128s correctly, with slow host-mapped loads, plain or
  .CI, so the wait-mask semantics are counting.)
- Register-bank note: in the K-quant streams the mask for field v and the
  activation register of the same XMAD share a bank for 2 of every 8 XMADs, and
  the combine's `acc_v + acc_{v+4}` pairs share banks (MaxAs reports 187
  conflicts for Q5_K K = 6144). A conflict-free assignment exists (masks in banks
  1,3,0,2; accumulators 2,1,3,0 / 3,0,2,1; high-bit sources `qh >> 4t` with masks
  `1 << v`). Not done: the kernels are at the memory ceiling, and per token the
  whole model's MACs are ~0.6 ms of issue time per die against ~4 ms of memory
  time.

### Matvec shapes in the file (K = contiguous dimension)

| Format | K | Rows | Count |
|---|---|---|---|
| Q2_0 | 2560 | 640 ×512 experts (gate/up), 640, 6144, 10240, 12288 | 96 expert tensors + 42 |
| Q2_0 | 640 | 2560 ×512 experts (down), 2560 | 48 expert tensors + 16 |
| Q3_K | 2560 / 6144 | 640, 512, 6144, 10240, 12288 / 2560 | 87 / 3 |
| IQ4_XS | 2560 / 6144 | 512, 640, 6144, 10240, 12288 / 2560 | 39 / 17 |
| Q4_K | 2560 / 6144 | 512, 640, 6144, 10240 / 2560 | 23 / 15 |
| Q5_K, Q6_K | 2560 / 6144 | 512, 640 / 2560 | 8+6 / 7+6 |
| Q4_0, Q5_0, Q8_0, IQ4_NL | 640 | 2560 | 32 |
| BF16 | 10240 | 320 (hc down), 4 (hc inject) | 96 + 96 |
| BF16 | 320 | 10240 (hc up) | 96 |
| BF16 | 2560 | 512 (router), 128/512 (indexer), 48 (ssm α/β), 2560 (ple_value) | 48 + 24 + 72 + 1 |

## Phase 2 design (block kernels on one die)

Primitives that exist and are bit-exact against C twins: the GEMV family
(every format/shape), `actq` (per-32 quantiser), `normq` (RMSNorm + quantiser),
`rmsnorm` (fp32, 2560/10240), `gridbar`, `mailbox`. Still to write, each with a
C twin and a tolerance test (transcendentals use MUFU, so bit-exactness ends
where exp/log/sigmoid begin; the twins mirror op order and the tests report max
abs/rel error): SiLU/sigmoid/softplus/exp vector ops, the DeltaNet head step,
conv1d state step, L2 normalisation, per-head RMSNorm + RoPE, attention scores +
softmax + KV cache append, the indexer, router softmax + top-10, the PLE hash +
row gather + gate, the token-embedding row decode (raw Q3_K), hyper-connection
read/write gates.

Phase sequence per block (each line = one grid phase of the persistent kernel;
a phase is a loop over its rows/heads by all warps of the die, then a grid
barrier; "1w" = tiny, one warp):

1. **hc_attn norm**: x̂_l = RMS_2560(S_l) ⊙ hc_norm[l], 4 warps (fp32 10240 out).
2. **hc read-down**: BF16 GEMV 10240→320 (4 slices accumulate) on x̂.
3. **hc read-up**: t = SiLU(0.25·down) (1w), BF16 GEMV 320→10240, g = σ(·);
   also **inject**: w_l = 2σ(0.25·(W_inj x̂)_l) (4 rows × 10240, 4 warps).
4. **mix**: z = 0.25·Σ_l g_l ⊙ x̂_l (2560) → `actq`(z) and keep z fp32.
5. **mixer projections** from z: DeltaNet: qkv (10240 rows), gate (6144),
   α/β (BF16 48+48, fp32 z); attention: q (12288), k, v (512 each), indexer
   q/k (BF16 512/128, fp32 z). One phase, rows of several matrices.
6. **mixer core**: DeltaNet: conv1d(k=4, state) + SiLU on 10240 channels, L2 on
   q/k heads, gates α_h/β_h (softplus/exp/σ), 48 recurrences (one CTA per head:
   S ← α S; r = k̂ᵀS; e = v − r; S += β k̂ eᵀ; o = q̂ᵀS; 64 KB state per head read
   and written), o/√128, RMS_128 ⊙ ssm_norm, ⊙ σ(gate) → 6144 fp32 → `actq`.
   Attention: q/k per-head RMS + RoPE(64), KV append, (indexer when t > 2048),
   24 heads × online softmax over admitted keys, ⊙ σ(gate) → 6144 → `actq`.
7. **out projection**: ssm_out / attn_output (K=6144) → y (2560).
8. **write**: S_l += w_l · y (4 lanes, 1 phase, 4 warps).
9–12. **hc_ffn**: norm, read-down, read-up + inject, mix → z' (+ `actq`).
13. **router**: softmax(W_r z') (BF16 512 rows), top-10 + renormalise (1w),
    shared gate s = σ(w_sh · z') (1w).
14. **experts gate/up**: 10 experts × (gate, up) Q2_0 640 rows each + shared
    gate/up (640 rows, mixed formats): 13,440 rows total in one phase, then
    h_e = SiLU(g_e) ⊙ u_e → `actq` per expert (11 × 640, per-32).
15. **experts down**: 10 × Q2_0 K=640 (2560 rows each) + shared down (b32),
    weighted sum Σ w_e y_e + s·y_s → out (2560) (each row's 11 contributions
    summed in a fixed order inside the row loop).
16. **write**: S_l += w'_l · out.
Block 1 adds the PLE step before phase 1 (hash → 16 rows from host RAM → K/V
projections → per-lane gates → conv over the 4 last tokens' G → S += out);
block 0 starts from the embedding row broadcast to 4 lanes; after block 47 the
head: output_hc norm/down/up gate → h → `output.weight` logits → argmax /
sampling (1w) → token to the ring buffer and the next die's mailbox.

≈ 16 barriers per block → ~800 per token across the 6 dies ≈ 1.5 ms (≈ 5% of
~30 ms); fusions (norm into mix, inject into read-up, router into the mix
phase) come later. State per die: DeltaNet states 3 MB × its DeltaNet blocks,
conv states (3 × 10240 fp32 per block), KV cache 2 × 256 × 2 × 4 B = 4 KB per
token per attention block (fp32; 32k tokens = 128 MB per block ✗ → fp16 KV at
64 MB per block, 2 attention blocks per die → 128 MB at 32k), indexer keys
(128 fp32 per token per attention block), PLE: the last 4 G vectors (4 × 10240).

Verification: each primitive against its C twin; each block against cx-flash's
node dumps (`RECIPE_TRACE_*` on archy, see `results/cx-flash-blockmath.md` §9)
at tolerance; end to end against `RECIPE_REFERENCE_WRITE` logits and greedy
tokens. The two cx-flash ambiguities (key-head mapping, readout scale) are
resolved by whichever choice reproduces llama.cpp's greedy output.

## Phases

0. **Probes** — done (table above).
1. **GEMV kernels** — **done (2026-10-16)**. Every weight format and matvec
   shape in the file has a MaxAs kernel that is bit-exact against the C
   reference on real tensors:

   | Kernel | Shapes | Registers | GB/s (large tensors) |
   |---|---|---|---|
   | `gemv_q2_0` | K=2560, 640 | 128 | 139–140 |
   | `gemv_bf16` | K=2560, 320; K=10240 as 4 accumulating slices | 134/74/136 | 134–137 |
   | `gemv_q4_k` / `gemv_q5_k` (one generator) | K=2560, 6144 | 168/192, 186/212 | 138 (output.weight), 117–129 (K=6144) |
   | `gemv_q3_k` | K=2560, 6144 | 172/192 | 107 |
   | `gemv_q6_k` | K=2560, 6144 | 204/230 | 112 |
   | `gemv_iq4_xs` | K=2560, 6144 | 184/206 | 118–129 |
   | `gemv_b32` → q4_0, q5_0, q8_0, iq4_nl | K=640 | 144–180 | small matrices |

   Scheduling lessons folded into the generators: a mask LOP must be >= 6 issues
   before the XMAD that reads it (the next word's masks precede this word's
   XMADs); a combine must not read an accumulator within 6 cycles of its last
   XMAD; predicates at the loop top; stall >= 2 after a variable-latency op
   before a dependent wait. Not yet done, for the tuning pass: overlapping the
   next row's loads with this row's compute (Q3_K/Q6_K/K=6144 kernels sit at
   75–85% of the ceiling because of it); the conflict-free register banks.
   Not GEMVs, handled in Phase 2: the n-gram table gather (IQ4_NL rows of 160),
   `token_embd` row gather (Q3_K), norms and conv weights (F32/F16).
2. **Block kernels**: DeltaNet, attention + indexer, MoE, hyper-connections, PLE,
   checked against cx-flash per-layer dumps on identical inputs.
3. **One die, 6 blocks**: persistent loop, grid barriers, correct state after 6 blocks.
4. **8-die pipeline**: mailboxes, feedback, sampling, greedy decode end to end.
5. **MTP**: draft, verify, rollback.
6. **Prefill and tuning**: register banks, reuse flags, stall counts, layer split.

## Verification

- Reference: cx-flash `recipe run qwen4exp.rs` on the same files for per-layer
  dumps and greedy tokens (`RECIPE_REFERENCE_WRITE`/`RECIPE_REFERENCE`).
- GEMV paths bit-exact against the C reference; block outputs within fp32 noise.
- 3 fixed prompts × 256 greedy tokens identical to the reference.
- Launch count: exactly 8 `cuLaunchKernel` per response (LD_PRELOAD counter).
- Speed from ring-buffer timestamps; MTP acceptance rate reported separately.

## Files

Sentry (192.168.1.20, not the .212 in the wake-device skill table) exports
`/data/models/flash-next-q2` read-only to archy; remounted on archy at
`/mnt/sentry-nfs/flash-next-q2` (NFSv3, ~70 MB/s, so a full cold load is ~16 min).
It holds both Q2_0 shards, `mtp-Qwen3.8-Flash-Next-Q8_0.gguf` (4.14 GB), the
UD-Q2_K_XL set, and a CUDA 11.8 toolkit. The mount is not in fstab; the loader
remounts it when missing (`mount -t nfs -o ro,vers=3,hard,rsize=1048576`).

## Design decisions taken in Phase 1

- **Repacked layout**: GGUF blocks are 18 B (Q2_0, Q4_0, IQ4_NL), so `qs` is only
  2-byte aligned and LDG.128 is impossible in place. At load the host permutes
  each row into `[fp16 d × B, padded to 16 B][16 B qs × B]`; rows padded to 16 B.
  Same bits, aligned loads. Every integer format gets the analogous row tile.
- **Q2_0 inner loop** (per 32-bit word = 16 weights): one LOP3 isolates two weights
  at the same bit position s in the two halves (mask `3<<s | 3<<(s+16)`), then two
  XMAD.U16.S16 multiply them by their int16 activations (H0/H1 selected by s) into
  the shift accumulator `acc_s += (q<<s)·a`. 8 accumulators per half-block, combined
  as `Σ_s acc_s >> s` (exact: each is a multiple of 2^s). The (q−1) offset is
  `− Σa` per half-block with Σa precomputed when the activations are quantised.
  Cost ≈ 1.5 issue slots per MAC (+ ~0.5 overhead) at full rate; measured XMAD
  rate is 123.7/clk/SM, so compute is roughly half the memory time of a Q2_0 matvec.
  A 2-terms-per-XMAD SWAR dot was checked and rejected: with int8 activations the
  low garbage field (≤765) overlaps the dot field at the 2^8 spacing XMAD's 16-bit
  operands allow; it only works for ≤7-bit activations.
- **Lane mapping** (K = 2560): a warp handles 2 rows, 16 lanes per row, 5
  half-blocks (32 weights, 8 B) per lane, so each lane keeps its 160 activations
  in 80 registers across all rows of the matvec (no shared-memory traffic).
  K = 640 (expert down): 8 lanes per row, 4 rows per warp, 20 int16 pairs... chosen
  per shape by the same rule (half-blocks / lanes per row integral).
- **Bit-exact definition**: per half-block int32 dot exact; fp32 accumulation is
  `acc = fma(d_b, float(dot − Σa), acc)` sequentially over a lane's half-blocks,
  then a SHFL butterfly (xor 8,4,2,1) in fp32, then × activation scale. The C
  reference performs the identical sequence with `fmaf`, so GPU and CPU outputs
  compare with `==`.

## Open questions

1. BF16 tensors (1.47 GB; hc mats, router, indexer, ssm α/β): keep as BF16 with
   FFMA (bit-exact, slower) or requantise to Q8_0 at load (integer, ~0.6 GB/token
   less traffic)? Default: keep.
2. IQ4_NL/IQ4_XS use a 16-entry int8 level table (PRMT-selected, no float
   dequant). Acceptable, or requantise those 13 + 56 tensors to Q4_0?
3. Die 7 carries the head and MTP and clocks lower: fewer blocks there is fine?
4. Prefill inside the same single dispatch (default), or a separate prefill
   dispatch per die?
5. Context target for v1: 32k?
