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

## Block math (from cx-flash `recipe.rs`/`qwen4exp.rs`, to be verified numerically)

- **Hyper-connections** (around both the mixer and the MoE): `xn` = per-lane
  RMSNorm(stream)·`hc_*_norm`; read gate `r = σ(W_up · SiLU(W_down · xn/4))`;
  write gate `w = 2σ(W_inject · xn/4)` per lane; branch input = read-gated sum of
  lanes; update `stream_l += w_l ⊙ Branch`.
- **DeltaNet** (36 blocks): qkv 2560→10240 (q 16×128, k 16×128, v 48×128), causal
  depthwise conv k=4 + SiLU, L2-normalise q and k per head; per value head with
  S∈R^{128×128}: `d = exp(a·softplus(α + dt_bias))`, `w = σ(β)`,
  `S ← d·S; r = k̂ᵀS; S ← S + w·k̂(v−r)ᵀ; o = (q̂/√128)ᵀS`; then RMSNorm(`ssm_norm`)
  ⊙ σ(`attn_gate`·x), `ssm_out`. Open: cx-flash may apply 1/√128 twice.
- **Full attention** (12 blocks): q 2560→12288 (q + gate), k/v 2560→512, q/k
  RMSNorm 256, neox RoPE on 64 dims, GQA 24:2, head dim 256, `attn_output`
  6144→2560. Indexer: q 2560→512 (4×128), k 2560→128, RMSNorm 128, blocks of 4
  keys, keep the top 2048 tokens (512 blocks).
- **MoE**: router BF16 2560→512, softmax, top-10, renormalise; expert
  `W_down(SiLU(W_gate x) ⊙ W_up x)`; shared expert 640-wide scaled by
  σ(`ffn_gate_inp_shexp`·x).
- **PLE** (block 1): gathered e (2560) → `key = RMSNorm(W_k e)`,
  `query = RMSNorm(stream lane)`, `s = Σ key⊙query / √2560`,
  `g = σ(sign(s)·√max(|s|,1e-6))`, `v = W_v e`, out = `stream + g⊗v +
  SiLU(DConv_{k=4,dil=3}(RMSNorm(g⊗v)))`.
- **Head**: `output_hc_*` collapse → RMSNorm → `output.weight`.
- **MTP**: RMSNorm(embedding) ‖ per-lane hnorm → `eh_proj` 5120→2560 → one block
  → shared head.

## Numerics

- Activations quantised per row to int8 + fp32 scale before each integer matmul.
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
  (launch-bound at 13 µs). Roofline is ~160 GB/s; profiling and scheduling next.
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

## Phases

0. **Probes** — done (table above).
1. **GEMV kernels**: Q2_0 K=2560 done and bit-exact; next Q2_0 K=640, then
   BF16 (biggest per-token bytes), Q3_K, IQ4_XS, Q4_K, Q5_K, Q6_K, the block-32
   formats. Each bit-exact against the C reference on real tensors.
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
