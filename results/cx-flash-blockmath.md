# cx-flash math for Qwen3.8-Flash-Next, one decode step

## 0. Sources and caveats

- `/home/nate/work/cx-flash` does not exist on engi. Its clone lives on archy (`/home/nate/work/heads/cx-flash-prompt.txt`). I used two engi copies of nm-z/recipe-dev:
  - **B (primary):** `/home/nate/work/cx-fl-exchanges-1092/` (worktree of `cx-fl-split-local`, branch `feature/flash-layer-exchanges-1092`). It has MTP, and its `qwen4exp.rs` loads your Q2 shards. Cited as `B recipe.rs:N` / `B amd-nv-cpu.ll:N` (IR library).
  - **A (cross-check):** `/home/nate/work/cx-recipe-dev-1089/` (clone, branch `cx/1089-fix`). It has the `[architecture.qwen4exp]` policy and a `hyper_gate`/`ple_math` DSL, but no MTP. Cited as `A ...`.
- Shapes and tensor names come from the Q2 inventory (`/home/nate/work/cx-flash-local/tensors.tsv`, `mtp-tensors.tsv`, produced on archy). GGUF metadata values are not verified because NFS is not mounted on engi. Read them with `recipe keys` on archy.
- Rule note: `cx-flash-local/AGENTS.md` forbids scanning `/home/nate` or `/home/nate/work`. Before I found that file I ran one `find /home/nate -maxdepth 4` and listed `/home/nate/work`. I have not scanned since. I wrote nothing in any repo and built nothing.

Verified shapes: hidden 2560; 4 hyper lanes (stream 10240); hc low rank 320; vocab 248320; 48 blocks = 36 Gated DeltaNet + 12 full attention (`blk.3, 7, ..., 47`), plus MTP block 48. DeltaNet: 16 key heads, 48 value heads, state 128, conv kernel 4, inner 6144. Attention: 24 query heads and 2 KV heads, head dim 256. Indexer: 4 heads of 128. MoE: 512 experts, expert hidden 640, shared hidden 640. PLE table `per_layer_token_embd.weight` is `[160, 320001536]` (iq4_nl), and PLE exists only in `blk.1`.

## 1. Key findings (check these first)

1. **Key-head mapping differs by path.** B always tiles: value head h reads key head `h mod 16` (B `lower_delta` sets `argument[6]=1`, B recipe.rs:22394-22431; B `delta_forward_body` uses `urem`, B amd-nv-cpu.ll:2993-3007). A's inference body (`delta_live_body`) defaults to grouped `h / 3` while A's training body is tiled (A amd-nv-cpu.ll:4161, 4210). Confirm the convention on layer 0 against llama.cpp. llama.cpp's `ggml_repeat` tiles, and HF's `repeat_interleave` groups.
2. **B applies the readout scale twice.** B's `delta_parameters` returns `query.scale = 1/sqrt(128)`, which `delta_step` multiplies into the readout (B amd-nv-cpu.ll:2861-2866, 2980). B `lower_delta` then applies `lower_scale(1/sqrt(128))` after the Delta node (B recipe.rs:22420). Net factor 1/128. A applies it once (A recipe.rs:23046). The following RMS norm is scale-invariant except through ε, so the error is small but nonzero. The usual convention (FLA's `chunk_gated_delta_rule`) scales q once by `1/sqrt(d_k)`.
3. **PLE acts on a 4-lane stream.** It sits at the top of layer 1, after layer 0 has expanded the stream. The gate is one scalar per lane from a 2560-wide dot product. The 160-wide table rows only feed the 2560-wide projections. The 16-by-160 concatenation is the gathered input, not a set of gates.
4. **MTP's hidden input is the head-normalized stream**, not the raw residual stream. `retain_output` lands on the RMS node that feeds the head's `Read` (B recipe.rs:16546-16555). MTP's `hnorm` then normalizes it a second time.
5. Hyper mean: A applies `1/lanes` as an explicit scale after `Read`, and B folds it into the `Read` body (B amd-nv-cpu.ll:2684-2720). The branch input is the same either way.

## 2. Hyper-connections and block order

Stream S ∈ R^{4×2560}. Layers L = 0..47 each run a mixer hyper-block (DeltaNet or attention, weights `blk.L.hc_attn_*`), then an FFN hyper-block (MoE, `blk.L.hc_ffn_*`). Layer structure: B `qwen4exp.rs:9-25`. No pre-norm or post-norm appears in the blocks (A Cargo.toml 468-500: `attn-pre-norm=false`, `ffn-pre-norm=none`).

- **Embedding and expansion:** x = `token_embd[id_t]` (q5_K, 2560). The first hyper-block expands `S[l] = x` for all four lanes (`Expand`, B `lower_hyper` recipe.rs:23131-23170; B amd-nv-cpu.ll:2664).
- **Hyper-block with branch width 2560:**
  - `x̂[l] = RMS_2560(S[l]) ⊙ hc_norm`, where `hc_norm` has 10240 weights and the RMS is per lane (B `lower_gates` recipe.rs:23171-23195).
  - Read gate: `g = σ( W_up · SiLU( 0.25 · W_down · x̂ ) )`, with `W_down` 10240→320 (`hc_*_down`) and `W_up` 320→10240 (`hc_*_up`). The 0.25 is `1/lanes`.
  - `z = 0.25 · Σ_l g[l] ⊙ x̂[l]` ∈ R^2560 (`Read`, mean over lanes).
  - `y = Branch(z)`, where Branch is DeltaNet, attention, or MoE.
  - Write gate: `w[l] = 2 · σ( 0.25 · (W_inj x̂)[l] )`, with `W_inj` 10240→4 (`hc_*_inject`, no low rank).
  - `S[l] ← S[l] + w[l] · y` (`Outer`, then `Add` onto the pre-gate stream; B `lower_hyper` 23131-23170).
- **Head:** the same read gate runs on the final stream with `output_hc_norm/down/up` (no inject): `h = 0.25 · Σ_l g^o[l] ⊙ x̂^o[l]` (B `lower_collapse` recipe.rs:23237; B `head_planes` 18438). There is no output norm (`output-norm = none`). `logits = output.weight · h`, with `output.weight` q4_K of shape 2560×248320. A soft-cap `cap·tanh(logits/cap)` applies only if `qwen4exp.final_logit_softcapping` exists (B recipe.rs:17144-17147).

## 3. Gated DeltaNet (36 layers)

Input z (2560), per token:

1. `qkv = W_qkv z` ∈ R^10240 = `[q 16×128 | k 16×128 | v 48×128]` (`attn_qkv`; B `delta_planes` 17388-17420).
2. Depthwise causal conv, kernel K = 4, no bias, over all 10240 channels: `y_c(t) = Σ_{k=0..3} w[c,k] · x_c(t−3+k)`, with zero before the sequence start. Weight index is `c·4 + k` and the oldest tap comes first. The state keeps the last 3 inputs per channel (B `dconv_forward_body` amd-nv-cpu.ll:2784; `lower_dconv` B recipe.rs:22269).
3. `u = SiLU(y)` on all 10240 channels, before the q/k/v split (B `lower_delta` 22394-22431).
4. `q̂ = q / max(‖q‖₂, ε)` and `k̂ = k / max(‖k‖₂, ε)` per 128-vector (L2 with an ε floor; A `NormalizeMode::L2`, A recipe.rs:7036-7046).
5. Gates: `a = W_α z + ssm_dt.bias` (48), `b = W_β z` (48). The bias offsets only the alpha half (A `delta_planes` recipe.rs:18189-18207).
   - `g_h = −exp(A_h) · softplus(a_h)`, where `A_h = ln(−ssm_a_h)`, because the file stores `ssm_a = −exp(A)`.
   - Decay `α_h = exp(g_h)`, write `β_h = σ(b_h)`.
   - `softplus(x) = max(x,0) + log(1 + exp(−|x|))` (A recipe.rs:22593-22600). A `lower_delta_gates` 22987-23013.
6. Recurrence per value head h, with key head kh(h) from Finding 1. State S ∈ R^{128×128}, with `S[i][c]` where i is the key index and c the value index. It is held in f64 in A (`double` buffers):
   ```
   S ← α_h · S                              decay in place
   r_c = Σ_i k̂_i · S[i][c]                  read
   e_c = v_c − r_c                          error
   S[i][c] ← S[i][c] + β_h · k̂_i · e_c      write
   o_c = Σ_i q̂_i · S[i][c]                  readout (after the write)
   ```
   A `delta_column` amd-nv-cpu.ll:4073-4120; B `delta_step` 2911-2990, same order. The chunk size (`RECIPE_DELTA_CHUNK`, 8) only controls state checkpoints, not the arithmetic (B amd-nv-cpu.ll:2993 comment).
7. Readout scale `o ← o / sqrt(128)` (see Finding 2).
8. `ssm_norm`: `ô = o / sqrt(mean_128(o²) + ε) ⊙ ssm_norm`, with 128 weights shared by all heads (A `lower_normalize` recipe.rs:23193; RMS emitter 7036-7046).
9. Output gate: `ô ⊙ σ(W_g z)`, where `W_g` is `attn_gate` 2560→6144. The gate is **sigmoid, not SiLU** (`delta_activations(Silu, Sigmoid)`: A `qwen4exp.rs:39`, B `qwen4exp.rs:19`). Then `ssm_out` 6144→2560.

## 4. Full attention (12 layers, L = 3, 7, ..., 47)

1. `z → attn_q` 2560→12288 = 24 heads × `[q_i (256) | g_i (256)]`. Head i owns rows `i·512 .. i·512+511`: q in the first 256, gate in the next 256. `attn_k` and `attn_v` are 2560→512 (2 KV heads × 256). No biases (A `attention_planes` recipe.rs:19328-19360; A `attention_gate_planes` 19427).
2. Per-head RMS norm on q and k, 256 weights each, shared over heads, with ε. v is not normalized (A `lower_attention` recipe.rs:23059-23075).
3. **RoPE, NeoX/halves**, on the first `d_r` channels of each q and k head (`d_r = rope.dimension_count`, default 256 if absent). For `p < d_r/2`, with `θ_p = base^(−2p/d_r)` and `a = pos · θ_p`:
   `x'_p = x_p·cos a − x_{p+d_r/2}·sin a`, and `x'_{p+d_r/2} = x_{p+d_r/2}·cos a + x_p·sin a`.
   `base = rope.freq_base` (default 10000). No rope factors (`rope-factors=none`). A `rope_pair` amd-nv-cpu.ll:3711, `rope_body` 3793-3835; B `rope_body` 2579.
4. `score = q·k / sqrt(256)` (A amd-nv-cpu.ll:6597-6601; B 5385). Query head i reads KV head `i / 12` (A 6602-6603). Softmax over admitted causal keys, then `out_i = Σ_j p_ij · v_j`. Online and full softmax differ only in rounding order.
5. `out_i ← out_i ⊙ σ(g_i)` (A `qwen4exp.rs:33-34`, `heads * gate`; in B the gate is an Attention-node flag, IR at B amd-nv-cpu.ll:5352-5359).
6. `attn_output` 6144→2560.

### 4a. Sparse indexer (every attention layer)

- From the block input z: `iq = W_iq z` (512 = 4×128) and `ik = W_ik z` (128). Tensors are `blk.L.indexer.{q_proj,k_proj}.weight` (bf16), with `q_norm` and `k_norm` of 128 each (A `indexer_planes` recipe.rs:19294-19316; B 18510).
- **Query:** `iq_h ← RMS128(iq_h) ⊙ q_norm`, then RoPE on the first `d_r` channels at position t, with the attention block's base (A `lower_attention` 23146-23175). The auto-score rule enables this when the norms exist (A 19084-19085; B 18344).
- **Block size** B = `attention.compress_ratios[L]` (max with 1).
- **Representative** of block b, pooled over its keys j ≤ t: `m = Σ_j ik_j / count`, then `r_b = RoPE_{pos = b·B}( RMS128(m) ⊙ k_norm )`. The order is pool, norm, rope, and the raw keys are pooled first (A `attention_index_cache` amd-nv-cpu.ll:5806-5942).
- **Score:** `S_{t,b} = Σ_{h=0..3} ReLU(iq_{t,h} · r_b)`, over blocks whose start is ≤ t. The block containing t uses the prefix representative (A `attention_select_body` 6057-6160, ReLU at 6154; B amd-nv-cpu.ll:4933).
- **Admission:** `keep = ceil(top_k / B)` (A `index_tokens` recipe.rs:14293-14295). Block b is admitted iff fewer than `keep` blocks outrank it: higher score, or equal score with a lower index (A amd-nv-cpu.ll:6200). Attention then covers every key j ≤ t inside admitted blocks.
- `top_k` = `attention.indexer.top_k` (a token budget). The compress ratio comes from `attention.compress_ratios[L]`.

## 5. MoE (48 layers, branch input z')

1. `p = softmax over 512 experts of (W_r z')`, where `W_r` is `ffn_gate_inp` 2560×512 (f32, no bias). `expert_gating_func`: 1 = softmax (default), 2 = sigmoid.
2. Top-10 by p, ties to the lower id. Renormalize: `w_e = p_e / Σ_{top-10} p` (`expert_weights_norm`, default true). `topk_score = exp(s − max)`, and the divisor is the kept total when renormalizing (B amd-nv-cpu.ll:3387-3394; A 4486).
3. Expert e: `y_e = W_down,e( SiLU(W_gate,e z') ⊙ (W_up,e z') )`. Gate and up are 2560→640, down is 640→2560, and tables are stored `[expert][hidden][channel]` (`ffn_gate_exps` shape 2560×640×512). `routed = Σ_{top-10} w_e · y_e`, with expert scale 1.0 (B `expert_in` 3596, `expert_out` 3689; A `lower_experts` recipe.rs:23389-23417).
4. Shared expert: `s = σ(ffn_gate_inp_shexp · z')` (a scalar), `y_s = W_down,s( SiLU(W_gate,s z') ⊙ W_up,s z' )` with width 640, and `out = routed + s · y_s` (A `lower_gguf_moe` 23418-23462; B 22837-22870). B takes the shared width equal to the routed width, which holds for this file (both 640).
5. No pre-norm or post-norm, and no selection bias (`exp_probs_b` is not bound for qwen4exp).

## 6. PLE (layer 1 only)

`ple.layers = [1]`. The inventory has `blk.1.ple_*` and no PLE tensors on other blocks.

- **Row addressing:** context `c_b = id_{t−b}` for `b = 0..ngram−1`, replaced by `absent` (the EOS id) if the position is before BOS or an EOS lies in between. For order `n = 2..ngram`: `mixed_n = XOR_{i<n} (c_i · layer_multipliers[i])`, with wrapping u64 multiplies. Heads `h ∈ [(n−2)·P, (n−1)·P)`, with `P = heads_per_ngram`, read row `head_offsets[h] + (mixed_n mod head_vocab_sizes[h])`. Order-2 heads come first (B `RowHash::rows_at` recipe.rs:11287-11300; B `Ngram` 11384).
- **Gather:** `g` is the concatenation of the 16 rows, 16×160 = 2560 values, host-gathered from `per_layer_token_embd.weight`.
- **Block at the top of layer 1**, on the 4-lane stream S:
  ```
  K   = W_k g                                ple_key 2560→10240
  k̂   = RMS_2560_per_lane(K) ⊙ ple_norm_key
  Q   = RMS_2560_per_lane(S) ⊙ ple_norm_query
  s_l = (1/sqrt(2560)) · Σ_{c in lane l} k̂_c · Q_c           one scalar per lane
  γ_l = σ( sgn(s_l) · sqrt(max(|s_l|, 1e-6)) )
  V   = W_v g                                ple_value 2560→2560, no norm
  G_l = γ_l · V                              V broadcast over lanes
  out = G + SiLU( DConv_{K=4, dilation=ngram}( RMS_2560(G) ⊙ ple_norm_conv ) )   ple_conv1d [4, 10240]
  S ← S + out
  ```
  A: `qwen4exp.rs:18-24` (`ple_math`), `lower_ple` recipe.rs:22874-22937. B: `lower_ple` 22281-22338 (same math, hard-coded).

## 7. Output, sampling, MTP

**Sampler.** `recipe.sampler()` defaults to temperature 1.0, top_k 0, top_p 1, min_p 0, repeat penalty 1 over a 64-token window (B recipe.rs:19107-19108). The placed decode path sets `temperature(0.0)`, which is greedy (B 18165). The serve path keeps 1.0 unless the request sets it (B 19236-19242). Confirm which path your entry point takes.

**MTP (block 48).** Built in `MtpHead::open` (B 16367-16470). The input for the draft after token t:

- `e = RMS_2560(token_embd[next]) ⊙ nextn.enorm`.
- For each lane l: `u_l = RMS_2560(S_l) ⊙ nextn.hnorm[l]`, where `S` is the retained head-normalized stream from Finding 4 and `hnorm` has 10240 per-lane weights.
- `lane_l = W_eh · [e ; u_l]`, where `eh_proj` is 5120→2560. The same `W_eh` applies to every lane (B `lower_mtp_join` 23203-23236).
- Block 48 is a normal hyper-block: gated attention and MoE, with FP32 for qk, rope, and experts (B 16418-16420), and its own `blk.48.hc_*` gates.
- Head: `nextn.hc_head_{norm,down,up}` (the same form as the main head), then its own `output.weight` (q8_0). It uses the target's `output.weight` when `nextn_shared_target_tensors` is true.

**Draft and verify** (B speculative loop 16895-16975). Propose K greedy drafts `d_1..d_K`. Each draft is produced by the MTP head, fed the previous draft and its own hidden output. K is `RECIPE_MTP_DRAFTS` if set, else `[package.metadata.mtp] tokens` = 3 (B Cargo.toml 63-65). `probs = 0` disables the confidence gate. Optional n-gram lookup drafts are tried first (`RECIPE_MTP_LOOKUP`, B 16788). The main model then runs one window over `[next, d_1..d_K]`. Draft `d_{i+1}` is accepted iff it equals the main model's choice at that position. Acceptance stops at the first mismatch, and rejected state is rolled back via `commit_mtp_prefix` (draft positions ≤ 4). Output matches greedy decoding exactly.

## 8. Numerics

- **Default profile `recipe`** (A Cargo.toml 144-163; B Cargo.toml:134): `sum=int8`, `embed/attn/rope/kv/atvn/norm/res = fp16`, `acc=fp32`, `train=fp32`, `rope-angle=direct`, `attn-softmax=full`, `math=portable`, `tolerance 0.05`, `step 32`.
- **Profile `llamacpp`** (A Cargo.toml 184-203): fp32 almost everywhere, `kv=fp16`, `norm-acc=fp64`, `rope-angle=chain`, `attn-softmax=online`, `math=libm`, `gelu-table=fp16`, `step 256`. Select with `RECIPE_CONFIG=llamacpp`.
- **Forced FP32:** routed and shared experts (`.fp(32)`, B `qwen4exp.rs:24`) and the MTP block (B 16418-16420).
- **Delta state:** A holds the state, gates, and planes in f64 (A amd-nv-cpu.ll:4073-4200; A recipe.rs:23002-23003). B uses the `RECIPE_STATE` type, whose precision per profile I did not verify.
- **ε:** GGUF `qwen4exp.attention.layer_norm_rms_epsilon` (A recipe.rs:17809-17810, 18426-18429), else 1e-5 (B Cargo.toml:105). It is used inside the RMS sqrt, as the L2 floor, in indexer norms, and in PLE norms.
- **Activation quantization** (packed-weight contractions under int8 sums): each 32-value block is quantized (B recipe.rs:7645-7700, device quantizer): `d = max|x|/127`, `q = clamp(round_half_away(x/d), −128, 127)`, `x' = d·q`. Weights are unchanged. I did not verify whether the f32/bf16 tensors (norms, `ssm_alpha/beta`, router, `hc_inject`, indexer) skip this path.
- **Clamps:** L2 floor ε, PLE gate floor 1e-6, stable softplus, max-subtraction in softmax. No attention logit cap.

## 9. Debug and verification

- **Run on archy only.** The repo's AGENTS forbids compiling or running on engi. B `qwen4exp.rs` reads the `GGUF` env var (default `/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf`) and the `MTP` env var (B 34). `--memory` prints the plan (B 35-41). Otherwise it prints `generation tokens N rate R tok/s`. Device syntax: A README.md line 8, `--device archy:nv0.nv1`.
- `.log([debug])` writes a per-dispatch trace to `RECIPE_TRACE_PATH` (default `<repo>/recipe.log`; B 12744-12751, 17814). `.log([hc_values])` exposes `hc_mixed`, `hc_inject`, and `hc_combine` (B 23145, 23162, 23189, 23243).
- **Node dumps** (B 25436-25495): `RECIPE_TRACE_NODES=a..b`, `RECIPE_TRACE_DUMP=a..b`, `RECIPE_TRACE_CHANNELS=c1,c2` (3 values per channel at the first and last positions), `RECIPE_TRACE_ROW=p|all` (full row), `RECIPE_TRACE_POSITIONS=1`.
- **Reference logits:** `RECIPE_REFERENCE_WRITE=path` records every step's f64 logits (magic `RCPREF01`, plus the winner). `RECIPE_REFERENCE=path` compares under the profile tolerance or exactly (A reference.rs 1-275). The two are exclusive.
- **Expert routing:** `RECIPE_EXPERT_PICK_LOG` (B 19795) and `RECIPE_EXPERT_PICK_COUNTS` (B 19577).
- **Tensor names:** `blk.L.attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`, `ssm_norm`, `ssm_out` (DeltaNet); `attn_q`, `attn_k`, `attn_v`, `attn_output`, `attn_q_norm`, `attn_k_norm`, `indexer.*` (attention); `ffn_gate_inp`, `ffn_{gate,up,down}_exps`, `ffn_*_shexp`, `ffn_gate_inp_shexp` (MoE); `hc_{attn,ffn}_{norm,down,up,inject}` (hyper, 10240 norm, 320 rank, 4 inject); `output_hc_{norm,down,up}` (head); `ple_{key,value,norm_key,norm_query,norm_conv,conv1d}` (PLE); `output.weight`, `token_embd.weight`, `per_layer_token_embd.weight`; MTP: `blk.48.nextn.{enorm,hnorm,eh_proj,hc_head_*,shared_head_norm}`.

### GGUF keys read (prefix `qwen4exp.`; B recipe.rs:17154-17260)

Values in parentheses are inferred from the tensor inventory. The rest need archy.

- `block_count` (48), `embedding_length` (2560), `attention.head_count` (24), `attention.head_count_kv` (2), `attention.key_length` (256), `attention.value_length`, `attention.sliding_window` (0), `full_attention_interval` (4), `rope.dimension_count` (default 256), `rope.freq_base` (default 10000), `rope.scaling.*` (YaRN, optional), `attention.layer_norm_rms_epsilon` (default 1e-5).
- `ssm.time_step_rank` (48), `ssm.group_count` (16), `ssm.state_size` (128), `ssm.conv_kernel` (4), `ssm.inner_size` (6144).
- `expert_count` (512), `expert_used_count` (10, per your note), `expert_feed_forward_length` (640), `expert_gating_func` (1), `expert_weights_norm` (true), `expert_shared_feed_forward_length` (640).
- `hyper_connection.count` (4), `hyper_connection.low_rank` (320).
- `attention.indexer.head_count` (4), `attention.indexer.key_length` (128), `attention.indexer.top_k` (token budget), `attention.compress_ratios` (per-layer array).
- `ple.ngram_size`, `ple.heads_per_ngram`, `ple.eos_token_id`, `ple.layer_multipliers`, `ple.head_offsets`, `ple.head_vocab_sizes`, `ple.image_token_id` (optional), `ple.layers` ([1]), `ple.conv_kernel` (4). A seeded variant (`ngram.heads`, `ngram.table`, `ngram.seeds`) is a different layout (B `Ngram::layout` 11384).
- MTP file: `nextn_predict_layers` (1) and `nextn_shared_target_tensors`. Optional: `final_logit_softcapping`.
- Dump all of them with `recipe keys <gguf>` and `recipe conventions <gguf>` (A README.md 311-321).

## 10. Gaps and caveats

- Metadata values (rope base and dims, ε, `top_k`, `compress_ratios`, ngram size and heads per order, multipliers) are unverified.
- Nothing was run or built. All line numbers come from reading the files, and approximate ranges are marked as such.
- The head uses its own `output_hc_*` gate, not the last block's `hc_ffn` gate.
- Finding 1 (key-head mapping) and Finding 2 (double readout scale) need a llama.cpp reference check before you port them. I did not run either.
