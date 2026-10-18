# flash-sass
An inference engine for **Qwen3.8-Flash-Next**, specifically the Q2 variant of: 
https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
Using **Maxwell SASS** (via [MaxAs](https://github.com/NervanaSystems/maxas)) for multiple **Tesla M60** GPUs. By default each die runs **one persistent kernel per response** Weights stay in the GGUF's own block formats and are dotted as integers with packed `XMAD`.

- row split
- expert split

## Measurments on 4 M60 cards, 6 dies used
- The full 48-block model (DeltaNet, attention + indexer, 512-expert MoE, hyper-connections, the
  n-gram "PLE" table read from pinned host RAM) as one program of ~1150 phases per token.
- Greedy output identical to upstream llama.cpp on the reference prompts used (every block type is
  also bit-exact against a C twin in `ref/`).
- **26.7 ms per position (about 37 tok/s)** for prompt and decode alike; expert and row split over
  the six dies, with the dies exchanging activations by peer-to-peer stores.
- In-kernel sampling with the model's own defaults (top-k 20, top-p 0.95, temperature 1.0), stop on
  end-of-turn, stop on client disconnect, and a prefix cache for multi-turn chats.
- A persistent host process (`flashcore`) that keeps the weights and the 28.8 GB table loaded while
  the engine library is reloaded, `flashd` (a llama-server-style HTTP/SSE front-end that renders the
  GGUF's own Jinja chat template), and a tokenizer written from the GGUF's metadata.

An opt-in **two-position pass** pairs known prompt positions with one synchronization and
publication per logical phase. Set `FLASH_TWO_POSITION=1` when starting or reloading the
engine. Five BF16 variants share weight loads; remaining operations replay their scalar
arithmetic with separate position buffers. The final prompt token and generation stay B1.
See [BATCH.md](BATCH.md) and [PAIR_TESTING.md](PAIR_TESTING.md).


#### Planned: 
- MTP head
- unified KV cache
- general matrix-batched prefill
- concurrent requests
- V340L version

## What one token runs

| Kind | What it does | CONFLICT TEST | µs per phase | ms per token (all phases of this kind) | exchanges it carries |
|---|---|---|---|---|---|
| **exchange and quantize** | | | | **6.26** | |
| `xsend` | send this die's slice to its peers (P2P stores, fence, flag) | 111 / 111 | 12.8 | 1.42 | 50 projection, 49 contraction, 13 broadcast (112 sent; the profile times 111) |
| `actq256` | quantize an activation to int8 + scale; where a die waits for its peers | 146 / 146 | 33.1 | 4.84 | |
| **hyper-connection** | | | | **8.26** | |
| `bf16hcrwx` | RMSNorm of the stream + hc down and inject (BF16), plus the pending stream write | 48 / 48 | 40.3 | 1.93 | 48 projection |
| `bf16hcrx` | the same without the pending write | 49 / 49 | 32.0 | 1.57 | 49 projection |
| `bf16hcup` | SiLU + hc up + lane mix + inbox write | 97 / 97 | 25.3 | 2.46 | 97 projection |
| `hcwsum` | stream write with the MoE partials' fixed-order sum fused in | 48 / 48 | 47.9 | 2.30 | |
| **MoE** | | | | **5.14** | |
| `router` | router GEMV and softmax | 48 / 48 | 23.3 | 1.12 | |
| `moesel` | top-10 expert selection | 48 / 48 | 5.6 | 0.27 | |
| `q2moegn` | expert gate and up GEMVs (Q2_0), this die's share of the experts | 96 / 96 | 18.4 | 1.77 | |
| `ffnact` | SiLU(gate) · up, quantized for the down projection | 48 / 48 | 13.0 | 0.62 | |
| `q2moedn` | expert down GEMV (Q2_0) | 48 / 48 | 21.2 | 1.02 | |
| `q2k2560`, `q2acc640`, `b32q40acc`, `b32q50acc`, `b32iq4nlacc` | shared-expert and other small matrices in their own formats | 20 / 24 | 15–26 | 0.34 | |
| **projections (GEMV by weight format and input length K)** | | | | **6.61** | |
| `q3kk2560` | Q3_K, K = 2560 | 50 / 49 | 34.2 | 1.71 | |
| `bf16k2560` | BF16, K = 2560 | 49 / 53 | 27.8 | 1.36 | |
| `iq4k2560`, `iq4k6144` | IQ4_XS, K = 2560 / 6144 | 19 + 17 / 22 + 17 | 42.8 / 33.1 | 0.81 + 0.56 | |
| `q4kk2560`, `q4kk6144` | Q4_K, K = 2560 / 6144 | 8 + 15 / 11 + 15 | 35.3 / 29.4 | 0.28 + 0.44 | |
| `q5kk2560`, `q5kk6144` | Q5_K, K = 2560 / 6144; one `q5kk2560` is the share of the 248,320-row output head (0.69 ms, the longest phase) | 1 + 7 / 4 + 7 | 688 / 41.0 | 0.69 + 0.29 | |
| `q6kk1536`, `q6kk1536acc` | Q6_K, K = 6144 as four K = 1536 slices summed in a fixed order | 6 + 18 / 6 + 18 | 18.3 / 13.6 | 0.11 + 0.24 | |
| `q3kk6144` | Q3_K, K = 6144 | 3 / 3 | 35.2 | 0.11 | |
| **DeltaNet (36 blocks)** | | | | **1.53** | |
| `bf16bias` | BF16 GEMV with bias | 36 / 36 | 7.0 | 0.25 | |
| `conv1d` | causal conv (k = 4) + SiLU | 36 / 36 | 5.7 | 0.20 | |
| `dnstep` | the recurrence step (state update, read-out) | 36 / 36 | 18.1 | 0.65 | |
| `dnnormx` | output norm and gate, quantized | 36 / 36 | 11.8 | 0.42 | 36 projection |
| **attention (12 layers, 4 on each attention die)** | | | | **0.48 on an attention die** | |
| `attnqkq` | q/k norm, RoPE, KV append | 0 / 4 | 6.1 | 0.02 | |
| `idxqk`, `idxscore` | indexer q·k and block scores | 0 / 4 + 4 | 5.6 / 4.4 | 0.04 | |
| `idxsel` | the five top-k selection passes of the indexer | 0 / 20 | 3.9 | 0.08 | |
| `kvgather` | gather the admitted KV blocks | 0 / 4 | 18.9 | 0.08 | |
| `attnpartn`, `attnmerge` | partial attention over the gathered keys, merge | 0 / 4 + 4 | 57.1 / 9.0 | 0.23 + 0.04 | |
| **embedding, PLE step, picking the token** | | | | **0.15** | |
| `xcopy` | copy the embedding stream that arrived from the die holding `token_embd` | 1 / 1 | 33.8 | 0.03 | |
| `plerows` | PLE: the 16 n-gram rows from the host (waits until the host has read them) | 1 / 1 | 57.3 | 0.06 | |
| `rms2560` | PLE norms of the key, the query and the conv input | 3 / 3 | 8.2 | 0.02 | |
| `plegate` | PLE gates | 1 / 1 | 6.1 | 0.01 | |
| `pleconv` | PLE conv over the last 4 tokens | 1 / 1 | 9.2 | 0.01 | |
| `argmax` | this die's largest logit and its index, over its share of the output head (greedy; the sampler has its own `topk` and `sample`) | 1 / 1 | 11.3 | 0.01 | |
| `amax` | the largest of the six dies' pairs: the token | 1 / 1 | 7.2 | 0.01 | |

Each token makes 342 exchanges, the same on every die: 280 projections (every die gathers the slices of a row-split result), 49 contractions (the MoE partial sums and the argmax pairs, summed across dies in a fixed order) and 13 broadcasts (the 12 attention outputs and the token embedding). The last column counts them; 230 are carried by the GEMV phases that produce the data and 112 by `xsend`.

The counts follow the model: 48 means once per block, 36 once per DeltaNet block, 4 once per
attention layer on a die that holds four. Each block pays one `router`, `moesel`, `ffnact` and
`q2moedn`, two `q2moegn` (gate, up) and two `bf16hcup` (plus one for the output collapse), about
three `actq256` and about two `xsend`. Sizes are in `SPEC.md` ("Phase sequence per block"; fusions
since then have merged some of those steps).

## Layout
| Path | Contents |
|---|---|
| `kern/*.sass` | the kernels: GEMVs per weight format, DeltaNet, attention, MoE router, hyper-connection ops, exchange, top-k and sampler |
| `kern/die.pl` | generator of the persistent kernel: phase dispatcher, grid barrier, token loop, exchange flags |
| `kern/prog.c`, `engine.c` | the per-token program builder and the reloadable engine library |
| `kern/flashcore.c`, `flashd.c`, `serve.sh` | the persistent core, the HTTP front-end, start/reload script |
| `kern/prof.sh` | per-phase profile of the last token (`PROF_ALL=1` lists every phase kind) |
| `kern/hold.c` | the weight holder (keeps the weights in GPU memory, shared by CUDA IPC) |
| `kern/tok.c`, `jinja.c`, `json.c` | tokenizer, chat-template interpreter, JSON |
| `kern/*_test.c` | tests of the kernels against their C twins |
| `ref/` | the CPU reference and the bit-exact C twins of the kernels |
| `maxas-patch/` | patches to MaxAs for CUDA 11.4 |
| `SPEC.md` | the design and the log of every measurement; `BATCH.md` is the plan for two positions per phase |

## Building
You need an sm_52 toolchain (CUDA 11.4's `ptxas` builds the kernel containers that MaxAs fills), a
MaxAs checkout with `maxas-patch/` applied, Perl, a C compiler, and the model files. Paths and
the six-die split (`SPLIT_SHARES`) are set for the author's machine; read `kern/Makefile`,
`kern/serve.sh` and `SPEC.md` before expecting it to run elsewhere.

```
make -C kern die.cubin libengine.so flashcore flashd hold
```

## Licence
MIT, see `LICENSE`. Derived parts and their notices are in `THIRD_PARTY.md`. No model weights are included.
