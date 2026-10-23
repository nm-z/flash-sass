# flash-sass
An inference engine for **Qwen3.8-Flash-Next**, specifically the Q2 variant of: 
https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF
Using **Maxwell SASS** (via [MaxAs](https://github.com/NervanaSystems/maxas) and https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF/blob/main/Qwen3.8-Flash-Next-DFlash2-Q4_K_M.gguf) for multiple **Tesla M60** GPUs. By default each die runs **one persistent kernel per response** Weights stay in the GGUF's own block formats and are dotted as integers with packed `XMAD`.

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

The **two-position pass** pairs known prompt positions with one synchronization and
publication per logical phase. `flashcore` defaults to `FLASH_PREFILL_CHUNK=64`, which
runs known prompt positions in headless chunks of at most 64 and uses two-position
microbatches. Five BF16 variants share weight loads; remaining operations replay their
scalar arithmetic with separate position buffers. The final prompt token and generation
stay B1. Set `FLASH_PREFILL_CHUNK=0` to disable chunking, or set it to a size from `1` to
`4096`. `FLASH_PREFILL_CHUNK=1` uses scalar microbatches; `FLASH_TWO_POSITION=0` forces
scalar microbatches for any chunk size. Set `FLASH_TWO_POSITION=1` to use paired execution
even when chunking is disabled. See [BATCH.md](BATCH.md) and [PAIR_TESTING.md](PAIR_TESTING.md).

`FLASH_ATTN_Q_SPLIT=1` distributes the 12,288-row attention query projection
across dies with space for local weight slices. At engine initialization it
copies those slices from the resident weight arenas. With at least three
workers available, the attention owner handles K/V and the indexer while the
other workers compute Q. Results use the existing broadcast and synchronization
path. The planner runs after context, image, and paired activation allocations,
reserves 8 MiB per participating die, and rejects the reload if fewer than two
workers fit. The default is `0`; `config` reports `attn_q_split` and the
hexadecimal `attn_q_mask`. See [the measured results](validation/issue82/attention-q-split/README.md).

## Timing measurements

[TIMINGS.md](TIMINGS.md) defines the shared CLI/HTTP timing schema, counts and
rate denominators. Enable CLI records with `timings 1`; HTTP includes the same
record under `timings.engine` and request/transport events under `timings.http`.
Host-observed decode, device publication intervals and complete request timing
are separate measurements. A request with fewer than two generated outputs has
no measured decode interval rate. Optional device tracing and its validation
commands are documented with the schema.

#### Planned: 
- MTP head
- unified KV cache
- general matrix-batched prefill
- concurrent requests
- V340L version

## What one token runs

| Kind | Operation | phases per token | µs per phase | ms per token | exchanges |
|---|---|---|---|---|---|
| **exchange and quantize** | | | | **6.26** | |
| `xsend` | P2P stores, fence, flag | 111 / 111 | 12.8 | 1.42 | 50 projection, 49 contraction, 13 broadcast |
| `actq256` | quantize an activation to int8 + scale | 146 / 146 | 33.1 | 4.84 | |
| **hyper-connection** | | | | **8.26** | |
| `bf16hcrwx` | RMSNorm, hc down, inject (BF16), write | 48 / 48 | 40.3 | 1.93 | 48 projection |
| `bf16hcrx` | RMSNorm, hc down, inject (BF16) | 49 / 49 | 32.0 | 1.57 | 49 projection |
| `bf16hcup` | SiLU + hc up + lane mix + inbox write | 97 / 97 | 25.3 | 2.46 | 97 projection |
| `hcwsum` | stream write, sum of the six MoE partials | 48 / 48 | 47.9 | 2.30 | |
| **MoE** | | | | **5.14** | |
| `router` | router GEMV and softmax | 48 / 48 | 23.3 | 1.12 | |
| `moesel` | top-10 expert selection | 48 / 48 | 5.6 | 0.27 | |
| `q2moegn` | expert gate and up GEMVs (Q2_0) | 96 / 96 | 18.4 | 1.77 | |
| `ffnact` | SiLU(gate) · up, quantize | 48 / 48 | 13.0 | 0.62 | |
| `q2moedn` | expert down GEMV (Q2_0) | 48 / 48 | 21.2 | 1.02 | |
| `q2k2560`, `q2acc640`, `b32q40acc`, `b32q50acc`, `b32iq4nlacc` | shared-expert, small matrices | 20 / 24 | 15–26 | 0.34 | |
| **projections (GEMV by weight format and input length K)** | | | | **6.61** | |
| `q3kk2560` | Q3_K, K = 2560 | 50 / 49 | 34.2 | 1.71 | |
| `bf16k2560` | BF16, K = 2560 | 49 / 53 | 27.8 | 1.36 | |
| `iq4k2560`, `iq4k6144` | IQ4_XS, K = 2560 / 6144 | 19 + 17 / 22 + 17 | 42.8 / 33.1 | 0.81 + 0.56 | |
| `q4kk2560`, `q4kk6144` | Q4_K, K = 2560 / 6144 | 8 + 15 / 11 + 15 | 35.3 / 29.4 | 0.28 + 0.44 | |
| `q5kk2560`, `q5kk6144` | Q5_K, K = 2560 / 6144; one `q5kk2560` is the share of the 248,320-row output head | 1 + 7 / 4 + 7 | 688 / 41.0 | 0.69 + 0.29 | |
| `q6kk1536`, `q6kk1536acc` | Q6_K, K = 6144 as four K = 1536 slices summed in a fixed order | 6 + 18 / 6 + 18 | 18.3 / 13.6 | 0.11 + 0.24 | |
| `q3kk6144` | Q3_K, K = 6144 | 3 / 3 | 35.2 | 0.11 | |
| **DeltaNet (36 blocks)** | | | | **1.53** | |
| `bf16bias` | BF16 GEMV with bias | 36 / 36 | 7.0 | 0.25 | |
| `conv1d` | causal conv (k = 4) + SiLU | 36 / 36 | 5.7 | 0.20 | |
| `dnstep` | recurrence, state update, read-out | 36 / 36 | 18.1 | 0.65 | |
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

Each token makes:

- 342 exchanges on each die
- 280 projections
- 49 contractions
- 13 broadcasts

## Layout
| Path | Contents |
|---|---|
| `kern/*.sass` | the kernels: GEMVs per weight format, DeltaNet, attention, MoE router, hyper-connection ops, exchange, top-k and sampler |
| `kern/die.pl` | generator of the persistent kernel: phase dispatcher, grid barrier, token loop, exchange flags |
| `kern/prog.c`, `engine.c` | the per-token program builder and the reloadable engine library |
| `kern/flashcore.c`, `flashd.c`, `serve.sh` | the persistent core, the HTTP front-end, start/reload script |
| `.github/` | CI: `scripts/pr-audit.sh` (language and issue-link rules, also `PR=N .github/scripts/pr-audit.sh` locally) on every PR push, and `scripts/draft-conflicting.sh`, which converts ready PRs with merge conflicts to drafts |
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

### Image encoder

`flashvis` requires a CUDA GPU and GPU Flash Attention. It selects the visible
GPU with the most free memory, or the ggml GPU name in `FLASHVIS_DEVICE` (for
example, `CUDA6`). `auto` selects a GPU; it does not select a CPU or disable FA.
A missing GPU, invalid device, unsupported attention shape, failed GPU encode,
or non-finite embedding produces an explicit error. There is no CPU mode,
CPU retry, or standard-attention path in the service.

A parent process starts before CUDA initialization and supervises a persistent
fork+exec GPU worker. It buffers the complete successful response before
publishing it. A native CUDA abort, truncated response or worker EOF returns
`err GPU encoder worker terminated` for that image. The image is not resubmitted.
After an error, the next request starts a fresh GPU worker.

The M60's installed FA tile kernel fails with a partial KV tile. The encoder
uses a C/PTX online-softmax FlashAttention kernel for those lengths and the
validated installed GPU FA kernel for KV lengths divisible by 256. This is a
fixed choice by tensor shape; execution errors do not change kernels. Both
paths visit every key and use GPU Flash Attention. The PTX path keeps tile
scores, probabilities, maxima, sums and PV accumulators in FP32 and does not
allocate a global attention matrix. A graph callback checks every compute
operation's actual GPU buffer and verifies that each FA operation reached the
required dispatch. Output is checked for finite values before publication.

The encoder uses the model's 65,536–16,777,216 pixel preprocessing policy:
64–16,384 image tokens with patch size 16 and spatial merge size 2. A
4096 × 4096 image produces all 16,384 tokens and a 128 × 128 M-RoPE grid.
The model's patch alignment still applies. Inputs above 16,777,216 pixels are
rejected explicitly. Image resolution and token counts are not reduced to fit
GPU memory. The engine's default image capacity is 16,384 tokens per response,
bounded by context capacity. The `vision-all` and `vision-core` Make launch
targets supply `FLASH_MAXIMG=16384`; this is an aggregate limit across images.
Cached embeddings use a new `16mp-` filename prefix so entries produced
with the previous resolution limit are not reused.
When `FLASHVIS_SOCK` enables vision in the HTTP frontend, the default
whole-request deadline is 2,400 seconds to allow a maximum-size image to finish
encoding and prompt processing after the configured queue wait. Text-only
service keeps its 300-second default. `FLASHD_REQUEST_MAX_S` overrides either
default; set it to `0` to disable the deadline.

The C adapter targets the installed llama.cpp CUDA ABI at
`fcac19079f3f2985fe74ed845f0880e7e9a8030a`. Build using headers and libraries
from that checkout; the prebuilt libraries are reused. The default tree is
`~/claude/llama.cpp-cuda`; `LLAMA_CPP` and `LLAMA_BUILD` select another matching
source/build directory. A CUDA 11.4 `ptxas` assembles `vis_fa.cubin` for sm_52.
The binary records its build's absolute module/backend paths, so rebuild it
when moving the checkout. The FA dispatch check rejects a missing or bypassed
hook; other dependency builds remain unverified.

```console
dsh build --host archy --require-tool cargo --require-tool cc -- make -f /home/nate/work/fs96-sol/validation/issue96/Makefile all cpu-check --artifact vision-artifacts
ssh archy env FLASHVIS_DEVICE=CUDA6 /home/nate/work/fs96-sol/kern/flashvis /path/to/mmproj.gguf /path/to/text.gguf /tmp/flashvis.sock 10
```

For complete engine startup, the Make targets `vision-all` and `vision-core`
in `kern/Makefile` supply the 16,384-token image capacity to the inherited
launcher. The engine's C default is also 16,384, bounded by context capacity.

`CUDA_VISIBLE_DEVICES` affects the ggml GPU names. Image file decoding and the
model's image preprocessing run on the host; projector computation and attention
run on the selected GPU. CUDA graph capture is disabled for the synchronous
C driver adapter. No setting disables GPU FA or enables CPU execution.

The daemon logs the FA implementation, actual GPU, shape, GPU event interval,
observed compute/FA counts, and sampled device memory usage. Those device memory
samples include other allocations on that GPU; they are not an isolated process
peak. The installed FA kernel's scratch/shared/register counts are unmeasured;
the C/PTX kernel uses zero global workspace, 25,376 shared bytes and 86 registers
per thread in the tested build. See [issue #96 validation](validation/issue96/README.md)
for uncached image timings, numerical references, failure checks, and exact
queue/build identities. The CPU numerical oracle is a separate test executable.

### Text engine

Refresh the files from `maxas-patch/` in an existing MaxAs checkout before building
the direct-dispatch kernel; its BRX encoding and label relocation support are
required. The default build uses `DIE_DISPATCH=jump`, `GEMV_HC_ROWS=4`, and
`GEMV_HC_SLICE_LOOP=1`. The prior linear dispatcher and eight-row/unrolled HC
variants remain selectable with their corresponding build variables.

```
make -C kern die.cubin libengine.so flashcore flashd hold
```

## PLE row reads

With `FLASH_PLE_DISK=1 FLASH_PLE_READERS=16`, the engine reads a position's sixteen
90-byte rows with a persistent pool of 16 readers. Serial reads remain the default;
values from 2 through 16 bound the number of concurrent readers. The pool does not pin
the full PLE file. See [PLE_READS.md](PLE_READS.md) for the lifecycle, configuration,
and C correctness and file-I/O benchmarks.

### Local PLE file cache

`FLASH_PLE_FILE=/absolute/path/to/copied-shard.gguf` redirects disk-mode PLE
row reads to a local copy. The copy must contain the entire original second
shard, including its header; the engine keeps the core's existing table offset.
An unavailable, non-regular, or differently sized cache falls back to the core source.
The setting does not change the GPU kernels, staging buffers, weights, context
capacity, or image capacity.

Create the copy under a temporary name and rename it only after copying succeeds.
The original model file remains the source of truth. File-size checking detects
an incomplete copy; it does not identify an unrelated model with the same size.

```sh
cp --sparse=always "$GGUF2" /local/cache/flash-ple.gguf.tmp &&
  mv /local/cache/flash-ple.gguf.tmp /local/cache/flash-ple.gguf
```

To apply it to an already resident model, use the engine's reload command:

```text
reload /absolute/path/to/libengine.so FLASH_PLE_FILE=/local/cache/flash-ple.gguf
```

The engine logs the selected file and reports `ple_file` in `config`. Reloading
with `FLASH_PLE_FILE=` returns to the core's original file descriptor. The engine
opens a cache descriptor for one synchronous row batch and closes it after all
readers finish; it never keeps a RAM file alive between batches. A short or failed
cache read is retried from the original source before publishing the rows.
Serial and parallel reader settings remain available for either source.

### RAM-backed PLE source

`kern/ple_ram` copies the source into an anonymous RAM-backed file, locks each
64 MiB chunk as it is filled, and seals the completed file against future
writes and size changes. This avoids dirty disk writeback and does not depend
on retaining an NFS mapping in the filesystem cache. It uses no CUDA context.

```sh
make -C kern ple_ram
./kern/ple_ram "$GGUF2" /tmp/flash-ple-ram.ready
```

The helper remains in the foreground to keep the mapping locked. Its nonempty
ready file appears after the copy and page-residency check finish. The first
line is a readable `/proc/PID/fd/FD` path; use that path as `FLASH_PLE_FILE` in
the reload command above. The following line records the byte and page counts
and the source file identity.

The loader requires the complete file size plus a 4 GiB available-memory reserve
and checks the reserve while loading and every 20 ms afterward. It withdraws
the cache when that reserve is crossed; serving then uses the original source. The locked-memory limit must also permit
the allocation. A failed read, memory lock, residency check, or file seal aborts
before publishing readiness. To release the RAM, reload the original source
with `FLASH_PLE_FILE=` and terminate the helper. SIGTERM and SIGINT remove its
ready file; a stale ready file must be resolved before starting another loader.

## Optional PLE file-cache warm-up

Set `FLASH_PLE_DISK=1 FLASH_PLE_PREWARM=1` when starting `kern/serve.sh core` or
`kern/serve.sh all` to read the complete `GGUF2` file before launching the core.
Pre-warming is off by default. Startup waits for the read to finish, and a read
error prevents core startup. The log reports bytes read and elapsed time.

To warm the file without starting or restarting any service:

```sh
GGUF2="/path/to/model shard.gguf" ./kern/serve.sh prewarm
```

The read uses the operating system's reclaimable filesystem page cache and an
8 MiB userspace buffer. It does not pin the file in RAM. Memory pressure can evict
cached pages; reading the whole file adds startup I/O and elapsed time.

## Licence
MIT, see `LICENSE`. Derived parts and their notices are in `THIRD_PARTY.md`. No model weights are included.
