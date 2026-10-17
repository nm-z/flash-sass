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

## Layout
| Path | Contents |
|---|---|
| `kern/*.sass` | the kernels: GEMVs per weight format, DeltaNet, attention, MoE router, hyper-connection ops, exchange, top-k and sampler |
| `kern/die.pl` | generator of the persistent kernel: phase dispatcher, grid barrier, token loop, exchange flags |
| `kern/prog.c`, `engine.c` | the per-token program builder and the reloadable engine library |
| `kern/flashcore.c`, `flashd.c`, `serve.sh` | the persistent core, the HTTP front-end, start/reload script |
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

## PLE row reads

With `FLASH_PLE_DISK=1`, the engine reads a position's sixteen 90-byte rows with a
persistent pool of 16 readers. `FLASH_PLE_READERS=1` selects serial reads; values
from 2 through 16 bound the number of concurrent readers. The pool does not pin
the full PLE file. See [PLE_READS.md](PLE_READS.md) for the lifecycle, configuration,
and C correctness and file-I/O benchmarks.

## Licence
MIT, see `LICENSE`. Derived parts and their notices are in `THIRD_PARTY.md`. No model weights are included.
