# flash-sass

An inference engine for one model, **Qwen3.8-Flash-Next** (the GSQ-RCO Q2_0 GGUF), written as
hand-assembled **Maxwell SASS** (via [MaxAs](https://github.com/NervanaSystems/maxas)) for **Tesla M60**
GPUs (sm_52, 8 GB dies). Each GPU die runs **one persistent kernel per response**: the token loop,
every layer, the sampler and the exchange with the other dies are inside it, so there are no
per-token launches, copies or host round trips. Weights stay in the GGUF's own block formats and
are dotted as integers with packed `XMAD`; the host side is plain C.

It is a research project for one model on one machine, not a general engine.

## What works (measured on 4 M60 cards, 6 dies used)

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

Not done: speculative decoding with the MTP head, batched prefill (the prompt is processed one
token at a time, at decode speed), more than one request at a time, any GPU but the M60.

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
git clone https://github.com/NervanaSystems/maxas maxas
cp maxas-patch/*.pm maxas/lib/MaxAs/
make -C kern die.cubin libengine.so flashcore flashd hold
```

`kern/bootstrap.sh [dir]` does all of that in a new directory from a fresh clone (about a minute), builds the binaries `serve.sh` starts and the unit tests,
and runs the tests that need no GPU and, on one free die (`BOOT_DIE`, default 6), the ones that do. It does not start the engine.

## Licence

MIT, see `LICENSE`. Derived parts and their notices are in `THIRD_PARTY.md`. No model weights are included.
