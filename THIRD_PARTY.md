# Third-party material

Everything in this repository is MIT-licensed (see `LICENSE`). Parts of it are derived from other
MIT-licensed projects, whose notices follow.

## llama.cpp / ggml

MIT License, Copyright (c) 2023-2026 The ggml authors. https://github.com/ggml-org/llama.cpp

- `ref/quant.c`, `ref/quant.h`: the block layouts and dequantisation of the standard formats (Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K,
  IQ4_NL, IQ4_XS) follow ggml (`ggml-quants`), reimplemented here as the bit-exact CPU twin of the GPU
  kernels. The GSQ-RCO Q2_0 layout is read from the published GGUF, not taken from ggml.
- `kern/tok.c`: the Qwen3.5-style pre-tokenizer split, the byte-level BPE merge loop and the
  special-token partition are ports of `llama-vocab.cpp` and `unicode.cpp`.
- `kern/ucd.h`: the Unicode category table is converted from `src/unicode-data.cpp`
  (generated there by `scripts/gen-unicode-data.py`).
- `kern/jinja.c` is an independent interpreter written against the behaviour of llama.cpp's chat
  template runtime (trim_blocks / lstrip_blocks on, the same filters and tests); it contains none of
  its code.
- llama.cpp is also the numerical reference the engine is checked against.

## MaxAs

MIT License, Copyright (c) 2014 Scott Gray (Nervana Systems). https://github.com/NervanaSystems/maxas

`maxas-patch/` holds patched copies of three MaxAs source files (`Cubin.pm`, `MaxAs.pm`,
`MaxAsGrammar.pm`) for use with CUDA 11.4 toolchains. They are derivative works of MaxAs and keep
its licence; apply them on top of a MaxAs checkout.

## Model

No model weights are included. The engine reads the published GGUF of Qwen3.8-Flash-Next
(Qwen team, Apache-2.0 per the file's metadata) quantised with GSQ-RCO by ISTA DASLab.
