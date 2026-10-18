# Independent full-model QSA capture

`kern/qsa_model_reference.c` executes the complete model through an independent
llama.cpp C API and observes that runtime's existing graph tensors. It supplies
the native full-model capture arm for issue #11. It does not call the flash-sass
instruction twins. The adapter derives its basic model-loading/logit export
from [PR #46's C adapter](https://github.com/nm-z/flash-sass/blob/fcf435f728760b64de30214478524666093468ec/kern/model_reference.c).

The initial measured run and its exact limitations are in
[validation/issue11/full-model](validation/issue11/full-model/README.md).
The captured model is the quantized GGUF release. Independent execution of those
weights does not establish original-checkpoint or conversion fidelity.

## Build a separate native reference

The tested reference source is
[ggml-org/llama.cpp at fcac19079f3f2985fe74ed845f0880e7e9a8030a](https://github.com/ggml-org/llama.cpp/tree/fcac19079f3f2985fe74ed845f0880e7e9a8030a).
Use a clean checkout and a separate build directory so the measurement does not
replace another worker's loaded libraries:

```sh
cmake -S /absolute/reference-source -B /absolute/reference-source/build \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
  -DGGML_CUDA=OFF -DGGML_VULKAN=OFF -DGGML_RPC=OFF \
  -DLLAMA_BUILD_COMMON=OFF -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_MTMD=OFF
cmake --build /absolute/reference-source/build --target llama --parallel 2
make -C kern qsa_model_reference qsa_logits_check \
  QSA_LLAMA_DIR=/absolute/reference-source
```

This uses the existing external C++ runtime; the new adapter and comparison
programs are C and Perl. No Python is required. Record the clean source revision,
compiler, CMake configuration, adapter hash and actual loaded library hashes.
The evidence includes a loader trace and matching before/after hashes for the
private `libllama`, `libggml`, `libggml-base`, and `libggml-cpu` builds.

## Capture an exact teacher-forced prefix

```sh
kern/qsa_model_reference \
  /absolute/model-00001-of-00002.gguf \
  /absolute/input-ids.txt 1,2047,2048,2049,2050,2051,2052 \
  /absolute/new-reference-capture 2 3600
```

The ID file contains one complete, whitespace-separated sequence of decimal
vocabulary IDs. The comma-separated list contains strictly increasing **visible
prefix lengths**. Its last value must equal the number of supplied IDs. Every
token is decoded, including positions whose logits are not retained. There is
no sampling or insertion of BOS, EOS, or other tokens. The capture requires the
48-block Qwen4Exp model and vocabulary size 248320. It supports up to 262144 input
positions, including the requested 262000–262003 tail cases.

The context has one sequence, one token per decode, no eviction, FP32 K/V, no
GPU layers or operation offload, no flash attention, and no MTP. The final two
arguments are CPU threads and a wall-time deadline in seconds. A deadline,
decode, readback, shape, coverage, nonfinite, or I/O failure prevents the final
completion marker. The output directory must be new.

Saving only requested logit rows limits output storage; it does not reduce the
number of tokens the reference must decode. This is a diagnostic capture, not a
throughput benchmark or a held-out quality evaluation.

### Files and identities

- `all_input_ids.u32`: the complete teacher-forced history, little-endian uint32.
- `positions.u32`: zero-based input positions corresponding to retained rows.
- `ids.u32`: input IDs at those selected positions.
- `logits.f32`: full-vocabulary FP32 rows in selected-position order, with 248320
  columns per row. Row `r` predicts after `all_input_ids[positions[r]]`.
- `vocab_ids.u32`: ordered vocabulary column IDs; `vocabulary_text.bin` preserves
  each native vocabulary string as a uint32 byte length followed by its bytes.
- `model_metadata.json`: scalar metadata returned by the runtime API. This is
  not a substitute for the original model/configuration and conversion record.
- `tensors.jsonl` and `Lxxxxxx-Byy-*.bin`: native per-layer snapshots. Each index
  entry records the original dtype, dimensions and byte strides. Files contain
  logical elements in `ne[0]`-fastest order, with original scalar dtype bits
  preserved even for strided views.
- `capture_info.json`: written last with `complete:true`, after logit and tensor
  output has closed and every requested position has passed coverage checks.

At each selected length the adapter requires these nodes at all twelve QSA
blocks, numbered 3, 7, …, 47:

| Native callback | Observation |
| --- | --- |
| `indexer_k_raw` | Current token's raw indexer-key projection |
| `indexer_k`, `indexer_q` | Cached pooled keys and normalized/rotated index query |
| `indexer_score` | Native scaled pool scores, including masked padding |
| `indexer_top_k` | Native unordered pool indices, which can include invisible padding |
| `indexer_sel` | Zero/negative-infinity attention mask over cache cells |
| `Qcur`, `Kcur`, `Vcur` | Current-token attention vectors at the native callback boundaries |
| `attn_pregate`, `attn_gated`, `attn_output` | Attention result before/after its gate and after output projection |

It also captures every block's `l_last` output, including recurrent blocks, and
records `indexer_pool_k_new` when present. Required nodes must occur exactly once
per block and selected length. A total tensor count alone cannot satisfy this
coverage requirement.

`indexer_sel` is **not an ID tensor**. The adapter preserves the mask and exports
its zero-valued cell indices separately as `selected_ids.u32`, in ascending set
order. With this one-sequence, non-evicting decode, cell indices equal prefix
positions. It rejects future cells, checks the incomplete tail, and requires
`4*min(floor(length/4),512) + length%4` selected positions. Negative infinity is
allowed only for masked scores and selection masks; activation tensors must be
finite. The selection export does not claim the reference's top-k ordering or a
particular tie-breaking policy.

The current-token K/V files do not contain the entire raw K/V prefix. Cached
index pools and the full selection mask are captured, but same-ID K/V replay and
candidate per-layer comparison require the corresponding target-side exports.

## Bind and compare complete histories

```sh
perl kern/qsa_model_compare.pl \
  /absolute/reference-capture /absolute/candidate-raw-capture \
  /absolute/candidate-frozen-manifest.json /absolute/new-comparison
```

The candidate format is the contiguous full-vocabulary capture from PR #46,
with its `flash-sass.frozen-logits.v1` artifact manifest. New candidate captures
must have a completed `flash-sass.raw-model-capture.v2` marker. Use the explicit
`--historical-v1` flag only to analyze an archived v1 capture; the resulting
report labels it historical. The archived metadata file may have been
normalized after freezing; the wrapper verifies the frozen IDs/logits and
vocabulary-column identities and separately records the supplied raw metadata
hash. It does not assert that the two metadata byte strings are identical.

The wrapper checks the **complete input history**, selected position order,
selected IDs, full vocabulary column order, file shapes, frozen candidate
hashes, and native layer coverage. It extracts the corresponding candidate rows
and calls the existing `qsa_logits_check` with the unchanged limits:
maximum absolute error 0.10 and normalized RMS error 0.01, each enforced on every
row and on the aggregate. Numerical failure returns 3 and preserves the report.
Malformed or inconsistent inputs cannot produce a completed comparison report.

The C checker's `position` field indexes retained rows. The wrapper's
`selected_positions` maps those rows back to the complete input prefix.
Reference activation norms and individual tensor hashes are retained in
`reference-layer-diagnostics.json` and `comparison.json`. The wrapper explicitly
reports candidate per-layer comparison, original-checkpoint conformance and
quality validation as unmeasured. A numerical pass cannot supply those gates.

The retained four-token fixtures also exercise seven negative controls:

```sh
perl kern/qsa_model_compare_test.pl /absolute/reference-capture \
  /absolute/historical-candidate /absolute/candidate-frozen-manifest.json
```

These reject a historical capture without the explicit flag, a changed earlier
input despite matching selected tokens, an incomplete capture, a missing QSA
output hidden by a consistent total count, a truncated attention tensor, a
future selected position, and changed candidate logits.

## Shared GPU validation

This CPU reference needs no GPU lock. Any target GPU execution must acquire
`/tmp/flash-sass-engine-validation.lock` on the GPU host and satisfy the relevant
runtime/memory preconditions. The lock does not authorize stopping holders,
reloading another worker's engine, or overwriting a shared build. Record each
attempt and coordinate before publishing to the shared PR branch.
