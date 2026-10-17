# Teacher-forced model captures and comparisons

This workflow supplies export and comparison tooling for [issue #22](https://github.com/nm-z/flash-sass/issues/22), building on [QSA_MODEL_TEST.md](QSA_MODEL_TEST.md). It separates local implementation agreement, same-implementation repeatability, independent numerical agreement and original-checkpoint quality.

## Capture actual GPU logits

`MODEL_CAPTURE=/fresh/directory` requires QSA mode, blocks 0–48, embedding and head, per-token launches, complete whitespace-separated token IDs in `MODEL_IDS`, and the real PLE table. Loop mode, truncation and mismatch injection are rejected. `MODEL_NOTWIN=1` permits GPU capture without the scalar tensor cache; it does not establish scalar agreement.

From the directory containing the exact built `model_test` and matching `die.cubin`:

```sh
MODEL_ATTN=qsa MODEL_NOTWIN=1 FLASH_NBLK=512 FLASH_KV_HOST=0 \
MODEL_IDS=/absolute/ids.txt MODEL_PLE=/absolute/ple-shard.gguf \
MODEL_PLE_OFF=192 MODEL_CAPTURE=/absolute/fresh-capture \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 CUDA_DEVICE_ORDER=PCI_BUS_ID \
SPLIT_SHARES=112,97,81,116,97,97 ATTN_DIE=1 \
ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5 \
./model_test /unused-with-MODEL_NOTWIN 0 48 4 eh \
/tmp/hold-h0.manifest /tmp/hold-h1.manifest /tmp/hold-h2.manifest \
/tmp/hold-h3.manifest /tmp/hold-h4.manifest /tmp/hold-h5.manifest
```

The example uses four supplied IDs and the loaded six-die split. Match the environment/manifests to the actual holders. The destination must not exist. Capture checks that head slices cover every vocabulary column exactly once, then writes:

- `ids.u32`: little-endian uint32 input IDs.
- `logits.f32`: little-endian FP32, shape `[positions,248320]`.
- `capture_info.json`: dimensions, dispatch/placement settings and build revision.

Four rows contain 3,973,120 logit bytes and 16 ID bytes. Row `t` predicts the token following input ID `t`. Keep the exact command, exit status, source/binary/cubin identities, GPU/driver, effective environment, weight identities and raw log. Capture metadata leaves checkpoint/reference provenance unset and reports `quality_validation=false`.

## Freeze and compare

```sh
python3 kern/model_quality.py freeze --kind candidate \
  --ids /absolute/fresh-capture/ids.u32 \
  --logits /absolute/fresh-capture/logits.f32 --vocab-size 248320 \
  --raw-info /absolute/fresh-capture/capture_info.json \
  --metadata /absolute/candidate-provenance.json \
  --out /absolute/frozen-candidate

python3 kern/model_quality.py compare \
  --candidate /absolute/frozen-candidate \
  --reference /absolute/frozen-reference \
  --out /absolute/comparison.json
```

Freeze the other capture separately with `--kind reference`. Copies have recorded byte counts and SHA-256 hashes, verified before comparison. `--vocab-ids` supplies a uint32 token ID per logit column when columns are reordered. Both captures require identical input IDs and vocabulary ID sets, and every input ID must belong to that vocabulary, including the first ID.

Core provenance fields are `teacher_forced=true`, `base_model.id`, immutable `base_model.revision`, `tokenizer.sha256`, checkpoint file hashes and paths/source URIs, `implementation.name`, immutable `implementation.revision`, and `command`. The full experiment also needs GGUF/conversion and original-checkpoint identities, template/IDs, runtime dtypes, seed and hardware. Keep unknown fields unknown; a config revision alone does not identify original checkpoint weights. Locally supplied checkpoint files are hashed; URI-only identities remain producer declarations, not execution attestation.

Use one capture per uninterrupted sequence. Concatenating independent examples would introduce incorrect continuation labels across example boundaries.

The comparator reports raw/centered logit error, probability error, KL divergence, deterministic top-token agreement/top-k overlap, and next-token loss/perplexity and ranks. A sequence of `T` rows supplies `T-1` continuation labels; a one-row capture has no loss/perplexity estimate. Perplexity overflow is explicit.

Choose numerical limits before examining results: `--max-mean-kl`, `--max-rmse`, `--max-abs-logit`, `--min-top1-agreement` and `--max-mean-nll-increase`. With no limits, the result is `metrics_only` when the core provenance fields are present. `declared_limits_pass` reports only the evaluated numerical criteria. Passing criteria with complete producer declarations yield `passed_declared_limits_unattested`; they never set `quality_pass=true`. The report retains `provenance_status=producer_declared_unattested` and `independent_checkpoint_evidence=false`. Numerical failures may set `quality_pass=false`; otherwise quality remains unestablished (`null`).

Missing provenance returns exit 2 unless `--diagnostic` permits diagnostic metrics. Invalid, changed or incompatible captures return exit 2. Nonfinite output and evaluated limit failures return exit 3. Supplying metadata or assigning a capture the `reference` role does not establish independent checkpoint evidence.

For repeatability, keep `implementation.independent_reference=false`, use `--diagnostic`, and label the result same-implementation repeatability.

## Validation and remaining measurements

The recorded CPU suite passes 14 tests, including analytical KL/NLL, vocabulary alignment, input-ID domain checks, ties, offsets, corrupt/incompatible/nonfinite captures, and representable/overflowing perplexities at NLL 700/710:

```sh
python3 -m unittest discover -s kern -p model_quality_test.py -v
```

These tests validate the tooling. Target-GPU results and their exact source/artifact identities belong in [validation/issue22](validation/issue22).

The current capture exports logits. Frozen layer inputs/weights, pool/score/cutoff/selection outputs, independent precision-isolated comparisons, sparse-length checkpoint measurements, a fixed held-out loss/perplexity and retrieval/task suite, and separately labeled free-running generation remain required. Native QSA pooling order, completed blocks plus tail, FP16/BF16 pool conversion, gamma conventions and selection ties need explicit comparisons. Dense attention is a separately labeled control. Unit tests, a short GPU capture, local byte parity and synthetic long-cache tests do not establish those remaining quality results.
