# Teacher-forced checkpoint comparison

This adds capture and comparison tools for issue #22. It does **not** establish that this checkpoint, the sparse approximation, or long-context generation has passed a quality evaluation. The committed CPU fixtures exercise the tools. An independently executed checkpoint reference and actual full-model capture are still required.

## Candidate capture

`MODEL_CAPTURE` records actual GPU head-row slices after each position of the existing per-token model harness. It requires all 48 blocks, embedding and head, production QSA, a complete teacher-forced ID sequence, a real PLE table, and no phase truncation or fault injection. It rejects missing or overlapping vocabulary slices before creating capture files. It writes little-endian FP32 logits in `[position,248320]` order, little-endian uint32 input IDs, and raw configuration metadata. It cannot attest the checkpoint identity from the holder's arena alone.

A command template, after building this PR and configuring the existing six holders, is:

```bash
MODEL_ATTN=qsa FLASH_NBLK=512 FLASH_KV_HOST=1 \
MODEL_IDS=/absolute/path/heldout-ids.txt MODEL_NOTWIN=1 \
MODEL_PLE=/mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf \
MODEL_PLE_OFF=192 MODEL_CAPTURE=/absolute/path/new-candidate-capture \
./model_test /absolute/path/full-tensor-cache 0 48 3100 eh \
  /tmp/hold-h0.manifest /tmp/hold-h1.manifest /tmp/hold-h2.manifest \
  /tmp/hold-h3.manifest /tmp/hold-h4.manifest /tmp/hold-h5.manifest
```

Use the actual holder split and attention map recorded with the run. The full tensor cache and held-out IDs must exist; the paths above are explicit required inputs, not evidence that a run occurred. `MODEL_NOTWIN=1` skips the instruction twin while capturing the GPU. It does not designate an independent reference. The raw directory must not already exist.

Capture writes one complete vocabulary row per input position. Its logits alone occupy `T*248320*4` bytes, about 260 GB at 262,000 positions. Capture readbacks and file I/O invalidate inference-throughput measurements. Run quality capture separately from performance benchmarks. The current tool supports a contiguous teacher-forced sequence; it does not implement packed-document masking or sampled position capture.

## Freeze identities and compare

Run the reference in an independently maintained implementation with the same base model, tokenizer, complete input IDs, vocabulary mapping and explicitly declared attention/quantization policy. Save FP32 logits after each input token. Record exact implementation revision, checkpoint file hashes, tokenizer hash, command, runtime, hardware and numerical settings. Distinguish comparisons against native sparse attention, dense attention, and the repository's selected subset.

```bash
python3 model_quality.py freeze --kind candidate \
  --ids /absolute/path/new-candidate-capture/ids.u32 \
  --logits /absolute/path/new-candidate-capture/logits.f32 --vocab-size 248320 \
  --raw-info /absolute/path/new-candidate-capture/capture_info.json \
  --metadata candidate-provenance.json --out frozen-candidate
python3 model_quality.py freeze --kind reference \
  --ids reference-ids.u32 --logits reference-logits.f32 --vocab-size 248320 \
  --metadata reference-provenance.json --out frozen-reference
python3 model_quality.py compare --candidate frozen-candidate \
  --reference frozen-reference --top-k 10 --out comparison.json
```

The metadata object contains `teacher_forced:true`, `base_model:{id,revision}`, `tokenizer:{sha256}`, `checkpoint:{files:[{path or source_uri,sha256}]}`, `implementation:{name,revision,independent_reference}`, and `command`. Immutable revision hashes and SHA-256 syntax are checked. Locally available checkpoint files are hashed and verified; remote identities remain producer declarations. The reference's independence boolean is also a declaration, not execution evidence.

Frozen artifacts are hashed again when compared. Inputs must match exactly, every input ID must be in the aligned vocabulary, and vocabulary columns are reordered by token ID. Nonfinite logits fail. The comparator reports raw and centered logit errors, relative L2, cosine similarity, reference-to-candidate KL, probability differences, top-1 agreement, top-k overlap, teacher-forced NLL/perplexity and target rank. The last row has no next-token label. Log-softmax subtracts the maximum before normalization to preserve probabilities under large shared logit offsets; cutoff ties choose lower token IDs.

Predeclare acceptance limits before running the held-out evaluation using the applicable `--max-mean-kl`, `--max-rmse`, `--max-abs-logit`, `--min-top1-agreement` and `--max-mean-nll-increase` options. There is no universal threshold that proves quality across quantizers or attention approximations.

## Interpreting results

`declared_limits_pass` describes only the supplied rows and limits. A passing comparison keeps `quality_pass:null`, `independent_checkpoint_evidence:false`, and `provenance_status:producer_declared_unattested`. Supplying plausible names and hashes cannot certify independent execution. Numerical or declared-limit failures can set `quality_pass:false`.

Missing required declarations return exit 2 unless `--diagnostic` is explicit; malformed inputs or changed frozen bytes return 2; nonfinite values and failed evaluated limits return 3; successful metric computation returns 0. Exit 0 is not checkpoint-quality acceptance.

## Remaining acceptance evidence

Freeze held-out token sequences at short lengths and at 2,048, 3,100, 8,000, 65,000 and 262,000 positions, subject to supported capacity and recorded resource limits. At a budget of 512, include positions with more than 512 complete blocks so that selection actually excludes keys. Report native sparse versus repository sparse differences separately from INT8-versus-higher-precision KV differences. Preserve per-position results, sequence-level aggregates, failed examples, layer-level diagnostics where needed, and the exact metadata and commands.

The independent execution records, candidate GPU captures, chosen acceptance thresholds, measured quality results, and long-context coverage have not been supplied by this PR. Store large captures outside Git and commit their hashes, reproducible acquisition commands, compact metrics and provenance records.
