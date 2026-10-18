# Teacher-forced checkpoint comparison

This adds capture and comparison tools for issue #22. It does **not** establish that this checkpoint, the sparse approximation, or long-context generation has passed a quality evaluation. Four actual full-model GPU captures are supplied below; they measure agreement between VRAM and pinned-host KV in the same implementation. An independently executed original-checkpoint reference and a held-out quality evaluation are still required.

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

Build the CPU-only Perl comparator with `make -C kern model_quality`. It uses core Perl modules, including `JSON::PP` and `Digest::SHA`, and streams one vocabulary row at a time. Run its fixture checks with `perl kern/model_quality_test.pl`. The following commands run from `kern`:

```bash
./model_quality freeze --kind candidate \
  --ids /absolute/path/new-candidate-capture/ids.u32 \
  --logits /absolute/path/new-candidate-capture/logits.f32 --vocab-size 248320 \
  --raw-info /absolute/path/new-candidate-capture/capture_info.json \
  --metadata candidate-provenance.json --out frozen-candidate
./model_quality freeze --kind reference \
  --ids reference-ids.u32 --logits reference-logits.f32 --vocab-size 248320 \
  --metadata reference-provenance.json --out frozen-reference
./model_quality compare --candidate frozen-candidate \
  --reference frozen-reference --top-k 10 --out comparison.json
```

The metadata object contains `teacher_forced:true`, `base_model:{id,revision}`, `tokenizer:{sha256}`, `checkpoint:{files:[{path or source_uri,sha256}]}`, `implementation:{name,revision,independent_reference}`, and `command`. Immutable revision hashes and SHA-256 syntax are checked. Locally available checkpoint files are hashed and verified; remote identities remain producer declarations. The reference's independence boolean is also a declaration, not execution evidence.

Frozen artifacts are hashed again when compared. Inputs must match exactly, every input ID must be in the aligned vocabulary, and vocabulary columns are reordered by token ID. Nonfinite logits fail. The comparator reports raw and centered logit errors, relative L2, cosine similarity, reference-to-candidate KL, probability differences, top-1 agreement, top-k overlap, teacher-forced NLL/perplexity and target rank. The last row has no next-token label. Log-softmax subtracts the maximum before normalization to preserve probabilities under large shared logit offsets; cutoff ties choose lower token IDs.

The native comparator reads one FP32 vocabulary row at a time and computes metrics in FP64. It uses a stable shifted log-softmax and accumulates per-row NLL differences directly. Centered RMSE centers the candidate-minus-reference difference vector; it must not subtract a large anchor logit separately from each input vector. Hash verification identifies supplied artifacts; files must remain unchanged between hashing and comparison. Neither hashes nor producer declarations attest how logits were generated.

Predeclare acceptance limits before running the held-out evaluation using the applicable `--max-mean-kl`, `--max-rmse`, `--max-abs-logit`, `--min-top1-agreement` and `--max-mean-nll-increase` options. There is no universal threshold that proves quality across quantizers or attention approximations.

## Interpreting results

`declared_limits_pass` describes only the supplied rows and limits. A passing comparison keeps `quality_pass:null`, `independent_checkpoint_evidence:false`, and `provenance_status:producer_declared_unattested`. Supplying plausible names and hashes cannot certify independent execution. Numerical or declared-limit failures can set `quality_pass:false`.

Missing required declarations return exit 2 unless `--diagnostic` is explicit; malformed inputs or changed frozen bytes return 2; nonfinite values and failed evaluated limits return 3; successful metric computation returns 0. Exit 0 is not checkpoint-quality acceptance.

## Remaining acceptance evidence

Freeze held-out token sequences at N=1, 2047..2052, 3100, 8000, 65000 and 262000, subject to supported capacity and recorded resource limits. At a budget of 512, N=2051 has 512 complete blocks and a three-token tail; N=2052 has 513 complete blocks and must exclude one. Add N=2053..2055 for sparse tail residues. Record actual emitted production instructions and selected IDs in addition to the requested attention mode. Report native sparse versus repository sparse differences separately from INT8-versus-higher-precision KV differences. Preserve per-position results, sequence-level aggregates, failed examples, layer-level diagnostics and the exact metadata and commands.

The four candidate GPU captures below are supplied. Independent original-checkpoint execution records, held-out acceptance thresholds and quality results, and full-model production-budget sparse-length coverage remain unmeasured. Store large captures outside Git and commit their hashes, reproducible acquisition commands, compact metrics and provenance records.

### Recorded actual-weight captures

[Archy evidence](validation/issue22/archy-06968d1) records executable source
06968d1a5d6d7fdaf12106236cc39a1cb5cdf389 and its binary/cubin identities.

All four full-48-layer GPU runs completed with actual embedding, disk-PLE and
output-head weights. They cover four positions at the production budget of 512
blocks and nine positions at a budget of one block, each with VRAM and pinned-host
KV. Every exported logit is finite. Each placement pair is byte-identical across
the full vocabulary: 993,280 floats per four-position run and 2,234,880 floats per
nine-position run. The input IDs are valid consecutive-token microfixtures,
not a held-out natural-language dataset.

Raw logits are retained as deterministic gzip files, alongside input IDs,
capture metadata, command/environment records, hashes and output. The freeze and
compare commands completed successfully; their diagnostic reports show zero
logit error and leave quality_pass null and independent_checkpoint_evidence false.
These are same-implementation placement comparisons without a scalar twin or
independent original checkpoint. The remaining quality measurements above are
still required.

## Frozen layer exports and numerical controls

`MODEL_LAYER_CAPTURE` adds a separate postprojection export to `model_test`.
Choose an attention block with `MODEL_LAYER_INDEX` and predeclare snapshot prefix
lengths with `MODEL_LAYER_LENGTHS`. The run must supply full blocks 0..48,
embedding, the output head, real PLE data, per-token QSA and exactly T input IDs.
The selected block's raw K, V and indexer-key prefix, supplied normalization
gammas and RoPE tables are saved. Each requested snapshot includes Q/gate and
indexer projections, normalized query, packed INT8 KV and FP16 scales, block
pools, selected block IDs, attention output and layer output. Scores are saved
only when their production operation ran; below the selection boundary, the
all-block selection is labeled implicit. Completion metadata is written after
the requested prefix has been captured.

For example, add these variables to the candidate command above and set T=2055:

```bash
MODEL_LAYER_CAPTURE=/absolute/path/new-layer-capture
MODEL_LAYER_INDEX=3
MODEL_LAYER_LENGTHS=1,2047,2048,2049,2050,2051,2052,2053,2054,2055
```

Hold `/tmp/flash-sass-engine-validation.lock` for the complete GPU command and
its children. Inspect `nvidia-smi` and `ps` while holding the lock, wait if another
worker is using the devices, and leave that worker's processes running. Holder
identities, device placement and the tested library/cubin must accompany a real
capture. The CPU readback mock checks export layout and error handling only.

`qsa_quality` is a C numerical control derived from the pinned Transformers
definition named in issue #22. It replays the indexer from the saved
postprojection inputs, without linking the local instruction twins. Its report
separates the supplied GPU score/selection consistency from recomputed score
arithmetic, cutoff margins and exact ties. It records pool dtype and return-cast
variants, supplied multiplicative gamma, the supplied RoPE table and the
explicit score-scale conversion. `qsa_quality_replay.pl` validates snapshot
metadata and hashes the consumed inputs, executable and produced arrays.

```bash
make -C kern qsa_quality
perl kern/qsa_quality_replay.pl /absolute/path/new-layer-capture 2052 \
  /absolute/path/new-replay.json
```

This export boundary leaves projection weights and activation quantization
upstream of the control. The indexer replay also leaves attention-output/INT8-KV
isolation and full-checkpoint quality open. The exported attention tensors make
those later comparisons possible; exporting them alone supplies no numerical
agreement result. Synthetic CPU fixtures retain their synthetic origin in each
report.

## Independent backend capture

`kern/model_reference.c` calls the C API of an explicitly supplied llama.cpp
source/build. It disables GPU offload, records all vocabulary logits after each
supplied ID, uses FP32 K/V, and writes the final completion marker only after
every row is saved. It also records the ordered vocabulary text, model metadata,
runtime settings and command. The same quantized GGUF is a useful backend
comparison arm; original-checkpoint conversion effects require their own arm.

```bash
make -C kern model_reference LLAMA_DIR=/absolute/path/pinned-llama-cpp
kern/model_reference /absolute/path/model.gguf /absolute/path/ids.txt \
  /absolute/path/new-reference-capture 2 1800
```

Pin and hash the chosen source revision, headers, built shared libraries and
GGUF shards alongside the output. The adapter's comments identify its tested
API revision. Build success does not establish which source produced an old
binary, and the model's own metadata does not prove original-checkpoint or
conversion provenance. Native QSA selection and dtype policy must be inspected
and recorded before a backend run is used for sparse-quality acceptance.

## Repeatable CPU validation

```bash
bash validation/issue22/run_cpu.sh /absolute/path/new-cpu-evidence
```

The runner rebuilds the tools, executes the Perl regressions and C analytic
checks, replays the committed placement captures, and generates/replays the
synthetic length matrix. It records exact commands, exit codes, runtime details
and source/binary hashes, rejecting source changes during the checks. The final
report identifies CPU success independently of GPU execution and checkpoint
quality. Its large synthetic inputs can be regenerated from the committed C
source; preserve the hashes and compact reports with an actual experiment.
