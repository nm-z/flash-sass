# Teacher-forced checkpoint comparison

This adds capture and comparison tools for issue #22. It does **not** establish that this checkpoint, the sparse approximation, or long-context generation has passed a quality evaluation. Four actual full-model GPU captures are supplied below; they measure agreement between VRAM and pinned-host KV in the same implementation. An independently executed original-checkpoint reference and a held-out quality evaluation are still required.

## Candidate capture

`MODEL_CAPTURE` records actual GPU head-row slices after each position of the existing per-token model harness. It requires all 48 blocks, embedding and head, production QSA, a complete teacher-forced ID sequence, a real PLE table, and no phase truncation or fault injection. It rejects missing or overlapping vocabulary slices before creating capture files. It writes little-endian FP32 logits in `[position,248320]` order, little-endian uint32 input IDs, and raw configuration metadata. It cannot attest the checkpoint identity from the holder's arena alone.

New captures use `flash-sass.raw-model-capture.v2`. The final
`capture_info.json` includes `dtype:"<f4"` and `complete:true` and is published
only after every row has been written and the logit stream has closed. A failed
readback, skipped/repeated position or interrupted run leaves no final marker.
Freezing a v2 capture requires that marker. Archived v1 captures remain
replayable under their original schema; explicit contradictory dtypes, byte
orders and completion flags are rejected in every schema.

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

The native comparator reads one FP32 vocabulary row at a time and computes metrics in FP64. It uses a stable shifted log-softmax and accumulates per-row NLL differences directly. Centered RMSE centers the candidate-minus-reference difference vector; it must not subtract a large anchor logit separately from each input vector. The comparator hashes the actual consumed rows and checks those digests, consumed IDs and vocabulary mapping against the frozen manifest. It also rechecks the manifest and artifacts after comparison. Neither hashes nor producer declarations attest how logits were generated.

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

The upstream file was fetched at revision
`8498ed425fa01bbee7d086efdf1a6634c2962aed` and its Git blob verified as
`946f5fd6e9ba21347654b1b2ca196e86b0f751b3`. The relevant indexer definition is
lines 664..770; RMSNorm at lines 143..163 multiplies by `1 + weight` before its
return cast. This replay consumes already converted multiplicative gamma and
does not add one. Its contiguous-prefix visibility model excludes packed or
otherwise masked documents. FP64 and scalar cast experiments remain numerical
controls, without a claim of framework/device bitwise equivalence.

```bash
make -C kern qsa_quality
perl kern/qsa_quality_replay.pl /absolute/path/new-layer-capture 2052 \
  /absolute/path/new-replay.json
```

Without a limits profile the replay is diagnostic: finite discrepancies are
reported with exit 0 and `numerical_gates.passed:null`. Add
`--limits /absolute/path/predeclared-limits.json` before the capture path to
evaluate explicit numerical gates. The profile must have schema
`flash-sass.qsa-numerical-limits.v1` and every following field:

| Field | Meaning |
| --- | --- |
| `raw_last_key_max_abs` | Maximum error between the snapshot key and prefix key |
| `candidate_raw_score_max_abs` | Maximum unscaled score error using the exported query and pool |
| `candidate_scaled_score_max_abs` | Maximum scaled candidate score error against the full FP64 control |
| `query_max_abs` | Maximum exported-query error against the FP64 control |
| `candidate_selection_exact_ties` | Must be `true`; require consistency with candidate score readback, allowing exact cutoff ties |
| `full_fp64_selection_min_overlap` | Minimum fraction shared with the canonical FP64 selected set, in [0,1] |

Numeric limits must be finite and nonnegative. An empty selected set has overlap
one. Candidate score and score-readback selector gates are not applicable only
when the validated N/budget pair has no sparse score dispatch. Other gates still
run. Missing/invalid profile fields fail; evaluated limit failures preserve the
full report and return 3. The profile bytes are hashed before and after replay.
This binds the chosen profile but does not prove it was predeclared: commit the
profile before the accepted experiment. These gates exclude projection,
attention-output and checkpoint quality, and never set `quality_validation` true.

This export boundary leaves projection weights and activation quantization
upstream of the indexer control. The attention control below uses the exported
attention tensors for a separate comparison. Neither control establishes
full-checkpoint quality. Synthetic fixtures retain their origin in each report.

## Frozen attention outputs and storage controls

`kern/qsa_attention_quality.c` independently reconstructs attention from frozen
postprojection inputs and an identical ordered set of selected block IDs. It
does not link the local SASS instruction twins. It uses FP64 RMS normalization,
the supplied multiplicative gamma and FP32 RoPE tables, GQA with 24 query heads
and two KV heads, dot-product scaling by 1/16, stable softmax and the sigmoid
output gate. Selected complete blocks are expanded in the supplied order and
the contiguous tail is included exactly once. Its scalar FP64 calculation is a
numerical control; it does not claim framework FP32, FP16 or BF16 equivalence.

The input boundary is after projection, before query/key normalization, RoPE
and KV quantization. Projection and upstream activation-quantization effects
remain outside this comparison. The baseline uses the exported normalized
query and exactly dequantized INT8 K/V with their exported FP16 group scales.
The following arms change one input source relative to that baseline unless
their interpretation explicitly combines effects:

| Arm | What the measured difference contains |
| --- | --- |
| Exported Q, dequantized K/V versus GPU output | Attention arithmetic and elementary-function differences on identical effective inputs |
| Reconstructed Q, dequantized K/V | Query normalization and RoPE sensitivity |
| Exported Q, dequantized K, raw V | V-storage quantization sensitivity |
| Exported Q, reconstructed raw K, dequantized V | Combined K normalization, RoPE and storage sensitivity |
| Exported Q, reconstructed raw K, raw V | Combined K normalization/RoPE and K/V storage sensitivity |
| Exported Q, separate GPU FP32 K append, dequantized V | Differences between K append paths, including any normalization/RoPE differences |
| Exported Q, separate GPU FP32 K append, raw V | Combined K append-path and V-storage sensitivity |

The last two arms require `normalized_k.f32`. That optional artifact comes
from a separate `OP_ATTNQK` execution. It is not an export of the exact
prequantization K inside `OP_ATTNQKQ`. Even byte-identical queries from those
operations cannot establish identical prequantization K. Accordingly
`pure_k_storage_isolated` and `k_storage_only_measured` stay false. The measured
effects are not assumed to add linearly.

```bash
make -C kern qsa_attention_quality
perl kern/qsa_attention_replay.pl \
  --limits validation/issue22/model-attention-limits.json \
  /absolute/path/new-layer-capture 2052 /absolute/path/new-attention-report.json
```

The Perl wrapper validates completion and snapshot metadata, byte order,
dimensions, epsilon, selected-token order and every reported mode. It hashes
the inputs, reference executable, wrapper and limits profile before and after
the replay. It rejects nonfinite data, malformed IDs or scales, contradictory
metadata, malformed limits, missing artifacts and attempts to overwrite an
existing report. These checks bind the observed files; they do not attest how
the producer generated them or prove immutability between observations.

The committed `flash-sass.qsa-attention-limits.v1` profile supplies four gates:

| Field | Declared limit |
| --- | --- |
| `arithmetic_max_abs` | 0.005 |
| `arithmetic_rmse` | 0.001 |
| `arithmetic_relative_l2` | 0.00005 |
| `query_max_abs` | 0.00001 |

These limits were committed in `affe73f3b62c94639875a1d5b2c5d6958752f7be`
before the first new GPU fixture capture. They are numerical regression limits
for the deterministic fixture, not original-checkpoint quality thresholds or
limits on the quantization-effect arms. A failed evaluated gate preserves the
complete report and returns 3. Invalid inputs return 2 without publishing a
final report. Without a profile the result is diagnostic. A passing report
never sets checkpoint or QSA quality acceptance to true. The profile hash alone
does not prove historical predeclaration; the recorded Git revision and run
identity supply that separate timing evidence.

### Deterministic GPU attention fixtures

`kern/qsa_attention_gpu.c` runs the production `OP_ATTNQKQ`, `OP_ATTNQK`,
`OP_KVGATHER`, `OP_ATTNPARTN` and `OP_ATTNMERGE` SASS operations on deterministic
postprojection inputs with seed 4242. It exports their actual readbacks and
records the device PCI address, CUDA driver version, build revision, opcodes,
fixed selected IDs and completion metadata. Tiny values, zero KV heads and
large V outliers exercise storage behavior. It does not execute the production
indexer, a full model or a teacher-forced checkpoint sequence.

```bash
make -C kern qsa_attention_quality qsa_attention_gpu die.cubin \
  FLASH_REV="$(git rev-parse HEAD)"
bash validation/issue22/run_attention_gpu.sh 6 262000 512 \
  1,2047,2048,2049,2050,2051,2052,2053,2054,2055,3100,8000,65000,262000 \
  /absolute/path/new-gpu-attention-run
perl kern/qsa_attention_replay.pl \
  --limits validation/issue22/attention-limits.json \
  /absolute/path/new-gpu-attention-run/capture 2052 \
  /absolute/path/new-gpu-attention-run/n2052-comparison.json
```

Choose the GPU from the actual machine inventory. The runner queues on
`/tmp/flash-sass-engine-validation.lock`, records an observation every three
minutes while retaining its place, and permits up to 30 minutes to acquire it.
Under the lock it records device/process state, verifies available memory,
limits the fixture runtime, checks source and binary hashes before and after,
and hashes every completed capture artifact. It leaves other workers running.
A successful capture still requires the independent replay command for each
snapshot. Fixed supplied IDs can demonstrate attention exclusion and tail
handling; they cannot demonstrate production score generation, top-k selection
or the full-model sparse dispatcher required by issue #22.

For the separate actual-model length experiment, commit
`model-indexer-limits.json` and `model-attention-limits.json` before capture and
pass the applicable file to each replay. The published profiles were committed
in `04e77e181fa6195f199ab384ad19e36fb18484d5` before the queued production-budget
model experiment could launch. The indexer profile requires an exact last-key
binding, raw-score error at most 0.001, scaled-score error at most 0.02, query
error at most 0.00001, candidate selection consistent with its scores allowing
exact ties, and at least 0.99 overlap with the independent FP64 selected set.
The model attention profile uses the four attention limits listed above.
These are numerical limits on the frozen-layer experiment. Held-out loss,
retrieval accuracy and original-checkpoint agreement need separate thresholds
and measurements. The earlier nine-token model smoke remains diagnostic; its
results were not retroactively accepted under these profiles.

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
checks, replays the committed placement and recovered CPU-reference captures,
and generates/replays 14 synthetic lengths, including sparse tails 2053..2055.
It records exact commands, exit codes, runtime details
and source/binary hashes, rejecting source changes during the checks. The final
report identifies CPU success independently of GPU execution and checkpoint
quality. Its large synthetic inputs can be regenerated from the committed C
source; preserve the hashes and compact reports with an actual experiment.

## Review measurements and current status

[Attention and model-layer follow-up at 2284a21](validation/issue22/review-2284a21/README.md)
records 250 passing analytic/regression checks, six attention analytic groups,
and all 14 synthetic indexer lengths. The new GPU attention experiment passed
the unchanged predeclared limits at all 14 synthetic lengths through 262000
with budget 512, plus four budget-one smoke snapshots. It uses fixed supplied IDs.
An additional full-48-layer nine-token capture with budget one completed with
actual model weights; its independent indexer and attention replays are
diagnostic. All nine vocabulary rows match the historical same-input capture
byte for byte. None of these observations establishes original-checkpoint or
held-out long-context quality. The linked evidence records subsequent model
preflight outcomes and the acceptance state separately.

[Capture-completion follow-up at c20f156](validation/issue22/review-c20f156/README.md)
records 204 passing analytic/regression checks, all 14 synthetic lengths,
recomputed archived comparisons, and a successful harness build. The new v2
capture completion rules are included in those checks. Both shared-GPU lock
attempts returned exit 75 after 45 seconds; no new GPU run occurred. The
machine-readable report leaves issue #22 unresolved and PR #46 in draft.

[Review evidence for f6ddcd5](validation/issue22/review-f6ddcd5/README.md)
records passing CPU checks: 74 comparator assertions, 48 export-mock assertions,
36 replay assertions, 16 analytic assertions, 32 model CLI boundary checks,
both historical placement comparisons and 14 synthetic lengths. Its machine
readable acceptance report keeps `issue22_resolved:false`.

The recovered four-position CPU llama.cpp capture is preserved in
[the reference archive](validation/issue22/archy-llama-7182ef7/README.md).
Comparing it with the historical GPU capture measures logit RMSE 0.1868455,
mean reference-to-candidate KL 0.0210944, top-1 agreement 4/4 and mean top-10
overlap 0.85. These are exploratory microfixture metrics with no acceptance
thresholds and incomplete historical/original-checkpoint provenance. They
cannot establish representative quality or isolate QSA and precision effects.

The earlier CPU-only reviews' blocked GPU attempts remain historical records.
PR #46 remains draft while the complete independent-comparison and held-out
quality acceptance in issue #22 is unmet. A completed capture, successful
replay or passing numerical profile must not set the overall issue or
checkpoint-quality status to accepted.

The continuation merged current `master` through commit `98c176a` and passed
the full 25-step CPU suite on Benji via DSH, plus the 93-case request-checker
and 76-case model-quality tests. The GPU attention fixture and cubin are built
on Benji with pinned MaxAs and CUDA 11.4 ptxas. A one-die synthetic smoke is
queued on Archy device 6; the six-die model run, original-checkpoint execution,
and held-out quality remain unmeasured. Current details and artifacts are in
[`validation/issue22/review-20261020/README.md`](validation/issue22/review-20261020/README.md)
and [`validation/issue22/review-20261021-cpu-summary/README.md`](validation/issue22/review-20261021-cpu-summary/README.md).
Criteria 1–6 remain unmet, and this work does not make the PR ready for review.
