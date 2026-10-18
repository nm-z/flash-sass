# Independent full-model QSA capture — issue #11

This follow-up adds and executes a C capture adapter for an independently
built CPU implementation. It preserves full-vocabulary teacher-forced logits
and per-layer QSA tensors from the real, same-quantized GGUF model. It does
**not** claim original-checkpoint conformance or a current GPU acceptance pass.
PR #31 remains draft.

## Sources and actual execution

| Component | Identity or measured scope |
|---|---|
| Full four-position capture adapter | `9809e6d9d93f85c5a8eb3956c110f5adbdb1f55e` |
| Final code and selected-position capture | `3f8a00f2803db6eafc53491856c8eae0eda415f2` |
| Independent llama.cpp source | `fcac19079f3f2985fe74ed845f0880e7e9a8030a` |
| Execution host | `engi`; CPU-only runtime, no CUDA/op/KV offload |
| Model | Both published Q2_0 shards of `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` |
| Published GGUF revision | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` |
| Inputs | Exactly `10,11,12,13`, no inserted tokens, no sampling |
| Model outputs | 4 rows × 248320 FP32 vocabulary entries = 993280 logits |
| Full trace | All 12 QSA layers at each of 4 positions; 252 named tensors per position, 1008 total |
| Filtered trace | The same four logit rows, tracing only visible lengths 1 and 4 |

The adapter was adapted from `kern/model_reference.c` in PR #46 at
`fcf435f728760b64de30214478524666093468ec`. The new capture path observes
independent graph nodes through its C API and does not construct reference
outputs with flash-sass scalar twins.

The independent runtime was rebuilt from a separate clean worktree; the
modified `common/debug.cpp` in the user's existing reference checkout was
not used or changed. The CMake configuration, complete build log, compiler,
adapter commands and measured library identities are retained under `build/`
and `full/`. Each executing process saved its loaded library paths. The
independent libraries and adapter inputs matched their pre-run hashes after
each run. Both complete GGUF files were hashed before the first capture and
after both captures and retained the published SHA-256 identities. File size,
mtime and ctime observations also remained unchanged.

The model runs used a user scope with `MemoryHigh=5G`, `MemoryMax=7G`,
`MemorySwapMax=0`, `CPUQuota=200%`, two threads and `nice -n 10`. Their internal
wall-clock totals, including model loading, were 99.441147 seconds for the
full trace and 42.060406 seconds for the filtered trace. These are execution
records, **not a matched performance benchmark**: cache conditions differed.
The trace-filter comparison is a numerical observation check only.

## Measured results

| Check | Result |
|---|---|
| Native numerical suite at clean source `3f8a00f` | PASS: 58 expected outcomes; unchanged sources and binaries |
| Trace serialization/domain tests | PASS: 156 checks |
| Same trace tests with AddressSanitizer and UndefinedBehaviorSanitizer | PASS: 156 checks, no sanitizer finding |
| Logit comparator regressions at clean source `3f8a00f` | PASS: 32 cases, zero failures |
| Fresh CPU full-model capture | Completed, exit 0; 4 logit rows and all 48 requested layer-position traces |
| Fresh filtered CPU capture | Completed, exit 0; logits and input IDs byte-identical to the full-trace run |
| Fresh CPU reference versus historical GPU candidate | **NUMERICAL_FAIL, exit 3**; all 4 rows violate at least one declared limit |
| Current PR #31 GPU source/image | Not executed in this follow-up; target lock remains occupied |
| Original-checkpoint/tokenizer/conversion conformance | Not established |

The full-model comparison deliberately retains the pre-existing limits:
maximum absolute logit error **0.10** and normalized RMS **0.01**, enforced for
each row and for the aggregate. The observed comparison was:

| Metric | Measured value |
|---|---:|
| Maximum absolute error | 1.807985246181488 |
| Normalized RMS error | 0.092794941228946159 |
| RMS error | 0.23558130937207228 |
| Maximum per-row normalized RMS | 0.14065448724523533 |
| Top-1 agreement | 3 of 4 rows |
| Rows failing the fixed numerical rules | 4 of 4 |

Its candidate is the preserved GPU run at source
`06968d1a5d6d7fdaf12106236cc39a1cb5cdf389`, T=4, budget=512 and KV placement=0,
from PR #46 artifact commit `fcf435f728760b64de30214478524666093468ec`.
**That candidate was not rerun here and is not the current PR #31 build.**
These measured errors cannot establish a current-branch failure or success,
or attribute the difference solely to QSA rather than reference/model changes,
activation precision, quantization or conversion effects.

An additional replay of the earlier recovered CPU reference against the
same historical GPU arrays is retained separately. It also fails these limits:
max absolute error 1.1895020008087158 and normalized RMS
0.073727772467838459, despite 4/4 top-token agreement. The recovered reference
retains its missing contemporaneous library provenance; it is not relabeled
as the new, freshly built runtime.

## Retained files and replay

`captures.tar.gz` contains three named directories and an inner `SHA256SUMS`:

- `reference/`: the complete first CPU capture, including all raw logical
  tensor files, full logits, IDs, ordered vocabulary, model metadata, loaded
  libraries and per-row completion/catalog files.
- `reference-filtered/`: the complete selected-position capture from final
  code, including its identical full logits and the two selected trace rows.
- `historical-candidate/`: the exact earlier GPU IDs/logits and capture metadata,
  with an explicit source/artifact identity and historical-scope record.

No model weights, executable or shared library is committed. Tensor catalogs
record native dtype, shape and original byte strides; files contain logical
elements with dimension zero fastest, excluding stride gaps. Intentional
negative-infinity causal-mask and invisible-score values are preserved.
Completion requires every declared QSA role at every requested layer; a
tolerance cannot turn missing data into a pass.

From the repository root, reproduce the comparison without a GPU, model load,
network access or Python:

```sh
bash validation/issue11/independent/replay.sh /absolute/new-replay-directory
```

**Expected recorded outcome: exit 3, not exit 0.** The script verifies the
outer and inner checksums, builds the strict C comparator and writes
`numerical.json` plus its exit code and logs. The report keeps
`independent_checkpoint_evidence=false` and `provenance_status=not_verified`.
A raw-array comparison cannot attest how either model was trained or converted.

To generate a fresh CPU capture, use the pinned build commands and
`qsa_model_reference` command in `NUMERICAL_VALIDATION.md`. The fixed local
paths in retained logs identify this measured process; they are not required
paths for another installation.

## Remaining acceptance and target availability

The consecutive-ID, four-position microfixture is not a held-out language
sample and cannot exercise production-budget sparse exclusion. The long
full-model matrix at 2047–2052, 3100, 8000, 65000 and 262000 plus all requested
tail residues remains unmeasured. The existing native generated-length suite
is retained as synthetic numerical coverage, not substituted for that matrix.
Original checkpoint and tokenizer revisions, exact conversion source/configuration
and normalization-weight lineage also remain unverified.

Read-only preflights of the M60 host `archy` are retained under `target/`.
At 2026-10-17 22:50:55 -07:00 the global validation lock was held by bash PID
`970242`; earlier checks found PID `910093`. Resident `hold`, `fence` and
`flashcore` processes remain on dies 0–5. Dies 6–7 having free memory does not
authorize bypassing the shared lock. No target GPU workload was launched, and
no other worker's process was stopped or resident engine reconfigured.

Before Ready for review, this branch still requires a current matching GPU
capture and byte-parity execution under the shared lock, verified original
checkpoint/conversion lineage, and the declared paired long-sparse full-model
evaluation. The failing historical diagnostic is evidence to investigate,
not permission to loosen those gates.
