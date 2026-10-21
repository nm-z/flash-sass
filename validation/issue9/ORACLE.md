# Offline CPU reference for whole-model QSA

This follow-up addresses [issue #9](https://github.com/nm-z/flash-sass/issues/9).
It extends the actual-weight model harness with a CUDA-free, resumable scalar
producer and a GPU consumer. The full 48-layer, six-die actual model matches
all nine scalar records in both host and VRAM KV per-token runs. Both single-launch
placements also match every prediction and final-position stages. Seven current-source
whole-model fault controls reject their intended mismatches. Deep production-budget
gates remain pending. The evidence below distinguishes
actual-model, synthetic-component and CPU format checks, including the separate
intermittent paired engine failure.

The current numerical source epoch is `e2ff58f`. Its optional four-worker CPU
rows match the serial nine-position stream and checkpoint exactly, with 17
API/CLI checks plus three actual input-rejection controls. The new consumer's
host and VRAM per-token GPU gates pass all nine complete records. Both
single-launch placements match all predictions and final-position stages.
The complete four-mode matrix at `36a8246` remains
separately identified evidence. See
[the current source-epoch record](whole-model-e2ff58f/README.md) and
[the parallel-row/input-root contract](PARALLEL_ORACLE.md).

## Execution and comparison

`kern/model_oracle.c` executes the existing instruction-order C reference:
embedding, blocks 0 through 47, block 1's PLE, and the complete output head.
It uses six logical dies, explicit physical die IDs, production split shares
`112,97,81,116,97,97`, and attention map
`1,1,1,1,4,4,4,4,5,5,5,5`. Every position consumes an explicit teacher-forced ID.
Dense-baseline and free-running measurements remain separate harness modes.

The producer writes one record per position and checkpoints at most 32 positions
per invocation. It validates tensor layouts before hashing source bytes.
Records contain labelled tensor dimensions, data types, byte
counts and SHA-256 digests. Selected block IDs and prediction tuples also have
raw payloads. The comparison contract is exact bytes, including F32 and F16
representations, with contraction disabled during compilation. It is the
instruction-order reference contract, rather than a tolerance comparison.

The GPU consumer validates the input seal, all record positions, teacher IDs,
prediction tuples and complete-stream EOF before `cuInit`. It then executes
the GPU model without executing the scalar model on that path. Per-token
launches compare each position's complete record. A single launch compares
every prediction and the final position's stage buffers, because the GPU trace
slots retain the final position. The consumer rejects nonfinite stream and
logit outputs even if their bytes would otherwise match.

Coverage includes all die/layer stream outputs, each attention owner's INT8
KV rows and F16 scales, indexer output/query/ring/pool, scores and selected IDs
when exclusion is active, canonical compact K/V, softmax partials/output,
DeltaNet and convolution state, PLE state, all logit slices, and final streams
and prediction tuples. Missing, duplicate, extra, malformed or differently
shaped observations fail comparison. Capture of real GPU selected block IDs
requires the full model, six distinct dies, explicit IDs, no injection and
actual exclusion. A budget of 512 first excludes at visible length 2052.

## Input identity and persistence

The input seal covers sorted relative source keys and contents (including the
generated opcode header consumed by the model), tile metadata
and contents, both GGUF checkpoint contents, teacher IDs, PLE offset, budget,
epsilon and topology. The initial CPU verification hashes large inputs. Later
CPU chunks and the GPU consumer reuse that receipt, rehash the small source
files, and require the large inputs' recorded sizes and modification times.
A replacement that preserves those metadata is outside the cached receipt's
verification contract. The GPU entry point requires an existing nonempty
receipt and rejects attempts to perform initial large-file hashing there.

`source_revision` is an annotation supplied by the runner. The relative source
content digest, routed DSH source snapshot/build log, compiler/link flags and
binary hashes provide the implementation evidence. Queue identity also checks
the cubin receipt against its SASS/generator inputs and the checked-in MaxAs
patches. It records CUDA driver, device, holder manifests and model hashes.

Records are flushed and synchronized before a checkpoint is synchronized and
published with an atomic rename. The CLI synchronizes the containing
directories. On resume, an incomplete suffix beyond the committed record
offset is truncated; a record stream shorter than that offset is rejected.
CPU records and checkpoints live outside GPUQ's captured output directory.
The CLI acquires an exclusive nonblocking record-file lock before opening the
shared receipt or checkpoint. The GPU consumer holds a shared record lock.
An existing uncheckpointed record stream is preserved and rejected, rather
than truncated by the CLI's initial file open.

## Remote commands

Use a fresh full-model tile export with all 48 blocks plus embedding and head.
The existing eight-block export is insufficient. The export and its hashes
must finish before the producer starts. On Archy the model files are read from
the existing Sentry NFS model mount. The active export uses two task-owned
volumes: blocks 0 through 35 plus embedding/head on the HDD build volume, and
blocks 36 through 47 in the isolated root-filesystem cache. The CPU tensor
directory links the second part's files and combines their catalogs. Each
tensor remains a separate file; arena offsets are catalog metadata and do not
address a shared CPU arena. CPU oracle output defaults to the task's own
directory on the HDD build volume.

```console
dsh build --host archy --admission-timeout 60 --require-tool cargo --require-tool cc -- make -f validation/issue9/Makefile oracle-check cpu-check
dsh build --host archy --admission-timeout 60 --require-tool cargo --require-tool cc -- make -f validation/issue9/Makefile TOKENS=9 BLOCKS=1 ORACLE_CHUNK=4 oracle-chunk
dsh build --host archy --admission-timeout 60 --require-tool cargo --require-tool cc -- make -f validation/issue9/Makefile TOKENS=9 BLOCKS=1 oracle-validate
ssh archy gpuq run -- make -f /home/nate/work/fs9-sol/validation/issue9/Makefile TOKENS=9 BLOCKS=1 HOST_KV=1 model-oracle
```

Give the first CPU invocation enough DSH time for the complete large-file hash
pass as well as its scalar chunk. Invoke the same CPU chunk target until its
structured completion field is true.
`oracle-batch` serializes eight independently checkpointed chunk invocations
within one routed job. It is CPU work and is rejected inside GPUQ.
Before submitting the GPU consumer, `oracle-validate` uses the production reader
to check the complete stream, shapes, teacher IDs and EOF on CPU.
All chunks for one record stream must use the same identities and geometry.
Keep `ORACLE_REVISION` fixed to the chosen implementation revision across later
documentation or request-runner commits.
`ORACLE_ARCH_FLAGS=-march=native` selects the build host's CPU ISA while retaining
`-ffp-contract=off`. Before using it for the deep reference, compare a complete
native/chunked record stream with an independent baseline/uninterrupted stream
using `oracle-parity`. Copying an input receipt between streams with identical
source, IDs and geometry reuses its content verification. Keep the architecture
flags fixed across chunks and retain each routed build's flags/binary identity.
Copy returned executables into the isolated Archy checkout before queueing GPU
work. The GPU targets compile nothing. Existing serving holders and services
stay under their owners' control.

## Measured checks so far

| Evidence | Observed result | Scope |
| --- | --- | --- |
| DSH `build-661648-1792555441846020340` on Benji at source `421a6df` | After merging master `c099273`, the prefill unit and all issue9 CPU targets exited 0: 17 oracle API, 46 boundary, 12 fault-runner, 20 engine-parser checks | CPU-only; `cuInit` intercepted, `gpu_calls=0`, no GPU or deep-reference validation. Full details: [post-merge CPU checks](whole-model-e2ff58f/benji-421a6df-checks.excerpt). |
| DSH `build-604008-1792554960906070300` on Benji at source `3ec4a25` | `oracle-check cpu-check` exited 0; 17 oracle API checks and 46 CPU boundary checks passed | `cuInit` intercepted; no GPU validation. Full summary and binary identities: [current-source CPU checks](whole-model-e2ff58f/benji-cpu-checks.excerpt). |
| DSH `build-620717-1792555110098571991` on Benji at source `3ec4a25` | 12 fault-runner fixtures and 20 engine-config parser checks passed; exit 0 | CPU-only; `gpu_calls=0`, no numerical GPU validation. Same excerpt. |
| DSH `build-2688266-1792494343255683276` on Archy | 9 public API cases passed; producer and consumer built | Valid raw/digest comparison; one-bit, size, type and shape mismatch; missing, duplicate and extra labels; empty-record rejection. No CUDA execution. |
| Same routed build | 46 boundary checks passed | Argument/capture/oracle scope and empty/missing receipt rejection before CUDA; positive controls intercepted `cuInit`. No numerical model validation. |
| DSH `build-2802738-1792494998585080917` on Archy | 11 API/CLI checks and 46 boundary checks passed | Adds concurrent-writer exclusion before verification and preservation of existing record bytes after failed verification. Builds the consumer with shared reader locking. No CUDA execution. |
| DSH `build-2853303-1792495224803651691` on Archy | 12 API/CLI checks passed | Adds the shared-reader/exclusive-writer conflict case. No CUDA execution; the earlier 46 boundary checks were not repeated for this test-only extension. |
| DSH `build-3053937-1792496534261548369` on Archy | 20 engine config-parser fixtures passed | Full 40-hex revision, duplicate/missing/mismatched scope and pair/chunk flag checks. No private core, CUDA or numerical model execution. |
| DSH `build-3160844-1792497271142504447` on Archy | 12 API/CLI cases passed with `-march=native` | Compiler/ISA variant built successfully. No CUDA execution. |
| GPUQ `796ebc95-d3cc-45dd-ad6b-b25710a895c0` | Failed at the first direct case with illegal instruction | Built with a stale external MaxAs lacking the repository's BRX encoding/relocation patches. This is retained as a failed attempt. |
| DSH `build-1864214-1792491727574313292` | Patched cubin rebuilt with input receipt | Private MaxAs files match `maxas-patch`; cubin SHA-256 `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`. This build result alone is not GPU validation. |
| GPUQ `ef56e0c2-ad05-488d-9cdc-e3992ba72f9e` | 14 direct cases / 56 positions passed | Synthetic history and deterministic FP32 inputs through BF16 projection fixtures; budget 512, both host/VRAM placements, all modulo-four tails at the requested depths through 262003. Not actual-weight whole-model evidence. |
| GPUQ `8f441d80-bace-4a50-a156-cdf6e40e30ba` | Full model passed, 1202 phases | Actual weights, embedding, all 48 blocks, PLE and full head on six production dies; one teacher-forced token, host KV, budget 1. Exact compared stages/states/logits/prediction; no selection exclusion at this depth. |
| Six direct negative controls | Each exited 3 with its specific mismatch | Selection, compact gather and attention-output perturbations, both KV placements, visible length 2052 / budget 512. IDs in the retained run record. These are successful rejection checks, not positive model runs. |
| DSH `build-3659594-1792500856648284611` | Native/chunked and portable/uninterrupted nine-position files matched | Both record and checkpoint bytes equal. This comparison used the old malformed serializer and is retained as compiler/resume evidence, not valid oracle or GPU acceptance. |
| DSH `build-3745644-1792501584558826668` | Two-thread compiler variant matched the same old record/checkpoint bytes | No performance claim: build conditions/timing domains differ. This experimental variant is not needed for the fresh reference. |
| GPUQ `16737acb-6900-46fa-8021-e1fa9e6424d9` | Failed before CUDA | Reader rejected `b3.ring` dimensions versus byte count. The producer multiplied the ring size twice by four; the same inspection found reused final-stream dimensions after token tuples. Both are fixed in `36a8246`; fresh records are required. |
| DSH `build-3758806-1792501690605794907` | 13 API/CLI cases passed | Includes malformed ring-size rejection. The later final-stream dimension fix is exercised by the fresh actual-model producer and full-stream validation. |
| DSH `build-4061563-1792504281619723007` | Corrected native/chunked and portable/uninterrupted nine-position records and checkpoints match exactly | Each complete stream first passed the production reader on CPU: 9 positions / 8280 items. Source `36a8246`; all actual weights and six-die topology. |
| GPUQ `ffde89d3-b9bb-4f2f-b08c-4d7844bbbd7f` and `bb7c5913-8f77-4c55-b154-74df7c047aef` | Host and VRAM per-token oracle runs passed | All nine complete records, layer/QSA/recurrent/PLE/logit state and predictions match exactly. Budget-1 exclusion active at 8/9; each run captures real selected IDs from all 12 attention layers at visible length 9. |
| GPUQ `769fccd6-8cd6-4f4a-bb31-aafae233f6dd` and `75880d0a-557a-476a-b5fe-b226759ac563` | Host and VRAM single-launch runs passed | Every prediction and final position's complete state match the scalar reference; 12 real selected-ID captures per run. Earlier stage buffers are not retained in this launch mode. |

Compressed durable database log exports and scope details are in
[the full-model run record](whole-model-3902787/README.md).
The corrected source's CPU excerpts, byte identities and GPU logs are in
[the corrected oracle run record](whole-model-36a8246/README.md).
Queue trace collection was unavailable for the completed GPU runs because the
resident engine had no enabled trace producer; it is not reported as zero.
The current source's seven whole-model injected-error controls pass their
expected-rejection contract. Deep production-budget exclusion remains required
before closing #9. Both KV placements and launch modes passed
their declared comparison scopes. Full request/snapshot reuse has the
separate intermittent paired failure described below.

## Complete fault controls

The `oracle-fault-matrix` target runs seven serial fault cases in one GPUQ
reservation. Six per-token cases perturb layer output, logits, selected IDs,
recurrent state, convolution state and PLE state; a VRAM single-launch case
perturbs final-position logits. The fixed workload is nine teacher-forced
positions with budget 1 and the existing complete scalar reference.
Each case flips one bit in a host copy of the selected GPU readback before
comparison.

The C runner requires the specific injection label, one failed comparison at
the intended position, the complete expected comparison count and the model's
mismatch exit through Make. It rejects successful exit, missing markers,
incomplete comparisons and successful selected-ID captures. Child execution is
bounded and cancellation is forwarded to the owned subprocess. A CPU fixture
suite checks the acceptance logic before queued GPU execution.

Build/check through DSH with `oracle-fault-check`, then execute the prebuilt
runner through GPUQ with `TOKENS=9 BLOCKS=1`, the pinned `ORACLE_REVISION` and
matching `ORACLE_DATA`. The runner does not produce a CPU reference inside the
GPU reservation. Its CPU/GPU results are recorded in the source-epoch run
record. Routed build `build-893011-1792513509803091078`
passed the strict C build and twelve CPU fixtures, including malformed JSON
rejection and cleanup through the production timeout wrapper. GPUQ
`1a7a5f7c-cf8d-47d7-90ad-bed1bd2904c0` then passes all seven expected-rejection
cases; complete output and per-case status are in the current source record.

## Private engine request validation

`engine_check.c` owns four private `flashcore` children, one at a time, for the
cross product of host/VRAM KV and scalar/paired-chunk launches. It leaves serving
holders under their owners' control. Effective config must describe all 48
layers and six dies; responses must include every die/layer state fingerprint,
PLE/logit fingerprints and consistent resume, pair and chunk counters. Fresh,
changed-prompt, repeated and snapshot/resume requests share GPU-versus-GPU
references across the four children. A separate generated-token request checks
the generation/chunk path. An empty `gen 0` must be rejected before valid
requests. These fingerprint checks do not claim CPU equality;
the scalar record consumer supplies that separate evidence.

The runner is built remotely with `engine-build` and executed through the queue
with `engine-matrix`. Supply the six verified live holder PIDs in logical-die
order through `HOLDER_PIDS`, and use a fresh queue-managed output directory.
`ENGINE_REVISION` must match the library's explicitly compiled `FLASH_REV`;
keep it fixed when later commits change documentation or validation tooling.
The PLE SHA comes from the offline checkpoint hash receipt. A bounded receipt
snapshot is both hashed and parsed; its identity and the live PLE file's
size/mtime/device/inode are checked before and after each child. The report
distinguishes that declared digest from a hash of the live shard. These checks
assume immutable model storage; metadata checks do not prove byte equality
against an offline digest or detect a transient replacement that restores the
same metadata between observations. The independent scalar input seal supplies
its separately documented content verification.
The first executed engine matrix (`ee5de8c3-3472-4ef4-9328-791e6c5cae20`)
passed VRAM/scalar reuse, then failed the paired mode's final repeated A
request with changed recurrent/logit fingerprints. An unchanged repeat
(`93d25b21-3730-4b71-91f6-f192c85227c7`) passed all 64 requests across four
private cores. The intermittent earlier failure is retained and investigated
in [issue #109](https://github.com/nm-z/flash-sass/issues/109); it has not been
reclassified as a passing result.
