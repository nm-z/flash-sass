# Offline CPU reference for whole-model QSA

This follow-up addresses [issue #9](https://github.com/nm-z/flash-sass/issues/9).
It extends the actual-weight model harness with a CUDA-free, resumable scalar
producer and a GPU consumer. The full-model numerical and request matrices are
still pending. The CPU API and argument checks below establish their stated
scope; they do not establish GPU agreement.

## Execution and comparison

`kern/model_oracle.c` executes the existing instruction-order C reference:
embedding, blocks 0 through 47, block 1's PLE, and the complete output head.
It uses six logical dies, explicit physical die IDs, production split shares
`112,97,81,116,97,97`, and attention map
`1,1,1,1,4,4,4,4,5,5,5,5`. Every position consumes an explicit teacher-forced ID.
Dense-baseline and free-running measurements remain separate harness modes.

The producer writes one record per position and checkpoints at most 32 positions
per invocation. Records contain labelled tensor dimensions, data types, byte
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
ssh archy gpuq run -- make -f /home/nate/work/fs9-sol/validation/issue9/Makefile TOKENS=9 BLOCKS=1 HOST_KV=1 model-oracle
```

Give the first CPU invocation enough DSH time for the complete large-file hash
pass as well as its scalar chunk. Invoke the same CPU chunk target until its
structured completion field is true.
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
| DSH `build-2688266-1792494343255683276` on Archy | 9 public API cases passed; producer and consumer built | Valid raw/digest comparison; one-bit, size, type and shape mismatch; missing, duplicate and extra labels; empty-record rejection. No CUDA execution. |
| Same routed build | 46 boundary checks passed | Argument/capture/oracle scope and empty/missing receipt rejection before CUDA; positive controls intercepted `cuInit`. No numerical model validation. |
| DSH `build-2802738-1792494998585080917` on Archy | 11 API/CLI checks and 46 boundary checks passed | Adds concurrent-writer exclusion before verification and preservation of existing record bytes after failed verification. Builds the consumer with shared reader locking. No CUDA execution. |
| DSH `build-2853303-1792495224803651691` on Archy | 12 API/CLI checks passed | Adds the shared-reader/exclusive-writer conflict case. No CUDA execution; the earlier 46 boundary checks were not repeated for this test-only extension. |
| DSH `build-3053937-1792496534261548369` on Archy | 20 engine config-parser fixtures passed | Full 40-hex revision, duplicate/missing/mismatched scope and pair/chunk flag checks. No private core, CUDA or numerical model execution. |
| DSH `build-3160844-1792497271142504447` on Archy | 12 API/CLI cases passed with `-march=native` | Compiler/ISA variant built successfully; actual-weight record parity remains pending. No CUDA or numerical model execution. |
| GPUQ `796ebc95-d3cc-45dd-ad6b-b25710a895c0` | Failed at the first direct case with illegal instruction | Built with a stale external MaxAs lacking the repository's BRX encoding/relocation patches. This is retained as a failed attempt. |
| DSH `build-1864214-1792491727574313292` | Patched cubin rebuilt with input receipt | Private MaxAs files match `maxas-patch`; cubin SHA-256 `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`. This build result alone is not GPU validation. |

The patched direct matrix, full-model reference/consumer parity, negative
controls, both KV placements and launch modes, deep production-budget exclusion,
and full-model request/snapshot reuse remain required before closing #9.

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
The engine request matrix has not run yet.
