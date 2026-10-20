# Optional parallel rows in the CPU oracle

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9).
The trial numerical source is
`e2ff58ff1b0e956d61a1c93830c5f66210d27788`. The new consumer canary passed
all nine complete scalar records and 12 actual selected-ID captures in GPUQ
`78df6e22-1a73-40f4-9ced-37c57a50d84c`. The tested code is adopted into
PR #105; the numerical files match that source epoch exactly.

## Row execution

`ORACLE_PARALLEL_ROWS=1` together with OpenMP enables static scheduling of
independent output rows in ten C GEMV entry points. Calls with at least 256
rows use four workers. Per-row lane arrays and scratch are private; each row
retains its existing FMA, butterfly and slice-add order. The normal build
executes the serial row loop. Inputs and scales must remain read-only, the
output must not alias those inputs, and environment settings must remain fixed
through the call. The current standalone oracle supplies those conditions.

The tested build adds these flags to the existing C11 / `-ffp-contract=off`
configuration:

```text
-march=native -O3 -fno-math-errno -fopenmp -DORACLE_PARALLEL_ROWS=1
```

The serial comparison adds `-march=native -O3 -fno-math-errno`.
Use a fresh routed source/build or distinct `OUT_DIR` for each compiler variant;
ordinary Make file timestamps do not track changes to command-line flags.
Binary hashes and exact compiler commands are retained with the run evidence.
Canonical Make recipes explicitly clear `GEMV_NOMIN`, the legacy attention-owner
override and the graph profiling overrides before inference.
No speedup is claimed from the concurrent runs' wall times or the short CPU
sample; they have different cache/I/O conditions.

## Full-hash roots and fresh references

A version-1 receipt is produced by the full input hash pass. CPU preparation
can derive a fresh version-2 receipt by setting `ORACLE_REUSE_RECEIPT` to that
full-hash root. It takes one bounded, stable snapshot of the root and hashes
the same bytes it parses. The new receipt binds the current small source
files, teacher IDs, geometry and topology. Cache/checkpoint digests, the full
large-file inventory, sizes and modification times must agree with the root.
No old record or checkpoint state is transferred to the new reference.

The derived receipt stores the root's SHA-256 and the CLI reports
`full_hash=false`. Version-2 receipts cannot be derivation roots. The output
receipt and record stream must be fresh, distinct files; unsafe aliases are
rejected before writing. Both the CPU validator and GPU consumer can read
version-1 and version-2 receipts. CPU preparation reads/hashes the root;
GPU consumption checks the derived identity and current cached metadata and
retains the recorded root hash for provenance. The cached verification contract
assumes immutable large inputs; metadata-preserving content replacement is
outside that contract.

## Measured trial checks

| Routed Archy build | Observed result |
| --- | --- |
| `build-130547-1792506606733982340` | 17 API/CLI cases passed with the four-worker build, including receipt aliases, empty root/output protection and record/receipt alias rejection. |
| `build-154782-1792506822173131858` | Four-worker producer completed nine positions and the production reader verified all 8280 items. |
| `build-154783-1792506822176279513` | Serial producer independently completed nine positions and verified 8280 items with the same source and geometry. |
| `build-245925-1792507667079498618` | Complete records and checkpoints compare byte for byte between the two runs. |
| `build-260985-1792507807278547126` | Expected rejection: a derived version-2 receipt cannot be a full-hash root. Record stream remains empty. |
| `build-267485-1792507867195499435` | Expected rejection: altered expected tensor SHA changes the large-input identity. Record stream remains empty. |
| `build-267484-1792507867192359790` | Expected rejection: an isolated copied norm tensor has a changed mtime. The production cache is untouched; the new record stream remains empty. |

Both positive receipts identify root SHA-256
`e7d0ff80c5b410dace18a069dcd425ab978c153af51f16845377a7c6bc700908`.
Their record SHA-256 is
`e3b6821bf962b5850471bf16db182ed06fee97af639ebbceb31b0a3b26a52ca9`;
their checkpoint SHA-256 is
`3f8479481248520bbf1d15e50bb3967440383f684894ba3c9444ae8713bd699d`.
The four-worker producer SHA-256 is
`423cf9c5dad8efbda0dbb39fc1eca0eb79b8a5341b199188813bb8a05a77c2b3`.
CPU excerpts and the successful GPU canary are retained in
[the source-epoch run record](whole-model-e2ff58f/README.md).
The canary compares all nine complete layer/QSA/recurrent/PLE/logit records and
predictions against the parallel reference on six real GPUs, with budget-1
exclusion active. Deep production-budget reference completion is pending.
