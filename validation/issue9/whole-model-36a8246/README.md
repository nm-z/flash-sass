# Corrected whole-model oracle evidence

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9) and
[PR #105](https://github.com/nm-z/flash-sass/pull/105).
The numerical/record source is fixed to
`36a8246b98df4beb1046f467cb66f9d6225d4d6f`. It corrects the serializer errors
documented in [the earlier run record](../whole-model-3902787/README.md).
Subsequent source changes to engine diagnostics and documentation do not change
the numerical files bound by the scalar input seal.

## CPU stream and checkpoint validation

All runs below executed on Archy through DSH, outside GPUQ. The packed model
export and checkpoint hashes are the complete inputs archived in the earlier
run record. Teacher IDs are 10 through 18, block budget 1, all 48 blocks plus
embedding/PLE/head, production six-die shares/map. Floating comparison is exact
bytes; compilation retains `-ffp-contract=off`.

| Routed build | Result | Retained excerpt |
| --- | --- | --- |
| `build-3770830-1792501786926561393` | Native producer: initial full input verification, four positions, durable checkpoint; exit 0 | `cpu-native-first4.excerpt` |
| `build-4031905-1792504020489746811` | Native producer resumes four more positions; exit 0 | `cpu-native-resume4.excerpt` |
| `build-4048333-1792504161104748827` | Native producer resumes final position; complete stream verified: 9 positions / 8280 items; exit 0 | `cpu-native-final1.excerpt` |
| `build-3991867-1792503664800062495` | Independent portable producer runs all nine positions uninterrupted; complete stream verified: 9 / 8280; exit 0 | `cpu-portable9.excerpt` |
| `build-4061563-1792504281619723007` | Both complete record files and checkpoints compare byte for byte; exit 0 | `cpu-parity9.excerpt` |

The portable run reuses the same fully verified input receipt in a fresh data
directory. The C producer still checks that seal against current source/input
identities. The native variant adds `-march=native`; the portable build adds no
architecture flags. Native producer SHA-256:
`161ad2edeb63f7e98af04f109a987e110bd96be7c38cdc10d59d7b1e14582691`.

Both complete record files have SHA-256
`cde924457c656f853c92d7c0a39a763c54cdd26c844e711e942993535663fe15`.
Both checkpoints have SHA-256
`20bbe8d8ec0de21c9b9c8fba7edd5dc475d966c3d60c461afb94035e990637c8`.
Raw reference/checkpoint files remain in task-owned persistent Archy storage;
they are not added to Git. `.excerpt` files retain compiler commands, relevant
identities and exact structured results from the DSH logs. Repeated process
telemetry is omitted; the full logs remain under their DSH build IDs.
These runs are not matched performance experiments, so no speedup is inferred.

## GPU comparisons

GPU consumer SHA-256:
`b1d31d16e36ae3edf97b7c87b9728122f2883d51e1bbc45d7359216681998584`.
The cubin remains
`aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`.
Queue identity validates the cubin/SASS/MaxAs receipt and records driver/device,
all six holder manifests, both checkpoint hashes and runtime arguments.

| GPUQ job | Placement / launch | Observed result |
| --- | --- | --- |
| `ffde89d3-b9bb-4f2f-b08c-4d7844bbbd7f` | Host KV / per-token | Exit 0, nine complete scalar records match, 9/9 teacher-forced predictions exact. Twelve actual GPU selected-ID captures, one for each attention layer at visible length 9, all match the scalar reference. Budget-1 exclusion is active at visible lengths 8 and 9. |
| `bb7c5913-8f77-4c55-b154-74df7c047aef` | VRAM KV / per-token | Exit 0, all nine complete records and predictions match. Twelve actual selected-ID captures at visible length 9, all attention layers, all match. |

Compressed `.logs.json.gz` files are exports of durable database documents.
The derived `.stdout` preserves readable positive output. The issue9 history
database is `/home/nate/.local/state/gpuq/issue9-history.sqlite3`, sharing the
global FIFO/reservation with default-database jobs. Optional resident trace
collection is unavailable, not zero. Clock domains remain separate; no
performance claim is made from CPU/GPU timing differences.

Remaining gates include both single-launch placements, whole-model injected
errors, and the deep production-budget reference/GPU
comparison. The independent paired engine reuse failure remains tracked in
[issue #109](https://github.com/nm-z/flash-sass/issues/109); its passing and
failing controls remain in the earlier run record.
