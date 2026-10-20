# Parallel-oracle source epoch and GPU evidence

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9) and
[PR #105](https://github.com/nm-z/flash-sass/pull/105).
Numerical source is fixed to `e2ff58ff1b0e956d61a1c93830c5f66210d27788`.
The adopted PR's numerical C/header files match this epoch; later documentation
and canonical environment-recipe commits may advance the branch head.
See [the parallel oracle contract](../PARALLEL_ORACLE.md).

## CPU and input gates

All checks use the complete 48-layer tile export, real checkpoint weights,
embedding, PLE and full head, six logical/physical dies, production shares/map,
IDs 10 through 18 and budget 1. The C comparison contract is exact bytes.

| DSH build | Result | Evidence |
| --- | --- | --- |
| `build-130547-1792506606733982340` | 17 API/CLI checks passed | `cpu-api17.excerpt` |
| `build-154782-1792506822173131858` | Four-worker row build completed nine positions; reader verified 9 positions / 8280 items | `cpu-parallel9.excerpt` |
| `build-154783-1792506822176279513` | Independent serial run completed and verified 9 / 8280 | `cpu-serial9.excerpt` |
| `build-245925-1792507667079498618` | Complete records and checkpoints compare byte for byte | `cpu-parity9.excerpt` |
| `build-260985-1792507807278547126` | Expected rejection: derived receipt supplied as a root | `reject-derived-root.excerpt` |
| `build-267485-1792507867195499435` | Expected rejection: altered expected tensor SHA | `reject-expected-sha.excerpt` |
| `build-267484-1792507867192359790` | Expected rejection: isolated copied norm tensor's mtime changed | `reject-file-mtime.excerpt` |

Each negative input case reports its intended error and leaves both the new
record and receipt files at zero bytes. The altered norm fixture is a separate
regular file; the production cache is unchanged. The failed DSH request status
is expected for these rejection checks. No CUDA/model execution is claimed by
those cases. Excerpts preserve compiler arguments, relevant byte identities,
structured results and errors; full telemetry remains in DSH's build logs.

The positive runs explicitly derive a version-2 receipt from the same version-1
full-hash root, SHA-256
`e7d0ff80c5b410dace18a069dcd425ab978c153af51f16845377a7c6bc700908`.
Both complete records have SHA-256
`e3b6821bf962b5850471bf16db182ed06fee97af639ebbceb31b0a3b26a52ca9`;
both checkpoints have SHA-256
`3f8479481248520bbf1d15e50bb3967440383f684894ba3c9444ae8713bd699d`.
The parallel producer SHA-256 is
`423cf9c5dad8efbda0dbb39fc1eca0eb79b8a5341b199188813bb8a05a77c2b3`.

## GPU consumer

The C GPU consumer was built by routed Archy build
`build-301653-1792508182812478140`; SHA-256 is
`3c0dafd512307c58091202472a0285e7f5a997bb58706eb39938fd032584832c`.
The cubin is unchanged:
`aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`.
The queue validates the cubin/SASS/MaxAs receipt and records device/driver,
holder manifests, checkpoint identities and runtime arguments.

| GPUQ job | Workload | Result |
| --- | --- | --- |
| `78df6e22-1a73-40f4-9ced-37c57a50d84c` | Host KV, per-token, all 48 layers/six dies, budget 1 | Exit 0; all nine complete scalar records and predictions match. Twelve actual selected-ID captures at visible length 9, one for every attention layer, all match. Exclusion is active at lengths 8 and 9. |

`.logs.json.gz` is a durable queue database export; `.stdout` is its readable
derived output. Optional resident trace collection was unavailable, not zero.
CPU/build wall time and device intervals are separate clock domains. No speedup
is inferred from concurrent CPU runs or from comparing these domains.

The new epoch's remaining KV/launch placements and fault gates are queued
serially. The uniform four-worker deep reference starts from position zero in
`par-e2ff-b512-t2055`, with 32-position durable checkpoints and source epoch
held fixed. Its initial eight-chunk routed job is
`build-352749-1792508656328957494`. Completion is pending. The older source-36a
native comparison stream was retired with 96 committed positions; its data and
checkpoints remain available as partial evidence. Its first 32-position record
prefix and frozen checkpoint matched the O3 source-36a control through routed
comparisons `build-227440-1792507498960444325` and
`build-227441-1792507498960899876`; these are not a completed 2055-position run.

The separate intermittent paired engine failure is tracked in
[issue #109](https://github.com/nm-z/flash-sass/issues/109). Passing and failing
request controls remain in [the earlier record](../whole-model-3902787/README.md).
