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
| `build-517679-1792510180423953110` | 46 pre-CUDA command-line boundary checks passed against the current GPU consumer | `cli-guards46.excerpt` |
| `build-849208-1792513127592872578` | Strict C fault-runner build and seven CPU fixtures passed, including constructed Make arguments and retirement of a TERM-ignoring descendant after its leader exits | `fault-runner-cpu.excerpt` |
| `build-877366-1792513372428956977` | Eight fixtures passed, adding the production GNU timeout wrapper and checking that the stubborn descendant shares the owned process group and is retired | `fault-runner-nested-timeout.excerpt` |
| `build-893011-1792513509803091078` | Current strict fault runner passed twelve fixtures, including four malformed scalar-JSON cases | `fault-runner-strict.excerpt` |
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

## Current-source CPU-only checks

At the earlier branch head `3ec4a25d54baf755f85c734d2e0a6e3789c2b741`, DSH
builds on Benji passed the issue9 API, CLI, fault-runner and engine-config CPU
checks. After merging master `c099273` as branch revision
`421a6df91afde108d4ad0aa30341010da125510d`, DSH build
`build-661648-1792555441846020340` passed those checks again and passed the new
`test-flashcore-prefill` target. These are CPU-only results; they do not
complete the 2055-position oracle or the production-budget GPU matrix. See the
[post-merge CPU check excerpt](benji-421a6df-checks.excerpt).

A short local Benji run of `make -C kern checked_test` and
`./kern/checked_test` also passes on `421a6df`. It exercises the checked QSA API
at positions 2050 and 2051 for B512 and remains CPU-only. The compile emits
existing misleading-indentation warnings; full output and source hashes are
in the same excerpt.

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
| `59f2ef9f-7b5c-4db7-90f4-a6a6ddcbca58` | VRAM KV, per-token, same model scope and budget | Exit 0; nine complete records and predictions match, with twelve matching actual selected-ID captures at visible length 9. |
| `ee9b79e4-f01a-4738-a835-3eaa278ba88a` | Host KV, one persistent launch, same model scope and budget | Exit 0; all nine predictions on all six dies match, plus the final token's complete stage record. Twelve actual selected-ID captures match at visible length 9. Earlier tokens' stage buffers are not retained by this launch mode. |
| `b68e8906-5c5c-4a71-b4ff-0811648d1729` | VRAM KV, one persistent launch, same model scope and budget | Exit 0; every prediction on every die and the final token's complete stage record match. Twelve matching actual selected-ID captures at visible length 9. Earlier tokens' stage buffers are not retained. |

`.logs.json.gz` is a durable queue database export; `.stdout` is its readable
derived output. Optional resident trace collection was unavailable, not zero.
CPU/build wall time and device intervals are separate clock domains. No speedup
is inferred from concurrent CPU runs or from comparing these domains.

All four KV/launch placements pass their stated nine-position scopes. The
prebuilt C fault runner groups seven expected mismatch cases into one queued
reservation. GPUQ `1a7a5f7c-cf8d-47d7-90ad-bed1bd2904c0` exits 0 with all seven
expected rejections accepted: layer/logit/recurrent/convolution/PLE copies fail
at position 0; selected IDs fail at position 7; the VRAM single-launch logit
copy fails at final position 8. Each per-token case has eight matching frames
and one mismatch; the single-launch case has one mismatching final frame.
Every case reports the intended label, Make exit 2 / model exit 3, no
successful selected-ID capture, pipe EOF and no remaining live child-group
member. This is comparison rejection coverage using a one-bit perturbation of
a host readback copy. Complete durable logs (401 stdout documents starting at
ordinal zero) and readable output are retained in `fault-matrix-1a7a.*`.
The runner's binary SHA-256 is
`1293d4c93e4852459a6d3a0f3619517c568144cbe80d0827dd2ae2a22b70ff83`;
its C source SHA-256 is
`c7a55cfdeaf61531d263f58d4c2eebb35229a68515f2fc7846f96f248bf960e4`.
The uniform four-worker deep reference starts from position zero in
`par-e2ff-b512-t2055`, with 32-position durable checkpoints and source epoch
held fixed. Its initial eight-chunk routed job is
`build-352749-1792508656328957494`; it exits 0 after eight committed chunks,
256 positions, as retained in `cpu-deep-first256.excerpt`. This is a completed
batch, not a completed 2055-position reference. Continuation
`build-1157614-1792515841388210491` resumes the same files for the remaining
57 chunks, followed by full-stream validation, under a longer routed timeout.
Its producer and source-manifest hashes match the first batch exactly. A direct
checkpoint read later reported `next_position=1248` and
`record_offset=131918388` bytes. `records.bin` is 133609644 bytes, leaving
1691256 bytes beyond the durable offset; that suffix is not committed oracle
output and must not be counted as completed positions. The DSH build has no
confirmed terminal result, its last captured log reports the Sentry NFS timeout,
and its previously recorded `model_oracle` PID was absent on the later Archy
process check. No full-stream validation or 2055-position completion is claimed.

Before any continuation, the three source model shards must be copied off NFS.
They total 78869128864 bytes. Archy had 1855864832 bytes free on `/` and
4123766784 bytes free on `/mnt/stop-hook-hdd`, so the model shards do not fit on
either local filesystem. The existing records, checkpoint, IDs and receipt
total 283372362 bytes and fit on the HDD; those partial artifacts remain there.
The optional Q8_0 MTP file is another 4137429120 bytes and is not used by this
scalar oracle. No replacement long run was started.

In this follow-up, the current shell was on Benji. Its `/mnt/sentry-nfs` path
was an unmounted, empty directory, and its only disk had 36,785,856,512 bytes
free. The required 78,869,128,864-byte shard set could not be copied there, so
no oracle resume was attempted. The stock DSH profile also omitted Benji; a
temporary build-only profile routed the CPU checks above through DSH. Archy did
not resolve from this host, so no new GPU job was submitted.

The older source-36a
native comparison stream was retired with 96 committed positions; its data and
checkpoints remain available as partial evidence. Its first 32-position record
prefix and frozen checkpoint matched the O3 source-36a control through routed
comparisons `build-227440-1792507498960444325` and
`build-227441-1792507498960899876`; these are not a completed 2055-position run.

The separate intermittent paired engine failure is tracked in
[issue #109](https://github.com/nm-z/flash-sass/issues/109). Passing and failing
request controls remain in [the earlier record](../whole-model-3902787/README.md).

## Production-budget request matrix: failed scalar gate

GPUQ `fe513a53-f621-4dff-946c-61d2f5a73351` exercises the canonical production
engine independently of the unfinished CPU reference: 2055-position requests,
budget 512, snapshots at 2048, 2051, 2052, 2053 and 2054, both KV placements and
scalar/paired modes planned. It fails in the first VRAM scalar mode; later
modes are not reached.

The raw transcript has seven requests ending at position 2055, including
resumed suffixes from snapshots 2048 and 2051. Reset-B request 9 then reports
`completed=1057 requested=2055`. Stderr reports a CUDA illegal memory access on
die 4 and a failed attempt to rebuild the contexts. The private core exits 1
without a signal, timeout or interruption. The matrix rejects that exit before
accepting request comparisons, so the report's accepted-check count is zero.
This is a failed gate, with no scalar-oracle agreement claimed. It is tracked
separately in [issue #110](https://github.com/nm-z/flash-sass/issues/110).

The engine revision is `11ac712913234331088808284524a11effca7208`; an empty git
diff confirms `engine.c`, `prog.c`, `prog.h` and `qsa_state.h` match this PR's
current files. Its binary SHA-256 is
`aa9ebd387020f1382e5805e4fc8d790807d0067b941b9af15c26f25a447d7cfd`,
and it loads canonical cubin
`aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`.
The runner SHA-256 is
`594c55c9716a52462ef0f908b6ff769f89ce0fc9083f0a31c34dce7e23104554`.
Math/profiling overrides were cleared and no fault injection was enabled.
All six original holder PIDs remained alive in the post-run observation, with
lifetimes spanning the test; they were not replaced during that observation.

The complete GPUQ export, report and child stdout/stderr are retained as
`production-b512-fe51.*`. Document 3656 supplies stderr, 3658 supplies stdout
and 3659 supplies the JSON report. These records retain the distinct request
timing observations and do not establish a performance comparison or the
offending instruction's identity.

The count 1057 is the last observed completion count from die 0, not a precise
faulting-token report for die 4. A bounded Archy kernel journal excerpt records
Xid 31 for that core PID on BDF `0000:0b:00`: an MMU PDE read fault. It does
not establish the source instruction or code cause. The excerpt, a failed
full-segment recorder attempt that reaches the five-minute kernel limit,
and subsequent bounded-window diagnostics are isolated in
[draft PR #112](https://github.com/nm-z/flash-sass/pull/112). That recorder's GPU
fixture and nine-position scalar-oracle canary pass their declared scopes;
its long request matrix remains pending. This PR retains the canonical cubin
and does not adopt that instrumentation as a runtime fix.

The bounded-window instrumented B512 run `ff6ae989` completes its full request
transcript without a CUDA error. Its verifier accepts two state fingerprints
then rejects `save-2048-A` against `fresh-A`; the queue job exits 2. The
instrumented matrix is not accepted and this outcome does not reproduce the
original device error. Complete records and limits remain isolated in draft
PR #112.
