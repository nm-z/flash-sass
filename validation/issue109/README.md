# Generic pair-body grid-barrier diagnostic

Refs [issue #109](https://github.com/nm-z/flash-sass/issues/109) and
[issue #9](https://github.com/nm-z/flash-sass/issues/9).
This is an experimental discriminator for the intermittent paired engine
failure. No root cause or successful fix is established by this source alone.

The unmodified generator places a CTA barrier and device memory fence between
the two bodies of a generic paired phase. The C postprocessor adds the existing
device-wide generation barrier at that boundary. Fused pair bodies and scalar
phases bypass the insertion. The exchange counter and peer flags remain
published once after both bodies; the insertion does not publish them.

The transform requires a unique generic-pair branch, its nearby mode test and
the following fence. It refuses existing output, aliases, altered anchors and
already patched input. The build verifies the original cubin against the
pinned baseline before applying the transform. The final kernel receipt binds
both generated SASS files, the C transform, op sources and private MaxAs files.
The inserted barrier uses the framework's generation storage, polling and
launch-expiry logic, including predicate-delay padding before its exit branch.

Builds and CPU structural checks run remotely through DSH:

```console
dsh build --host archy --require-tool cargo --require-tool cc -- make -f validation/issue109/Makefile pair-build pair-check --artifact kern/die.cubin --artifact kern/die.sass --artifact kern/issue9-kernel.sha256 --artifact kern/libengine.so --artifact kern/flashcore --artifact kern/model_test --artifact validation/issue9/bin/engine_check --artifact validation/issue109/bin
```

GPU request controls use the inherited `engine-matrix` recipe through Archy's
FIFO, with the diagnostic engine's full `ENGINE_REVISION`. The Make recipe
resolves the six current holder PIDs after admission, matching checkpoint path,
full layer scope, device, manifest name and split arguments; duplicate or
missing matches fail before the private runner starts. Explicit `HOLDER_PIDS`
can override that lookup. The private test children import the holders' existing
allocations. Keep the diagnostic cubin in its separate checkout, retain the
unchanged baseline controls and record passing and failing runs. A bounded
series of passing repeats would provide evidence about this discriminator;
it would not establish the original failure's instruction-level cause.

## Measured build checks

| Routed Archy build | Result |
| --- | --- |
| `build-660111-1792511465775211408` | Original baseline hash matched; transformed SASS assembled with 249 registers. Engine/client builds completed. The final structural-test compilation then rejected an unused test constant under `-Werror`, so the overall request exited 2. |
| `build-674345-1792511587574712206` | After removing that unused test constant, strict C builds and the real generated-source structural checks passed, including malformed/missing/duplicate anchor rejection, already patched input, existing output and alias preservation. Exit 0. |

The transformed cubin SHA-256 is
`c2d8fe082adf54dfb35fcd6ef522baa20cf32f9660026ef7a98e40eab6f7e0e8`.
The engine SHA-256 is
`aa9ebd387020f1382e5805e4fc8d790807d0067b941b9af15c26f25a447d7cfd`;
it reports source revision `11ac712913234331088808284524a11effca7208`.
The test-only constant removal does not change either production artifact.
The final receipt also passed verification after copying the artifacts into the
separate Archy checkout. Build excerpts retain both the initial failure and
the corrected successful check.

## Matched stock-kernel control

GPUQ `7ca22683-676f-43c5-b9dd-cf63c78b5152` uses the same engine/client
artifacts and source revision as the diagnostic, with the stock cubin
`aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`.
Canonical graph profiling overrides are cleared. The six holder PIDs are
resolved after admission; the serving owner had restored them while the job
waited, so the original PID list would have been stale.

The control reproduces the intermittent failure: VRAM scalar, VRAM paired and
host scalar each pass all 16 request checks. Host paired passes eight checks,
then `save-3-A` has a GPU state fingerprint different from its A reference.
All four private children exit 0, with no timeout, interruption or signal.
The matrix therefore exits nonzero. The durable logs and report are retained
as `matched-stock-7ca2.logs.json.gz` and `matched-stock-7ca2.results.json`.
This report establishes the fingerprint mismatch; it does not identify the
first divergent mathematical operation.

## Transformed-kernel result

GPUQ `8f6d8ae4-49ba-445c-9691-f06de1d55589` loads the transformed cubin
`c2d8fe082adf54dfb35fcd6ef522baa20cf32f9660026ef7a98e40eab6f7e0e8`
with the same engine/client binaries, declared source revision, profiling
settings and workload. VRAM scalar passes all 16 requests; VRAM paired passes
two checks, then `save-1-A` has a different state fingerprint from its A
reference. Both children exit 0 without a signal, timeout or interruption.
The runner stops before the host modes after this mismatch.

The extra pair-body grid barrier does not make this request control pass.
These two failing matrices do not establish that their mathematical divergence
has the same instruction-level cause. The transform stays isolated as an
experimental discriminator. Its durable export and report are
`genbar-8f6d.logs.json.gz` and `genbar-8f6d.results.json`; scalar and failing
paired child outputs are retained beside both run records. PR #105 retains
the canonical kernel and continues its independent scalar-oracle work.

## PLE mapped-row visibility discriminator

The next isolated variant addresses a different hypothesis. The CPU stager
primes one position before launch, then writes mapped-host row payloads and
release-publishes a per-position ready value while the kernel runs. The PLE
loader polls ready with `LDG.E.CV`, then reads the payload with ordinary
`LDG.E.U16` loads. The rows occupy 1440 bytes per position.

The GNU gawk transform validates the unique PLE block, all 45 ordered payload
loads and their halfword offsets before emitting changed Assembly. It makes
those payload loads `LDG.E.CV.U16` and adds `MEMBAR.SYS` after the ready wait.
The Perl generators are unchanged. NVIDIA describes `.cv` as refetching
system-memory cache lines and states that cache operators do not change memory
consistency; see the [CUDA 11.4 PTX cache rules](https://docs.nvidia.com/cuda/archive/11.4.4/parallel-thread-execution/index.html#cache-operators).
The GPU fence and cache change are an experimental discriminator. They do not
by themselves prove a host/device synchronization contract or the cause of
the observed request mismatch.

The first routed build request, `build-1194327-1792516174768826308`, was refused
during staging admission before compilation. The fsync deadline/reporting
feedback is [DSH #2114](https://github.com/nm-z/nates-dsh/issues/2114).
Retry `build-1215643-1792516366759787392` exits 0: the stock baseline hash
matches, the transformed kernel assembles with 249 registers, and disassembly
contains exactly 45 `LDG.E.CV.U16` loads. The excerpt is retained as
`ple-visibility-build.excerpt`.

The variant cubin SHA-256 is
`a57514187a81a017d7b08da31355182019468514a0e3b6cf0400dd491eca56b9`;
the transform's SHA-256 is
`744b19ed917f970b8cc214bdfeb622f7fab60bdd31293780f952aadb14b55485`.
The request control reuses the matched engine/client binaries and explicit
engine revision above. GPUQ `81654b75-8185-424e-97e6-e4d472d2ff7c` still fails:
VRAM scalar, VRAM paired and host scalar each pass all 16 checks; host paired
fails its first `fresh-A` fingerprint comparison. All four children exit 0
without a signal, timeout or interruption. The cache/fence change is therefore
not sufficient to make this control pass. Durable logs, the report and the
host scalar/paired outputs are retained as `ple-visibility-8165.*`.
