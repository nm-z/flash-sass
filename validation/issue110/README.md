# Last observed phase diagnostic for issue #110

Refs [issue #110](https://github.com/nm-z/flash-sass/issues/110) and
[issue #9](https://github.com/nm-z/flash-sass/issues/9).
The canonical B512 request gate reports a CUDA illegal memory access on die 4
after repeated requests and snapshot/resume. Its last observed die-0 completion
count is 1057; the faulting die's token and instruction are not established.
The failed run and identities are retained in
[the #9 record](../issue9/whole-model-e2ff58f/README.md).

This checkout adds a diagnostic, not an established runtime fix. A small mapped
host buffer records one 64-bit progress word per CTA on each selected die. The
fault path reads those host words directly, avoiding a device readback after a
CUDA error. The values are last observations and can be stale; they are not
fault PCs or a synchronized snapshot of all CTAs.

## Record format

Each die has sixteen naturally aligned 8-byte slots. The control block at
`ctl + 0xe8` supplies that die's mapped pointer. The proposed dispatch hook
stores this word from thread zero of each CTA:

| Bits | Meaning |
| --- | --- |
| 63..32 | Kernel token base |
| 31..5 | Physical primary descriptor index plus one |
| 4..0 | Pair mode: 0, 8 or 24 |

A zero word means no observation. Mode 8 names the generic first body; mode 24
names a fused pair. Mode 0 names a scalar body or a generic second body, which
the uploaded primary descriptor flags distinguish. The host retains opcode,
unit count and flags for every uploaded descriptor and rejects tags that do
not fit the segment, descriptor range or pair layout.

Recording is restricted to kernel token bases `[1024,1280)`. Both bodies of a
pair share its base, so the second body's decoded token can be 1280 when its
base is 1279. The request still executes all 2055 model positions. Segments
outside the recording window require zero tags; paired launches account for
the parity of their first base at the window boundary.

The dispatch transform preserves the model's live registers and saves/restores
the incoming per-thread P4 predicate. Pointer and shared-memory loads use
separate dependency slots, both waited before use. It adds no model descriptor, grid barrier or
memory fence. Its memory accesses and instruction timing can still alter the
failure's behavior. PR #105 retains its canonical kernel.

## Host lifecycle

The engine enables the helper through an explicit `--phase-probe` runner
option, applied after inherited engine overrides are cleared. The runner
requires the engine config to report `phase_probe=1` and records that option in
its JSON report.

Before a launch, the host records the request and segment bounds and clears
the progress slots. Uploading another program marks the old segment inactive.
After successful synchronization and the kernel-limit check, the helper checks
all expected CTA slots for segments that execute a base inside the window.
Missing or invalid records exit 71. On a CUDA error, the enabled helper reports
the host observations and exits 70 before a CUDA error-string query or context
recovery. It does not replay the request. Active mapped storage is not freed
through the ordinary cleanup path.
The persistent kernel's own five-minute limit reports the still-active host
observations and exits 72. That reason is distinct from a CUDA API error.

[CUDA 11.4's error table](https://docs.nvidia.com/cuda/archive/11.4.0/cuda-driver-api/group__CUDA__TYPES.html)
describes illegal-address errors as leaving the process unusable for further
CUDA work and requiring process termination/relaunch. Its
[mapped-memory rules](https://docs.nvidia.com/cuda/archive/11.4.4/cuda-c-programming-guide/index.html#mapped-memory)
describe preservation of naturally aligned 1-, 2-, 4- and 8-byte device accesses
to host memory, while requiring synchronization for host/device read and write hazards.
This diagnostic makes no claim that every pending store becomes observable
after a fault.

## Checks and current status

Initial routed build `build-2276488-1792524088849213919` passes 21 CPU-only
decoder cases using literal wire-format tags. No CUDA initialization occurs in
that executable. Initial engine/client build
`build-2290672-1792524194978680204` passes 24 configuration fixtures, including
required, disabled, enabled and unexpected probe settings. The excerpts retain
the source hashes of those initial checks; the public host-observation accessor
and final Assembly integration still require the combined build.

```console
dsh build --host archy --require-tool cargo --require-tool cc -- make -f validation/issue110/Makefile phase-check
```

## Combined build and initial GPU unit gate

Routed build `build-2435895-1792525343874000357` passes the complete Assembly,
engine/client/fixture build plus 21 decoder and 24 configuration checks. The
source revision is `222ed2e9683cf6631dc24abf16e310e90ae4ef3d`; the excerpt is
`combined-build.excerpt`. The diagnostic cubin SHA-256 is
`582e02f224838a468264ba688869b9723fb7065263683cef69db8688c9051334`,
the engine SHA-256 is
`2fdcd14bca0915412c896a76763299621bf66b3cb543a39870cfcb98943573c7`,
and the GPU fixture SHA-256 is
`34c1db37cfd4140b73dcb95571649ce82febf843c41cd46d742a903deea36ea0`.

GPUQ `09d4da30-632a-4724-b305-7f1e674d8d9a` passes the actual GPU fixture:
null-pointer copy, scalar tag/copy, generic second-body tag/copies, and fused
BF16 tag/results. All sixteen CTA words match their expected value; destination
guards are intact, and each fused BF16 output equals 160. The test context uses
the engine's `CU_CTX_SCHED_SPIN` flags. The overall job then exits 2 before the
model canary executes, because the deployment omitted generated `kern/dieops.h`
from the artifact copy. Its complete stream is retained as `canary-09d4.*`.

Repair build `build-2490356-1792525674895186565` generates that header through
DSH. Its SHA-256,
`5032a1e6012f19f30b922ede9fdb4662a19f17080163506b4edc5290f9f51356`,
matches the frozen reference checkout's header. The successful GPU unit gate
is not repeated. Model-only GPUQ
`73c3fb17-f733-4508-949d-6c101ca895da` then exits 0: all nine complete frames
report scalar agreement and all nine tokens match. The case uses host KV,
budget 1, per-token launches and numerical source epoch `e2ff58f...`; actual-ID
JSON capture is disabled. The kernel's null-pointer dispatch path is exercised
with the unchanged consumer SHA-256
`3c0dafd512307c58091202472a0285e7f5a997bb58706eb39938fd032584832c`.
Its complete export and output are `oracle9-73c3.*`.

The deployment must copy the generated opcode header as well as the cubin,
engine/client, receipt, transformed SASS and GPU fixture. The build recipe now
includes that header in the receipt. Example routed artifact list:

```console
dsh build --host archy --require-tool cargo --require-tool cc --require-tool gawk -- make -f validation/issue110/Makefile ENGINE_REVISION=222ed2e9683cf6631dc24abf16e310e90ae4ef3d phase-build phase-check --artifact kern/die.cubin --artifact kern/die.sass --artifact kern/dieops.h --artifact kern/issue9-kernel.sha256 --artifact kern/libengine.so --artifact kern/flashcore --artifact kern/model_test --artifact validation/issue9/bin/engine_check --artifact validation/issue110/bin
```

The first instrumented production-budget reproduction is GPUQ
`16a16fe1-38b0-40f7-b3eb-2392b41bc6c4`: 2055-position requests, block budget
512, snapshots 2048/2051/2052/2053/2054 and explicit `--phase-probe`. It uses
the same pinned binaries and has passed its receipt check, including the
added generated-header entry. It exits 2 after the first request reaches the
kernel's five-minute limit. The protocol reports completion count 1358; no
illegal-address error is recorded. The driver accepts zero matrix checks.
The full-segment recorder calls its successful-segment helper before checking
the limit, so that version does not dump individual CTA words on this exit.
The retained scalar stdout/stderr and JSON report are `probe-16a1.*`.

The subsequent source limits recording to `[1024,1280)` and moves the successful
segment check after the limit check, with the host-only exit-72 dump described
above. The GPU fixture now covers enabled but filtered copies before/after the
window, a pair straddling its lower boundary, both scalar endpoints, generic
second-body decoding, null-pointer dispatch inside the window and fused BF16.
These changes require a fresh routed build and GPU checks; no successful fault
reproduction or runtime fix is claimed.

## Bounded-window build and GPU gate

Source `0f60a77cc4dec4ab307a86d0fd44dcd36cdffd4c` is built by DSH
`build-2738671-1792527179092277722` (exit 0), including all 21 decoder and
24 engine configuration fixtures. The build excerpt is `window-build.excerpt`.
The diagnostic engine reports revision `0f60a77`; its content is bound by
the full source revision and binary SHA-256 below.

| Artifact | SHA-256 |
| --- | --- |
| Diagnostic cubin | `f0e9bde32056612cc1f91e16cc38928a07d99c43a57bb3e2f8c41024589c2d8d` |
| Engine | `45b0e9e2ea5d413cd9d24cd3671c40f01230f10955a416b41c7baa4ef0c2a576` |
| Core | `0257e1fbf4a6505bc54ed230b93abe22d5f21ed66daa22345a85b73f6a673d06` |
| GPU fixture | `5070f4a2482f71a598fe78d06b8c9423f97f52ee5d467ceb771395177f29241e` |

GPUQ `bbe8d096-8f94-4893-8c27-94208f546af2` exits 0. The actual GPU fixture
passes the filtered, null-pointer, scalar endpoint, generic pair and fused
BF16 cases above. All expected sixteen-CTA tag sets, copies and output guards
match. The following host-KV/B1/per-token model canary matches all nine
complete scalar records and predictions. The unchanged GPU consumer's hash
remains `3c0dafd512307c58091202472a0285e7f5a997bb58706eb39938fd032584832c`.
The queue export and output are `window-canary-bbe8.*`.

The unchanged bounded-window binaries ran the full 2055-position, B512
request/snapshot matrix in GPUQ `ff6ae989-f1d8-42f2-9b41-8ea92d4914ee`.
The core child exits 0 without timeout or CUDA error after emitting 19 state
fingerprints and the planned snapshot/resume requests. The verifier checks two
fingerprints, then rejects `save-2048-A` because its fingerprint differs from
the earlier `fresh-A` reference. The queue job exits 2 and the full request
matrix is not accepted. Comparing raw statehash records gives 76 differing
fields out of 234; the first stream-ordered mismatch is die 0, layer 33. This
is a state-consistency failure under the
instrumented run, not a reproduction of the prior CUDA illegal access. The
complete child streams and report are `window-b512.*`.

## Deliberate CUDA-error handler fixture

Source `2ca99e1d9620901b067af9f21089c9b849f3b790` adds a two-child GPU fixture
for the host-only failure exits. It executes one valid in-window copy before
calling the recorder's host-limit exit 72, then executes an XCOPY with a
deliberately unmapped source read in a separate child. That child requires
`CUDA_ERROR_ILLEGAL_ADDRESS`, confirms at least one expected mapped-host tag,
and exits through the recorder's error-70 dump without a post-error CUDA call.
The parent captures and validates both children’s bounded stderr dumps, exit
codes and all 16 exact CTA tag records; an unexpected read result, missing dump,
wrong tag or wrong exit fails the fixture. The limit child also completes
`copy_case`: every tag, the copied test output and both destination guards pass
before it calls the synthetic host-limit exit. The fixture’s HtoD and Memset
writes target its device control, descriptor, input and output test allocations;
the dispatch probes write only the diagnostic mapped tag buffer. The synthetic
host-limit fixture does not wait for, or claim to reproduce, an actual kernel
time-limit event.

The fixture and CPU decoder check build on Benji as
`build-3444639-1792529270056721202` (exit 0); no Perl/Ruby/Bash source or build
step is used. The fixture binary SHA-256 is
`d5062d8013b6b8f9591927cf2025b15aea44d1e08e3c9fe12e0ea4e8adff2f99`; the
copied opcode header SHA-256 is
`5032a1e6012f19f30b922ede9fdb4662a19f17080163506b4edc5290f9f51356`. GPUQ
`7eab82bf-78b3-42d5-9adc-5ddead179716` exits 0. Its first child receives CUDA
error 700 on the deliberate invalid read, reads all 16 host observations,
reports zero CUDA calls after the error, and exits 70. A second child exercises
one in-window copy and exits through the synthetic host-limit dump at 72. The
parent in this version validated both child exit codes; the retained streams
show the dump records. Complete output is `failure-handler.gpu.*`.

Luna's read-only review found that the parent did not check either dump's
contents and that the limit child exited before `copy_case` checked the copy
bytes and destination guards. Revision `18bd65ec71f4c27f0f38fc41ce44385e7c105014`
captures each child's stderr through a bounded pipe, then checks the dump
header, terminator and all 16 exact CTA records. The limit child now completes
`copy_case`, verifies the copy, tags and guards, and calls exit 72 afterwards.
Benji build `build-3971517-1792530780129130442` passes the strict build and 21
CPU decoder checks. The rebuilt fixture SHA-256 is
`1b46db65b8c21694d4db2dc8b94a252ff97f3be422e6f32dd0f6186ac12ea6f2`. One GPUQ
rerun of the corrected assertions on GPU6 is pending.

The first deliberate fault-handler GPUQ run's pre/post telemetry
for NVML index 6 (documents 66610 and 66650 in that GPUQ job) maps it to BDF
`12:00` and the UUID below; it remained listed
with 7,547 MiB free, no active compute process and zero utilization. Its
volatile and aggregate ECC totals were zero before and after; retired-page
counts were zero before and after, with no pending retirement or blacklist.
The bounded kernel journal shows the expected Xid 31 at address zero on this
GPU and no Xid 79 or fallen-off-bus message. No reset was required.

## Canonical fault's kernel journal

`xid31-kernel-journal.log` retains a bounded read-only Archy kernel journal
excerpt around the canonical `fe513a53...` job. At epoch `1792522477.212058`,
Xid 31 identifies an MMU PDE read fault on BDF `0000:0b:00`, PID `3003319`.
The retained pre/post NVML XML joins that BDF to UUID
`GPU-906e6ede-6cde-d0dd-5110-d1bb8bf2458d` at NVML index 4. The saved data does
not map that UUID to an engine CUDA ordinal, so the `engine: die 4 failed`
message is not attributed to the same adapter by ordinal. The Xid does not
identify the source instruction or underlying code cause. The pre/post XML
records are documents 3199 and 3653 in the `fe513a53...` GPUQ run.

The deliberate GPU6 error-handler fixture also coincided with an Xid 31 on
BDF `0000:12:00.0`, address zero, PID `3667902`. Its bounded log is
`xid31-gpu6-fault-fixture.log`. GPUQ inventory maps BDF `12:00` to NVML index 6
and UUID `GPU-431371bd-b482-96b2-4b70-04f83c6d075b`; post-run sampling showed
zero utilization and no compute process. The PIDs differ; the GPUQ evidence does
not join the fixture Xid PID to the child process. The events are separate and
neither identifies the other’s source or cause.
