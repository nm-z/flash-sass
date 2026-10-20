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
After a successful synchronization, the helper checks all expected CTA slots.
Missing or invalid records exit 71. On a CUDA error, the enabled helper reports
the host observations and exits 70 before a CUDA error-string query or context
recovery. It does not replay the request. Active mapped storage is not freed
through the ordinary cleanup path.

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

The planned GPU gate checks exact per-CTA progress words and copy/pair outputs,
then compares one complete nine-position model case against the existing CPU
oracle to check the changed dispatch path. A subsequent production-budget
reproduction will use separate source and binary identities. No GPU execution
or successful reproduction is claimed for this diagnostic yet.
