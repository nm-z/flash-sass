# Issue 15 timing validation

## Current status: queued target retry on 2026-10-18

Frozen source and target build revision:
`7f52b0886829236a833c637ea3a87777b76648fe` (host implementation `bec699c`).
The current PR source was transferred from engi with `git archive` to
`/home/nate/claude/flash-sass-resolve/timing-7f52b08` on archy. The ordinary and
traced cubins, engine library, core, frontend, and validation binaries all built
successfully (exit 0) with archy's installed CUDA toolchain and driver library.
The [build log](20261018-queued/cuda-build.log.gz) and
[artifact hashes](20261018-queued/artifact-sha256.txt) are recorded. This build
does not establish a successful GPU request.

The shared lock was available at 05:19:29 UTC. The current-revision preflight
acquired it at 05:20:56 UTC and released it after recording a **different
blocker**: the six required model GPUs still have existing allocations. The
recorded free memory for ordinals 0–5 is respectively **52, 16, 77, 7, 3, and
65 MiB**. The first rejected compute process is weight-holder PID **974018**;
the snapshot also contains the other weight holders, resident core
**3575268**, and frontend **753710**. The
[occupancy and selected process evidence](20261018-queued/occupancy-1.log)
and [raw GPU queries](20261018-queued/gpu-preflight-1.log) were captured while
the validator held the shared lock. No existing process was stopped or
reconfigured, and no owned core or request was launched.

| Retry, UTC on 2026-10-18 | Invocation | Measured result |
| --- | --- | --- |
| 05:20:56–05:20:57 | Current-source preflight | **BLOCKED, exit 75** after acquiring the lock; resident GPU allocations ([result](20261018-queued/preflight-1-results.json)) |
| 05:23:52 | Full driver, current target build | **BLOCKED, exit 75** before GPU queries; lock unavailable ([result](20261018-queued/attempt-1-results.json)) |
| 05:26:04 | Full driver retry | **BLOCKED, exit 75** before GPU queries; lock unavailable ([result](20261018-queued/attempt-2-results.json)) |

At 05:24:53 UTC, the observed lock owner was **PID 847383**, running
`flock -w 300 -E 73 /tmp/flash-sass-engine-validation.lock bash --noprofile --norc -i`;
see [the owner observation](20261018-queued/lock-owner-2.log). The earlier
issue-19 validator PID is not the owner observed in this retry. The later
lock-blocked attempts did not remeasure GPU occupancy.

**PR #50 remains draft.** A permitted setup with enough capacity on the
required GPUs is still needed in addition to the shared lock. The real request,
event reconciliation, and observation/transport overhead gate has not passed.
The [retry summary and commands](20261018-queued/results.json) distinguish
these blocked attempts from validation success. The built source and existing
guarded driver are staged for that setup.

## Previous resumed review on 2026-10-18

Current host implementation: `bec699ca07f6cf1ac7452a93698d41509aa58576`.
The resumed review found that a failed host program allocation returned an
error while retaining `error_code: 0` and trace status `running`. The fix gives
that failure its own `-4` code and `allocation_error` status, measures build
time, leaves unobserved counts and run time null, and emits an explicit CLI
error. It does not trigger a device-context rebuild.

The CPU regression forces each of the three program allocations to fail and
checks the real session's timing/error order and stale-trace rejection. It
failed against the preceding implementation (exit 134), then passed with the
fix (exit 0). The complete current commands, exit codes and artifact hashes are
in [the resumed evidence](20261018-resume/results.json).

| Current check | Result | Evidence |
| --- | --- | --- |
| Canonical count, timing and device-integer fixtures | PASS, exit 0 | [Timing fixtures](20261018-resume/timings.log) |
| Real session, callback/cancellation and forced allocation failures | PASS, exit 0 | [Session fixtures](20261018-resume/session.log) |
| Actual frontend JSON/SSE fixtures | PASS, exit 0; 33 groups, 30 transport cases | [Frontend TAP](20261018-resume/http-fixtures.tap.gz) |
| Shared lock guard | PASS, exit 0; 16 assertions | [Guard TAP](20261018-resume/engine-guard.tap) |
| Prefill planner | PASS, exit 0 | [Planner fixture](20261018-resume/prefill-plan.log) |
| Production engine library from the committed source, CUDA stub link on engi | PASS, exit 0 | [Build log](20261018-resume/engine-build.log), [hashes](20261018-resume/artifact-sha256.txt) |
| Final target preflight at 2026-10-18T03:32:14Z | **BLOCKED, exit 75**; lock unavailable, zero GPU queries and zero cores | [Preflight result](20261018-resume/target-preflight-final.json) |

The shared lock on archy was held by PID 4010221, the issue-19 model validator;
see the [targeted lock-owner observation](20261018-resume/lock-owner.txt).
The first preflight at 03:21:26Z and the final check both returned 75. Neither
check acquired the lock or measured current GPU occupancy. The older resident
allocation observations below must not be treated as a fresh occupancy check.

Both preflights used the existing frozen `5f6c120` source on archy. The guard and
device source are identical to the current implementation; only the guard was
invoked. The new host implementation has not run a target-GPU request. The
existing workers and shared lock inode were left in place.

**PR #50 remains draft.** Its real target-GPU request, raw device/host event
reconciliation, and separate measured observation/transport overhead remain
unresolved. CPU fixtures and a successful build do not satisfy that gate.

## Earlier validation at 5f6c120

The preceding timing implementation is in commit
`5f6c120bab7b5cd2fc734575a127b40a9394b6ff`, which includes the callback-return fix
`9be16f4` and merges default-branch commit
`bf53cfd7c2b89972fa6adb2235531afa2da05a63`.
Later evidence/documentation commits do not change the measured implementation.

**PR #50 remains draft. No successful real target-GPU request or measured
observation/transport overhead is claimed.** The source compiles and the CPU
checks pass, but the shared-host preflight blocks a new core while other
workers own the GPUs.

| Check | Result | Evidence |
| --- | --- | --- |
| Canonical counts, rates, aliases, errors, aborts, device integer stamps | PASS, exit 0 | [CPU fixture](20261018-followup/merged-timings.log) |
| Real session parser, input EOF, negotiated cancellation, callback-return ordering | PASS, exit 0 | [Session fixture](20261018-followup/merged-session.log) |
| Actual frontend with deterministic engine events, JSON and SSE | PASS, exit 0; 33 top-level fixture groups, including 30 transport cases | [Frontend TAP](20261018-followup/merged-http-fixtures.tap.gz) |
| Shared lock, inherited child lifetime, unchanged lock inode, busy/invalid observations | PASS, exit 0; 16 assertions | [Guard TAP](20261018-followup/merged-engine-guard.tap) |
| Merged prefill planner | PASS, exit 0 | [Planner fixture](20261018-followup/merged-prefill-plan.log) |
| Frozen-source CUDA build on archy | PASS, exit 0 | [Build log](20261018-followup/cuda-build.log.gz), [artifact hashes](20261018-followup/target/artifact-sha256.txt) |
| Session and real-tokenizer output regression on archy, without GPU initialization | PASS, both exit 0 | [Command output](20261018-followup/target/commands-output.log) |
| Target request preflight | **BLOCKED, exit 75**; no core started | [Machine-readable result](20261018-followup/target/results.json) |
| Real GPU event/count/rate reconciliation and observation/transport overhead | **Not run** | Blocked before a request |

The callback regression exercises the actual engine delivery path with a callback
that requests cancellation. Its first cancellation-driver shim asserts that
`callback_end` has already been recorded. GPU initialization, launch, asynchronous
copy and synchronization are trapped in this CPU fixture. This prevents
synchronous cancellation propagation from being included in callback duration.

The merge retains `prefill_chunk`, `last_prefill_chunks`, the run-stat counter,
planner tests and prefill documentation, alongside the request timing fields.

## CPU commands on engi

The first local link attempt could not find `-lcuda`. The successful CPU-only
session build uses CUDA's stub library, with a private loader path; it does not
create a GPU context. Build output is in
[merged-cpu-build.log](20261018-followup/merged-cpu-build.log).

```bash
make -C kern 'CC=cc -L/opt/cuda/targets/x86_64-linux/lib/stubs' \
  timings_test flashd_timing_test flashd trace_session_test output_tokens_test prefill_plan_test
stub_dir=$(mktemp -d /tmp/issue15-cuda-stubs.XXXXXX)
ln -s /opt/cuda/targets/x86_64-linux/lib/stubs/libcuda.so "$stub_dir/libcuda.so.1"

./kern/timings_test
LD_LIBRARY_PATH="$stub_dir" ./kern/trace_session_test
./kern/prefill_plan_test
perl validation/issue15/engine_guard_test.pl
perl kern/flashd_timing_test.pl
perl -c kern/timings_core_check.pl
perl -c kern/timings_http_real.pl
perl -c validation/issue15/run_target.pl
cc -std=gnu99 -O2 -fsyntax-only kern/engine.c -Iref -I/opt/cuda/include
git diff --check

rm "$stub_dir/libcuda.so.1"
rmdir "$stub_dir"
```

All successful commands above exited 0. The session and tokenizer fixtures also
passed on archy using its installed driver library without initializing CUDA.

## Frozen-source target build and shared-host check

The source was transferred with `git archive` of the full implementation commit
to `/home/nate/claude/flash-sass-resolve/timing-5f6c120` on archy. It was built with:

```bash
source_dir=/home/nate/claude/flash-sass-resolve/timing-5f6c120
make -C "$source_dir/kern" CUDA=/opt/cuda PTXAS11=/opt/cuda-11.4/bin/ptxas \
  'MAXAS=perl -I/home/nate/claude/flash-sass/maxas/lib /home/nate/claude/flash-sass/maxas/bin/maxas.pl' \
  FLASH_REV=5f6c120bab7b5cd2fc734575a127b40a9394b6ff \
  die.cubin die_trace.cubin libengine.so flashcore flashd timings_test \
  flashd_timing_test trace_session_test output_tokens_test prefill_plan_test
```

Build exit: **0**. The ordinary and traced cubin SHA-256 hashes exactly match the
earlier build at `a2d339b`; the callback-return change affects host code.
The new engine-library hash is
`92715386df4ebfbeccfd65809395ace7d6617d45d222f16971e9b38b746ccd1b`.
The old generated-code comparison also matched all 45,330 ordinary instructions
against base `a1b4064` with tracing disabled (instruction-stream SHA-256:
`3c225681ce0509a02cd585e6c014393f186f5ff044c685d9ff72b8f68c877a29`).

The following ran on archy:

```bash
perl "$source_dir/validation/issue15/run_target.pl" \
  --source "$source_dir" \
  --out /home/nate/claude/flash-sass-resolve/timing-5f6c120-preflight \
  --source-sha 5f6c120bab7b5cd2fc734575a127b40a9394b6ff \
  --preflight-only
```

Exit: **75**, status **BLOCKED**. The runner acquired
`/tmp/flash-sass-engine-validation.lock` on archy at
`2026-10-18T00:21:57Z` and released it at `00:21:58Z`.
Full `nvidia-smi`, GPU inventory, compute-process query and `ps` each exited 0.
The recorded core list is empty.

The process snapshots contain the existing six weight holders, fence, resident
core PID 1575007 and frontend PID 1578451. The six model GPUs have respectively
52, 16, 77, 7, 3 and 65 MiB free. Zero utilization does not make those allocations
available. No existing process was stopped, reloaded or reconfigured.
[Selected process rows](20261018-followup/target/preflight-processes-selected.log)
retain the relevant identities; the complete local `ps` snapshot is not
published because unrelated command arguments are outside this evidence.

## Runner policy and remaining gate

`run_target.pl` acquires the shared lock before occupancy checks, checks again
before each owned core launch, and releases its descriptor after owned children
are reaped. It never unlinks the lock inode. Owned exec children inherit the
lock so an unexpected supervisor exit cannot immediately admit a second
validator while those children remain alive. Query failure, malformed
observations, existing compute processes (including resident weight holders),
or a worker found in `ps` prevent launch. Blocked runs exit 75; failed
validation exits 1. `--preflight-only` never reports request validation PASS.

This conservative policy currently blocks the recorded shared IPC layout.
A permitted validation setup with sufficient memory and without conflicting
processes is still needed; this PR does not unload other workers or provide a
bypass for their resident allocations.

When a permitted owned setup is available, run the same driver without
`--preflight-only`, using a new output directory. The CLI and HTTP checkers
require the source revision to match the resident engine and compare
request-correlated raw device and host events, generated counts, and named
rates. Their output reports callback and transport intervals separately and
never subtracts device and host epochs. See [TIMINGS.md](../../TIMINGS.md) for
the individual commands and clock definitions.

Until those real request checks pass, the issue's final acceptance item remains
open and the PR must stay draft.

## Historical startup failure

[20261017-a2d339b](20261017-a2d339b/) retains the earlier implementation's CPU
fixtures, toolchain, successful target build, and failed owned-core startup.
At `2026-10-17T22:59:34Z`, that core exited 1 before any request:
`cannot create a context on die 0`. The old runner returned 25.
A small diagnostic using the same context API reported
`CUDA_ERROR_OUT_OF_MEMORY` on ordinal 0 (exit 1) and success on the spare ordinal 6
(exit 0). Its source is [context_probe.c](context_probe.c).
These are historical results from before the shared-lock update and are not
a successful model request. They were not rerun during the occupied-host
follow-up.
