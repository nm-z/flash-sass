# Issue 14 follow-up validation

## Scope of this evidence

This directory records the follow-up to merged PR #32, whose `Refs #14`
link did not claim a completed matched benchmark. The original issue, issue
comments (none), PR conversation and Files changed resource were opened.

The new branch started from default branch `master` at
`3d57d734d63aa74f167a6b5a02c616ddc018d2ff`.

The GPU preflight is a read-only availability check, not an inference request,
throughput benchmark or validation of the new executable. It ran on `archy`
through the connected Codex session on `engi`. The shared lock was acquired
before `nvidia-smi` and `ps`, and released afterward. No process was stopped.

## CPU validation

The checked implementation commit is
`9db8780e1caf16897db9d45c1ab2b36234080fb3`, with a clean source tree before
validation. The subsequent evidence commit only adds these records and this
explanation. `cpu-validation.json` preserves exact command argument arrays,
individual exit codes, source-file hashes, log hashes and the linked artifact
identity. Every recorded command exited zero.

| Check | Result and raw record |
| --- | --- |
| Four Perl regression suites | 60 top-level tests passed; three optional historical full-array fixtures skipped (`perl-tests.log`) |
| C metadata helper, strict C11 build | Passed with `-Wall -Wextra -Werror -pedantic` (`requestmeta-build.log`) |
| C metadata lifecycle tests | `request_meta_test: ok` (`requestmeta-test.log`) |
| Generated opcode header | Passed (`generate-header.log`) |
| Engine C syntax | Passed with `-Wall -Werror` (`engine-syntax.log`) |
| Engine shared-library build | Passed using the CUDA SDK stub library (`engine-stub-link.log`) |
| Whitespace check | Passed (`diff-check.log`) |

The link command explicitly uses `CC=cc -L/opt/cuda/lib64/stubs` because
`engi` does not have the NVIDIA driver library in its normal linker path.
This verifies compilation and linkage against the SDK stub; it is not a
driver load or target-GPU execution test. The link log preserves six
`-Wmaybe-uninitialized` warnings in the unchanged `ref/quant.c` functions.

## Observed blocker

At `2026-10-18T02:04:05Z`, the six dies holding the model reported these free
memory amounts: 52, 16, 77, 7, 3 and 65 MiB. The resident weight holders and
`flashcore` PID 3575268 already occupied those dies. The other two dies do not
hold this six-die model partition.

The only queried resident core reported:

- revision `edc66d9`;
- `kv_host=0`, `ple_disk=1`;
- all 48 layers and a 512-block QSA budget;
- capacity 230000, below the requested 262000-position case;
- no `requestmeta_schema`, `trace_schema`, or `trace_tokens` capability.

There was no second matched host-KV core available for this experiment.
Creating an additional full six-die core would require resources currently
occupied by other processes. Replacing or reloading the shared service would
also change someone else's active runtime. The follow-up did neither.

See `archy-preflight.log` for the actual commands, exit codes, GPU UUIDs,
process inventory and unmodified `config` response.

A second read-only preflight at `2026-10-18T02:58:54Z` found the shared
validation lock busy and exited 73. It performed no GPU query, process
inspection or socket command after that failed acquisition. See
`archy-preflight-latest.log`. The earlier hardware/configuration snapshot
therefore remains historical evidence, not a claim about the current owner
of the lock or current GPU state.

## Acceptance still requires real measurements

CPU regression tests verify the new metadata, lifecycle, parser and rejection
logic. They do not establish model correctness, target-GPU execution,
host/VRAM speed, profiler counters, trace overhead or long-context behavior.

Before issue #14 can close, collect a new campaign with the actual new runtime
and complete build/model/conversion provenance, validated post-template prompts
and answers, five alternating host/VRAM pairs at each supported required
depth, and 129 outputs per request. Keep the schema-2 phase traces and profiler
measurements separate from authoritative throughput, and retain raw files and
explicit capacity skips. The phase-trace runtime is still tracked by PR #30.

Until those records exist and satisfy the gates, the follow-up PR must stay
draft. No result from the old metadata smoke or another executable is relabeled
as evidence for this new runtime.
