# PR #78 resume: integration and validation

## Scope and source identity

This follow-up resumes issue #14 on PR #78. The issue, merged PR #32 and its
Files changed resource, and PR #78 and its Files changed resource were opened.
PR #32's metadata smoke did not supply the completed matched campaign.

The implementation tested here is **83644519f87c9f8c9f005b596feb1cba3d2534a3**.
The source tree was clean before validation, and default branch `master` was
still `3d57d734d63aa74f167a6b5a02c616ddc018d2ff`. This subsequent evidence commit
only preserves the records and explanation in this directory.

## Changes on the PR branch

- Integrate the schema-2 SASS trace producer, C exporter, and Ruby validation
  tools from PR #30 at `257ea63c6cc431e3a3c5b32f52ce0cfdfb8bfdff`.
  Historical PR #30 measurements retain their original source identity; they
  are not results for this integrated runtime.
- Keep request metadata and trace exports on one request ID, including
  rejected protocol commands. Invalidate stale captures at every attempt.
- Export trace layer labels through the real descriptor index, including
  programs containing descriptors that span two slots.
- Report host program allocation failures as `allocation_error` / return -4
  in both metadata protocols, preserving the observed build interval and
  leaving execution boundaries unobserved.
- Require profiler schema 2 with counter references to verified raw JSON
  records. Extract each finite value from the referenced bytes, preserve
  integer precision, and reject unbound or duplicate declarations.
- Apply the same executable/library, tensor, launch and PLE cache identity
  checks to separate profiler records as to attribution records.

Authored and integrated implementation languages are C, Perl, Ruby, Make and
SASS assembly. No Python was added.

## Locked CPU validation

The validator acquired `/tmp/flash-sass-engine-validation.lock` on `engi`,
checked `nvidia-smi` and `ps`, held the lock through all checks, and released
it. `nvidia-smi` returned 127 because it is unavailable on this CPU host.
All 12 validation commands exited zero. `cpu-validation.json` preserves
exact argv, individual exits, source-file hashes, build-artifact identities,
log hashes and the lock lifecycle. The exact validator is `verify-cpu.pl`.

| Check | Result |
| --- | --- |
| Four Perl suites | 60 top-level tests passed; three optional historical full-array fixtures skipped |
| C request metadata helper | Strict C11 build and lifecycle test passed |
| C engine syntax | Passed with `-Wall -Werror` |
| C engine integration | Rejected-command identities, stale-trace invalidation, all three program allocation failures and protocol recovery passed |
| Actual C exporter to Perl parser | Synthetic CPU fixture: 129 outputs, 128 intervals and 258 phase rows; descriptor mapping and integer timestamps above 2^53 preserved |
| Ruby trace reconciliation | Exact partitions, independent epoch shifts, causal bounds and malformed coverage tests passed |
| SASS generation | Ordinary, CTA-0 trace and 16-CTA trace source generation passed |
| Engine shared-library linkage | Passed against the CUDA SDK stub; six existing `ref/quant.c` warnings retained |
| Whitespace | Passed |

`engine-cpu-trace.txt` is the actual C exporter output for **artificial CPU
input timestamps**. It is neither a GPU trace nor a model benchmark. The C
test replaces exercised CUDA calls with CPU mocks. No driver execution or
cubin assembly was validated by these CPU checks.

## Exact current blocker

Both target-host retries, at **2026-10-18T05:40:16Z** and
**2026-10-18T05:45:22Z**, failed to acquire
`/tmp/flash-sass-engine-validation.lock` on **archy** and exited **73**.
See `archy-preflight.log`, `archy-preflight-latest.log` and the exact
`archy-preflight.sh` used through SSH from engi. Each attempt stopped at
failed acquisition, before GPU queries, process inspection or socket commands.
No inference, benchmark, runtime reload or process termination was issued.
The earlier GPU occupancy snapshot in `../20261018/` remains historical.

## Issue #14 acceptance status

The implementation has CPU coverage, but all eight criteria remain **unmet
as a complete evidence-backed campaign** until the integrated runtime can be
validated and its real measurements collected.

| Criterion | Implementation and remaining evidence |
| --- | --- |
| One-command complete manifest | Runner and provenance checks implemented; actual full runtime/build/model/conversion/GPU campaign manifest still missing |
| Prompt/output/runtime state | Capture and rejection gates implemented; real validated post-template prompts, outputs and state records missing |
| Executed dense/QSA and selected keys | Producer, exporter and parser integrated; real source-matched executed traces missing |
| Required depths and matched placements | Matrix and capacity-skip policy implemented; real host/VRAM pairs and capacity records missing |
| 129 outputs, repetitions, raw median/p95 | Fixed-policy and aggregate gates implemented; genuine raw successful campaign missing |
| Separately named timing boundaries | Host records and device trace parsing implemented; actual measurements missing, and device-prefill time remains explicitly unavailable in these host records |
| Separate profiler collection | Raw-bound counter acceptance implemented; real separate profiler reports and instrumentation records missing |
| Reject differences and reconcile increments | Identity checks and reconciliation implemented; actual throughput/attribution overhead and context-increment comparison missing |

PR #78 must remain draft until the real campaign satisfies these acceptance
criteria. The busy shared lock is the current execution blocker; CPU fixtures
and historical measurements do not substitute for the missing evidence.
