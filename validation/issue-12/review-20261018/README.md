# Issue #12 review checkpoint — 2026-10-18

**Builds and CPU validation pass. GPU acceptance remains blocked; PR #29 stays draft.**

Tested code commit: `9e3050ba3e9f6c1fe53d16e9fe7f2179e3974aa4`.
This merges current master `3d57d734d63aa74f167a6b5a02c616ddc018d2ff`
into the existing PR branch. The subsequent checkpoint commit changes evidence
and documentation only.

## Changes reviewed

- Replace the former Python matrix with Perl, retaining the shared host lock,
  GPU and process checks, raw child results, and explicit incomplete-run status.
- Add actual leading/trailing output canaries, physical-CTA work counters, fresh
  launch nonces, and rejection tests. Historical output initialization was not a
  boundary guard.
- Merge the trace hooks into the current paired dispatcher after descriptor
  reload. Paired-prefix passes leave the cleared scalar trace unwritten.
  With tracing disabled, the generated SASS is byte-identical to current master.
- Build a validation-only library around the production engine for matched
  six-die decode runs, with fixed snapshots, token parity, two warmup pairs,
  alternating mapping/stage order, and host/device timing definitions.
- Validate every raw whole-model CSV record, including all 16 CTAs, and recompute
  all device timing terms before accepting the JSON. Reject arbitrary nonempty
  CSVs, incomplete traces, invalid work/identity/opcodes, stale launches, and
  internally consistent JSON metrics that disagree with the raw timestamps.

## Build result

Six isolated builds passed on **engi**, with no GPU execution:

| Variant | Source | Trace |
| --- | --- | --- |
| historical-old | `c6d3de8180d8b87caf0d2f61fa52edc9247daded` | off |
| historical-new | `66725432f86cc193b75cd51230216aed44c60d3f` | off |
| matched-old | tested code, reverse mapping patch only | off |
| matched-new | tested code | off |
| trace-old | tested code, reverse mapping patch only | on |
| trace-new | tested code | on |

The matched old/new host libraries are byte-identical, as are the traced
old/new host libraries. The build also compiled the updated regression harness,
`libengine_cta_remap.so`, `flashcore`, and `hold`.

[Build status](build/status.txt), [source/binary hashes](build/),
[toolchain](build/toolchain.txt), and [command excerpts](build/command-excerpts.log)
are retained. The full transcripts contain about 5 MB of assembler output;
their hashes are in [full-log-hashes.txt](build/full-log-hashes.txt), and the
original logs and isolated trees remain at the output path below. The excerpts
retain each compiler/build command and compiler warning. All variants emitted
the existing `ref/quant.c` maybe-uninitialized warnings; the builds exited zero.

Exact build invocation from the branch checkout:

```bash
PR29_EVIDENCE=/home/nate/codex/flash-sass-pr29-20261017/resume-20261018.rqrKTU
LIBRARY_PATH=/opt/cuda/lib64/stubs \
PTXAS11="$PR29_EVIDENCE/toolchain/ptxas" \
CUDA=/opt/cuda MAXAS_ROOT="$PR29_EVIDENCE/toolchain/maxas" \
bash kern/cta_remap_build.sh "$PR29_EVIDENCE/builds" \
    9e3050ba3e9f6c1fe53d16e9fe7f2179e3974aa4
```

The output directory must be new when reproducing. The Maxwell-capable ptxas
is CUDA 11.4.152; its hash and the CUDA linker stub hash are recorded.
MaxAs upstream identity, local patch, and file hashes are retained beside the
build hashes. All six variants used the same frozen MaxAs snapshot.

The untraced production/merged SASS comparison passed with SHA-256
`2e65b0b8e605dc2a4eff30820f3a287e6884171b50d4b580b4b8c2ac088a495f`
for both outputs; see [untraced-comparison.sha256](build/untraced-comparison.sha256).

## CPU validation

| Check | Result |
| --- | --- |
| Freshly built harness `--self-test` | 16 negative canary cases, 14 negative trace cases, 64 mapping cases passed |
| Perl matrix `--self-test` | 15 checks passed |
| Whole-model summary fixtures | 24 checks passed |
| Shared process/CSV evidence helper | 10 checks passed |
| Build script syntax | passed |

Commands, executed from the tested checkout:

```bash
LD_LIBRARY_PATH=/home/nate/codex/flash-sass-pr29-20261017/cuda-stubs \
    "$PR29_EVIDENCE/builds/matched-new/kern/attnpartn_regress" --self-test
perl kern/cta_remap_matrix.pl --self-test
perl kern/cta_remap_summary_test.pl
perl kern/QsaEvidence_test.pl
bash -n kern/cta_remap_build.sh
```

The runtime stub directory contains a `libcuda.so.1` symlink to
`/opt/cuda/lib64/stubs/libcuda.so`; it is used only for the CPU self-test.
The [CPU logs and exit codes](cpu/) contain no target-GPU measurement.

## GPU availability and remaining acceptance

A fresh check on **archy at 2026-10-18 03:41 UTC** exited **75** with
`BLOCKED_BUSY`, `gpu_executed=false`, and `gpu_launches=0).
The shared lock `/tmp/flash-sass-engine-validation.lock` was held by
**Ruby PID 4010221**. The inventory also shows **flashcore PID 3575268**,
six resident holders, a fence, and GPUs 0–5 at 100% utilization.

The runner and shared helper were copied from the tested checkout into an
isolated temporary directory. The command run through SSH from engi was:

```bash
perl /tmp/flash-sass-issue12-gate.bZqgCy/cta_remap_matrix.pl \
    --check-only \
    --gpu GPU-7ed9f65b-0233-0113-dfc1-05f4039decb4 \
    --out /tmp/flash-sass-issue12-gate.bZqgCy/result \
    --wait-seconds 0
```

[Gate status](gate/status.json), [manifest](gate/manifest.json),
[exit code](gate/exit.txt), and [process/GPU inventory](gate/inventory.txt)
record the blocker. The check did not stop, reload, or attach to another worker's
engine.

The new canary matrix, physical-CTA trace/rejection runs, and matched six-die
whole-model experiment still require an idle, exclusively locked host. The
historical 48 component cases remain preserved in the parent directory. They
do not validate these new trace/canary changes, and their component timing does
not establish a whole-model speedup. No new GPU result or estimated whole-model
gain is claimed by this checkpoint.
