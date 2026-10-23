# Follow-up verification of PR #44

This is an evidence-only follow-up. The requested implementation and completed
GPU/checkpoint records were already committed at
`3c81fa166079041787b5eb3c87c8da191b4e97d8` when this verification began. No
runtime, quantizer, assembler, or runner source is changed here.

## Work performed on engi

- Opened issue #19, PR #44, and its Files changed HTML page successfully
  (HTTP 200 for all three); inspected the changed quantizer, boundary fixtures,
  per-layer diagnostics and Ruby runner sources.
- Executed the committed Ruby runner's CPU mode at `3c81fa1`. Both commands
  exited 0. All 44 signed fixtures passed, including both signs of the adjacent
  inputs `0x367e0001` / `0x367e0002` and expected half scales `0x0000` / `0x0001`.
  The additional finite-range, reconstruction and unchanged-rejection-output
  checks passed. The runner worked with Ruby 3.4.10 without an optional CSV gem.
- Checked the SHA-256 of all three delivered raw archives against the published
  values, extracted fresh copies, and ran the committed offline auditor. It
  exited 0 with `status=PASS`.
- The audit checked the archived command/output hashes, token slices, complete
  layer coverage, count/maxima consistency, recorded restoration and merged
  128-position parity. The three cubin hashes match between the original and
  merged records; all 24 layer rows and 234 state-hash payload records match
  for the 128-position comparison.
- Inspected all 107 files returned by GitHub's paginated PR-files endpoint.
  No Python files are added or modified. None of the three raw archives contains
  a Python source or notebook member. There were no unresolved inline review
  threads; the earlier top-level Python-runner comment is addressed by removal
  of that runner and the working Ruby replacement.

## Fresh CPU commands

Executed from `/home/nate/codex/flash-sass-issue19-20261018` at the source above:

```bash
ruby validation/issue19/run.rb --mode cpu \
  --out /home/nate/codex/flash-pr44-recheck.DRgLkhbB/runner-cpu
ruby -c validation/issue19/run.rb
ruby -c validation/issue19/20261018/audit-completed.rb
```

[cpu.json](cpu.json) contains the exact compiler invocation, source/runner/binary
identities, command exits, elapsed times and raw-output hashes. Raw stdout and
stderr files are included beside it. [toolchain.txt](toolchain.txt) records the
host, compiler and Ruby version.

## Fresh archive-audit command

After extracting the original CPU/build/GPU and model archives into `original`
and the integration archive into `merged`, executed:

```bash
ruby validation/issue19/20261018/audit-completed.rb \
  /home/nate/codex/flash-pr44-recheck.DRgLkhbB/original \
  /home/nate/codex/flash-pr44-recheck.DRgLkhbB/merged
```

[offline-audit.json](offline-audit.json) is the newly produced report.
[archive-sha256.txt](archive-sha256.txt) records the freshly checked archive
hashes. Original raw evidence remains in [the completed report](../20261018/README.md).

## Scope of the archived observations

| Actual measured revision | Positions | K/V rows | Groups | Underflow | Clipped | Would overflow |
|---|---:|---:|---:|---:|---:|---:|
| `6da421b` | 128 | 24 | 49,152 | 0 | 0 | 0 |
| `6da421b` | 32,768 | 24 | 12,582,912 | 0 | 0 | 0 |
| `6da421b` | 65,536 | 24 | 25,165,824 | 0 | 0 | 0 |
| `d46d188` | 128 | 24 | 49,152 | 0 | 0 | 0 |

The long-context GPU measurements were **not repeated in this follow-up** or
at the merged revision. Their applicability after the merge remains the
explicit, revision-scoped inference in the completed report; a fresh offline
audit is not a fresh GPU execution. No GPU command, resident-engine mutation,
process termination or GPU-lock acquisition was performed in this follow-up.
The new evidence does not expand the measured checkpoint/workload scope or
establish whole-model quality or throughput effects.
