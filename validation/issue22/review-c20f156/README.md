# Issue 22 follow-up: capture completion and final CPU evidence

Executed source: **c20f1560f984f7476a249f0a9fd28f437f3f62a5**, clean at the
start of the recorded suite. The evidence commit changes no executable source.
This work integrates the previously published review at
541e9090d0dffcd848572aa7337d9610cc85538c.

## Changes verified

New GPU logit captures publish `capture_info.json` only after all teacher-forced
rows have been written and the logit stream has closed. The v2 schema declares
little-endian FP32 and a true completion flag. Interrupted captures, failed
readbacks, skipped or duplicate positions, missing/overlapping/out-of-range
vocabulary slices and existing output directories cannot produce a final
completion marker.

The comparator requires dtype and completion fields for the new schema while
preserving archived v1 captures. It hashes the consumed logits, IDs and
vocabulary mapping and checks frozen artifacts before and after comparison.
The added readback mock exercises file publication and layout on the CPU; it
does not execute a device kernel.

## Recorded results

| Check | Result |
| --- | --- |
| Comparator regressions | PASS, 76 assertions |
| Logit-export readback mock | PASS, 28 assertions |
| Layer-export readback mock | PASS, 48 assertions |
| Frozen indexer replay | PASS, 36 assertions |
| Analytic indexer checks | PASS, 16 checks |
| Synthetic indexer matrix | PASS, all 14 listed lengths |
| Archived placement pairs | Replayed successfully; each pair remains byte-identical |
| Recovered four-position CPU reference | Diagnostic recomputed successfully |
| Full-model harness | Compiled and linked against the CUDA driver stub; no GPU execution |
| Current GPU gate | BLOCKED, exit 75 after each 45-second lock wait |

The 204 analytic/regression checks are separate from the 14 synthetic length
replays and the archived-capture comparisons. Synthetic coverage is N=1,
2047 through 2055, 3100, 8000, 65000 and 262000. It exercises sparse selection
geometry and includes all sparse tail residues, using generated postprojection
inputs. It supplies no actual production GPU dispatch or checkpoint-quality
measurement.

The recovered four-position comparison reports top-token agreement 4/4,
mean reference-to-candidate KL 0.0210943973410854, logit RMSE
0.186845505716465, and maximum absolute logit error 1.18950200080872.
No acceptance thresholds were applied. The inputs are consecutive token IDs
10,11,12,13; they are not a held-out language sample. The exact historical loaded
backend identity and original-checkpoint/conversion lineage remain unverified.
See [the preserved capture explanation](../archy-llama-7182ef7/README.md).

## Reproduce and inspect

From the repository root:

```bash
bash validation/issue22/run_cpu.sh /absolute/path/new-cpu-evidence
LIBRARY_PATH=/opt/cuda/lib64/stubs make -B -C kern model_test
```

The second command is a build-only check on engi. Linking a CUDA driver stub
does not establish target execution.

`commands.txt`, `exits.tsv` and the logs contain the CPU commands and outcomes.
`source-before.sha256` and `source-after.sha256` match.
`executable.sha256` identifies all four CPU executables;
`model-final.sha256` records the separately compiled harness and generated
opcode header. `runtime.txt` records the host and compiler/runtime versions.
The startup local-proof record retains its original observation time;
`source-commit.txt` identifies the tested code.

The compact [synthetic summary](synthetic-summary.json) contains the measured
geometry, query/score errors and selection overlap for every length.
`synthetic-report.sha256` identifies the full generated reports retained in
`/home/nate/codex/flash-issue22-worker-xmX3xTlj/cpu-final/synthetic-replay`.
The command above regenerates the larger synthetic inputs and output arrays.
The committed historical captures and their original execution identities
remain unchanged.

The shared GPU preflights were initiated through engi. Each tried the required
lock on archy, waited 45 seconds, and observed Ruby process 4010221 holding it.
No new GPU workload was started.

## Acceptance state

[acceptance.json](acceptance.json) keeps `issue22_resolved:false`,
`ready_for_review:false`, and original-checkpoint quality unmeasured.

The following acceptance remains open:

- Actual GPU layer exports and the production-budget sparse length matrix.
- Same-input, same-selection attention output comparisons and isolated
  activation/projection and INT8 KV effects.
- Verified original checkpoint, tokenizer/template and GGUF conversion lineage.
- Predeclared held-out long-context loss/perplexity and retrieval/task accuracy.
- Separately labeled free-running generation and controls for independent
  model differences tracked in #1/#2.

PR #46 remains draft while those requirements are unmet.
