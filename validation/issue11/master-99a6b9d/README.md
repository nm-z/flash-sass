# Host validation after the master merge

Master `adf0fa0f824fec2dcaec7ab44906db544ed7ee7d` was merged into the shared issue #11 branch at `99a6b9de4bacda71545f5dd476bb8a4eae3e574f`. The two conflicting files were resolved by keeping both features: production QSA capture and master's KV trace / PLE reader changes. The engine keeps separate completion flags, applies the capture-mode fresh-prefix requirement before setting the KV trace start, and retains both session commands. The build keeps `qsa_capture.c`, all header dependencies, and master's pthread link option. The engine reuse test also depends on the capture header.

The subsequent shared-branch merge `c363177e0c80787242346a9c88be14c764a28179` incorporates the other worker's `151af5c43401136aef6fa8680ec9ce5aed533fb2`. It adds their regression evidence without changing any `kern/` or `ref/` source relative to the tested master merge; `merge151-source-diff.txt` is the empty, successful comparison. Both independent C reference commands and all previously committed evidence remain present.

## Measured checks

| Check | Result |
|---|---|
| Engine and capture syntax | Passed |
| Ordinary and reuse-test engine libraries | Compiled and linked, including capture and pthread support |
| Both independent model capture commands | Built with the existing strict numerical flags |
| Native numerical suite | `PASS_NATIVE_SYNTHETIC`, 58/58 expected outcomes |
| Candidate capture serialization | 92 assertions passed |
| Production capture descriptor comparison | 90 cases, 2250 descriptors, 1320 expected pointer rewrites passed |
| Independent trace helper | 156 checks passed |
| Device/lock gate | 32 CPU-only checks passed, including real local queue and timeout cases |
| Source identity | No source changed during the numerical suite; before/after hashes match |

The native suite ran on `engi` from 2026-10-18 06:44:05 to 06:44:25 UTC, taking 19.3811020851135 seconds. Its required failing controls returned their expected exit codes. The report explicitly records `gpu_executed: false` and `checkpoint_conformance: false`.

Engine library linking used `-L/opt/cuda/lib64/stubs` for this build-only check. No stub library was placed in a runtime search path and neither library was loaded into a resident engine. The engine syntax check produced no diagnostics. The ordinary engine link retained the existing `ref/quant.c` maybe-uninitialized warnings; the numerical helper builds use `-Werror` and passed.

The first validation invocation omitted the required new-directory argument to `qsa_capture_test` and exited 2 with usage text after the builds had succeeded. `initial-invocation-exit.txt`, `capture-io-usage.log`, and `build-and-initial-invocation.sh` preserve that invocation. The corrected invocation supplied fresh output directories to the I/O, trace, and gate tests and completed with exit 0, as recorded in `corrected-tests.sh` and `tests-exit.txt`. It did not change product source to obtain a pass.

## Evidence and reproduction

`native-report.json` contains every numerical command, expected/actual exit, metric, and source/binary/host identity. `native-artifacts.json` retains the generated artifact size/hash catalog. The generated arrays remain in `/home/nate/codex/flash-pr31-evidence-20261018/master99a6-validation/native` on engi; they are not claimed to be committed in this directory. The portable numerical replay is:

```sh
perl kern/qsa_oracle.pl --full --out /absolute/new-native-output
```

Build the independent capture commands as described in [QSA_MODEL_REFERENCE.md](../../../QSA_MODEL_REFERENCE.md). The already recorded real-model captures, repeated byte agreement, per-layer coverage, and seven rejection controls remain in [full-model](../full-model/README.md); these adapter/comparator sources were not changed by the master integration.

The additional host commands use fresh directories:

```sh
make -C kern dieops.h qsa_capture_test qsa_capture_prog_test qsa_reference_trace_test
kern/qsa_capture_test /absolute/new-capture-io-output
kern/qsa_capture_prog_test
kern/qsa_reference_trace_test /absolute/new-reference-trace-output
bash kern/qsa_oracle_gpu_gate_test.sh /absolute/new-gate-test-output
```

The retained shell scripts and logs record the exact paths and commands used here. `SHA256SUMS` covers the committed evidence, including the scripts and this explanation.

## Target and readiness limits

The actual M60 component pass in [target-ca4a476](../target-ca4a476/README.md) remains valid evidence for its recorded source and production image. Master changes the production attention/quantization kernels and references, so that earlier pass is **not** presented as an execution of this master merge. The updated image needs a new target run.

The fresh full-model probe in that directory acquired the shared lock but exited 1 at `model_test.c:73: cuMemAlloc(&p, bytes) -> out of memory`. It produced no completed logits or layer capture. The production-engine observation mode also requires additional scratch during initialization and an enabled/disabled target comparison. No resident model was stopped, reloaded, or reconfigured here.

GPU availability checks continued using `/tmp/flash-sass-engine-validation.lock`. Checks 14–16 ended without a lock result; their partial output is retained, together with the successful read-only diagnostic following check 14. Check 17 at 2026-10-18 06:44:39 UTC observed Bash PID 1244313 holding the lock and only 3–77 MiB free on resident GPUs 0–5; the bounded lock probe timed out with exit 124. No GPU workload launched from these checks.

Issue #11 remains blocked on a completed current full-model target capture paired with the independent logits and per-layer observations, the required long sparse-context/tail acceptance, and verified original checkpoint/tokenizer/conversion lineage. Historical short-prefix logits exceed the fixed limits and have not localized the mismatch to QSA. This host integration does not establish those missing acceptance results; PR #31 remains draft.
