# Native oracle evidence for PR #31 / issue #11

Measured code commit: **8ea6b4aebcc1b069037e8dd49807982650dba67e**.
Subsequent evidence commits do not change the tested code. Each oracle report
records that exact start/end HEAD and verifies stable source and binary hashes.

| Gate | Measured result | Evidence |
|---|---|---|
| Portable C/Perl full suite | PASS: 56 invocations; 13 frozen probes, 23 generated index/sparse cases, 18 generated attention cases, host checks and the expected failing score control | [Report](cpu/report.json), [log](cpu/run.log), [artifact hashes](cpu/artifacts.json) |
| Complete historical archive replay plus full generated matrix | PASS: 114 invocations, including all 71 frozen NPZ fixtures | [Report](archive/report.json), [log](archive/run.log), [artifact hashes](archive/artifacts.json) |
| Correct score RMS gate | Maximum measured score NRMS 0.000160764298943595; limit 0.005. The deliberate NRMS 0.0069282032302755096 case exits 3 as required | Both oracle reports |
| C logit comparator | PASS: 32 synthetic regressions, zero failures | [Report](logits-checker/report.json), [retained fixtures](logits-checker/fixtures.tar.gz), [log](logits-checker/run.log) |
| M60 image and CUDA C adapter build | PASS, build only; no GPU launch | [Adapter build](build/identity.json), [compiler log](build/run.log), [image assembly identity](cubin-build.json) |
| Target GPU execution | BLOCKED_BUSY, exit 75; no command launched | [Final idle-check result](archy-idle-final/check.json), [nvidia-smi](archy-idle-final/final-nvidia-smi.txt), [ps subset](archy-idle-final/final-gpu-processes.txt) |
| Independent checkpoint teacher-forced logits | NOT MEASURED | [Availability and remaining gaps](checkpoint-status.json) |

The M60 host remained occupied by resident holders, a fence and flashcore.
An earlier 180-second wait and a final 30-second wait both ended busy.
No other worker's process was stopped. Free individual devices were not used
to bypass the shared-host rule.

The CPU runs use native independent mathematical reductions and preserved
external outputs. They do not claim fresh Transformers execution, target
execution or full-model quality. The logit comparator tests use synthetic
arrays and explicitly keep independent_checkpoint_evidence=false.

Large generated inputs and outputs remain in the run directories recorded in
the reports on engi. The portable command in NUMERICAL_VALIDATION.md recreates
the full native matrix from the checked-in compact goldens. The optional
historical replay requires the preserved original archive. The committed
artifact catalogs are hash manifests, not claims that every large array is
checked into Git.

PR #31 must remain draft until the target execution and independent checkpoint
requirements are fulfilled. It retains **Refs #11**.
