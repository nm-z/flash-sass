# Updated M60 execution after the master merge

The production numerical suite at `f6c54db8b26251f3e1772e9672c8e70f33294b81` passed on an actual Tesla M60 after the master integration. This closes the component-target revalidation gap recorded in [master-99a6b9d](../master-99a6b9d/README.md). It does not establish full-model or original-checkpoint conformance.

## Measured outcome

| Item | Recorded value |
|---|---|
| Source | `f6c54db8b26251f3e1772e9672c8e70f33294b81` |
| Master incorporated | `adf0fa0f824fec2dcaec7ab44906db544ed7ee7d` through `99a6b9d` |
| Controller / execution host | engi / archy |
| Physical GPU | Tesla M60 #6, `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` |
| Child CUDA ordinal | 0, scoped to that UUID by the shared-lock gate |
| Shared lock | `/tmp/flash-sass-engine-validation.lock` |
| Start / end | 2026-10-18 06:54:36 / 06:55:44 UTC |
| Duration | 68.0273389816284 seconds |
| Result | `PASS_TARGET_SYNTHETIC`, exit 0 |
| Expected outcomes | 58/58 passed |
| Target-bearing invocations | 54 |
| Actual CUDA kernel launches | 1898 |
| Exact target/scalar byte comparisons | 707 |
| Target parity failures | 0 |
| Maximum scaled-score normalized RMS | 0.000157283398165829 |
| Unchanged score normalized RMS limit | 0.005 |
| Run artifacts copied to engi and verified | 1347, by both size and SHA-256 |
| Source/binary/host identity during the suite | Consistent; no changed sources |

The updated production image SHA-256 is
`81d0a1df8031ca79fd0eafd55a492b12d1120015eb5fe8ff63584763f538ef6e`.
The actual executed adapter SHA-256 is
`10874128757cc37f62ceb8b834b938e9219e1892ce072335f0c380e5ad62a3fd`.
Both match before and after execution. The generated opcode header remains
`5032a1e6012f19f30b922ede9fdb4662a19f17080163506b4edc5290f9f51356`.

The four CPU-only invocations in the suite are the host/domain checks, required failing score control, required failing one-ULP/signed-zero parity control, and frozen BF16 boundary check. The negative controls return their required exit 3 and are not target failures. Their local parity rejection counts are excluded from the 707 target checks and zero target failures above.

## Coverage and build identity

Thirteen pinned external fixtures were replayed. Generated index/sparse lengths were `1`, `2047..2052`, `3100..3103`, `8000..8003`, `65000..65003`, and `262000..262003`, including all modulo-four tails at the long lengths. Same-ID attention lengths were `1..4`, `1023..1027`, `2047..2052`, and `8191..8193`. The production quantizer boundary/batching cases also passed. These are component fixtures; the long lengths are not a newly executed full-model token corpus.

The image was assembled on engi using the recorded CUDA 11.4 assembler and a private copy of the repository's patched MaxAs inputs. `build/controller-*` records the controller build, original MaxAs commit and per-file hashes, assembler identity, and generated image/header/assembly hashes. The exact MaxAs patch and build log are retained as gzip files. The controller's stub-linked adapter was a build check and was not executed on the GPU.

A separate, isolated sparse checkout on archy was pinned to the exact commit. The image/header/assembly transferred there matched the controller hashes. The adapter was rebuilt on archy against `/usr/lib/libcuda`; `build/adapter-ldd.txt` records the actual loader resolution, and `build/adapter-build.log` records the strict floating-point flags. CUDA stubs were not added to a runtime search path. The tested tracked checkout was clean before and after the suite.

The gate acquired the shared lock, re-resolved the device UUID under that lock, and found no compute user on the selected device. It exposed only the selected M60 to the child. No resident process was stopped, reloaded, or reconfigured. Check 20, after this run, found another `gpuq` process holding the lock and active work on model GPUs; that observation is retained separately from the completed suite.

## Retained evidence and reproduction

`gpu/report.json` contains each command, expected and actual exit, metric, source hash and execution identity. `gpu/checks/` retains every child JSON report and stderr. `gpu/artifacts.json` records all 1347 generated run files. `gpu/pinned-and-quantizer-captures.tar.gz` retains the complete frozen replay and quantizer raw inputs/observed outputs.

The full approximately 1.2-GiB run is retained on engi at:

`/home/nate/codex/flash-pr31-evidence-20261018/master-target-run/gpu`

`artifacts-verified.txt` records the successful size/SHA-256 verification after copying every cataloged run artifact. The large generated long-case arrays are not claimed to be committed to Git. `gate-file-identities.txt` records the complete gate inventory, including the full host process listing retained with the local run. That full process-argument listing is omitted from the public evidence; the device-specific compute inventories and their empty selected-device result are committed. `SHA256SUMS` covers the files actually committed here.

From the tested source, with the recorded CUDA/MaxAs setup:

```sh
make -C kern die.cubin dieops.h qsa_oracle_gpu CUDA_LIBDIR=/usr/lib
bash kern/qsa_oracle_gpu_gate.sh --device 6 --wait 600 --run-timeout 300 \
  --out /absolute/new-gate-output -- \
  perl kern/qsa_oracle.pl --full --gpu kern/die.cubin \
  --binary kern/qsa_oracle_gpu --device 0 --out /absolute/new-target-output
```

The physical ordinal is specific to this host; the gate must resolve and verify a suitable idle device under the same shared lock. Do not bypass the lock when a device appears free.

## Full-model readiness remains blocked

The independent CPU logits and per-layer QSA observations are retained in [full-model](../full-model/README.md) and [independent](../independent/README.md). The selected-row capture retains 408 tensor snapshots over all 12 QSA layers and all 48 block outputs; a second actual model run reproduces 440 comparable files byte for byte. The four-row and selected-row references agree on their overlapping logits. Their comparisons with the archived GPU run exceed the fixed 0.10 maximum-absolute / 0.01 normalized-RMS limits, and the short-prefix discrepancy is not localized to QSA.

The earlier fresh six-die full-model probe in [target-ca4a476/model-probe](../target-ca4a476/model-probe) acquired the shared lock but failed before producing a completed capture:

```text
model_test.c:73: cuMemAlloc(&p, bytes) -> out of memory
```

This newer component run does not supply the missing current full-model logits or candidate per-layer trace. The opt-in resident-engine capture path needs additional scratch at initialization and an enabled/disabled target comparison. A completed target run at the required sparse model lengths/tails and verified original checkpoint/tokenizer/conversion lineage are still absent. Those acceptance gates remain required before PR #31 can be marked Ready for review. No Python was written or executed.
