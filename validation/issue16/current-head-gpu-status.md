# Current-head GPU verification status

Implementation source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`; Makefile driver-API fix commit: `461c119a954667e6ad2097d9a47afa45e55e0fce`. PR branch now merges master `98c176a80e82cd31736ed3165a134cfbbae45338` at `64cf729d87e6d51718f32b05b0b6d52842dacdd2`.

The current-head C/CUDA build completed through DSH on Benji. The original binary was built with CUDA 13.4 headers and ptxas 11.4; an M60 smoke attempt failed before a kernel launch because driver 470.256 does not export `cuEventElapsedTime_v2`. The error is in GPUQ job `77b3da64-aec0-4573-8de4-f6f83c6d839c`, document 76441.

The QSA Makefile targets now use CUDA 11.4 driver headers and link stubs. Rebuilding through plain DSH routing succeeded on Benji at the merged head with staged CUDA 11.4.148 `cuda.h`, the matching stub, ptxas 11.4.152, and patched MaxAs. `nm -D` confirms both rebuilt binaries import legacy `cuEventElapsedTime`; hashes, input identities, and the raw DSH log are in [current-head-build-64cf.json](current-head-build-64cf.json) and [build-current-head-64cf.log](build-current-head-64cf.log). The earlier compatibility rebuild record is retained in `current-head-build.json`.

The old queued smoke request has completed with `/dev/null: File exists` before CUDA initialization. The corrected pre-merge smoke `94cf312f-88b8-4577-b579-a52ad34a878f` later succeeded with exit 0 using the ABI-correct binary from `461c119`; it does not by itself verify merged source revision `64cf729`.

A one-launch merged-head smoke ran through GPUQ as `33205075-1c6b-46e1-a995-100540f50727` and succeeded with exit 0. Its raw JSON output is in [current-head-smoke-64cf.json](current-head-smoke-64cf.json); it reports source revision `64cf729`, strict VRAM placement, `preflight_verified_all: true`, and 41.024 microseconds for the one measured launch. The CSV output is not retained; the GPUQ stdout JSON contains the complete one-launch summary. This smoke alone does not satisfy the matrix, capture, or counter-profile requirements.

The merged-head 40-pair matrix has now completed as GPUQ job `cfacfaa9-faa5-4b06-adf1-64f3eeea452e` (exit 0). It covers `nv=128,2000`, tails 0–3, five patterns, eight iterations per arm, and both placements (80 arms total). The runner verified every output before timing and checked paired cache, selection, and output signatures; all 40 pairs pass. The paired report, provenance, and GPU identity are in [matrix-64cf](matrix-64cf/); raw CSVs, logs, and IDs are in [raw-results.tar.gz](matrix-64cf/raw-results.tar.gz), SHA-256 `8b5d1c8d81a095ecd8bcc7b9435b7a279c292bfcb1ad79461ed7f35a24c4d6b4`. The source bundle is GPUQ document 79399 (`a2d4ef67af22a8b8ba095bb399abafc27dbd60b80a62566c0a78d44e12aa2c24`). Its `captured` pattern uses generated IDs, not a model capture. The matrix does not include the separate `changing-token` condition.

The queue has since gained a queued FS13 job. I will wait for GPUQ to clear before running the supplemental per-token pair, the verified `idxfull` capture pair, or profiler work.

The GPUQ inventory found an idle Tesla M60 (`GPU-431371bd-b482-96b2-4b70-04f83c6d075b`, driver `470.256.02`, 7,547 MiB free at the inventory sample). A short GPUQ negative test confirmed `QSA_HOSTKV=2` is rejected with status 2 and the expected message.

Earlier matrix submissions were lost before a GPU child started; intermittent database locks are recorded in [gpuq #35](https://github.com/nm-z/gpuq/issues/35). The corrected smoke remains in the FIFO; no duplicate is submitted while it is queued.

## Remaining acceptance evidence

- ABI-corrected current-head M60 smoke.
- Current-head 40-pair matrix across both placements, all tails, and layouts.
- A verified `OP_IDXSEL` capture from `qsa_test idxfull` and the paired pinned-host/VRAM gather run. This capture path uses synthetic scores, not checkpoint-derived model output.
- Separate current-head system-memory/VRAM byte, transaction, and stall profiles on Maxwell.

The PR remains draft. Historical issue #10 profiles and the older matrix retain their own source, binary, and cubin identities.
