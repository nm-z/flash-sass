# Current-head GPU verification status

Implementation source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`; Makefile driver-API fix commit: `461c119a954667e6ad2097d9a47afa45e55e0fce`.

The current-head C/CUDA build completed through DSH on Benji. The original binary was built with CUDA 13.4 headers and ptxas 11.4; an M60 smoke attempt failed before a kernel launch because driver 470.256 does not export `cuEventElapsedTime_v2`. The error is in GPUQ job `77b3da64-aec0-4573-8de4-f6f83c6d839c`, document 76441.

The QSA Makefile targets now use CUDA 11.4 driver headers and link stubs. Rebuilding through plain DSH routing succeeded on azb1 with staged CUDA 11.4.148 `cuda.h` and the matching stub. `nm -D` confirms both rebuilt binaries import legacy `cuEventElapsedTime`; hashes are in `current-head-build.json`.

The old queued smoke request has completed with `/dev/null: File exists` before CUDA initialization. The corrected one-launch smoke is now queued as `94cf312f-88b8-4577-b579-a52ad34a878f`, behind earlier GPUQ jobs. The updated binaries are staged under `/dev/shm/fs16-a623032/kern`. No matrix, capture, or counter profile has run yet.

The GPUQ inventory found an idle Tesla M60 (`GPU-431371bd-b482-96b2-4b70-04f83c6d075b`, driver `470.256.02`, 7,547 MiB free at the inventory sample). A short GPUQ negative test confirmed `QSA_HOSTKV=2` is rejected with status 2 and the expected message.

Earlier matrix submissions were lost before a GPU child started; intermittent database locks are recorded in [gpuq #35](https://github.com/nm-z/gpuq/issues/35). The corrected smoke remains in the FIFO; no duplicate is submitted while it is queued.

## Remaining acceptance evidence

- ABI-corrected current-head M60 smoke.
- Current-head 40-pair matrix across both placements, all tails, and layouts.
- A verified `OP_IDXSEL` capture from `qsa_test idxfull` and the paired pinned-host/VRAM gather run. This capture path uses synthetic scores, not checkpoint-derived model output.
- Separate current-head system-memory/VRAM byte, transaction, and stall profiles on Maxwell.

The PR remains draft. Historical issue #10 profiles and the older matrix retain their own source, binary, and cubin identities.
