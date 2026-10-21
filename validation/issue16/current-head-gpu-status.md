# Current-head GPU verification status

Implementation source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`; Makefile driver-API fix commit: `461c119a954667e6ad2097d9a47afa45e55e0fce`.

The current-head C/CUDA build completed on Benji through DSH. The original binary was built with CUDA 13.4 headers and ptxas 11.4; a GPUQ smoke job failed before launching a kernel because the M60 driver 470.256 does not export `cuEventElapsedTime_v2`. The runner log is GPUQ job `77b3da64-aec0-4573-8de4-f6f83c6d839c`, document 76441.

The QSA Makefile targets now use CUDA 11.4 driver headers and stubs by default. Rebuilding through plain DSH routing succeeded on azb1 with CUDA 11.4.148 `cuda.h` and stub library staged as build inputs. `nm -D` confirms both rebuilt binaries import legacy `cuEventElapsedTime`. Their hashes and the exact DSH request are in `current-head-build.json`.

A GPUQ inventory job found an idle Tesla M60 `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` (driver `470.256.02`, 7,547 MiB free at the inventory sample). A GPUQ negative test ran `QSA_HOSTKV=2` and returned the expected status 2 with `QSA_HOSTKV must be exactly 0 or 1`.

## GPU queue status

The old ABI binary smoke request is still queued, ahead of another running job. I have not changed the staged binaries while that job is pending. The ABI-corrected binaries are built locally on Engi through DSH but have not yet replaced the staged copies.

No current-head paired matrix, `OP_IDXSEL` capture pair, or profiler counters have completed. Earlier matrix attempts were lost before a child start, and GPUQ intermittently returned `database is locked`; the incidents are tracked in [gpuq #35](https://github.com/nm-z/gpuq/issues/35). I will use the current queued smoke job's final status before submitting the corrected binary.

## Remaining acceptance evidence

- Run the ABI-corrected binary on the M60 and confirm the one-pair gather smoke.
- Complete the current-head 40-pair matrix across both placements, all tails, and layouts.
- Capture a verified `OP_IDXSEL` selection and run the pinned-host/VRAM gather pair. The available capture path uses synthetic scores; it is not checkpoint-derived model output.
- Collect separate current-head system-memory/VRAM byte, transaction, and stall profiles on Maxwell.

The PR remains draft. Historical issue #10 profiles and the older matrix retain their own source, binary, and cubin identities.
