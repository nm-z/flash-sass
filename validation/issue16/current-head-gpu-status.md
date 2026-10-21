# Current-head GPU verification status

Implementation source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`. PR #43 includes the current master merge and validation updates.

The current-head C/CUDA build completed on Benji through DSH. Its request, task-local build root, toolchain inputs, artifact hashes, and raw log are recorded in `build-current-head.json` and `build-current-head.log`.

A GPUQ inventory job completed successfully. It found an idle Tesla M60 `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` (driver `470.256.02`, 7,547 MiB free at the inventory sample). A GPUQ negative test ran `QSA_HOSTKV=2` and returned the expected status 2 with `QSA_HOSTKV must be exactly 0 or 1`.

## Matrix attempts

No paired matrix has completed on the current head. Four attempts have not started a GPU child:

- Initial matrix request `a34b0f45-b983-447e-8be4-67056e8d7161`: `lost` before ticket/start; logs empty.
- Staged-script retry `5f97798e-3bd5-49df-a80d-efbb9c3d4b6e`: `lost` before start.
- A `bash -c` submission returned exit 2; its history lookup initially hit `database is locked` and later found no row.
- DSH-built C launcher `e93afb10-dcb4-4c8c-a23c-dc2cf43f05ee`: `lost` with `started_ns=null`, `ticket=null`, `exit_code=null`.

The failures are recorded in [gpuq #35](https://github.com/nm-z/gpuq/issues/35). I am not submitting more jobs until GPUQ queue ownership and database state are reliable.

The binaries, source identity files, Perl runner, CUDA 11.4.120 `nvprof`/CUPTI libraries, and temporary C launcher are staged under `/dev/shm/fs16-a623032`. NVProf 11.4.120 reports its version on Archy, but no current-head metric query or counter profile has run.

## Remaining acceptance evidence

- Current-head 40-pair matrix across both placements, all tails and layouts.
- A verified `OP_IDXSEL` capture from `qsa_test idxfull` and the paired pinned-host/VRAM gather run. The current capture path uses synthetic scores; it is not checkpoint-derived model output.
- Separate current-head system-memory/VRAM byte, transaction, and stall profiles on Maxwell.

The PR remains draft. Historical issue #10 profiles and the older matrix retain their own source, binary, and cubin identities.
