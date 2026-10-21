# Current-head GPU verification status

Implementation source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`. PR #43 currently includes the merged master and validation commits.

The current-head C/CUDA build completed on Benji through DSH. Its DSH request, task-local build root, ptxas/MaxAs inputs, artifact hashes, and raw build log are recorded in `build-current-head.json` and `build-current-head.log`.

A GPUQ inventory job completed successfully. It found an idle Tesla M60 `GPU-431371bd-b482-96b2-4b70-04f83c6d075b` (driver `470.256.02`, 7,547 MiB free at the inventory sample). A short GPUQ test ran `QSA_HOSTKV=2` and returned the expected status 2 with `QSA_HOSTKV must be exactly 0 or 1`.

## Matrix attempts

No paired matrix has completed on the current head.

- The first matrix request (`a34b0f45-b983-447e-8be4-67056e8d7161`) ended `lost` with `started_ns=null`, `ticket=null`, and `exit_code=null`; its logs are empty.
- A staged-script retry (`5f97798e-3bd5-49df-a80d-efbb9c3d4b6e`) also ended `lost` before start, with no ticket or child logs.
- A later `bash -c` submission returned exit 2. The following read-only history query failed with `database is locked`, so that attempt's final state is unconfirmed.

All queue failures are recorded in [gpuq #35](https://github.com/nm-z/gpuq/issues/35). No more GPU jobs are being submitted while the queue history is locked or the latest job's state is uncertain.

The binaries, source identity files, Perl runner, and CUDA 11.4.120 `nvprof`/CUPTI libraries are staged under `/dev/shm/fs16-a623032`. NVProf 11.4.120 reports its version on Archy, but no current-head metric query or counter profile has run.

## Remaining acceptance evidence

- Current-head 40-pair matrix across both placements, all tails and layouts.
- A verified `OP_IDXSEL` capture from `qsa_test idxfull` and the paired pinned-host/VRAM gather run. The available capture path uses synthetic scores; it is not checkpoint-derived model output.
- Separate current-head system-memory/VRAM byte, transaction, and stall profiles on Maxwell.

The PR remains draft. Historical issue #10 profiles and the older matrix are labelled with their own source, binary, and cubin identities.
