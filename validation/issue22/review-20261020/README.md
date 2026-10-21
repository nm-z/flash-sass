# Issue #22 continuation review

The isolated PR branch includes current master `98c176a80e82cd31736ed3165a134cfbbae45338` through merge commit `1ed9702f89482aa551ee130f261cc0af1a194b0e`, after the earlier merges at `ad2a7532` and `c7f90f7`.

## CPU validation

The current-source CPU suite passed on Benji through `dsh build` ID `build-2630825-1792564997771592028`, source commit `6b52888d5fd354184a2af80671be2309c92f7f3f`. All 25 recorded steps exited 0. It covered comparator and export checks, analytic and frozen indexer/attention tests, placement and recovered-reference replays, and synthetic inputs at N=1, 2047–2055, 3100, 8000, 65000, and 262000. The compact report, final per-length JSON, logs, source hashes, and completion record are in [`review-20261021-cpu-summary`](../review-20261021-cpu-summary/).

Additional current-source checks passed: 93 request-checker cases and 76 model-quality comparator cases. No C compilation ran on engi.

A prior full-output DSH run took about 10 minutes to unpack and return 308 MiB of reproducible raw synthetic tensors and temporary FP64 intermediates. That tool behavior is tracked as [nates-dsh #2998](https://github.com/nm-z/nates-dsh/issues/2998), a sub-issue of the open [#2385](https://github.com/nm-z/nates-dsh/issues/2385). The compact runner returns only reviewable summaries and avoids the large transfer.

## GPU validation

GPUQ preflight job `22e69a40-8536-4fb8-ae44-0536b74dcf84` found devices 0–5 with only 214–300 MiB free, while device 6 had 7547 MiB and device 7 had 4626 MiB. The short one-die synthetic attention smoke (N=1,4,8,9; budget 1; device 6) is queued as job `ce33eb63-7448-4b6b-9397-2a226af036aa` in FIFO order. No model-quality GPU measurement has completed yet.

The fixture executable and cubin were built on Benji via DSH from source commit `6b52888d5fd354184a2af80671be2309c92f7f3f`, using pinned MaxAs `54eda7af086a46c9dae1688b691968235d560164`, repository patches, and CUDA 11.4 ptxas 11.4.152. The first attempt with Benji's CUDA 13.4 ptxas failed because it no longer supports `sm_52`; the final build used the 11.4 ptxas staged on Benji. Only the fixture executable, cubin, generated header, source, and runner files were copied to a unique temporary directory on Archy for the queued job.

This smoke uses supplied synthetic IDs; it does not run the production selector, a full model, or the original checkpoint. The six-die production matrix still needs a free window. No original checkpoint weights have been downloaded or executed.

## Acceptance status

Criterion 7, qualified reporting, is met. Criteria 1–6 remain **unmet**: exact original-to-GGUF conversion lineage, original-checkpoint execution, complete precision/mask controls, independent production-length logits, full production-budget model coverage, and held-out long-context quality remain unestablished. `issue22_resolved` and `ready_for_review` remain false. PR #46 remains draft and must not be merged until these gates are completed.
