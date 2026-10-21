# GPUQ evidence helpers

These C helpers are specific to the issue-16 validation run. They were compiled through DSH, copied to Archy tmpfs, and invoked only with `ssh archy gpuq run -- ...`. The helpers do not implement product behavior.

- `issue16_matrix_launcher.c` runs the existing `run_qsa_gather_pair.pl` matrix, puts a compressed raw-results bundle into GPUQ, and removes its temporary tmpfs directory.
- `issue16_capture_launcher.c` runs the verified `qsa_test idxfull` capture, checks the sorted ID row, pairs captured and per-token cases across host/VRAM, and uploads the raw results.
- `issue16_profile_launcher.c` runs one separate nvprof metric group for each paired host/VRAM job, checks the outputs, and uploads the logs and counter CSVs. On Archy, unprivileged nvprof returned `ERR_NVGPUCTRPERM`; the successful helper invokes nvprof through `sudo -n` inside the GPUQ-locked process. It does not change driver settings.

The helpers use the `/dev/shm/fs16-64cf` paths and the M60 UUID recorded in the manifests. Their output manifests, exact binary/cubin hashes, GPUQ job identities, and raw results are in the parent directory.
