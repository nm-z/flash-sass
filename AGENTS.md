# Performance target

- Decode 140 tok/s and prompt processing 1,200 tok/s. This is the calculated hardware ceiling. Do not re-derive it.

# Job measurements and reuse

- GPU jobs on archy use `ssh archy gpuq run -- COMMAND ...`. The queue records each job in its SQLite history.
- Before writing another benchmark or scratch test, query the database for an existing job, source snapshot, metric, or fixture. Use `gpuq search`, `gpuq jobs`, `gpuq query`, and `gpuq rerun`.
- Treat the database as the primary interface for run results. `gpuq run` is quiet; retrieve stdout/stderr with `gpuq logs JOB_ID` and individual records with `gpuq document DOCUMENT_ID`.
- Do not redirect job output to ad hoc `.txt` or `.log` files. Emit structured metrics with `gpuq emit`, send text directly with `gpuq put`, or write legacy output under `$GPUQ_RUN_DIR`. Newly generated text in managed/captured output directories is ingested and removed after durable storage. Source files and repository contents remain version controlled.
- Optional collectors are enabled by default. Use `gpuq run --exclude gpu,trace -- COMMAND ...` when deliberately omitting them. Every omission is recorded. Identity, input/output capture, source snapshots and text-artifact retention remain mandatory.
- Distinguish excluded, unavailable and unknown historical measurements from zero. Keep engine/client/device clock domains, units, workload, sampler, trace mode, warm/cold conditions and implementation identities distinct when comparing jobs. The database records observations; it does not establish that incomparable runs demonstrate a speedup.
- An existing shared engine's snapshot may include work outside the queued job. Such records are labelled unattributed context. Use the job's own emitted timing/trace records for attribution.
- See [the queue/database interface](tools/gpuq/README.md). All authored code must follow the user's allowed-language requirements.
