# Measuring and selecting the six-die split

Issue [#54](https://github.com/nm-z/flash-sass/issues/54) requires a measured placement: an `actq256` spread of at most 0.5 ms over dies 0–5, at least 35.9 decode tokens/s under the `linkmatrix_test.sh` request protocol, identical output token IDs, and the chosen configuration and profile in `SPEC.md`.

The capture and analysis tools prepare that experiment. They do not apply a placement or claim that an unmeasured proposal meets those requirements.

The [2026-10-18 resident baseline](../validation/issue54/baseline-20261018T053200Z/README.md) was captured successfully under the shared lock. Its 2.2927 ms worst spread and 35.7767 tok/s mean decode miss the acceptance limits; a candidate placement remains to be measured.

## Capture one loaded configuration

Run the collector through `gpuq` on the GPU host, with the existing core and
front-end PIDs. Query `gpuq search` for reusable captures before a new run:

```bash
gpuq run --name die-balance-baseline -- sh -c '
exec kern/die_balance_capture \
  --out "$GPUQ_RUN_DIR/baseline" \
  --label baseline \
  --core-pid CORE_PID \
  --flashd-pid FLASHD_PID \
  --socket /tmp/flash.sock'
```

Use a fresh output directory for every capture. The default shared lock is
`/tmp/flash-sass-engine-validation.lock`. The queue waits for its current owner,
then passes its lease to the collector through `GPUQ_LOCK_FD`. The collector
validates the descriptor against the requested lock and duplicates it, so it
does not contend with its own queue reservation or unlock the parent on exit.
Direct callers also wait, but GPU measurements must use the queue for FIFO
ordering and retained evidence. Never remove the lock file or stop its owner.
The collector checks GPU/process state before inference and retains the
preflight output. Retrieve records with `gpuq logs JOB_ID`, `gpuq metrics JOB_ID`,
and `gpuq document DOCUMENT_ID`. Text under `$GPUQ_RUN_DIR` is ingested into the
database; it is not a persistent filesystem capture directory.

The request body is the one in `kern/linkmatrix_test.sh`:

```json
{"messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":40,"temperature":0}
```

One warm-up request precedes at least three measured requests. PCIe links remain at their current settings. Do not run the original link-matrix script to capture a placement: that script changes and retrains root-port links.

The collector starts its own temporary HTTP front end using the running `flashd` executable and its behavior settings. A private socket relay connects it to the existing core. The relay records the actual prompt and output IDs, including tokens omitted from the rendered answer. After the core's `done` record, it reads `config` and `prof <die> all` for all six dies over the same connection, before releasing the response. This associates the six profiles with that response even if another client is waiting. Cleanup applies only to the collector's own front end and sockets.

`capture.json` retains the request, runtime configuration, core/frontend/engine-library SHA-256 identities, model file identity, hardware/link snapshot, HTTP timings, token IDs, and raw profiles for each run. A failed or partial capture must not be passed off as a completed measurement. The checkpoint identity is path/stat metadata, not a full content hash. The runtime revision in `config` and the collector checkout revision in `manifest.source_revision` are separate; neither is relabeled as the other.

## Compare measured candidates

After loading a candidate placement in an approved maintenance window, capture it with a new label and output directory. Hold constant the engine implementation, checkpoint, kernel image, logical-to-physical die mapping, context capacity, KV placement, layer count, block budget, front-end settings, PCIe links, and request protocol. Capture a fresh baseline again if those conditions change.

```bash
kern/die_balance --baseline /path/to/baseline/capture.json \
  /path/to/candidate-a/capture.json /path/to/candidate-b/capture.json
```

The analyzer rejects incomplete profiles, different input/output IDs, mismatched runtime conditions, invalid timing values, and candidates that miss the acceptance thresholds. Its objective is the largest per-die profile span in any measured request, `max_run(max_die(total_ms))`; durations on parallel dies are never added. The selected configuration is the best of the measured candidates supplied to the tool. A baseline without a measured candidate does not establish a new winning split.

The old `prof` command measures the last token of the response. Its differences omit phase zero, and `actq256` includes both the phase's work and waiting for an earlier exchange. It is the metric named in #54, not a complete separation of compute and communication. Retain all opcode rows and repeat the experiment to identify clock, temperature, or PCIe drift; use the phase-trace work when finer attribution is needed.

## Prepare placement candidates

`SPLIT_SHARES` controls the holder's row and expert boundaries as well as the engine's program. `ATTN_MAP` controls which holder owns each attention layer's projections and where that layer's runtime state lives. Changing either requires compatible holders and an engine initialized with the same placement. Reloading only the engine library does not move the resident weights.

The routed experts use the same cumulative boundaries as rows:

```text
start(d) = floor(512 * sum(shares[0...d]) / sum(shares))
end(d)   = floor(512 * sum(shares[0...d+1]) / sum(shares))
```

The shared expert of block `b` runs on die `b % 6`. With the existing shares `112,97,81,116,97,97`, routed expert ranges are:

| Die | Share | Routed experts, half-open range | Attention blocks |
|---|---:|---|---|
| 0 | 112 | [0, 95) | — |
| 1 | 97 | [95, 178) | 3, 7, 11, 15 |
| 2 | 81 | [178, 247) | — |
| 3 | 116 | [247, 346) | — |
| 4 | 97 | [346, 429) | 19, 23, 27, 31 |
| 5 | 97 | [429, 512) | 35, 39, 43, 47 |

Wait times alone cannot predict the next split: each move changes discrete row/head/expert boundaries, attention placement changes memory needs, and the selected experts depend on the prompt. Use the per-op profiles to choose local placement candidates, check their memory budgets, then rank their measured results. Do not replace `serve.sh` defaults using only a fitted or hand-computed wait estimate.

`HOLD_DRY=1` can calculate a candidate holder's weight size without a CUDA allocation. It still unlinks the supplied manifest's `.ready` file before doing the size calculation, so use a fresh private manifest path. Never point a dry run at `/tmp/hold-h*.manifest` belonging to the live service. Weight size is only part of the budget; include fenced memory, contexts, scratch, PLE/indexer state, and the unchanged KV capacity before loading a candidate.

## Record the decision

For a passing candidate, retain both baseline and candidate capture directories and record in `SPEC.md`:

- Exact engine source revision, executable/kernel identity, checkpoint identity, and hardware/link state.
- `SPLIT_SHARES`, `ATTN_MAP`, routed-expert boundaries, and shared-expert placement.
- All six `actq256` totals and phase counts for each measured request, the worst spread, and the per-die profile totals used for selection.
- All measured decode rates and their arithmetic mean, plus identical input/output token sequences.
- The other measured candidates and the reason the chosen candidate wins the busiest-die objective.

The owner requested review of the collector and measured baseline without testing a different split. The resident model cannot be unloaded or its holders replaced for this work. The PR is submitted for review with that limitation; the remaining placement measurements and #54 performance criteria are still unmet.
