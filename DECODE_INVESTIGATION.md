# Decode-depth investigation

This report replaces an unqualified extrapolation from the historical 36/28/19 tok/s note with reproducible, matched measurements. The original result remains **unreproduced as a matched experiment**: SPEC records the numbers, but the record does not identify the executable, complete configuration, input IDs, clocks or timing window.

The historical 3,100-to-8,000 comparison would be 35.714 to 52.632 ms/token, a 16.917 ms/token increment. It does not establish a linear context cost or a 200k-token throughput prediction.

## Completed experiment

On archy, reuse the six resident weight arenas and compare the same full 48-layer model with budget 512, optional disk PLE, and host-pinned versus VRAM KV. The paired two-core setup uses an 8,192-position engine capacity so both placements fit while preserving the resident holders. The production core normally has a 230,000-position serving limit; its original library, cubin and settings were restored after the experiment.

Each prompt consists of fixed synthetic token IDs from seed 4242. These exercise the actual checkpoint forward pass but do not test language quality. First populate a snapshot through prompt position N-2. Then run five interleaved placement pairs with the same prompt and continuation, greedy sampling and stop tokens disabled. Each measurement generates 129 outputs and measures the 128 intervals between them. Reported N is **prompt length**; the measured visible length grows from N through N+128.

The complete 60-run trace series passed response-count, same-output, snapshot-resume, phase-order, publication, short-read and realized-placement checks. Every measured token includes all 12 production QSA chains: INT8 append, indexer, five selection phases, gather, partial attention and merge. Every die and phase zero is captured.

The engine source used for this series is [bac5b88b665ee6973488dd40b363ded3134136f6](https://github.com/nm-z/flash-sass/commit/bac5b88b665ee6973488dd40b363ded3134136f6). The later PR-head merge of current master is not retroactively labeled as the tested executable. The binary, cubin, GGUF shards, runner, host and device identities are in the archived manifest. The GGUF conversion revision is unknown and remains null.

### Traced results

All rates below include instrumentation. Median is the median of five run means; p95 is the nearest-rank percentile of all 640 individual intervals. No first repetition is discarded.

| Prompt tokens | Host KV median ms/token | Host KV tok/s | VRAM KV median ms/token | VRAM KV tok/s |
|---:|---:|---:|---:|---:|
| 256 | 29.612 | 33.770 | 29.009 | 34.472 |
| 2,048 | 35.000 | 28.572 | 31.910 | 31.339 |
| 2,051 | 34.952 | 28.611 | 31.858 | 31.389 |
| 2,052 | 35.038 | 28.540 | 31.944 | 31.305 |
| 3,100 | 35.011 | 28.563 | 31.924 | 31.325 |
| 8,000 | 35.350 | 28.288 | 32.330 | 30.931 |

The traced median increment from 3,100 to 8,000 is 0.340 ms/token with host KV and 0.406 ms/token with VRAM KV. Those values do not reproduce the historical 16.917 ms/token increment. The 65,000 and 262,000 cases are explicitly capacity-skipped in this controlled setup, not passed. These results do not prove flat decode through 262k.

## Where the time went

Each die has its own timer epoch. The analysis partitions a die's publication-to-publication interval into dispatch, exchange wait, body excluding PLE wait, barrier/send, PLE wait, between-phase gaps, publication footer and between-token gap. These components reconcile to the same die's measured interval. The maximum accepted arithmetic residual in the analysis is 0.00001 microseconds; the recorded residuals are floating-point summation noise below that bound.

Do not add the six complete die timelines. They overlap. Sender work and the receiver waiting for it are also two observations of the same dependency.

For the illustrative matched repetition 2, the local die-0 timeline gives:

| 3,100→8,000 increment | Host KV, µs/token | VRAM KV, µs/token |
|---|---:|---:|
| Client-observed interval | +339.527 | +413.861 |
| Die-0 publication interval | +340.136 | +413.208 |
| Exchange wait | +379.976 | +461.224 |
| Body excluding PLE wait | +2.624 | −8.640 |
| Barrier/send | −41.168 | −36.960 |
| Other recorded boundaries and waits | −1.296 | −2.416 |

All five repetitions are retained. Repetition 2 is an illustration, not an outlier-exclusion policy.

Summing each distinct attention layer's gather operation once gives about 5.03 ms/token with host KV and 1.96 ms/token with VRAM KV at 3,100 tokens. This is operation work across the 12 serial model layers, not a sum of complete parallel die timelines. In repetition 2, the host gather change at 8,000 tokens is **−2.576 µs/token** across all 12 layers; score scan and five-pass selection increase by 5.848 and 17.416 µs/token respectively. Those measured changes cannot explain an extra 16.917 ms/token.

Die 0 does not own attention in this placement. Its large actq256/exchange-wait bucket includes waiting for the dies that do. Calling that entire bucket “KV quantization time” would misattribute the wait.

### Slow first repetitions are visible in the record

At prompt length 2,048, host repetition 0 takes 44.093 ms/token. Its die-0 PLE wait is 9.090 ms/token, versus 0.00184 ms/token in repetition 2. At 2,051, the corresponding first repetition takes 55.841 ms/token, including 20.898 ms/token of PLE wait; repetition 2 returns to 34.952 ms/token with about 0.00189 ms/token of PLE wait.

This identifies the extra wait in these runs. It does not identify the historical run's PLE configuration, nor isolate NFS service time from operating-system cache state. No storage-cache flush was performed.

At 8,000 tokens, early repetitions also show changing die-4 clocks: the post-run samples include 1,101, 1,126 and 1,151 MHz before returning to 1,177 MHz. Per-layer work on that die and the other dies' receiver waits change with the repetitions. A sampled clock is not a continuous measurement over the entire interval.

## Completed untraced follow-up and remaining gates

The second series completed all 20 measured requests without tracing: five interleaved pairs at each of 3,100 and 8,000 prompt tokens, again with 129 outputs and 128 intervals per request. All continuations matched across KV placement. Both jobs exited 0 and recorded restoration of the normal 230,000-position serving configuration.

| Prompt tokens | Host KV median ms/token | Host KV tok/s | VRAM KV median ms/token | VRAM KV tok/s |
|---:|---:|---:|---:|---:|
| 3,100 | 49.593 | 20.164 | 46.775 | 21.379 |
| 8,000 | 45.664 | 21.899 | 42.747 | 23.394 |

These are the recorded observations, including every repetition. They cannot establish a benefit from longer context or a cost of tracing. During this series die 4 fell to 531 MHz while applications clocks requested 1,177 MHz. The sampled driver report shows **SW Power Cap active**, with **SW Thermal Slowdown inactive**, a temperature of 86°C, and a target temperature of 85°C. The recorded flag does not justify relabeling the cause as thermal throttling. The clock sample also does not continuously measure every token interval.

This clock-confounded series is unsuitable for a clean estimate of tracing overhead against the earlier series. Remaining acceptance includes a matched instrumentation-overhead experiment with controlled clocks, profiler traffic/stall counters, and supported longer-context cases. The original 36/28/19 provenance remains unavailable.

## Metadata audit and limits

The two completed series used the same engine-library SHA256 and core-executable SHA256. The traced and untraced cubins have separately recorded identities. Engine source is pinned to bac5b88; the resident core's source revision was not embedded, so its binary identity is the supported statement. Neither a later source merge nor a later metadata fix changes what these historical runs executed.

The old runner attempted nvidia-smi -q -d ECC,PCIE,CLOCK,TEMPERATURE. Driver 470 rejected that display selector with exit 2. The raw failure remains in the manifests; the ordinary per-run GPU query still supplied UUID, driver, clocks, ECC mode, PCIe link, temperature and memory snapshots. Full before/after driver-counter capture was therefore incomplete. PR #32 now requires a successful full XML query and rejects metadata failures.

The traced runner also captured an empty parent environment. The setup script records the private core's explicit launch settings and the resident engine's reload arguments, but it does not recover every original core launch setting. The untraced series explicitly records the runner/private-core launch environment. These operator setup records are retained separately and are not presented as a retrospectively complete trace manifest.

The resident core was constructed with a larger capacity than the private host core, even though both engine configurations report 8,192. Core-level PLE staging allocation can consequently differ. The weight holders are shared, disk PLE uses shard two at offset 192, and storage cache state was not flushed or independently controlled. Effective configuration equality does not erase those allocation and cache-state limits.

The compact artifacts preserve raw client observation times, generated IDs, every run mean, every die/opcode total and all rejected capacity cases. The analyzer requires successful capture exits, complete five-pair groups, matching configurations apart from KV placement, matching continuations, and exact reconstruction of the 128 client intervals before producing its summaries.

## Reproduction and evidence

Run the resident-core benchmark from PR #32 using the supplied model/placement manifests. The runner itself does not load, reload or kill cores. The setup jobs and restoration records are kept separately so their mutations and effective settings can be reviewed.

Analyze the captured outputs:

~~~bash
python3 kern/decode_forensics.py \
  --traced /path/to/paired-depth-trace3/runs \
  --untraced /path/to/paired-depth-untraced4/runs \
  --out validation/issue10
~~~

- [All run intervals, manifests, outputs and die-0 timelines](validation/issue10/traced.json.gz)
- [Untraced intervals, manifests and all 20 runs](validation/issue10/untraced.json.gz)
- [Untraced per-depth summary and every run mean](validation/issue10/untraced-summary.json)
- [All die/opcode totals for all 60 runs](validation/issue10/traced-all-die-phase-totals.json.gz)
- [Per-depth summary and every run mean](validation/issue10/traced-summary.json)
- [Illustrative matched phase deltas](validation/issue10/matched-phase-deltas.json)
- [Later die-4 power/clock snapshot](validation/issue10/die4-power-clock-snapshot.txt)
- [Raw two-position phase excerpts and complete illustrative host traces](validation/issue10/review/raw-excerpts)
- [Full raw archive identities](validation/issue10/review/raw-trace-archive.json)
- [Setup scripts, commands and restoration records](validation/issue10/review/setup)
- [Artifact SHA256 identities](validation/issue10/IDENTITIES.json)

The complete raw per-phase CSV collection is retained at /home/nate/claude/flash-sass-prs/paired-depth-trace3/runs on archy. Its full content is not duplicated in git. The compact review artifacts preserve every run and opcode total. Two-position phase excerpts are attached for dies 0 and 4 at 3,100/8,000 in both KV placements, together with the complete corresponding host traces. The archive manifest identifies every original CSV stream by size and SHA256.

Refs #10. Trace format and lifecycle work are in #13; reproducible capture in #14; shared timing API in #15; representative gather inputs in #16.
