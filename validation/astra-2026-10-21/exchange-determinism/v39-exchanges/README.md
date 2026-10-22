# Final v39 exchange measurements and tracing overhead

All GPU work is recorded in archy's canonical `~/.local/state/gpuq/history.sqlite3`.

| Measurement | GPUQ job |
|---|---|
| Untraced control | `4c91b069-5d11-4576-942c-fb35d3b9e93e` |
| Traced split comparison | `73ed8630-36f3-47a5-9e7d-36fd7db0584f` |

Both jobs use frozen v39 images/library, the full 48-layer model and full head,
FP32 residuals, no MTP, 499 phases, six-die attention balance, a 64-token prompt
and 16 generated tokens. Both arms perform four requests; the first is warmup.
The candidate replicates eight FFN GR stages. Profiling and post-request
state/consensus/history audits are enabled in both jobs. All requests pass
same-image repeatability and consensus. The traced images additionally record
one token per request and dump the trace after each request.

`trace-warm.tsv.gz` contains 12,456 valid records: three measured requests ×
six dies × (354 normal-split + 338 replicated-split exchanges). The report has
six independent arm/request/position clock groups, complete die coverage for
every exchange epoch, and consistent causal clock bounds. Run0 is excluded.
`exchanges.csv` retains every observation; `exchange-summary.md` sums by die;
`exchange-operations.csv` groups producer/consumer families. Grouped attention
rows are normalized only over the die/token samples on which they occur.

The routed-down → GR-read family illustrates the decomposition:

| Split | Finish → acquire, µs/token/die | Producer-skew bounds | Remaining publication/communication/poll bounds |
|---|---:|---:|---:|
| Normal | 2959.076 | 2543.730–2821.973 | 137.102–415.346 |
| Eight stages replicated | 2940.359 | 2527.801–2810.084 | 130.276–412.558 |

These are six-die averages of instrumented observations. Device-local
finish→acquire intervals are measured directly. Cross-device skew uses causal
clock-offset bounds, assuming negligible drift within one token. The remainder
includes publication, transport, invalidation, polling and CTA rendezvous;
it is **not a measurement of pure wire latency**. Earlier tables that charged
this interval to routed-down/GR-read cannot distinguish those components.

The matched overhead calculation uses the three warm requests per arm and
host-monotonic decode intervals. Latency is the arithmetic mean of each
request's `decode_ms / decode_intervals`; TPS is the arithmetic mean of each
request's reported decode TPS.

| Split | Untraced ms/token | Traced ms/token | Added ms/token | Added latency | Untraced TPS | Traced TPS |
|---|---:|---:|---:|---:|---:|---:|
| Normal | 31.722857 | 31.939147 | 0.216289 | 0.682% | 31.5230 | 31.3095 |
| Eight stages replicated | 32.998848 | 33.161631 | 0.162783 | 0.493% | 30.3041 | 30.1553 |

This is one sequential control/traced comparison, not an interleaved experiment
or a statistical upper bound. The differences include ordinary run-to-run
drift as well as instrumentation. Traced intervals are not silently substituted
for uninstrumented timings. Replication removes exchanges but is slower in
this matched workload too.

Reproduce the report without executing the model:

```
gzip -dc trace-warm.tsv.gz > /tmp/flash-v39-warm.tsv
lua kern/phase/exchange_report.lua /tmp/flash-v39-warm.tsv OUTPUT_DIRECTORY
```

The compressed raw GPUQ JSON is `../job-73ed8630.json.gz`; the untraced export
is `../job-4c91b069.json`. They are exports from the single canonical database,
not separate measurement databases.
