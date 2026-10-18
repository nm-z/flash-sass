# Issue #13: schema-2 acceptance evidence

Engine source: **c7d0a6e14e99edd9e91ea16c010ae7938ac647e0**.
The guarded reload helper is from **99c7134a0ef68ae9a37a827de05e8ab2fd8df17b**.
The later callback-window assertion revalidates the already captured cancellation row.
These later changes do not change the measured C engine or kernel images.

The runs used the real 48-layer, six-die model on archy's Tesla M60s.
The CTA-0 matrix began at **2026-10-18 01:13:18 UTC**; the 16-CTA matrix began at
**01:28:02 UTC**. All figures below are derived from the committed raw captures,
not synthetic model timings. [summary.json](summary.json) contains the full
derived results and exact artifact identities.

## Acceptance results

| Requirement | Measured evidence |
| --- | --- |
| Every phase, phase zero, bounded token positions and matching request identity on all dies | 12 successful traced benchmark requests contain **1,534,464 phase records**. There are 1,154 phases on dies 0, 2 and 3, and 1,214 on dies 1, 4 and 5. All 48 observed opcode names and all phase instances are retained. |
| Dispatch, sampled body, exchange readiness, PLE readiness and framework intervals | Schema 2 samples selected-opcode entry after dispatch and opcode return before framework barriers. Both actual GPU layout tests passed ordering, peer-generation/publication brackets, canaries, and layout/stride rejection. |
| Per-die reconciliation and overlap | **1,296 CTA/token partitions** and **216 die/token envelopes** in the benchmark captures have **0 ns arithmetic residual**. Every captured CTA is checked; die envelopes use interval unions. |
| Cross-die critical path | All 12 benchmark windows have feasible generation-matched clock constraints and finite GPU-span bounds, with **6,144–7,168 ns interval widths**. Reports retain the causal witness and its bucket accounting. |
| Lifecycle | **10 real cases passed**: one output, four outputs, snapshot/resume with both stop slots, callback cancellation, argument/image errors, recovery, and paired prefix/resume. Eight available diagnostic traces also reconcile exactly. |
| Callback timing on cancellation | At aborted position 2, callback duration was **7,354 ns**, callback return preceded abort entry by **224 ns**, and the separately recorded abort interval was **93,079 ns**. The latest checker requires this position to be present in the validated window. |
| Matched tracing overhead | Three ABBA blocks per layout, six measured requests per mode, two warmups before each measured slot, identical output IDs within each matrix, and raw client timestamps. Results are below. |
| Allowed implementation languages | The Python checker and committed Python runner were removed. Current validators, reconciliation, controller and experiment runner use Ruby; engine/layout changes use C, assembly and Perl. |

### Matched overhead

Both sides of each comparison use the same engine library and runtime settings:
48/48 layers, six dies, `maxt=512`, `img_rows=1`, `nblk=512`,
`kv_host=0`, disk PLE, `two_position=0`, `prefill_chunk=0`, greedy sampling,
prompt IDs `10 11`, and the recorded split/attention map. Only the selected image
and enabled trace capacity differ. The ordinary image has no trace instructions.
The engine library SHA-256 is
`2bfea9ea0f82929fc1f37b5de82ddb7bf6894356033353bb3f471083cb661347`.

| Layout and output count | Ordinary steady delivery, ms/token | Traced steady delivery, ms/token | Mean ABBA overhead | ABBA block range |
| --- | ---: | ---: | ---: | ---: |
| CTA 0, four outputs | 27.479251 | 31.452815 | **14.460%** | 14.383–14.516% |
| All 16 CTAs, two outputs | 27.503921 | 32.865557 | **19.494%** | 19.218–19.724% |

Steady delivery is the client-observed interval from first to last token arrival
divided by the number of intervening tokens. Socket batching remains part of
this metric. Ordinary/traced sample standard deviations were respectively
0.016938/0.022327 ms for CTA 0 and 0.071471/0.029042 ms for 16 CTAs.
The overhead statistic averages the three within-block ratios, each comparing
two ordinary and two traced measurements.

Mean full-request client wall times were **142.551595 / 162.200139 ms**
for CTA 0 (**13.783%** mean block overhead) and
**87.682894 / 103.497653 ms** for 16 CTAs (**18.037%**).
First-token and rounded engine timings are retained in the two summary JSON
files; [overhead-samples.csv](overhead-samples.csv) contains every measured sample.

Each matrix had 24 warmups and 12 measurements. All 72 accepted request
preflights recorded idle GPU utilization and a successful `ps` inventory.
All recorded before/after SM and memory clocks on dies 0–5 were
**1,177 / 2,505 MHz**. Temperatures and other GPU observations are in the raw
records and summary. The shared validation lock was held throughout each
matrix, and unknown GPU processes or active workers cause the runner to wait.

These are short-generation measurements. The two layout matrices have
different output counts and are compared to their own matched ordinary runs.
The 16-CTA count was reduced after the documented allocation failure below.

### Exact per-die reconciliation

For example, position 2 of the first 16-CTA measurement gives:

| Die | Envelope duration, ns | Union of phase envelopes, ns | Gaps, ns | Publication tail, ns | Residual, ns |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 32,861,184 | 32,825,344 | 34,816 | 1,024 | **0** |
| 1 | 32,860,160 | 32,817,152 | 41,984 | 1,024 | **0** |
| 2 | 32,861,184 | 32,838,656 | 20,480 | 2,048 | **0** |
| 3 | 32,860,160 | 32,846,848 | 12,288 | 1,024 | **0** |
| 4 | 32,861,184 | 32,825,344 | 34,816 | 1,024 | **0** |
| 5 | 32,860,160 | 32,819,200 | 39,936 | 1,024 | **0** |

The full [per-die table](per-die-reconciliation.csv) retains overlap separately.
Individual CTA partitions contain setup, dispatch, sampled body, exchange wait,
PLE wait, barriers/send brackets, phase gaps and publication tail. The
previous-publication transition is separate from the current token's phase-zero
bucket. Single-position analyses of all positions in both representative
captures also have zero residual and no imported predecessor duration.

### Cross-die reconstruction and its precision

The first CTA-0 capture has a conditional enclosing GPU span of
**126.035968–126.042112 ms**. The first all-CTA capture has a span of
**65.872896–65.879040 ms**. Each interval is **6.144 µs** wide.
Their complete witnesses are in
[the CTA-0 report](reconciliation/cta1-slot-02-window.json.gz) and
[the 16-CTA report](reconciliation/cta16-slot-02-window.json.gz).

The graph matches outgoing and incoming exchange generations, uses
`send_before` as the causal publication boundary, joins all observed CTA
arrivals at actual grid barriers, and takes maxima at branch merges.
It preserves independent device epochs and measured CTA start skew.
The all-CTA example includes 684 matched generations and 12,480 grid joins.
Its witness's measured local buckets total 65,873,920 ns; subtracting its
1,024 ns crossing allowance gives the 65,872,896 ns sampled chain.

Bounds assume constant relative GPU clock offsets during each captured window
and a **1,024 ns timestamp allowance**. There is no independently measured
drift bound. The span interval width is temporal uncertainty, separately from
the exact integer reconciliation residual. A witness includes sampled wait
residency; publication brackets and receiver polling can leave the exact last
visible producer ambiguous. CTA-0 data observes only one CTA per die, whereas
the all-CTA matrix observes all 16. Neither mode sums work across simultaneous
dies or CTAs. Host times remain in their own clock domain.

### Host and lifecycle evidence

The raw host records retain PLE staging/read start and end, bytes, short reads,
readiness publication brackets, polling/readback batches and callback events.
For positions 1 and 2 of the representative 16-CTA request, the PLE read loops
took 21,904 and 27,116 ns, each reading 1,440 bytes with zero short reads.
These are observed read-loop durations through the filesystem cache.

The cancellation trace identifies `abort_reason=callback` at position 2:
`callback_end_ns=68843962922038` and
`abort_start_ns=68843962922262`. Its completed diagnostic prefix covers
positions 1–10 of a request that requested 42 full positions.
[The lifecycle results](lifecycle-results.json) and
[the explicit callback-window recheck](callback-window-check.json) preserve
the evidence. Prefix, snapshot and cancellation captures are kept separate
from benchmark samples.

### Memory failure and service restoration

The initial four-output, 16-CTA attempt requested a 32 MiB trace allocation per
die while die 3 had approximately 31 MiB free. CUDA allocation failed during
reload. The core retried its former library with the newly requested trace
environment, failed again, and exited. That attempt contains no successful
traced measurement and contributes no overhead result.

The original service was restarted from the unchanged baseline library and
cubin, with its recorded configuration, using the existing weight holders.
The restoration wrapper's inherited validation lock was also corrected by
closing its descriptor before starting the replacement core. No other worker's
process was killed. The final restored core is PID 3575268; the restoration
records show revision `edc66d9`, the original cubin FNV
`61f38fc0784534a2`, and all original reported runtime fields.

The new `trace_control.rb` memory preflight requires an ordinary configuration
with identical shared settings before enabling tracing, credits only the trace
buffer guaranteed to be released on reload, and retains an 8 MiB reserve on
each configured GPU. A real regression check rejected the oversized request
**before sending reload**, confirmed that the ordinary engine remained
available, and restored the baseline. The successful 16-CTA matrix uses a
16 MiB buffer per die. This guard protects the controlled validation workflow;
raw core reload commands still require appropriate resource sizing.

## Artifacts and reproduction

- [CTA-0 raw bundle](cta1-raw.tar.gz): all 36 request/config/event records, measured six-die/host traces, initial controller inputs, exact source/build identities, and restoration.
- [16-CTA raw bundle](cta16-raw.tar.gz): the matched two-output matrix, controller inputs, identities, memory checks and restoration.
- [Lifecycle raw bundle](lifecycle-raw.tar.gz): all ten real cases, preflight inventories, protocol, raw traces and paired state hashes.
- [Operational diagnostics](operational-diagnostics.tar.gz): failed allocation, guarded rejection, and recovery records.
- [Reload inventories](reload-preflights.tar.gz) and [layout inventories](layout-preflights.tar.gz): the corresponding GPU/process checks.
- [Reconciliation reports](reconciliation/): all 12 benchmark windows, representative individual positions, and eight available lifecycle traces. The eight lifecycle analyses use `--allow-incomplete` so stopped and aborted prefixes remain diagnostic data.
- [Recorded commands](commands/recorded-commands.md), [source identities for CTA 0](source-manifest-cta1.txt), and [source identities for 16 CTAs](source-manifest-cta16.txt).
- [Artifact hashes](SHA256SUMS) cover every committed evidence file except that checksum file itself.

The GPU layout logs exercise the actual images with zero-work opcode bodies
and six local flag banks. They validate the ABI and control flow. The
[CPU regression log](reconcile-cpu.log) covers independent epoch shifts above
2^53, known span bounds, single-position scope, missing CTAs and inconsistent
generations. Model output equivalence is measured separately by the real
ordinary/traced requests.

To reproduce analysis without a GPU, extract the raw bundles into one scratch
directory, run `ruby kern/trace_reconcile.rb --dies 6 --out REPORT.json` with
one measured directory's six `trace-?.raw.gz` files and its
`hosttrace.raw.gz`, and compare the report's residuals and bounds.
The committed summary script regenerates the aggregate JSON and CSV files
from the extracted runs and reconciliation reports.
