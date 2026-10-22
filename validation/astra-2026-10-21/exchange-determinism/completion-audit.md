# Evidence against the three requested items

This audit concerns the exchange/determinism request. It does not claim the
60/140 TPS targets, 2–4 exchanges per layer, MTP completion, arbitrary-prompt
correctness, or a speedup from replication.

| Requirement | Authoritative evidence | Outcome and limit |
|---|---|---|
| Measure every exchange from each die finishing its work to acquire completion | Final trace job `73ed8630-36f3-47a5-9e7d-36fd7db0584f`; [every warm observation](v39-exchanges/exchanges.csv); compressed raw JSON and TSV | 12,456 valid observations; three measured requests per image; all six dies; 354 normal/338 replicated exchanges per token/die. Raw records retain all 16 CTA timestamps and all 96 producer-flag observations. |
| Separate producer skew from the remaining exchange wait | [Per-die report](v39-exchanges/exchange-summary.md), [operation families](v39-exchanges/exchange-operations.csv), `phase/exchange_report.lua` | Direct device-local finish→acquire, plus causal cross-device clock bounds per arm/request/position. Remainder includes publication/transport/cache/polling/rendezvous; no pure wire-latency claim. |
| Account for instrumentation overhead | Control `4c91b069-5d11-4576-942c-fb35d3b9e93e` versus trace job; [method and numbers](v39-exchanges/README.md) | Same frozen model/split/settings, four64/16 requests per image, first warmup. Added mean0.216289/0.162783 ms/token. One sequential comparison, not a statistical bound. |
| Cut actual exchanges by changing the split, preserving the full model | Matched job `b761ed99-563c-45d7-ae52-16665c5c6bd5`; per-die prof headers and wait counters; `phase/gr_replica.h`, `phase/gr_local.lua` | Eight replicated FFN GR stages:354→338 peer acquires,144→160 local joins,499 phases,48 layers and full head. No additional fusion. Larger48/24-stage replication OOM;2–4 exchanges/layer is not achieved. |
| Measure the revised kernel rather than infer a speedup | [Matched operation profile](v39-split-profile/operations.md), raw job export, [control/trace rates](v39-exchanges/README.md) | Matched64/128 warm31.4909 TPS normal versus30.1612 replicated. The split is slower and remains opt-in. |
| Establish original-baseline repeatability before using it as an oracle | Original v39 long job `6842afd1-0e3b-4063-aaa6-56bdb6736d81`; original-only ordinary `2ce8b88b-79bb-41b6-bc35-37490b36392f`, fixed `374f3bd1-212a-48d8-a9b4-f100c202c329` | Two4096/16, four64/128 and four fixed256-position requests pass; output tokens,216 recurrent-state hashes,432 component-value rows/request, six-die token consensus and PLE history are checked. Finite-workload evidence, not proof over all inputs. |
| Establish the same gates for the normal and replicated full production images | Normal long: `6842afd1-0e3b-4063-aaa6-56bdb6736d81`; replicated long: `bf4c9410-f917-4b51-bc87-761bcdfe6f51`; both ordinary: `b761ed99-563c-45d7-ae52-16665c5c6bd5`; both fixed: `5930993a-ed67-43cc-8e28-acc21ad9cf93` | All selected gates pass with zero output/state/component/consensus/history mismatches. The complete48-layer schedule and full head execute, with499 phases for both phase images. |
| Perform cross-image output comparisons only after repeatability gates | Saved first requests in the completed gate jobs | Zero original/normal differences for128 short-generated,256 fixed-input and16 long-generated outputs; zero normal/replicated differences for the same counts. These are greedy finite workloads; no lossless stochastic-sampling claim. |
| Preserve source and measurement provenance | `v39-SHA256SUMS`, immutable v39 source archive and cubins, canonical GPUQ jobs and raw exports | All comparisons use frozen v39 images. Interrupted capture and pre-admission retry are separately recorded and not counted as kernel results. |
| Restore serving configuration | Final config job `78faf82c-a716-4d9d-b31c-718b3760be34` | Full48-layer v8 serving baseline restored, `baseline-amax13.cubin`, FNVd215dc737080be6d, context32768, FP32 residuals, MTP off. Experimental replication is not left serving. |

The final recurrent-state fix retains the merged bodies and retires pending
completion barriers before crossing execution pipelines. Earlier readiness,
operand and publication repairs remain. The passing gates support the combined
v39 image; they do not independently prove that every conservative repair was
necessary. The documented earlier failures are retained instead of relabeled
as passing or removed from the record.

The Desktop export was refreshed after the last GPUQ job using SQLite's backup
API, then transferred to engi and checked with `PRAGMA quick_check` (`ok`).
Both the canonical database and snapshot contain1,166 jobs, including final
serving-config job78faf82c-a716-4d9d-b31c-718b3760be34. The delivered file is
`/home/nate/Desktop/flash-sass-gpuq-history-2026-10-21.sqlite3`,4,039,528,448
bytes, mode0600. Its journal mode is DELETE so the delivered artifact is one
standalone file, with no Desktop WAL/SHM sidecars. The existing filename is
retained even though this refresh completed on2026-10-22. The stale polling
shell paused for capture was resumed after the runs and backup completed.
