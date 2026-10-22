# Complete balanced kernel: first execution and phase evidence

The objective remains open: every compute die must perform useful work within
ten percent of the six-die mean for every full-model phase. Neither equal
phase counts nor a mostly passing table proves completion.

The production image runs all48 layers and the full head, with493 physical
phases and980 descriptors on every die. MTP and FP8 are off. Logical compute
dies0–5 map to physical GPUs0,1,6,3,4,5; GPU2 retains its protective fence.
All jobs below are in archy's single canonical GPUq database:
`/home/nate/.local/state/gpuq/history.sqlite3`.

## Successful production executions

| Canonical job | Workload | Result |
|---|---|---|
| `81b8aeaa-8f0b-413b-9d01-17b1bd2e4ed8` | Full64-token prompt/128 generated, first execution | Completed191 causal positions;17.039430938 TPS. No reference check. |
| `4f8abd51-8bff-4e65-870d-d0d37c9da28e` | Same workload, two requests |17.358816652/17.361331239 TPS. All128 outputs repeat; both requests have zero six-die/stream/input-history mismatches. This is same-kernel repeatability, not original-model correctness. |
| `9e6a0a07-7079-472c-91e3-2e46996ca7e7` | Same full workload after GPU6 clock correction |19.573653181 TPS;20.359337432 prompt tokens/s. Full191 positions complete. |

The production cubin FNV is `6a3d00cf0fbb0882`, archived under archy's
`/home/nate/work/flash-per-phase-balance-v52`. The corrected QSA cloner is in
commit6d5b0e1. Before that fix, complete model attempts faulted in the first
QSA composite. Host-mapped diagnostic job
`357dfcb4-fb54-4bad-a154-9fb9bcbecb44` showed all six dies completing physical
phase40, then failing in phase41/descriptor69. The cloner had treated SASS
completion masks as labels and lost instructions.

GPUq samples in job4f8abd51 show all five ordinary compute GPUs fixed at1177MHz,
but physical GPU6 at873MHz; all six memory clocks were2505MHz. Query job
`5331c729-c718-4553-8ca1-c39487100cd2` records GPU6's original application
clock873MHz and supported1177MHz setting. Job
`bf227698-75cf-43f5-9493-019f9000ff70` changes only GPU6 to2505/1177MHz.
The19.57 TPS request follows that change. This is still slower than the
historical ordinary baseline; no speedup over that baseline is claimed.

## Complete per-phase body capture

Job `603a3a90-fd06-425e-ae7a-8e6fc94eafcc`, image archived under
`/home/nate/work/flash-per-phase-balance-v58`, completes the full model and
captures positions189 and190 on every compute die. [body-v58.tsv](body-v58.tsv)
retains every phase; the C reporter consumes the original integer JSON clocks
directly from the database. It reports:

```
positions=2/2 phases_per_position=493 cta_records=94656
invalid_records=0 invalid_phases=0 failing_phases=11
max_deviation_pct=68.953069 coverage_complete=1
body_timing_within_10pct=0
```

Thus975 of986 phase comparisons are inside the current timing gate, and11
are not. PLE gathering is the clear persistent outlier: physical GPU6 takes
156.672–159.744us against approximately80–83us on the other GPUs, even after
explicit host-ready polling is subtracted. Other failures include argmax,
the layer38 Q6 projection, RMSNorm, router, and small HCWSUM/AMAX phases.

These are body-minus-explicit-poll observations, not pure arithmetic times.
They retain local barriers, publication/dispatch and trace overhead. The
GLOBALTIMER samples are quantized to1024ns, making some small-phase failures
ambiguous; finer timing and an overhead/uncertainty budget remain required.
This finite two-position capture is not an all-context qualification.

All16 resident CTA SM IDs are captured. The source uses the ptxas11.4-verified
Maxwell encoding `SR_VIRTID` followed by `BFE.U32 ...,0x914`; the compile-only
fixture is `kern/balance/smid.ptx`. Resident placement is **not** useful
occupancy. For example, the current PLE gather computes on CTA0, although all
16 CTAs remain resident. A complete operation/CTA/warp work ledger is still
required.

Earlier capture `cc271f6f-0645-4064-ad22-b2d2823a2a7a` has complete body timing
coverage, but its SM-ID instruction incorrectly encoded LANEID. Its retained
[body-v57.tsv](body-v57.tsv) must not be used for SM occupancy. The fixed
tracer also keeps branches warp-uniform and derives metadata addresses with
positive offsets: the earlier negative u32 adjustment advanced the high
pointer word by4GiB for CTA>0. The matching driver faults were writes.

## Exchange observations before clock correction

Job `3f1044a2-bf97-41ec-a35d-dfeae365c142` completes the same full request with
exchange-only instrumentation. All4824 records are valid:402 exchanges per
token, two positions, six dies. The [exchange summary](exchange-summary.md),
[every exchange](exchanges.csv), [operation summary](exchange-operations.csv)
and [integer TSV input](exchanges.tsv) are derived from that canonical job.

The first five logical dies spend11.477–12.516ms/token from local completion
to acquisition, versus4.370ms on logical die2. The report separates causal
producer-skew intervals from communication/polling intervals; it does not
claim exact cross-device clock alignment or isolated link latency. GPU6 was
still at873MHz during this capture, so it is not the final balanced-clock
exchange measurement.

## Original-model fidelity is not qualified

[original-v39-short-reference.ids](original-v39-short-reference.ids) is the
first128 generated IDs from the completed original-baseline repeatability
job `2ce8b88b-79bb-41b6-bc35-37490b36392f`, whose frozen source uses the same
64-token fixture and greedy decoding. Its SHA256 is
`a79d6da2253f4524ae45e9c38e8e1460cd17e2ee2e593a568534e1813014d43e`.
[balanced-short-output.ids](balanced-short-output.ids) comes from the full
production job9e6a0a07. The two128-ID sequences differ at117 positions, first
at index8. Later mismatches follow divergent generated histories and are not
117 independent numerical failures. Teacher-forced/state comparison is still
needed to locate the first numerical difference; repeatability alone is not
an oracle pass. Preserve the balanced architecture while fixing that cause.

The model holders remain resident. The main public flash engine is not
serving; these requests use private complete-model cores. GPU2's fence,
GPU7's original128K-context service, CPU service, persistence mode and thermal/
power monitoring were restored after the cold boots. GPU6's1177MHz application
clock is an intentional recorded change.
