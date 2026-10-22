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

## Cooperative PLE and compact per-GPU prefetch cache

`kern/balance/ple.lua` replaces the balanced image's whole gather/quantization
operation in place. Sixteen CTAs own the sixteen160-value n-gram heads. Each
CTA hashes its head in lane0, explicitly reconverges the warp, cooperatively
loads the45 packed u16 words in two warp instructions, and uses five warps
to decode/quantize its five32-value IQ4_NL blocks. Hash and dequantization
arithmetic retain the production constants/order. Every CTA performs useful
work; no phase or exchange is added. The complete program remains493 phases.

The first complete-model attempt, job
`8e2a6c19-5af9-4d57-b4f2-a7828338c644`, failed with a hardware stack error:
the SSY target was the SYNC instruction itself. Moving the reconvergence
target past SYNC corrected that control flow without reverting the operation.
Job `76963f67-bb35-485b-9763-83a3858c9f96` then completed64/128 at19.930911532
TPS. Its128 output IDs exactly match `balanced-short-output.ids`.

The first cooperative trace, job
`04ef1e32-0dc8-4b94-80b1-7ad42b65d60d`, has complete94656-record coverage;
[body-v60.tsv](body-v60.tsv) retains every phase. It reduces PLE execution
substantially, but host-mapped reads still leave GPU6 slower. The six-die
maximum CTA body-minus-poll observations for position189 are:

| PLE implementation | Die0 | Die1 | Die2 / GPU6 | Die3 | Die4 | Die5 |
|---|---:|---:|---:|---:|---:|---:|
| Original balanced gather, v58 |81.920|81.920|159.744|80.896|80.896|81.920|
| Cooperative host reads, v60 |18.432|16.384|38.912|22.528|23.552|26.624|
| Cooperative device-row cache, v62 |16.384|16.384|14.336|16.384|15.360|14.336|

All values are instrumented microseconds; the same GLOBALTIMER resolution and
overhead limitations apply. These are measured operator observations, not a
statistical speedup claim for complete decode.

The engine now DMA-stages only the compact prefetched rows on each balanced
GPU, preserving the full table's off-accelerator capacity. At context32768,
each die reserves47188800 row bytes plus131080 epoch bytes. Each position's
1440-byte payload and4-byte epoch copy are ordered in that die's nonblocking
stream. The kernel polls its own device epoch, allowing each die to start as
soon as its own payload arrives. There is no cross-die readiness padding or
runtime cache fallback. Epoch spans are cleared before a new request.
The pinned epoch copy avoids launching a helper memset kernel alongside the
fully resident model. Host queue-submission time is not reported as actual
device readiness; that point is measured by the kernel's poll clocks.

Job `77f3113c-74d5-4936-899b-376919d191f9` completes the complete64/128 model at
19.921056333 TPS with the compact caches. All128 IDs again exactly match the
pre-change balanced output. It stages191 positions,1650240 payload bytes
across six dies,4584 epoch bytes, and11.025221ms of CPU queue-submission time.
Production cubin FNV is `e9bc7ec3e5262163`; its SHA256 is
`16efd39bbdf2fa85b0643e9698e431869e2e3d7590059d5c68a49214d24b896b`.
The cooperative images are archived in archy's `flash-per-phase-balance-v60`;
the engine with epoch-span clearing and the updated client are in v62.

Full-model trace job `fbf5984a-6130-4800-ad84-78d002aa7cea` produces
[body-v62.tsv](body-v62.tsv): all94656 records and986 phase comparisons have
valid coverage. PLE is inside ten percent at both positions (maximum
deviations7.692308% and6.666667%). Eight other comparisons fail the current
gate: HCWSUM, AMAX, argmax and two GDN observations. The maximum observed
deviation is20%, with the short-phase quantization limitation still open.
This is978/986 comparisons inside the current gate, not goal completion.

## Fixed-input numerical boundary

The private client now honours `BENCH_SOCKET_TIMEOUT` consistently with the
resident client, to permit future long-prompt qualification without the old
120-second private-socket timeout. No long-context pass is claimed yet.

Fixed-input job `89b8999a-ab42-404f-9221-2a33e7019f1e` completes all256 causal
positions and full-head outputs against the original baseline fixture from
job374f3bd1. It reports58 differing outputs, first at position6. The fixture
is four copies of the same64-token prompt; the prompt constant matches commit
00e0981 exactly. All positions execute before the client reports failure;
there is no differing generated history in this comparison.

The original and balanced layouts have the same eight global GDN heads on
logical dies0,1,4,5. Their comparable combined recurrent/convolution hashes
match through layers0,1,2 and first differ at layer4 on all four dies. The
[144-row comparison](common-gdn-comparison.tsv) preserves every comparable
GDN layer, with [original hashes](original-common-gdn-hashes.tsv) and
[balanced hashes](balanced-common-gdn-hashes.tsv). This narrows the numerical
boundary to the interval after GDN2 and before GDN4: output/FFN2 and QSA/FFN3
still need to be separated. It does not yet prove QSA alone is responsible.

The [256-position original fixture](original-v39-fixed-reference.ids),
[balanced outputs](balanced-fixed-output.ids) and [all58 mismatches](fixed-mismatches.tsv)
are retained. The balanced architecture remains in place while its numerical
cause is investigated; the original-model correctness gate is still open.
