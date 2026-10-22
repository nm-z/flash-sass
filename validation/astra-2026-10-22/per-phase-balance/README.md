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

## KV-major gather fix and restored short-context fidelity (v65)

The optional `qsaaudit` engine command reads layer47's retained attention
scratch after a complete scalar model request. For each of six dies it checks
15 components against the existing C stage twins. Each stage uses actual GPU
inputs, so upstream discrepancies are not attributed to downstream arithmetic.
Only valid keys and real local heads are compared. The reference's additional
heads are copies used to satisfy its fixed24-Q/2-KV layout; they do not enter
the comparisons. No model arithmetic or partial-model GPU launch is added.
`BENCH_QSA_AUDIT=1` labels the workload as a component audit and collects all
results before reporting failure. Balanced raw-cache state hashes now use the
correct256-byte codes and16-byte scales per position instead of legacy widths.

The initial audit job `b2487649-4444-42c0-b85a-04e5c2df22b7` completed inference
but rejected the readback because its guard incorrectly excluded scalar prompt
groups counted as prefill chunks. The guard was corrected; no production-kernel
change was involved in that diagnostic failure.

Job `c9aa7c73-f13e-4fa1-aced-f678fe6dc53a` found exactly12 failing components:
K gather and V gather on every die. All other78 checks matched bitwise. The
[full audit](qsa-audit-v64.log) shows the first mismatch at key1 and roughly half
the gathered values wrong. The source specializes an expanded KV-major loop
(head0 tokens0..3, then head1 tokens0..3), but retained alternating blocks as if
it were token-major. It thus omitted odd-position keys from head0. The fix
retains all four consecutive head0 blocks and asserts their raw-source and
compact-destination offsets while generating SASS. The493-phase architecture
and per-die head split remain intact.

Job `1e47a219-e8da-47bc-8f1f-253ce952a4d1` runs all256 fixed-input positions
and the full head before auditing the corrected image. It has zero output
mismatches against the original fixture (previously58), and all90 component
checks pass. The [complete log](qsa-fixed-v65.log) retains the results and state
hashes. All144 comparable recurrent/convolution hashes now match the original
exactly; see [corrected hashes](balanced-fixed-v65-common-gdn.tsv) and the
existing original-common-gdn-hashes.tsv. This comparison covers the same global
head ranges on logical dies0,1,4,5, not incompatible shards on dies2,3.

Ordinary decode job `be8d413d-bc4c-49d1-8ce8-be65ec5dd083` uses64 prompt tokens
and generates128, with127 timed decode intervals. It measures19.850973046 TPS
and20.421824092 prompt positions/s. All128 generated IDs match the original
frozen baseline; [log](decode-v65.log). It is one measured execution, not a
statistical speedup claim. The former19.92 TPS run had incorrect output.

Artifacts: v63 benchmark, v64 audit-enabled engine, v65 corrected production
and body images on archy. Production SHA256:
`8abf0a7abeda96f68bf32868b7d480cb375c9b70fc6f41975af910592d5e8c2a`.
Body image SHA256:
`1cb471837975af0d628103e8c77f8dd7b859b24ecf94e7eabbeeafc7aa6a61f6`.
All canonical runs remain in the single gpuq history database. Long-context
selection and complete per-phase balance qualification remain open.

The corrected body-image job `23637881-f770-4fe8-86b8-e6beda26d0e6` also
matches all128 original generated IDs and passes all90 attention checks,
including the191-key tail case. Its [complete phase table](body-v65.tsv) and
[coverage summary](body-v65-summary.txt) contain94656 valid CTA records:
493 physical phases at two positions on six dies with16 CTAs each. All986
comparisons have complete coverage;980 are inside ten percent and six outside.
This is a new capture with corrected token history, not a matched timing
improvement over the former incorrect output.

| Position | Phase | Operation | Layer | Largest deviation from mean |
|---:|---:|---|---:|---:|
|189|11|hcwsum|0|12.195122%|
|189|181|gdn_full|17|10.344828%|
|189|461|gdn_full|45|10.059172%|
|189|491|argmax|head|25.000000%|
|189|492|amax|47|15.384615%|
|190|491|argmax|head|17.647059%|

The same1024ns GLOBALTIMER resolution and instrumentation/publication
limitations apply. HCWSUM and AMAX are short enough that one timer tick can
cross the threshold; this does not exempt them from the gate. Argmax needs
its local reduction/publication cost separated, and GDN's remaining timing
variation needs resolution. Sixteen resident SMs per die do not establish
sixteen SMs of equal useful work. The goal remains incomplete.

## Cycle-resolution diagnostic and concurrent argmax publication

Trace ABI3 adds per-SM CLOCKLO pairs alongside every existing body and explicit
poll timestamp. The reserved upper-half trace records hold six32-bit pairs per
CTA without growing the allocation or changing the production register count.
Differences are taken only within one resident CTA/SM. Modulo subtraction
handles wrap through zero; intervals of one second or more are rejected as
ambiguous. The [clock query](clocks-v66.csv), canonical job
`24638a3d-000e-4337-ba5a-4f6be693b500`, confirms the compute GPUs' current and
maximum SM clocks are1177MHz, below the2GHz wrap-bound assumption. Cycle counts
remain separate from nanoseconds and are not converted using an assumed clock.
Both measures retain instrumentation, local barriers and publication overhead.

The report accepts ABI2 evidence unchanged: its original21 columns exactly
reproduce the v65 table. ABI3 appends six die cycle maxima, their deviation and
complete-coverage/pass flags. The existing nanosecond gate is unchanged; absent
cycle data in an old trace are reported as missing, not a passing cycle check.

Cycle-image job `8d0b0d29-a9c0-46f3-b927-87f29466355e` completes the full model
and matches all128 frozen-baseline IDs. All94656 records and986 cycle comparisons
are valid. The [v66 table](body-v66.tsv) and [summary](body-v66-summary.txt) show
12 failures in nanoseconds and eight in cycles. HCWSUM and AMAX are inside ten
percent in cycles for this capture. Argmax remains outside at both positions,
with a maximum24.099911% cycle deviation. Other cycle failures are PLE's Q2
projection at both positions, two router observations, one GDN and one RMSNorm.
This exposes measurement variation and does not establish that the previously
failing small operations pass for all inputs. The full gate remains open.

The v66 image and engine are archived on archy. Body image SHA256:
`12e296fe4df84f61830e3e8939808e02cf5f73255a42167dee1a87ed5e6d5242`.

The balanced argmax specialization now uses lanes0..5 of its surviving warp
to publish the winning pair to six inboxes concurrently. The final butterfly
already leaves the same winning index/value in every lane, so no reduction
arithmetic, destination offset, local result/counter write or footer system
fence changes. The old path fetched each peer pointer and sent each pair
serially from lane0. Top-k retains its separate existing path. This targets
publication latency without padding or changing the493-phase split.

Ordinary job `371eb7c4-05e9-440d-b7cd-38ea953bcdaa` runs the complete candidate
and matches all128 original-baseline IDs. It records20.070463380 decode TPS;
[log](decode-v67.log). This is one execution, not sufficient evidence of a
statistical speedup over19.85 TPS.

Full body job `6514d0b6-a9ce-4dea-8c4f-3616d8fca335` preserves all128 baseline
IDs and yields94656 valid records and986 valid cycle comparisons. The
[v67 table](body-v67.tsv) and [summary](body-v67-summary.txt) contain seven
nanosecond failures and four cycle failures. The latter are router layer30,
GDN layer42 (10.150634%), and argmax at both positions. Argmax's cycle
deviations are24.901655% and18.808543%; concurrent publication did not resolve
its imbalance. The production output remains correct for the exercised fixture,
but the full balance goal is still open. The reduced total count of failing
observations is not attributed entirely to this change: other operations vary
between captures, and no repeated statistical qualification has been performed.

The next structural candidate is eliminating argmax's last-CTA atomic handoff
and local final reduction. Its16 per-CTA winners can instead be forwarded to
the already-existing final AMAX reduction, with a matching wider exchange
payload and base-index mapping. That requires a complete coordinated producer,
descriptor and consumer change, followed by a full-model run. It has not yet
been implemented; no completion or performance claim relies on it.

## Per-CTA head reduction and batched winner transport (v68–v70)

The structural head change is now implemented. Each of16 CTAs scans its
original logit stripe, performs the unchanged warp/CTA reduction, and writes
one `(local index, max bits)` pair. There is no last-CTA atomic increment or
second local winner reduction. The final AMAX warp reads all96 pairs: lane l
reduces pairs l,l+32,l+64, adds the vocabulary base for `pair_index/16`, then
uses the existing lowest-index tie-breaking butterfly. Its six-die layout is
guarded by `flash_balance_head_cta_abi_v1` at module load.

Initial direct-CTA publication job `0d33c1e1-bc0f-4bde-9f71-998ca6801e45`
matches all128 original IDs and measures20.106445361 TPS; [log](decode-v68.log).
Its complete trace, job `24994125-09e9-426d-bef8-37f29447901e`, has all94656
valid records and matches the original IDs. The [table](body-v68.tsv) and
[summary](body-v68-summary.txt) show five cycle failures and nine nanosecond
failures. Argmax still fails at29.578321% and19.720080% in cycles; AMAX is
inside ten percent at both positions. Scattering each CTA pair directly to
six peers therefore does not resolve the transport imbalance.

The final v70 program retains the contiguous exchange-copy descriptor for
the128-byte winner block from each die. The producer finishes all16 local
pairs before that copy, and the existing exchange acquire protects the AMAX
consumer. This replaces96 tiny peer payload writes per die with the existing
coalesced copy path. The direct-publication branch in the compatible v68 image
is disabled by the descriptor flag. All other producer fusions remain enabled.
The complete program now has494 physical phases on each die, below500, with
the same cross-die exchange count and one additional local phase boundary.

The optional `headaudit` command runs only after completed full-model scalar
inference. It reads all248320 retained logits, independently scans each CTA's
stripe, checks all96 partial indices and exact max bits, and checks each die's
final winner and recorded token. It includes losing partials; matching only
the final winner is insufficient. It reports nonfinite logits as failures.
Fixed-input job `7fa35b3b-322c-4e33-b99c-678de56e6525` has zero differences
across all256 original-baseline outputs and zero head-audit failures;
[log](head-audit-v69.log).

The final batched-copy job `b1c3b0b4-77c8-4e02-a63b-dcf5343ea324` matches all128
original IDs, passes all96 partial checks across248320 logits, and confirms
the same final winner on all six dies. It records20.101631624 timed decode
TPS and494 phases; [log](head-audit-v70.log). The audit occurs after the timed
model execution. This remains one timing observation, not a statistical
speedup claim.

The report now also validates the routed-expert records already emitted by
the full trace. For every captured position and all48 layers, all six dies
must report the same ten distinct global expert IDs, each in0..511. Missing,
duplicate or inconsistent records fail the report. The complete v67 capture
retains identical timing columns under this additional audit and has576/576
valid expert records. The v68 capture also passes all576 records. This proves
the observed routing inputs agree; it is not a substitute for full useful-work
or per-phase timing qualification.

Artifacts: v68 production/body images, v69 audit client, v70 final engine on
archy. Production image SHA256:
`aef76ab30cef5e85f8fe5f20f1338b848761279af771f78f9a9f714b09e86932`.
Body image SHA256:
`cca75cf5938e60d7e7db46d7f703a676aac67dc63e83ef793b52a1d9baf825a2`.

Final batched-copy trace job `adc291d7-4ebd-439b-9ab9-872a479e6d7f` completes
all128 original-baseline IDs with zero differences. Its [full table](body-v70.tsv)
and [summary](body-v70-summary.txt) contain94848 valid CTA records, all988 phase
comparisons, and576/576 consistent routed-expert records. There are five
nanosecond failures and six cycle failures. The head breakdown is:

| Position | Phase | Operation | Maximum cycle deviation |
|---:|---:|---|---:|
|189|491|argmax|1.545057%|
|190|491|argmax|2.111779%|
|189|492|xsend|29.552089%|
|190|492|xsend|27.182771%|
|189|493|amax|1.813933%|
|190|493|amax|2.052387%|

This isolates the remaining head timing imbalance in transport: it does not
make it disappear or satisfy the full objective. The other four cycle failures
are two Q4 projection observations, one BF16 gated read and one RMSNorm. One
AMAX nanosecond observation fails although its cycle variation is2.05%; the
unchanged nanosecond gate still retains that failure. No phase has been padded,
omitted or exempted from the complete report.

Operational constraint from Nate: physical GPUs2 and7 are reserved for his
work. Keep flash-sass computation on0,1,6,3,4,5 (logical0..5 respectively).
Existing GPU2 protected-memory fence/anchor and GPU7 stop-hook allocations
were disclosed; do not move a model shard or launch additional workloads onto
either reserved GPU. The six weight holders remain resident.

## Direct peer reads of head winners (v71)

The separate winner-copy phase is removed. Argmax writes its16 local winner
pairs with WT stores; its ordinary phase footer performs the system fence and
publishes all16 producer flags. The existing AMAX acquire waits for all96 flags
before reading the six producer-local buffers through a48-byte pointer table.
Each die still consumes all96 winners and uses the original global-index tie
rule. The full graph is493 physical phases again. No workload is added to the
reserved physical GPUs2 or7.

The scalar full-model graph protects reuse: after AMAX, the next token's
embedding exchange requires every die to have finished AMAX before any die
can progress through the model to overwrite its head partials. Stop/exit paths
do not write another set of partials. Balanced initialization already excludes
paired and multi-position execution. The pointer table is built from the
actual six compute contexts after all partial buffers exist.

AMAX issues three independent64-bit peer reads per lane before waiting for
their results, with CV loads to avoid retaining an earlier token's peer data.
Independent vocabulary-base loads are also overlapped. The peer pairs are
issued in logical order0/1,4/5,2/3; the last group includes the chipset-side
physical GPU6 in the current mapping. The [PCI topology](pci-topology-v71.txt)
and [driver bus map](gpu-bus-map-v71.txt) retain that hardware evidence. The
ordering changes latency scheduling only; all six buffers are always read.
The new `flash_balance_head_peer_abi_v1` image marker guards the pointer-table
interpretation against the older flat copied-pair format.

Complete-model job `1e1052f4-e850-4f98-b426-b64a89096c49` matches all128 frozen
baseline IDs, passes all96 partial checks across248320 retained logits, and
reports493 phases. It records20.093097685 decode TPS before the post-run audit;
[log](head-audit-v71.log). This one observation does not show a throughput
improvement over the preceding20.10 TPS result.

The v71 engine and images are archived on archy. Production SHA256:
`23016a38959b46bc8718676b30e6a38a435a4715afc1d962c35c46b5b732d876`.
Body image SHA256:
`67eca18a0ea3451c187726906dc272d1aa03cfdd4361dc2e33424317c452e298`.

Complete trace job `844d9836-64a3-4c15-ba27-73edf5771b22` matches all128
original IDs and has94656 valid CTA records, all986 valid phase/cycle
comparisons, and576 consistent expert records. Its [full table](body-v71.tsv)
and [summary](body-v71-summary.txt) show eight nanosecond failures and five
cycle failures. Argmax fails at24.707730% and20.930514%; AMAX fails at29.657860%
and26.618313%. The remaining cycle failure is GDN layer26 at13.309264%.
Logical die2 / physical GPU6 is the slowest head participant at both positions.
The direct-read path is correct on this fixture and removes a copy phase, but
does not satisfy the balance objective or show a throughput improvement.

The comparison isolates two costs that remain despite equal logit stripes:
system-fenced producer completion and peer reads. This is consistent with the
asymmetric PCI topology, not proof that topology alone explains every timing
difference. The next candidate is producing CTA winners inside the output GEMV
itself, avoiding the separate logit scan and its synchronization boundary.
That requires a coordinated production-kernel change and is not implemented
by v71. No completion claim is made from the eliminated copy descriptor.
