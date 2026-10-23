# Issue #13 schema-2 evidence on source commit `69b1e7c`

This evidence set records the full six-die, 48/48-layer phase-trace validation on source commit `69b1e7ce5da3d11fd33fe6a5cf23a34e2eb12170`, before master `84ceabf` added automatic prefill selection. The current merged-head checks and additional ABBA blocks are in [schema2-c250e88](../schema2-c250e88/README.md). These captures remain the full three-block overhead matrices and ten-case lifecycle run; their engine/library identity is recorded separately and is not presented as the `c250e88` library.

## Source, weights and GPUQ runs

[`source-manifest.txt`](source-manifest.txt) and [`weights-manifest.txt`](weights-manifest.txt) identify the engine, kernels, validators, runner, model inputs and PLE shard used by these captures. The real model ran on six Tesla M60 dies across all 48 layers.

| Artifact | SHA-256 |
| --- | --- |
| `libengine.so` | `39b6059978b9bdb1df126b2d6ca8155b29c4542f98773bf5be2281dea331be33` |
| Ordinary `die.cubin` | `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423` |
| CTA1 `die_trace.cubin` | `8ad2ae307c6802ac539de3261f6584aa838ee2f83563656efac69d336a37dd56` |
| CTA16 `die_trace16.cubin` | `de7b02550392244e30c5e9bbf21a12e866037f8ca34503dd32cef56045323b43` |
| `trace_layout_test` | `6094cf4ea07307d37913b9834ed6c0bcb7e7ff7519a9908815b78fada9455b99` |
| `trace_preflight` | `c8252c2f6f64b5390fd8dd50a4ab9680022be854191d6504620eb6604b3fdee0` |

The current source tree keeps `DIE_TRACE` and `DIE_CTA_TRACE` mutually exclusive because their control-register ABIs overlap. The traced Make targets explicitly set `DIE_CTA_TRACE=0`; the generator rejects both modes being enabled.

| Run | GPUQ job IDs | Status |
| --- | --- | --- |
| CTA1, three ABBA blocks | `ea4f6791-115a-4af5-92d2-edef5f9392fc`, `52eeb266-b745-42a8-ae7e-e1601618144a`, `6633d340-8881-4859-8910-e94325779a07` | All succeeded; original configuration restored after each block |
| CTA16, three ABBA blocks | `1c6b24b1-d36f-4abb-83b9-5f8b529f31f5`, `619640fb-f0b3-4051-9c95-4335520b60f1`, `16fb492c-7294-4c0f-b0df-e501fed60d4d` | All succeeded; original configuration restored after each block |
| CTA1 layout test | `acef6590-250c-4f72-acab-d7159aea3e8b` | Passed |
| CTA16 layout test | `bfa7afd0-29ea-42ed-9aac-a17bc2ed1264` | Passed |
| Ten-case lifecycle suite | `9cca824f-d276-4ce5-ab62-634c9fb44267` | Passed; original configuration restored |

The accepted runners use the basename wrapper in [`commands/`](commands/) so GPUQ does not scan an absolute output directory. The separate validation lock avoids colliding with GPUQ's shared engine lock. Earlier malformed, partial or collector-failed jobs are excluded from the accepted results; the accepted IDs above have direct argv and exit-zero records in the GPUQ history.

[`gpuq-jobs.json`](gpuq-jobs.json) preserves the exact argv, provenance and status for all nine accepted GPUQ jobs above; each succeeded with exit code 0.

[`SHA256SUMS`](SHA256SUMS) covers all committed evidence in this directory except the checksum file itself.

## Matched tracing overhead

Each block used ordinary, traced, traced, ordinary slots, two warmups per slot and matched runtime settings within its CTA layout. All 12 measured requests per layout returned identical token IDs.

| Layout | Ordinary steady delivery | Traced steady delivery | Mean ABBA overhead | Block range | Wall overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| CTA1, 3 outputs | 24.607620 ms/token | 28.815131 ms/token | 17.0984% | 17.0736–17.1183% | 15.9220% (15.5140–16.1368%) |
| CTA16, 2 outputs | 24.645580 ms/token | 29.909494 ms/token | 21.3585% | 21.2331–21.5256% | 19.9895% (19.8753–20.1233%) |

CTA1 first-token overhead averaged 14.3225%; rounded engine run-time overhead averaged 16.5234%. CTA16 first-token overhead averaged 19.3648%; rounded run-time overhead averaged 20.4774%. The matrices have different output counts and each layout is compared with its own matched ordinary mode.

The per-block records, raw client timestamps, output IDs, GPU/process inventories, configurations and summaries are retained in the six `overhead-cta*-b*-gpuqfix2/` directories and the CTA1/CTA16 summary JSON files.

## Per-die reconciliation, cross-die spans and host events

| Layout | Traced captures | CTA/token partitions | Die/token envelopes | Arithmetic residual | Causal data per capture |
| --- | ---: | ---: | ---: | --- | --- |
| CTA1 | 6 | 144 | 144 | All 0 ns | 1,416 generations; 25,488 grid-barrier joins; 7,168 ns span-bound width |
| CTA16 | 6 | 1,152 | 72 | All 0 ns | 708 generations; 12,744 grid-barrier joins; 7,168–8,192 ns span-bound widths |

The reports retain phase-zero buckets, exchange and PLE waits, dispatch/body intervals, publication brackets, per-token phase totals, overlap-aware die envelopes and generation-matched cross-die witnesses. Timer values remain integer nanoseconds in independent per-device clock domains; finite span bounds assume constant relative clock offsets over each bounded window and report their timestamp allowance separately from the exact zero arithmetic residual.

CTA1 host summaries cover six captures and 24 host-token rows; CTA16 covers six captures and 12 rows. Every row staged 1,440 PLE bytes with zero short reads. Mean PLE read/staging/callback times were 24,682/63,323/12,373 ns for CTA1 and 21,552/55,688/12,478 ns for CTA16. Raw host traces retain readiness publication, polling, readback and callbacks in the host clock domain.

The two actual GPU layout tests exercised dispatch/body boundaries, generation, peer-publication brackets, phase zero, per-token reconciliation, CTA-mode mismatch canaries and stride mismatch canaries.

## Lifecycle

The successful GPUQ lifecycle suite passed all ten real cases: single and multiple outputs, stop slots, snapshot/resume, cancellation, argument/image errors, recovery and paired prefix/resume. Eight available diagnostic traces were reconciled; argument/image errors correctly have no trace records. Those eight captures contain 174 zero-residual CTA/token partitions and 174 zero-residual die/token envelopes. The aborted callback case records `aborted=1, ended=0`; the checker accepts either documented device `ended` bit for an aborted request while validating agreement across dies and host events.

## Reproduction and regressions

The DSH CPU regression `build-3432057-1792569758014988847` passed all five trace reconciliation regressions. The final all-capture reconciliation build was `build-660395-1792586109582327767`; lifecycle reconciliation was `build-733499-1792586867479625292`.

The raw traces, layout logs, lifecycle captures, configs, source manifests, reconciliation reports and per-layout overhead summaries are retained under this directory. Current-head follow-up runs, current-head measurements and their recorded commands are in [schema2-c250e88](../schema2-c250e88/README.md).
