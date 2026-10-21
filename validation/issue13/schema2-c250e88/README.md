# Issue #13 acceptance evidence on merged master

This supplement validates the current PR head, merge commit `c250e883611caec5eb85ea22114dad6068b2d85c` (master `84ceabf`). The full issue acceptance matrix, historical raw traces and three-block measurements on source commit `69b1e7c` remain in [the earlier evidence set](../schema2-69b1e7c/README.md). The current-head checks below use the same six Tesla M60 dies and 48/48 model layers.

Exact Benji build, ABI, GPUQ layout, ABBA and lifecycle commands are listed in [`commands/recorded-commands.md`](commands/recorded-commands.md). The GPUQ job records in this directory preserve the actual argv and outcomes.

[`SHA256SUMS`](SHA256SUMS) covers the raw traces, summaries, run records and source/weights manifests in this directory, excluding the checksum file itself.

## Current source and image identities

[`source-manifest.txt`](source-manifest.txt) lists the complete engine inputs, trace generator, validators and build rules. [`weights-manifest.txt`](weights-manifest.txt) records the resident model inputs. The exact current run configs and reload/restore controls are [`config-cta1.json`](config-cta1.json), [`config-cta16.json`](config-cta16.json) and [`control.json`](control.json).

| Artifact | SHA-256 |
| --- | --- |
| Current `libengine.so` built from `c250e88` | `daee76446895a816158db875b9e627995d4fc1a1b0e9a9fbb46a3eee408f096a` |
| Ordinary `die.cubin` | `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423` |
| CTA1 `die_trace.cubin` | `8ad2ae307c6802ac539de3261f6584aa838ee2f83563656efac69d336a37dd56` |
| CTA16 `die_trace16.cubin` | `de7b02550392244e30c5e9bbf21a12e866037f8ca34503dd32cef56045323b43` |
| `trace_layout_test` | `6094cf4ea07307d37913b9834ed6c0bcb7e7ff7519a9908815b78fada9455b99` |
| `trace_preflight` | `c8252c2f6f64b5390fd8dd50a4ab9680022be854191d6504620eb6604b3fdee0` |
| `source-manifest.txt` | `9274cb10043de498e9734f2ccf6eb608a0f06cffd622c84a474323e23a552d9d` |
| `weights-manifest.txt` | `f3ee5175d91b119077fd874b497f065d5541fe2a149adb47522defc850da3f3f` |

The master merge adds automatic prompt chunk selection in `engine.c`. Both current run configs explicitly set `FLASH_PREFILL_CHUNK=0`; their recorded engine config reports `prefill_auto=0`. This keeps the measured path fixed and explicit. `kern/die.pl`, `kern/trace.pl` and the trace layout test source have the same hashes as the earlier successful cubin build; the reused cubins match those recorded hashes. The current source manifest records the `DIE_TRACE` and `DIE_CTA_TRACE` inputs. The Makefile passes `DIE_CTA_TRACE=0` for both trace images. A direct generator check rejected both flags enabled together and accepted `DIE_TRACE=1/16` with `DIE_CTA_TRACE=0`.

## Issue objective coverage

| Issue #13 objective | Current evidence |
| --- | --- |
| Bounded per-token traces on all participating dies, including phase zero and request/token identity | Current CTA1 and CTA16 raw traces plus actual GPU layout tests; each trace run records all six dies. |
| Separate opcode work from exchange, PLE readiness, dispatch, barriers and publication | Schema-2 phase records, host traces and source format; current reconciliation reports retain per-token buckets and overlap. |
| Export every model phase, including indexer selection, gather, attention and remaining operations | Current per-die traces record the generated phase lists (1,178 or 1,230 phases by die); full raw rows remain in the accepted outputs. |
| Record host PLE reads, bytes, short reads, readiness, polling, readback and callback events | Current CTA1/CTA16 host summaries and the paired raw `hosttrace` files. |
| Validate single/multiple output, stop slots, snapshot/resume, cancellation, errors and recovery | Current ten-case lifecycle run, including pair prefix/resume and callback abort; two expected error cases have no trace rows. |
| Reconcile each CTA/token and die/token and reconstruct the overlap-aware cross-die path | Current DSH reports: all arithmetic residuals are 0 ns with finite causal span bounds. |
| Measure instrumentation overhead against a matched ordinary engine | Current CTA1 and CTA16 ABBA blocks and the earlier three-block matrices for each layout. |
| Preserve exact commands, source/cubin identity, raw traces and results | Source/weights manifests, GPUQ job records/logs, command record, raw outputs and checksummed reports. |

DSH build `build-832079-1792587863466072272` on Benji built the merged engine, layout test, preflight and prefill planner test. The prefill planner suite passed, as did all five CPU trace reconciliation regressions. A second DSH run, `build-800204-1792587542522324977`, also passed the five trace reconciliation regressions. The Benji image lacks the external MaxAs checkout required by the full cubin target, so that target did not regenerate cubins there; the exact source and image hashes above were checked, and both images passed the live layout tests below.

## Current-head layout checks

The actual current-head test binary passed against both Tesla M60 images:

| Layout | GPUQ job | Result |
| --- | --- | --- |
| CTA1 | `a2473749-0bb6-46a5-97a5-0adf78da52b1` | Dispatch/body boundaries, generation, peer-publication brackets, phase zero, per-token reconciliation, CTA-mode mismatch canaries and stride mismatch canaries passed. |
| CTA16 | `81c7f7e6-f01e-4174-9454-802b20881de6` | The same checks passed across all 16 CTAs. |

GPUQ's optional resident-engine trace collector reported that the uninstrumented resident engine had no trace producer. The layout executable still returned exit 0 and emitted the successful layout checks; its own output is retained in [`layout/`](layout/).

## Current-head matched overhead

Each current-head run used one four-slot ABBA block, two warmups per slot, the same prompt/settings within its layout, and identical output IDs across all 12 measured requests. The complete samples, per-request host traces and configs are in the two `overhead-*-c250e88-accepted/` directories. GPUQ job records and logs are saved alongside them.

| Layout | Ordinary steady delivery | Traced steady delivery | Steady overhead | Ordinary wall | Traced wall | Wall overhead | Output IDs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| CTA1, 3 outputs | 24.580683 ms/token | 28.793407 ms/token | 17.1384% | 128.096636 ms | 148.779424 ms | 16.1462% | `[220, 15, 13, 15]` |
| CTA16, 2 outputs | 24.697596 ms/token | 29.849323 ms/token | 20.8592% | 79.491900 ms | 94.825969 ms | 19.2901% | `[220, 15]` |

The CTA1 matrix also measured 14.6633% first-token overhead and 16.5793% from rounded engine run time. CTA16 measured 18.6228% first-token overhead and 20.0935% from rounded engine run time. These are one-block current-head spot checks. The stronger three-block matrices on the earlier engine build are retained for variance context: CTA1 steady overhead 17.0984% (range 17.0736–17.1183%); CTA16 21.3585% (range 21.2331–21.5256%). Those earlier measurements explicitly used `FLASH_PREFILL_CHUNK=0` as well.

## Current-head trace reconciliation and host events

DSH build `build-957994-1792589164408047935` reconciled all four current-head traced captures. The reports and checksums are in [`reconciliation-current/`](reconciliation-current/); no incomplete samples were skipped.

Each CTA1 report covers four token positions and each CTA16 report covers two. The current per-die phase counts are 1,178 or 1,230, according to the die's generated phase list; phase zero is included in the raw records and each CTA/token partition is checked.

| Layout | Captured requests | CTA/token partitions per capture | Die/token envelopes per capture | Residuals | Causal witness |
| --- | ---: | ---: | ---: | --- | --- |
| CTA1 | 2 | 24 | 24 | All arithmetic residuals 0 ns | 1,416 generations and 25,488 grid joins per capture; 7,168 ns finite span-bound width |
| CTA16 | 2 | 192 | 12 | All arithmetic residuals 0 ns | 708 generations and 12,744 grid joins per capture; 7,168–8,192 ns finite span-bound width |

The current host summaries are [`cta1-host-current-summary.json`](cta1-host-current-summary.json) and [`cta16-host-current-summary.json`](cta16-host-current-summary.json). CTA1 has 8 host-token rows and CTA16 has 4; every row read 1,440 PLE bytes with zero short reads. Mean read/staging/readiness-bracket/callback times were 24,527/62,490/82/9,371 ns for CTA1 and 22,910/56,579/87/17,613 ns for CTA16. Raw host captures retain readiness publication, polling, readback and callback events in the host clock domain.

## Lifecycle

Current-head GPUQ job `a87330b2-d055-42dc-8161-b29cbb42b0fb` passed all ten real lifecycle cases with `source_sha=c250e88`; the pair check was requested and executed, and no device fault was injected. [`lifecycle-c250e88-final/`](lifecycle-c250e88-final/) retains the raw six-die and host traces, protocol and state hashes. The controller's final restore record confirms the original `attention-q-release` library, cubin and runtime settings.

The cases cover single and normal output, single and multiple stop slots, disconnect cancellation, argument/image errors, recovery and paired prefix/resume. Eight cases produced traces; argument/image errors correctly produced none. DSH build `build-1092297-1792590467137970187` reconciled those eight traces: 174 CTA/token partitions and 174 die/token envelopes, all with 0 ns residual. The skipped cases and reasons are listed in [`reconciliation-lifecycle-current/index.json`](reconciliation-lifecycle-current/index.json).

The disconnect case is request 5, aborted at position 2 with `abort_reason=callback`; its validated prefix is 11 of 42 positions and its documented `ended=0` bit is accepted. The host trace records callback return at 152,372,282,743,901 ns and abort entry at 152,372,282,744,023 ns, a 122 ns ordering gap, followed by a 114,666 ns abort interval. The callback, PLE read/staging, readiness, polling and readback events remain in their host clock domain.

## Historical full matrix

The earlier [schema-2 evidence set](../schema2-69b1e7c/README.md) contains six full ABBA blocks, the ten-case real lifecycle suite, layout logs, source and cubin identities, raw traces and host events. Its CTA1 captures have 144 zero-residual CTA/token partitions and 144 zero-residual die/token envelopes; CTA16 has 1,152 and 72 respectively. That dataset remains separately labeled with its measured engine/library identity. The new current-head spot checks above verify the merged engine source and current master behavior.

## GPUQ collector retry

The first current-head CTA1 submission failed before starting (`started_ns=null`, no GPU work) because GPUQ's provenance write hit `SQLite3::BusyException`. The reproduction is filed as [GPUQ #66](https://github.com/nm-z/gpuq/issues/66), attached under existing [GPUQ #54](https://github.com/nm-z/gpuq/issues/54). The later accepted CTA1 run used the same queue after it cleared and completed successfully. A control-file allowlist correction was also required for the new `c250e88` engine revision; the accepted overhead and lifecycle runs and both layout checks record or verify restoration to the original `attention-q-release` configuration.

The intervening CTA1 control attempt completed only ordinary warmups and one ordinary sample, then refused the traced reload because `c250e88` was missing from `allowed_revisions`. Its output is preserved under [`operational-diagnostics/`](operational-diagnostics/). The explicit restore helper initially refused for the same allowlist reason; after adding the known current revision, GPUQ job `ac408909-6957-4918-a3bb-0ae270905664` restored the original library, cubin and `attention-q-release` config and printed its `PASS` confirmation. None of the partial ordinary measurements is included in the accepted ABBA summaries.
