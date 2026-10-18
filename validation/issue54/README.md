# Issue #54: balance experiment status

This directory contains the CPU tests and a real six-die resident baseline. It does not contain a passing measured placement.

## Validation

`make -C kern die_balance_test` runs the analyzer's synthetic acceptance fixtures and the collector's fake UNIX-socket engine tests. The fake engine exercises pipelined commands/tokens, same-connection profile collection before `done`, error/EOF handling, peer-PID verification, and refusal of a busy lock. Regression tests cover Ruby 3.4 without the optional CSV gem and recognition of flashcore's mapped engine generation. These tests use no CUDA calls or model inference.

The original test transcript and source hashes are in `offline-validation.txt`; the continuation is in `offline-validation-resume.txt`. The initial busy-lock observation remains in `shared-host-preflight.txt` as historical evidence.

## Baseline measured on archy

The collector acquired `/tmp/flash-sass-engine-validation.lock` and completed a warmup plus three measured requests at **2026-10-18 05:32 UTC**. The raw capture, protocol transcript, source patch/hashes, telemetry, and measured results are in [baseline-20261018T053200Z](baseline-20261018T053200Z/README.md).

| Requirement from #54 | Current evidence |
|---|---|
| Six-die `actq256` spread at most 0.50 ms | Baseline worst measured spread **2.2927 ms**; fails |
| Mean decode at least 35.9 tok/s under the fixed linkmatrix request protocol | Baseline rates 35.77, 35.77, 35.79; mean **35.7767 tok/s**; fails |
| Identical baseline/candidate output token IDs | All four baseline requests have identical 64 input and 38 output IDs; candidate comparison outstanding |
| New shares/map and measured justification in `SPEC.md` | Baseline recorded; no winning placement selected |

Changing `SPLIT_SHARES` or `ATTN_MAP` requires compatible weight holders. The existing holders and core were not restarted or reconfigured by this work. Testing other placements remains blocked by the user's standing prohibition on restarting or reconfiguring the resident engine and holders.

See [the experiment procedure](../../docs/die-balance.md) for capture/comparison commands and the required evidence. The runtime source revision in `config` identifies the measured engine; `manifest.source_revision` identifies the collector checkout. The mapped engine library and core/frontend executables have separate SHA-256 identities. The checkpoint is identified by path and file metadata, which is explicitly weaker than a full content hash.

The PR remains draft until an actual measured candidate satisfies the issue's requirements.
