# Issue #54: balance experiment status

This directory contains the CPU tests and the recorded host preflight for the six-die balance tooling. It does not contain a passing measured placement.

## Validation

`make -C kern die_balance_test` runs the analyzer's synthetic acceptance fixtures and the collector's fake UNIX-socket engine tests. The fake engine exercises pipelined commands/tokens, same-connection profile collection before `done`, error/EOF handling, peer-PID verification, and refusal of a busy lock. These tests use no CUDA calls or model inference. The fixture numbers are synthetic and must not be cited as hardware performance.

The final test transcript and source hashes are recorded in `offline-validation.txt`. `shared-host-preflight.txt` records the attempted nonblocking lock acquisition on archy, which returned `LOCK_BUSY` and exit code 75. PID `4010221`, running `validation/issue19/run.rb --mode model`, owned the lock at that check.

## Outstanding acceptance

| Requirement from #54 | Status |
|---|---|
| Six-die `actq256` spread at most 0.50 ms | No new hardware measurement |
| Mean decode at least 35.9 tok/s under the fixed linkmatrix request protocol | No new hardware measurement |
| Identical baseline/candidate output token IDs | No candidate capture |
| New shares/map and the measured justification in `SPEC.md` | No winning placement selected |

Changing `SPLIT_SHARES` or `ATTN_MAP` requires compatible weight holders. The existing holders and core were not restarted or reconfigured. A shared-host maintenance window is required to load and compare different placements under the user's existing restrictions.

See [the experiment procedure](../../docs/die-balance.md) for capture/comparison commands and the required evidence. The runtime source revision in `config` identifies the measured engine; `manifest.source_revision` identifies the collector checkout. The mapped engine library and the core/frontend executables have separate SHA-256 identities. The checkpoint is identified by path and file metadata, which is explicitly weaker than a full content hash.

The PR remains draft until an actual measured candidate satisfies the issue's requirements.
