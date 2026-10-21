# Resident baseline, 2026-10-18 05:32 UTC

This is a successful capture of the existing placement on archy, **not a passing candidate**. It was collected with `/tmp/flash-sass-engine-validation.lock` held, using the resident core PID 3575268 and frontend PID 753710. The collector released the lock after its identity checks and cleanup.

## Source and command

The collector checkout was `08c741228d03addb222f290e4f233147d8f34779` with the changes in [collector.patch](collector.patch). Apply this zero-context patch to the recorded parent with `git apply --unidiff-zero`. The patch removes the optional CSV gem dependency, recognizes the engine generation that flashcore actually maps, and preserves the frontend request timeout setting. The exact source and capture hashes are in [source-and-capture.sha256](source-and-capture.sha256); they matched between archy and engi after copying the evidence. The collector's dirty source status is retained in the capture.

On archy, Ruby 3.4.10:

```bash
ruby /tmp/flash-sass-issue54-08c7412/kern/die_balance_capture.rb \
  --out /tmp/flash-sass-issue54-baseline-20261018T0528Z \
  --label baseline-20261018T0528Z \
  --core-pid 3575268 --flashd-pid 753710 --socket /tmp/flash.sock
```

Exit status: 0. The label and original output path were chosen before waiting for the lock; `capture.json.created_at` records the actual start, `2026-10-18T05:32:00.850787Z`.

The initial run from the unmodified checkout failed before inference because Ruby 3.4 does not bundle the CSV gem. Inspection then found that flashcore maps `/tmp/flashcore-3575268-gen21.so`, rather than a filename beginning with `libengine`. Both problems are covered by the added CPU regression tests.

## Measured results

One warmup preceded three measured requests using the exact linkmatrix request. Every request had 64 prompt IDs and the same 38 output IDs, including reasoning and stop tokens; the rendered answer was `Paris`.

| Run | Die 0 actq256 ms | Die 1 | Die 2 | Die 3 | Die 4 | Die 5 | Spread ms | Decode tok/s | Each die's profile span ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Warmup, excluded | 5.0524 | 3.7233 | 5.9423 | 4.9603 | 3.6250 | 3.6055 | 2.3368 | 11.72 | 135.94 |
| 1 | 5.0975 | 3.7478 | 5.8358 | 4.9807 | 3.6280 | 3.6506 | 2.2078 | 35.77 | 28.10 |
| 2 | 5.0606 | 3.7458 | 5.8880 | 4.9715 | 3.5953 | 3.6280 | 2.2927 | 35.77 | 28.09 |
| 3 | 5.0698 | 3.7089 | 5.8880 | 4.9531 | 3.6076 | 3.6004 | 2.2876 | 35.79 | 28.08 |

All four runs have phase counts `1154,1214,1154,1154,1214,1214`; every die has 146 `actq256` phases. Warmup's large `plerows` time is retained in the raw profiles and is excluded only as prescribed by the one-warmup protocol.

Worst measured spread: **2.2927 ms**, over the 0.50 ms limit. Arithmetic mean measured decode: **35.7766666667 tok/s**, below 35.9. Objective `max_run(max_die(profile_span_ms))`: **28.10 ms**. No candidate has been selected.

Token-sequence SHA-256, encoding each ID as an unsigned 32-bit little-endian integer:

- Input: `11fcb9de0469d54bdc9eb4ce430fe1fabe7d6eb4c3c30758dfa28fd74f376fb4`.
- Output: `daf096b3860066b3506a4eb7b0f7944eee95a31a95e0dd2635e85b417897452e`.

The IDs and all six raw profiles per response are in [capture.json](capture.json); [protocol.jsonl](protocol.jsonl) retains the same-connection commands, responses, and timestamps.

## Runtime

The configuration before and after capture was identical:

```text
rev=edc66d9 kv_host=0 ple_disk=1 maxt=230000 layers=48/48 nblk=512 dies=6
attn_dies=1,4,5 split=112,97,81,116,97,97
attn_map=1,1,1,1,4,4,4,4,5,5,5,5
img_rows=4096 two_position=0 prefill_chunk=0
cubin_fnv=669dbb2e43186b1a
```

Root-port links were unchanged: `00:01.0 8GT/s x8`, `00:02.0 8GT/s x16`, and `00:03.0 5GT/s x16`. Core, frontend, mapped engine, and model stat identities were unchanged. The engine generation SHA-256 is `e0ce409d7f23f8ee924cf312c77eac4fbce89bfca6d180c6328f55b12e771eb2`; executable and checkpoint identities are in the capture and [resident-processes.json](resident-processes.json).

## Reproduce the analysis

From the repository root:

```bash
ruby kern/die_balance.rb \
  --baseline validation/issue54/baseline-20261018T053200Z/capture.json
```

Exit status: 0; analysis status: `PENDING`, because only a baseline is supplied. This is not an acceptance result. [analysis.json](analysis.json) and [analysis.txt](analysis.txt) retain that output.

The next required experiment is a matched baseline/candidate comparison after loading compatible weight holders. The user's standing instruction prohibits restarting or reconfiguring the resident engine and holders; no candidate layout was loaded during this capture.
