# Issue #125: Nushell runner port and GPUQ parity

PR #43 introduced `kern/run_qsa_gather_pair.pl` in merge `316117c5`. This PR removes that file, replaces it with [`kern/run_qsa_gather_pair.nu`](../../kern/run_qsa_gather_pair.nu) plus the captured-ID helper [`kern/write_qsa_gather_ids.nu`](../../kern/write_qsa_gather_ids.nu), and updates the issue-16 matrix launcher and instructions to use Nu. The pre-port runner hash was `ab702baed911e998143d9815efec3fe63ab6f29a5e12712ab40b156d7b9e9c3c`; the final Nu runner and helper hashes are `188cbd850a025d8a318d714a737da9c5b48975abcaeb008085699855b95822fb` and `62e59c0dc40f4852b50df36517d3f89b0716f8d826997ee2258d7a5760661777`.

The Nu runner keeps the prior CLI, default and matrix cases, input/correctness checks, report fields, and captured-fixture algorithm. Its captured IDs are still generated with `lcg32-1664525-1013904223-partial-fisher-yates-v1` and written as sorted little-endian uint32 values. A DSH matrix self-test on Benji exercised all 20 cases for `nv=128`, tails 0–3, and all five layouts; all fake-benchmark arms passed the pair checks, and all four captured files had the expected 512-byte size. That self-test is orchestration coverage, not GPU evidence. The DSH build ID was `build-153047-1792580804452616034`.

The updated C matrix launcher compiled successfully through DSH on Benji (`build-161416-1792580847786577754`). It now invokes `~/.local/bin/nu` (overridable with `NU_BIN`) and the staged Nu runner/helper.

## GPUQ before/after pair

The baseline used the merged-master Perl runner; the post-port run used the final Nu runner. Both runs used seed `4242`, four measured iterations per arm, the same QSA benchmark binary and cubin, the same visible M60, and the default `nv=513`, `tail=1`, `changing` case. The baseline job `7f781111-6a43-42f4-8a63-0c0a4ba7143f` and final Nu job `fb84f561-1c92-4a13-814b-9e6129c48434` both exited 0. An earlier failed post-port attempt used the same runner before a captured-helper timeout-only increase; it is not included below.

Both GPUQ runs used `qsa_gather_bench` SHA-256 `246caa8f78d65579b7d6cdc8005a69357972b5b911702d5a67f0243df03af892` and `die.cubin` SHA-256 `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`. The binary/cubin came from the prior DSH build `build-3919491-1792578277376915060` (source `a81cef3`). The QSA benchmark sources listed in the manifests are identical between that source and merged master `316117c`; the comparison therefore isolates the runner port. A fresh current-master CUDA rebuild was attempted through DSH but could not produce these artifacts because Benji lacks `/opt/cuda-11.4/bin/ptxas`.

The runs agreed on both placements for all stable benchmark fields: source revision `a81cef3`, cache hash `e3a8f87d8c355d72`, selection-table hash `82f446fa0725f0e9`, three selection changes, three verified-output-signature changes, and `preflight_verified_all=true`. Each arm reported position 2052, tail residue 0, budget 512, 2,049 logical keys, 2,052 physical rows, 2,232,576 source unique-address bytes, and 8,404,992 destination unique-address bytes. The four per-iteration selection and gathered-output signatures were identical across both placements and both runner versions:

| Iteration | Selection hash | Verified output hash |
| ---: | --- | --- |
| 0 | `963cec86ff4615fe` | `9bc0d7e6086b5a76` |
| 1 | `f1a76199f5a84e25` | `aec30a0cef12e8e4` |
| 2 | `c3eb74ae9bf43109` | `7f20fd2c38e106e7` |
| 3 | `ba39034e9a4e856e` | `e0e1b518ee477c12` |

As expected, timing samples differed: device-VRAM p50 was 203.200 µs before and 200.064 µs after; pinned-host p50 was 3,517.440 µs before and 3,520.384 µs after. The parity check compares the deterministic input, selection, count, placement, and output-signature fields; it does not require timing distributions to be identical.

## Raw outputs

The before and after directories contain each report, provenance manifest, two raw CSVs, two per-arm logs, and GPU inventory:

- [Before: report](before/report.json), [host 0 CSV](before/nv513-tail1-changing-host0.csv), [host 1 CSV](before/nv513-tail1-changing-host1.csv), and [provenance](before/provenance.json).
- [After: report](after/report.json), [host 0 CSV](after/nv513-tail1-changing-host0.csv), [host 1 CSV](after/nv513-tail1-changing-host1.csv), and [provenance](after/provenance.json).

The pre-port report also preserves the original source/binary/cubin hashes. The final Nu report includes the captured-ID generator helper hash. [The machine-readable parity check](parity.json) records exact equality of the stable report fields and all four rows' selection/output signatures; only the measured timing values differ.
