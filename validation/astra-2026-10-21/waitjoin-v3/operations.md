# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `d0b1773c-1e9d-4680-b88d-bef51d2b1ca5`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3959.1 | +3959.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3009.2 | +3009.2 |
| `router_full` | 0 | 0.0 | 48 | 2573.0 | +2573.0 |
| `q2moegn` | 96 | 1447.8 | 48 | 1877.3 | +429.5 |
| `q3kk2560` | 53–56 | 1685.4 | 23–24 | 1673.1 | -12.3 |
| `bf16hcrwx` | 48 | 1587.5 | 48 | 1591.1 | +3.6 |
| `q2moedn` | 48 | 856.7 | 48 | 1470.1 | +613.4 |
| `q4kk6144` | 15 | 405.9 | 15 | 1272.0 | +866.1 |
| `iq4k2560` | 22–24 | 868.4 | 16–18 | 1265.7 | +397.3 |
| `gdn_full` | 0 | 0.0 | 36 | 1164.4 | +1164.4 |
| `q6kk1536` | 6 | 102.7 | 6 | 1001.1 | +898.4 |
| `q5kk6144` | 7 | 269.3 | 7 | 978.6 | +709.3 |
| `iq4k6144` | 17 | 471.7 | 17 | 807.9 | +336.2 |
| `q5kk2560` | 1–4 | 620.4 | 1–3 | 618.0 | -2.4 |
| `q4kk2560` | 8–12 | 279.1 | 4–6 | 338.1 | +59.0 |
| `xsend` | 123–124 | 1549.8 | 10 | 215.4 | -1334.4 |
| `q2k2560` | 12–14 | 214.2 | 3–4 | 193.0 | -21.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 189.4 | +189.4 |
| `q3kk6144` | 3 | 99.5 | 3 | 157.7 | +58.2 |
| `attnqkq` | 0–4 | 84.8 | 2 | 143.5 | +58.7 |
| `amax` | 1 | 94.9 | 1 | 95.7 | +0.9 |
| `bf16hcrx` | 49 | 1178.1 | 2 | 74.4 | -1103.7 |
| `hcwsum` | 48 | 2368.0 | 1 | 73.9 | -2294.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 | +66.6 |
| `kvgather` | 0–4 | 42.0 | 2 | 41.1 | -0.9 |
| `rms2560` | 3 | 23.2 | 1 | 31.6 | +8.3 |
| `bf16k2560` | 49–53 | 1382.9 | 1 | 16.6 | -1366.4 |
| `q6kk2560` | 0–4 | 26.6 | 0–1 | 14.5 | -12.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.8 | +12.8 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 | +12.3 |
| `argmax` | 1 | 10.9 | 1 | 12.1 | +1.2 |
| `pleconv` | 1 | 10.0 | 1 | 10.8 | +0.7 |
| `actq256` | 146 | 2780.9 | 0 | 0.0 | -2780.9 |
| `attnmerge` | 0–4 | 16.2 | 0 | 0.0 | -16.2 |
| `attnpartn` | 0–4 | 127.7 | 0 | 0.0 | -127.7 |
| `b32iq4nlacc` | 0–2 | 26.1 | 0 | 0.0 | -26.1 |
| `b32q40acc` | 2–4 | 49.0 | 0 | 0.0 | -49.0 |
| `b32q50acc` | 0–2 | 29.7 | 0 | 0.0 | -29.7 |
| `b32q80acc` | 0–1 | 8.2 | 0 | 0.0 | -8.2 |
| `bf16bias` | 36 | 230.4 | 0 | 0.0 | -230.4 |
| `bf16hcup` | 97 | 2376.0 | 0 | 0.0 | -2376.0 |
| `conv1d` | 36 | 219.3 | 0 | 0.0 | -219.3 |
| `dnnormx` | 36 | 502.8 | 0 | 0.0 | -502.8 |
| `dnstep` | 36 | 617.8 | 0 | 0.0 | -617.8 |
| `ffnact` | 48 | 644.4 | 0 | 0.0 | -644.4 |
| `idxqk` | 0–4 | 20.8 | 0 | 0.0 | -20.8 |
| `idxscore` | 0–4 | 5.1 | 0 | 0.0 | -5.1 |
| `idxsel` | 0–20 | 27.5 | 0 | 0.0 | -27.5 |
| `moesel` | 48 | 215.7 | 0 | 0.0 | -215.7 |
| `plegate` | 1 | 7.9 | 0 | 0.0 | -7.9 |
| `plerows` | 1 | 55.1 | 0 | 0.0 | -55.1 |
| `q2acc640` | 1–4 | 40.1 | 0 | 0.0 | -40.1 |
| `q6kk1536acc` | 18 | 218.1 | 0 | 0.0 | -218.1 |
| `router` | 48 | 1084.1 | 0 | 0.0 | -1084.1 |
| `xcopy` | 0–1 | 27.0 | 0 | 0.0 | -27.0 |
| **All recorded intervals** | 1,177–1,229 | **25009.7** | 498 | **24960.2** | **-49.6** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3546.1 | 4107.4 | +561.3 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6695.8 | 4429.7 | -2266.2 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.6 | 12.3 | -20.3 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 31.1 | 44.4 | +13.3 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 320.8 | 1001.1 | +680.3 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3977.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 2989.1 |
| `router_full` | 0 | 0.0 | 48 | 2527.2 |
| `q2moegn` | 96 | 1517.6 | 48 | 1961.0 |
| `q3kk2560` | 54 | 1683.5 | 24 | 1680.4 |
| `bf16hcrwx` | 48 | 1535.0 | 48 | 1557.5 |
| `q2moedn` | 48 | 895.0 | 48 | 1499.1 |
| `q4kk6144` | 15 | 425.0 | 15 | 1188.9 |
| `iq4k2560` | 24 | 903.2 | 18 | 1308.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1223.7 |
| `q6kk1536` | 6 | 105.5 | 6 | 926.7 |
| `q5kk6144` | 7 | 279.6 | 7 | 1099.8 |
| `iq4k6144` | 17 | 493.6 | 17 | 829.4 |
| `q5kk2560` | 2 | 688.1 | 1 | 660.5 |
| `q4kk2560` | 10 | 277.5 | 4 | 312.3 |
| `xsend` | 124 | 1629.2 | 10 | 208.9 |
| `q2k2560` | 12 | 210.9 | 3 | 159.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 198.7 |
| `q3kk6144` | 3 | 104.4 | 3 | 159.7 |
| `attnqkq` | 0 | 0.0 | 2 | 181.2 |
| `amax` | 1 | 22.5 | 1 | 26.6 |
| `bf16hcrx` | 49 | 1216.5 | 2 | 49.2 |
| `hcwsum` | 48 | 2242.6 | 1 | 6.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 42.0 |
| `rms2560` | 3 | 23.6 | 1 | 31.7 |
| `bf16k2560` | 49 | 1317.9 | 1 | 15.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 11.3 |
| `actq256` | 146 | 3063.8 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 35.8 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 238.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2322.4 | 0 | 0.0 |
| `conv1d` | 36 | 229.4 | 0 | 0.0 |
| `dnnormx` | 36 | 566.3 | 0 | 0.0 |
| `dnstep` | 36 | 615.4 | 0 | 0.0 |
| `ffnact` | 48 | 646.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 59.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 222.2 | 0 | 0.0 |
| `router` | 48 | 1090.6 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3890.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 3025.9 |
| `router_full` | 0 | 0.0 | 48 | 2568.2 |
| `q2moegn` | 96 | 1454.1 | 48 | 1873.9 |
| `q3kk2560` | 53 | 1692.7 | 24 | 1693.7 |
| `bf16hcrwx` | 48 | 1536.0 | 48 | 1531.9 |
| `q2moedn` | 48 | 867.3 | 48 | 1484.8 |
| `q4kk6144` | 15 | 412.7 | 15 | 1187.8 |
| `iq4k2560` | 24 | 865.3 | 16 | 1198.1 |
| `gdn_full` | 0 | 0.0 | 36 | 1221.6 |
| `q6kk1536` | 6 | 103.4 | 6 | 922.6 |
| `q5kk6144` | 7 | 271.4 | 7 | 1094.7 |
| `iq4k6144` | 17 | 477.2 | 17 | 814.1 |
| `q5kk2560` | 4 | 630.8 | 1 | 575.5 |
| `q4kk2560` | 11 | 302.1 | 6 | 461.8 |
| `xsend` | 123 | 1546.2 | 10 | 199.7 |
| `q2k2560` | 14 | 192.5 | 3 | 166.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 201.7 |
| `q3kk6144` | 3 | 101.4 | 3 | 159.7 |
| `attnqkq` | 4 | 290.8 | 2 | 177.2 |
| `amax` | 1 | 114.7 | 1 | 112.6 |
| `bf16hcrx` | 49 | 1176.6 | 2 | 78.8 |
| `hcwsum` | 48 | 2336.8 | 1 | 128.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `kvgather` | 4 | 85.0 | 2 | 42.0 |
| `rms2560` | 3 | 22.5 | 1 | 25.6 |
| `bf16k2560` | 53 | 1449.0 | 1 | 16.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2090.0 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 37.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 243.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2496.5 | 0 | 0.0 |
| `conv1d` | 36 | 226.3 | 0 | 0.0 |
| `dnnormx` | 36 | 557.1 | 0 | 0.0 |
| `dnstep` | 36 | 614.4 | 0 | 0.0 |
| `ffnact` | 48 | 647.2 | 0 | 0.0 |
| `idxqk` | 4 | 64.5 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 219.1 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 60.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1091.6 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4172.8 |
| `hcupq_x` | 0 | 0.0 | 97 | 3131.4 |
| `router_full` | 0 | 0.0 | 48 | 2589.7 |
| `q2moegn` | 96 | 1244.2 | 48 | 1665.0 |
| `q3kk2560` | 56 | 1632.3 | 23 | 1585.2 |
| `bf16hcrwx` | 48 | 1818.6 | 48 | 1655.8 |
| `q2moedn` | 48 | 744.4 | 48 | 1384.4 |
| `q4kk6144` | 15 | 357.4 | 15 | 1423.4 |
| `iq4k2560` | 23 | 825.3 | 18 | 1267.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1054.7 |
| `q6kk1536` | 6 | 95.2 | 6 | 940.0 |
| `q5kk6144` | 7 | 237.6 | 7 | 917.5 |
| `iq4k6144` | 17 | 422.9 | 17 | 872.4 |
| `q5kk2560` | 1 | 485.4 | 1 | 484.4 |
| `q4kk2560` | 8 | 238.6 | 4 | 291.8 |
| `xsend` | 123 | 1439.7 | 10 | 233.5 |
| `q2k2560` | 14 | 224.3 | 4 | 205.8 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 186.4 |
| `q3kk6144` | 3 | 89.1 | 3 | 162.8 |
| `attnqkq` | 0 | 0.0 | 2 | 179.2 |
| `amax` | 1 | 200.7 | 1 | 203.8 |
| `bf16hcrx` | 49 | 1122.3 | 2 | 77.8 |
| `hcwsum` | 48 | 2721.8 | 1 | 73.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 67.6 |
| `kvgather` | 0 | 0.0 | 2 | 39.9 |
| `rms2560` | 3 | 23.6 | 1 | 31.7 |
| `bf16k2560` | 49 | 1312.8 | 1 | 18.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3898.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 24.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 209.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2124.8 | 0 | 0.0 |
| `conv1d` | 36 | 201.7 | 0 | 0.0 |
| `dnnormx` | 36 | 311.3 | 0 | 0.0 |
| `dnstep` | 36 | 596.0 | 0 | 0.0 |
| `ffnact` | 48 | 636.9 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 216.1 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 1 | 15.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 208.9 | 0 | 0.0 |
| `router` | 48 | 1082.4 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3836.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 2925.6 |
| `router_full` | 0 | 0.0 | 48 | 2564.1 |
| `q2moegn` | 96 | 1531.9 | 48 | 1957.9 |
| `q3kk2560` | 56 | 1827.8 | 23 | 1731.6 |
| `bf16hcrwx` | 48 | 1548.3 | 48 | 1617.9 |
| `q2moedn` | 48 | 904.2 | 48 | 1565.7 |
| `q4kk6144` | 15 | 421.9 | 15 | 1364.0 |
| `iq4k2560` | 23 | 932.9 | 16 | 1254.4 |
| `gdn_full` | 0 | 0.0 | 36 | 1093.6 |
| `q6kk1536` | 6 | 105.5 | 6 | 1134.6 |
| `q5kk6144` | 7 | 279.6 | 7 | 696.3 |
| `iq4k6144` | 17 | 489.5 | 17 | 755.7 |
| `q5kk2560` | 1 | 679.9 | 3 | 835.6 |
| `q4kk2560` | 8 | 281.6 | 4 | 332.8 |
| `xsend` | 123 | 1569.8 | 10 | 210.9 |
| `q2k2560` | 14 | 230.4 | 4 | 217.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q3kk6144` | 3 | 102.4 | 3 | 147.5 |
| `attnqkq` | 0 | 0.0 | 2 | 180.2 |
| `amax` | 1 | 5.1 | 1 | 8.2 |
| `bf16hcrx` | 49 | 1217.5 | 2 | 80.9 |
| `hcwsum` | 48 | 2282.5 | 1 | 74.8 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 38.9 |
| `rms2560` | 3 | 24.6 | 1 | 29.7 |
| `bf16k2560` | 49 | 1312.8 | 1 | 14.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 12.3 | 1 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2967.6 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 216.1 | 0 | 0.0 |
| `bf16hcup` | 97 | 2347.0 | 0 | 0.0 |
| `conv1d` | 36 | 208.9 | 0 | 0.0 |
| `dnnormx` | 36 | 446.5 | 0 | 0.0 |
| `dnstep` | 36 | 639.0 | 0 | 0.0 |
| `ffnact` | 48 | 649.2 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 213.0 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 54.3 | 0 | 0.0 |
| `q2acc640` | 3 | 44.0 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 227.3 | 0 | 0.0 |
| `router` | 48 | 1077.2 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4075.5 |
| `hcupq_x` | 0 | 0.0 | 97 | 3012.6 |
| `router_full` | 0 | 0.0 | 48 | 2595.8 |
| `q2moegn` | 96 | 1401.9 | 48 | 1826.8 |
| `q3kk2560` | 53 | 1615.9 | 24 | 1714.2 |
| `bf16hcrwx` | 48 | 1519.6 | 48 | 1583.1 |
| `q2moedn` | 48 | 823.3 | 48 | 1379.3 |
| `q4kk6144` | 15 | 409.6 | 15 | 1210.4 |
| `iq4k2560` | 24 | 866.3 | 17 | 1290.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1187.8 |
| `q6kk1536` | 6 | 104.4 | 6 | 956.4 |
| `q5kk6144` | 7 | 270.3 | 7 | 1094.7 |
| `iq4k6144` | 17 | 477.2 | 17 | 790.5 |
| `q5kk2560` | 2 | 593.9 | 1 | 575.5 |
| `q4kk2560` | 12 | 310.3 | 4 | 315.4 |
| `xsend` | 123 | 1586.2 | 10 | 219.1 |
| `q2k2560` | 13 | 212.0 | 4 | 208.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q3kk6144` | 3 | 99.3 | 3 | 158.7 |
| `attnqkq` | 4 | 197.6 | 2 | 121.9 |
| `amax` | 1 | 114.7 | 1 | 112.6 |
| `bf16hcrx` | 49 | 1162.2 | 2 | 79.9 |
| `hcwsum` | 48 | 2432.0 | 1 | 67.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 82.9 | 2 | 42.0 |
| `rms2560` | 3 | 22.5 | 1 | 31.7 |
| `bf16k2560` | 53 | 1447.9 | 1 | 17.4 |
| `q6kk2560` | 2 | 50.2 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 2239.5 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 44.0 | 0 | 0.0 |
| `b32q40acc` | 4 | 72.7 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 234.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2501.6 | 0 | 0.0 |
| `conv1d` | 36 | 224.3 | 0 | 0.0 |
| `dnnormx` | 36 | 569.3 | 0 | 0.0 |
| `dnstep` | 36 | 618.5 | 0 | 0.0 |
| `ffnact` | 48 | 645.1 | 0 | 0.0 |
| `idxqk` | 4 | 44.0 | 0 | 0.0 |
| `idxscore` | 4 | 9.2 | 0 | 0.0 |
| `idxsel` | 20 | 55.3 | 0 | 0.0 |
| `moesel` | 48 | 212.0 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1087.5 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3802.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 2970.6 |
| `router_full` | 0 | 0.0 | 48 | 2592.8 |
| `q2moegn` | 96 | 1537.0 | 48 | 1979.4 |
| `q3kk2560` | 54 | 1659.9 | 23 | 1633.3 |
| `bf16hcrwx` | 48 | 1567.7 | 48 | 1600.5 |
| `q2moedn` | 48 | 906.2 | 48 | 1507.3 |
| `q4kk6144` | 15 | 408.6 | 15 | 1257.5 |
| `iq4k2560` | 22 | 817.2 | 17 | 1274.9 |
| `gdn_full` | 0 | 0.0 | 36 | 1205.2 |
| `q6kk1536` | 6 | 102.4 | 6 | 1126.4 |
| `q5kk6144` | 7 | 277.5 | 7 | 968.7 |
| `iq4k6144` | 17 | 470.0 | 17 | 785.4 |
| `q5kk2560` | 4 | 644.1 | 1 | 576.5 |
| `q4kk2560` | 9 | 264.2 | 4 | 314.4 |
| `xsend` | 123 | 1527.8 | 10 | 220.2 |
| `q2k2560` | 13 | 215.0 | 4 | 199.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 185.3 |
| `q3kk6144` | 3 | 100.4 | 3 | 157.7 |
| `attnqkq` | 4 | 20.5 | 2 | 21.5 |
| `amax` | 1 | 111.6 | 1 | 110.6 |
| `bf16hcrx` | 49 | 1173.5 | 2 | 79.9 |
| `hcwsum` | 48 | 2192.4 | 1 | 93.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 84.0 | 2 | 42.0 |
| `rms2560` | 3 | 22.5 | 1 | 38.9 |
| `bf16k2560` | 53 | 1457.2 | 1 | 17.4 |
| `q6kk2560` | 4 | 109.6 | 1 | 87.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 2425.9 | 0 | 0.0 |
| `attnmerge` | 4 | 31.7 | 0 | 0.0 |
| `attnpartn` | 4 | 256.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 239.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2463.7 | 0 | 0.0 |
| `conv1d` | 36 | 225.3 | 0 | 0.0 |
| `dnnormx` | 36 | 566.3 | 0 | 0.0 |
| `dnstep` | 36 | 623.6 | 0 | 0.0 |
| `ffnact` | 48 | 642.0 | 0 | 0.0 |
| `idxqk` | 4 | 16.4 | 0 | 0.0 |
| `idxscore` | 4 | 10.2 | 0 | 0.0 |
| `idxsel` | 20 | 56.3 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 216.1 | 0 | 0.0 |
| `router` | 48 | 1075.2 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

