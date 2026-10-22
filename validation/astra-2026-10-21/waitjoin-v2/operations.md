# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `9c00680c-b88e-4583-baf8-4d0127c6c379`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5656.1 | +5656.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3276.3 | +3276.3 |
| `router_full` | 0 | 0.0 | 48 | 2575.4 | +2575.4 |
| `q2moegn` | 96 | 1450.3 | 48 | 1880.1 | +429.7 |
| `q3kk2560` | 53–56 | 1689.9 | 23–24 | 1696.6 | +6.7 |
| `bf16hcrwx` | 48 | 1587.7 | 48 | 1585.8 | -1.9 |
| `q2moedn` | 48 | 855.5 | 48 | 1464.2 | +608.6 |
| `iq4k2560` | 22–24 | 863.6 | 16–18 | 1276.2 | +412.7 |
| `q4kk6144` | 15 | 405.2 | 15 | 1273.3 | +868.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1185.3 | +1185.3 |
| `q6kk1536` | 6 | 102.9 | 6 | 994.5 | +891.6 |
| `q5kk6144` | 7 | 268.1 | 7 | 979.6 | +711.5 |
| `iq4k6144` | 17 | 473.7 | 17 | 807.9 | +334.2 |
| `q5kk2560` | 1–4 | 618.8 | 1–3 | 618.8 | +0.0 |
| `q4kk2560` | 8–12 | 276.7 | 4–6 | 343.9 | +67.3 |
| `xsend` | 123–124 | 1558.0 | 10 | 212.8 | -1345.2 |
| `q2k2560` | 12–14 | 212.7 | 3–4 | 192.2 | -20.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 190.0 | +190.0 |
| `q3kk6144` | 3 | 99.0 | 3 | 149.3 | +50.3 |
| `attnqkq` | 0–4 | 80.5 | 2 | 143.5 | +63.0 |
| `amax` | 1 | 97.6 | 1 | 94.9 | -2.7 |
| `bf16hcrx` | 49 | 1179.7 | 2 | 73.2 | -1106.5 |
| `hcwsum` | 48 | 2383.7 | 1 | 71.0 | -2312.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 67.1 | +67.1 |
| `kvgather` | 0–4 | 41.5 | 2 | 41.6 | +0.2 |
| `rms2560` | 3 | 23.0 | 1 | 34.8 | +11.8 |
| `bf16k2560` | 49–53 | 1381.2 | 1 | 16.2 | -1365.0 |
| `q6kk2560` | 0–4 | 27.0 | 0–1 | 14.5 | -12.5 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.8 | +12.8 |
| `argmax` | 1 | 10.8 | 1 | 12.5 | +1.7 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.5 | +12.5 |
| `pleconv` | 1 | 10.0 | 1 | 10.9 | +0.9 |
| `actq256` | 146 | 2785.8 | 0 | 0.0 | -2785.8 |
| `attnmerge` | 0–4 | 16.2 | 0 | 0.0 | -16.2 |
| `attnpartn` | 0–4 | 127.8 | 0 | 0.0 | -127.8 |
| `b32iq4nlacc` | 0–2 | 25.9 | 0 | 0.0 | -25.9 |
| `b32q40acc` | 2–4 | 48.8 | 0 | 0.0 | -48.8 |
| `b32q50acc` | 0–2 | 29.7 | 0 | 0.0 | -29.7 |
| `b32q80acc` | 0–1 | 8.2 | 0 | 0.0 | -8.2 |
| `bf16bias` | 36 | 229.4 | 0 | 0.0 | -229.4 |
| `bf16hcup` | 97 | 2371.2 | 0 | 0.0 | -2371.2 |
| `conv1d` | 36 | 218.6 | 0 | 0.0 | -218.6 |
| `dnnormx` | 36 | 495.1 | 0 | 0.0 | -495.1 |
| `dnstep` | 36 | 621.5 | 0 | 0.0 | -621.5 |
| `ffnact` | 48 | 644.4 | 0 | 0.0 | -644.4 |
| `idxqk` | 0–4 | 24.6 | 0 | 0.0 | -24.6 |
| `idxscore` | 0–4 | 6.0 | 0 | 0.0 | -6.0 |
| `idxsel` | 0–20 | 26.8 | 0 | 0.0 | -26.8 |
| `moesel` | 48 | 216.2 | 0 | 0.0 | -216.2 |
| `plegate` | 1 | 8.5 | 0 | 0.0 | -8.5 |
| `plerows` | 1 | 56.0 | 0 | 0.0 | -56.0 |
| `q2acc640` | 1–4 | 40.3 | 0 | 0.0 | -40.3 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 | -217.1 |
| `router` | 48 | 1081.5 | 0 | 0.0 | -1081.5 |
| `xcopy` | 0–1 | 26.6 | 0 | 0.0 | -26.6 |
| **All recorded intervals** | 1,177–1,229 | **25023.4** | 498 | **26963.9** | **+1940.4** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3563.3 | 5800.2 | +2236.9 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6692.3 | 4718.6 | -1973.7 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.8 | 12.5 | -20.3 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 31.6 | 47.6 | +16.0 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 320.0 | 994.5 | +674.5 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5684.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 3231.7 |
| `router_full` | 0 | 0.0 | 48 | 2533.4 |
| `q2moegn` | 96 | 1517.6 | 48 | 1969.2 |
| `q3kk2560` | 54 | 1694.7 | 24 | 1719.3 |
| `bf16hcrwx` | 48 | 1515.5 | 48 | 1540.1 |
| `q2moedn` | 48 | 902.1 | 48 | 1505.3 |
| `iq4k2560` | 24 | 897.0 | 18 | 1316.9 |
| `q4kk6144` | 15 | 419.8 | 15 | 1187.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1250.3 |
| `q6kk1536` | 6 | 108.5 | 6 | 923.6 |
| `q5kk6144` | 7 | 277.5 | 7 | 1102.8 |
| `iq4k6144` | 17 | 496.6 | 17 | 807.9 |
| `q5kk2560` | 2 | 679.9 | 1 | 662.5 |
| `q4kk2560` | 10 | 278.5 | 4 | 314.4 |
| `xsend` | 124 | 1646.6 | 10 | 210.9 |
| `q2k2560` | 12 | 207.9 | 3 | 158.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 200.7 |
| `q3kk6144` | 3 | 104.4 | 3 | 149.5 |
| `attnqkq` | 0 | 0.0 | 2 | 182.3 |
| `amax` | 1 | 29.7 | 1 | 24.6 |
| `bf16hcrx` | 49 | 1218.6 | 2 | 48.1 |
| `hcwsum` | 48 | 2244.6 | 1 | 5.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 43.0 |
| `rms2560` | 3 | 23.6 | 1 | 35.8 |
| `bf16k2560` | 49 | 1329.2 | 1 | 17.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 11.3 |
| `argmax` | 1 | 11.3 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3070.0 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 236.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2319.4 | 0 | 0.0 |
| `conv1d` | 36 | 224.3 | 0 | 0.0 |
| `dnnormx` | 36 | 555.0 | 0 | 0.0 |
| `dnstep` | 36 | 618.5 | 0 | 0.0 |
| `ffnact` | 48 | 651.3 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 218.1 | 0 | 0.0 |
| `plegate` | 1 | 5.1 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 4 | 60.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 223.2 | 0 | 0.0 |
| `router` | 48 | 1084.4 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5603.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3275.8 |
| `router_full` | 0 | 0.0 | 48 | 2549.8 |
| `q2moegn` | 96 | 1457.2 | 48 | 1891.3 |
| `q3kk2560` | 53 | 1681.4 | 24 | 1718.3 |
| `bf16hcrwx` | 48 | 1540.1 | 48 | 1542.1 |
| `q2moedn` | 48 | 860.2 | 48 | 1471.5 |
| `iq4k2560` | 24 | 864.3 | 16 | 1217.5 |
| `q4kk6144` | 15 | 412.7 | 15 | 1184.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1248.3 |
| `q6kk1536` | 6 | 102.4 | 6 | 915.5 |
| `q5kk6144` | 7 | 270.3 | 7 | 1094.7 |
| `iq4k6144` | 17 | 479.2 | 17 | 806.9 |
| `q5kk2560` | 4 | 628.7 | 1 | 581.6 |
| `q4kk2560` | 11 | 302.1 | 6 | 468.0 |
| `xsend` | 123 | 1559.6 | 10 | 197.6 |
| `q2k2560` | 14 | 187.4 | 3 | 166.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 200.7 |
| `q3kk6144` | 3 | 101.4 | 3 | 145.4 |
| `attnqkq` | 4 | 251.9 | 2 | 180.2 |
| `amax` | 1 | 117.8 | 1 | 106.5 |
| `bf16hcrx` | 49 | 1177.6 | 2 | 78.8 |
| `hcwsum` | 48 | 2358.3 | 1 | 124.9 |
| `ple_rowsq` | 0 | 0.0 | 1 | 67.6 |
| `kvgather` | 4 | 80.9 | 2 | 41.0 |
| `rms2560` | 3 | 25.6 | 1 | 25.6 |
| `bf16k2560` | 53 | 1452.0 | 1 | 15.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 11.3 | 1 | 11.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 2112.5 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 257.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 35.8 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 251.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2471.9 | 0 | 0.0 |
| `conv1d` | 36 | 230.4 | 0 | 0.0 |
| `dnnormx` | 36 | 543.7 | 0 | 0.0 |
| `dnstep` | 36 | 619.5 | 0 | 0.0 |
| `ffnact` | 48 | 649.2 | 0 | 0.0 |
| `idxqk` | 4 | 102.4 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 54.3 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 59.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 214.0 | 0 | 0.0 |
| `router` | 48 | 1085.4 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5847.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 3438.6 |
| `router_full` | 0 | 0.0 | 48 | 2607.1 |
| `q2moegn` | 96 | 1242.1 | 48 | 1664.0 |
| `q3kk2560` | 56 | 1630.2 | 23 | 1596.4 |
| `bf16hcrwx` | 48 | 1820.7 | 48 | 1624.1 |
| `q2moedn` | 48 | 742.4 | 48 | 1376.3 |
| `iq4k2560` | 23 | 819.2 | 18 | 1271.8 |
| `q4kk6144` | 15 | 360.4 | 15 | 1418.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1057.8 |
| `q6kk1536` | 6 | 95.2 | 6 | 928.8 |
| `q5kk6144` | 7 | 236.5 | 7 | 912.4 |
| `iq4k6144` | 17 | 423.9 | 17 | 919.6 |
| `q5kk2560` | 1 | 478.2 | 1 | 484.4 |
| `q4kk2560` | 8 | 232.4 | 4 | 297.0 |
| `xsend` | 123 | 1446.9 | 10 | 229.4 |
| `q2k2560` | 14 | 221.2 | 4 | 208.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `q3kk6144` | 3 | 87.0 | 3 | 166.9 |
| `attnqkq` | 0 | 0.0 | 2 | 177.2 |
| `amax` | 1 | 208.9 | 1 | 203.8 |
| `bf16hcrx` | 49 | 1123.3 | 2 | 74.8 |
| `hcwsum` | 48 | 2739.2 | 1 | 70.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 67.6 |
| `kvgather` | 0 | 0.0 | 2 | 43.0 |
| `rms2560` | 3 | 22.5 | 1 | 31.7 |
| `bf16k2560` | 49 | 1313.8 | 1 | 17.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 3918.8 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 206.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2116.6 | 0 | 0.0 |
| `conv1d` | 36 | 199.7 | 0 | 0.0 |
| `dnnormx` | 36 | 309.2 | 0 | 0.0 |
| `dnstep` | 36 | 607.2 | 0 | 0.0 |
| `ffnact` | 48 | 635.9 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 213.0 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 1 | 14.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 207.9 | 0 | 0.0 |
| `router` | 48 | 1077.2 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5532.7 |
| `hcupq_x` | 0 | 0.0 | 97 | 3183.6 |
| `router_full` | 0 | 0.0 | 48 | 2582.5 |
| `q2moegn` | 96 | 1531.9 | 48 | 1943.6 |
| `q3kk2560` | 56 | 1827.8 | 23 | 1751.0 |
| `bf16hcrwx` | 48 | 1549.3 | 48 | 1628.2 |
| `q2moedn` | 48 | 897.0 | 48 | 1560.6 |
| `iq4k2560` | 23 | 926.7 | 16 | 1267.7 |
| `q4kk6144` | 15 | 423.9 | 15 | 1366.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1102.8 |
| `q6kk1536` | 6 | 105.5 | 6 | 1126.4 |
| `q5kk6144` | 7 | 279.6 | 7 | 695.3 |
| `iq4k6144` | 17 | 492.5 | 17 | 759.8 |
| `q5kk2560` | 1 | 682.0 | 3 | 836.6 |
| `q4kk2560` | 8 | 275.5 | 4 | 340.0 |
| `xsend` | 123 | 1557.5 | 10 | 210.9 |
| `q2k2560` | 14 | 236.5 | 4 | 216.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 183.3 |
| `q3kk6144` | 3 | 102.4 | 3 | 141.3 |
| `attnqkq` | 0 | 0.0 | 2 | 178.2 |
| `amax` | 1 | 6.1 | 1 | 8.2 |
| `bf16hcrx` | 49 | 1220.6 | 2 | 80.9 |
| `hcwsum` | 48 | 2310.1 | 1 | 70.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 41.0 |
| `rms2560` | 3 | 22.5 | 1 | 33.8 |
| `bf16k2560` | 49 | 1315.8 | 1 | 14.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 2959.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 37.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 50.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 213.0 | 0 | 0.0 |
| `bf16hcup` | 97 | 2354.2 | 0 | 0.0 |
| `conv1d` | 36 | 206.8 | 0 | 0.0 |
| `dnnormx` | 36 | 449.5 | 0 | 0.0 |
| `dnstep` | 36 | 642.0 | 0 | 0.0 |
| `ffnact` | 48 | 643.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 210.9 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 3 | 46.1 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 227.3 | 0 | 0.0 |
| `router` | 48 | 1080.3 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5750.8 |
| `hcupq_x` | 0 | 0.0 | 97 | 3301.4 |
| `router_full` | 0 | 0.0 | 48 | 2585.6 |
| `q2moegn` | 96 | 1417.2 | 48 | 1835.0 |
| `q3kk2560` | 53 | 1635.3 | 24 | 1731.6 |
| `bf16hcrwx` | 48 | 1527.8 | 48 | 1589.2 |
| `q2moedn` | 48 | 829.4 | 48 | 1372.2 |
| `iq4k2560` | 24 | 863.2 | 17 | 1299.5 |
| `q4kk6144` | 15 | 408.6 | 15 | 1221.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1222.7 |
| `q6kk1536` | 6 | 102.4 | 6 | 951.3 |
| `q5kk6144` | 7 | 271.4 | 7 | 1096.7 |
| `iq4k6144` | 17 | 476.2 | 17 | 776.2 |
| `q5kk2560` | 2 | 598.0 | 1 | 571.4 |
| `q4kk2560` | 12 | 305.2 | 4 | 321.5 |
| `xsend` | 123 | 1598.5 | 10 | 218.1 |
| `q2k2560` | 13 | 210.9 | 4 | 200.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `q3kk6144` | 3 | 98.3 | 3 | 146.4 |
| `attnqkq` | 4 | 210.9 | 2 | 120.8 |
| `amax` | 1 | 111.6 | 1 | 115.7 |
| `bf16hcrx` | 49 | 1163.3 | 2 | 78.8 |
| `hcwsum` | 48 | 2435.1 | 1 | 63.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 67.6 |
| `kvgather` | 4 | 82.9 | 2 | 39.9 |
| `rms2560` | 3 | 21.5 | 1 | 41.0 |
| `bf16k2560` | 53 | 1441.8 | 1 | 16.4 |
| `q6kk2560` | 2 | 51.2 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 10.2 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2222.1 | 0 | 0.0 |
| `attnmerge` | 4 | 31.7 | 0 | 0.0 |
| `attnpartn` | 4 | 257.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 43.0 | 0 | 0.0 |
| `b32q40acc` | 4 | 72.7 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 228.4 | 0 | 0.0 |
| `bf16hcup` | 97 | 2505.7 | 0 | 0.0 |
| `conv1d` | 36 | 226.3 | 0 | 0.0 |
| `dnnormx` | 36 | 557.1 | 0 | 0.0 |
| `dnstep` | 36 | 618.5 | 0 | 0.0 |
| `ffnact` | 48 | 643.1 | 0 | 0.0 |
| `idxqk` | 4 | 28.7 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 216.1 | 0 | 0.0 |
| `plegate` | 1 | 9.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 216.1 | 0 | 0.0 |
| `router` | 48 | 1082.4 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5518.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3226.6 |
| `router_full` | 0 | 0.0 | 48 | 2593.8 |
| `q2moegn` | 96 | 1536.0 | 48 | 1977.3 |
| `q3kk2560` | 54 | 1670.1 | 23 | 1663.0 |
| `bf16hcrwx` | 48 | 1572.9 | 48 | 1591.3 |
| `q2moedn` | 48 | 902.1 | 48 | 1499.1 |
| `iq4k2560` | 22 | 811.0 | 17 | 1284.1 |
| `q4kk6144` | 15 | 405.5 | 15 | 1261.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1229.8 |
| `q6kk1536` | 6 | 103.4 | 6 | 1121.3 |
| `q5kk6144` | 7 | 273.4 | 7 | 975.9 |
| `iq4k6144` | 17 | 474.1 | 17 | 777.2 |
| `q5kk2560` | 4 | 646.1 | 1 | 576.5 |
| `q4kk2560` | 9 | 266.2 | 4 | 322.6 |
| `xsend` | 123 | 1539.1 | 10 | 209.9 |
| `q2k2560` | 13 | 212.0 | 4 | 201.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 186.4 |
| `q3kk6144` | 3 | 100.4 | 3 | 146.4 |
| `attnqkq` | 4 | 20.5 | 2 | 22.5 |
| `amax` | 1 | 111.6 | 1 | 110.6 |
| `bf16hcrx` | 49 | 1174.5 | 2 | 77.8 |
| `hcwsum` | 48 | 2214.9 | 1 | 91.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 85.0 | 2 | 42.0 |
| `rms2560` | 3 | 22.5 | 1 | 41.0 |
| `bf16k2560` | 53 | 1434.6 | 1 | 16.4 |
| `q6kk2560` | 4 | 110.6 | 1 | 87.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 10.2 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 11.3 |
| `actq256` | 146 | 2432.0 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 252.9 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 54.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 239.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2459.6 | 0 | 0.0 |
| `conv1d` | 36 | 224.3 | 0 | 0.0 |
| `dnnormx` | 36 | 556.0 | 0 | 0.0 |
| `dnstep` | 36 | 623.6 | 0 | 0.0 |
| `ffnact` | 48 | 644.1 | 0 | 0.0 |
| `idxqk` | 4 | 16.4 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 222.2 | 0 | 0.0 |
| `plegate` | 1 | 9.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 214.0 | 0 | 0.0 |
| `router` | 48 | 1079.3 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

