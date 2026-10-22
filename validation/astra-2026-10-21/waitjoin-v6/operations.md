# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `84dff40d-5cc1-44a7-a99f-7c26cd02a28c`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3941.0 | +3941.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 2993.5 | +2993.5 |
| `router_full` | 0 | 0.0 | 48 | 2545.7 | +2545.7 |
| `q2moegn` | 96 | 1449.6 | 48 | 1855.6 | +406.0 |
| `q3kk2560` | 53–56 | 1680.9 | 23–24 | 1673.4 | -7.5 |
| `bf16hcrwx` | 48 | 1594.0 | 48 | 1560.4 | -33.6 |
| `q2moedn` | 48 | 857.8 | 48 | 1482.4 | +624.6 |
| `q4kk6144` | 15 | 403.6 | 15 | 1291.8 | +888.2 |
| `iq4k2560` | 22–24 | 867.1 | 16–18 | 1275.0 | +407.9 |
| `gdn_full` | 0 | 0.0 | 36 | 1167.9 | +1167.9 |
| `q6kk1536` | 6 | 103.8 | 6 | 1004.0 | +900.3 |
| `q5kk6144` | 7 | 268.4 | 7 | 989.9 | +721.5 |
| `iq4k6144` | 17 | 475.7 | 17 | 802.8 | +327.1 |
| `q5kk2560` | 1–4 | 618.5 | 1–3 | 618.9 | +0.4 |
| `q4kk2560` | 8–12 | 278.9 | 4–6 | 341.0 | +62.1 |
| `xsend` | 123–124 | 1555.4 | 10 | 223.9 | -1331.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 189.1 | +189.1 |
| `q2k2560` | 12–14 | 212.3 | 3–4 | 177.2 | -35.2 |
| `q3kk6144` | 3 | 100.2 | 3 | 159.9 | +59.7 |
| `attnqkq` | 0–4 | 75.5 | 2 | 145.1 | +69.6 |
| `amax` | 1 | 93.7 | 1 | 100.9 | +7.2 |
| `bf16hcrx` | 49 | 1174.9 | 2 | 73.0 | -1101.9 |
| `hcwsum` | 48 | 2377.4 | 1 | 71.0 | -2306.4 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 | +59.4 |
| `kvgather` | 0–4 | 41.1 | 2 | 42.7 | +1.6 |
| `rms2560` | 3 | 23.2 | 1 | 30.2 | +6.9 |
| `bf16k2560` | 49–53 | 1385.8 | 1 | 17.7 | -1368.1 |
| `q6kk2560` | 0–4 | 27.1 | 0–1 | 14.5 | -12.6 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.8 | +12.8 |
| `argmax` | 1 | 10.6 | 1 | 12.5 | +1.9 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.5 | +12.5 |
| `pleconv` | 1 | 10.0 | 1 | 10.6 | +0.5 |
| `actq256` | 146 | 2790.0 | 0 | 0.0 | -2790.0 |
| `attnmerge` | 0–4 | 16.6 | 0 | 0.0 | -16.6 |
| `attnpartn` | 0–4 | 128.3 | 0 | 0.0 | -128.3 |
| `b32iq4nlacc` | 0–2 | 25.9 | 0 | 0.0 | -25.9 |
| `b32q40acc` | 2–4 | 49.0 | 0 | 0.0 | -49.0 |
| `b32q50acc` | 0–2 | 29.5 | 0 | 0.0 | -29.5 |
| `b32q80acc` | 0–1 | 8.0 | 0 | 0.0 | -8.0 |
| `bf16bias` | 36 | 233.6 | 0 | 0.0 | -233.6 |
| `bf16hcup` | 97 | 2367.7 | 0 | 0.0 | -2367.7 |
| `conv1d` | 36 | 217.6 | 0 | 0.0 | -217.6 |
| `dnnormx` | 36 | 489.5 | 0 | 0.0 | -489.5 |
| `dnstep` | 36 | 618.3 | 0 | 0.0 | -618.3 |
| `ffnact` | 48 | 645.0 | 0 | 0.0 | -645.0 |
| `idxqk` | 0–4 | 29.4 | 0 | 0.0 | -29.4 |
| `idxscore` | 0–4 | 5.8 | 0 | 0.0 | -5.8 |
| `idxsel` | 0–20 | 26.9 | 0 | 0.0 | -26.9 |
| `moesel` | 48 | 214.3 | 0 | 0.0 | -214.3 |
| `plegate` | 1 | 8.7 | 0 | 0.0 | -8.7 |
| `plerows` | 1 | 62.8 | 0 | 0.0 | -62.8 |
| `q2acc640` | 1–4 | 40.3 | 0 | 0.0 | -40.3 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 | -217.1 |
| `router` | 48 | 1082.7 | 0 | 0.0 | -1082.7 |
| `xcopy` | 0–1 | 26.6 | 0 | 0.0 | -26.6 |
| **All recorded intervals** | 1,177–1,229 | **25019.4** | 498 | **24896.1** | **-123.3** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3552.2 | 4085.0 | +532.8 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6690.9 | 4409.9 | -2281.0 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.8 | 12.5 | -20.3 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 31.9 | 43.0 | +11.0 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 320.9 | 1004.0 | +683.2 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3964.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 2973.7 |
| `router_full` | 0 | 0.0 | 48 | 2492.4 |
| `q2moegn` | 96 | 1521.7 | 48 | 1940.5 |
| `q3kk2560` | 54 | 1698.8 | 24 | 1696.8 |
| `bf16hcrwx` | 48 | 1527.8 | 48 | 1535.0 |
| `q2moedn` | 48 | 907.3 | 48 | 1509.4 |
| `q4kk6144` | 15 | 419.8 | 15 | 1214.5 |
| `iq4k2560` | 24 | 892.9 | 18 | 1315.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1216.5 |
| `q6kk1536` | 6 | 107.5 | 6 | 929.8 |
| `q5kk6144` | 7 | 278.5 | 7 | 1112.1 |
| `iq4k6144` | 17 | 493.6 | 17 | 799.7 |
| `q5kk2560` | 2 | 681.0 | 1 | 661.5 |
| `q4kk2560` | 10 | 281.6 | 4 | 317.4 |
| `xsend` | 124 | 1632.3 | 10 | 214.0 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 200.7 |
| `q2k2560` | 12 | 212.0 | 3 | 146.4 |
| `q3kk6144` | 3 | 106.5 | 3 | 161.8 |
| `attnqkq` | 0 | 0.0 | 2 | 184.3 |
| `amax` | 1 | 24.6 | 1 | 30.7 |
| `bf16hcrx` | 49 | 1209.3 | 2 | 48.1 |
| `hcwsum` | 48 | 2251.8 | 1 | 6.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 0 | 0.0 | 2 | 42.0 |
| `rms2560` | 3 | 23.6 | 1 | 31.7 |
| `bf16k2560` | 49 | 1310.7 | 1 | 16.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 3078.1 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 35.8 | 0 | 0.0 |
| `b32q50acc` | 2 | 49.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 244.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2303.0 | 0 | 0.0 |
| `conv1d` | 36 | 227.3 | 0 | 0.0 |
| `dnnormx` | 36 | 556.0 | 0 | 0.0 |
| `dnstep` | 36 | 615.4 | 0 | 0.0 |
| `ffnact` | 48 | 651.3 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 215.0 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 62.5 | 0 | 0.0 |
| `q2acc640` | 4 | 61.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 225.3 | 0 | 0.0 |
| `router` | 48 | 1086.5 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3867.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3019.8 |
| `router_full` | 0 | 0.0 | 48 | 2541.6 |
| `q2moegn` | 96 | 1456.1 | 48 | 1857.5 |
| `q3kk2560` | 53 | 1665.0 | 24 | 1689.6 |
| `bf16hcrwx` | 48 | 1549.3 | 48 | 1497.1 |
| `q2moedn` | 48 | 875.5 | 48 | 1495.0 |
| `q4kk6144` | 15 | 407.6 | 15 | 1203.2 |
| `iq4k2560` | 24 | 869.4 | 16 | 1215.5 |
| `gdn_full` | 0 | 0.0 | 36 | 1219.6 |
| `q6kk1536` | 6 | 102.4 | 6 | 920.6 |
| `q5kk6144` | 7 | 273.4 | 7 | 1104.9 |
| `iq4k6144` | 17 | 481.3 | 17 | 800.8 |
| `q5kk2560` | 4 | 630.8 | 1 | 576.5 |
| `q4kk2560` | 11 | 302.1 | 6 | 466.9 |
| `xsend` | 123 | 1557.5 | 10 | 208.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 202.8 |
| `q2k2560` | 14 | 191.5 | 3 | 152.6 |
| `q3kk6144` | 3 | 103.4 | 3 | 165.9 |
| `attnqkq` | 4 | 237.6 | 2 | 183.3 |
| `amax` | 1 | 109.6 | 1 | 116.7 |
| `bf16hcrx` | 49 | 1181.7 | 2 | 77.8 |
| `hcwsum` | 48 | 2347.0 | 1 | 126.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 4 | 80.9 | 2 | 41.0 |
| `rms2560` | 3 | 22.5 | 1 | 24.6 |
| `bf16k2560` | 53 | 1444.9 | 1 | 18.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 11.3 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 11.3 |
| `actq256` | 146 | 2121.7 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 37.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 234.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2475.0 | 0 | 0.0 |
| `conv1d` | 36 | 227.3 | 0 | 0.0 |
| `dnnormx` | 36 | 548.9 | 0 | 0.0 |
| `dnstep` | 36 | 622.6 | 0 | 0.0 |
| `ffnact` | 48 | 644.1 | 0 | 0.0 |
| `idxqk` | 4 | 114.7 | 0 | 0.0 |
| `idxscore` | 4 | 10.2 | 0 | 0.0 |
| `idxsel` | 20 | 55.3 | 0 | 0.0 |
| `moesel` | 48 | 209.9 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 62.5 | 0 | 0.0 |
| `q2acc640` | 4 | 59.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 215.0 | 0 | 0.0 |
| `router` | 48 | 1094.7 | 0 | 0.0 |
| `xcopy` | 1 | 30.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4178.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3098.6 |
| `router_full` | 0 | 0.0 | 48 | 2567.2 |
| `q2moegn` | 96 | 1239.0 | 48 | 1636.4 |
| `q3kk2560` | 56 | 1641.5 | 23 | 1583.1 |
| `bf16hcrwx` | 48 | 1840.1 | 48 | 1587.2 |
| `q2moedn` | 48 | 729.1 | 48 | 1390.6 |
| `q4kk6144` | 15 | 356.4 | 15 | 1458.2 |
| `iq4k2560` | 23 | 827.4 | 18 | 1266.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1061.9 |
| `q6kk1536` | 6 | 98.3 | 6 | 944.1 |
| `q5kk6144` | 7 | 235.5 | 7 | 929.8 |
| `iq4k6144` | 17 | 429.1 | 17 | 909.3 |
| `q5kk2560` | 1 | 481.3 | 1 | 482.3 |
| `q4kk2560` | 8 | 232.4 | 4 | 289.8 |
| `xsend` | 123 | 1435.6 | 10 | 240.6 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q2k2560` | 14 | 214.0 | 4 | 190.5 |
| `q3kk6144` | 3 | 90.1 | 3 | 163.8 |
| `attnqkq` | 0 | 0.0 | 2 | 181.2 |
| `amax` | 1 | 202.8 | 1 | 212.0 |
| `bf16hcrx` | 49 | 1116.2 | 2 | 78.8 |
| `hcwsum` | 48 | 2735.1 | 1 | 67.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 0 | 0.0 | 2 | 45.1 |
| `rms2560` | 3 | 23.6 | 1 | 27.6 |
| `bf16k2560` | 49 | 1329.2 | 1 | 22.5 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 9.2 | 1 | 11.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3918.8 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 23.6 | 0 | 0.0 |
| `bf16bias` | 36 | 209.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2106.4 | 0 | 0.0 |
| `conv1d` | 36 | 194.6 | 0 | 0.0 |
| `dnnormx` | 36 | 293.9 | 0 | 0.0 |
| `dnstep` | 36 | 599.0 | 0 | 0.0 |
| `ffnact` | 48 | 643.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 218.1 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 62.5 | 0 | 0.0 |
| `q2acc640` | 1 | 15.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 206.8 | 0 | 0.0 |
| `router` | 48 | 1079.3 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3814.4 |
| `hcupq_x` | 0 | 0.0 | 97 | 2900.0 |
| `router_full` | 0 | 0.0 | 48 | 2540.5 |
| `q2moegn` | 96 | 1537.0 | 48 | 1938.4 |
| `q3kk2560` | 56 | 1823.7 | 23 | 1732.6 |
| `bf16hcrwx` | 48 | 1564.7 | 48 | 1614.8 |
| `q2moedn` | 48 | 903.2 | 48 | 1580.0 |
| `q4kk6144` | 15 | 422.9 | 15 | 1373.2 |
| `iq4k2560` | 23 | 930.8 | 16 | 1271.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1095.7 |
| `q6kk1536` | 6 | 109.6 | 6 | 1136.6 |
| `q5kk6144` | 7 | 278.5 | 7 | 702.5 |
| `iq4k6144` | 17 | 495.6 | 17 | 743.4 |
| `q5kk2560` | 1 | 677.9 | 3 | 842.8 |
| `q4kk2560` | 8 | 277.5 | 4 | 332.8 |
| `xsend` | 123 | 1577.0 | 10 | 223.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q2k2560` | 14 | 231.4 | 4 | 192.5 |
| `q3kk6144` | 3 | 102.4 | 3 | 148.5 |
| `attnqkq` | 0 | 0.0 | 2 | 179.2 |
| `amax` | 1 | 6.1 | 1 | 8.2 |
| `bf16hcrx` | 49 | 1211.4 | 2 | 77.8 |
| `hcwsum` | 48 | 2271.2 | 1 | 68.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 0 | 0.0 | 2 | 44.0 |
| `rms2560` | 3 | 23.6 | 1 | 34.8 |
| `bf16k2560` | 49 | 1327.1 | 1 | 14.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 11.3 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2959.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 215.0 | 0 | 0.0 |
| `bf16hcup` | 97 | 2348.0 | 0 | 0.0 |
| `conv1d` | 36 | 201.7 | 0 | 0.0 |
| `dnnormx` | 36 | 440.3 | 0 | 0.0 |
| `dnstep` | 36 | 644.1 | 0 | 0.0 |
| `ffnact` | 48 | 647.2 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 213.0 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 63.5 | 0 | 0.0 |
| `q2acc640` | 3 | 45.1 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 224.3 | 0 | 0.0 |
| `router` | 48 | 1075.2 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4058.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 2994.2 |
| `router_full` | 0 | 0.0 | 48 | 2552.8 |
| `q2moegn` | 96 | 1400.8 | 48 | 1814.5 |
| `q3kk2560` | 53 | 1603.6 | 24 | 1707.0 |
| `bf16hcrwx` | 48 | 1517.6 | 48 | 1573.9 |
| `q2moedn` | 48 | 828.4 | 48 | 1392.6 |
| `q4kk6144` | 15 | 405.5 | 15 | 1222.7 |
| `iq4k2560` | 24 | 870.4 | 17 | 1300.5 |
| `gdn_full` | 0 | 0.0 | 36 | 1201.2 |
| `q6kk1536` | 6 | 102.4 | 6 | 957.4 |
| `q5kk6144` | 7 | 270.3 | 7 | 1104.9 |
| `iq4k6144` | 17 | 478.2 | 17 | 778.2 |
| `q5kk2560` | 2 | 590.8 | 1 | 578.6 |
| `q4kk2560` | 12 | 310.3 | 4 | 318.5 |
| `xsend` | 123 | 1588.2 | 10 | 230.4 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q2k2560` | 13 | 204.8 | 4 | 191.5 |
| `q3kk6144` | 3 | 100.4 | 3 | 160.8 |
| `attnqkq` | 4 | 194.6 | 2 | 122.9 |
| `amax` | 1 | 114.7 | 1 | 114.7 |
| `bf16hcrx` | 49 | 1166.3 | 2 | 77.8 |
| `hcwsum` | 48 | 2443.3 | 1 | 65.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 4 | 82.9 | 2 | 43.0 |
| `rms2560` | 3 | 23.6 | 1 | 30.7 |
| `bf16k2560` | 53 | 1452.0 | 1 | 17.4 |
| `q6kk2560` | 2 | 52.2 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2251.8 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 259.1 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 43.0 | 0 | 0.0 |
| `b32q40acc` | 4 | 73.7 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 249.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2496.5 | 0 | 0.0 |
| `conv1d` | 36 | 229.4 | 0 | 0.0 |
| `dnnormx` | 36 | 540.7 | 0 | 0.0 |
| `dnstep` | 36 | 614.4 | 0 | 0.0 |
| `ffnact` | 48 | 646.1 | 0 | 0.0 |
| `idxqk` | 4 | 45.1 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 216.1 | 0 | 0.0 |
| `plegate` | 1 | 9.2 | 0 | 0.0 |
| `plerows` | 1 | 63.5 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 214.0 | 0 | 0.0 |
| `router` | 48 | 1083.4 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3762.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 2974.7 |
| `router_full` | 0 | 0.0 | 48 | 2579.5 |
| `q2moegn` | 96 | 1543.2 | 48 | 1946.6 |
| `q3kk2560` | 54 | 1652.7 | 23 | 1631.2 |
| `bf16hcrwx` | 48 | 1564.7 | 48 | 1554.4 |
| `q2moedn` | 48 | 903.2 | 48 | 1526.8 |
| `q4kk6144` | 15 | 409.6 | 15 | 1279.0 |
| `iq4k2560` | 22 | 812.0 | 17 | 1280.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1212.4 |
| `q6kk1536` | 6 | 102.4 | 6 | 1135.6 |
| `q5kk6144` | 7 | 274.4 | 7 | 985.1 |
| `iq4k6144` | 17 | 476.2 | 17 | 785.4 |
| `q5kk2560` | 4 | 649.2 | 1 | 571.4 |
| `q4kk2560` | 9 | 269.3 | 4 | 320.5 |
| `xsend` | 123 | 1542.1 | 10 | 226.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `q2k2560` | 13 | 220.2 | 4 | 189.4 |
| `q3kk6144` | 3 | 98.3 | 3 | 158.7 |
| `attnqkq` | 4 | 20.5 | 2 | 19.5 |
| `amax` | 1 | 104.4 | 1 | 122.9 |
| `bf16hcrx` | 49 | 1164.3 | 2 | 77.8 |
| `hcwsum` | 48 | 2215.9 | 1 | 92.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 59.4 |
| `kvgather` | 4 | 82.9 | 2 | 41.0 |
| `rms2560` | 3 | 22.5 | 1 | 31.7 |
| `bf16k2560` | 53 | 1451.0 | 1 | 17.4 |
| `q6kk2560` | 4 | 110.6 | 1 | 87.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2410.5 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 256.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 54.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 247.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2477.1 | 0 | 0.0 |
| `conv1d` | 36 | 225.3 | 0 | 0.0 |
| `dnnormx` | 36 | 557.1 | 0 | 0.0 |
| `dnstep` | 36 | 614.4 | 0 | 0.0 |
| `ffnact` | 48 | 638.0 | 0 | 0.0 |
| `idxqk` | 4 | 16.4 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 214.0 | 0 | 0.0 |
| `plegate` | 1 | 9.2 | 0 | 0.0 |
| `plerows` | 1 | 62.5 | 0 | 0.0 |
| `q2acc640` | 2 | 29.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1077.2 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

