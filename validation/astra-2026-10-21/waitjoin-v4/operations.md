# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `bbcf3f39-3404-455f-a010-83eda2c9ed17`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3968.7 | +3968.7 |
| `hcupq_x` | 0 | 0.0 | 97 | 2993.8 | +2993.8 |
| `router_full` | 0 | 0.0 | 48 | 2573.8 | +2573.8 |
| `q2moegn` | 96 | 1450.0 | 48 | 1872.1 | +422.1 |
| `q3kk2560` | 53–56 | 1676.1 | 23–24 | 1669.8 | -6.3 |
| `bf16hcrwx` | 48 | 1587.7 | 48 | 1590.3 | +2.6 |
| `q2moedn` | 48 | 853.2 | 48 | 1467.4 | +614.2 |
| `q4kk6144` | 15 | 405.0 | 15 | 1276.6 | +871.6 |
| `iq4k2560` | 22–24 | 870.2 | 16–18 | 1265.7 | +395.5 |
| `gdn_full` | 0 | 0.0 | 36 | 1172.8 | +1172.8 |
| `q6kk1536` | 6 | 103.6 | 6 | 1003.2 | +899.6 |
| `q5kk6144` | 7 | 268.1 | 7 | 980.5 | +712.4 |
| `iq4k6144` | 17 | 472.2 | 17 | 805.9 | +333.7 |
| `q5kk2560` | 1–4 | 620.2 | 1–3 | 619.0 | -1.2 |
| `q4kk2560` | 8–12 | 276.8 | 4–6 | 339.6 | +62.8 |
| `xsend` | 123–124 | 1562.0 | 10 | 221.0 | -1340.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 189.8 | +189.8 |
| `q2k2560` | 12–14 | 213.7 | 3–4 | 181.1 | -32.6 |
| `q3kk6144` | 3 | 99.5 | 3 | 154.1 | +54.6 |
| `attnqkq` | 0–4 | 80.0 | 2 | 144.2 | +64.2 |
| `amax` | 1 | 95.2 | 1 | 98.3 | +3.0 |
| `bf16hcrx` | 49 | 1177.1 | 2 | 73.6 | -1103.5 |
| `hcwsum` | 48 | 2363.6 | 1 | 71.3 | -2292.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.4 | +66.4 |
| `kvgather` | 0–4 | 42.0 | 2 | 40.8 | -1.2 |
| `rms2560` | 3 | 23.2 | 1 | 30.9 | +7.6 |
| `bf16k2560` | 49–53 | 1381.2 | 1 | 16.7 | -1364.5 |
| `q6kk2560` | 0–4 | 27.0 | 0–1 | 14.5 | -12.5 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 | +13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 | +12.3 |
| `argmax` | 1 | 10.4 | 1 | 12.1 | +1.7 |
| `pleconv` | 1 | 9.9 | 1 | 10.6 | +0.7 |
| `actq256` | 146 | 2779.7 | 0 | 0.0 | -2779.7 |
| `attnmerge` | 0–4 | 16.2 | 0 | 0.0 | -16.2 |
| `attnpartn` | 0–4 | 127.3 | 0 | 0.0 | -127.3 |
| `b32iq4nlacc` | 0–2 | 25.8 | 0 | 0.0 | -25.8 |
| `b32q40acc` | 2–4 | 48.8 | 0 | 0.0 | -48.8 |
| `b32q50acc` | 0–2 | 29.7 | 0 | 0.0 | -29.7 |
| `b32q80acc` | 0–1 | 8.2 | 0 | 0.0 | -8.2 |
| `bf16bias` | 36 | 228.5 | 0 | 0.0 | -228.5 |
| `bf16hcup` | 97 | 2375.5 | 0 | 0.0 | -2375.5 |
| `conv1d` | 36 | 217.1 | 0 | 0.0 | -217.1 |
| `dnnormx` | 36 | 488.6 | 0 | 0.0 | -488.6 |
| `dnstep` | 36 | 620.5 | 0 | 0.0 | -620.5 |
| `ffnact` | 48 | 645.8 | 0 | 0.0 | -645.8 |
| `idxqk` | 0–4 | 25.4 | 0 | 0.0 | -25.4 |
| `idxscore` | 0–4 | 5.6 | 0 | 0.0 | -5.6 |
| `idxsel` | 0–20 | 26.9 | 0 | 0.0 | -26.9 |
| `moesel` | 48 | 214.9 | 0 | 0.0 | -214.9 |
| `plegate` | 1 | 8.0 | 0 | 0.0 | -8.0 |
| `plerows` | 1 | 55.5 | 0 | 0.0 | -55.5 |
| `q2acc640` | 1–4 | 40.3 | 0 | 0.0 | -40.3 |
| `q6kk1536acc` | 18 | 217.4 | 0 | 0.0 | -217.4 |
| `router` | 48 | 1085.3 | 0 | 0.0 | -1085.3 |
| `xcopy` | 0–1 | 27.1 | 0 | 0.0 | -27.1 |
| **All recorded intervals** | 1,177–1,229 | **24986.2** | 498 | **24950.1** | **-36.1** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3540.7 | 4113.6 | +572.9 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6680.4 | 4422.8 | -2257.6 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.6 | 12.3 | -20.3 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 31.2 | 44.2 | +12.9 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 321.0 | 1003.2 | +682.2 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3973.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 2981.9 |
| `router_full` | 0 | 0.0 | 48 | 2536.4 |
| `q2moegn` | 96 | 1525.8 | 48 | 1961.0 |
| `q3kk2560` | 54 | 1676.3 | 24 | 1680.4 |
| `bf16hcrwx` | 48 | 1522.7 | 48 | 1553.4 |
| `q2moedn` | 48 | 895.0 | 48 | 1498.1 |
| `q4kk6144` | 15 | 420.9 | 15 | 1196.0 |
| `iq4k2560` | 24 | 903.2 | 18 | 1310.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1231.9 |
| `q6kk1536` | 6 | 108.5 | 6 | 929.8 |
| `q5kk6144` | 7 | 275.5 | 7 | 1102.8 |
| `iq4k6144` | 17 | 490.5 | 17 | 803.8 |
| `q5kk2560` | 2 | 685.1 | 1 | 662.5 |
| `q4kk2560` | 10 | 276.5 | 4 | 314.4 |
| `xsend` | 124 | 1653.8 | 10 | 208.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 201.7 |
| `q2k2560` | 12 | 214.0 | 3 | 151.6 |
| `q3kk6144` | 3 | 104.4 | 3 | 160.8 |
| `attnqkq` | 0 | 0.0 | 2 | 182.3 |
| `amax` | 1 | 23.6 | 1 | 27.6 |
| `bf16hcrx` | 49 | 1217.5 | 2 | 48.1 |
| `hcwsum` | 48 | 2227.2 | 1 | 5.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 42.0 |
| `rms2560` | 3 | 21.5 | 1 | 29.7 |
| `bf16k2560` | 49 | 1317.9 | 1 | 17.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 11.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3043.3 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 244.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2335.7 | 0 | 0.0 |
| `conv1d` | 36 | 221.2 | 0 | 0.0 |
| `dnnormx` | 36 | 550.9 | 0 | 0.0 |
| `dnstep` | 36 | 623.6 | 0 | 0.0 |
| `ffnact` | 48 | 651.3 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 210.9 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 60.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 223.2 | 0 | 0.0 |
| `router` | 48 | 1090.6 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3912.7 |
| `hcupq_x` | 0 | 0.0 | 97 | 2997.2 |
| `router_full` | 0 | 0.0 | 48 | 2573.3 |
| `q2moegn` | 96 | 1459.2 | 48 | 1866.8 |
| `q3kk2560` | 53 | 1650.7 | 24 | 1683.5 |
| `bf16hcrwx` | 48 | 1531.9 | 48 | 1528.8 |
| `q2moedn` | 48 | 864.3 | 48 | 1479.7 |
| `q4kk6144` | 15 | 413.7 | 15 | 1191.9 |
| `iq4k2560` | 24 | 876.5 | 16 | 1202.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1234.9 |
| `q6kk1536` | 6 | 103.4 | 6 | 921.6 |
| `q5kk6144` | 7 | 273.4 | 7 | 1095.7 |
| `iq4k6144` | 17 | 477.2 | 17 | 807.9 |
| `q5kk2560` | 4 | 634.9 | 1 | 576.5 |
| `q4kk2560` | 11 | 300.0 | 6 | 464.9 |
| `xsend` | 123 | 1557.5 | 10 | 212.0 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 201.7 |
| `q2k2560` | 14 | 190.5 | 3 | 151.6 |
| `q3kk6144` | 3 | 101.4 | 3 | 154.6 |
| `attnqkq` | 4 | 248.8 | 2 | 181.2 |
| `amax` | 1 | 109.6 | 1 | 115.7 |
| `bf16hcrx` | 49 | 1182.7 | 2 | 79.9 |
| `hcwsum` | 48 | 2340.9 | 1 | 123.9 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 84.0 | 2 | 41.0 |
| `rms2560` | 3 | 22.5 | 1 | 24.6 |
| `bf16k2560` | 53 | 1445.9 | 1 | 17.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2101.2 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 228.4 | 0 | 0.0 |
| `bf16hcup` | 97 | 2486.3 | 0 | 0.0 |
| `conv1d` | 36 | 225.3 | 0 | 0.0 |
| `dnnormx` | 36 | 561.2 | 0 | 0.0 |
| `dnstep` | 36 | 618.5 | 0 | 0.0 |
| `ffnact` | 48 | 646.1 | 0 | 0.0 |
| `idxqk` | 4 | 104.4 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 219.1 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 4 | 62.5 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 216.1 | 0 | 0.0 |
| `router` | 48 | 1093.6 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4202.5 |
| `hcupq_x` | 0 | 0.0 | 97 | 3119.1 |
| `router_full` | 0 | 0.0 | 48 | 2590.7 |
| `q2moegn` | 96 | 1246.2 | 48 | 1648.6 |
| `q3kk2560` | 56 | 1627.1 | 23 | 1581.1 |
| `bf16hcrwx` | 48 | 1825.8 | 48 | 1628.2 |
| `q2moedn` | 48 | 739.3 | 48 | 1378.3 |
| `q4kk6144` | 15 | 360.4 | 15 | 1426.4 |
| `iq4k2560` | 23 | 824.3 | 18 | 1263.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1056.8 |
| `q6kk1536` | 6 | 96.3 | 6 | 944.1 |
| `q5kk6144` | 7 | 234.5 | 7 | 917.5 |
| `iq4k6144` | 17 | 422.9 | 17 | 903.2 |
| `q5kk2560` | 1 | 480.3 | 1 | 482.3 |
| `q4kk2560` | 8 | 233.5 | 4 | 290.8 |
| `xsend` | 123 | 1446.9 | 10 | 237.6 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `q2k2560` | 14 | 224.3 | 4 | 194.6 |
| `q3kk6144` | 3 | 88.1 | 3 | 159.7 |
| `attnqkq` | 0 | 0.0 | 2 | 180.2 |
| `amax` | 1 | 205.8 | 1 | 209.9 |
| `bf16hcrx` | 49 | 1112.1 | 2 | 75.8 |
| `hcwsum` | 48 | 2704.4 | 1 | 72.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 0 | 0.0 | 2 | 42.0 |
| `rms2560` | 3 | 23.6 | 1 | 31.7 |
| `bf16k2560` | 49 | 1330.2 | 1 | 17.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 14.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 9.2 | 1 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 3904.5 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 24.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 201.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2113.5 | 0 | 0.0 |
| `conv1d` | 36 | 199.7 | 0 | 0.0 |
| `dnnormx` | 36 | 311.3 | 0 | 0.0 |
| `dnstep` | 36 | 600.1 | 0 | 0.0 |
| `ffnact` | 48 | 641.0 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 54.3 | 0 | 0.0 |
| `q2acc640` | 1 | 14.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 206.8 | 0 | 0.0 |
| `router` | 48 | 1079.3 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3842.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 2913.3 |
| `router_full` | 0 | 0.0 | 48 | 2560.0 |
| `q2moegn` | 96 | 1535.0 | 48 | 1965.1 |
| `q3kk2560` | 56 | 1817.6 | 23 | 1728.5 |
| `bf16hcrwx` | 48 | 1552.4 | 48 | 1625.1 |
| `q2moedn` | 48 | 898.0 | 48 | 1555.5 |
| `q4kk6144` | 15 | 420.9 | 15 | 1364.0 |
| `iq4k2560` | 23 | 943.1 | 16 | 1254.4 |
| `gdn_full` | 0 | 0.0 | 36 | 1090.6 |
| `q6kk1536` | 6 | 107.5 | 6 | 1131.5 |
| `q5kk6144` | 7 | 277.5 | 7 | 699.4 |
| `iq4k6144` | 17 | 492.5 | 17 | 768.0 |
| `q5kk2560` | 1 | 681.0 | 3 | 837.6 |
| `q4kk2560` | 8 | 272.4 | 4 | 333.8 |
| `xsend` | 123 | 1586.2 | 10 | 220.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `q2k2560` | 14 | 229.4 | 4 | 197.6 |
| `q3kk6144` | 3 | 103.4 | 3 | 143.4 |
| `attnqkq` | 0 | 0.0 | 2 | 178.2 |
| `amax` | 1 | 5.1 | 1 | 8.2 |
| `bf16hcrx` | 49 | 1223.7 | 2 | 81.9 |
| `hcwsum` | 48 | 2268.2 | 1 | 71.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `kvgather` | 0 | 0.0 | 2 | 39.9 |
| `rms2560` | 3 | 24.6 | 1 | 35.8 |
| `bf16k2560` | 49 | 1319.9 | 1 | 14.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 11.3 | 1 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 10.2 |
| `actq256` | 146 | 2951.2 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 21.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 205.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2353.2 | 0 | 0.0 |
| `conv1d` | 36 | 206.8 | 0 | 0.0 |
| `dnnormx` | 36 | 448.5 | 0 | 0.0 |
| `dnstep` | 36 | 635.9 | 0 | 0.0 |
| `ffnact` | 48 | 647.2 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 213.0 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 3 | 43.0 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 225.3 | 0 | 0.0 |
| `router` | 48 | 1074.2 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 4086.8 |
| `hcupq_x` | 0 | 0.0 | 97 | 2988.0 |
| `router_full` | 0 | 0.0 | 48 | 2597.9 |
| `q2moegn` | 96 | 1394.7 | 48 | 1821.7 |
| `q3kk2560` | 53 | 1611.8 | 24 | 1715.2 |
| `bf16hcrwx` | 48 | 1527.8 | 48 | 1594.4 |
| `q2moedn` | 48 | 819.2 | 48 | 1377.3 |
| `q4kk6144` | 15 | 404.5 | 15 | 1217.5 |
| `iq4k2560` | 24 | 866.3 | 17 | 1293.3 |
| `gdn_full` | 0 | 0.0 | 36 | 1205.2 |
| `q6kk1536` | 6 | 103.4 | 6 | 960.5 |
| `q5kk6144` | 7 | 272.4 | 7 | 1094.7 |
| `iq4k6144` | 17 | 475.1 | 17 | 773.1 |
| `q5kk2560` | 2 | 598.0 | 1 | 579.6 |
| `q4kk2560` | 12 | 312.3 | 4 | 316.4 |
| `xsend` | 123 | 1594.4 | 10 | 222.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 183.3 |
| `q2k2560` | 13 | 212.0 | 4 | 194.6 |
| `q3kk6144` | 3 | 98.3 | 3 | 153.6 |
| `attnqkq` | 4 | 212.0 | 2 | 121.9 |
| `amax` | 1 | 111.6 | 1 | 112.6 |
| `bf16hcrx` | 49 | 1156.1 | 2 | 76.8 |
| `hcwsum` | 48 | 2437.1 | 1 | 64.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 85.0 | 2 | 39.9 |
| `rms2560` | 3 | 22.5 | 1 | 31.7 |
| `bf16k2560` | 53 | 1435.6 | 1 | 17.4 |
| `q6kk2560` | 2 | 52.2 | 0 | 0.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 14.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2238.5 | 0 | 0.0 |
| `attnmerge` | 4 | 31.7 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 44.0 | 0 | 0.0 |
| `b32q40acc` | 4 | 73.7 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 252.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2502.7 | 0 | 0.0 |
| `conv1d` | 36 | 225.3 | 0 | 0.0 |
| `dnnormx` | 36 | 529.4 | 0 | 0.0 |
| `dnstep` | 36 | 622.6 | 0 | 0.0 |
| `ffnact` | 48 | 645.1 | 0 | 0.0 |
| `idxqk` | 4 | 29.7 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1086.5 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 3794.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 2963.5 |
| `router_full` | 0 | 0.0 | 48 | 2584.6 |
| `q2moegn` | 96 | 1539.1 | 48 | 1969.2 |
| `q3kk2560` | 54 | 1673.2 | 23 | 1630.2 |
| `bf16hcrwx` | 48 | 1565.7 | 48 | 1611.8 |
| `q2moedn` | 48 | 903.2 | 48 | 1515.5 |
| `q4kk6144` | 15 | 409.6 | 15 | 1263.6 |
| `iq4k2560` | 22 | 807.9 | 17 | 1269.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1217.5 |
| `q6kk1536` | 6 | 102.4 | 6 | 1131.5 |
| `q5kk6144` | 7 | 275.5 | 7 | 972.8 |
| `iq4k6144` | 17 | 475.1 | 17 | 779.3 |
| `q5kk2560` | 4 | 642.0 | 1 | 575.5 |
| `q4kk2560` | 9 | 266.2 | 4 | 317.4 |
| `xsend` | 123 | 1532.9 | 10 | 225.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 185.3 |
| `q2k2560` | 13 | 212.0 | 4 | 196.6 |
| `q3kk6144` | 3 | 101.4 | 3 | 152.6 |
| `attnqkq` | 4 | 19.5 | 2 | 21.5 |
| `amax` | 1 | 115.7 | 1 | 115.7 |
| `bf16hcrx` | 49 | 1170.4 | 2 | 78.8 |
| `hcwsum` | 48 | 2203.6 | 1 | 90.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `kvgather` | 4 | 82.9 | 2 | 39.9 |
| `rms2560` | 3 | 24.6 | 1 | 31.7 |
| `bf16k2560` | 53 | 1437.7 | 1 | 16.4 |
| `q6kk2560` | 4 | 109.6 | 1 | 87.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 14.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `argmax` | 1 | 10.2 | 1 | 13.3 |
| `pleconv` | 1 | 9.2 | 1 | 11.3 |
| `actq256` | 146 | 2439.2 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 254.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 21.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 53.2 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 237.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2461.7 | 0 | 0.0 |
| `conv1d` | 36 | 224.3 | 0 | 0.0 |
| `dnnormx` | 36 | 530.4 | 0 | 0.0 |
| `dnstep` | 36 | 622.6 | 0 | 0.0 |
| `ffnact` | 48 | 644.1 | 0 | 0.0 |
| `idxqk` | 4 | 18.4 | 0 | 0.0 |
| `idxscore` | 4 | 10.2 | 0 | 0.0 |
| `idxsel` | 20 | 55.3 | 0 | 0.0 |
| `moesel` | 48 | 212.0 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 216.1 | 0 | 0.0 |
| `router` | 48 | 1087.5 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

