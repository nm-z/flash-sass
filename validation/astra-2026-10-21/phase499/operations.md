# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `03bff6fb-db1f-4844-9d91-0e2e381ea17f`, fetched with `gpuq logs ID 10000`. No new GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5621.6 | +5621.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3272.0 | +3272.0 |
| `router_full` | 0 | 0.0 | 48 | 2579.1 | +2579.1 |
| `q3kk2560` | 53–56 | 1690.1 | 23–24 | 2167.6 | +477.5 |
| `bf16hcrwx` | 48 | 1584.0 | 48 | 2080.4 | +496.5 |
| `q2moegn` | 96 | 1450.8 | 48 | 1871.5 | +420.7 |
| `q2moedn` | 48 | 854.7 | 48 | 1771.0 | +916.3 |
| `q4kk6144` | 15 | 404.7 | 15 | 1719.8 | +1315.2 |
| `iq4k2560` | 22–24 | 867.5 | 16–18 | 1601.4 | +733.9 |
| `iq4k6144` | 17 | 473.4 | 17 | 1300.8 | +827.4 |
| `q5kk6144` | 7 | 267.9 | 7 | 1189.5 | +921.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1176.2 | +1176.2 |
| `q6kk1536` | 6 | 103.3 | 6 | 1088.5 | +985.2 |
| `q5kk2560` | 1–4 | 619.9 | 1–3 | 627.4 | +7.5 |
| `q2k2560` | 12–14 | 211.6 | 3–4 | 392.0 | +180.4 |
| `xsend` | 123–124 | 1562.5 | 10 | 367.4 | -1195.0 |
| `q4kk2560` | 8–12 | 277.8 | 4–6 | 348.8 | +71.0 |
| `attnqkq` | 0–4 | 77.7 | 2 | 306.0 | +228.3 |
| `q3kk6144` | 3 | 99.2 | 3 | 245.2 | +146.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 190.5 | +190.5 |
| `amax` | 1 | 97.5 | 1 | 97.3 | -0.2 |
| `bf16hcrx` | 49 | 1177.9 | 2 | 74.0 | -1103.9 |
| `hcwsum` | 48 | 2363.4 | 1 | 72.0 | -2291.3 |
| `bf16k2560` | 49–53 | 1380.0 | 1 | 70.7 | -1309.4 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.3 | +65.3 |
| `rms2560` | 3 | 23.4 | 1 | 41.1 | +17.7 |
| `kvgather` | 0–4 | 42.0 | 2 | 39.2 | -2.7 |
| `plegate_norm` | 0 | 0.0 | 1 | 26.3 | +26.3 |
| `q6kk2560` | 0–4 | 26.8 | 0–1 | 16.1 | -10.8 |
| `argmax` | 1 | 10.2 | 1 | 13.0 | +2.8 |
| `qsa_score_select` | 0 | 0.0 | 2 | 11.9 | +11.9 |
| `pleconv` | 1 | 10.0 | 1 | 10.6 | +0.5 |
| `actq256` | 146 | 2800.8 | 0 | 0.0 | -2800.8 |
| `attnmerge` | 0–4 | 15.9 | 0 | 0.0 | -15.9 |
| `attnpartn` | 0–4 | 127.8 | 0 | 0.0 | -127.8 |
| `b32iq4nlacc` | 0–2 | 26.3 | 0 | 0.0 | -26.3 |
| `b32q40acc` | 2–4 | 48.6 | 0 | 0.0 | -48.6 |
| `b32q50acc` | 0–2 | 29.7 | 0 | 0.0 | -29.7 |
| `b32q80acc` | 0–1 | 8.0 | 0 | 0.0 | -8.0 |
| `bf16bias` | 36 | 228.5 | 0 | 0.0 | -228.5 |
| `bf16hcup` | 97 | 2378.3 | 0 | 0.0 | -2378.3 |
| `conv1d` | 36 | 217.3 | 0 | 0.0 | -217.3 |
| `dnnormx` | 36 | 493.7 | 0 | 0.0 | -493.7 |
| `dnstep` | 36 | 622.6 | 0 | 0.0 | -622.6 |
| `ffnact` | 48 | 646.3 | 0 | 0.0 | -646.3 |
| `idxqk` | 0–4 | 27.7 | 0 | 0.0 | -27.7 |
| `idxscore` | 0–4 | 5.5 | 0 | 0.0 | -5.5 |
| `idxsel` | 0–20 | 27.1 | 0 | 0.0 | -27.1 |
| `moesel` | 48 | 215.1 | 0 | 0.0 | -215.1 |
| `plegate` | 1 | 8.4 | 0 | 0.0 | -8.4 |
| `plerows` | 1 | 55.6 | 0 | 0.0 | -55.6 |
| `q2acc640` | 1–4 | 40.6 | 0 | 0.0 | -40.6 |
| `q6kk1536acc` | 18 | 218.3 | 0 | 0.0 | -218.3 |
| `router` | 48 | 1081.0 | 0 | 0.0 | -1081.0 |
| `xcopy` | 0–1 | 27.1 | 0 | 0.0 | -27.1 |
| **All recorded intervals** | 1,177–1,229 | **25026.5** | 498 | **30454.6** | **+5428.1** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3541.3 | 5767.7 | +2226.4 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6712.0 | 4704.1 | -2007.9 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.6 | 11.9 | -20.7 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 31.8 | 67.4 | +35.7 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 321.5 | 1088.5 | +767.0 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5542.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3265.5 |
| `router_full` | 0 | 0.0 | 48 | 2540.5 |
| `q3kk2560` | 54 | 1695.7 | 24 | 2376.7 |
| `bf16hcrwx` | 48 | 1535.0 | 48 | 1493.0 |
| `q2moegn` | 96 | 1527.8 | 48 | 1954.8 |
| `q2moedn` | 48 | 899.1 | 48 | 1913.9 |
| `q4kk6144` | 15 | 420.9 | 15 | 1812.5 |
| `iq4k2560` | 24 | 895.0 | 18 | 1849.3 |
| `iq4k6144` | 17 | 496.6 | 17 | 1496.1 |
| `q5kk6144` | 7 | 276.5 | 7 | 1395.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1241.1 |
| `q6kk1536` | 6 | 109.6 | 6 | 1037.3 |
| `q5kk2560` | 2 | 683.0 | 1 | 656.4 |
| `q2k2560` | 12 | 210.9 | 3 | 330.8 |
| `xsend` | 124 | 1626.1 | 10 | 75.8 |
| `q4kk2560` | 10 | 280.6 | 4 | 318.5 |
| `attnqkq` | 0 | 0.0 | 2 | 305.2 |
| `q3kk6144` | 3 | 104.4 | 3 | 286.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 202.8 |
| `amax` | 1 | 29.7 | 1 | 29.7 |
| `bf16hcrx` | 49 | 1212.4 | 2 | 48.1 |
| `hcwsum` | 48 | 2224.1 | 1 | 5.1 |
| `bf16k2560` | 49 | 1325.1 | 1 | 85.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `rms2560` | 3 | 22.5 | 1 | 15.4 |
| `kvgather` | 0 | 0.0 | 2 | 38.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 11.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 3113.0 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 35.8 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 234.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2308.1 | 0 | 0.0 |
| `conv1d` | 36 | 228.4 | 0 | 0.0 |
| `dnnormx` | 36 | 549.9 | 0 | 0.0 |
| `dnstep` | 36 | 616.4 | 0 | 0.0 |
| `ffnact` | 48 | 652.3 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 207.9 | 0 | 0.0 |
| `plegate` | 1 | 6.1 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 62.5 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 222.2 | 0 | 0.0 |
| `router` | 48 | 1088.5 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5460.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 3293.2 |
| `router_full` | 0 | 0.0 | 48 | 2544.6 |
| `q3kk2560` | 53 | 1677.3 | 24 | 2478.1 |
| `bf16hcrwx` | 48 | 1528.8 | 48 | 1628.2 |
| `q2moegn` | 96 | 1460.2 | 48 | 1870.8 |
| `q2moedn` | 48 | 863.2 | 48 | 1894.4 |
| `q4kk6144` | 15 | 407.6 | 15 | 1764.4 |
| `iq4k2560` | 24 | 874.5 | 16 | 1594.4 |
| `iq4k6144` | 17 | 478.2 | 17 | 1449.0 |
| `q5kk6144` | 7 | 272.4 | 7 | 1378.3 |
| `gdn_full` | 0 | 0.0 | 36 | 1240.1 |
| `q6kk1536` | 6 | 101.4 | 6 | 1029.1 |
| `q5kk2560` | 4 | 632.8 | 1 | 570.4 |
| `q2k2560` | 14 | 189.4 | 3 | 328.7 |
| `xsend` | 123 | 1556.5 | 10 | 73.7 |
| `q4kk2560` | 11 | 303.1 | 6 | 493.6 |
| `attnqkq` | 4 | 245.8 | 2 | 311.3 |
| `q3kk6144` | 3 | 102.4 | 3 | 278.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 202.8 |
| `amax` | 1 | 113.7 | 1 | 117.8 |
| `bf16hcrx` | 49 | 1174.5 | 2 | 78.8 |
| `hcwsum` | 48 | 2335.7 | 1 | 127.0 |
| `bf16k2560` | 53 | 1454.1 | 1 | 81.9 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `rms2560` | 3 | 21.5 | 1 | 18.4 |
| `kvgather` | 4 | 84.0 | 2 | 38.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 15.4 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 10.2 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2110.5 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 256.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 246.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2506.8 | 0 | 0.0 |
| `conv1d` | 36 | 222.2 | 0 | 0.0 |
| `dnnormx` | 36 | 549.9 | 0 | 0.0 |
| `dnstep` | 36 | 618.5 | 0 | 0.0 |
| `ffnact` | 48 | 651.3 | 0 | 0.0 |
| `idxqk` | 4 | 105.5 | 0 | 0.0 |
| `idxscore` | 4 | 10.2 | 0 | 0.0 |
| `idxsel` | 20 | 54.3 | 0 | 0.0 |
| `moesel` | 48 | 220.2 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 62.5 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 219.1 | 0 | 0.0 |
| `router` | 48 | 1082.4 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5881.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3400.7 |
| `router_full` | 0 | 0.0 | 48 | 2609.2 |
| `q3kk2560` | 56 | 1634.3 | 23 | 2017.3 |
| `bf16hcrwx` | 48 | 1815.6 | 48 | 2286.6 |
| `q2moegn` | 96 | 1243.1 | 48 | 1654.8 |
| `q2moedn` | 48 | 737.3 | 48 | 1637.4 |
| `q4kk6144` | 15 | 360.4 | 15 | 1830.9 |
| `iq4k2560` | 23 | 825.3 | 18 | 1658.9 |
| `iq4k6144` | 17 | 423.9 | 17 | 1331.2 |
| `q5kk6144` | 7 | 235.5 | 7 | 1098.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1053.7 |
| `q6kk1536` | 6 | 96.3 | 6 | 1032.2 |
| `q5kk2560` | 1 | 483.3 | 1 | 486.4 |
| `q2k2560` | 14 | 218.1 | 4 | 417.8 |
| `xsend` | 123 | 1451.0 | 10 | 341.0 |
| `q4kk2560` | 8 | 232.4 | 4 | 295.9 |
| `attnqkq` | 0 | 0.0 | 2 | 343.0 |
| `q3kk6144` | 3 | 88.1 | 3 | 245.8 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `amax` | 1 | 205.8 | 1 | 201.7 |
| `bf16hcrx` | 49 | 1113.1 | 2 | 76.8 |
| `hcwsum` | 48 | 2722.8 | 1 | 68.6 |
| `bf16k2560` | 49 | 1317.9 | 1 | 65.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `rms2560` | 3 | 24.6 | 1 | 70.7 |
| `kvgather` | 0 | 0.0 | 2 | 38.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 31.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 9.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3942.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 55.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 24.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 23.6 | 0 | 0.0 |
| `bf16bias` | 36 | 203.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2124.8 | 0 | 0.0 |
| `conv1d` | 36 | 198.7 | 0 | 0.0 |
| `dnnormx` | 36 | 299.0 | 0 | 0.0 |
| `dnstep` | 36 | 605.2 | 0 | 0.0 |
| `ffnact` | 48 | 635.9 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 12.3 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 1 | 14.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 206.8 | 0 | 0.0 |
| `router` | 48 | 1081.3 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5555.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 3181.6 |
| `router_full` | 0 | 0.0 | 48 | 2587.6 |
| `q3kk2560` | 56 | 1820.7 | 23 | 2204.7 |
| `bf16hcrwx` | 48 | 1549.3 | 48 | 2202.6 |
| `q2moegn` | 96 | 1537.0 | 48 | 1943.6 |
| `q2moedn` | 48 | 899.1 | 48 | 1797.1 |
| `q4kk6144` | 15 | 422.9 | 15 | 1774.6 |
| `iq4k2560` | 23 | 945.2 | 16 | 1524.7 |
| `iq4k6144` | 17 | 492.5 | 17 | 1196.0 |
| `q5kk6144` | 7 | 278.5 | 7 | 906.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1109.0 |
| `q6kk1536` | 6 | 106.5 | 6 | 1212.4 |
| `q5kk2560` | 1 | 682.0 | 3 | 906.2 |
| `q2k2560` | 14 | 226.3 | 4 | 438.3 |
| `xsend` | 123 | 1604.6 | 10 | 346.1 |
| `q4kk2560` | 8 | 276.5 | 4 | 336.9 |
| `attnqkq` | 0 | 0.0 | 2 | 360.4 |
| `q3kk6144` | 3 | 100.4 | 3 | 243.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 183.3 |
| `amax` | 1 | 5.1 | 1 | 5.1 |
| `bf16hcrx` | 49 | 1232.9 | 2 | 81.9 |
| `hcwsum` | 48 | 2266.1 | 1 | 71.7 |
| `bf16k2560` | 49 | 1304.6 | 1 | 73.7 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `rms2560` | 3 | 23.6 | 1 | 50.2 |
| `kvgather` | 0 | 0.0 | 2 | 39.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 23.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 14.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2953.2 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 51.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 214.0 | 0 | 0.0 |
| `bf16hcup` | 97 | 2350.1 | 0 | 0.0 |
| `conv1d` | 36 | 207.9 | 0 | 0.0 |
| `dnnormx` | 36 | 436.2 | 0 | 0.0 |
| `dnstep` | 36 | 656.4 | 0 | 0.0 |
| `ffnact` | 48 | 650.2 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 212.0 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 3 | 45.1 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 226.3 | 0 | 0.0 |
| `router` | 48 | 1074.2 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5767.2 |
| `hcupq_x` | 0 | 0.0 | 97 | 3266.6 |
| `router_full` | 0 | 0.0 | 48 | 2596.9 |
| `q3kk2560` | 53 | 1632.3 | 24 | 2016.3 |
| `bf16hcrwx` | 48 | 1517.6 | 48 | 2424.8 |
| `q2moegn` | 96 | 1402.9 | 48 | 1828.9 |
| `q2moedn` | 48 | 826.4 | 48 | 1645.6 |
| `q4kk6144` | 15 | 409.6 | 15 | 1541.1 |
| `iq4k2560` | 24 | 858.1 | 17 | 1500.2 |
| `iq4k6144` | 17 | 476.2 | 17 | 1178.6 |
| `q5kk6144` | 7 | 270.3 | 7 | 1239.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1200.1 |
| `q6kk1536` | 6 | 102.4 | 6 | 1028.1 |
| `q5kk2560` | 2 | 596.0 | 1 | 572.4 |
| `q2k2560` | 13 | 209.9 | 4 | 423.9 |
| `xsend` | 123 | 1589.2 | 10 | 670.7 |
| `q4kk2560` | 12 | 310.3 | 4 | 325.6 |
| `attnqkq` | 4 | 199.7 | 2 | 306.2 |
| `q3kk6144` | 3 | 100.4 | 3 | 200.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `amax` | 1 | 114.7 | 1 | 113.7 |
| `bf16hcrx` | 49 | 1157.1 | 2 | 79.9 |
| `hcwsum` | 48 | 2435.1 | 1 | 66.6 |
| `bf16k2560` | 53 | 1443.8 | 1 | 54.3 |
| `ple_rowsq` | 0 | 0.0 | 1 | 64.5 |
| `rms2560` | 3 | 24.6 | 1 | 44.0 |
| `kvgather` | 4 | 82.9 | 2 | 41.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 42.0 |
| `q6kk2560` | 2 | 51.2 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 11.3 |
| `actq256` | 146 | 2244.6 | 0 | 0.0 |
| `attnmerge` | 4 | 30.7 | 0 | 0.0 |
| `attnpartn` | 4 | 256.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 4 | 73.7 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 233.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2502.7 | 0 | 0.0 |
| `conv1d` | 36 | 224.3 | 0 | 0.0 |
| `dnnormx` | 36 | 576.5 | 0 | 0.0 |
| `dnstep` | 36 | 614.4 | 0 | 0.0 |
| `ffnact` | 48 | 645.1 | 0 | 0.0 |
| `idxqk` | 4 | 42.0 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 54.3 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 8.2 | 0 | 0.0 |
| `plerows` | 1 | 57.3 | 0 | 0.0 |
| `q2acc640` | 2 | 30.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 218.1 | 0 | 0.0 |
| `router` | 48 | 1078.3 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5522.4 |
| `hcupq_x` | 0 | 0.0 | 97 | 3224.6 |
| `router_full` | 0 | 0.0 | 48 | 2595.8 |
| `q3kk2560` | 54 | 1680.4 | 23 | 1912.8 |
| `bf16hcrwx` | 48 | 1557.5 | 48 | 2447.4 |
| `q2moegn` | 96 | 1534.0 | 48 | 1976.3 |
| `q2moedn` | 48 | 903.2 | 48 | 1737.7 |
| `q4kk6144` | 15 | 406.5 | 15 | 1595.4 |
| `iq4k2560` | 22 | 806.9 | 17 | 1480.7 |
| `iq4k6144` | 17 | 473.1 | 17 | 1154.0 |
| `q5kk6144` | 7 | 274.4 | 7 | 1119.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1213.4 |
| `q6kk1536` | 6 | 103.4 | 6 | 1191.9 |
| `q5kk2560` | 4 | 642.0 | 1 | 572.4 |
| `q2k2560` | 13 | 215.0 | 4 | 412.7 |
| `xsend` | 123 | 1547.3 | 10 | 697.3 |
| `q4kk2560` | 9 | 264.2 | 4 | 322.6 |
| `attnqkq` | 4 | 20.5 | 2 | 209.9 |
| `q3kk6144` | 3 | 99.3 | 3 | 216.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 187.4 |
| `amax` | 1 | 115.7 | 1 | 115.7 |
| `bf16hcrx` | 49 | 1177.6 | 2 | 78.8 |
| `hcwsum` | 48 | 2196.5 | 1 | 93.2 |
| `bf16k2560` | 53 | 1434.6 | 1 | 63.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `rms2560` | 3 | 23.6 | 1 | 48.1 |
| `kvgather` | 4 | 85.0 | 2 | 37.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 33.8 |
| `q6kk2560` | 4 | 109.6 | 1 | 96.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2441.2 | 0 | 0.0 |
| `attnmerge` | 4 | 31.7 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 53.2 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 238.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2477.1 | 0 | 0.0 |
| `conv1d` | 36 | 222.2 | 0 | 0.0 |
| `dnnormx` | 36 | 550.9 | 0 | 0.0 |
| `dnstep` | 36 | 624.6 | 0 | 0.0 |
| `ffnact` | 48 | 643.1 | 0 | 0.0 |
| `idxqk` | 4 | 18.4 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 54.3 | 0 | 0.0 |
| `moesel` | 48 | 216.1 | 0 | 0.0 |
| `plegate` | 1 | 9.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 2 | 28.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1081.3 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

