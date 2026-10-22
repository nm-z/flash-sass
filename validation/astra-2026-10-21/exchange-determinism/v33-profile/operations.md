# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `dd7a2119-eeb3-4a45-b1bd-88eedabe5336`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`. Baseline label: `baseline`; candidate label: `kernel-candidate`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: four requests per arm,64 prompt/128 generated tokens,greedy,no MTP,no exchange tracing; final-token profile with post-completion state readbacks. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5792.1 | +5792.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3420.7 | +3420.7 |
| `bf16hcrwx` | 48 | 2093.2 | 48 | 3260.6 | +1167.4 |
| `router_full` | 0 | 0.0 | 48 | 2947.9 | +2947.9 |
| `q2moegn` | 96 | 1741.6 | 48 | 2074.3 | +332.6 |
| `q3kk2560` | 53–56 | 1958.1 | 23–24 | 2042.5 | +84.5 |
| `q2moedn` | 48 | 978.8 | 48 | 1767.9 | +789.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1555.6 | +1555.6 |
| `iq4k2560` | 22–24 | 940.9 | 16–18 | 1500.2 | +559.3 |
| `q4kk6144` | 15 | 584.0 | 15 | 1472.2 | +888.1 |
| `q5kk6144` | 7 | 357.0 | 7 | 1088.8 | +731.8 |
| `q6kk1536` | 6 | 123.6 | 6 | 1063.9 | +940.3 |
| `iq4k6144` | 17 | 560.8 | 17 | 961.6 | +400.8 |
| `q5kk2560` | 1–4 | 628.8 | 1–3 | 630.3 | +1.5 |
| `q4kk2560` | 8–12 | 309.2 | 4–6 | 419.7 | +110.4 |
| `xsend` | 123–124 | 2174.6 | 10 | 363.3 | -1811.3 |
| `q2k2560` | 12–14 | 248.3 | 3–4 | 227.7 | -20.6 |
| `q3kk6144` | 3 | 134.6 | 3 | 198.0 | +63.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 195.9 | +195.9 |
| `attnqkq` | 0–4 | 92.5 | 2 | 148.1 | +55.6 |
| `amax` | 1 | 96.4 | 1 | 97.4 | +1.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 96.3 | +96.3 |
| `bf16hcrx` | 49 | 2415.6 | 2 | 85.5 | -2330.1 |
| `hcwsum` | 48 | 2776.9 | 1 | 80.0 | -2696.9 |
| `kvgather` | 0–4 | 45.6 | 2 | 44.4 | -1.2 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.7 | +35.7 |
| `rms2560` | 3 | 73.0 | 1 | 27.3 | -45.8 |
| `bf16k2560` | 49–53 | 1726.8 | 1 | 27.1 | -1699.7 |
| `q6kk2560` | 0–4 | 31.1 | 0–1 | 18.4 | -12.6 |
| `argmax` | 1 | 10.9 | 1 | 15.6 | +4.6 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.6 | +13.6 |
| `pleconv` | 1 | 13.0 | 1 | 12.8 | -0.2 |
| `actq256` | 146 | 3049.5 | 0 | 0.0 | -3049.5 |
| `attnmerge` | 0–4 | 16.9 | 0 | 0.0 | -16.9 |
| `attnpartn` | 0–4 | 130.4 | 0 | 0.0 | -130.4 |
| `b32iq4nlacc` | 0–2 | 27.8 | 0 | 0.0 | -27.8 |
| `b32q40acc` | 2–4 | 54.8 | 0 | 0.0 | -54.8 |
| `b32q50acc` | 0–2 | 32.8 | 0 | 0.0 | -32.8 |
| `b32q80acc` | 0–1 | 8.5 | 0 | 0.0 | -8.5 |
| `bf16bias` | 36 | 465.8 | 0 | 0.0 | -465.8 |
| `bf16hcup` | 97 | 2764.5 | 0 | 0.0 | -2764.5 |
| `conv1d` | 36 | 297.8 | 0 | 0.0 | -297.8 |
| `dnnormx` | 36 | 685.6 | 0 | 0.0 | -685.6 |
| `dnstep` | 36 | 768.5 | 0 | 0.0 | -768.5 |
| `ffnact` | 48 | 674.7 | 0 | 0.0 | -674.7 |
| `idxqk` | 0–4 | 16.6 | 0 | 0.0 | -16.6 |
| `idxscore` | 0–4 | 6.2 | 0 | 0.0 | -6.2 |
| `idxsel` | 0–20 | 31.8 | 0 | 0.0 | -31.8 |
| `moesel` | 48 | 241.2 | 0 | 0.0 | -241.2 |
| `plegate` | 1 | 14.0 | 0 | 0.0 | -14.0 |
| `plerows` | 1 | 74.6 | 0 | 0.0 | -74.6 |
| `q2acc640` | 1–4 | 49.3 | 0 | 0.0 | -49.3 |
| `q6kk1536acc` | 18 | 277.8 | 0 | 0.0 | -277.8 |
| `router` | 48 | 1132.9 | 0 | 0.0 | -1132.9 |
| `xcopy` | 0–1 | 26.1 | 0 | 0.0 | -26.1 |
| **All recorded intervals** | 1177–1229 | **30963.2** | 498 | **31685.5** | **+722.3** |

The programs have 1178–1230 versus 499 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` + `grwrite_read` | `hcwsum` + `bf16hcrx` + `grwrite_read` | 5192.5 | 5957.7 | +765.1 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 7787.7 | 5268.5 | -2519.2 |
| QSA score + selection | `idxscore` + `idxsel` + `qsa_score_select` | `idxscore` + `idxsel` + `qsa_score_select` | 37.9 | 13.6 | -24.3 |
| All PLE normalizations + gate | `rms2560` + `plegate` + `plegate_norm` | `rms2560` + `plegate` + `plegate_norm` | 87.0 | 63.0 | -24.1 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` + `q6kk1536acc` | 401.4 | 1063.9 | +662.5 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5784.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3478.5 |
| `bf16hcrwx` | 48 | 2108.4 | 48 | 3245.1 |
| `router_full` | 0 | 0.0 | 48 | 2894.8 |
| `q2moegn` | 96 | 1795.1 | 48 | 2171.9 |
| `q3kk2560` | 54 | 2020.4 | 24 | 2103.3 |
| `q2moedn` | 48 | 994.3 | 48 | 1796.1 |
| `gdn_full` | 0 | 0.0 | 36 | 1613.8 |
| `iq4k2560` | 24 | 970.8 | 18 | 1547.3 |
| `q4kk6144` | 15 | 553.0 | 15 | 1340.4 |
| `q5kk6144` | 7 | 350.2 | 7 | 1208.3 |
| `q6kk1536` | 6 | 122.9 | 6 | 986.1 |
| `iq4k6144` | 17 | 592.9 | 17 | 937.0 |
| `q5kk2560` | 2 | 690.2 | 1 | 670.7 |
| `q4kk2560` | 10 | 303.1 | 4 | 387.1 |
| `xsend` | 124 | 2257.9 | 10 | 353.3 |
| `q2k2560` | 12 | 252.9 | 3 | 188.4 |
| `q3kk6144` | 3 | 134.1 | 3 | 195.6 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 207.9 |
| `attnqkq` | 0 | 0.0 | 2 | 187.4 |
| `amax` | 1 | 29.7 | 1 | 22.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 96.3 |
| `bf16hcrx` | 49 | 2426.9 | 2 | 59.4 |
| `hcwsum` | 48 | 2673.7 | 1 | 6.1 |
| `kvgather` | 0 | 0.0 | 2 | 45.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 32.8 |
| `rms2560` | 3 | 71.7 | 1 | 27.6 |
| `bf16k2560` | 49 | 1662.0 | 1 | 30.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 15.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |
| `actq256` | 146 | 3330.0 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 44.0 | 0 | 0.0 |
| `b32q50acc` | 2 | 57.3 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 510.0 | 0 | 0.0 |
| `bf16hcup` | 97 | 2718.7 | 0 | 0.0 |
| `conv1d` | 36 | 304.1 | 0 | 0.0 |
| `dnnormx` | 36 | 709.6 | 0 | 0.0 |
| `dnstep` | 36 | 770.0 | 0 | 0.0 |
| `ffnact` | 48 | 675.8 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 242.7 | 0 | 0.0 |
| `plegate` | 1 | 12.3 | 0 | 0.0 |
| `plerows` | 1 | 74.8 | 0 | 0.0 |
| `q2acc640` | 4 | 74.8 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 278.5 | 0 | 0.0 |
| `router` | 48 | 1130.5 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5688.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3460.1 |
| `bf16hcrwx` | 48 | 2018.3 | 48 | 3202.0 |
| `router_full` | 0 | 0.0 | 48 | 2926.6 |
| `q2moegn` | 96 | 1755.1 | 48 | 2070.5 |
| `q3kk2560` | 53 | 1911.8 | 24 | 2070.5 |
| `q2moedn` | 48 | 1007.6 | 48 | 1819.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1610.8 |
| `iq4k2560` | 24 | 929.8 | 16 | 1437.7 |
| `q4kk6144` | 15 | 606.2 | 15 | 1374.2 |
| `q5kk6144` | 7 | 369.7 | 7 | 1213.4 |
| `q6kk1536` | 6 | 126.0 | 6 | 978.9 |
| `iq4k6144` | 17 | 575.5 | 17 | 940.0 |
| `q5kk2560` | 4 | 651.3 | 1 | 576.5 |
| `q4kk2560` | 11 | 335.9 | 6 | 576.5 |
| `xsend` | 123 | 2154.5 | 10 | 331.8 |
| `q2k2560` | 14 | 223.2 | 3 | 196.6 |
| `q3kk6144` | 3 | 141.3 | 3 | 200.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 209.9 |
| `attnqkq` | 4 | 324.6 | 2 | 184.3 |
| `amax` | 1 | 112.6 | 1 | 119.8 |
| `ple_rowsq` | 0 | 0.0 | 1 | 97.3 |
| `bf16hcrx` | 49 | 2388.0 | 2 | 94.2 |
| `hcwsum` | 48 | 2694.1 | 1 | 139.3 |
| `kvgather` | 4 | 91.1 | 2 | 43.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 73.7 | 1 | 24.6 |
| `bf16k2560` | 53 | 1840.1 | 1 | 24.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 15.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 14.3 |
| `pleconv` | 1 | 12.3 | 1 | 12.3 |
| `actq256` | 146 | 2378.8 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 261.1 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 41.0 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 521.2 | 0 | 0.0 |
| `bf16hcup` | 97 | 2867.2 | 0 | 0.0 |
| `conv1d` | 36 | 304.1 | 0 | 0.0 |
| `dnnormx` | 36 | 734.2 | 0 | 0.0 |
| `dnstep` | 36 | 764.9 | 0 | 0.0 |
| `ffnact` | 48 | 681.0 | 0 | 0.0 |
| `idxqk` | 4 | 45.1 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 63.5 | 0 | 0.0 |
| `moesel` | 48 | 238.6 | 0 | 0.0 |
| `plegate` | 1 | 15.4 | 0 | 0.0 |
| `plerows` | 1 | 74.8 | 0 | 0.0 |
| `q2acc640` | 4 | 73.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 274.4 | 0 | 0.0 |
| `router` | 48 | 1141.8 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 6163.5 |
| `hcupq_x` | 0 | 0.0 | 97 | 3396.6 |
| `bf16hcrwx` | 48 | 2373.6 | 48 | 3371.0 |
| `router_full` | 0 | 0.0 | 48 | 2964.5 |
| `q2moegn` | 96 | 1434.6 | 48 | 1791.0 |
| `q3kk2560` | 56 | 1932.3 | 23 | 1920.0 |
| `q2moedn` | 48 | 824.3 | 48 | 1636.4 |
| `gdn_full` | 0 | 0.0 | 36 | 1465.3 |
| `iq4k2560` | 23 | 885.8 | 18 | 1467.4 |
| `q4kk6144` | 15 | 506.9 | 15 | 1673.2 |
| `q5kk6144` | 7 | 310.3 | 7 | 1010.7 |
| `q6kk1536` | 6 | 122.9 | 6 | 1008.6 |
| `iq4k6144` | 17 | 496.6 | 17 | 1100.8 |
| `q5kk2560` | 1 | 486.4 | 1 | 488.4 |
| `q4kk2560` | 8 | 265.2 | 4 | 361.5 |
| `xsend` | 123 | 2129.9 | 10 | 394.2 |
| `q2k2560` | 14 | 246.8 | 4 | 243.7 |
| `q3kk6144` | 3 | 120.8 | 3 | 199.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 190.5 |
| `attnqkq` | 0 | 0.0 | 2 | 192.5 |
| `amax` | 1 | 204.8 | 1 | 208.9 |
| `ple_rowsq` | 0 | 0.0 | 1 | 95.2 |
| `bf16hcrx` | 49 | 2382.8 | 2 | 85.0 |
| `hcwsum` | 48 | 3267.6 | 1 | 82.9 |
| `kvgather` | 0 | 0.0 | 2 | 45.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `rms2560` | 3 | 74.8 | 1 | 27.6 |
| `bf16k2560` | 49 | 1637.4 | 1 | 26.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 9.2 | 1 | 14.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |
| `actq256` | 146 | 4167.7 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 48.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 63.5 | 0 | 0.0 |
| `b32q50acc` | 1 | 28.7 | 0 | 0.0 |
| `b32q80acc` | 1 | 25.6 | 0 | 0.0 |
| `bf16bias` | 36 | 376.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2500.6 | 0 | 0.0 |
| `conv1d` | 36 | 286.7 | 0 | 0.0 |
| `dnnormx` | 36 | 532.5 | 0 | 0.0 |
| `dnstep` | 36 | 760.8 | 0 | 0.0 |
| `ffnact` | 48 | 668.7 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 244.7 | 0 | 0.0 |
| `plegate` | 1 | 16.4 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 1 | 19.5 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 264.2 | 0 | 0.0 |
| `router` | 48 | 1126.4 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5578.8 |
| `hcupq_x` | 0 | 0.0 | 97 | 3437.6 |
| `bf16hcrwx` | 48 | 2013.2 | 48 | 3248.1 |
| `router_full` | 0 | 0.0 | 48 | 2949.1 |
| `q2moegn` | 96 | 1925.1 | 48 | 2201.6 |
| `q3kk2560` | 56 | 2130.9 | 23 | 2050.0 |
| `q2moedn` | 48 | 1032.2 | 48 | 1852.4 |
| `gdn_full` | 0 | 0.0 | 36 | 1503.2 |
| `iq4k2560` | 23 | 1040.4 | 16 | 1486.8 |
| `q4kk6144` | 15 | 630.8 | 15 | 1570.8 |
| `q5kk6144` | 7 | 378.9 | 7 | 798.7 |
| `q6kk1536` | 6 | 122.9 | 6 | 1191.9 |
| `iq4k6144` | 17 | 556.0 | 17 | 909.3 |
| `q5kk2560` | 1 | 684.0 | 3 | 879.6 |
| `q4kk2560` | 8 | 305.2 | 4 | 401.4 |
| `xsend` | 123 | 2182.1 | 10 | 370.7 |
| `q2k2560` | 14 | 270.3 | 4 | 247.8 |
| `q3kk6144` | 3 | 135.2 | 3 | 191.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 190.5 |
| `attnqkq` | 0 | 0.0 | 2 | 181.2 |
| `amax` | 1 | 6.1 | 1 | 8.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 96.3 |
| `bf16hcrx` | 49 | 2431.0 | 2 | 92.2 |
| `hcwsum` | 48 | 2652.2 | 1 | 78.8 |
| `kvgather` | 0 | 0.0 | 2 | 44.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 37.9 |
| `rms2560` | 3 | 71.7 | 1 | 28.7 |
| `bf16k2560` | 49 | 1589.2 | 1 | 22.5 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 12.3 | 1 | 15.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 13.3 | 1 | 12.3 |
| `actq256` | 146 | 3246.1 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 39.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 55.3 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 379.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2816.0 | 0 | 0.0 |
| `conv1d` | 36 | 289.8 | 0 | 0.0 |
| `dnnormx` | 36 | 645.1 | 0 | 0.0 |
| `dnstep` | 36 | 775.2 | 0 | 0.0 |
| `ffnact` | 48 | 673.8 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 240.6 | 0 | 0.0 |
| `plegate` | 1 | 12.3 | 0 | 0.0 |
| `plerows` | 1 | 74.8 | 0 | 0.0 |
| `q2acc640` | 3 | 54.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 298.0 | 0 | 0.0 |
| `router` | 48 | 1123.3 | 0 | 0.0 |
| `xcopy` | 1 | 30.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5907.5 |
| `hcupq_x` | 0 | 0.0 | 97 | 3382.3 |
| `bf16hcrwx` | 48 | 2030.6 | 48 | 3242.0 |
| `router_full` | 0 | 0.0 | 48 | 2990.1 |
| `q2moegn` | 96 | 1687.6 | 48 | 2014.2 |
| `q3kk2560` | 53 | 1825.8 | 24 | 2102.3 |
| `q2moedn` | 48 | 964.6 | 48 | 1702.9 |
| `gdn_full` | 0 | 0.0 | 36 | 1569.8 |
| `iq4k2560` | 24 | 943.1 | 17 | 1543.2 |
| `q4kk6144` | 15 | 594.9 | 15 | 1411.1 |
| `q5kk6144` | 7 | 366.6 | 7 | 1208.3 |
| `q6kk1536` | 6 | 122.9 | 6 | 1022.0 |
| `iq4k6144` | 17 | 574.5 | 17 | 945.2 |
| `q5kk2560` | 2 | 600.1 | 1 | 583.7 |
| `q4kk2560` | 12 | 351.2 | 4 | 393.2 |
| `xsend` | 123 | 2182.1 | 10 | 368.6 |
| `q2k2560` | 13 | 247.8 | 4 | 241.7 |
| `q3kk6144` | 3 | 137.2 | 3 | 198.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 188.4 |
| `attnqkq` | 4 | 208.9 | 2 | 121.9 |
| `amax` | 1 | 115.7 | 1 | 112.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 96.3 |
| `bf16hcrx` | 49 | 2428.9 | 2 | 92.2 |
| `hcwsum` | 48 | 2826.2 | 1 | 72.7 |
| `kvgather` | 4 | 92.2 | 2 | 44.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 72.7 | 1 | 29.7 |
| `bf16k2560` | 53 | 1814.5 | 1 | 27.6 |
| `q6kk2560` | 2 | 58.4 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 16.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 13.3 | 1 | 12.3 |
| `actq256` | 146 | 2491.4 | 0 | 0.0 |
| `attnmerge` | 4 | 34.8 | 0 | 0.0 |
| `attnpartn` | 4 | 260.1 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 48.1 | 0 | 0.0 |
| `b32q40acc` | 4 | 79.9 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 504.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2855.9 | 0 | 0.0 |
| `conv1d` | 36 | 304.1 | 0 | 0.0 |
| `dnnormx` | 36 | 752.6 | 0 | 0.0 |
| `dnstep` | 36 | 764.9 | 0 | 0.0 |
| `ffnact` | 48 | 675.8 | 0 | 0.0 |
| `idxqk` | 4 | 35.8 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 62.5 | 0 | 0.0 |
| `moesel` | 48 | 237.6 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 74.8 | 0 | 0.0 |
| `q2acc640` | 2 | 36.9 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 274.4 | 0 | 0.0 |
| `router` | 48 | 1140.7 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5630.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 3369.0 |
| `bf16hcrwx` | 48 | 2015.2 | 48 | 3255.3 |
| `router_full` | 0 | 0.0 | 48 | 2962.4 |
| `q2moegn` | 96 | 1852.4 | 48 | 2196.5 |
| `q3kk2560` | 54 | 1927.2 | 23 | 2009.1 |
| `q2moedn` | 48 | 1049.6 | 48 | 1800.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1570.8 |
| `iq4k2560` | 22 | 875.5 | 17 | 1518.6 |
| `q4kk6144` | 15 | 612.4 | 15 | 1463.3 |
| `q5kk6144` | 7 | 366.6 | 7 | 1093.6 |
| `q6kk1536` | 6 | 123.9 | 6 | 1196.0 |
| `iq4k6144` | 17 | 569.3 | 17 | 937.0 |
| `q5kk2560` | 4 | 660.5 | 1 | 582.7 |
| `q4kk2560` | 9 | 294.9 | 4 | 398.3 |
| `xsend` | 123 | 2141.2 | 10 | 361.5 |
| `q2k2560` | 13 | 248.8 | 4 | 247.8 |
| `q3kk6144` | 3 | 139.3 | 3 | 201.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 188.4 |
| `attnqkq` | 4 | 21.5 | 2 | 21.5 |
| `amax` | 1 | 109.6 | 1 | 112.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 96.3 |
| `bf16hcrx` | 49 | 2436.1 | 2 | 90.1 |
| `hcwsum` | 48 | 2547.7 | 1 | 100.4 |
| `kvgather` | 4 | 90.1 | 2 | 45.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 33.8 |
| `rms2560` | 3 | 73.7 | 1 | 25.6 |
| `bf16k2560` | 53 | 1817.6 | 1 | 30.7 |
| `q6kk2560` | 4 | 128.0 | 1 | 110.6 |
| `argmax` | 1 | 11.3 | 1 | 16.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 13.3 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |
| `actq256` | 146 | 2682.9 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 261.1 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 3 | 60.4 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 25.6 | 0 | 0.0 |
| `bf16bias` | 36 | 501.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2828.3 | 0 | 0.0 |
| `conv1d` | 36 | 298.0 | 0 | 0.0 |
| `dnnormx` | 36 | 739.3 | 0 | 0.0 |
| `dnstep` | 36 | 775.2 | 0 | 0.0 |
| `ffnact` | 48 | 672.8 | 0 | 0.0 |
| `idxqk` | 4 | 18.4 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 64.5 | 0 | 0.0 |
| `moesel` | 48 | 242.7 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 74.8 | 0 | 0.0 |
| `q2acc640` | 2 | 36.9 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 277.5 | 0 | 0.0 |
| `router` | 48 | 1134.6 | 0 | 0.0 |
| `xcopy` | 1 | 29.7 | 0 | 0.0 |

