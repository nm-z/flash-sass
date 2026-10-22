# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `b761ed99-563c-45d7-ae52-16665c5c6bd5`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`. Baseline label: `phase-baseline`; candidate label: `split-candidate`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: 64 prompt /128 generated; four requests per arm; eight FFN GR stages replicated; both pass same-kernel repeatability. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5795.0 | 47 | 5858.3 | +63.3 |
| `bf16hcrwx` | 48 | 3265.0 | 48 | 4188.5 | +923.5 |
| `hcupq_x` | 97 | 3458.7 | 97 | 3769.5 | +310.8 |
| `router_full` | 48 | 2950.7 | 48 | 2973.9 | +23.2 |
| `q2moegn` | 48 | 2122.3 | 48 | 2110.3 | -12.0 |
| `q3kk2560` | 23–24 | 2038.1 | 23–24 | 2093.4 | +55.3 |
| `q2moedn` | 48 | 1793.5 | 48 | 1779.7 | -13.8 |
| `iq4k2560` | 16–18 | 1510.4 | 16–18 | 1546.9 | +36.5 |
| `gdn_full` | 36 | 1539.1 | 36 | 1539.4 | +0.4 |
| `q4kk6144` | 15 | 1507.5 | 15 | 1489.8 | -17.8 |
| `q5kk6144` | 7 | 1099.6 | 7 | 1080.3 | -19.2 |
| `q6kk1536` | 6 | 1077.4 | 6 | 1071.5 | -6.0 |
| `iq4k6144` | 17 | 970.4 | 17 | 992.4 | +22.0 |
| `q5kk2560` | 1–3 | 629.2 | 1–3 | 627.4 | -1.9 |
| `q4kk2560` | 4–6 | 420.0 | 4–6 | 427.4 | +7.4 |
| `xsend` | 10 | 352.2 | 10 | 350.9 | -1.4 |
| `q2k2560` | 3–4 | 222.0 | 3–4 | 230.9 | +8.9 |
| `q3kk6144` | 3 | 204.6 | 3 | 201.9 | -2.7 |
| `qsa_part_merge` | 2 | 197.3 | 2 | 195.2 | -2.0 |
| `attnqkq` | 2 | 149.4 | 2 | 147.6 | -1.8 |
| `amax` | 1 | 96.2 | 1 | 102.1 | +5.8 |
| `hcwsum` | 1 | 85.9 | 1 | 91.8 | +6.0 |
| `bf16hcrx` | 2 | 85.0 | 2 | 83.1 | -1.9 |
| `ple_rowsq` | 1 | 88.9 | 1 | 81.6 | -7.4 |
| `kvgather` | 2 | 45.1 | 2 | 45.6 | +0.5 |
| `plegate_norm` | 1 | 34.8 | 1 | 36.2 | +1.4 |
| `bf16k2560` | 1 | 24.7 | 1 | 29.9 | +5.1 |
| `rms2560` | 1 | 27.1 | 1 | 25.3 | -1.9 |
| `q6kk2560` | 0–1 | 18.4 | 0–1 | 17.9 | -0.5 |
| `argmax` | 1 | 16.1 | 1 | 15.7 | -0.4 |
| `qsa_score_select` | 2 | 14.5 | 2 | 13.8 | -0.7 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 | +0.0 |
| **All recorded intervals** | 498 | **31852.6** | 498 | **33231.5** | **+1378.9** |

The programs have 499 versus 499 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` + `grwrite_read` | `hcwsum` + `bf16hcrx` + `grwrite_read` | 5965.8 | 6033.2 | +67.4 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 5284.0 | 5585.8 | +301.7 |
| QSA score + selection | `idxscore` + `idxsel` + `qsa_score_select` | `idxscore` + `idxsel` + `qsa_score_select` | 14.5 | 13.8 | -0.7 |
| All PLE normalizations + gate | `rms2560` + `plegate` + `plegate_norm` | `rms2560` + `plegate` + `plegate_norm` | 61.9 | 61.4 | -0.5 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` + `q6kk1536acc` | 1077.4 | 1071.5 | -6.0 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5801.0 | 47 | 5771.3 |
| `bf16hcrwx` | 48 | 3264.5 | 48 | 4207.6 |
| `hcupq_x` | 97 | 3466.2 | 97 | 3726.3 |
| `router_full` | 48 | 2877.4 | 48 | 2915.3 |
| `q2moegn` | 48 | 2264.1 | 48 | 2258.9 |
| `q3kk2560` | 24 | 2129.9 | 24 | 2197.5 |
| `q2moedn` | 48 | 1804.3 | 48 | 1862.7 |
| `iq4k2560` | 18 | 1580.0 | 18 | 1644.5 |
| `gdn_full` | 36 | 1587.2 | 36 | 1586.2 |
| `q4kk6144` | 15 | 1364.0 | 15 | 1355.8 |
| `q5kk6144` | 7 | 1224.7 | 7 | 1194.0 |
| `q6kk1536` | 6 | 996.4 | 6 | 992.3 |
| `iq4k6144` | 17 | 927.7 | 17 | 946.2 |
| `q5kk2560` | 1 | 666.6 | 1 | 664.6 |
| `q4kk2560` | 4 | 398.3 | 4 | 399.4 |
| `xsend` | 10 | 334.8 | 10 | 338.9 |
| `q2k2560` | 3 | 187.4 | 3 | 184.3 |
| `q3kk6144` | 3 | 191.5 | 3 | 196.6 |
| `qsa_part_merge` | 2 | 209.9 | 2 | 207.9 |
| `attnqkq` | 2 | 190.5 | 2 | 186.4 |
| `amax` | 1 | 24.6 | 1 | 32.8 |
| `hcwsum` | 1 | 10.2 | 1 | 9.2 |
| `bf16hcrx` | 2 | 58.4 | 2 | 60.4 |
| `ple_rowsq` | 1 | 88.1 | 1 | 84.0 |
| `kvgather` | 2 | 47.1 | 2 | 47.1 |
| `plegate_norm` | 1 | 33.8 | 1 | 34.8 |
| `bf16k2560` | 1 | 27.6 | 1 | 32.8 |
| `rms2560` | 1 | 25.6 | 1 | 24.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 16.4 | 1 | 16.4 |
| `qsa_score_select` | 2 | 14.3 | 2 | 14.3 |
| `pleconv` | 1 | 14.3 | 1 | 14.3 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5709.8 | 47 | 5777.4 |
| `bf16hcrwx` | 48 | 3182.6 | 48 | 4120.6 |
| `hcupq_x` | 97 | 3527.7 | 97 | 3824.6 |
| `router_full` | 48 | 2915.3 | 48 | 2993.2 |
| `q2moegn` | 48 | 2127.9 | 48 | 2099.2 |
| `q3kk2560` | 24 | 2063.4 | 24 | 2145.3 |
| `q2moedn` | 48 | 1818.6 | 48 | 1776.6 |
| `iq4k2560` | 16 | 1438.7 | 16 | 1487.9 |
| `gdn_full` | 36 | 1583.1 | 36 | 1565.7 |
| `q4kk6144` | 15 | 1419.3 | 15 | 1373.2 |
| `q5kk6144` | 7 | 1230.8 | 7 | 1195.0 |
| `q6kk1536` | 6 | 995.3 | 6 | 984.1 |
| `iq4k6144` | 17 | 958.5 | 17 | 996.4 |
| `q5kk2560` | 1 | 585.7 | 1 | 582.7 |
| `q4kk2560` | 6 | 572.4 | 6 | 598.0 |
| `xsend` | 10 | 331.8 | 10 | 316.4 |
| `q2k2560` | 3 | 185.3 | 3 | 193.5 |
| `q3kk6144` | 3 | 207.9 | 3 | 197.6 |
| `qsa_part_merge` | 2 | 208.9 | 2 | 203.8 |
| `attnqkq` | 2 | 186.4 | 2 | 184.3 |
| `amax` | 1 | 108.5 | 1 | 116.7 |
| `hcwsum` | 1 | 148.5 | 1 | 158.7 |
| `bf16hcrx` | 2 | 91.1 | 2 | 90.1 |
| `ple_rowsq` | 1 | 90.1 | 1 | 80.9 |
| `kvgather` | 2 | 45.1 | 2 | 46.1 |
| `plegate_norm` | 1 | 35.8 | 1 | 35.8 |
| `bf16k2560` | 1 | 22.5 | 1 | 30.7 |
| `rms2560` | 1 | 25.6 | 1 | 23.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 15.4 | 1 | 15.4 |
| `qsa_score_select` | 2 | 14.3 | 2 | 12.3 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 6198.3 | 47 | 6133.8 |
| `bf16hcrwx` | 48 | 3402.8 | 48 | 4302.8 |
| `hcupq_x` | 97 | 3464.2 | 97 | 3788.8 |
| `router_full` | 48 | 2967.6 | 48 | 3080.2 |
| `q2moegn` | 48 | 1807.4 | 48 | 1848.3 |
| `q3kk2560` | 23 | 1896.4 | 23 | 1935.4 |
| `q2moedn` | 48 | 1663.0 | 48 | 1627.1 |
| `iq4k2560` | 18 | 1460.2 | 18 | 1503.2 |
| `gdn_full` | 36 | 1462.3 | 36 | 1481.7 |
| `q4kk6144` | 15 | 1701.9 | 15 | 1712.1 |
| `q5kk6144` | 7 | 1000.4 | 7 | 1029.1 |
| `q6kk1536` | 6 | 1023.0 | 6 | 1022.0 |
| `iq4k6144` | 17 | 1093.6 | 17 | 1051.6 |
| `q5kk2560` | 1 | 490.5 | 1 | 487.4 |
| `q4kk2560` | 4 | 357.4 | 4 | 365.6 |
| `xsend` | 10 | 385.0 | 10 | 377.9 |
| `q2k2560` | 4 | 232.4 | 4 | 247.8 |
| `q3kk6144` | 3 | 218.1 | 3 | 220.2 |
| `qsa_part_merge` | 2 | 192.5 | 2 | 188.4 |
| `attnqkq` | 2 | 190.5 | 2 | 184.3 |
| `amax` | 1 | 204.8 | 1 | 213.0 |
| `hcwsum` | 1 | 92.2 | 1 | 89.1 |
| `bf16hcrx` | 2 | 87.0 | 2 | 88.1 |
| `ple_rowsq` | 1 | 89.1 | 1 | 80.9 |
| `kvgather` | 2 | 44.0 | 2 | 45.1 |
| `plegate_norm` | 1 | 34.8 | 1 | 37.9 |
| `bf16k2560` | 1 | 24.6 | 1 | 28.7 |
| `rms2560` | 1 | 30.7 | 1 | 26.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 15.4 | 1 | 14.3 |
| `qsa_score_select` | 2 | 14.3 | 2 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5553.2 | 47 | 5741.6 |
| `bf16hcrwx` | 48 | 3246.1 | 48 | 4155.4 |
| `hcupq_x` | 97 | 3476.5 | 97 | 3793.9 |
| `router_full` | 48 | 2939.9 | 48 | 2844.7 |
| `q2moegn` | 48 | 2253.8 | 48 | 2232.3 |
| `q3kk2560` | 23 | 2036.7 | 23 | 2077.7 |
| `q2moedn` | 48 | 1905.7 | 48 | 1915.9 |
| `iq4k2560` | 16 | 1490.9 | 16 | 1483.8 |
| `gdn_full` | 36 | 1510.4 | 36 | 1523.7 |
| `q4kk6144` | 15 | 1592.3 | 15 | 1597.4 |
| `q5kk6144` | 7 | 810.0 | 7 | 792.6 |
| `q6kk1536` | 6 | 1211.4 | 6 | 1214.5 |
| `iq4k6144` | 17 | 924.7 | 17 | 952.3 |
| `q5kk2560` | 3 | 876.5 | 3 | 874.5 |
| `q4kk2560` | 4 | 405.5 | 4 | 403.5 |
| `xsend` | 10 | 355.3 | 10 | 360.4 |
| `q2k2560` | 4 | 241.7 | 4 | 254.0 |
| `q3kk6144` | 3 | 199.7 | 3 | 194.6 |
| `qsa_part_merge` | 2 | 190.5 | 2 | 190.5 |
| `attnqkq` | 2 | 186.4 | 2 | 187.4 |
| `amax` | 1 | 7.2 | 1 | 8.2 |
| `hcwsum` | 1 | 85.0 | 1 | 89.1 |
| `bf16hcrx` | 2 | 92.2 | 2 | 88.1 |
| `ple_rowsq` | 1 | 89.1 | 1 | 80.9 |
| `kvgather` | 2 | 45.1 | 2 | 45.1 |
| `plegate_norm` | 1 | 33.8 | 1 | 35.8 |
| `bf16k2560` | 1 | 26.6 | 1 | 32.8 |
| `rms2560` | 1 | 26.6 | 1 | 22.5 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 16.4 | 1 | 15.4 |
| `qsa_score_select` | 2 | 14.3 | 2 | 13.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5901.3 | 47 | 6003.7 |
| `bf16hcrwx` | 48 | 3249.2 | 48 | 4163.6 |
| `hcupq_x` | 97 | 3402.8 | 97 | 3737.6 |
| `router_full` | 48 | 3034.1 | 48 | 3037.2 |
| `q2moegn` | 48 | 2057.2 | 48 | 2034.7 |
| `q3kk2560` | 24 | 2097.2 | 24 | 2160.6 |
| `q2moedn` | 48 | 1708.0 | 48 | 1675.3 |
| `iq4k2560` | 17 | 1543.2 | 17 | 1591.3 |
| `gdn_full` | 36 | 1554.4 | 36 | 1539.1 |
| `q4kk6144` | 15 | 1454.1 | 15 | 1423.4 |
| `q5kk6144` | 7 | 1230.8 | 7 | 1197.1 |
| `q6kk1536` | 6 | 1032.2 | 6 | 1022.0 |
| `iq4k6144` | 17 | 966.7 | 17 | 998.4 |
| `q5kk2560` | 1 | 577.5 | 1 | 575.5 |
| `q4kk2560` | 4 | 390.1 | 4 | 401.4 |
| `xsend` | 10 | 352.3 | 10 | 352.3 |
| `q2k2560` | 4 | 239.6 | 4 | 252.9 |
| `q3kk6144` | 3 | 205.8 | 3 | 199.7 |
| `qsa_part_merge` | 2 | 190.5 | 2 | 189.4 |
| `attnqkq` | 2 | 122.9 | 2 | 120.8 |
| `amax` | 1 | 116.7 | 1 | 122.9 |
| `hcwsum` | 1 | 74.8 | 1 | 88.1 |
| `bf16hcrx` | 2 | 91.1 | 2 | 87.0 |
| `ple_rowsq` | 1 | 89.1 | 1 | 80.9 |
| `kvgather` | 2 | 45.1 | 2 | 45.1 |
| `plegate_norm` | 1 | 35.8 | 1 | 35.8 |
| `bf16k2560` | 1 | 21.5 | 1 | 30.7 |
| `rms2560` | 1 | 29.7 | 1 | 23.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 16.4 | 1 | 16.4 |
| `qsa_score_select` | 2 | 14.3 | 2 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 47 | 5606.4 | 47 | 5722.1 |
| `bf16hcrwx` | 48 | 3245.1 | 48 | 4181.0 |
| `hcupq_x` | 97 | 3415.0 | 97 | 3745.8 |
| `router_full` | 48 | 2969.6 | 48 | 2972.7 |
| `q2moegn` | 48 | 2223.1 | 48 | 2188.3 |
| `q3kk2560` | 23 | 2005.0 | 23 | 2043.9 |
| `q2moedn` | 48 | 1861.6 | 48 | 1820.7 |
| `iq4k2560` | 17 | 1549.3 | 17 | 1570.8 |
| `gdn_full` | 36 | 1537.0 | 36 | 1540.1 |
| `q4kk6144` | 15 | 1513.5 | 15 | 1476.6 |
| `q5kk6144` | 7 | 1100.8 | 7 | 1074.2 |
| `q6kk1536` | 6 | 1206.3 | 6 | 1194.0 |
| `iq4k6144` | 17 | 951.3 | 17 | 1009.7 |
| `q5kk2560` | 1 | 578.6 | 1 | 579.6 |
| `q4kk2560` | 4 | 396.3 | 4 | 396.3 |
| `xsend` | 10 | 354.3 | 10 | 359.4 |
| `q2k2560` | 4 | 245.8 | 4 | 252.9 |
| `q3kk6144` | 3 | 204.8 | 3 | 202.8 |
| `qsa_part_merge` | 2 | 191.5 | 2 | 191.5 |
| `attnqkq` | 2 | 19.5 | 2 | 22.5 |
| `amax` | 1 | 115.7 | 1 | 118.8 |
| `hcwsum` | 1 | 104.4 | 1 | 116.7 |
| `bf16hcrx` | 2 | 90.1 | 2 | 85.0 |
| `ple_rowsq` | 1 | 88.1 | 1 | 81.9 |
| `kvgather` | 2 | 44.0 | 2 | 45.1 |
| `plegate_norm` | 1 | 34.8 | 1 | 36.9 |
| `bf16k2560` | 1 | 25.6 | 1 | 23.6 |
| `rms2560` | 1 | 24.6 | 1 | 30.7 |
| `q6kk2560` | 1 | 110.6 | 1 | 107.5 |
| `argmax` | 1 | 16.4 | 1 | 16.4 |
| `qsa_score_select` | 2 | 15.4 | 2 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 12.3 |

