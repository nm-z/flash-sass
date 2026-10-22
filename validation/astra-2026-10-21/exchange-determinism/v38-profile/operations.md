# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `45806366-8fdb-40fa-9649-887225f752c7`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`. Baseline label: `baseline`; candidate label: `kernel-candidate`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: 4096 prompt / 16 generated; two requests per arm; candidate fails state repeatability. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5829.1 | +5829.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3482.0 | +3482.0 |
| `bf16hcrwx` | 48 | 2085.0 | 48 | 3261.8 | +1176.8 |
| `router_full` | 0 | 0.0 | 48 | 3002.7 | +3002.7 |
| `q4kk6144` | 15 | 585.4 | 15 | 2440.4 | +1855.0 |
| `q2moegn` | 96 | 1770.7 | 48 | 2122.2 | +351.6 |
| `q3kk2560` | 53–56 | 1985.5 | 23–24 | 2043.4 | +57.8 |
| `q5kk6144` | 7 | 354.8 | 7 | 1997.5 | +1642.7 |
| `q6kk1536` | 6 | 124.9 | 6 | 1994.6 | +1869.7 |
| `q2moedn` | 48 | 995.0 | 48 | 1804.3 | +809.3 |
| `iq4k2560` | 22–24 | 949.2 | 16–18 | 1509.2 | +560.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1508.9 | +1508.9 |
| `iq4k6144` | 17 | 567.0 | 17 | 1024.9 | +457.9 |
| `q5kk2560` | 1–4 | 625.0 | 1–3 | 632.0 | +7.0 |
| `q4kk2560` | 8–12 | 310.6 | 4–6 | 421.7 | +111.1 |
| `xsend` | 123–124 | 2175.8 | 10 | 364.2 | -1811.6 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 350.4 | +350.4 |
| `kvgather` | 0–4 | 348.3 | 2 | 343.9 | -4.4 |
| `q2k2560` | 12–14 | 251.1 | 3–4 | 231.4 | -19.6 |
| `q3kk6144` | 3 | 137.9 | 3 | 207.5 | +69.6 |
| `attnqkq` | 0–4 | 92.7 | 2 | 148.1 | +55.5 |
| `qsa_score_select` | 0 | 0.0 | 2 | 110.4 | +110.4 |
| `amax` | 1 | 93.2 | 1 | 102.1 | +8.9 |
| `hcwsum` | 48 | 2744.7 | 1 | 90.6 | -2654.1 |
| `bf16hcrx` | 49 | 2425.7 | 2 | 86.5 | -2339.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.5 | +80.5 |
| `plegate_norm` | 0 | 0.0 | 1 | 37.7 | +37.7 |
| `bf16k2560` | 49–53 | 1750.7 | 1 | 32.1 | -1718.6 |
| `rms2560` | 3 | 73.5 | 1 | 28.8 | -44.7 |
| `q6kk2560` | 0–4 | 31.1 | 0–1 | 18.1 | -13.0 |
| `argmax` | 1 | 11.1 | 1 | 15.5 | +4.4 |
| `pleconv` | 1 | 13.0 | 1 | 13.3 | +0.3 |
| `actq256` | 146 | 5784.6 | 0 | 0.0 | -5784.6 |
| `attnmerge` | 0–4 | 17.2 | 0 | 0.0 | -17.2 |
| `attnpartn` | 0–4 | 280.6 | 0 | 0.0 | -280.6 |
| `b32iq4nlacc` | 0–2 | 28.0 | 0 | 0.0 | -28.0 |
| `b32q40acc` | 2–4 | 55.5 | 0 | 0.0 | -55.5 |
| `b32q50acc` | 0–2 | 32.6 | 0 | 0.0 | -32.6 |
| `b32q80acc` | 0–1 | 8.7 | 0 | 0.0 | -8.7 |
| `bf16bias` | 36 | 475.3 | 0 | 0.0 | -475.3 |
| `bf16hcup` | 97 | 2774.5 | 0 | 0.0 | -2774.5 |
| `conv1d` | 36 | 302.4 | 0 | 0.0 | -302.4 |
| `dnnormx` | 36 | 668.5 | 0 | 0.0 | -668.5 |
| `dnstep` | 36 | 778.9 | 0 | 0.0 | -778.9 |
| `ffnact` | 48 | 686.6 | 0 | 0.0 | -686.6 |
| `idxqk` | 0–4 | 17.4 | 0 | 0.0 | -17.4 |
| `idxscore` | 0–4 | 31.8 | 0 | 0.0 | -31.8 |
| `idxsel` | 0–20 | 91.6 | 0 | 0.0 | -91.6 |
| `moesel` | 48 | 250.6 | 0 | 0.0 | -250.6 |
| `plegate` | 1 | 14.3 | 0 | 0.0 | -14.3 |
| `plerows` | 1 | 89.8 | 0 | 0.0 | -89.8 |
| `q2acc640` | 1–4 | 49.5 | 0 | 0.0 | -49.5 |
| `q6kk1536acc` | 18 | 281.2 | 0 | 0.0 | -281.2 |
| `router` | 48 | 1139.2 | 0 | 0.0 | -1139.2 |
| `xcopy` | 0–1 | 27.1 | 0 | 0.0 | -27.1 |
| **All recorded intervals** | 1177–1229 | **34387.8** | 498 | **35335.8** | **+948.0** |

The programs have 1178–1230 versus 499 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` + `grwrite_read` | `hcwsum` + `bf16hcrx` + `grwrite_read` | 5170.3 | 6006.3 | +835.9 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 10696.5 | 5421.7 | -5274.8 |
| QSA score + selection | `idxscore` + `idxsel` + `qsa_score_select` | `idxscore` + `idxsel` + `qsa_score_select` | 123.4 | 110.4 | -13.0 |
| All PLE normalizations + gate | `rms2560` + `plegate` + `plegate_norm` | `rms2560` + `plegate` + `plegate_norm` | 87.9 | 66.5 | -21.3 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` + `q6kk1536acc` | 406.2 | 1994.6 | +1588.4 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5550.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3351.6 |
| `bf16hcrwx` | 48 | 2132.0 | 48 | 3301.4 |
| `router_full` | 0 | 0.0 | 48 | 2995.2 |
| `q4kk6144` | 15 | 556.0 | 15 | 2185.2 |
| `q2moegn` | 96 | 1953.8 | 48 | 2447.4 |
| `q3kk2560` | 54 | 2033.7 | 24 | 2167.8 |
| `q5kk6144` | 7 | 349.2 | 7 | 2291.7 |
| `q6kk1536` | 6 | 127.0 | 6 | 1806.3 |
| `q2moedn` | 48 | 1069.1 | 48 | 1940.5 |
| `iq4k2560` | 24 | 986.1 | 18 | 1622.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1578.0 |
| `iq4k6144` | 17 | 601.1 | 17 | 923.6 |
| `q5kk2560` | 2 | 695.3 | 1 | 671.7 |
| `q4kk2560` | 10 | 304.1 | 4 | 402.4 |
| `xsend` | 124 | 2298.9 | 10 | 354.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 370.7 |
| `kvgather` | 0 | 0.0 | 2 | 345.1 |
| `q2k2560` | 12 | 243.7 | 3 | 190.5 |
| `q3kk6144` | 3 | 136.2 | 3 | 191.5 |
| `attnqkq` | 0 | 0.0 | 2 | 189.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 115.7 |
| `amax` | 1 | 16.4 | 1 | 30.7 |
| `hcwsum` | 48 | 2410.5 | 1 | 10.2 |
| `bf16hcrx` | 49 | 2427.9 | 2 | 63.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 82.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `bf16k2560` | 49 | 1687.6 | 1 | 34.8 |
| `rms2560` | 3 | 74.8 | 1 | 29.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 12.3 | 1 | 15.4 |
| `pleconv` | 1 | 13.3 | 1 | 14.3 |
| `actq256` | 146 | 6589.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 43.0 | 0 | 0.0 |
| `b32q50acc` | 2 | 57.3 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 523.3 | 0 | 0.0 |
| `bf16hcup` | 97 | 2715.6 | 0 | 0.0 |
| `conv1d` | 36 | 313.3 | 0 | 0.0 |
| `dnnormx` | 36 | 687.1 | 0 | 0.0 |
| `dnstep` | 36 | 781.3 | 0 | 0.0 |
| `ffnact` | 48 | 698.4 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 252.9 | 0 | 0.0 |
| `plegate` | 1 | 12.3 | 0 | 0.0 |
| `plerows` | 1 | 90.1 | 0 | 0.0 |
| `q2acc640` | 4 | 75.8 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 283.6 | 0 | 0.0 |
| `router` | 48 | 1139.7 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 6242.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3559.4 |
| `bf16hcrwx` | 48 | 2024.4 | 48 | 3193.9 |
| `router_full` | 0 | 0.0 | 48 | 2946.0 |
| `q4kk6144` | 15 | 602.1 | 15 | 2270.2 |
| `q2moegn` | 96 | 1501.2 | 48 | 1813.5 |
| `q3kk2560` | 53 | 1946.6 | 24 | 2052.1 |
| `q5kk6144` | 7 | 371.7 | 7 | 2307.1 |
| `q6kk1536` | 6 | 124.9 | 6 | 1827.8 |
| `q2moedn` | 48 | 868.4 | 48 | 1702.9 |
| `iq4k2560` | 24 | 947.2 | 16 | 1430.5 |
| `gdn_full` | 0 | 0.0 | 36 | 1524.7 |
| `iq4k6144` | 17 | 582.7 | 17 | 1035.3 |
| `q5kk2560` | 4 | 647.2 | 1 | 582.7 |
| `q4kk2560` | 11 | 340.0 | 6 | 582.7 |
| `xsend` | 123 | 2179.1 | 10 | 342.0 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 359.4 |
| `kvgather` | 4 | 692.2 | 2 | 344.1 |
| `q2k2560` | 14 | 228.4 | 3 | 194.6 |
| `q3kk6144` | 3 | 143.4 | 3 | 215.0 |
| `attnqkq` | 4 | 322.6 | 2 | 186.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 110.6 |
| `amax` | 1 | 108.5 | 1 | 117.8 |
| `hcwsum` | 48 | 3059.7 | 1 | 107.5 |
| `bf16hcrx` | 49 | 2402.3 | 2 | 89.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `bf16k2560` | 53 | 1881.1 | 1 | 32.8 |
| `rms2560` | 3 | 73.7 | 1 | 24.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 16.4 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |
| `actq256` | 146 | 4576.3 | 0 | 0.0 |
| `attnmerge` | 4 | 34.8 | 0 | 0.0 |
| `attnpartn` | 4 | 559.1 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 41.0 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 513.0 | 0 | 0.0 |
| `bf16hcup` | 97 | 2868.2 | 0 | 0.0 |
| `conv1d` | 36 | 306.2 | 0 | 0.0 |
| `dnnormx` | 36 | 728.1 | 0 | 0.0 |
| `dnstep` | 36 | 776.2 | 0 | 0.0 |
| `ffnact` | 48 | 691.2 | 0 | 0.0 |
| `idxqk` | 4 | 49.2 | 0 | 0.0 |
| `idxscore` | 4 | 62.5 | 0 | 0.0 |
| `idxsel` | 20 | 184.3 | 0 | 0.0 |
| `moesel` | 48 | 249.9 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 89.1 | 0 | 0.0 |
| `q2acc640` | 4 | 73.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 274.4 | 0 | 0.0 |
| `router` | 48 | 1141.8 | 0 | 0.0 |
| `xcopy` | 1 | 33.8 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5954.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3471.4 |
| `bf16hcrwx` | 48 | 2352.1 | 48 | 3378.2 |
| `router_full` | 0 | 0.0 | 48 | 3021.8 |
| `q4kk6144` | 15 | 507.9 | 15 | 2811.9 |
| `q2moegn` | 96 | 1617.9 | 48 | 1976.3 |
| `q3kk2560` | 56 | 1950.7 | 23 | 1919.0 |
| `q5kk6144` | 7 | 306.2 | 7 | 1813.5 |
| `q6kk1536` | 6 | 124.9 | 6 | 1844.2 |
| `q2moedn` | 48 | 927.7 | 48 | 1778.7 |
| `iq4k2560` | 23 | 883.7 | 18 | 1451.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1466.4 |
| `iq4k6144` | 17 | 503.8 | 17 | 1135.6 |
| `q5kk2560` | 1 | 478.2 | 1 | 489.5 |
| `q4kk2560` | 8 | 268.3 | 4 | 360.4 |
| `xsend` | 123 | 2162.7 | 10 | 396.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 343.0 |
| `kvgather` | 0 | 0.0 | 2 | 354.3 |
| `q2k2560` | 14 | 242.7 | 4 | 246.8 |
| `q3kk6144` | 3 | 120.8 | 3 | 217.1 |
| `attnqkq` | 0 | 0.0 | 2 | 188.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 206.8 | 1 | 216.1 |
| `hcwsum` | 48 | 2970.6 | 1 | 104.4 |
| `bf16hcrx` | 49 | 2412.5 | 2 | 88.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 78.8 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `bf16k2560` | 49 | 1659.9 | 1 | 28.7 |
| `rms2560` | 3 | 73.7 | 1 | 33.8 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |
| `actq256` | 146 | 7452.7 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 49.2 | 0 | 0.0 |
| `b32q40acc` | 3 | 65.5 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 25.6 | 0 | 0.0 |
| `bf16bias` | 36 | 397.3 | 0 | 0.0 |
| `bf16hcup` | 97 | 2509.8 | 0 | 0.0 |
| `conv1d` | 36 | 290.8 | 0 | 0.0 |
| `dnnormx` | 36 | 499.7 | 0 | 0.0 |
| `dnstep` | 36 | 774.1 | 0 | 0.0 |
| `ffnact` | 48 | 678.9 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 249.9 | 0 | 0.0 |
| `plegate` | 1 | 17.4 | 0 | 0.0 |
| `plerows` | 1 | 89.1 | 0 | 0.0 |
| `q2acc640` | 1 | 19.5 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 271.4 | 0 | 0.0 |
| `router` | 48 | 1140.7 | 0 | 0.0 |
| `xcopy` | 1 | 30.7 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5646.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3544.1 |
| `bf16hcrwx` | 48 | 1998.8 | 48 | 3225.6 |
| `router_full` | 0 | 0.0 | 48 | 3008.5 |
| `q4kk6144` | 15 | 632.8 | 15 | 2704.4 |
| `q2moegn` | 96 | 1922.0 | 48 | 2203.6 |
| `q3kk2560` | 56 | 2183.2 | 23 | 2038.8 |
| `q5kk6144` | 7 | 379.9 | 7 | 1357.8 |
| `q6kk1536` | 6 | 124.9 | 6 | 2316.3 |
| `q2moedn` | 48 | 1041.4 | 48 | 1845.2 |
| `iq4k2560` | 23 | 1053.7 | 16 | 1485.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1495.0 |
| `iq4k6144` | 17 | 560.1 | 17 | 974.8 |
| `q5kk2560` | 1 | 677.9 | 3 | 884.7 |
| `q4kk2560` | 8 | 302.1 | 4 | 400.4 |
| `xsend` | 123 | 2145.3 | 10 | 373.8 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 340.0 |
| `kvgather` | 0 | 0.0 | 2 | 334.8 |
| `q2k2560` | 14 | 275.5 | 4 | 258.0 |
| `q3kk6144` | 3 | 143.4 | 3 | 198.7 |
| `attnqkq` | 0 | 0.0 | 2 | 181.2 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 6.1 | 1 | 7.2 |
| `hcwsum` | 48 | 2698.2 | 1 | 107.5 |
| `bf16hcrx` | 49 | 2440.2 | 2 | 94.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 78.8 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `bf16k2560` | 49 | 1598.5 | 1 | 32.8 |
| `rms2560` | 3 | 72.7 | 1 | 26.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 16.4 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |
| `actq256` | 146 | 6504.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 24.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 42.0 | 0 | 0.0 |
| `b32q50acc` | 2 | 55.3 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 376.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2829.3 | 0 | 0.0 |
| `conv1d` | 36 | 294.9 | 0 | 0.0 |
| `dnnormx` | 36 | 634.9 | 0 | 0.0 |
| `dnstep` | 36 | 788.5 | 0 | 0.0 |
| `ffnact` | 48 | 683.0 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 245.8 | 0 | 0.0 |
| `plegate` | 1 | 12.3 | 0 | 0.0 |
| `plerows` | 1 | 88.1 | 0 | 0.0 |
| `q2acc640` | 3 | 55.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 303.1 | 0 | 0.0 |
| `router` | 48 | 1134.6 | 0 | 0.0 |
| `xcopy` | 1 | 32.8 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5625.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3487.7 |
| `bf16hcrwx` | 48 | 1981.4 | 48 | 3229.7 |
| `router_full` | 0 | 0.0 | 48 | 3056.6 |
| `q4kk6144` | 15 | 603.1 | 15 | 2304.0 |
| `q2moegn` | 96 | 1924.1 | 48 | 2254.8 |
| `q3kk2560` | 53 | 1867.8 | 24 | 2085.9 |
| `q5kk6144` | 7 | 360.4 | 7 | 2309.1 |
| `q6kk1536` | 6 | 123.9 | 6 | 1859.6 |
| `q2moedn` | 48 | 1089.5 | 48 | 1806.3 |
| `iq4k2560` | 24 | 942.1 | 17 | 1543.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1501.2 |
| `iq4k6144` | 17 | 580.6 | 17 | 1036.3 |
| `q5kk2560` | 2 | 596.0 | 1 | 585.7 |
| `q4kk2560` | 12 | 354.3 | 4 | 387.1 |
| `xsend` | 123 | 2133.0 | 10 | 364.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 343.0 |
| `kvgather` | 4 | 693.2 | 2 | 344.1 |
| `q2k2560` | 13 | 252.9 | 4 | 250.9 |
| `q3kk6144` | 3 | 142.3 | 3 | 213.0 |
| `attnqkq` | 4 | 213.0 | 2 | 119.8 |
| `qsa_score_select` | 0 | 0.0 | 2 | 110.6 |
| `amax` | 1 | 110.6 | 1 | 117.8 |
| `hcwsum` | 48 | 2563.1 | 1 | 103.4 |
| `bf16hcrx` | 49 | 2437.1 | 2 | 94.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 41.0 |
| `bf16k2560` | 53 | 1826.8 | 1 | 31.7 |
| `rms2560` | 3 | 73.7 | 1 | 26.6 |
| `q6kk2560` | 2 | 54.3 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 14.3 |
| `pleconv` | 1 | 12.3 | 1 | 12.3 |
| `actq256` | 146 | 4695.0 | 0 | 0.0 |
| `attnmerge` | 4 | 34.8 | 0 | 0.0 |
| `attnpartn` | 4 | 563.2 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 47.1 | 0 | 0.0 |
| `b32q40acc` | 4 | 79.9 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 520.2 | 0 | 0.0 |
| `bf16hcup` | 97 | 2879.5 | 0 | 0.0 |
| `conv1d` | 36 | 308.2 | 0 | 0.0 |
| `dnnormx` | 36 | 732.2 | 0 | 0.0 |
| `dnstep` | 36 | 774.1 | 0 | 0.0 |
| `ffnact` | 48 | 684.0 | 0 | 0.0 |
| `idxqk` | 4 | 35.8 | 0 | 0.0 |
| `idxscore` | 4 | 65.5 | 0 | 0.0 |
| `idxsel` | 20 | 181.2 | 0 | 0.0 |
| `moesel` | 48 | 252.9 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 91.1 | 0 | 0.0 |
| `q2acc640` | 2 | 35.8 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 274.4 | 0 | 0.0 |
| `router` | 48 | 1138.7 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5955.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3477.5 |
| `bf16hcrwx` | 48 | 2021.4 | 48 | 3242.0 |
| `router_full` | 0 | 0.0 | 48 | 2988.0 |
| `q4kk6144` | 15 | 610.3 | 15 | 2366.5 |
| `q2moegn` | 96 | 1705.0 | 48 | 2037.8 |
| `q3kk2560` | 54 | 1931.3 | 23 | 1996.8 |
| `q5kk6144` | 7 | 361.5 | 7 | 1905.7 |
| `q6kk1536` | 6 | 123.9 | 6 | 2313.2 |
| `q2moedn` | 48 | 973.8 | 48 | 1752.1 |
| `iq4k2560` | 22 | 882.7 | 17 | 1522.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1487.9 |
| `iq4k6144` | 17 | 573.4 | 17 | 1043.5 |
| `q5kk2560` | 4 | 655.4 | 1 | 577.5 |
| `q4kk2560` | 9 | 294.9 | 4 | 397.3 |
| `xsend` | 123 | 2136.1 | 10 | 354.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 346.1 |
| `kvgather` | 4 | 704.5 | 2 | 341.0 |
| `q2k2560` | 13 | 263.2 | 4 | 247.8 |
| `q3kk6144` | 3 | 141.3 | 3 | 209.9 |
| `attnqkq` | 4 | 20.5 | 2 | 23.6 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 110.6 | 1 | 122.9 |
| `hcwsum` | 48 | 2765.8 | 1 | 110.6 |
| `bf16hcrx` | 49 | 2434.0 | 2 | 90.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 39.9 |
| `bf16k2560` | 53 | 1850.4 | 1 | 31.7 |
| `rms2560` | 3 | 72.7 | 1 | 31.7 |
| `q6kk2560` | 4 | 132.1 | 1 | 108.5 |
| `argmax` | 1 | 11.3 | 1 | 16.4 |
| `pleconv` | 1 | 13.3 | 1 | 13.3 |
| `actq256` | 146 | 4889.6 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 561.2 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 3 | 61.4 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 26.6 | 0 | 0.0 |
| `bf16bias` | 36 | 521.2 | 0 | 0.0 |
| `bf16hcup` | 97 | 2844.7 | 0 | 0.0 |
| `conv1d` | 36 | 301.1 | 0 | 0.0 |
| `dnnormx` | 36 | 729.1 | 0 | 0.0 |
| `dnstep` | 36 | 779.3 | 0 | 0.0 |
| `ffnact` | 48 | 684.0 | 0 | 0.0 |
| `idxqk` | 4 | 19.5 | 0 | 0.0 |
| `idxscore` | 4 | 62.5 | 0 | 0.0 |
| `idxsel` | 20 | 184.3 | 0 | 0.0 |
| `moesel` | 48 | 251.9 | 0 | 0.0 |
| `plegate` | 1 | 15.4 | 0 | 0.0 |
| `plerows` | 1 | 91.1 | 0 | 0.0 |
| `q2acc640` | 2 | 36.9 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 280.6 | 0 | 0.0 |
| `router` | 48 | 1139.7 | 0 | 0.0 |
| `xcopy` | 1 | 33.8 | 0 | 0.0 |

