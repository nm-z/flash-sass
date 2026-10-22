# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `6842afd1-0e3b-4063-aaa6-56bdb6736d81`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`. Baseline label: `baseline`; candidate label: `kernel-candidate`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: 4096 prompt / 16 generated; two requests per arm; both pass same-kernel repeatability. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5836.4 | +5836.4 |
| `hcupq_x` | 0 | 0.0 | 97 | 3670.3 | +3670.3 |
| `bf16hcrwx` | 48 | 2109.4 | 48 | 3259.0 | +1149.6 |
| `router_full` | 0 | 0.0 | 48 | 2959.4 | +2959.4 |
| `q4kk6144` | 15 | 580.6 | 15 | 2454.0 | +1873.4 |
| `q2moegn` | 96 | 1776.8 | 48 | 2159.1 | +382.3 |
| `q3kk2560` | 53–56 | 1984.0 | 23–24 | 2057.8 | +73.7 |
| `q5kk6144` | 7 | 354.3 | 7 | 2012.0 | +1657.7 |
| `q6kk1536` | 6 | 129.0 | 6 | 2004.5 | +1875.4 |
| `q2moedn` | 48 | 1000.1 | 48 | 1808.6 | +808.5 |
| `iq4k2560` | 22–24 | 944.3 | 16–18 | 1526.3 | +582.0 |
| `gdn_full` | 0 | 0.0 | 36 | 1511.4 | +1511.4 |
| `iq4k6144` | 17 | 567.5 | 17 | 1059.5 | +492.0 |
| `q5kk2560` | 1–4 | 628.4 | 1–3 | 628.9 | +0.5 |
| `q4kk2560` | 8–12 | 315.2 | 4–6 | 424.4 | +109.2 |
| `xsend` | 123–124 | 2258.6 | 10 | 356.2 | -1902.4 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 351.1 | +351.1 |
| `kvgather` | 0–4 | 335.9 | 2 | 344.1 | +8.2 |
| `q2k2560` | 12–14 | 260.2 | 3–4 | 231.6 | -28.6 |
| `q3kk6144` | 3 | 132.3 | 3 | 216.9 | +84.6 |
| `attnqkq` | 0–4 | 93.4 | 2 | 148.1 | +54.8 |
| `qsa_score_select` | 0 | 0.0 | 2 | 110.4 | +110.4 |
| `amax` | 1 | 100.0 | 1 | 101.5 | +1.5 |
| `hcwsum` | 48 | 2748.6 | 1 | 90.1 | -2658.5 |
| `bf16hcrx` | 49 | 2421.4 | 2 | 85.8 | -2335.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 81.1 | +81.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.5 | +36.5 |
| `rms2560` | 3 | 73.9 | 1 | 28.7 | -45.2 |
| `bf16k2560` | 49–53 | 1732.9 | 1 | 28.2 | -1704.8 |
| `q6kk2560` | 0–4 | 31.2 | 0–1 | 18.3 | -13.0 |
| `argmax` | 1 | 10.8 | 1 | 15.9 | +5.1 |
| `pleconv` | 1 | 12.5 | 1 | 13.1 | +0.7 |
| `actq256` | 146 | 5741.6 | 0 | 0.0 | -5741.6 |
| `attnmerge` | 0–4 | 16.9 | 0 | 0.0 | -16.9 |
| `attnpartn` | 0–4 | 280.9 | 0 | 0.0 | -280.9 |
| `b32iq4nlacc` | 0–2 | 28.3 | 0 | 0.0 | -28.3 |
| `b32q40acc` | 2–4 | 55.6 | 0 | 0.0 | -55.6 |
| `b32q50acc` | 0–2 | 32.8 | 0 | 0.0 | -32.8 |
| `b32q80acc` | 0–1 | 8.7 | 0 | 0.0 | -8.7 |
| `bf16bias` | 36 | 484.7 | 0 | 0.0 | -484.7 |
| `bf16hcup` | 97 | 2775.2 | 0 | 0.0 | -2775.2 |
| `conv1d` | 36 | 308.2 | 0 | 0.0 | -308.2 |
| `dnnormx` | 36 | 683.7 | 0 | 0.0 | -683.7 |
| `dnstep` | 36 | 773.8 | 0 | 0.0 | -773.8 |
| `ffnact` | 48 | 683.2 | 0 | 0.0 | -683.2 |
| `idxqk` | 0–4 | 15.9 | 0 | 0.0 | -15.9 |
| `idxscore` | 0–4 | 31.7 | 0 | 0.0 | -31.7 |
| `idxsel` | 0–20 | 92.8 | 0 | 0.0 | -92.8 |
| `moesel` | 48 | 247.0 | 0 | 0.0 | -247.0 |
| `plegate` | 1 | 14.5 | 0 | 0.0 | -14.5 |
| `plerows` | 1 | 73.5 | 0 | 0.0 | -73.5 |
| `q2acc640` | 1–4 | 50.0 | 0 | 0.0 | -50.0 |
| `q6kk1536acc` | 18 | 278.9 | 0 | 0.0 | -278.9 |
| `router` | 48 | 1135.6 | 0 | 0.0 | -1135.6 |
| `xcopy` | 0–1 | 28.3 | 0 | 0.0 | -28.3 |
| **All recorded intervals** | 1177–1229 | **34443.1** | 498 | **35629.2** | **+1186.1** |

The programs have 1178–1230 versus 499 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` + `grwrite_read` | `hcwsum` + `bf16hcrx` + `grwrite_read` | 5170.0 | 6012.4 | +842.4 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` + `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 10653.9 | 5613.9 | -5040.0 |
| QSA score + selection | `idxscore` + `idxsel` + `qsa_score_select` | `idxscore` + `idxsel` + `qsa_score_select` | 124.6 | 110.4 | -14.1 |
| All PLE normalizations + gate | `rms2560` + `plegate` + `plegate_norm` | `rms2560` + `plegate` + `plegate_norm` | 88.4 | 65.2 | -23.2 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` + `q6kk1536acc` | 407.9 | 2004.5 | +1596.6 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5543.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3442.7 |
| `bf16hcrwx` | 48 | 2147.3 | 48 | 3367.9 |
| `router_full` | 0 | 0.0 | 48 | 2948.1 |
| `q4kk6144` | 15 | 542.7 | 15 | 2152.4 |
| `q2moegn` | 96 | 1981.4 | 48 | 2529.3 |
| `q3kk2560` | 54 | 2000.9 | 24 | 2203.6 |
| `q5kk6144` | 7 | 342.0 | 7 | 2300.9 |
| `q6kk1536` | 6 | 126.0 | 6 | 1808.4 |
| `q2moedn` | 48 | 1076.2 | 48 | 1954.8 |
| `iq4k2560` | 24 | 975.9 | 18 | 1650.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1605.6 |
| `iq4k6144` | 17 | 604.2 | 17 | 932.9 |
| `q5kk2560` | 2 | 687.1 | 1 | 669.7 |
| `q4kk2560` | 10 | 314.4 | 4 | 414.7 |
| `xsend` | 124 | 2397.2 | 10 | 344.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 373.8 |
| `kvgather` | 0 | 0.0 | 2 | 352.3 |
| `q2k2560` | 12 | 258.0 | 3 | 192.5 |
| `q3kk6144` | 3 | 131.1 | 3 | 196.6 |
| `attnqkq` | 0 | 0.0 | 2 | 190.5 |
| `qsa_score_select` | 0 | 0.0 | 2 | 117.8 |
| `amax` | 1 | 33.8 | 1 | 27.6 |
| `hcwsum` | 48 | 2420.7 | 1 | 10.2 |
| `bf16hcrx` | 49 | 2434.0 | 2 | 62.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 84.0 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 73.7 | 1 | 29.7 |
| `bf16k2560` | 49 | 1668.1 | 1 | 29.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 16.4 |
| `pleconv` | 1 | 12.3 | 1 | 14.3 |
| `actq256` | 146 | 6544.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 44.0 | 0 | 0.0 |
| `b32q50acc` | 2 | 58.4 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 523.3 | 0 | 0.0 |
| `bf16hcup` | 97 | 2718.7 | 0 | 0.0 |
| `conv1d` | 36 | 316.4 | 0 | 0.0 |
| `dnnormx` | 36 | 720.9 | 0 | 0.0 |
| `dnstep` | 36 | 772.1 | 0 | 0.0 |
| `ffnact` | 48 | 685.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 248.8 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 4 | 76.8 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 276.5 | 0 | 0.0 |
| `router` | 48 | 1134.6 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 6258.7 |
| `hcupq_x` | 0 | 0.0 | 97 | 3770.4 |
| `bf16hcrwx` | 48 | 1999.9 | 48 | 3161.1 |
| `router_full` | 0 | 0.0 | 48 | 2913.3 |
| `q4kk6144` | 15 | 599.0 | 15 | 2299.9 |
| `q2moegn` | 96 | 1516.5 | 48 | 1831.9 |
| `q3kk2560` | 53 | 1973.2 | 24 | 2071.6 |
| `q5kk6144` | 7 | 367.6 | 7 | 2328.6 |
| `q6kk1536` | 6 | 132.1 | 6 | 1840.1 |
| `q2moedn` | 48 | 854.0 | 48 | 1706.0 |
| `iq4k2560` | 24 | 940.0 | 16 | 1445.9 |
| `gdn_full` | 0 | 0.0 | 36 | 1520.6 |
| `iq4k6144` | 17 | 577.5 | 17 | 1067.0 |
| `q5kk2560` | 4 | 651.3 | 1 | 579.6 |
| `q4kk2560` | 11 | 344.1 | 6 | 577.5 |
| `xsend` | 123 | 2268.2 | 10 | 327.7 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 361.5 |
| `kvgather` | 4 | 677.9 | 2 | 342.0 |
| `q2k2560` | 14 | 242.7 | 3 | 197.6 |
| `q3kk6144` | 3 | 137.2 | 3 | 226.3 |
| `attnqkq` | 4 | 327.7 | 2 | 184.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 109.6 |
| `amax` | 1 | 112.6 | 1 | 116.7 |
| `hcwsum` | 48 | 3093.5 | 1 | 105.5 |
| `bf16hcrx` | 49 | 2402.3 | 2 | 91.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `rms2560` | 3 | 72.7 | 1 | 28.7 |
| `bf16k2560` | 53 | 1839.1 | 1 | 25.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 17.4 |
| `pleconv` | 1 | 12.3 | 1 | 12.3 |
| `actq256` | 146 | 4508.7 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 562.2 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 23.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 39.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 26.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 531.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2904.1 | 0 | 0.0 |
| `conv1d` | 36 | 314.4 | 0 | 0.0 |
| `dnnormx` | 36 | 732.2 | 0 | 0.0 |
| `dnstep` | 36 | 772.1 | 0 | 0.0 |
| `ffnact` | 48 | 691.2 | 0 | 0.0 |
| `idxqk` | 4 | 41.0 | 0 | 0.0 |
| `idxscore` | 4 | 64.5 | 0 | 0.0 |
| `idxsel` | 20 | 185.3 | 0 | 0.0 |
| `moesel` | 48 | 244.7 | 0 | 0.0 |
| `plegate` | 1 | 15.4 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 4 | 76.8 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 272.4 | 0 | 0.0 |
| `router` | 48 | 1142.8 | 0 | 0.0 |
| `xcopy` | 1 | 33.8 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 6000.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3708.9 |
| `bf16hcrwx` | 48 | 2356.2 | 48 | 3363.8 |
| `router_full` | 0 | 0.0 | 48 | 2970.6 |
| `q4kk6144` | 15 | 499.7 | 15 | 2833.4 |
| `q2moegn` | 96 | 1583.1 | 48 | 1969.2 |
| `q3kk2560` | 56 | 1939.5 | 23 | 1913.9 |
| `q5kk6144` | 7 | 308.2 | 7 | 1828.9 |
| `q6kk1536` | 6 | 129.0 | 6 | 1867.8 |
| `q2moedn` | 48 | 944.1 | 48 | 1768.4 |
| `iq4k2560` | 23 | 883.7 | 18 | 1477.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1460.2 |
| `iq4k6144` | 17 | 508.9 | 17 | 1194.0 |
| `q5kk2560` | 1 | 484.4 | 1 | 487.4 |
| `q4kk2560` | 8 | 278.5 | 4 | 362.5 |
| `xsend` | 123 | 2253.8 | 10 | 387.1 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 343.0 |
| `kvgather` | 0 | 0.0 | 2 | 336.9 |
| `q2k2560` | 14 | 248.8 | 4 | 244.7 |
| `q3kk6144` | 3 | 117.8 | 3 | 213.0 |
| `attnqkq` | 0 | 0.0 | 2 | 188.4 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 210.9 | 1 | 213.0 |
| `hcwsum` | 48 | 2977.8 | 1 | 103.4 |
| `bf16hcrx` | 49 | 2395.1 | 2 | 88.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 78.8 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `rms2560` | 3 | 76.8 | 1 | 29.7 |
| `bf16k2560` | 49 | 1648.6 | 1 | 29.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 14.3 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |
| `actq256` | 146 | 7461.9 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 48.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 64.5 | 0 | 0.0 |
| `b32q50acc` | 1 | 28.7 | 0 | 0.0 |
| `b32q80acc` | 1 | 25.6 | 0 | 0.0 |
| `bf16bias` | 36 | 402.4 | 0 | 0.0 |
| `bf16hcup` | 97 | 2508.8 | 0 | 0.0 |
| `conv1d` | 36 | 295.9 | 0 | 0.0 |
| `dnnormx` | 36 | 497.7 | 0 | 0.0 |
| `dnstep` | 36 | 771.1 | 0 | 0.0 |
| `ffnact` | 48 | 685.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 249.9 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 72.7 | 0 | 0.0 |
| `q2acc640` | 1 | 18.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 269.3 | 0 | 0.0 |
| `router` | 48 | 1126.4 | 0 | 0.0 |
| `xcopy` | 1 | 34.8 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5644.3 |
| `hcupq_x` | 0 | 0.0 | 97 | 3749.9 |
| `bf16hcrwx` | 48 | 2017.3 | 48 | 3224.6 |
| `router_full` | 0 | 0.0 | 48 | 2921.5 |
| `q4kk6144` | 15 | 627.7 | 15 | 2724.9 |
| `q2moegn` | 96 | 1920.0 | 48 | 2251.8 |
| `q3kk2560` | 56 | 2169.9 | 23 | 2059.3 |
| `q5kk6144` | 7 | 379.9 | 7 | 1366.0 |
| `q6kk1536` | 6 | 128.0 | 6 | 2322.4 |
| `q2moedn` | 48 | 1046.5 | 48 | 1886.2 |
| `iq4k2560` | 23 | 1038.3 | 16 | 1486.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1484.8 |
| `iq4k6144` | 17 | 563.2 | 17 | 1023.0 |
| `q5kk2560` | 1 | 688.1 | 3 | 883.7 |
| `q4kk2560` | 8 | 300.0 | 4 | 401.4 |
| `xsend` | 123 | 2283.5 | 10 | 356.4 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 341.0 |
| `kvgather` | 0 | 0.0 | 2 | 337.9 |
| `q2k2560` | 14 | 282.6 | 4 | 254.0 |
| `q3kk6144` | 3 | 137.2 | 3 | 215.0 |
| `attnqkq` | 0 | 0.0 | 2 | 181.2 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 6.1 | 1 | 8.2 |
| `hcwsum` | 48 | 2657.3 | 1 | 106.5 |
| `bf16hcrx` | 49 | 2437.1 | 2 | 94.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 80.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 72.7 | 1 | 26.6 |
| `bf16k2560` | 49 | 1587.2 | 1 | 29.7 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 15.4 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |
| `actq256` | 146 | 6460.4 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 24.6 | 0 | 0.0 |
| `b32q40acc` | 2 | 43.0 | 0 | 0.0 |
| `b32q50acc` | 2 | 55.3 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 378.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2820.1 | 0 | 0.0 |
| `conv1d` | 36 | 303.1 | 0 | 0.0 |
| `dnnormx` | 36 | 665.6 | 0 | 0.0 |
| `dnstep` | 36 | 783.4 | 0 | 0.0 |
| `ffnact` | 48 | 675.8 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 250.9 | 0 | 0.0 |
| `plegate` | 1 | 15.4 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 3 | 54.3 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 306.2 | 0 | 0.0 |
| `router` | 48 | 1131.5 | 0 | 0.0 |
| `xcopy` | 1 | 34.8 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5621.8 |
| `hcupq_x` | 0 | 0.0 | 97 | 3687.4 |
| `bf16hcrwx` | 48 | 2066.4 | 48 | 3216.4 |
| `router_full` | 0 | 0.0 | 48 | 3025.9 |
| `q4kk6144` | 15 | 598.0 | 15 | 2329.6 |
| `q2moegn` | 96 | 1934.3 | 48 | 2306.0 |
| `q3kk2560` | 53 | 1889.3 | 24 | 2093.1 |
| `q5kk6144` | 7 | 368.6 | 7 | 2323.5 |
| `q6kk1536` | 6 | 130.0 | 6 | 1870.8 |
| `q2moedn` | 48 | 1105.9 | 48 | 1781.8 |
| `iq4k2560` | 24 | 946.2 | 17 | 1552.4 |
| `gdn_full` | 0 | 0.0 | 36 | 1502.2 |
| `iq4k6144` | 17 | 579.6 | 17 | 1072.1 |
| `q5kk2560` | 2 | 599.0 | 1 | 574.5 |
| `q4kk2560` | 12 | 353.3 | 4 | 393.2 |
| `xsend` | 123 | 2162.7 | 10 | 363.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 343.0 |
| `kvgather` | 4 | 667.6 | 2 | 346.1 |
| `q2k2560` | 13 | 257.0 | 4 | 250.9 |
| `q3kk6144` | 3 | 136.2 | 3 | 226.3 |
| `attnqkq` | 4 | 207.9 | 2 | 121.9 |
| `qsa_score_select` | 0 | 0.0 | 2 | 108.5 |
| `amax` | 1 | 119.8 | 1 | 122.9 |
| `hcwsum` | 48 | 2563.1 | 1 | 103.4 |
| `bf16hcrx` | 49 | 2428.9 | 2 | 92.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 81.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 73.7 | 1 | 31.7 |
| `bf16k2560` | 53 | 1826.8 | 1 | 23.6 |
| `q6kk2560` | 2 | 55.3 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 17.4 |
| `pleconv` | 1 | 12.3 | 1 | 13.3 |
| `actq256` | 146 | 4641.8 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 560.1 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 49.2 | 0 | 0.0 |
| `b32q40acc` | 4 | 81.9 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 534.5 | 0 | 0.0 |
| `bf16hcup` | 97 | 2849.8 | 0 | 0.0 |
| `conv1d` | 36 | 309.2 | 0 | 0.0 |
| `dnnormx` | 36 | 737.3 | 0 | 0.0 |
| `dnstep` | 36 | 771.1 | 0 | 0.0 |
| `ffnact` | 48 | 686.1 | 0 | 0.0 |
| `idxqk` | 4 | 33.8 | 0 | 0.0 |
| `idxscore` | 4 | 64.5 | 0 | 0.0 |
| `idxsel` | 20 | 184.3 | 0 | 0.0 |
| `moesel` | 48 | 243.7 | 0 | 0.0 |
| `plegate` | 1 | 14.3 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 2 | 36.9 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 270.3 | 0 | 0.0 |
| `router` | 48 | 1139.7 | 0 | 0.0 |
| `xcopy` | 1 | 34.8 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5949.4 |
| `hcupq_x` | 0 | 0.0 | 97 | 3662.8 |
| `bf16hcrwx` | 48 | 2069.5 | 48 | 3220.5 |
| `router_full` | 0 | 0.0 | 48 | 2976.8 |
| `q4kk6144` | 15 | 616.4 | 15 | 2383.9 |
| `q2moegn` | 96 | 1725.4 | 48 | 2066.4 |
| `q3kk2560` | 54 | 1931.3 | 23 | 2005.0 |
| `q5kk6144` | 7 | 359.4 | 7 | 1924.1 |
| `q6kk1536` | 6 | 129.0 | 6 | 2317.3 |
| `q2moedn` | 48 | 973.8 | 48 | 1754.1 |
| `iq4k2560` | 22 | 881.7 | 17 | 1544.2 |
| `gdn_full` | 0 | 0.0 | 36 | 1495.0 |
| `iq4k6144` | 17 | 571.4 | 17 | 1068.0 |
| `q5kk2560` | 4 | 660.5 | 1 | 578.6 |
| `q4kk2560` | 9 | 301.1 | 4 | 397.3 |
| `xsend` | 123 | 2186.2 | 10 | 358.4 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 344.1 |
| `kvgather` | 4 | 669.7 | 2 | 349.2 |
| `q2k2560` | 13 | 272.4 | 4 | 249.9 |
| `q3kk6144` | 3 | 134.1 | 3 | 224.3 |
| `attnqkq` | 4 | 24.6 | 2 | 22.5 |
| `qsa_score_select` | 0 | 0.0 | 2 | 109.6 |
| `amax` | 1 | 116.7 | 1 | 120.8 |
| `hcwsum` | 48 | 2779.1 | 1 | 111.6 |
| `bf16hcrx` | 49 | 2431.0 | 2 | 87.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 79.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 36.9 |
| `rms2560` | 3 | 73.7 | 1 | 25.6 |
| `bf16k2560` | 53 | 1827.8 | 1 | 30.7 |
| `q6kk2560` | 4 | 132.1 | 1 | 109.6 |
| `argmax` | 1 | 10.2 | 1 | 14.3 |
| `pleconv` | 1 | 13.3 | 1 | 12.3 |
| `actq256` | 146 | 4832.3 | 0 | 0.0 |
| `attnmerge` | 4 | 33.8 | 0 | 0.0 |
| `attnpartn` | 4 | 563.2 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 24.6 | 0 | 0.0 |
| `b32q40acc` | 3 | 60.4 | 0 | 0.0 |
| `b32q50acc` | 1 | 27.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 26.6 | 0 | 0.0 |
| `bf16bias` | 36 | 537.6 | 0 | 0.0 |
| `bf16hcup` | 97 | 2849.8 | 0 | 0.0 |
| `conv1d` | 36 | 310.3 | 0 | 0.0 |
| `dnnormx` | 36 | 748.5 | 0 | 0.0 |
| `dnstep` | 36 | 773.1 | 0 | 0.0 |
| `ffnact` | 48 | 675.8 | 0 | 0.0 |
| `idxqk` | 4 | 20.5 | 0 | 0.0 |
| `idxscore` | 4 | 61.4 | 0 | 0.0 |
| `idxsel` | 20 | 187.4 | 0 | 0.0 |
| `moesel` | 48 | 243.7 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 73.7 | 0 | 0.0 |
| `q2acc640` | 2 | 36.9 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 278.5 | 0 | 0.0 |
| `router` | 48 | 1138.7 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

