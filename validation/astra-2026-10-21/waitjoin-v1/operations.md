# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `83e2e806-d830-4d6d-a4d2-d60921fbf171`, fetched with `gpuq logs ID 10000`. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Each arm profiled the final token after four requests of 64 prompt / 128 generated tokens. Phase counts show the range across dies. Sum across dies divided by six preserves the twelve attention layers despite their changed ownership. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5633.0 | +5633.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 3299.7 | +3299.7 |
| `router_full` | 0 | 0.0 | 48 | 2601.0 | +2601.0 |
| `q3kk2560` | 53–56 | 1685.0 | 23–24 | 2180.3 | +495.3 |
| `bf16hcrwx` | 48 | 1584.6 | 48 | 2082.2 | +497.5 |
| `q2moegn` | 96 | 1449.2 | 48 | 1867.4 | +418.3 |
| `q2moedn` | 48 | 854.0 | 48 | 1742.1 | +888.1 |
| `q4kk6144` | 15 | 404.5 | 15 | 1725.8 | +1321.3 |
| `iq4k2560` | 22–24 | 864.2 | 16–18 | 1602.0 | +737.8 |
| `iq4k6144` | 17 | 473.8 | 17 | 1321.3 | +847.5 |
| `q5kk6144` | 7 | 267.4 | 7 | 1181.7 | +914.3 |
| `gdn_full` | 0 | 0.0 | 36 | 1152.7 | +1152.7 |
| `q6kk1536` | 6 | 103.6 | 6 | 1086.5 | +982.9 |
| `q5kk2560` | 1–4 | 618.7 | 1–3 | 627.4 | +8.7 |
| `q2k2560` | 12–14 | 213.8 | 3–4 | 387.6 | +173.7 |
| `q4kk2560` | 8–12 | 277.8 | 4–6 | 352.3 | +74.4 |
| `xsend` | 123–124 | 1564.0 | 10 | 334.2 | -1229.8 |
| `attnqkq` | 0–4 | 78.9 | 2 | 303.4 | +224.6 |
| `q3kk6144` | 3 | 99.2 | 3 | 237.1 | +137.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 190.1 | +190.1 |
| `amax` | 1 | 98.0 | 1 | 96.1 | -1.8 |
| `bf16hcrx` | 49 | 1179.0 | 2 | 73.2 | -1105.8 |
| `hcwsum` | 48 | 2363.6 | 1 | 71.2 | -2292.4 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.7 | +65.7 |
| `bf16k2560` | 49–53 | 1381.2 | 1 | 52.2 | -1329.0 |
| `rms2560` | 3 | 22.5 | 1 | 51.9 | +29.3 |
| `kvgather` | 0–4 | 41.8 | 2 | 38.7 | -3.1 |
| `plegate_norm` | 0 | 0.0 | 1 | 24.1 | +24.1 |
| `q6kk2560` | 0–4 | 27.0 | 0–1 | 16.2 | -10.8 |
| `argmax` | 1 | 10.6 | 1 | 12.5 | +1.9 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.0 | +12.0 |
| `pleconv` | 1 | 9.9 | 1 | 10.6 | +0.7 |
| `actq256` | 146 | 2779.3 | 0 | 0.0 | -2779.3 |
| `attnmerge` | 0–4 | 16.4 | 0 | 0.0 | -16.4 |
| `attnpartn` | 0–4 | 127.0 | 0 | 0.0 | -127.0 |
| `b32iq4nlacc` | 0–2 | 25.9 | 0 | 0.0 | -25.9 |
| `b32q40acc` | 2–4 | 49.7 | 0 | 0.0 | -49.7 |
| `b32q50acc` | 0–2 | 29.4 | 0 | 0.0 | -29.4 |
| `b32q80acc` | 0–1 | 8.2 | 0 | 0.0 | -8.2 |
| `bf16bias` | 36 | 232.1 | 0 | 0.0 | -232.1 |
| `bf16hcup` | 97 | 2380.6 | 0 | 0.0 | -2380.6 |
| `conv1d` | 36 | 217.1 | 0 | 0.0 | -217.1 |
| `dnnormx` | 36 | 496.8 | 0 | 0.0 | -496.8 |
| `dnstep` | 36 | 619.3 | 0 | 0.0 | -619.3 |
| `ffnact` | 48 | 647.5 | 0 | 0.0 | -647.5 |
| `idxqk` | 0–4 | 26.3 | 0 | 0.0 | -26.3 |
| `idxscore` | 0–4 | 5.8 | 0 | 0.0 | -5.8 |
| `idxsel` | 0–20 | 26.8 | 0 | 0.0 | -26.8 |
| `moesel` | 48 | 214.5 | 0 | 0.0 | -214.5 |
| `plegate` | 1 | 8.2 | 0 | 0.0 | -8.2 |
| `plerows` | 1 | 55.6 | 0 | 0.0 | -55.6 |
| `q2acc640` | 1–4 | 40.6 | 0 | 0.0 | -40.6 |
| `q6kk1536acc` | 18 | 217.6 | 0 | 0.0 | -217.6 |
| `router` | 48 | 1083.1 | 0 | 0.0 | -1083.1 |
| `xcopy` | 0–1 | 26.4 | 0 | 0.0 | -26.4 |
| **All recorded intervals** | 1,177–1,229 | **25006.5** | 498 | **30432.0** | **+5425.5** |

The uploaded programs have 1,178–1,230 versus 499 phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

## Comparisons that can be formed without allocating shared opcode pools

| Operation family | Baseline members | Candidate members | Baseline µs/token/die | Candidate µs/token/die | Difference |
|---|---|---|---:|---:|---:|
| All residual sum + GR down reads | `hcwsum` + `bf16hcrx` | `grwrite_read` + `hcwsum` + `bf16hcrx` | 3542.6 | 5777.4 | +2234.9 |
| Activation producers, quantization, attention partition/merge | `bf16hcup` + `conv1d` + `dnstep` + `dnnormx` + `plerows` + `actq256` + `attnpartn` + `attnmerge` | `hcupq_x` + `gdn_full` + `ple_rowsq` + `qsa_part_merge` | 6692.2 | 4708.2 | -1984.0 |
| QSA score + selection | `idxscore` + `idxsel` | `qsa_score_select` | 32.6 | 12.0 | -20.6 |
| All PLE normalizations + gate | `rms2560` + `plegate` | `rms2560` + `plegate_norm` | 30.7 | 75.9 | +45.2 |
| Q6 slice bodies (publication excluded on baseline) | `q6kk1536` + `q6kk1536acc` | `q6kk1536` | 321.2 | 1086.5 | +765.3 |

The first four rows cover complete opcode families. The activation row does **not** include the baseline XSEND that was also absorbed by attention merge; XSEND is a shared unallocated pool, so this is a family subtotal, not a complete causal fusion comparison. The residual row retains the two unfused reads and final write on the candidate. The PLE row retains all three normalizations, including the two now grouped under one primary RMS opcode. The Q6 row explicitly lacks the absorbed XSEND on the baseline. None of these differences is an isolated fusion speedup or slowdown.

## Die 0

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5543.9 |
| `hcupq_x` | 0 | 0.0 | 97 | 3288.1 |
| `router_full` | 0 | 0.0 | 48 | 2569.2 |
| `q3kk2560` | 54 | 1694.7 | 24 | 2383.9 |
| `bf16hcrwx` | 48 | 1535.0 | 48 | 1503.2 |
| `q2moegn` | 96 | 1521.7 | 48 | 1947.6 |
| `q2moedn` | 48 | 901.1 | 48 | 1885.2 |
| `q4kk6144` | 15 | 419.8 | 15 | 1834.0 |
| `iq4k2560` | 24 | 886.8 | 18 | 1835.0 |
| `iq4k6144` | 17 | 494.6 | 17 | 1531.9 |
| `q5kk6144` | 7 | 277.5 | 7 | 1370.1 |
| `gdn_full` | 0 | 0.0 | 36 | 1208.3 |
| `q6kk1536` | 6 | 108.5 | 6 | 1039.4 |
| `q5kk2560` | 2 | 682.0 | 1 | 661.5 |
| `q2k2560` | 12 | 209.9 | 3 | 326.7 |
| `q4kk2560` | 10 | 275.5 | 4 | 323.6 |
| `xsend` | 124 | 1636.4 | 10 | 72.7 |
| `attnqkq` | 0 | 0.0 | 2 | 290.8 |
| `q3kk6144` | 3 | 103.4 | 3 | 267.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 202.8 |
| `amax` | 1 | 29.7 | 1 | 23.6 |
| `bf16hcrx` | 49 | 1217.5 | 2 | 49.2 |
| `hcwsum` | 48 | 2243.6 | 1 | 6.1 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `bf16k2560` | 49 | 1308.7 | 1 | 64.5 |
| `rms2560` | 3 | 23.6 | 1 | 25.6 |
| `kvgather` | 0 | 0.0 | 2 | 39.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 11.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 3074.0 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 0 | 0.0 | 0 | 0.0 |
| `b32q40acc` | 2 | 36.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 50.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 247.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2311.2 | 0 | 0.0 |
| `conv1d` | 36 | 226.3 | 0 | 0.0 |
| `dnnormx` | 36 | 547.8 | 0 | 0.0 |
| `dnstep` | 36 | 619.5 | 0 | 0.0 |
| `ffnact` | 48 | 654.3 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 217.1 | 0 | 0.0 |
| `plegate` | 1 | 5.1 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 4 | 61.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 222.2 | 0 | 0.0 |
| `router` | 48 | 1086.5 | 0 | 0.0 |
| `xcopy` | 0 | 0.0 | 0 | 0.0 |

## Die 1

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5483.5 |
| `hcupq_x` | 0 | 0.0 | 97 | 3317.8 |
| `router_full` | 0 | 0.0 | 48 | 2568.2 |
| `q3kk2560` | 53 | 1680.4 | 24 | 2458.6 |
| `bf16hcrwx` | 48 | 1530.9 | 48 | 1628.2 |
| `q2moegn` | 96 | 1459.2 | 48 | 1877.0 |
| `q2moedn` | 48 | 864.3 | 48 | 1844.2 |
| `q4kk6144` | 15 | 408.6 | 15 | 1777.7 |
| `iq4k2560` | 24 | 870.4 | 16 | 1589.2 |
| `iq4k6144` | 17 | 476.2 | 17 | 1485.8 |
| `q5kk6144` | 7 | 271.4 | 7 | 1352.7 |
| `gdn_full` | 0 | 0.0 | 36 | 1210.4 |
| `q6kk1536` | 6 | 101.4 | 6 | 1029.1 |
| `q5kk2560` | 4 | 629.8 | 1 | 574.5 |
| `q2k2560` | 14 | 191.5 | 3 | 335.9 |
| `q4kk2560` | 11 | 304.1 | 6 | 493.6 |
| `xsend` | 123 | 1568.8 | 10 | 72.7 |
| `attnqkq` | 4 | 241.7 | 2 | 304.1 |
| `q3kk6144` | 3 | 102.4 | 3 | 263.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 203.8 |
| `amax` | 1 | 114.7 | 1 | 113.7 |
| `bf16hcrx` | 49 | 1177.6 | 2 | 78.8 |
| `hcwsum` | 48 | 2321.4 | 1 | 127.0 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `bf16k2560` | 53 | 1456.1 | 1 | 64.5 |
| `rms2560` | 3 | 23.6 | 1 | 30.7 |
| `kvgather` | 4 | 80.9 | 2 | 39.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 12.3 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 10.2 |
| `actq256` | 146 | 2080.8 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 254.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 37.9 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 243.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2514.9 | 0 | 0.0 |
| `conv1d` | 36 | 228.4 | 0 | 0.0 |
| `dnnormx` | 36 | 554.0 | 0 | 0.0 |
| `dnstep` | 36 | 614.4 | 0 | 0.0 |
| `ffnact` | 48 | 650.2 | 0 | 0.0 |
| `idxqk` | 4 | 111.6 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 208.9 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 4 | 60.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 217.1 | 0 | 0.0 |
| `router` | 48 | 1095.7 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 2

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5886.0 |
| `hcupq_x` | 0 | 0.0 | 97 | 3418.1 |
| `router_full` | 0 | 0.0 | 48 | 2624.5 |
| `q3kk2560` | 56 | 1632.3 | 23 | 2001.9 |
| `bf16hcrwx` | 48 | 1818.6 | 48 | 2320.4 |
| `q2moegn` | 96 | 1240.1 | 48 | 1652.7 |
| `q2moedn` | 48 | 731.1 | 48 | 1622.0 |
| `q4kk6144` | 15 | 354.3 | 15 | 1851.4 |
| `iq4k2560` | 23 | 820.2 | 18 | 1620.0 |
| `iq4k6144` | 17 | 426.0 | 17 | 1371.1 |
| `q5kk6144` | 7 | 235.5 | 7 | 1062.9 |
| `gdn_full` | 0 | 0.0 | 36 | 1032.2 |
| `q6kk1536` | 6 | 96.3 | 6 | 1016.8 |
| `q5kk2560` | 1 | 478.2 | 1 | 487.4 |
| `q2k2560` | 14 | 221.2 | 4 | 422.9 |
| `q4kk2560` | 8 | 238.6 | 4 | 299.0 |
| `xsend` | 123 | 1447.9 | 10 | 329.7 |
| `attnqkq` | 0 | 0.0 | 2 | 371.7 |
| `q3kk6144` | 3 | 89.1 | 3 | 222.2 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 184.3 |
| `amax` | 1 | 209.9 | 1 | 200.7 |
| `bf16hcrx` | 49 | 1117.2 | 2 | 76.8 |
| `hcwsum` | 48 | 2733.1 | 1 | 66.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `bf16k2560` | 49 | 1316.9 | 1 | 51.2 |
| `rms2560` | 3 | 21.5 | 1 | 79.9 |
| `kvgather` | 0 | 0.0 | 2 | 37.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 26.6 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 11.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 3917.8 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 45.1 | 0 | 0.0 |
| `b32q40acc` | 3 | 56.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 24.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 207.9 | 0 | 0.0 |
| `bf16hcup` | 97 | 2111.5 | 0 | 0.0 |
| `conv1d` | 36 | 199.7 | 0 | 0.0 |
| `dnnormx` | 36 | 314.4 | 0 | 0.0 |
| `dnstep` | 36 | 596.0 | 0 | 0.0 |
| `ffnact` | 48 | 644.1 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 215.0 | 0 | 0.0 |
| `plegate` | 1 | 13.3 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 1 | 15.4 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 206.8 | 0 | 0.0 |
| `router` | 48 | 1078.3 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 3

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5550.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3221.5 |
| `router_full` | 0 | 0.0 | 48 | 2606.1 |
| `q3kk2560` | 56 | 1820.7 | 23 | 2189.3 |
| `bf16hcrwx` | 48 | 1550.3 | 48 | 2192.4 |
| `q2moegn` | 96 | 1540.1 | 48 | 1933.3 |
| `q2moedn` | 48 | 905.2 | 48 | 1792.0 |
| `q4kk6144` | 15 | 429.1 | 15 | 1792.0 |
| `iq4k2560` | 23 | 942.1 | 16 | 1531.9 |
| `iq4k6144` | 17 | 494.6 | 17 | 1243.1 |
| `q5kk6144` | 7 | 277.5 | 7 | 877.6 |
| `gdn_full` | 0 | 0.0 | 36 | 1093.6 |
| `q6kk1536` | 6 | 107.5 | 6 | 1212.4 |
| `q5kk2560` | 1 | 682.0 | 3 | 896.0 |
| `q2k2560` | 14 | 233.5 | 4 | 443.4 |
| `q4kk2560` | 8 | 273.4 | 4 | 341.0 |
| `xsend` | 123 | 1587.2 | 10 | 343.0 |
| `attnqkq` | 0 | 0.0 | 2 | 355.3 |
| `q3kk6144` | 3 | 102.4 | 3 | 208.9 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `amax` | 1 | 5.1 | 1 | 7.2 |
| `bf16hcrx` | 49 | 1226.8 | 2 | 80.9 |
| `hcwsum` | 48 | 2268.2 | 1 | 67.6 |
| `ple_rowsq` | 0 | 0.0 | 1 | 66.6 |
| `bf16k2560` | 49 | 1313.8 | 1 | 54.3 |
| `rms2560` | 3 | 22.5 | 1 | 62.5 |
| `kvgather` | 0 | 0.0 | 2 | 36.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 19.5 |
| `q6kk2560` | 0 | 0.0 | 0 | 0.0 |
| `argmax` | 1 | 11.3 | 1 | 13.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 10.2 | 1 | 11.3 |
| `actq256` | 146 | 2948.1 | 0 | 0.0 |
| `attnmerge` | 0 | 0.0 | 0 | 0.0 |
| `attnpartn` | 0 | 0.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 22.5 | 0 | 0.0 |
| `b32q40acc` | 2 | 37.9 | 0 | 0.0 |
| `b32q50acc` | 2 | 50.2 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 202.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2351.1 | 0 | 0.0 |
| `conv1d` | 36 | 200.7 | 0 | 0.0 |
| `dnnormx` | 36 | 448.5 | 0 | 0.0 |
| `dnstep` | 36 | 646.1 | 0 | 0.0 |
| `ffnact` | 48 | 642.0 | 0 | 0.0 |
| `idxqk` | 0 | 0.0 | 0 | 0.0 |
| `idxscore` | 0 | 0.0 | 0 | 0.0 |
| `idxsel` | 0 | 0.0 | 0 | 0.0 |
| `moesel` | 48 | 214.0 | 0 | 0.0 |
| `plegate` | 1 | 7.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 3 | 45.1 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 225.3 | 0 | 0.0 |
| `router` | 48 | 1074.2 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 4

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5785.6 |
| `hcupq_x` | 0 | 0.0 | 97 | 3298.3 |
| `router_full` | 0 | 0.0 | 48 | 2619.4 |
| `q3kk2560` | 53 | 1620.0 | 24 | 2071.6 |
| `bf16hcrwx` | 48 | 1515.5 | 48 | 2397.2 |
| `q2moegn` | 96 | 1398.8 | 48 | 1821.7 |
| `q2moedn` | 48 | 825.3 | 48 | 1613.8 |
| `q4kk6144` | 15 | 408.6 | 15 | 1530.9 |
| `iq4k2560` | 24 | 855.0 | 17 | 1524.7 |
| `iq4k6144` | 17 | 475.1 | 17 | 1169.4 |
| `q5kk6144` | 7 | 268.3 | 7 | 1270.8 |
| `gdn_full` | 0 | 0.0 | 36 | 1186.8 |
| `q6kk1536` | 6 | 103.4 | 6 | 1026.0 |
| `q5kk2560` | 2 | 599.0 | 1 | 574.5 |
| `q2k2560` | 13 | 213.0 | 4 | 407.6 |
| `q4kk2560` | 12 | 312.3 | 4 | 326.7 |
| `xsend` | 123 | 1596.4 | 10 | 583.7 |
| `attnqkq` | 4 | 210.9 | 2 | 298.0 |
| `q3kk6144` | 3 | 97.3 | 3 | 224.3 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 182.3 |
| `amax` | 1 | 112.6 | 1 | 113.7 |
| `bf16hcrx` | 49 | 1158.1 | 2 | 76.8 |
| `hcwsum` | 48 | 2427.9 | 1 | 65.5 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `bf16k2560` | 53 | 1438.7 | 1 | 39.9 |
| `rms2560` | 3 | 21.5 | 1 | 53.2 |
| `kvgather` | 4 | 85.0 | 2 | 39.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 35.8 |
| `q6kk2560` | 2 | 52.2 | 0 | 0.0 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 11.3 |
| `pleconv` | 1 | 10.2 | 1 | 10.2 |
| `actq256` | 146 | 2237.4 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 252.9 | 0 | 0.0 |
| `b32iq4nlacc` | 2 | 44.0 | 0 | 0.0 |
| `b32q40acc` | 4 | 74.8 | 0 | 0.0 |
| `b32q50acc` | 0 | 0.0 | 0 | 0.0 |
| `b32q80acc` | 0 | 0.0 | 0 | 0.0 |
| `bf16bias` | 36 | 243.7 | 0 | 0.0 |
| `bf16hcup` | 97 | 2520.1 | 0 | 0.0 |
| `conv1d` | 36 | 225.3 | 0 | 0.0 |
| `dnnormx` | 36 | 551.9 | 0 | 0.0 |
| `dnstep` | 36 | 619.5 | 0 | 0.0 |
| `ffnact` | 48 | 644.1 | 0 | 0.0 |
| `idxqk` | 4 | 29.7 | 0 | 0.0 |
| `idxscore` | 4 | 11.3 | 0 | 0.0 |
| `idxsel` | 20 | 53.2 | 0 | 0.0 |
| `moesel` | 48 | 222.2 | 0 | 0.0 |
| `plegate` | 1 | 8.2 | 0 | 0.0 |
| `plerows` | 1 | 56.3 | 0 | 0.0 |
| `q2acc640` | 2 | 31.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 218.1 | 0 | 0.0 |
| `router` | 48 | 1081.3 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

## Die 5

| Opcode | Baseline phases | Baseline µs/token | Candidate phases | Candidate µs/token |
|---|---:|---:|---:|---:|
| `grwrite_read` | 0 | 0.0 | 47 | 5549.1 |
| `hcupq_x` | 0 | 0.0 | 97 | 3254.3 |
| `router_full` | 0 | 0.0 | 48 | 2618.4 |
| `q3kk2560` | 54 | 1662.0 | 23 | 1976.3 |
| `bf16hcrwx` | 48 | 1557.5 | 48 | 2451.5 |
| `q2moegn` | 96 | 1535.0 | 48 | 1972.2 |
| `q2moedn` | 48 | 897.0 | 48 | 1695.7 |
| `q4kk6144` | 15 | 406.5 | 15 | 1568.8 |
| `iq4k2560` | 22 | 811.0 | 17 | 1511.4 |
| `iq4k6144` | 17 | 476.2 | 17 | 1126.4 |
| `q5kk6144` | 7 | 274.4 | 7 | 1156.1 |
| `gdn_full` | 0 | 0.0 | 36 | 1184.8 |
| `q6kk1536` | 6 | 104.4 | 6 | 1195.0 |
| `q5kk2560` | 4 | 641.0 | 1 | 570.4 |
| `q2k2560` | 13 | 214.0 | 4 | 389.1 |
| `q4kk2560` | 9 | 263.2 | 4 | 329.7 |
| `xsend` | 123 | 1547.3 | 10 | 603.1 |
| `attnqkq` | 4 | 20.5 | 2 | 200.7 |
| `q3kk6144` | 3 | 100.4 | 3 | 236.5 |
| `qsa_part_merge` | 0 | 0.0 | 2 | 185.3 |
| `amax` | 1 | 115.7 | 1 | 117.8 |
| `bf16hcrx` | 49 | 1176.6 | 2 | 76.8 |
| `hcwsum` | 48 | 2187.3 | 1 | 94.2 |
| `ple_rowsq` | 0 | 0.0 | 1 | 65.5 |
| `bf16k2560` | 53 | 1453.1 | 1 | 38.9 |
| `rms2560` | 3 | 22.5 | 1 | 59.4 |
| `kvgather` | 4 | 85.0 | 2 | 37.9 |
| `plegate_norm` | 0 | 0.0 | 1 | 38.9 |
| `q6kk2560` | 4 | 109.6 | 1 | 97.3 |
| `argmax` | 1 | 10.2 | 1 | 12.3 |
| `qsa_score_select` | 0 | 0.0 | 2 | 12.3 |
| `pleconv` | 1 | 9.2 | 1 | 10.2 |
| `actq256` | 146 | 2417.7 | 0 | 0.0 |
| `attnmerge` | 4 | 32.8 | 0 | 0.0 |
| `attnpartn` | 4 | 255.0 | 0 | 0.0 |
| `b32iq4nlacc` | 1 | 21.5 | 0 | 0.0 |
| `b32q40acc` | 3 | 54.3 | 0 | 0.0 |
| `b32q50acc` | 1 | 25.6 | 0 | 0.0 |
| `b32q80acc` | 1 | 24.6 | 0 | 0.0 |
| `bf16bias` | 36 | 246.8 | 0 | 0.0 |
| `bf16hcup` | 97 | 2475.0 | 0 | 0.0 |
| `conv1d` | 36 | 222.2 | 0 | 0.0 |
| `dnnormx` | 36 | 564.2 | 0 | 0.0 |
| `dnstep` | 36 | 620.5 | 0 | 0.0 |
| `ffnact` | 48 | 650.2 | 0 | 0.0 |
| `idxqk` | 4 | 16.4 | 0 | 0.0 |
| `idxscore` | 4 | 12.3 | 0 | 0.0 |
| `idxsel` | 20 | 54.3 | 0 | 0.0 |
| `moesel` | 48 | 209.9 | 0 | 0.0 |
| `plegate` | 1 | 8.2 | 0 | 0.0 |
| `plerows` | 1 | 55.3 | 0 | 0.0 |
| `q2acc640` | 2 | 29.7 | 0 | 0.0 |
| `q6kk1536acc` | 18 | 216.1 | 0 | 0.0 |
| `router` | 48 | 1082.4 | 0 | 0.0 |
| `xcopy` | 1 | 31.7 | 0 | 0.0 |

