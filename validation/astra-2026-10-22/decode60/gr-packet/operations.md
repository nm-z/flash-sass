# Complete saved PROF_ALL comparison

Source: canonical archy GPUQ job `96ed25b0-6c12-4470-aebd-9577dbbb4076 and a44645b8-a365-405c-be03-68924ccc8787`, read from the canonical SQLite stdout stream. This extraction performs no GPU run. Both arms sent `prof DIE all`, the same engine command used by `PROF_ALL=1 kern/prof.sh`. Baseline label: `v104`; candidate label: `v119`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: 64 prompt tokens and128 generated outputs, scalar decode on six dies; final request2 of each job; separate observations, not randomized paired A/B. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `hcupq_x` | 97 | 5993.8 | 97 | 5929.7 | -64.1 |
| `q2moegn` | 48 | 5033.5 | 48 | 5008.2 | -25.3 |
| `q2moedn` | 48 | 4859.9 | 48 | 4892.7 | +32.8 |
| `router_full` | 48 | 3075.2 | 48 | 3072.8 | -2.4 |
| `bf16hcrwx` | 48 | 3064.0 | 48 | 2966.2 | -97.8 |
| `grwrite_read` | 47 | 2939.1 | 47 | 2809.4 | -129.7 |
| `q3kk2560` | 24 | 2346.7 | 24 | 2334.0 | -12.6 |
| `gdn_full` | 36 | 2034.0 | 36 | 2036.4 | +2.4 |
| `qsa_full_bal` | 12 | 2000.4 | 12 | 1994.9 | -5.5 |
| `iq4k2560` | 17 | 1703.8 | 17 | 1714.2 | +10.4 |
| `iq4k6144` | 17 | 1420.6 | 17 | 1414.3 | -6.3 |
| `q4kk6144` | 15 | 1239.6 | 15 | 1257.6 | +18.1 |
| `q5kk6144` | 7 | 657.9 | 7 | 654.0 | -3.9 |
| `q5kk2560` | 1 | 599.2 | 1 | 603.5 | +4.3 |
| `q6kk1536` | 6 | 458.4 | 6 | 456.7 | -1.7 |
| `q4kk2560` | 4 | 403.6 | 4 | 403.6 | +0.0 |
| `q2k2560` | 4 | 368.8 | 4 | 374.1 | +5.3 |
| `q3kk6144` | 3 | 267.9 | 3 | 270.3 | +2.4 |
| `bf16hcrx` | 2 | 77.3 | 2 | 75.8 | -1.5 |
| `bf16k2560` | 1 | 53.9 | 1 | 54.4 | +0.5 |
| `rms2560` | 1 | 30.4 | 1 | 29.2 | -1.2 |
| `amax` | 1 | 24.6 | 1 | 25.9 | +1.3 |
| `plegate_norm` | 1 | 17.9 | 1 | 18.4 | +0.5 |
| `ple_rowsq` | 1 | 15.0 | 1 | 15.4 | +0.4 |
| `pleconv` | 1 | 12.5 | 1 | 12.1 | -0.3 |
| `hcwsum` | 1 | 10.8 | 1 | 11.4 | +0.7 |
| **All recorded intervals** | 491 | **38708.7** | 491 | **38435.4** | **-273.3** |

The programs have 492 versus 492 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

