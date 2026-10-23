# Complete saved PROF_ALL comparison

Sources: canonical archy GPUq jobs `8e3af1ee-31af-4160-b9ee-c4a38f087534` (v97) and `96ed25b0-6c12-4470-aebd-9577dbbb4076` (v104), read from the SQLite stdout stream. No new GPU run was performed. These are separate observations, not a matched randomized A/B experiment. Both used `prof DIE all`, the engine command used by `PROF_ALL=1 kern/prof.sh`.

These are inclusive final-token phase times, averaged over six dies, **not** averages over the measured requests or isolated operation execution times. Workload: separate full-model jobs, 64 prompt tokens and128 generated tokens; v97 final request1, v104 final request2; scalar decode, MTP off. Phase counts show the range across dies. Averaging all six die totals accounts for all model attention layers. Synchronization and waits are included; grouped descriptors are charged to their primary opcode. The first timestamp has no preceding interval and is omitted by the profiler. Zero means no recorded interval for this opcode.

| Profile opcode | Baseline phases/die | Baseline µs/token/die | Candidate phases/die | Candidate µs/token/die | Difference |
|---|---:|---:|---:|---:|---:|
| `hcupq_x` | 97 | 5923.5 | 97 | 5993.8 | +70.3 |
| `q2moegn` | 48 | 5010.1 | 48 | 5033.5 | +23.4 |
| `q2moedn` | 48 | 4869.1 | 48 | 4859.9 | -9.2 |
| `router_full` | 48 | 3111.6 | 48 | 3075.2 | -36.3 |
| `bf16hcrwx` | 48 | 3545.9 | 48 | 3064.0 | -481.9 |
| `grwrite_read` | 47 | 3469.2 | 47 | 2939.1 | -530.1 |
| `q3kk2560` | 24 | 2324.2 | 24 | 2346.7 | +22.5 |
| `gdn_full` | 36 | 2041.3 | 36 | 2034.0 | -7.3 |
| `qsa_full_bal` | 12 | 2018.8 | 12 | 2000.4 | -18.4 |
| `iq4k2560` | 17 | 1709.6 | 17 | 1703.8 | -5.8 |
| `iq4k6144` | 17 | 1426.8 | 17 | 1420.6 | -6.1 |
| `q4kk6144` | 15 | 1244.1 | 15 | 1239.6 | -4.6 |
| `q5kk6144` | 7 | 660.1 | 7 | 657.9 | -2.2 |
| `q5kk2560` | 1 | 600.0 | 1 | 599.2 | -0.9 |
| `q6kk1536` | 6 | 459.3 | 6 | 458.4 | -0.8 |
| `q4kk2560` | 4 | 405.3 | 4 | 403.6 | -1.7 |
| `q2k2560` | 4 | 370.3 | 4 | 368.8 | -1.6 |
| `q3kk6144` | 3 | 276.0 | 3 | 267.9 | -8.0 |
| `bf16hcrx` | 2 | 81.6 | 2 | 77.3 | -4.3 |
| `bf16k2560` | 1 | 55.8 | 1 | 53.9 | -1.8 |
| `rms2560` | 1 | 27.6 | 1 | 30.4 | +2.7 |
| `amax` | 1 | 24.7 | 1 | 24.6 | -0.2 |
| `plegate_norm` | 1 | 18.6 | 1 | 17.9 | -0.7 |
| `ple_rowsq` | 1 | 15.5 | 1 | 15.0 | -0.5 |
| `pleconv` | 1 | 12.5 | 1 | 12.5 | +0.0 |
| `hcwsum` | 1 | 11.3 | 1 | 10.8 | -0.5 |
| **All recorded intervals** | 491 | **39712.9** | 491 | **38708.7** | **-1004.1** |

The programs have 492 versus 492 physical phases; the interval counts above exclude their first timestamp. CSV contains each die separately.

The CSV retains every die separately. All operation families and dispatch descriptors are the same in these two versions; only the memory-instruction schedule changes.
