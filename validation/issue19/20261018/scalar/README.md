# Checked scalar API overhead

Measured on engi from commit **`5b3ea525005fa5f8573ab544bfbeee6783c62e25`**. The [manifest](scalar.json) reports `PASS` and records source hashes, the exact compiler/run commands, executable hash, host inventory and every raw-output hash. All 14 raw stdout/stderr hashes were recomputed before this evidence was copied. The [source-scope check](source-scope.stdout) exited zero: `kern/` and `ref/` at this commit are unchanged from merged revision `d46d188a435c5d959993f16dc1f78c837cc9f70d`.

The [C benchmark](../../scalar_bench.c) calls the real `kv_quant32_checked` and `kv_quant32_gpu` functions compiled from the same `ref/qsa.c`. Inputs are 64 deterministic groups of 32 finite, in-range components, with positive, negative and zero channels. Rounding is set and checked as `FE_TONEAREST`. This comparison includes the checked API's argument/rounding/finite-input checks and temporary-output copy on successful calls; both variants use the same bounded quantization arithmetic.

Two warmup rounds run 4,096 calls per variant. Then 12 paired samples run 262,144 calls per variant, alternating which variant runs first. Inputs, output comparisons, checksums and printing are outside the timed intervals. Separate compilation without LTO preserves the real function calls. Every checked call succeeds, and all code bytes and half-scale bits match after every timed interval: **3,145,728 timed calls per variant and 24 timed output checks**. See [all raw samples](benchmark.stdout).

## Observed timing

| Clock | Unchecked median (ns/group) | Checked median (ns/group) | Median paired difference (ns/group) | Median paired difference (%) |
|---|---:|---:|---:|---:|
| Thread CPU time | 94.2586555 | 113.424664 | +16.9233723 | +18.0249748% |
| Monotonic wall time | 105.020226 | 142.078272 | +19.0568008 | +17.2543837% |

Each paired difference is computed within its sample before taking the median. It is not the subtraction of the two variant medians. Thread CPU time excludes descheduled intervals; wall time retains them. The benchmark ran on the shared host without fixed CPU affinity. Background-process inventory is in [processes.stdout](processes.stdout), and individual samples preserve the resulting variation. No GPU was accessed.

The recorded host is an AMD Ryzen 5 7600X, with GCC 16.2.1 and `-std=gnu99 -O2 -ffp-contract=off -UNDEBUG -Wall -Wextra -Werror`; see [CPU inventory](cpu.stdout) and [compiler output](compiler.stdout). These measurements describe successful scalar calls for this fixture set and environment. They do not establish rejected-input latency, GPU cost or whole-model overhead.

## Command

Run at the measured commit from the repository root; choose a writable output path for `scalar_bench`:

```bash
cc -std=gnu99 -O2 -ffp-contract=off -UNDEBUG -Wall -Wextra -Werror \
  -o scalar_bench validation/issue19/scalar_bench.c \
  ref/qsa.c ref/fmath.c -Iref -lm
./scalar_bench 262144
```
