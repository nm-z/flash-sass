# Per-exchange timing

All values below are microseconds per token, summed over exchanges on each die. `exchanges.csv` contains every measured exchange. Finish is the latest of the die’s 16 CTA body-completion observations. Acquire is the latest of its 16 completion observations. Publication and polling instrumentation add overhead; these are instrumented timings, not uninstrumented kernel timings.

Clock offsets are bounded independently for each token using the causal fact that a receiving GPU cannot observe a producer flag before that producer finishes. Relative drift within one token is assumed negligible. Producer skew and communication/polling are intervals, not a falsely precise link-latency measurement. The latter includes publication, transport, cache invalidation, polling and CTA rendezvous.

| Arm | Die | Exchanges/token | Finish → acquire | Producer skew bounds | Communication/poll bounds |
|---|---:|---:|---:|---:|---:|
| phase-baseline | 0 | 354 | 7258.795 | 4584.107–6162.091 | 1096.704–2674.688 |
| phase-baseline | 1 | 354 | 7483.733 | 4740.779–6385.664 | 1098.069–2742.955 |
| phase-baseline | 2 | 354 | 8599.211 | 5621.419–7570.091 | 1029.120–2977.792 |
| phase-baseline | 3 | 354 | 6767.275 | 4043.776–5648.043 | 1119.232–2723.499 |
| phase-baseline | 4 | 354 | 7371.776 | 4592.640–6289.408 | 1082.368–2779.136 |
| phase-baseline | 5 | 354 | 8024.747 | 5231.275–6946.133 | 1078.613–2793.472 |
| split-candidate | 0 | 338 | 7093.931 | 4450.304–5955.243 | 1138.688–2643.627 |
| split-candidate | 1 | 338 | 7306.923 | 4722.005–6339.243 | 967.680–2584.917 |
| split-candidate | 2 | 338 | 8318.635 | 5472.256–7328.427 | 990.208–2846.379 |
| split-candidate | 3 | 338 | 6586.027 | 4003.499–5564.075 | 1021.952–2582.528 |
| split-candidate | 4 | 338 | 7307.605 | 4586.496–6325.931 | 981.675–2721.109 |
| split-candidate | 5 | 338 | 7907.328 | 5227.520–6883.328 | 1024.000–2679.808 |
