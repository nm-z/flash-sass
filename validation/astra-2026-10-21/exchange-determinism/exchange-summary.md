# Per-exchange timing

All values below are microseconds per token, summed over exchanges on each die. `exchanges.csv` contains every measured exchange. Finish is the latest of the die’s 16 CTA body-completion observations. Acquire is the latest of its 16 completion observations. Publication and polling instrumentation add overhead; these are instrumented timings, not uninstrumented kernel timings.

Clock offsets are bounded independently for each token using the causal fact that a receiving GPU cannot observe a producer flag before that producer finishes. Relative drift within one token is assumed negligible. Producer skew and communication/polling are intervals, not a falsely precise link-latency measurement. The latter includes publication, transport, cache invalidation, polling and CTA rendezvous.

| Arm | Die | Exchanges/token | Finish → acquire | Producer skew bounds | Communication/poll bounds |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 354 | 7940.608 | 4393.728–6561.536 | 1379.072–3546.880 |
| baseline | 1 | 354 | 7261.696 | 3716.352–5772.544 | 1489.152–3545.344 |
| baseline | 2 | 354 | 9258.496 | 5128.704–7872.512 | 1385.984–4129.792 |
| baseline | 3 | 354 | 7293.696 | 4060.416–5863.424 | 1430.272–3233.280 |
| baseline | 4 | 354 | 7135.744 | 3628.544–5743.360 | 1392.384–3507.200 |
| baseline | 5 | 354 | 7584.768 | 4075.264–6160.128 | 1424.640–3509.504 |
| kernel-candidate | 0 | 354 | 8323.840 | 4327.168–6392.064 | 1931.776–3996.672 |
| kernel-candidate | 1 | 354 | 8828.672 | 4669.696–6996.736 | 1831.936–4158.976 |
| kernel-candidate | 2 | 354 | 9586.688 | 5200.640–7700.992 | 1885.696–4386.048 |
| kernel-candidate | 3 | 354 | 7533.568 | 3955.712–5382.144 | 2151.424–3577.856 |
| kernel-candidate | 4 | 354 | 8427.776 | 4379.648–6675.712 | 1752.064–4048.128 |
| kernel-candidate | 5 | 354 | 8887.552 | 4793.856–6972.416 | 1915.136–4093.696 |
