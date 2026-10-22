# Per-exchange timing

All values below are microseconds per token, summed over exchanges on each die. `exchanges.csv` contains every measured exchange. Finish is the latest of the die’s 16 CTA body-completion observations. Acquire is the latest of its 16 completion observations. Publication and polling instrumentation add overhead; these are instrumented timings, not uninstrumented kernel timings.

Clock offsets are bounded independently for each token using the causal fact that a receiving GPU cannot observe a producer flag before that producer finishes. Relative drift within one token is assumed negligible. Producer skew and communication/polling are intervals, not a falsely precise link-latency measurement. The latter includes publication, transport, cache invalidation, polling and CTA rendezvous.

| Arm | Die | Exchanges/token | Finish → acquire | Producer skew bounds | Communication/poll bounds |
|---|---:|---:|---:|---:|---:|
| row-balanced | 0 | 402 | 11476.992 | 5481.472–9918.976 | 1558.016–5995.520 |
| row-balanced | 1 | 402 | 11550.720 | 5404.672–9667.584 | 1883.136–6146.048 |
| row-balanced | 2 | 402 | 4369.920 | 0.000–1187.840 | 3182.080–4369.920 |
| row-balanced | 3 | 402 | 12515.840 | 6284.800–10704.896 | 1810.944–6231.040 |
| row-balanced | 4 | 402 | 12390.400 | 5914.624–10455.040 | 1935.360–6475.776 |
| row-balanced | 5 | 402 | 12456.448 | 5968.896–10535.424 | 1921.024–6487.552 |
