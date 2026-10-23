# Saved first five-position verification profile

`generic`: canonical job `e87e07d5-7d82-4e74-9e1f-f1da9e2e330d`.

`native`: canonical job `ed1ecc89-60f1-4835-ab47-5116009e06b9`.

`receiver_quant`: canonical job `06793fe8-3008-4abc-b98a-65a21db8feaf`.

Full48-layer model;64 prompt tokens and128 committed outputs; four-draft prompt lookup with greedy acceptance and recovery. Each profile covers the first five-position target call (start66). Values are inclusive device phase intervals, averaged over six dies, in **µs per five-position pass**. They include waits and are not isolated arithmetic or request averages. Phase0 has no preceding timestamp. These are separate observations, not a randomized paired experiment.

| Operation| generic µs| native µs| receiver_quant µs|
|---|---:|---:|---:|
| `q2moegn`| 19847.5| 19753.3| 19705.2|
| `hcupq_x`| 19628.4| 30567.8| 10859.0|
| `q2moedn`| 15805.8| 15866.2| 15828.7|
| `router_full`| 10922.8| 12292.9| 10597.5|
| `grwrite_read`| 9671.3| 9716.2| 9643.7|
| `bf16hcrwx`| 9507.7| 9504.9| 9531.6|
| `q3kk2560`| 9200.1| 9924.6| 9045.8|
| `qsa_full_bal`| 6506.0| 6418.1| 6452.6|
| `gdn_full`| 6149.5| 6159.2| 6123.3|
| `iq4k2560`| 6638.9| 7151.8| 6569.6|
| `iq4k6144`| 4259.0| 4308.0| 4323.2|
| `q4kk6144`| 4101.3| 4113.2| 4085.9|
| `q5kk2560`| 2902.7| 2930.0| 2908.5|
| `q5kk6144`| 2096.1| 2121.9| 2123.1|
| `q4kk2560`| 1551.0| 1678.2| 1561.6|
| `q6kk1536`| 1587.9| 1607.5| 1572.9|
| `q2k2560`| 1350.0| 1430.2| 1355.1|
| `q3kk6144`| 788.8| 778.6| 792.7|
| `bf16hcrx`| 327.3| 487.3| 501.9|
| `rms2560`| 160.3| 157.7| 175.6|
| `amax`| 116.4| 88.4| 93.4|
| `bf16k2560`| 124.2| 125.8| 113.8|
| `ple_rowsq`| 61.4| 61.6| 61.1|
| `plegate_norm`| 81.6| 79.7| 81.2|
| `pleconv`| 56.8| 57.0| 56.5|
| `hcwsum`| 44.7| 45.1| 42.0|
| **All491 intervals**| **133487.6**| **147425.1**| **124205.6**|

The receiver-local quantization prefix is charged to the following consumer (often `router_full`), while its producer omits quantized peer payloads. Compare the complete pass as well as `hcupq_x`. CSV retains each die and phase count.
