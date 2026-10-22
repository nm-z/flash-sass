# Updated residual fingerprint comparison

583680 records; 24320 residual coordinates per request/die. All request/die coordinate sets match. 0 differences across repeated fixed-input requests; 0 differences from GPU0 within the same request. These hashes fingerprint all10240 updated residual values already computed by the original operation. They are diagnostics, not an independent output-correctness oracle.

| Comparison | Request | Die | First position | GR consumer layer | Operation | Phase |
|---|---:|---:|---:|---:|---|---:|

Layer -1 denotes the output head. `grwrite_read` applies the preceding MoE residual update before the named consumer layer; `bf16hcrwx` applies that layer’s attention residual update. The first differing hash brackets a producer/consumer boundary; it does not by itself prove which side is defective.
