# PLE serving result

Three requests went through the existing HTTP frontend and six-die resident
model after loading the PLE file into a sealed, locked memfd. They used the
original serving cubin, all 48 layers, 512 selected blocks, 230,000-position
capacity, 4,096 image rows, scalar prompt processing, and one PLE reader.

| Request | Prompt tokens | Generated tokens | Prompt tokens/s | Decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Same prompt the user reported at 16.9 decode tokens/s | 65 | 256 | 35.57 | 35.19 |
| Fresh explanation prompt | 67 | 194 | 35.53 | 35.24 |
| Fresh explanation prompt | 65 | 139 | 35.54 | 35.31 |

The first request reached the 256-output request cap; the user's earlier
response had 264 outputs. Decode uses the frontend's generated-minus-one
interval count, not the total generated count. This is a serving-path result,
not a new kernel-speed claim or evidence that the 120–144 target was reached.

The helper reported 28,800,138,432 copied bytes and 7,031,284 resident pages.
`smaps_rollup` reported 28,125,136 KiB locked and 28,127,216 KiB RSS. Dirty pages
were about 2.5 MiB with zero writeback in the post-run sample. The engine loaded
host code from commit 5601c92; the RAM loader's completed path is the same as
906ee6f, which additionally rejects an already-existing ready file before loading.

The earlier alternatives failed: the direct NFS mapping did not retain all
pages despite `mlock` returning success, and a buffered copy to the local
filesystem caused approximately 17 GB of dirty writeback and regressed fresh
HTTP decode to 3.02–4.29 tokens/s. That copy was removed and its deployment
reverted before this RAM-file result. The RAM loader uses anonymous tmpfs pages,
locks each completed chunk, verifies full residency, and publishes its descriptor
only after sealing it.

The machine-readable record retains the actual frontend timing objects, wall
durations, first-text observations, and effective engine configuration. The
serving helper and engine descriptors keep this RAM source alive. Recreating it
requires enough available RAM for the complete file plus the loader's reserve.

Refs #82.

## Kernel and paired-prefill deployment

The subsequent curated kernel build uses direct jump-table dispatch, four-row
HC-down geometry, compact HC slice loops, fuller HC-write geometry and coalesced
peer writes. It retains the original shared phase state and barriers. The
register-state and single-publication experiments that regressed are absent.
Shared RoPE tables allow paired prefill with the full capacities above.

The kernel SHA-256 is
`7aca64b21ea9904dce3c278a944785b673d150cedeeaff2b6fe1219d6043fc92`, byte-for-byte
the image that completed two 2,048-token prompt runs. The source integration
preserves master's optional timing instrumentation and watchdog handling.

With the RAM source active in both arms:

| Matched measurement | Original kernel | Candidate |
| --- | ---: | ---: |
| Short decode, tokens/s | 35.5898 | 37.7343 |
| 512-token prompt, ms to first output | 14,463.8 | 10,308.8 |

All output token IDs matched across both arms. The candidate improves measured
decode by 6.03% and prompt throughput by 40.31% in this comparison.

Fresh HTTP requests on the combined serving configuration measured 37.28 and
37.29 decode tokens/s, with prompt rates 50.03 and 49.96. A 1,574-token natural
inventory prompt completed at 47.65 prompt tokens/s and 34.59 decode tokens/s;
its visible answer was the expected `17`. The candidate remains deployed with
230,000-position capacity, 4,096 image rows, `rope_owner=5`, `two_position=1`, and
`prefill_chunk=64`.

`kernel-serving.json` retains the matched prompt/output IDs, arrival times,
engine timing records and fresh HTTP timings. These results remain below the
120–144 decode tokens/s objective. No graph-executor result is included in the
serving candidate: that experimental route was slower.
