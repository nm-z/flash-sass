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
