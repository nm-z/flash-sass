# Concurrent sequence slots

`FLASH_SEQUENCES` selects 1–16 independent sequence slots. It defaults to 1. The engine divides the configured total position budget among the slots, rounding each slot down to a multiple of 128 positions. For example, `FLASH_ENGINE_MAXT=8192 FLASH_SEQUENCES=2` gives each slot 4096 positions. A multi-slot configuration that cannot provide at least 128 positions per slot is rejected.

Each slot owns its token IDs, sampling random values, raw INT8 KV, pooled indexer keys and raw-key ring, DeltaNet matrices, convolution histories, PLE history, image positions, and prefix snapshot. Weights and temporary activation buffers are shared. In disk PLE mode each slot also owns its staged rows and readiness flags. VRAM remains the default placement selected by the serving launcher; `FLASH_KV_HOST=1` explicitly selects mapped host KV.

## Scheduling

The core accepts concurrent socket connections. Admission waits for an available slot; `chat_slot <slot> <n_new> <ids...>` and `gen_slot <slot> <n_new> <ids...>` select a particular slot. Ordinary `chat` and `gen` use the first available slot. Slot allocation serves the oldest eligible waiter, and waiting clients can cancel. A request for an occupied explicit slot does not block an unrelated free slot. The HTTP frontend keeps its own connection for each request and locks only the tokenizer's shared BPE scratch.

With multiple slots, GPU work uses the existing one-position kernels. A single slot preserves the two-position prefill path from master. `FLASH_TWO_POSITION=1` requires `FLASH_SEQUENCES=1`; the combined setting is rejected during initialization. A FIFO scheduler gives each admitted request a turn. During decode a turn runs one position. During prefill it runs at most `FLASH_QUANTUM` positions (1–128, default 1); it yields at the prefill boundary and at a requested snapshot. This bounds how much prefill work one turn can hold. It does not promise increased aggregate token throughput or reuse a weight read across multiple positions.

Device exchanges and grid-barrier counters keep increasing across slot switches. Only the token, end and abort counters are restored for the next turn. A slot's state stays allocated until engine unload; releasing a request makes that slot available and retains its own prefix snapshot. A CUDA failure marks the scheduler broken so waiting requests fail and the core can drain callers before rebuilding its contexts. Reload and shutdown close admission and wait for active sessions before unloading the library.

`info` returns `info <positions-per-slot> <image-capacity> <slots> <prefill-quantum>`. `config` includes `sequences`, `quantum` and `two_position`. `statehash` requires one sequence slot and waits for the whole request to release that slot before reading state. The `done` record appends the actual slot ID; HTTP timing records include `sequence_slot`. Existing readers of the original leading fields remain compatible.

Snapshots contain a text prefix. Introducing an image before an existing snapshot boundary must invalidate that snapshot even when all placeholder token IDs match. A request interrupted before its requested snapshot boundary reports no new snapshot.

## Acceptance checks

The Perl driver `kern/sequence_check.pl` uses real model requests and ordered engine turn observations. It requires all 48 layers on six dies, at least two slots, enough image storage, and an explicitly expected build revision. It records every prompt, full token stream, completion record, configuration and the observation records used by each assertion.

The required assertions are:

- Greedy and fixed-seed token equality between serial and interleaved requests, including exchanging their slot assignments.
- Independent snapshots at different prefix lengths. Invalidating one slot's prefix must leave the other slot's snapshot and output intact.
- A long request has completed a prefill turn before the short request is submitted. The short request must finish before the long request reaches its final prompt position. Finishing the short request first without this ordering evidence does not pass.
- A request is demonstrably active when its client disconnects. Its observed completion must be aborted, far short of its planned end, and its slot must produce the original baseline after reuse. Eventual slot reuse alone does not pass.
- Nonzero image embeddings and different M-RoPE positions remain isolated from simultaneous text and image requests. Image-to-text reuse and introducing an image inside a cached text prefix must agree with cold execution.

`FLASH_SEQUENCE_TRACE=/absolute/path/turns.jsonl` enables JSONL observations for a dedicated validation core. The trace records ordered request admission, completed GPU turns and request termination, including after the client disconnects. It is disabled by default. This observer establishes execution order and position counts; it is not a performance benchmark.

## Reproduction

Run the CPU regressions from committed source with `kern/sequence_cpu_check.sh NEW_OUTPUT_DIR`. This checks scheduler admission and turns, prefill boundaries, the checker's ordering predicates, and HTTP disconnect detection while engine output remains buffered. It also builds a control that omits the HTTP disconnect check and requires that control to fail the same socket regression. These checks do not execute CUDA or replace the full-model matrix.

Build the exact committed source before running a matrix:

```sh
revision=$(git rev-parse HEAD)
make -C kern CUDA=/opt/cuda FLASH_REV="$revision" \
    libengine.so flashcore flashd die.cubin seqsched_test output_tokens_test
./kern/seqsched_test
```

Use `sequence_validate.sh` on the same host as the six resident weight holders. Supply the model's placement explicitly. The launcher takes `/tmp/flash-sass-engine-validation.lock`, checks `nvidia-smi` and `ps`, waits for other GPU work or resident cores to clear, and only starts a private core when that gate passes. It never stops another process. Output directories must be new; core sockets use a generated temporary directory. Build prerequisites, including the patched MaxAs checkout, are described in `README.md` and `maxas-patch/`.

Example placement for the current archy holders:

```sh
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,7
export SPLIT_SHARES=112,97,81,116,97,97 ATTN_DIE=1
export ATTN_MAP=1,1,1,1,4,4,4,4,5,5,5,5
export FLASH_ANCHOR_DIE=7 FLASH_MAXT=1024 FLASH_ENGINE_MAXT=1024 FLASH_MAXIMG=2
export SEQUENCE_REVISION="$revision"
# PLE_SHARD is the real second GGUF shard; MANIFEST0..5 identify the live holders.
FLASH_QUANTUM=1 FLASH_NBLK=1 FLASH_KV_HOST=0 \
    kern/sequence_validate.sh "$RESULTS/vram-q1-b1" "$PLE_SHARD" \
    "$MANIFEST0" "$MANIFEST1" "$MANIFEST2" "$MANIFEST3" "$MANIFEST4" "$MANIFEST5"
FLASH_QUANTUM=8 FLASH_NBLK=512 FLASH_KV_HOST=1 FLASH_PREFILL_CHUNK=4 \
    kern/sequence_validate.sh "$RESULTS/host-q8-b512" "$PLE_SHARD" \
    "$MANIFEST0" "$MANIFEST1" "$MANIFEST2" "$MANIFEST3" "$MANIFEST4" "$MANIFEST5"
```

The small QSA budget exercises selection and gather within the bounded slot capacity. The second matrix preserves the production budget while varying the prefill quantum and KV placement, and exercises the merged headless prefill path with four-position chunks. Results establish agreement between serial and concurrent execution of this implementation; they are not an independent model-quality oracle.

For a core already created by the same validation job:

```sh
perl kern/sequence_check.pl --revision "$revision" \
    "$PRIVATE_SOCKET" "$TRACE_JSONL" "$RESULTS/sequence-results.json"
```

`kern/sequence_mutant.sh serial NEW_DIRECTORY` builds a library whose core serializes entire client sessions. `kern/sequence_mutant.sh cancellation NEW_DIRECTORY` builds a library that ignores client cancellation callbacks. Each preserves its exact source patch, revision label, compiler command, status and library hash. Run those libraries through the same launcher using `SEQUENCE_LIBRARY`, `SEQUENCE_REVISION` from `control.json`, and `SEQUENCE_CASE=progress` or `SEQUENCE_CASE=cancellation`. Each check must fail on its named acceptance assertion; a build, connection or configuration failure is not a successful negative control.

## Evidence status

The historical `validation/issue-7/archy-c67d391/` files preserve the earlier driver's actual outputs at source `c67d391c8d04e790d1b25a4ced31e5515c41e7fe`. That driver reported passes for two configurations and recorded serial/interleaved tokens, but its prefill-progress and cancellation assertions were insufficient. Those historical pass labels do not establish the stronger acceptance conditions above, and do not validate later source changes. New measurements and any resource blockers belong beside those retained records with their actual tested revision and exit codes.
