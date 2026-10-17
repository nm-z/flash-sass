# Concurrent sequence slots

`FLASH_SEQUENCES` selects 1–16 independent sequence slots. It defaults to 1. The engine divides the configured total position budget among the slots, rounding each slot down to a multiple of 128 positions. For example, `FLASH_ENGINE_MAXT=8192 FLASH_SEQUENCES=2` gives each slot 4096 positions. A multi-slot configuration that cannot provide at least 128 positions per slot is rejected.

Each slot owns its token IDs, sampling random values, raw INT8 KV, pooled indexer keys and raw-key ring, DeltaNet matrices, convolution histories, PLE history, image positions, and prefix snapshot. Weights and temporary activation buffers are shared. In disk PLE mode each slot also owns its staged rows and readiness flags. VRAM remains the default placement selected by the serving launcher; `FLASH_KV_HOST=1` explicitly selects mapped host KV.

## Scheduling

The core accepts concurrent socket connections. Admission waits for an available slot; `chat_slot <slot> <n_new> <ids...>` and `gen_slot <slot> <n_new> <ids...>` select a particular slot. Ordinary `chat` and `gen` use the first available slot. Slot allocation serves the oldest eligible waiter, and waiting clients can cancel. A request for an occupied explicit slot does not block an unrelated free slot. The HTTP frontend keeps its own connection for each request and locks only the tokenizer's shared BPE scratch.

GPU work still uses the existing one-position kernels. A FIFO scheduler gives each admitted request a turn. During decode a turn runs one position. During prefill it runs at most `FLASH_QUANTUM` positions (1–128, default 1); it yields at the prefill boundary and at a requested snapshot. This lets short requests make progress during another request's long prefill. It does not promise increased aggregate token throughput or reuse a weight read across multiple positions. The two-position kernel work is separate.

Device exchanges and grid-barrier counters keep increasing across slot switches. Only the token, end and abort counters are restored for the next turn. A slot's state stays allocated until engine unload; releasing a request makes that slot available and retains its own prefix snapshot. A CUDA failure marks the scheduler broken so waiting requests fail and the core can drain callers before rebuilding its contexts. Reload and shutdown close admission and wait for active sessions before unloading the library.

`info` now returns `info <positions-per-slot> <image-capacity> <slots> <prefill-quantum>`. `config` includes `sequences` and `quantum`. The `done` record appends the actual slot ID; HTTP timing records include `sequence_slot`. Existing readers of the original leading fields remain compatible.

## Validation

CPU scheduler regression:

```sh
make -C kern seqsched_test
./kern/seqsched_test
```

The GPU integration check needs a dedicated core socket running this exact build, at least two slots and 128 positions per slot, and matching resident weight-holder manifests. It compares serial and interleaved token IDs under greedy and fixed-seed sampling, independent prefix snapshots, stop tokens, cancellation and slot reuse. A short request must finish while another slot is still prefilling. When image storage is enabled, it also compares an image/M-RoPE slot with a simultaneous text-only slot.

```sh
python3 kern/sequence_check.py /path/to/test-core.sock
```

Use both `FLASH_QUANTUM=1` and a larger quantum, and test both KV placements when memory permits. Set `FLASH_NBLK=1` in a small-context integration run to exercise selection and gather while two slots are interleaved; also preserve a run with the production budget of 512. A CUDA build or CPU scheduler pass alone does not establish model output equivalence. Exact tested commits, commands and results belong in `validation/issue-7/`.
