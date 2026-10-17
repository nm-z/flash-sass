# Two-position validation

The paired pass is opt-in with `FLASH_TWO_POSITION=1`, or automatically enabled
by `FLASH_PREFILL_CHUNK` sizes above one. Explicit `FLASH_TWO_POSITION=0` retains
scalar execution. The settings are read during engine initialization. The pass
consumes two known prompt IDs, stopping before the final prompt position.
It is not a speculative accept/reject implementation.

## Component checks

Build with the repository's CUDA 11.4 / MaxAs toolchain:

```sh
make -C kern die.cubin die_test pair_prog_test pair_bf16_plan_test pair_state_test pair_bf16_test
cd kern
./pair_prog_test
./pair_bf16_plan_test
# Select and lock a free M60 die before the following GPU commands.
./die_test
./pair_state_test
PAIR_MIXED_WARP=1 ./pair_state_test
./pair_bf16_test
```

- `pair_prog_test` checks token offsets, unchanged scalar descriptor parameters,
  logical exchange counts, rejection of incompatible programs, and preservation
  of the output program when pairing fails.
- `pair_bf16_plan_test` checks all five fusion mappings and fallback for different
  weights/dimensions, token masks, aliases and unverified peer destinations.
- `pair_state_test` compares scalar-position order with paired-phase order on the
  same convolution, DeltaNet, quantizer and argmax kernels. It compares every byte
  of the recurrent matrices/rings and both positions' outputs, validates argmax
  independently, and checks token counts and grid generations. Starts include
  convolution-ring boundaries and positions 127/128 and 2047/2048. The mixed-warp
  variant leaves seven warps idle in the active quantizer CTA to exercise replay
  rendezvous with unequal work.
- `pair_bf16_test` compares each shared-weight BF16 variant with two real scalar
  phases. It covers non-multiple row counts, virtual CTA loops, separate activations
  and biases, twelve XOUT destinations, different output offsets, and guard bytes.

These tests compare scheduling and arithmetic equivalence with B1. They do not
establish new agreement with an external model oracle.

## Full 48-layer comparison on archy

The full-model comparison harness uses archy's existing six IPC weight holders
and GGUF files. It launches separate short-lived cores, obtains
compatible shared GPU locks, uses a small 128-position cache, and sends commands
only to its own stdin/stdout cores. The resident serving process is not reloaded.
Review the paths and six-die split in the runner before using another machine.

The machine-specific comparison harness is maintained outside this branch. The
PR validation record identifies the tested source revision, commands, effective
configuration, binary hashes and result directories. Build its engine artifacts
with `make -C kern die.cubin libengine.so flashcore tok_test`.

The comparison checks generated token arrays, completed-position counts, and
`statehash` readbacks after identical request sequences. Readbacks cover all live
DeltaNet and convolution states, the PLE ring, valid positional KV codes/scales,
indexer key rings/pools and final vocabulary logits on each die. They exclude
transient compact-cache padding outside the causal window. Checksums are FNV-1a
fingerprints; the component tests use direct byte comparisons.

Cases include one-token prompts, odd and even prefixes, lengths crossing complete
four-token indexer blocks, a real prompt encoded by the model's tokenizer, seeded
sampling, an exact five-position snapshot and its reuse, and stop-token termination.
`FLASH_NBLK=4` makes the longer prompts exercise sparse selection, while retaining
all 48 model layers. This is not a production-size 512-block/deep-context sweep or
multimodal validation.

Each output directory records commands, stdout, stderr, effective configuration,
source revision and binary SHA-256 hashes in `results.json`. Run the final commit,
not an earlier development cubin, when attaching evidence to a review.

The `statehash` command is an explicit validation readback after a completed
request. It is outside the timed generation path. The `config` response and
`gen` statistics report whether pairing is enabled and how many positions used it;
the existing streaming `chat` response format is preserved.

No throughput improvement is claimed from these correctness comparisons. Disk PLE
rows warm between runs, and neighboring GPU activity can change timings. Controlled
warm repetitions and equivalent cache settings are needed for performance claims.

## Chunk boundaries and headless prefix equivalence

```sh
make -C kern prefill_plan_test
./kern/prefill_plan_test
# After recording a successful model comparison at this revision:
python3 kern/prefill_cancel_test.py . /tmp/flash-chunk-cancel-check \
  --reference /tmp/flash-chunk-check/results.json --sha "$(git rev-parse HEAD)"
```

The host planner test needs only a C compiler. It exhaustively checks small prompt,
resume, snapshot and chunk combinations for exact ordered coverage, odd tails and
exclusion of the final prompt position. It also checks cancellation/resume planning,
strict environment parsing and arithmetic near `INT_MAX`.

The external model harness compares each mode with the full-head scalar reference. Chunk
sizes one, two and three exercise B1-only chunks, pairs and odd scalar tails;
64 exercises a chunk larger than the prompt. Automatic pairing and its explicit
scalar override are tested separately. In addition to tokens and state readbacks,
each case checks the actual paired-position and completed-chunk counters against
the expected execution plan. `gen 0` confirms that the teacher-forced full-head
diagnostic is preserved. These are correctness checks, not performance benchmarks.

The comparison matrix uses chunk sizes 1, 2, 3 and 64 plus scalar chunk 64 with
VRAM KV, and repeats chunk 64 with host KV. Its successful `results.json` supplies
the scalar `p17` reference for the cancellation check below. The PR records the
commands, source revision, binary hashes and result paths for completed runs.

The cancellation check starts its own core on a unique temporary Unix socket,
requests a snapshot after five positions, sends a 61-position prompt with chunks
of four, and closes that client immediately. The next connection is a completion
barrier. At most one chunk may complete, the core must remain usable, and a normal
17-position prompt must match the scalar reference's tokens and state. The next
request must not reuse the incomplete snapshot. Zero completed positions is valid
when the disconnect is observed before the first launch; this case establishes
boundary cancellation and recovery, not a guaranteed in-flight abort observation.
The runner checks the source revision, binary hashes and configuration against
the reference before comparing. It sends no commands to the resident server.
