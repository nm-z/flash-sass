# Two-position validation

The paired pass is opt-in with `FLASH_TWO_POSITION=1`; unset or `0` retains
scalar execution. The setting is read during engine initialization. The pass
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
