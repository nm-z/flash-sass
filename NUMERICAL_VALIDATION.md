# QSA numerical validation

This suite compares the local instruction-order scalar routines with a pinned
external implementation and independent float64 attention calculations. GPU/C
parity and checkpoint quality are separate gates. A passing synthetic run is
reported as PASS_SYNTHETIC; it does not establish model conformance.

## Run

From the repository root, with NumPy and PyTorch installed:

    python3 kern/qsa_oracle.py --full --out /tmp/qsa-oracle-new-run

The output directory must be new. The suite downloads the pinned source below,
verifies its Git blob identity, then executes the literal required definitions.
Use --hf-source FILE for an already downloaded copy of that exact blob.
Compiler flags, source hashes, package versions, deterministic inputs, raw arrays
and numerical reports are retained. An assertion or invalid input returns nonzero.

## Reference and arithmetic contract

The external indexer is [Transformers commit 8498ed4, Qwen4ExpTextQSAIndexer](https://github.com/huggingface/transformers/blob/8498ed425fa01bbee7d086efdf1a6634c2962aed/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py#L664-L770),
Git blob 946f5fd6e9ba21347654b1b2ca196e86b0f751b3. Instrumentation only observes
values immediately before top-k. The candidate remains the unmodified local C
routines. Attention uses independent PyTorch float64 dot products, softmax,
weighted sums and gates.

Required host arithmetic: IEEE binary32, FE_TONEAREST, no fast-math, no implicit
FMA contraction. Explicit fmaf calls are retained. The environment guard rejects
a deliberately changed rounding mode and restores the requested mode. These are
instruction-order comparison conditions, not a claim that every framework uses
the same reduction tree.

The external block mean accumulates in FP32 and casts to the input dtype before
RMSNorm. A BF16 fixture tests that boundary explicitly. Stored norm offsets use
the external 1+weight convention. The candidate stores pooled keys as FP16.
That quantization and the order of reductions are separately observable sources
of difference.

Scores are divided by sqrt(128) in the external implementation. Multiplication
by a positive constant preserves order in exact arithmetic; finite-precision
division can collapse adjacent scores to a tie. The suite preserves raw and
scaled score bits and tests two assignments of block IDs to the cutoff pair.
Local radix tie handling and external top-k tie handling need not choose the
same IDs. Acceptance checks that every chosen ID is admissible at that
implementation's cutoff, that all strictly larger scores are included, and
that an adequately separated cutoff produces the same set. It reports overlap
and output effects instead of treating every tie-order difference as a failure.

All-zero index scores must be canonical +0. Pooled zero values may carry either
zero sign after RoPE. NaN and infinity are rejected by the test boundary.
Only contiguous causal prefixes are covered; arbitrary masks are not claimed.

## Quantization, padding and empty inputs

The baseline quantizer derives INT8 codes from an FP32 maximum and rounds its
stored scale to FP16. A tiny nonzero group can therefore reconstruct to zero;
this is measured, not silently treated as bitwise FP32 preservation. The oracle
boundary rejects nonfinite or out-of-range input, including the 1e8 overflow
probe. Production range enforcement is tracked separately in #19.

Attention fixtures initialize finite zero padding over the complete partition
sweep. The local C routine skips masked arithmetic; that alone does not prove
arbitrary NaN/Inf padding safe in SASS. Production masked-padding hardening is
tracked in #18. Empty decode is rejected before the scalar attention call.
Bounds and capacity enforcement are tracked in #20.

exp_gpu is tested over 17,401 finite points in [-87,87], its declared clamp
domain. Reciprocal preserves the sign of zero. These elementary checks support
the specified domain and do not claim arbitrary nonfinite equivalence.

## Acceptance thresholds fixed before execution

| Comparison | Maximum absolute error | Normalized RMS error |
|---|---:|---:|
| Same-ID attention reduction | 0.00003 | 0.00003 |
| Same-ID KV-quantized attention | 0.025 | 0.025 |
| Teacher-forced logits (future independent capture) | 0.10 | 0.01 |

The score gate is normalized RMS <=0.005, with detailed absolute/relative errors
also reported. Near-zero denominators can make pointwise relative error large.
Selection-only, KV-only and reduction-only errors are measured separately.
The declared logit thresholds are not an executed model-quality result.

Full coverage includes visible lengths 1, 2047..2052, all tail residues near
3100, 8000, 65000 and 262000; attention includes 1..4 and boundaries around
1024, 2048 and 8192. The end-to-end cutoff fixture starts with raw indexer
inputs and norm/pool stages, rather than injecting only final score arrays.

## Recorded run

validation/issue11 contains the full machine-readable report, compiler log,
all artifact hashes and small raw cutoff fixtures from the full CPU run.
The complete 466 MB set of generated NPZ inputs and outputs is retained at
/home/nate/claude/flash-sass-prs/issue-11/oracle-run2 on engi and can be reproduced
with the command above. The manifest identifies every file; omitted large arrays
are not represented as checked-in artifacts.

The full run passed the declared synthetic gates. It also reproduced differing
admissible cutoff selections. No target-GPU execution, frozen checkpoint logits,
perplexity result or long-context answer-quality result is implied by this run.
Those remain separate #9, #17 and #22 evidence.
