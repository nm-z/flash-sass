# QSA numerical validation

Refs #11. This PR has separate gates for native mathematical agreement, frozen
external-reference agreement, target M60 execution, and independent checkpoint
teacher-forced logits. A synthetic CPU pass does not establish the last two.
The PR stays draft while required target and checkpoint evidence is missing.

## Reproduce without Python

From the repository root, with a C compiler, Make, and Perl (including
IO::Uncompress::Unzip, JSON::PP and Digest::SHA):

    perl kern/qsa_oracle.pl --full --out /tmp/qsa-oracle-new-run

The directory must be new. The driver rebuilds the default C binary with strict
floating-point flags, verifies the pinned golden hashes, and runs the full
matrix. It saves raw inputs, scores, selected IDs, attention partials and outputs;
report.json records every command, expected and actual exit, metrics, source and
binary hashes, and start/end host/compiler identity. artifacts.json identifies
the retained data. An assertion, malformed child report, changed source/binary,
or violated numerical limit prevents a suite pass.

An explicit --binary is recorded as a supplied binary with its measured hash;
the driver does not claim to have compiled it. The normal command uses the
rebuilt default binary.

The checked-in goldens contain thirteen original external fixtures. The
remaining historical captures can also be replayed when that archive is
available:

    perl kern/qsa_oracle.pl --full --archive /path/to/oracle-run2 --out /tmp/qsa-archive-new-run

That optional archive is not needed for the portable full matrix. New native
inputs use the documented xorshift64star generator; they are not byte-for-byte
reproductions of the retired NumPy generator. Preserve old NPZ files as frozen
evidence. The C/Perl port never executes Python or Transformers.

## Reference identity and independence

The external indexer is
[Qwen4ExpTextQSAIndexer at Transformers commit 8498ed425fa01bbee7d086efdf1a6634c2962aed](https://github.com/huggingface/transformers/blob/8498ed425fa01bbee7d086efdf1a6634c2962aed/src/transformers/models/qwen4_exp/modeling_qwen4_exp.py).
The source blob is 946f5fd6e9ba21347654b1b2ca196e86b0f751b3.

The historical capture executed the required definitions with observation
before top-k. Its separate attention reference used PyTorch float64 dot
products, softmax, weighted sums and gates. The original report SHA-256 is
d213b972c0e20552c45e6b2fd880036618090430d4893dbdb4e686d7a46e3b41;
the original artifact catalog SHA-256 is
e6588cc3686becef542f02a8f42de7483ae8e11fe160a5b08b482ec3eaf86280.
Both identities and each included fixture are verified by the Perl driver.

The new mathematical reference in kern/qsa_oracle.c uses independent scalar
binary64 reductions and explicit dtype boundaries. It does not call the
candidate's normalization, scoring, selection, quantizer, or attention routines
to construct the mathematical attention or score reference. Archived external
outputs check the transcription separately; generated cases compare the
candidate against the native mathematical reference. This is not a fresh
execution of the external framework or an original-checkpoint quality result.

## Arithmetic and dtype contract

The runner requires little-endian IEEE binary32/binary64, FE_TONEAREST and
FLT_EVAL_METHOD=0. It builds with -fno-fast-math, -ffp-contract=off and
-frounding-math. Explicit candidate fmaf operations remain explicit.
The host guard is tested under FE_DOWNWARD and then restored to FE_TONEAREST.

The pinned indexer first accumulates the four-key mean in FP32 and casts it
back to the raw-key dtype before RMSNorm. The frozen BF16 probe verifies that
rounding boundary. RMSNorm uses the stored offset convention gamma=1+weight;
native fixtures retain both BF16-rounded offsets and full FP32 gammas. The
native mathematical RMS reduction is binary64 and is explicitly compared with
frozen external outputs; it does not pretend to reproduce every backend
reduction tree. The candidate stores pooled keys in FP16.

The reference divides scores by sqrt(128). Positive scaling preserves strict
order over exact real numbers. Rounded FP32 division can merge adjacent
scores into a cutoff tie. The divisor represented in FP32 is
11.313708305358887. The preserved adjacent pair
1.4142136573791504 and 1.41421377658844 both divide to
0.1250000149011612 in the witnessed operation sequence. Both assignments to
the lower/higher block ID are tested, together with equal-score and zero-score
fixtures and two complete raw-key-to-cutoff witnesses.

Candidate top-k uses deterministic lower-ID ties. External top-k may select a
different admissible set. Checks require unique in-range IDs, every score
strictly above the cutoff, the declared tie policy for each implementation, and
identical sets when the measured error and cutoff margin imply stability.
Selection overlap and its attention-output effect are retained separately.

Only contiguous causal prefixes are covered. All-zero index scores must be
canonical positive zero; pooled zero values may have either sign after RoPE.

## Enforced limits and decomposed errors

All normalized RMS metrics use

    RMS(candidate - reference) / max(RMS(reference), 1e-8)

Pointwise relative error is reported with the same denominator floor but is
not substituted for the normalized RMS gate.

| Comparison | Maximum absolute error | Normalized RMS limit |
|---|---:|---:|
| Scaled index scores | Reported | 0.005 |
| Same-ID attention reduction | 0.00003 | 0.00003 |
| Same-ID KV quantization effect | 0.025 | 0.025 |
| Supplied teacher-forced logit rows | 0.10 | 0.01 |

The score regression uses reference [1,0,0,0] and candidate
[1,0.004,0.004,0.004]. The obsolete maximum-error/maximum-reference check
accepts it, but its normalized RMS is 0.0069282032302755096.
The corrected executable returns 3. The suite requires that failure.

Fresh synthetic fixtures separate selection-only error, the same-selection
KV quantization effect, and attention reduction error. Frozen probes also
compare independent native math with external results. A reported
selection-only measurement is not an invented whole-model acceptance limit.

The INT8 quantizer derives codes from an FP32 maximum and rounds the stored
scale to FP16. Tiny nonzero groups can reconstruct to zero; the runner records
that loss. It checks scale/code/dequantization behavior independently and
rejects nonfinite or out-of-range inputs. Production range handling remains
the subject of #19.

Attention buffers contain finite zero padding over the complete partition
sweep. Invalid K and V padding are rejected at the harness boundary.
This does not establish arbitrary poisoned-padding safety in production SASS
(#18), or all public API capacity checks (#20). Empty attention is rejected.
The host exponential approximation is checked at 17,401 points in [-87,87],
at clamp boundaries, and alongside signed-zero reciprocal behavior.

## Coverage

Full generated indexer plus sparse-attention lengths are 1, 2047..2052,
3100..3103, 8000..8003, 65000..65003 and 262000..262003: 23 cases covering
all four pooling-tail residues at the long positions.

Same-ID attention lengths are 1..4, 1023..1027, 2047..2052 and 8191..8193:
18 cases covering tile and partition boundaries and empty partitions.

The portable suite also checks six pinned cutoff fixtures, the BF16 boundary,
index length 2052, and attention lengths 1,2,3,4,1025. The host arithmetic
checks and required failing score control are separate invocations.

## Target M60 path and shared-host rule

Build the production image using the repository's pinned MaxAs patch and
CUDA 11.4 ptxas, then build the C driver:

    make -C kern die.cubin dieops.h
    make -C kern qsa_oracle_gpu

Every target run must go through the shared-host gate:

    bash kern/qsa_oracle_gpu_gate.sh --out /tmp/qsa-gpu-gate-new --wait 180 -- \
      perl kern/qsa_oracle.pl --full --gpu kern/die.cubin \
      --binary kern/qsa_oracle_gpu --device 0 --out /tmp/qsa-gpu-new-run

The gate checks nvidia-smi and ps, acquires
/tmp/flash-sass-engine-validation.lock, rechecks, and holds the lock throughout
the run. Resident holders or another worker's model keep the host busy even
when utilization is zero. It waits and returns 75 if the host remains busy;
it never kills other workers' processes. Gate logs contain process data;
publish only the relevant sanitized evidence.

The C adapter executes production IDXQK normalization/RoPE/pooling in bounded
batches, IDXSCORE and five IDXSEL passes with the actual GPU-produced histogram,
and ATTNPARTN/ATTNMERGE. It saves the observed candidate tensors and partials,
requires SM52, propagates CUDA failures, and records launches. A requested
target fixture cannot silently use the CPU backend. A frozen zero-block
index fixture is explicitly a no-op; generated length 1 still exercises
target normalization and attention.

Dense all-block selection and fixture construction run on the host. Device KV
quantization and whole-model logits are not covered by this adapter. Compiling
against the CUDA stub library is a build-only check; do not put the stub
directory in the runtime library search path.

## Checkpoint logits remain a separate acceptance gate

    make -C kern qsa_logits_check
    kern/qsa_logits_check reference_ids.u32 candidate_ids.u32 \
      reference_logits.f32 candidate_logits.f32 ROWS VOCAB result.json

This C checker requires exact little-endian file sizes, identical in-range
input IDs, finite logits, and unchanged inputs. It enforces both fixed limits
for every row and the aggregate, and reports top-1 agreement. Exit 0 means a
numerical pass, 3 a numerical failure, and 2 malformed input or I/O failure.
All reports explicitly retain independent_checkpoint_evidence=false and
provenance_status=not_verified: arrays alone cannot prove how they were made.

    perl kern/qsa_logits_check_test.pl --checker kern/qsa_logits_check --out /tmp/logit-checks-new

The 32 synthetic regression cases include absolute-only and NRMS-only
failures, the near-zero denominator floor, a bad row hidden by aggregate
metrics, mismatched IDs, nonfinite logits, malformed shapes, overflow, and
preservation of an existing result file.

Issue #11 still needs measured full-checkpoint teacher-forced candidate and
independent reference logits from identical token IDs, verified original
checkpoint/tokenizer and GGUF shard identities, conversion source revision,
runtime/build identities, full vocabulary coverage, and per-layer QSA
selection/output diagnostics. The pinned GGUF repository/revision in the
historical work is ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF at
ed59f92082b1e93c0e96d60a8b11aab089b52f09; that metadata alone does not establish
the original checkpoint or conversion lineage. Same-implementation placement
comparisons and isolated block-3 captures from #9/#22 do not supply this gate.

## Recorded evidence

Current native reports, logs, manifests, build identity and sanitized archy
idle checks are under validation/issue11/native. The portable frozen data and
its provenance are under validation/issue11/goldens. Generated large buffers
are represented by measured hashes and can be regenerated with the command
above; they are not claimed as checked-in files.

The original validation/issue11/report.json and run.log remain historical.
That run reported PASS_SYNTHETIC, but its score check enforced maximum
absolute error relative to the largest reference score. Its reported
normalized RMS was not the enforced gate. Preserving those captures does not
retroactively correct the old runner or imply target/model conformance.
