# Native five-position GR up/mix/quantization

The first implementation (v112) shares one BF16 weight stream across five
position accumulators, retaining the scalar lane0 dot association. It stages
the original SiLU-transformed input, computes the five projections, and uses
the original gate/mix/quantization/publication body on the resulting raw rows.
It remains inside the complete production kernel; no standalone matrix test
substitutes for the full model. Register allocation remains249.

Canonical job `88e8ada9-1b46-49b8-841d-2689ba6f7b0a` executes the full model,
128 generated tokens, four-draft prompt lookup, acceptance and recovery.
All128 frozen output tokens and288 retained state hashes match; the complete
head audit passes. The new body executes4,753 times (49 five-position passes
x97 GR-up phases). End-to-end committed throughput is10.694023488TPS. The49
five-position target calls total6725.040095ms, or137.245716ms each, versus
about136.18ms for the retained generic five-position target. This observation
shows no throughput gain.

The scalar default is unchanged. Enable this experimental target with
`FLASH_BALANCED_HCUP5=1` and a batch module carrying
`flash_balance_hcup5_abi_v1`. The builder requires five positions, matching
weights/row counts/offsets, sixteen CTAs, and complete128-row tiles. The original
per-die row ownership is retained.
