# IQ4 packed lookup and biased dot

Refs #82. This is a small kernel improvement; the 120–144 tokens/s target remains unmet.

The default IQ4_XS lookup uses two PRMTs per four decoded values. The negative table returns `0xff` for an upper-half selector; the positive table, biased by one, returns zero for a lower-half selector. `((lo & 0x7f7f7f7f) + hi)` produces unsigned `level+128` bytes. Its largest byte sum is 241, so there are no carries between bytes. Each integer dot starts at `-128*S32[subblock]`; its final value is exactly the previous signed-weight dot. The floating-point scales and accumulation order are unchanged.

The existing descriptor argument at 0x150 must point to the activation's per-32 int32 sums. Production ph_gemvk already supplies this argument. `IQ4_BIASED_DOT=0` retains the previous arithmetic for comparisons. The default generator emits exactly the assembly measured with IQ4_BIASED_DOT=1.

`ruby identity.rb` checks all 65,536 packed four-value selectors. The resident six-die model was compared reference/candidate/reference using the prefill-start library, complete 48 layers and 512 experts, context230000, image4096, local PLE file, and the existing model's weight arenas. The control benchmark required identical output IDs across every arm.

- Fixed64-token prompt: candidate37.933/37.960 tok/s; reference37.728/37.751 before and37.720/37.717 after. Approximately0.6% improvement on this workload.
- Paired512-token prompt: complete output-ID matches in both repetitions of each arm. First reference TTFT10298.9/10300.1ms; candidate timings are in prefill.jsonl. The final reference drifted to10405.6/10592.5ms, so this run establishes compatibility but is not a strong prompt-speed estimate.
- These are fixed control-socket benchmark results, not fresh HTTP chat-speed claims. Existing real chat baseline is approximately34tok/s.

All GPU comparisons ran through gpuq and restored the original serving configuration. The candidate cubin SHA256 and default assembly equivalence are recorded in provenance.txt.
