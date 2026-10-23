# Current work toward60 TPS

The60TPS goal remains active and unachieved. GPUs2 and7 remain reserved.
Retained scalar default: v104,25.72–25.79TPS,492 physical phases and maximum
pairwise assigned-work ratio1.076923077. Scalar packet candidate v119 measures
25.97/25.95TPS; both requests match all576 state hashes and256 frozen IDs,
and actual492-phase descriptors pass all7,913 work-component checks.

The complete greedy speculative controller includes acceptance/rejection,
snapshot/recovery replay, PLE restaging and accounting for discarded work.
Its proposer is prompt lookup. Native five-position GR-up shares the weight
stream and sends only FP32 Z; the consumer reconstructs exact quantization
locally. Verification falls from about136.2 to126.5ms per five-position call.
Complete requests measure11.06/11.19TPS,16/205 drafts accepted,689.2578125
synchronization phases/committed token. The final scalar program's492 phases
do not establish compliance for the complete speculative request.

Preserve the causal batch handoff before the end decision, including the last
position. Omitting it caused282 differing state hashes despite identical IDs.
The diagnostic all-phase joins are not enabled in the retained controller.

Objective weights: throughput1.0, fidelity1.0, phase/work requirements1.0,
engineering time0.8, cost0.8. The lower-effort communication alternative changes
existing tiny peer stores' issue schedule: predicted small gain, low cost.
The higher-effort alternative collects a whole die's small GR-down vector,
then sends contiguous chunks: predicted fewer peer transactions, with a local
completion wait. Selected and implemented the latter across all97 GR-down
reads. The observed scalar improvement is small; retain it as an opt-in path.

GR packet preserves each CTA's dot rows. CTA0 waits for local readiness then
publishes; other CTAs proceed to the existing footer. No new full-grid phase
or peer exchange is introduced. Words20..21 reference position-private ldown;
each launch clears readiness. A module marker prevents incompatible images.

The stage trace shows payload costs beyond exchange counts. The final native
HCUP's die0 active CTAs spent about227µs publishing FP32 plus quantized forms;
receiver-local quantization reduces this stage to4.096µs. Projection remains
18.432µs. This is one instrumented operation, not a97-phase average. Corrected
SM IDs show16 resident CTAs on16 SMs,13 owning matrix rows on die0. The earlier
SM field was invalid because MaxAs encoded SR_SMID as lane ID.

Next: native multi-position execution beyond GR-up and a strong balanced
proposer. Existing DFlash2 uses a single extra device, CPU operations and
unchecked drafts; it cannot be enabled unchanged. Cached config: hidden2560,
intermediate7680, five draft layers, block8, feature taps5/15/25/35/45. Native
MTP also needs balanced row ownership instead of unequal whole-expert ranges.

Nate's8.2ms budget remains the accounting reference, not a revised ceiling.
402 peer acquires at the original6µs sum to2.412ms, which does not explain
38.8ms scalar decode. Do not infer a hardware ceiling from current verifier
measurements or the weak lookup acceptance rate.
