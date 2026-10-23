# Contiguous GR-down peer publication

v119 keeps all GR normalization, residual updates, dot-product association and
per-die/CTA row ownership. The original producer sends four-row fragments from
each CTA. The new path writes each die's53/54 rows into local position-private
storage, then CTA0 sends two contiguous chunks to each of six destinations.
All16 CTAs publish local readiness; only CTA0 acquires those words. Other CTAs
proceed to the existing footer. CTA0's ordinary peer release covers the payload.
There is no extra peer exchange or full-grid phase.

Descriptor words20..21 reference the existing ldown allocation:64 floats and16
readiness words. Position cloning already gives each speculative position its
own ldown. Every segment launch clears readiness; the outgoing exchange epoch
labels arrivals. The builder requires16 CTAs and at most64 rows, rejects field
conflicts, and forbids the incompatible GR trace/replication modes. The dedicated
image marker must agree with `FLASH_BALANCED_GR_PACKET=1`.

Objective priorities were performance1.0, correctness1.0, phase/work compliance1.0,
time0.8 and cost0.8. Compared with merely rescheduling the four-row stores,
collecting a full packet was expected to save more peer transactions, while
adding a local completion wait. The measured gain is small.

Canonical job `a44645b8-a365-405c-be03-68924ccc8787` runs two complete48-layer/head
requests with64 prompt tokens and128 generated tokens. Decode throughput is
**25.970348940 and25.953858470 TPS**, versus retained v104 observations of
25.721831473/25.792883107TPS. These are separate observations, not randomized
paired estimates. This does not establish a large speedup or achieve60TPS.

Both requests externally match all256 frozen original IDs and576 v78 state
hashes. Same-kernel GPU/stream/next-input and full-head audits pass. The actual
program readback has492 phases and979 descriptors/die. All7,913 work-component
comparisons pass; maximum pairwise assigned-work ratio is1.076923077. See
[work table](work.tsv), [summary](work-summary.txt), [saved per-operation
comparison](operations.md) and [per-die timing CSV](operations.csv).

The4096-token prompt/two-output job `6f58fcb3-285d-4fb8-b36a-f83bd5573f09` emits
the frozen IDs760/1156. All288 state hashes match retained scalar v104 job
`989775df-8bbb-4503-b55f-22a1ec386c18`; the full-head audit passes. This is
scalar long-context evidence, not native five-position speculative coverage.

Artifacts: `/home/nate/work/flash-per-phase-balance-v119/` on archy contains
`die_balanced_grpacket.cubin`, its SASS, the matching library and source snapshot.
Compute uses physical0,1,6,3,4,5; GPUs2 and7 remain reserved. The candidate
remains opt-in and does not replace the retained scalar image.
