# Masked attention and readable padding

The dense and capped attention kernels predicate both QK accumulation and V accumulation on the logical key mask. An invalid key's K or V bytes may therefore contain arbitrary FP32 bit patterns, including NaN and infinities, without participating in arithmetic. Zero probability alone is insufficient: IEEE `0 * NaN` and `0 * infinity` are not zero. Valid Q/K/V retain the existing arithmetic and reduction order.

This change does not remove speculative vector loads. Allocations must still cover each kernel's prefetch extent. Dense `attnpart` retains its conservative `n + 8192 + 64` readable-position allocation contract; the blocked K allocation must also round to whole 32-position blocks. The compact QSA cache retains the engine/harness's `4*(NBLK+1)+2048` readable positions, rounded to 32. The gather itself may write all four rows of the unfinished source block although fewer rows are logically visible. The mask uses the logical length, not that physical row count. Padding contents need not be cleared on fresh allocation, reuse or rollback, but pointers and readable extents must remain valid.

QK predicates P0..P3 correspond to the thread's four score slots and remain live through QK accumulation. In V accumulation, stream A's key is `tbase+(tid&3)+8*iteration` and stream B is four keys later. P0/P1 are recomputed for those indices before their FFMAs. Dense uses `param_n`; capped attention uses its computed `nn`. Neither change changes CTA ownership, tile boundaries or the order of valid operations.

## Regression controls

`ATTN_POISON_K` and `ATTN_POISON_V` independently accept `off`, `nan`, `inf`, `-inf` or `max`. The dense test computes the reference from the valid data, then poisons only GPU padding. It compares maxima, denominators, weighted partials and final output.

`QSA_POISON_K` / `QSA_POISON_V` apply the same policies to compact-cache padding at the final sequential test position, after append/gather and before capped attention. Visible gathered rows still compare; intentionally changed invalid rows are excluded from the gather comparison. `QSA_KV_LAST_ONLY=1` keeps append/query checks at every position and compares expensive scalar attention at the last position. This is labeled explicitly and is not a substitute for the separate direct-position all-stage test.

The regression matrix must include independent K/V poison, tails, empty partitions, tile/chunk transitions and both KV placements. Real model cache reuse/snapshot flows and masking overhead require separate whole-model validation. This policy is about masked padding; it does not promise finite output for arbitrary unbounded or nonfinite **valid** input values.
