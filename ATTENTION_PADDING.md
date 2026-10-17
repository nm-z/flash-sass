# Masked attention and readable padding

The dense and capped attention kernels predicate both QK accumulation and V accumulation on the logical key mask. An invalid key's K or V bytes may therefore contain arbitrary FP32 bit patterns, including NaN and infinities, without participating in arithmetic. Zero probability alone is insufficient: IEEE `0 * NaN` and `0 * infinity` are not zero. Valid Q/K/V retain the existing arithmetic and reduction order.

This change does not remove speculative vector loads. Allocations must still cover each kernel's prefetch extent. Dense `attnpart` retains its conservative `n + 8192 + 64` readable-position allocation contract; the blocked K allocation must also round to whole 32-position blocks. The compact QSA cache retains the engine/harness's `4*(NBLK+1)+2048` readable positions, rounded to 32. The gather itself may write all four rows of the unfinished source block although fewer rows are logically visible. The mask uses the logical length, not that physical row count. Padding contents need not be cleared on fresh allocation, reuse or rollback, but pointers and readable extents must remain valid.

QK predicates P0..P3 correspond to the thread's four score slots and remain live through QK accumulation. In V accumulation, stream A's key is `tbase+(tid&3)+8*iteration` and stream B is four keys later. P0/P1 are recomputed for those indices before their FFMAs. Dense uses `param_n`; capped attention uses its computed `nn`. Neither change changes CTA ownership, tile boundaries or the order of valid operations.

## Actual read extents

For a positive logical key count `n`, let `q=ceil(n/8192)` and `C=1024*q`.
Partition `c` starts at `c*C` and sets `tend=min((c+1)*C,n)`. The entry branch
skips a partition starting at or beyond `n`; each remaining partition advances
by 1024 while below `tend`. Thus the final executed tile starts at
`B=1024*floor((n-1)/1024)`.

K addresses start at `2048*B + 32768*kv + 65536*warp + 16*lane`.
Four score slots add `slot*0x80000`. The final unused QK prefetch accesses
dimension chunk 64, adding `64*0x200`. With `kv=1`, `warp=7`, `lane=31`,
`slot=3`, and the full 16-byte vector, the exclusive byte bound is
`2048*round_up(n,1024)+512` relative to KC. Whole blocked K storage for
`round_up(n,1024)+32` positions suffices.

V executes `I=ceil(min(n-B,1024)/8)` iterations in that final tile. A/B streams
load positions `B+8*i+r` and `B+8*i+r+4`, and the final unused next-A prefetch
loads through `B+8*I+3`, for `r=tid&3`. Its exclusive position bound is
`round_up(n,8)+4`. Both existing conservative allocation contracts exceed these
bounds for supported lengths/budgets. At zero keys neither cache is loaded,
but the checked attention API rejects the zero-key softmax rather than defining
an output for it.

The V predicates have at least 21 explicitly scheduled cycles before their
first consumer, including stall 6 on both comparisons and the final probability
load. Correctness does not depend on a cache miss hiding predicate latency.

## Regression controls

`ATTN_POISON_K` and `ATTN_POISON_V` independently accept `off`, `nan`, `inf`, `-inf` or `max`. The dense test computes the reference from the valid data, then poisons only GPU padding. It compares maxima, denominators, weighted partials and final output.

`QSA_POISON_K` / `QSA_POISON_V` apply the same policies to compact-cache padding at the final sequential test position, after append/gather and before capped attention. Visible gathered rows still compare; intentionally changed invalid rows are excluded from the gather comparison. `QSA_KV_LAST_ONLY=1` keeps append/query checks at every position and compares expensive scalar attention at the last position. This is labeled explicitly and is not a substitute for the separate direct-position all-stage test.

The regression matrix must include independent K/V poison, tails, empty partitions, tile/chunk transitions and both KV placements. Real model cache reuse/snapshot flows and masking overhead require separate whole-model validation. This policy is about masked padding; it does not promise finite output for arbitrary unbounded or nonfinite **valid** input values.
