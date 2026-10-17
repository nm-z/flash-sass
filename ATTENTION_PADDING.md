# Masked attention and readable padding

Dense `attnpart` and capped `attnpartn` predicate both QK and V accumulation
on the logical key mask. Invalid K and V rows may contain arbitrary FP32 bit
patterns, including NaN and either infinity, independently of one another.
Zero probability alone would be insufficient: IEEE `0 * NaN` and
`0 * infinity` are not zero. Valid operations retain their arithmetic and
reduction order.

This is a hardening contract for invalid padding. Valid Q/K/V must be finite
and scaled within the existing arithmetic domain. Finite inputs alone do not
guarantee finite products, scores, or a nonzero softmax denominator. The bounded
test fixtures use independently generated valid values and change only padding.

## Count and allocation contracts

Raw C `attn_decode_gpu` and `attn_partials_gpu` return 1 for `n == 0` or
`n > UINT_MAX-8191`, before reading inputs or changing outputs; success
returns 0. The checked decode wrapper propagates that status. Empty softmax is
rejected. The scalar limit prevents overflow in `n+8191`; other pointer,
capacity, and numerical preconditions remain caller duties in the raw twins.

The SASS address proof below applies to `1 <= n <= 2^21`: the initial
`tbase<<11` byte offset is computed in 32 bits. The application and test CLIs
configure at most 262144 visible positions. Capped attention additionally
requires a budget `1 <= NBLK <= 4096`. These raw kernel parameter preconditions
are caller duties. Checked scalar capacities cover valid rows and do not
include speculative GPU reads.

Predicated arithmetic does not remove vector prefetches. KC and VC must remain
readable through every load, including loads whose results are discarded.

| Buffer | Sufficient physical extent |
| --- | --- |
| Dense blocked K | `round_up(n,1024)+32` positions, 512 FP32 values per position |
| Dense position-major V | `round_up(n,8)+4` positions, 512 FP32 values per position |
| Existing conservative dense allocation | `round_up(n+8192+64,32)` positions for each cache |
| Compact QSA K and V | `round_up(4*(NBLK+1)+2048,32)` positions for each cache |

K positions are physically stored in whole blocks of 32. The compact formula
covers gathered writes as well as prefetches. Gather may write four rows of an
unfinished source block even when fewer are logically visible; attention uses
the logical count rather than that physical row count.

### Derivation of the final loads

For positive logical count `n`, let `q=ceil(n/8192)` and `C=1024*q`.
Partition `c` starts at `c*C` and sets `tend=min((c+1)*C,n)`.
The entry branch skips a partition starting at or beyond `n`, before K/V
loads. Other partitions advance by 1024 while below `tend`. The final
executed tile therefore starts at `B=1024*floor((n-1)/1024)`.

K addresses start at `2048*B + 32768*kv + 65536*warp + 16*lane`.
Four score slots add `slot*0x80000`. The final unused QK prefetch accesses
dimension chunk 64, adding `64*0x200`. With `kv=1`, `warp=7`,
`lane=31`, `slot=3`, and the full 16-byte vector, the exclusive byte bound
relative to KC is `2048*round_up(n,1024)+512`. Whole blocked K storage for
`round_up(n,1024)+32` positions suffices. Earlier tiles have smaller bounds.

V executes `I=ceil(min(n-B,1024)/8)` iterations in the final tile. A/B streams
load positions `B+8*i+r` and `B+8*i+r+4`; the final unused next-A prefetch
loads through `B+8*I+3`, with `r=tid&3`. The exclusive position bound is
`round_up(n,8)+4`. Earlier full tiles have smaller bounds.

For capped attention substitute its computed `nn` for `n` throughout.
With budget `b`, `nn <= 4*b+3` and gather may write `4*(b+1)` rows.
The compact allocation formula exceeds both derived bounds for every supported
budget 1..4096. Readability is required even when arithmetic predicates reject
the loaded value. The bounds derive from instruction addresses; allocation
canaries detect writes and do not prove the absence of out-of-bounds reads.

### Mask lifetimes

QK predicates P0..P3 describe the thread's four score slots and remain live
through QK accumulation. In V accumulation, stream A's key is
`tbase+(tid&3)+8*iteration` and stream B is four keys later. P0/P1 are
recomputed for these indices before their FFMAs. Dense uses `param_n`;
capped uses `nn`. CTA ownership, tile boundaries and valid arithmetic order
are unchanged.

The V predicates have at least 21 explicitly scheduled cycles before their
first consumer, including stall 6 on both comparisons and the final probability
load. Correctness does not depend on a cache miss hiding predicate latency.

## Fresh allocation, reuse and prefix rollback

Engine allocations are initially zeroed as an implementation choice. The
attention contract does not require invalid rows to remain zero.

A fresh response resets recurrent DeltaNet, convolution and PLE state. It
writes positional K/V rows before they become visible; attention caches,
indexer pool and ring are not fully cleared between responses. A shorter
response can leave an arbitrary old suffix outside its logical length.

Prefix resume preserves positional K/V and completed indexer pool entries
below the snapshot. It restores recurrent matrices, convolution windows, the
PLE ring, and the 2048-byte raw indexer ring needed to finish a partially
completed four-position block. Later rows are overwritten before visibility.
Gather rebuilds compact scratch for the current selection. No padding clear
is necessary for any of these transitions.

These invariants require separate tests: direct compact attention alone does
not exercise append, selection, gather, or model snapshot restoration.
`kern/engine_reuse_test.c` includes the production engine as a validation-only
library and compares identical requests across fresh allocation, longer-to-
shorter reuse and divergent suffix rollback at every prefix residue modulo
four. It checks all three emitted token IDs and all final logits, plus outer
allocation guards. See [the validation guide](validation/issue18/README.md)
for the measured scope and reproduction commands.

## Regression interfaces

`ATTN_POISON_K` / `ATTN_POISON_V` and `QSA_POISON_K` / `QSA_POISON_V`
independently accept `off`, `nan`, `inf`, `-inf`, `finite`, or `max`.
The bounded `finite` mode alternates +3.25 and -3.25 by row, dimension and
KV head. The valid data and reference are generated before padding changes.
The retained `max` mode is an additional historical stress mode.

Dense tests compare pm, pd, po and the merged output. `ATTN_HOSTKV=1`
places the dense caches in mapped host memory; an unset or zero value uses
VRAM. Compact `qsa_test padding VISIBLE BUDGET [ITERS]` covers production
boundaries without constructing a sequential history. It checks every partial,
output, unchanged K/V input and allocation guard.

`qsa_test kv` applies poison after append/gather at its final sequential
position. Visible gathered rows still compare; intentionally changed invalid
rows do not. `QSA_KV_LAST_ONLY=1` retains query/append checks at every position
and compares expensive scalar attention only at the last position. That mode
is explicitly labeled and is not a substitute for the complete direct-position
QSA driver. `QSA_HOSTKV` uses presence semantics: unset it for device memory;
setting it to `0` still selects host memory.

The single Bash [runner](validation/issue18/run.sh) records independent
K-only and V-only NaN, positive infinity, negative infinity and bounded finite
arms, both placements, tails, empty partitions, tile/chunk boundaries, capped
budgets, maximum configured position, controls and timing. Every GPU invocation
is preceded by recorded `nvidia-smi` and `ps` checks and waits for an idle
host. Historical evidence retains its original revision and scope; it is not
a result for a later head.
