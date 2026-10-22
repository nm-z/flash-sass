# Next candidate: distribute activation preparation

Current validated default is v91:24.793978201 TPS,492 phases. The60-TPS goal
remains unchanged. The following is a proposed next full-kernel change, not
implemented behavior or an achieved performance claim.

The current gate/up group broadcasts FP32 values to all six dies. Each die
then repeats the same7040-value SiLU/up multiplication and quantization before
its routed-down rows. Consider assigning each complete32-value quantization
group to one die, with group owner `group_index % 6`. There are220 groups,
so dies receive36 or37 groups, within the pairwise work bound. Weight rows
and down-projection ownership remain unchanged.

Each owning warp must read the four eight-row source stripes from the
existing per-die `balance_gate_local` and `balance_up_local` buffers. Those
buffers remain rank-major with104/112 local rows. Shared-expert rank10 follows
the ten routed ranks. A CPU-built immutable pointer map can select the
correct source and local row without changing expert IDs or arithmetic.
Gate/up still needs its system-visible producer release before remote reads.

Do not send four separately addressed quantization arrays for every group:
that may replace the saved bytes with more PCIe transactions. A possible
transport record contains64 code bytes plus S32, both S16 sums and the FP32
activation scale (80 useful bytes), aligned to a fixed record stride. The
complete K640 consumers—Q2 routed/down/shared and all B32 shared-down
formats—must consume that record directly, preserving every original value.
The current mutable activation loads are coherent `.CG` loads; do not turn
them into stale read-only-cache accesses.

All six dies must observe completion of distributed quantization before any
down matrix reads its local received records. The current preparation join
is local, so it is insufficient for this layout. Implement the real system
release/acquire dependency within the existing down phase, retain epoch
uniqueness, and account for its added communication explicitly. Fewer bytes
does not establish a speedup if that dependency or peer-read latency costs
more than it saves.

Implement producer publication, ownership, the complete consumer format and
the cross-die readiness path together before a GPU run. Then execute the full
48-layer model and head through GPUq, using the existing reference and state
fixtures. Do not substitute a partial-layer kernel, lower-quality outputs,
unchecked proposals, or an averaged-work gate for the60-TPS objective.
