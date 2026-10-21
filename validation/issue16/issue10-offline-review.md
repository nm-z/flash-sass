# Archived issue #10 evidence review for issue #16

Reviewed the issue #10 acceptance inventory and archived M60 counter component matrix in [PR #38 at head `27d9f5f`](https://github.com/nm-z/flash-sass/pull/38).

## Reusable offline evidence

The archived issue #10 campaign confirms the Maxwell metric groups needed for a gather profile and preserves their raw host/VRAM runs:

- Traffic: `sysmem_read_bytes`, `sysmem_write_bytes`, `dram_read_bytes`, `dram_write_bytes`.
- Transactions: `sysmem_read_transactions`, `sysmem_write_transactions`, `dram_read_transactions`, `dram_write_transactions`.
- Stalls: `stall_memory_dependency`, `stall_memory_throttle`, `stall_sync`.
- Supporting metrics: `achieved_occupancy` and `eligible_warps_per_cycle`.

The archive reports 64 successful profiler commands and 208 metric rows across eight 3,100/8,000-position layout cases. For example, its `nv2000-changing` run is a 2,000-block, tail-0, 8,000-visible-position, budget-512 host/VRAM pair. The archived [traffic profile](https://github.com/nm-z/flash-sass/blob/27d9f5ff9d6c125554aae9ab376a4bf7dd150c60/validation/issue10/counters/workloads/matrix-3100-8000/nv2000-changing/traffic-host0.accepted.json), [transaction profile](https://github.com/nm-z/flash-sass/blob/27d9f5ff9d6c125554aae9ab376a4bf7dd150c60/validation/issue10/counters/workloads/matrix-3100-8000/nv2000-changing/transactions-host0.accepted.json), [stall profile](https://github.com/nm-z/flash-sass/blob/27d9f5ff9d6c125554aae9ab376a4bf7dd150c60/validation/issue10/counters/workloads/matrix-3100-8000/nv2000-changing/stalls-host0.accepted.json), and [CUDA 11.4 metric inventory](https://github.com/nm-z/flash-sass/blob/27d9f5ff9d6c125554aae9ab376a4bf7dd150c60/validation/issue10/counters/profiler/nvprof114-metrics.txt) let the current-head metric groups be prepared without querying the GPU again.

Those profiler runs used source revision `7c6f34d4372567fd97f1cef9b609d3dc0c81b0d3`, `qsa_gather_bench` SHA-256 `126ba2ed56f7623a0ee1eeb5f658d719e057ef3c3f54eb1d5d2746f8776523b8`, and `die.cubin` SHA-256 `0fa2a17fa41c07aac507f9bc218a777601719a8e2b6eb8b40ec93867d4768027`. These identities match the prior issue #16 `perl-matrix` provenance.

## Why these values do not finish issue #16

The current-head artifacts differ: `qsa_gather_bench` is `f982b117724816c7d910a8188d585e1ef5f51c75322e21f9175b0ef6fb2b6ecc`, and `die.cubin` is `aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`. The source changes since `7c6f34d` add the current position/capture cases and profile range, and the kernel/assembly sources changed. The archived #10 profiler results therefore remain historical evidence for the older binary/cubin; they are not relabeled as current-head counter results.

The issue #10 acceptance inventory still marks all eight criteria `PARTIAL` and `full_issue_acceptance=false`. Its criterion 7 specifically records no accepted checkpoint-derived selected-ID extraction lineage. The archived component matrix used generated changing IDs and cannot supply production model selections for issue #16.

## Offline preparation completed

[`capture-manifest.json`](capture-manifest.json) records the exact current-head `QSA_CAPTURE_IDS` fixture command, expected one-row format, matching position, direct host/VRAM `GATHER_IDS` commands, and the limitation that `idxfull` uses synthetic inputs. [`profile-manifest.json`](profile-manifest.json) records the metric names confirmed by the archived CUDA 11.4 M60 runs and the separate current-head profiling groups.

This closes the offline metric-discovery and command-preparation gaps. A current-head GPU capture, paired host/VRAM matrix, and current-head byte/transaction/stall values still require a new GPUQ run; no acceptance item requiring those hardware observations is marked complete here.
