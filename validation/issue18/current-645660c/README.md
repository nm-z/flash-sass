# Issue 18 acceptance: implementation 645660c

The acceptance suites completed successfully on archy at **2026-10-18 00:49:37 UTC**.
The tested C/SASS implementation is
[`645660cd27a0ab7c2f28faeb9bddfcf2f33886cd`](https://github.com/nm-z/flash-sass/commit/645660cd27a0ab7c2f28faeb9bddfcf2f33886cd).
The final Bash runner is from
[`cee182df737fd7501060288402f14202b9a186fe`](https://github.com/nm-z/flash-sass/commit/cee182df737fd7501060288402f14202b9a186fe).
That later commit changes orchestration and documentation; it does not change the
measured C/SASS sources or binaries.

## Evidence and provenance

- [results.json](results.json) contains the acceptance totals, independent poison arms, every timing pair, medians and ranges.
- [raw-evidence.tar.gz](raw-evidence.tar.gz) contains the commands, environments, stdout/stderr, exit records, idle checks, engine configurations, build logs, source/driver/cubin hashes, and dependency patch.
- [raw-evidence.sha256](raw-evidence.sha256) checks the archive. Its SHA-256 is `310f767b4a7a88b8ee697ea55fb08af37309c5af59495453e5315e3ab15fca92`.
- [report.jq](report.jq) regenerates the summary from those recorded inputs.

Both candidate and control checkouts were clean before and after the suites.
Both final builds exited 0. The control revision is
`f88dfc56c3045023b32ca7eec62b0dd7bfd3f9a2`: it uses the candidate source with only
`kern/attnpart.sass` and `kern/attnpartn.sass` replaced by their unmasked versions
from `aa0b898607f0c070bda100e10dfb6bb6391e55a4`. The complete control patch is in
`raw/build/final-control-kernels.patch`; it can be applied to the tested candidate
without needing the temporary control branch. Source, binary and cubin hashes
were captured again after all tests.

The component runs used Tesla M60 GPU
`GPU-431371bd-b482-96b2-4b70-04f83c6d075b` (physical GPU 6), driver 470.256.02,
CUDA 11.4, ptxas 11.4.152 and GCC 16.2.1 20260810. MaxAs was based on
`54eda7af086a46c9dae1688b691968235d560164`; its three local changes, patch and hashes
are recorded under `raw/dependencies/` and `raw/maxas-provenance.txt`.

Before every component GPU process, the runner recorded `nvidia-smi` GPU activity,
compute owners and `ps`, and waited for an idle host. It held the supplied GPU
coordination lock. Engine orchestration held both the component and engine locks
and checked idleness before each reload and fixture request. Resident weight
holders, the fence and an idle flashcore remained loaded. Other workers' processes
were not killed. Published process records retain the PID, command name and wait
channel (also parent PID for engine records); process arguments are omitted.
`raw/PUBLICATION.txt` documents this projection. Raw test outputs are unchanged.
The archive's 17,937 manifest entries were verified after transfer.

## Acceptance results

| Check | Measured result |
| --- | --- |
| Exposed C count contract | Three invalid counts reject before reading NULL inputs or changing output sentinels; one-key decode/partials and empty partitions pass |
| Checked scalar interfaces | Strict parsing, empty/overflow/capacity/budget rejection, unchanged outputs, one-key attention and selected gather pass |
| Invalid CLI inputs | 46/46 reject with exit 2 while CUDA entry-point traps remain uncalled |
| GPU smoke cases | 4/4 pass, including capped maximum visible count and host budget 4096 |
| Independent padding matrix | 1536/1536 pass numerical and allocation-guard checks |
| Complete direct QSA | 10/10 cases, 34/34 visible positions report EXACT with nonzero guarded allocations |
| Negative/observation controls | 26/26 produce their required outcomes |
| Paired component timings | 40/40 runs pass correctness and guard checks; five image pairs per component/placement |
| Actual model reuse/rollback | Four fixtures, 80 requests, 32 bitwise comparisons and 84 guard checks pass |
| Shared engine restoration | Original full configuration and original library hash match after restoration |

### Independent K and V padding

Each arm below contains 96 device-memory and 96 mapped-host-memory cases. Valid
data is generated before padding changes and uses the same seed across poison
arms. The bounded finite mode varies padding between +3.25 and -3.25 by row,
dimension and KV head, independently of valid data.

| Invalid K | Invalid V | Passing cases |
| --- | --- | ---: |
| NaN | Off | 192 |
| Positive infinity | Off | 192 |
| Negative infinity | Off | 192 |
| Bounded finite | Off | 192 |
| Off | NaN | 192 |
| Off | Positive infinity | 192 |
| Off | Negative infinity | 192 |
| Off | Bounded finite | 192 |

The matrix contains 912 dense cases, 288 sequential QSA cases and 336 compact
attention cases. The sequential cases compare query/append and gather/attention
at every position: 4,288 positions in total. The runner clears inherited QSA
settings, including the option that would limit attention checking to the final
position.

Dense coverage includes counts 1..33, quarter/tile/chunk boundaries, 2048..2055,
and 8191..8193. Compact coverage includes production budget 512, maximum supported
budget 4096 and 262,144 visible positions (zero-based position 262,143). The
complete QSA driver covers visible counts 3100..3103, 8000..8003, 65000..65003,
262000..262003 and 262144 in both placements. Exact commands and all counts are
in each suite's `plan.tsv` and case records.

The compact fixture isolates attention and merge. Complete direct QSA uses
synthetic history with original-input projections and deterministic synthetic
weights; it is not a maximum-context model run. The numerical and allocation
preconditions, including the instruction-derived read bounds for every supported
budget, are in [ATTENTION_PADDING.md](../../../ATTENTION_PADDING.md).

### Controls establish what the tests detect

The unmasked control passed all 12 K-only NaN/Inf cases. Its later score mask can
overwrite the effect of invalid K arithmetic, so these passes are recorded as
observations and are not called regression detections.

All 12 V-only NaN/Inf controls produced numerical mismatches while allocation
guards passed. Dense controls exited 3; compact controls exited 1. At dense
`n=1`, the NaN control left pm/pd exact but contaminated po and the merged output,
matching the issue's conditional counterexample.

Both intentional prefix-guard corruptions exited 3 and reported the guard
mismatch while pm, pd, po and output remained bit-exact. These corrupt the QO
allocation's device prefix in each cache-placement configuration; they do not
claim an injected host-memory guard fault.

### Actual engine allocation, reuse and rollback

The validation-only C library runs the production 48-layer engine with its actual
PLE table and core-owned weights. Each placement/image fixture starts with an
acknowledged fresh reload, then runs `reusecheck 96 64 16`. The configuration uses
`MAXT=256`, budget 512, two-position mode off and chunked prefill off. Candidate and
control use the same library, differing only in the loaded cubin. Device KV uses
control then candidate; host KV reverses that order.

For each snapshot residue, a 64-token reference is compared with the identical
request after a different 96-token request, then with a request resumed after a
96-token divergent suffix. Ordinary reuse must report resume 0; rollback must
report the actual snapshot 16, 17, 18 or 19. Each comparison checks all three
emitted token IDs and all 248,320 final logits for bitwise equality and
finiteness. Every fixture reports 20 completed requests, eight comparisons and
21 nonzero allocation-guard checks, including the fresh allocation check.

These guards cover engine device allocations and the outer mapped host KV arena.
They do not separately enclose every logical subarray within that arena,
core-owned weights/PLE, or polling arrays allocated through `cuMemAllocHost`.
Canaries detect boundary writes; the documented address derivation establishes
read extents. Intermediate generation steps' complete logit buffers are not
compared. The engine fixtures establish these reuse transitions at their stated
prompt sizes and are not a long-context throughput or model-quality benchmark.

`raw/engine/` preserves all four reload/configuration/fixture transcripts, input
hashes and restoration records. The final status is `workload_exit=0` and
`restore_verified=true`; the original and restored full configuration responses
are identical, including the original cubin fingerprint.

## Measured masking cost

### Attention components

Each sample uses 200 iterations, visible count 2051 and poison off. Compact
attention uses budget 512. Candidate/control order alternates across five pairs.
Times are CUDA-event microseconds per iteration for attention plus merge.
A paired change is `100 * (candidate/control - 1)`; its median need not equal the
ratio of the two independently computed timing medians.

| Component | KV placement | Control median, us | Candidate median, us | Median paired change | Paired change range |
| --- | --- | ---: | ---: | ---: | ---: |
| Dense | Device | 153.000 | 155.700 | +1.76% | +1.24% to +2.62% |
| Compact | Device | 153.577 | 155.985 | +1.54% | +1.40% to +1.81% |
| Dense | Mapped host | 30201.100 | 30209.600 | +0.02% | -0.05% to +25.66% |
| Compact | Mapped host | 30192.842 | 30189.622 | -0.01% | -0.36% to +0.01% |

The third dense host pair measured 37,950.8 us for the candidate and 30,201.1 us
for the control, a +25.66% change. It is retained in the summary and raw evidence;
its cause was not established. These samples do not establish a fixed upper
bound on host-memory overhead or a systematic host-memory speedup.

### Actual model requests

The production engine reports `run_ms` before allocation-guard scans and final
logit readback. The following medians use four matched requests per stage and
placement. `build_ms` and every individual pair remain in `results.json`.

| KV placement | Stage | Control median, ms | Candidate median, ms | Median paired change |
| --- | --- | ---: | ---: | ---: |
| Device | Short reference | 1837.844 | 1847.839 | +0.54% |
| Device | Longer request | 2731.649 | 2748.309 | +0.59% |
| Device | Short after longer | 1837.646 | 1848.821 | +0.61% |
| Device | Longer with snapshot | 2765.024 | 2768.455 | +0.14% |
| Device | Short after rollback | 1361.536 | 1368.301 | +0.60% |
| Mapped host | Short reference | 1921.587 | 1882.056 | -2.06% |
| Mapped host | Longer request | 2860.107 | 2800.057 | -2.10% |
| Mapped host | Short after longer | 1925.248 | 1890.623 | -2.21% |
| Mapped host | Longer with snapshot | 2894.941 | 2841.726 | -2.14% |
| Mapped host | Short after rollback | 1452.478 | 1411.382 | -1.98% |

These short-fixture observations include load order, page-cache and system
variation. Negative changes are observations, not evidence that predication
speeds up the model. The component measurements isolate the added attention
work more directly. No full-cache clearing was added.

## Reproduction and retained diagnostics

Follow [the validation guide](../README.md) to build and run the parameterized
Bash component and engine runners. Build with the full `FLASH_REV` shown there so
the engine configuration can be checked against the source HEAD. To reconstruct
the control, apply the archived control patch to the tested candidate and rebuild
with the recorded MaxAs patch/toolchain.

To verify the archive and regenerate this recorded summary from this directory:

```bash
sha256sum -c raw-evidence.sha256
tar -xzf raw-evidence.tar.gz
(cd raw && sha256sum --quiet -c MANIFEST.sha256)
jq -n --slurpfile padding raw/padding/results.json \
    --slurpfile direct raw/direct/results.json \
    --slurpfile controls raw/controls/results.json \
    --slurpfile timing raw/timing/results.json \
    --slurpfile engine raw/engine/timings.jsonl \
    --slurpfile engine_status raw/engine/status.json \
    --slurpfile cli raw/cli/cli-report.json \
    --rawfile completed raw/timing/finished.txt \
    -f report.jq > results.rebuilt.json
cmp results.json results.rebuilt.json
```

Two initial direct-suite attempts each ran one numerically exact case but were
rejected by the runner: first it counted a metadata header as a result row, then
it matched the phrase "0 mismatching steps" as a failure. Both rejected attempts
are preserved under `raw/direct-parser-*`, excluded from final acceptance, and
followed by a complete ten-case rerun. The parser checks also reject missing
positions, a missing terminal summary, the wrong placement and actual failure
markers.

The padding suite used the committed 645660c runner. The final direct, controls
and timing suites record their corrected runner source and hash. The measured
engine script and committed `engine.sh` differ only in whether process arguments
are recorded. Historical evidence under `../archy-final` retains its original
revision and scope; it is not relabeled as current-head evidence.
