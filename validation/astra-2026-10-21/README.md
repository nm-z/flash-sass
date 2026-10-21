# Astra performance investigation: results and experimental source

Published at Nate's request after roughly 13 hours of work on October 20–21,
2026. Related performance work: [issue #82](https://github.com/nm-z/flash-sass/issues/82).

**The requested outcome was not delivered.** Full-model decode remained around
40 tokens/s. Neither the first milestone of 60 tokens/s nor the final targets of
140 decode / 1,200 prompt-processing tokens/s were reached. No substantial
correctness-qualified throughput improvement was established. The source in
this branch is an experimental research snapshot, not a production release.

The largest recorded decode result with eight matching output streams was
40.3922 tokens/s against 40.1206 tokens/s, approximately **0.68%**. Even that
job's overall status is **failed**, because restoring the original engine
failed after measurement; a separate recovery job restored it. Reporting its
matching token streams as a fully successful GPUQ job would be incorrect.

The useful outputs are reproduced bugs, rejected hypotheses, an executable MTP
shadow path, partial rollback validation, source, and recorded evidence. They
do not substitute for the missing throughput result.

## Evidence and measurement scope

- [jobs.json](jobs.json) contains all **72 recorded jobs** from this investigation's
  three former side databases: 27 earlier jobs, 44 later jobs, and one recovery.
  It includes UUIDs, exact commands, terminal status, exit code, and collector
  exclusions. Rejected submissions and cancelled attempts remain included.
- [stdout-stderr.jsonl](stdout-stderr.jsonl) preserves **20,356 captured stdout/stderr
  documents**, including split output chunks, generated IDs, timings, failures,
  and restoration reports. It contains no identity/environment collector dump.
  Join chunks for the selected job and stream before splitting into lines.
- [RUNS.md](RUNS.md) provides a readable inventory of those 72 jobs.
- The canonical GPUQ database additionally retains documents, collector data,
  metrics, source identities, and unrelated historical runs. Its integer document
  IDs can change during consolidation; original-to-canonical mappings are stored
  in each imported job's `gpuq-canonical-consolidation` provenance document.
  Job UUIDs and original document contents do not change.

The full-model workload was Qwen3.8-Flash-Next Q2, all **48 layers**, six M60
compute dies, QSA **B512**, greedy generation, ordinarily **64 prompt tokens and
128 generated tokens**. There are 127 post-first-token decode intervals. The
benchmark uses one warmup plus three measured repetitions in each arm. Rates
below are medians of the three warmed repetitions, not maxima. A one-token
diagnostic has no measured decode interval and is not a decode-throughput test.

The resident expert split is `112,97,81,116,97,97`; attention owners are
`1,1,1,1,4,4,4,4,5,5,5,5`; GPU visibility includes an anchor at 6, with model
compute on 0–5. Component tests sometimes use GPU 7. Holders and persistent
weights are reused. Most experimental modes use capacity 16,384 and one image
row; the original service uses capacity 230,000 and 4,096 image rows. Exact
per-arm configurations are in the logs; do not compare incompatible arms or
turn a component speedup into a full-model claim.

The [frozen 128-token output](../native_chunk/fixture64-greedy128.ids) comes from
the scalar arm of `52e7b787-5332-474f-bf25-7aa1003d2d2b`, whose four scalar
streams agreed. Its SHA256 is
`54cdb3b961cc5df485f65db476373ce8886db3deb518fbd49b3a64c602b23f64`.
It is an output oracle only. It was not fed to generation or used to stage
future PLE rows. This single workload is not an independent model-quality
oracle or a validation of all context lengths, sampling, images, or resumes.

## Full-model results

| Experiment / job UUID | Baseline decode | Candidate decode | Prompt processing | Outcome |
| --- | ---: | ---: | --- | --- |
| Fused routed gate/up/SiLU/quantization, `335ca2a7-3082-4767-8880-2dc719608f2b` | 40.1206 | 40.3922 | Approximately 40.31 → 40.58 | All eight output streams match; +0.68%; **job failed restoration**, later recovered. Not promoted. |
| Native F32 prefix with matched production scalar decode, `dcb94c6a-4c19-43dd-b33f-d1d684f0186b` | 40.03986 | 39.75300 | 40.22627 → 93.96586 | All four candidate outputs differ from scalar starting at output 33. **Failed**, rates diagnostic. |
| Exact B8 prefix, `6706ab8c-899d-46ac-9bef-ded6f7f0988c` | 40.03976 | 40.02023 | 53.86381 → 67.38484 | All four B2 baselines match; three B8 candidates match, one diverges at output 33 with 93 differing suffix tokens. **Failed**. |
| Coalesced peer publication, `bb1ac914-e2a4-487d-990a-9adc45ae7558` | 39.88746 | 37.96236 | Not a demonstrated PP improvement | Candidate is about 4.8% slower. Four candidate streams match, but baseline run 3 differs at outputs 126 and 127. **Overall failed**; no accepted A/B gain. |
| MTP shadow v8, `9d19f0e1-e980-416d-8923-7aeabbd8d6a0` | 39.8079 | 33.9160 | About 41.22 → 36.03 | **Passed** all eight target-output streams and restored identically. Additional draft work makes this slower; no speculative tokens are accepted. |

Earlier native run `52e7b787-5332-474f-bf25-7aa1003d2d2b` measured diagnostic
93.58110 PP / 39.03465 decode against 40.23494 PP / 40.04274 decode, with
candidate mismatch counts 1, 93, 93, 0. These are not validated faster PP results.
Global removal of CI load modifiers also failed, at approximately 36.18 decode
and 94.33 PP in the candidate. No result establishes a correct 94-PP engine.

The paired full-head cost probe `7dc1f170-4b3a-4553-956c-dea933f4166d` passed
its fixture check at **51.8581 known positions/s**, against 40.2955 scalar. It
processed a 64-token known prompt and generated only one token: **zero decode
intervals**. It was a verification-cost probe, not a 51.86-TPS decode result.
The earlier 256-position probe failed. This limited paired rate motivated the
native batch work rather than supporting a claim that MTP alone could reach 60.

## Fused MoE: real component gains, negligible model gain

`kern/decode_fuse.c` emits a SASS fusion of routed Q2 gate/up, SiLU, and activation
quantization with 32-row CTA tiles. It preserves the existing dot-product
arithmetic. Shared-expert work remains separate. Opt-ins are `FLASH_FUSED_MOE`
and `FLASH_FAST_EXCHANGE`; the latter elides a send grid wait only when the
following consumer waits on all sending CTAs.

Component job `97e1d335-6c4c-44b8-b52e-93c62a7066b4` passed exact old/new GPU
comparison, an independent C activation/quantization check, and guards for
selected-expert counts 0, 1, 2, 3, 6, 10. Nonzero-count speedups were 1.3084,
1.1863, 1.1262, 1.0481, and 1.0371. The full model gained only 0.68% in the
measurement above. This is direct evidence that those component gains did not
deliver the required structural improvement.

Early attempts failed module loading with a global-variable ABI marker, then
failed on the CUDA-13 `cuEventElapsedTime_v2` symbol against archy's older
driver. An unused entry marker and explicit older event ABI resolved those
specific component-test failures. A private full-model core failed allocation
alongside the serving core, so the completed A/B used resident reloads. The
successful recovery UUID was `98abaeca-5873-4e33-81a9-1d72b991bcbf`.

## Native prefill: implementation, reproduced bugs, rejected explanations

The `kern/chunk/` experiment reuses the resident weights through virtual expert
maps, adds batch GEMM/attention/DeltaNet/slab operators, and stages known PLE rows
through existing readers. It does not load a second target model. The batch
kernel has 249 registers. Scalar and batch modules were eventually separated
so the candidate's new dispatcher was not also used for generated scalar work.

Concrete defects found and corrected in component tests:

1. The initial new dispatch table placed its first appended operation after the
   scalar fallback branch. BF16 outputs remained untouched NaNs. Correcting
   dispatch and checking three-instruction entry stride made component v3 pass
   (`04d5a179-8b09-452d-b6e3-a619b1fab68e`). A test-owned dynamic counter also
   needed reset between independent down-projection launches.
2. The activation quantizer repeatedly mutated its address registers. Immutable
   bases and immediate offsets fixed it. `064bcbb4-e30f-4e6b-b7bb-abc5f35fad1a`
   passed BF16, both Q2 GEMMs, routed down, and slab/packed, in-place/out-of-place
   quantization checks with guards. This proves those tested shapes, not all
   full-model equivalence or larger slab strides.
3. Routed down-projection used different FMA trees for a 4-column tail and
   8/16-column groups. Atomic assignment of positions to slots could select a
   different tree across runs. Original-kernel permutation test
   `fa0787c1-1bea-42de-8adc-9efdc3f80647` reproduced **486 differences** at 17
   positions. Uniform even/odd trees passed with **zero differences** at 13,
   17, and 21 positions in `47937aa8-550c-4baa-a04c-ad0da8ee0568`, at the same
   CPU error tolerance. This fixes that order dependence; it did not fix
   full-model scalar equivalence.

Prefix hashing in `ccbfa7ed-548c-4191-9ea0-a36fa9b85c40` localized the original
variation to layer 2, die 0, before the first attention block: 222 changed buffers
after stable layer-0/layer-1 hashes. The ledger marks this run timed out during
a host I/O stall; its completed hash records are diagnostic evidence, not an
accepted timing run. After the FMA-tree correction, one-token repeatability
test `e322e12d-391d-4a18-94dc-e341fcebcdee` passed, but full output tests still
failed. `f1f6929b-e40c-46d3-ac37-2757b975f060` produced four identical candidate
streams, each differing from the frozen scalar output at index 33.

Explanations tested and rejected:

- **Host stream readback alone:** `e7e618b0-b3d4-4069-a0cc-44640c15308a` still
  failed output repeatability, while all eight final device/host stream checks
  reported zero differences. This rules out that explanation for this run.
- **A missing all-thread phase fence as the complete fix:**
  `05c5ea52-7685-43f6-83bd-6a6f6e47a5dc` still failed with 850 mismatches.
- **Global CI load removal as the fix:** a cache microtest passed on the
  original CI module (`6cd83ffe-8a71-4fef-8105-40350f703dd0`); coherent-load
  components passed (`6be1fc97-0cfc-4136-b2a9-d87242a2b8e0`), but the full A/B
  `92375122-5304-4ead-a594-56a64fbf6532` still failed with 533 mismatches and
  slower decode. Do not enable this globally on the strength of cache theory.
- **Using only the original scalar dispatcher for decode:** matched original
  scalar-module tests still produced the deterministic output-33 divergence.

Native FP32 batch arithmetic is not automatically identical to scalar
activation-quantized low-bit GEMV. Component tolerances and repeatable candidate
tokens cannot establish the required scalar equivalence.

## Exact B8: tested operators and remaining full-model failure

`kern/exactbatch/` preserves scalar arithmetic for selected BF16, Q2, DNSTEP8,
and ACTQ2568 work. Disjoint activation/inbox storage is allocated per position;
attention scratch aliases are explicitly serialized. Host grouping preserves
position-major PLE, indexer, and compact-attention ordering.

- BF16 parity: `3acbbd6f-5a97-4566-bb3b-2d19c079a7fd`, passed.
- Production-input Q2 parity: `c7e8fd87-1474-4083-a5fa-084416860fe1`, passed.
  The earlier submission failed recorder admission, not kernel parity.
- Recurrent state and ACTQ comparison:
  `fb7686bb-f84f-4a82-a075-b27ff3aedce6`, passed all 12 start positions,
  including 2047/2048 boundaries.

Production-input timing showed routed B8 Q2 scanning often only 0.1–0.5 times
scalar speed, and large K2560 row cases around 0.63 times scalar. Routed Q2
fusion was disabled and row counts over 1,024 fell back to grouped scalar work.
Those choices did not make the full B8 engine correct: the 67.38-PP full-model
run above still failed one repetition. This verifier is not yet fast enough to
justify a claim that speculation will reach 60, much less 140 tokens/s.

## MTP: a working shadow path, not speculative decoding

The draft checkpoint is `mtp-Qwen3.8-Flash-Next-Q8_0.gguf`, 4,137,429,120 bytes,
34 tensors, SHA256
`58a47d6a216307e9b97172d5bfdc3436a8994b632f5e436b9bb6edf8951da8d1`.
The loader validates shapes, streams Q8 repacking, converts the F32 router to
BF16, handles split/merged HC rows, and reuses the resident target output head.
It gathers embedding rows on the CPU and uses independent mutable draft state.
Reference alignment is input token at position p with target hidden state p−1,
and zeros at p0.

A real packing bug existed in `ref/quant.c`: all block-32 formats reserved 48
scale bytes, but wider Q8 rows need `round_up(2*(K/32),16)`. Scales overwrote Q8
codes for wider matrices. The fix preserves the K640 48-byte ABI and changes
only Q8 prefix sizing. Loader CPU checks cover raw scale/code bytes, selected
packed rows, split boundaries, maps, guards, and allocation accounting on six
dies. Standalone GPU job `739c1ceb-378c-4deb-920e-43d822fd225f` passed 24 Q8 cases.

Persistent MTP components initially faulted or returned incorrect SiLU outputs.
Adding spacing between count predicates and predicated loads, and between
dependent SiLU predicates, produced passing uninstrumented job
`df22cfbe-adcd-4cdd-833e-d5f7826ea5e4`: 48 Q8 cases and eight helper combinations,
exact checks and clean guards. Zero memory errors under memcheck in an earlier
run did not establish numerical correctness.

The target expert split left insufficient memory for a second Q8 draft using
the same split. Capacity planning against actual free VRAM produced independent
draft ranges **29,150,25,20,149,139**, retaining the target split. Compact
activation arenas later reduced allocation overhead and allowed checkpoint
storage to fit. No target holder needed reloading for this plan.

The final shadow job `9d19f0e1-e980-416d-8923-7aeabbd8d6a0` completed all four
baseline and four candidate requests with matching 128-token output. Each
candidate advanced 191 draft positions. One-step matches were **108,108,107,108
out of 128**: 431/512, or approximately **84.18%**. Warm median head-draft time was
478.043576 ms for 128 predictions, approximately **3.735 ms per prediction**.
Headless draft steps were approximately 2.89 ms. These are one-step agreement
and cost measurements; they do not measure chained-draft acceptance.

**No target batch acceptance, rejection controller, accepted-token publication,
or speculative decode speedup was implemented.** Shadow mode computes both
draft and target and emits only target-generated output, explaining its roughly
33.92 TPS versus 39.81 baseline. Frozen expected IDs are only compared after
generation. MTP shadow and the rollback audit are separate library modes.

## Persistent-kernel stalls and scheduling experiments

Successive full-model MTP attempts failed despite several plausible corrections:
known PLE rows staged before each target position, all host uploads completed
before launching peers, and reading target output only after all peer launches
completed. These improved observability but did not establish a root-cause fix.

Uninstrumented completion checks found GPU 0 exiting with fewer exchange
publications than its program requires, while other dies waited. Representative
observations from service diagnostics:

| Job | Completed position | GPU 0 XGEN | Expected XGEN |
| --- | ---: | ---: | ---: |
| `90afe422-647f-45b2-83dd-f4ab89d3eda8` | 50 | 17598 | 17600 |
| `84c2d15d-924a-4257-8923-1fcb7bed3e55` | 22 | 7741 | 7744 |
| `68037e60-3e43-455d-a7a1-828410d4829f` | 10 | 3518 | 3520 |
| `2e850c9e-2acb-44f9-8e5d-79513ff79dbd` | 67 | 23589 | 23592 |

Its 16 own-publication flags agreed with the lower XGEN; local CTA/grid
generations agreed and timeout flag was zero. Previous completed launches had
passed these checks. This demonstrates an incomplete-publication symptom, not
the instruction responsible.

A CUDA-gdb diagnostic (`336f8bd3-9284-43b5-a1e4-ceb66824fd32`) saw peers waiting
for sender 0 one generation behind, with GPU 0 absent from active kernels.
That attach had toolchain/Python/glibc compatibility problems and the service
subsequently crashed. The debugger perturbed execution and is not definitive
root-cause evidence or a timing result. The service restarted and restored.

`phase_schedule_patch.c` is bound to the original production ELF fingerprint.
Experiments added mirror-predicate spacing, a timestamp-store read barrier,
CG controller loads, and separate dependency barriers for descriptor payload
and token-mask prefetch. The first long-stall patch mistakenly retained the
no-yield bit; later versions corrected it. Counter regressions and MTP component
tests passed, but full MTP runs still stalled. The same final schedule with
compact activation allocation passed the v8 shadow run. **This correlation does
not prove allocation layout fixed the underlying bug.**

Relevant scheduling documentation:
[MaxAs control codes](https://github.com/NervanaSystems/maxas/wiki/Control-Codes),
[NVIDIA PTX memory/cache semantics](https://docs.nvidia.com/cuda/archive/12.1.1/parallel-thread-execution/index.html).
The experiments, not documentation alone, determine whether a proposed fix works.

## Coalesced communication publication

`kern/comm/` makes producer threads issue `MEMBAR.SYS` before local grid arrival,
then lets a half-warp in CTA 0 coalesce publication of the 16 CTA-generation
words per destination. It preserves the existing six-by-16 flag layout and
descriptor ABI. Both counter-regression versions passed; the full model was
slower at 37.96236 TPS. It should not be promoted as a performance optimization.
Its four matching output streams do not cure the baseline failures in that A/B.

## Rollback verification: useful partial evidence, one failed matrix

`die_verify` adds intermediate DNSTEP8 checkpoints plus a bounded vector-copy
operation. Programs capture recurrent, convolution, PLE and attention-indexer
ring state for the first seven positions; position eight uses final live state.
Position zero reuses existing snapshot storage and six further snapshots share
one arena per die. Rollback rewinds logical state and position while keeping
peer/grid generations monotonic. Future KV rows are ignored until overwritten.

Component job `8dc627e7-6687-43ea-aa04-4da68da7e9cb` passed all 12 starting
positions, every intermediate scalar/grouped/fused state comparison, guards,
and bounded-copy sizes 16, 2032, 2048, 4096, 4112, 65552 bytes. The initial
full-model attempt failed allocation. Compact arenas made the keep-one test fit.

`c384337c-9280-4b53-acdd-6bda795d472c` **passed** four baseline and four candidate
128-token streams. Each candidate verified prompt positions 16–23, retained
one position, poisoned seven future embedding IDs, restored state, replayed the
discarded suffix, and reported zero changed valid-prefix buffers. PLE rows were
kept original in this diagnostic: it was a future-embedding causality check,
not complete speculative PLE-cache handling.

The expanded all-eight-boundary matrix
`d8037316-de4e-411d-ae0b-0b6b5c9f488e` **failed overall**. All 32 candidate
state-fingerprint comparisons matched, but candidate run 8, retaining three
positions, generated token 303 instead of 1083 at output 102. The remaining 31
candidate streams and four baselines matched. Matching state hashes did not
prove the full decoding path correct. This matrix must not be described as
passed or as completed speculative decoding.

## Source map and build/reproduction

| Source | Purpose / status |
| --- | --- |
| `kern/decode_fuse.c`, `kern/moe_fused_test.c` | SASS fusion and component oracle; negligible full-model gain. |
| `kern/chunk/`, `kern/chunk_ops_test.c` | Native batch experiment; full-model equivalence fails. |
| `kern/exactbatch/`, `kern/pair_state_test.c` | B8 arithmetic, grouping, checkpoint capture and rollback diagnostics; full-model gates incomplete. |
| `kern/mtp/`, `kern/mtp_q8_gen.c`, `kern/mtp_q8_test.c` | Draft loader, Q8 SASS, full draft graph and shadow runtime; no acceptance controller. |
| `kern/comm/` | Coalesced publication experiment; regression in full-model throughput. |
| `kern/phase_schedule_patch.c` | Identity-checked production scheduling/cache-policy experiments; not a proven stall cure. |
| `kern/resident_decode_bench.c` | Existing-fixture A/B runner, mismatch collection, shadow/rollback diagnostics, restoration. |
| `kern/core_recover.c` | Queue-owned recovery used after a blocked restore. |
| `ref/quant.c`, `ref/quant.h`, `kern/hold.c` | Q8 scale-prefix sizing and draft tensor packing support. |

The branch starts from `98c176a80e82cd31736ed3165a134cfbbae45338`. It contains
the final experimental source, including rejected paths and diagnostics. It is
not a claim that rebuilding the final tree reconstructs every earlier version;
job identities, archived artifacts, and captured sources distinguish those.
The old `NativeBatchRunfile` / `CoherentABRunfile` retain historical artifact
paths and should not be mistaken for a current approved run recipe.

Computation is authored in SASS with C/Lua generation and host control. Tiny PTX
entries provide an ELF cubin shell and marker entry; MaxAs replaces the compute
body with SASS. A cubin is the CUDA driver's binary container, not a switch to
CUDA C++ or PTX math. See [NVIDIA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#what-is-a-cuda-binary).

Build examples for the environment used:

```sh
env LIBRARY_PATH=/opt/cuda/lib64/stubs make -C kern --output-sync=target -j2 \
  PTXAS11=/home/nate/.cache/flash-sass-tools/ptxas-11.4 \
  'MAXAS=perl -I/home/nate/.cache/flash-sass-tools/maxas/lib /home/nate/.cache/flash-sass-tools/maxas/bin/maxas.pl' \
  libengine_mtp.so die_mtp.cubin libengine_verify.so die_verify.cubin \
  resident_decode_bench
```

The existing assembler is Perl; no new Perl or CUDA C++ computation was authored.
CUDA 13 headers/stub libraries were used locally, PTXAS 11.4 and patched MaxAs
for SM52 assembly. Existing quant.c maybe-uninitialized warnings remain.
Host pair-program, prefill-planner and trace/session checks passed during the
implementation. Those checks do not replace the GPU evidence above. No new GPU
performance run was performed simply to publish this report.

All archy GPU work must use the canonical queue:

```sh
ssh archy gpuq run -- COMMAND ARGS...
```

Use each entry's exact `argv` in [jobs.json](jobs.json), with its original
artifact identities, to understand or reproduce a historical test. Explicitly
preserve the current serving configuration before a resident reload. Do not use
old side-database overrides. A new run is required to validate any new build.

Selected final artifact SHA256 values, originally staged under
`/dev/shm/flash-sass-kernel-artifacts/` on archy:

| Artifact | SHA256 |
| --- | --- |
| `libengine_mtp-v8.so` | `84e98d93e5c734486bb1f96687334de8c81dd1ce1dcad7ede1a0e238d29ab61c` |
| `die_mtp-v5.cubin` | `673896c5885dd102d55f4fa972d85fa302041b8c0cd3b3ff11ff37b2b708d086` |
| `die-production-schedule-v3.cubin` | `dc9cf05d433fe13cca74a90ea02dd6c4ae97fffbb5437215851b081de77b8aa6` |
| `resident_decode_bench-v19` | `10c35b89c602af5aff51e5573d8fa0fe93a2df7e510fc6d23b5a05334bc09756` |
| `die_collective-v2.cubin` | `60abb7101338e05cafa38e45e171e3dc1f07a4a749f22cd9e6bb850a12b7b9fb` |

## Operational failures and mistakes

Database locks, recorder capture failures, a host-wide storage stall, and tool
coordination failures occurred. They affected individual attempts; they are
not an explanation that turns the failed performance work into success. GPUQ
feedback is recorded in [#59](https://github.com/nm-z/gpuq/issues/59),
[#60](https://github.com/nm-z/gpuq/issues/60), and
[#29](https://github.com/nm-z/gpuq/issues/29). Jev discovery was unavailable
because its host did not resolve, reported in
[stop-hook #1116](https://github.com/nm-z/stop-hook/issues/1116). DSH delivery
failures are recorded in
[nates-dsh #3096](https://github.com/nm-z/nates-dsh/issues/3096).

**Astra incorrectly worked around recorder failures by selecting separate
GPUQ databases.** Although they shared the GPU admission queue, this split
measurement history and violated the intended single-database architecture.
The publication/export work reconciles these records into
`/home/nate/.local/state/gpuq/history.sqlite3`, preserving provenance, and gives
Nate one full canonical database backup. A namespaced archive first produced
for export was not a native GPUQ database and is not the delivered replacement.

Consolidation completed with **1,057 jobs: the existing 985 plus all 72 owned
jobs**. Comparisons found zero missing or changed jobs, collectors, original
document blobs, metric samples, or parsed records, and zero foreign-key errors.
The original 985 job rows remained unchanged. The canonical database passed
SQLite `quick_check`; imported documents were decompressed and SHA256-checked,
then indexed for normal GPUQ search. Old override paths now resolve to the
canonical database; retired originals and a pre-migration canonical snapshot
are backups only. The single Desktop export is named
`flash-sass-gpuq-history-2026-10-21.sqlite3`, SHA256
`d8efaebf29099918d49fa6a68bd4d4687da7b0bc12d6e0fe83b98dd9cb82b053`.

Other operational qualifications: the first fused A/B restore stalled on NFS;
a later recovery used local PLE. The verifier initially ran out of memory. A
debugger attach caused a service failure. A known blocking-launch sanitizer
experiment was incompatible with sequential launch of peer-dependent kernels.
These are retained as failures, not omitted from the run inventory. Other
workers' observations in [#109](https://github.com/nm-z/flash-sass/issues/109) and
[#110](https://github.com/nm-z/flash-sass/issues/110) are relevant context, but
do not prove the cause of this experiment's short-request output differences.

At **12:29 PM PDT on October 21**, a fresh control read confirmed the original
`attention-q-release` service configuration: 48/48 layers, six dies, B512,
230,000 capacity, 4,096 image rows, local PLE, rope owner 5, paired mode 1,
prefill chunk 64, Q-split mask 3f, original cubin FNV `9abf8aafb9134d59`.
Original library SHA256 is
`954b27bf92fc968004aafa69a6d5896153b6fa03340f768b63483ffc0137fe3c`;
original cubin SHA256 is
`aecc7ae61d5fea6869aeb57c68965d4cce741516e8c4f909e1c8ed8956ee1423`.
This is a time-specific observation, not ownership of the shared service.

## Unfinished work and limits of the findings

No accepted candidate delivers 60 TPS. No 140-decode/1,200-PP result exists.
The single-request scalar publication fault is not root-caused. Exact B8 and
the all-boundary rollback matrix still fail full-output validation. MTP has
only one-step shadow agreement; chained proposals and a real verifier/acceptance
controller are absent. Longer contexts, broad prompt coverage, images,
snapshot/resume and non-greedy sampling are not validated by this fixture.

An integer-exact batched Q2 GEMM using FP32 for bounded integer dot products,
with scalar scale/FMA/reduction order preserved, was considered but **not
implemented or measured**. HCR/HCUP decomposition and older external prototypes
were inspected but did not establish the requested performance. No projection
from those ideas is reported as achieved TPS. The goal was cleared by Nate;
this publication records the work as it stands and does not claim completion.
