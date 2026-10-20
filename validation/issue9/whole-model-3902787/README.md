# Full-model validation inputs and run record

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9) and
[PR #105](https://github.com/nm-z/flash-sass/pull/105).
Complete issue acceptance is pending. These files preserve the input export,
completed direct-component checks, a one-token actual-model result, and the
failed multi-position oracle preflight. Each result has a limited scope below.

## Input export

DSH `build-2756458-1792494763943062965` completed on Archy with exit 0 in
2,829 seconds, including compilation, export and hashes. The export has 1,079
tensor files for all 48 blocks plus embedding and output head. Blocks 0–35 and
the embedding/head are stored on the task's HDD cache; blocks 36–47 are stored
in the task's root-filesystem cache and linked into the same tensor directory.

- `manifest.new.gz` is the complete CPU catalog, compressed for retention. Its
  `.new` suffix is the existing
  `hold_cache` output convention. Its header explicitly distinguishes it from
  a CUDA IPC manifest. Arena offsets restart in the second export; CPU tensors
  are addressed by independent file names.
- `tensors.sha256.gz` records every packed tensor file.
- `checkpoints.sha256` records both complete GGUF source files. The primary
  shard SHA-256 is `69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720`;
  the PLE shard SHA-256 is `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`.

The earlier partial Sentry exports are not used as completed inputs. The second
Sentry export was cancelled after slow progress under observed shared memory
and I/O pressure. Its replacement is the successful Archy export above.

## Implementation and builds

The completed direct/coupled runs and private engine use revision
`39027871256e578fa8339681a661f59de53233c4`. The CPU record serializer changed in
`36a8246b98df4beb1046f467cb66f9d6225d4d6f` after its reader rejected the old
ring byte count. New oracle runs require a fresh source seal/reference.
Documentation/Make changes may advance the PR head independently; each seal
also verifies numerical source contents, including the generated opcode header.

| Artifact | SHA-256 | Routed build |
| --- | --- | --- |
| GPU `model_test` | `da1ef219748ec35d2354d6ea2eaa8840f1ded98397fb39fd3efcdd0b10745c10` | `build-3086677-1792496730776973030` |
| Portable CPU producer | `d67500c26a40e58fc63ff7fec6c2ccfa56e3b190a5ed18250baa55ad48f9a6ec` | Same build; identical to the tested earlier producer |
| Native CPU producer | `8601e20ee61f9a6191a8519989159f2123a9c25dac52d69d7abe144298f7a0c9` | `build-3160844-1792497271142504447` |
| Private engine library | `882195eb79be5ca52f731a949bfbef749e23f23679c2e97909a1c59ba5b829a1` | `build-3099676-1792496828342326933`; full fixed `FLASH_REV` |
| Private `flashcore` | `0257e1fbf4a6505bc54ed230b93abe22d5f21ed66daa22345a85b73f6a673d06` | Same engine build |
| Request checker | `594c55c9716a52462ef0f908b6ff769f89ce0fc9083f0a31c34dce7e23104554` | Same engine build; config fixtures separately passed |

Native producer flags add `-march=native` to C11, `-O2`, and
`-ffp-contract=off`; the build host is Archy's E5-2620 v3. This architecture
variant's 12 API/CLI checks passed. Old-format actual-weight record/checkpoint
equality was measured as described below; fresh corrected-format equality is
still required before the deep reference.

## Completed GPU checks

The issue9 jobs use `/home/nate/.local/state/gpuq/issue9-history.sqlite3` on
Archy and share the global FIFO/reservation with the default database. The
compressed `.logs.json.gz` files are exports of durable database documents,
not output redirected during execution. Derived `.stdout` files are provided
for reading the two positive runs.

| GPUQ job | Result and scope | Retained export |
| --- | --- | --- |
| `ef56e0c2-ad05-488d-9cdc-e3992ba72f9e` | Exit 0, 14 direct cases / 56 positions, both KV placements, budget 512, starts 1/2048/2052/3100/8000/65000/262000 plus three tails. Synthetic history, deterministic FP32 inputs and BF16 projection fixtures; no checkpoint weight provenance. | `direct-ef56.logs.json.gz`, `direct-ef56.stdout` |
| `8f441d80-bace-4a50-a156-cdf6e40e30ba` | Exit 0, full 48 blocks + embedding/PLE/head, all six production dies, 1202 phases, one teacher-forced ID 10, prediction 18, host KV / budget 1. Exact compared stages, states and logits. No exclusion at one token. | `model-t1-8f44.logs.json.gz`, `model-t1-8f44.stdout` |
| `c8289d09-b214-4acf-bf09-7ead03cc9add` | Expected exit 3, selected-ID perturbation, VRAM KV | `negative-kv0-selection.logs.json.gz` |
| `23f81eb9-78f7-4fd1-a5b5-5bbf93a31dc0` | Expected exit 3, compact-gather perturbation, VRAM KV | `negative-kv0-gather.logs.json.gz` |
| `56a37b5f-83c0-4425-a489-4269ea60d6e1` | Expected exit 3, attention-output perturbation, VRAM KV | `negative-kv0-output.logs.json.gz` |
| `c3d745e8-c5b8-4da7-88e0-67ca15b85c5a` | Expected exit 3, selected-ID perturbation, host KV | `negative-kv1-selection.logs.json.gz` |
| `bc5e6bf2-3e8e-4dfb-b6eb-74922b95102f` | Expected exit 3, compact-gather perturbation, host KV | `negative-kv1-gather.logs.json.gz` |
| `4a01410d-4fb2-4e7d-9539-4fd19145cb8d` | Expected exit 3, attention-output perturbation, host KV | `negative-kv1-output.logs.json.gz` |
| `16737acb-6900-46fa-8021-e1fa9e6424d9` | Exit 2 before CUDA: invalid ring dimensions versus byte count in old CPU records. This is a failed gate, not GPU numerical evidence. | `oracle-preflight-1673.logs.json.gz` |

All six negative cases use visible length 2052, budget 512 and seed 4242. Each
reports its intended specific mismatch. Queue status `failed` is expected for
these deliberately rejected cases. Initial parallel-admission attempts that
were lost or exited before execution are retained in the database and are not
counted as numerical results; serial reruns above supplied the evidence.

Optional resident trace collection was unavailable: the collector reported no
enabled resident trace producer. Device timing, CPU reference wall time and
client/build time are different clock domains. No speedup is inferred here.

## CPU reference attempts and fresh run

The old native/chunked stream used DSH builds
`build-3257364-1792497853067902053` (four positions),
`build-3588582-1792500249730496407` (four resumed positions), and
`build-3636801-1792500664399514576` (final position).
An independent portable/uninterrupted nine-position run completed in
`build-3600061-1792500347151099419`. DSH parity check
`build-3659594-1792500856648284611` compared both complete files exactly:

- Records SHA-256 `fe424adb48dbe81c97050086550c456e7001adb590f795b74cd0f696704ab1cb`.
- Checkpoint SHA-256 `394355fd52f6004ce6e4ab27cb3cc3387b3ce2cc68777a1b505dca03f9ef42de`.

These streams have the same malformed serializer, so their equality does not
establish a valid oracle or GPU agreement. They remain compiler/resume evidence.
An experimental two-thread loop compiler variant also matched those bytes in
`build-3745644-1792501584558826668`; no speedup is claimed. An O3/LTO experiment
was rejected by strict compiler diagnostics and was not used.

The fresh serializer validates shape/type/byte products before hashing any
source tensor. It corrects ring lengths and resets final-stream dimensions for
each die. CPU `oracle-validate` parses the complete stream using the production
reader before GPU submission. Fresh native reference generation started with
`build-3770830-1792501786926561393`, using full revision `36a8246...` in a new
data directory. A preceding invocation with an incorrectly typed short revision
annotation was cancelled by its owner before use; its partial artifacts are
not acceptance evidence.

## Pending execution

Corrected-format CPU parity/continuation, multi-position GPU comparisons, both
KV placements/launch modes, actual production-budget exclusion, and private
full-model request/snapshot matrices remain pending. The engine request gate
`06cb459e-0a40-4810-a87b-6bbab40758b4` failed in queue capture before core
startup: document 613 records `SQLite3::BusyException: database is locked`.
It is being replayed serially. An earlier queued invocation with incorrect SSH
argument grouping was cancelled by its owner before execution.

The replay `ee5de8c3-3472-4ef4-9328-791e6c5cae20` executed the engine matrix
and failed a real fingerprint comparison: VRAM/scalar passed all 16 requests;
VRAM/paired mode passed 14 before `final-A` changed recurrent and logit hashes.
The report and both captured response streams are retained in `engine-ee5d/`.
An unchanged repeat `93d25b21-3730-4b71-91f6-f192c85227c7` passed all four
placement/launch modes. The earlier mismatch is intermittent and remains
under investigation in [issue #109](https://github.com/nm-z/flash-sass/issues/109);
the passing repeat does not erase that failure or establish a root cause.

The first direct attempt failed; its record is retained in the database and
described in [the oracle notes](../ORACLE.md). No GPU job or resident holder
has been interrupted to advance this validation.
