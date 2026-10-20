# Full-model validation inputs and run record

Refs [issue #9](https://github.com/nm-z/flash-sass/issues/9) and
[PR #105](https://github.com/nm-z/flash-sass/pull/105).
Numerical/GPU acceptance is pending. These files first preserve the completed
input export; they do not establish model correctness.

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

The numerical/reference and engine revision is fixed to
`39027871256e578fa8339681a661f59de53233c4`. Later documentation and validation
tooling commits may advance the PR head; each scalar input seal also verifies
the actual numerical source contents, including the generated opcode header.

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
variant's 12 API/CLI checks passed. Actual-weight record equality against the
portable baseline is still required before using it for the deep reference.

## Pending execution

- Native nine-position, budget-1 scalar reference: first four-position chunk
  started through DSH `build-3257364-1792497853067902053`, outside GPUQ.
- Independent portable/uninterrupted record parity, checkpoint continuation,
  GPU comparisons and full request/snapshot matrices remain pending.
- Patched direct component matrix is queued as
  `ef56e0c2-ad05-488d-9cdc-e3992ba72f9e` in the issue9 GPUQ history database.

The first direct attempt failed; its record is retained in the database and
described in [the oracle notes](../ORACLE.md). No GPU job or resident holder
has been interrupted to advance this validation.
