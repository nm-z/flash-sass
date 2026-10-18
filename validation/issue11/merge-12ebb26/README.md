# Combined PR #31 regression check after 12ebb26

The shared branch was fetched and merged before pushing. This run tests the
combined source `12ebb26f0f83ad1c1331862f160d2fb633b6d896` on `engi`, preserving
all collaborators' capture and comparison commands.

| Check | Measured result |
|---|---|
| Candidate capture serializer | PASS: 92 assertions |
| Descriptor preservation | PASS: 90 cases, 2250 descriptors, 1320 expected pointer substitutions |
| Independent reference trace helper | PASS: 156 checks |
| GPU gate with mocked inventory and real flock/timeout | PASS: 32 checks |
| Strict logit comparator | PASS: 32 cases, zero failures |
| Native oracle suite | PASS: 58 expected outcomes, provenance consistent, no changed sources |

The combined command exited 0. Its log and the native/comparator reports are
retained here. These are CPU checks; they are not a new GPU measurement.

The actual M60 suite at `ca4a476` is preserved separately under
`../target-ca4a476`. The source-equivalence check confirms that its component
oracle, production kernel/program builders and assembler patch inputs did
not change in this merge. The added production engine capture mode is a
separate target-validation requirement and is **not** covered by that statement.

## Remaining current-model blocker

The fresh full-model probe under the shared lock failed `cuMemAlloc(&p, bytes)`
with `out of memory`, exit 1. Its exact source/image identities, command and
memory inventory are under `../target-ca4a476/model-probe`.

The collaborator's `FLASH_QSA_CAPTURE=1` path was inspected. It is selected at
`engine_init`, allocates additional per-QSA-layer GPU scratch, and requires a
fresh completed teacher-forced request before export. It is not a read-only
switch that can collect previously discarded outputs from the current shared
engine. Enabling it there would require a coordinated engine reinitialization
or reload; the shared lock alone does not authorize reconfiguring another
worker's engine. No such operation was performed.

A successful current full-model capture therefore still requires an available
validation runtime with enough device memory, or explicitly coordinated use
of a capture-enabled resident runtime. Original checkpoint/tokenizer and exact
conversion/gamma lineage, capture-enabled/disabled target output parity, and
the paired long-sparse full-model acceptance remain unverified. PR #31 remains
draft; the earlier historical numerical failure is not relabeled as a current
model result.
