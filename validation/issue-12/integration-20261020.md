# Trace integration with current master

The merge incorporates master `c91e46a1c60ff3ecbea89de3a2b6a6790f8c0d18` while preserving both opt-in tracing implementations and the current jump/tree dispatcher. `DIE_TRACE` and `DIE_CTA_TRACE` now fail before assembly output when both are enabled: their control fields overlap at `ctl+0xd0..0xe4`, but their record layouts differ.

CPU checks on engi, 2026-10-20:

- Seven environment combinations passed: unset/unset, phase only, CTA only, both enabled (required rejection with no output), and each explicit-zero combination.
- Ordinary generated assembly is byte-identical to the same master generator: SHA-256 `2bb0d65a69d37baad1ad922c4a9e1011ecc34f61d3f6804136c680dd29fa7776`.
- Phase-traced generated assembly is byte-identical to master with `DIE_TRACE=1`: SHA-256 `a53d68cc7c4c88bce9fab2d3d3ba8c240a3e86ebf6f179f60cc7f19e999b6c60`.
- `perl kern/cta_remap_matrix.pl --self-test`: 61 checks passed.
- `perl kern/cta_remap_summary_test.pl`: 24 checks passed.
- `prove kern/QsaEvidence_test.pl`: 10 checks passed.
- Fresh native `attnpartn_regress` build and `--self-test`: 16 canary negative cases, 14 trace negative cases and 64 mapping cases passed. CUDA stubs were used for this CPU-only check.
- `git diff --check`: passed.

The 144-case archived GPU component run retains its original source and binary identities. This merge is not represented as a new GPU execution. The whole-model experiment remains pending under issue #12. The latest owner review requested this concrete generator integration without repeating the large component matrix.
