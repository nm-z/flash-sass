# Production-QSA harness integration

Merged master `c91e46a1c60ff3ecbea89de3a2b6a6790f8c0d18`, preserving the shared QSA layout/binding helpers, partial-layer fixtures and `hold_cache`, together with master's phase tracing, timing records, PLE readers, guarded direct-test prerequisites and pthread linking. Layer binding is followed by the current `kv_rawmax` assignment so trace state is preserved.

CPU/build verification on engi, 2026-10-20:

- Fresh `libengine.so`, `dies_test`, `model_test`, `hold_cache` and `qsa_direct_test` builds passed, using the CUDA headers and link stubs at `/opt/cuda`. This is a build check, not target execution.
- `perl kern/test_qsa_engine_requests.pl`: 93 assertions passed, including 28 complete synthetic captures and 25 corrupt captures.
- `perl kern/run_qsa_model_matrix.pl --self-test`: 8 child-result checks, 3 stale/scope checks and selection/control planning passed.
- `CUDA_STUB_LIBRARY=/opt/cuda/lib64/stubs/libcuda.so bash kern/test_model_cli.sh`: 26 checks passed. The CUDA boundary was intercepted; no GPU calls executed.
- `git diff --check`: passed.

The recorded block-3 target runs retain their original executable source identity. No new GPU result is claimed by this merge. Review readiness covers the implemented production-QSA harness and recorded checks, as scoped in the PR description; full-model/checkpoint-quality acceptance remains separate and incomplete.
