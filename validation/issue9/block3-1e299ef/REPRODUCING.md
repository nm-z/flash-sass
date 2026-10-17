# Reproducing the recorded block check

The original machine-specific runner source has been removed from this PR tip to keep Python out of the change. The original `IDENTITIES.json` remains an immutable record of that capture, including the historical runner hash. The command arrays, environment, expected and actual exit codes, GPU inventory and raw outputs remain in this directory.

Use the exact command and environment recorded in `gpu-results.json` when reproducing the six block checks with the built `kern/model_test`. Coordinate exclusive execution access to all six GPU dies named by the holder configuration before launching; a lock on a different GPU does not reserve those dies. Keep the existing holders intact. Reproduction requires the recorded real tensor cache, holder manifests, CUDA/MaxAs build and model/PLE files.

The two placements each contain a positive case and selection/layer fault-injection controls. These runs cover actual weights for block 3 with budget 1 and nine teacher-forced microfixture positions. The reduced budget ensures the indexer excludes complete blocks. They do not cover a complete 48-layer quality evaluation, embedding/head correctness, production-budget long-context selection, or natural-language recall. Those remain separate gates described in the production QSA test documentation and issue #22.
