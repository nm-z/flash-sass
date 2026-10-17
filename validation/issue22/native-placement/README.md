# Native comparison of recorded placement captures

The C++17 comparator reprocessed the committed four-position, budget-512 and nine-position, budget-1 captures through the complete vocabulary of 248,320 logits per position. Both VRAM/pinned-host pairs passed exact zero-error limits for RMSE and mean KL and a top-1 agreement limit of one. This replay ran on the CPU; it did not execute a new GPU capture.

`validation/issue22/replay_placement.sh NEW_OUTPUT_DIRECTORY` verifies the original compressed logits, IDs, capture-info and metadata hashes before decompression, freezes new manifests, and runs the native comparison. `native-identities.txt` identifies the comparator binary and its source. The 54-assertion CPU suite is recorded separately in `../native-cpu.tap`.

These are same-implementation placement comparisons. Both reports retain `quality_pass:null` and `independent_checkpoint_evidence:false`. Missing original-checkpoint/tokenizer provenance and an independently executed reference remain explicit. The original captured IDs are short microfixtures, not a held-out natural-language evaluation or a production-budget long-context sparse-quality test.

Historical build/capture runners were removed from the PR tip to keep executable Python out of the change. The original `archy-06968d1/IDENTITIES.json`, commands, raw GPU records and output hashes remain unchanged as historical evidence. Its two runner entries describe files in the earlier recorded revision, not present-day deliverables. The current reproduction commands use the native comparator and shell script.
