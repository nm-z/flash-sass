# Compact CPU suite result

The issue #22 CPU suite ran through DSH on Benji in build
`build-2630825-1792564997771592028`, using source commit
`6b52888d5fd354184a2af80671be2309c92f7f3f`. All 25 entries in `exits.tsv`
are zero. It covered comparator and export validation, analytic and frozen
indexer/attention checks, placement and recovered-reference replays, and
synthetic lengths N=1, 2047–2055, 3100, 8000, 65000, and 262000.

`complete.json` confirms unchanged source hashes, CPU-only execution, and
`issue22_resolved:false`. This is not GPU execution or original-checkpoint
quality evidence. Large reproducible synthetic tensors and temporary FP64
intermediates are omitted; `validation/issue22/run_cpu.sh` regenerates them.
