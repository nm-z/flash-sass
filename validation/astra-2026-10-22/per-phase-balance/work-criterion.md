# Per-phase work requirement

Nate clarified the requirement on October 22:

> less than 500 phases/token, and within each phase, each die must do the same amount of work within 10% for any given die.

Use the pairwise bound `max(work across dies) / min(work across dies) <= 1.10`
for every physical phase. Do not substitute deviation from the six-die mean,
or an average across token positions. Account for every child of a compound
phase and input-dependent work. A phase with some idle dies and some working
dies fails; equal zero-work components do not establish useful balance.

The current program has 492 phases per token on each of six compute dies.
Known source assignments include 104/112 routed gate rows, 424/432 down rows,
416/448 GR columns, eight GDN heads and four QSA query heads per die. These
dimensions individually satisfy the pairwise bound. The completed v79 audit
now expands all 979 descriptors per die and compares 7,091 geometry
components, including padded matrix rows, with no failures. The v80 full-model
run additionally exercises long-context QSA selection and confirms identical
query values, scores and selected IDs across the six dies; see
[the production work audit](work-audit-v79.md).

Elapsed time and exchange waiting remain separate performance evidence.
Passing all 492 window-averaged timings does not establish this requirement;
neither does equal residency establish equal useful work or active SM count.

`BENCH_WORK_DUMP=1` requests the descriptors retained on the GPUs after a
completed full-model run. The optional readback includes all children, words,
die/phase indices and coverage counts; it does not execute additional kernels
or assert that the work is balanced. It has been exercised end to end in
canonical job `89cebaae-2527-4d69-832e-c54c9f093664`; the actual work table
matches the offline production builder's table exactly.

Physical GPUs 2 and 7 remain reserved for Nate. Compute uses 0, 1, 6, 3, 4, 5.
