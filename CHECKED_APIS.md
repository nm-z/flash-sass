# Checked attention dimensions

ref/checked.h provides checked scalar entry points. They return 0 on success and
1 for invalid input; rejected calls preserve outputs and output counts/sizes.
The original instruction-order *_gpu twins retain their signatures and require
valid caller-supplied bounds.

An attention key count is nonzero. Size checks occur before unsigned rounding
and size_t byte multiplication. A zero-based QSA position must be below physical
capacity; budgets are 1..4096. The unfinished source block requires four
allocated positions even when fewer are logically visible. Selection IDs must
be valid, unique and ascending when complete blocks exceed the budget.
Destination float capacities include every physically gathered row.

These are scalar layout/bounds checks. They do not establish SASS prefetch
allocation bounds or support arbitrary nonfinite padding; see #18/#19.

Both attention test CLIs accept unsigned decimal digits only. They reject
empty/signed/malformed/overflowing arguments, unknown modes, extra arguments,
zero keys, unsupported budgets and visible lengths above 262144 before allocation
or CUDA initialization. Supported zero-based positions are 0..262143. idxscore
accepts zero complete blocks and uses position 0: one current token is visible
after append. idxsel/idxfull require a nonzero block count. The idxscore fixture
position is computed and checked before CUDA initialization; its generated tail
is limited to the remaining capacity. At 65536 complete blocks it uses position
262143 with no tail. The RNG draw order for the fixture data is preserved.

CPU validation:

    cc -std=gnu99 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra \
       -Iref kern/checked_test.c ref/checked.c ref/qsa.c ref/attn.c \
       ref/fmath.c -lm -o /tmp/flash-checked-test
    /tmp/flash-checked-test

    make -C kern idxscore_position_test
    ./kern/idxscore_position_test

After building qsa_test and attn_test on the target host:

    cd kern
    perl checked_cli_test.pl --out /tmp/checked-cli-new

The negative CLI suite hides CUDA and intercepts cuInit/cuLaunchKernel with a
guard that exits 99. Invalid invocations must return 2 instead. The recorded
43 cases passed on archy. Scalar tests passed on engi, including one-key attention,
gather success and unchanged outputs on invalid calls. Raw logs and binary hashes
are under validation/issue20.

## Refreshed master integration

Source 682433079f9d2d02682d32a0a1b2407c50a06488 includes the current master fixes.
The archy build of checked_test, qsa_test and attn_test succeeded; the scalar
suite passed and all 46 invalid CLI cases returned 2 before the CUDA traps.
The outer SSH wrapper timed out at 30 seconds, so its exit 124 is retained
separately from the completed child-case report. These are boundary tests;
no successful GPU kernel execution is claimed by the CLI guard suite.
Evidence is in validation/issue20/current-archy. The unsuccessful engi GPU
link attempt is retained in current-engi; archy supplied the successful build.

## Maximum-position follow-up

The idxscore position regression covers all four generated tails at zero and
maximum complete blocks, the adjacent lower block count, a partial final block
at a smaller capacity, arithmetic near UINT_MAX, and unchanged outputs on
invalid dimensions. UINT_MAX cases test the helper's arithmetic only; the GPU
harness capacity remains 262144 and the scalar APIs retain their own size bounds.

The CLI suite now checks 52 invalid invocations (exit 2) and seven accepted
boundary controls (exit 99 at the CUDA trap), including zero and maximum complete
blocks with budgets 1 and 4096. The controls also verify that the trap is active.
These CPU and prelaunch checks do not claim GPU arithmetic or model validation.
