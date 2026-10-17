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
zero keys, unsupported budgets and positions above 262144 before allocation or
CUDA initialization. idxscore accepts zero complete blocks because a nonempty
prefix may have no completed block; idxsel/idxfull require a nonzero block count.

CPU validation:

    cc -std=gnu99 -O2 -fno-fast-math -ffp-contract=off -Wall -Wextra \
       -Iref kern/checked_test.c ref/checked.c ref/qsa.c ref/attn.c \
       ref/fmath.c -lm -o /tmp/flash-checked-test
    /tmp/flash-checked-test

After building qsa_test and attn_test on the target host:

    cd kern
    python3 checked_cli_test.py --out /tmp/checked-cli-new

The negative CLI suite hides CUDA and intercepts cuInit/cuLaunchKernel with a
guard that exits 99. Invalid invocations must return 2 instead. The recorded
43 cases passed on archy. Scalar tests passed on engi, including one-key attention,
gather success and unchanged outputs on invalid calls. Raw logs and binary hashes
are under validation/issue20.
