# Issue #20: maximum idxscore fixture position

Merged PR #33 supplies the checked scalar APIs, output-preserving rejection,
strict CLI parsing, and the original prelaunch rejection evidence. One endpoint
remained wrong on master `bf53cfd7c2b89972fa6adb2235531afa2da05a63`:
`qsa_test idxscore 65536 512` passed its input bounds, then generated position
262145 (262146 visible tokens) for a capacity of 262144.

The reproduction script extracts the baseline's RNG and actual fixture
expression into a CPU program. Its output is in `before.log`; the expected
failed capacity assertion returns 1, recorded in `before-status.txt`. This is
a CPU reproduction of the dimension bug, not a GPU execution.

## Fix and acceptance

The checked fixture helper calculates the position before `dev_init()`. It
preserves the requested complete blocks and limits the generated tail to the
remaining capacity. Maximum complete blocks use position 262143 for every tail
choice; zero complete blocks still use one visible current token at position 0.
The random draw sequence for query and pool contents is unchanged.

| Issue #20 acceptance | Evidence |
| --- | --- |
| Reject empty attention with unchanged outputs | Existing `checked_test` passes at this source; `ref/checked.c` and `ref/checked.h` are unchanged. |
| State and enforce positions, budgets and capacities | `CHECKED_APIS.md`, the checked APIs, and the pre-CUDA fixture helper. |
| CPU zero/one-key and output-contract tests | `checked-api.log` records the existing suite passing with exit 0. |
| Reject invalid GPU harness inputs before launch | 52 negative CLI cases return 2; seven accepted controls return 99 at the CUDA-entry trap. The production `engine_run` empty-prompt guard is unchanged. |
| Valid maximum positions against capacity; saved commands/status | 35 CPU position cases pass, including all tails at blocks 65536/capacity 262144; `status.tsv` records every verification command and exit status. |

The fixture's tests near `UINT_MAX` check integer arithmetic only. They do not
expand supported GPU capacity or the scalar APIs' allocation bounds.

## Reproduce

From the repository root:

```bash
bash validation/issue20/position-capacity/reproduce-before.sh \
  bf53cfd7c2b89972fa6adb2235531afa2da05a63 /tmp/issue20-before
# Expected exit 1: position=262145, capacity=262144.

bash validation/issue20/position-capacity/verify.sh /tmp/issue20-verify
# Expected exit 0.
```

Use new output directories. `verify.sh` uses the CUDA toolkit's link stub and
the CLI suite's `cuInit`/`cuLaunchKernel` trap, so neither CUDA initialization
nor a GPU launch occurs. Set `CUDA_STUB_DIR` if the toolkit's stub directory
differs from the default path. This issue's boundary acceptance does not require
a numerical GPU run; no shared archy process was touched.

## Recorded run

Tested implementation commit: `99c2fc0b9f879be657e58cf68d66c5ce676a5908`.
Host, UTC time, compiler, and source identity are in `identity.txt`; source and
binary hashes are retained separately. The tested code under kern/ref is
identical to the final PR head. All five verification commands returned 0. The expected baseline
reproduction returned 1.

`cli-report.json` retains all 52 invalid cases and seven accepted controls.
Accepted-control exit 99 proves the trap is active and that valid boundary
arguments reach it. It does not report GPU arithmetic, model quality, or any
part of the separate long-context numerical matrix.
