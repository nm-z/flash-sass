# Current-head GPU verification status

Source revision: `a62303283f993acb8a13188941b00f5bc9ef340b`.

The current-head C/CUDA build completed on Benji through DSH. The exact build log, DSH configuration, toolchain hashes, and returned artifact hashes are in `build-current-head.log` and `build-current-head.json`.

No current-head GPU matrix or profiler job has run. The required Archy queue access failed before `gpuq` started:

```text
ssh archy gpuq search --help
Permission denied, please try again.
Permission denied, please try again.
nate@192.168.1.218: Permission denied (publickey,password).
```

This is tracked in [gpuq #25](https://github.com/nm-z/gpuq/issues/25), a sub-issue of #12. A fresh KWin session attempt also exceeded its 20-second host-scan limit and is recorded in [kwin-mcp #722](https://github.com/nm-z/kwin-mcp/issues/722), a sub-issue of #704.

The checked-in 40-pair matrix uses source revision `7c6f34d4372567fd97f1cef9b609d3dc0c81b0d3`; its captured-format IDs are generated fixtures. It does not verify this head's per-token conditions or production-captured selections. The matrix report marks `hardware_counter_gate` as `not_run`. The earlier profiler probe exited 124 with `CUDA profiling error` and collected no counters.

Next step: restore authenticated Archy queue access through the viewer, query GPUQ history for reusable runs, then run the current-head paired matrix and separate Maxwell-compatible system-memory/VRAM byte, transaction, and stall profiles. Keep the PR draft until these results are recorded.
