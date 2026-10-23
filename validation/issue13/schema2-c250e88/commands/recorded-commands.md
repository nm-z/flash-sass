# Current-head reproduction commands

All GPU work ran through `ssh archy gpuq run --`; all builds and longer checks ran through `dsh build` on Benji. Temporary GPU inputs were staged under `/dev/shm/fs13-c250e88` and will be removed after evidence collection.

## Benji build and CPU checks

Successful current-head build `build-832079-1792587863466072272`:

```sh
dsh build --host benji --require-tool cc --require-tool perl -- bash -lc 'set -euo pipefail; make -C kern libengine.so trace_layout_test prefill_plan_test; ./kern/prefill_plan_test; ruby kern/trace_reconcile_test.rb; mkdir -p validation/issue13/schema2-c250e88/build-artifacts; cp kern/libengine.so kern/trace_layout_test validation/issue13/schema2-c250e88/build-artifacts/; cc -std=c11 -O2 -Wall -Wextra -o validation/issue13/schema2-c250e88/build-artifacts/trace_preflight validation/issue13/schema2-69b1e7c/commands/trace_preflight.c' --artifact validation/issue13/schema2-c250e88/build-artifacts/libengine.so --artifact validation/issue13/schema2-c250e88/build-artifacts/trace_layout_test --artifact validation/issue13/schema2-c250e88/build-artifacts/trace_preflight
```

Current-head GPU trace captures were reconciled by `build-957994-1792589164408047935`:

```sh
dsh build --host benji --require-tool ruby -- bash -lc 'set -euo pipefail; mkdir -p validation/issue13/schema2-c250e88/reconciliation-current; ruby validation/issue13/schema2-69b1e7c/commands/reconcile_captures.rb validation/issue13/schema2-c250e88 validation/issue13/schema2-c250e88/reconciliation-current; tar -czf validation/issue13/schema2-c250e88/reconciliation-current.tar.gz validation/issue13/schema2-c250e88/reconciliation-current' --artifact validation/issue13/schema2-c250e88/reconciliation-current.tar.gz
```

The direct `die.pl` ABI check run from `kern/` was:

```sh
if DIE_TRACE=1 DIE_CTA_TRACE=1 perl die.pl >/dev/null 2>&1; then exit 1; fi
DIE_TRACE=1 DIE_CTA_TRACE=0 perl die.pl >/dev/null
DIE_TRACE=16 DIE_CTA_TRACE=0 perl die.pl >/dev/null
```

It rejected the overlapping flags and generated both supported schema-2 trace modes.

The short host summaries were regenerated from the current reconciliations with:

```sh
ruby validation/issue13/schema2-69b1e7c/commands/summarize_host.rb cta1 validation/issue13/schema2-c250e88/reconciliation-current validation/issue13/schema2-c250e88/cta1-host-current-summary.json
ruby validation/issue13/schema2-69b1e7c/commands/summarize_host.rb cta16 validation/issue13/schema2-c250e88/reconciliation-current validation/issue13/schema2-c250e88/cta16-host-current-summary.json
```

## GPUQ layout and overhead runs

| Check | Exact GPUQ command | Job |
| --- | --- | --- |
| CTA1 layout | `ssh archy gpuq run --name fs13-current-c250e88-layout-cta1 -- /dev/shm/fs13-c250e88/kern/trace_layout_test /dev/shm/fs13-c250e88/kern/die_trace.cubin 1` | `a2473749-0bb6-46a5-97a5-0adf78da52b1` |
| CTA16 layout | `ssh archy gpuq run --name fs13-current-c250e88-layout-cta16 -- /dev/shm/fs13-c250e88/kern/trace_layout_test /dev/shm/fs13-c250e88/kern/die_trace16.cubin 16` | `81c7f7e6-f01e-4174-9454-802b20881de6` |
| CTA1 ABBA | `ssh archy gpuq run --timeout 600 --name fs13-current-c250e88-overhead-cta1-accepted -- ruby /dev/shm/fs13-c250e88/commands/run_trace_validation.rb /dev/shm/fs13-c250e88 /dev/shm/fs13-c250e88/evidence/schema2/config-cta1.json overhead-cta1-c250e88-accepted` | `7709b8dd-b35d-4a38-893b-6f8a723e2764` |
| CTA16 ABBA | `ssh archy gpuq run --timeout 600 --name fs13-current-c250e88-overhead-cta16 -- ruby /dev/shm/fs13-c250e88/commands/run_trace_validation.rb /dev/shm/fs13-c250e88 /dev/shm/fs13-c250e88/evidence/schema2/config-cta16.json overhead-cta16-c250e88-accepted` | `1385e908-b577-40ec-aa6e-eb4caa281b63` |
| Restore after control-file correction | `ssh archy gpuq run --timeout 120 --name fs13-current-c250e88-emergency-restore -- env FLASH_TRACE_VALIDATION_LOCK_HELD=1 ruby /dev/shm/fs13-c250e88/kern/trace_control.rb /dev/shm/fs13-c250e88/evidence/schema2/control.json restore` | `ac408909-6957-4918-a3bb-0ae270905664` |

## Lifecycle run

The current-head lifecycle wrapper uses the engine revision string `c250e88` because the checker compares it directly with `config rev=`. The successful job was `a87330b2-d055-42dc-8161-b29cbb42b0fb`:

```sh
ssh archy gpuq run --timeout 900 --name fs13-current-c250e88-lifecycle-final -- ruby /dev/shm/fs13-c250e88/commands/run_lifecycle.rb /dev/shm/fs13-c250e88 c250e88 lifecycle-c250e88-final lifecycle-control-c250e88-final
```

The eight available lifecycle captures were reconciled on Benji with `build-1092297-1792590467137970187`:

```sh
dsh build --host benji --require-tool ruby -- bash -lc 'set -euo pipefail; mkdir -p validation/issue13/schema2-c250e88/reconciliation-lifecycle-current; ruby validation/issue13/schema2-69b1e7c/commands/reconcile_lifecycle.rb validation/issue13/schema2-c250e88 lifecycle-c250e88-final validation/issue13/schema2-c250e88/reconciliation-lifecycle-current; tar -czf validation/issue13/schema2-c250e88/reconciliation-lifecycle-current.tar.gz validation/issue13/schema2-c250e88/reconciliation-lifecycle-current' --artifact validation/issue13/schema2-c250e88/reconciliation-lifecycle-current.tar.gz
```
