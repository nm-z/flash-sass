# Commands and execution identities

Repository work and analysis ran on **engi** in
`/home/nate/codex/flash-pr30-work`, on branch `perf/phase-trace`.
GPU commands ran through SSH on **archy** in
`/home/nate/codex/flash-pr30-run`.
Raw inputs retain the command arguments, request bytes, responses, environment
settings, process identities and artifact hashes.

## Build

The trace images and layout executable were built from the unchanged kernel
sources identified in the manifests. The ordinary cubin hash remained unchanged.
The final engine library was rebuilt after commit c7d0a6e with:

```bash
LIBRARY_PATH=/opt/cuda/lib64/stubs make -C kern libengine.so -W engine.c FLASH_REV="$(git rev-parse HEAD)" > /home/nate/codex/flash-pr30-evidence/build-final.log 2>&1
```

At that invocation HEAD was
`c7d0a6e14e99edd9e91ea16c010ae7938ac647e0`.
The image/layout build targets are:

```bash
LIBRARY_PATH=/opt/cuda/lib64/stubs make -C kern die.cubin die_trace.cubin die_trace16.cubin trace_layout_test PTXAS11=/home/nate/codex/flash-pr30-evidence/ptxas11
```

Compiler, assembler version, build output and image hashes are retained.
The CUDA 11.4 ptxas executable SHA-256 was
`6180efbc0877b5cf127bc99c5d51be6a779d98a03811cc1c24ef61621dc20976`.

## Layout checks

Each command was run under the global validation lock and the GPU 6 reservation
lock, after checking `nvidia-smi`, `ps`, zero utilization and no GPU 6 compute
process. The Ruby preflight also checked the configured resident-holder identities.

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 timeout 10s kern/trace_layout_test kern/die_trace.cubin 1
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 timeout 10s kern/trace_layout_test kern/die_trace16.cubin 16
```

The locks were `/tmp/flash-sass-engine-validation.lock` and
`/home/nate/codex/asm-port/GPU-431371bd-b482-96b2-4b70-04f83c6d075b.lock`.

## Live lifecycle and overhead

The original lifecycle invocation used the c7d0a6e controller and checker:

```bash
bash /home/nate/codex/flash-pr30-run/evidence/schema2/run-lifecycle.sh
ruby /home/nate/codex/flash-pr30-run/kern/trace_validation.rb --config /home/nate/codex/flash-pr30-run/evidence/schema2/config-cta1.json --output /home/nate/codex/flash-pr30-run/evidence/schema2/overhead-cta1
```

The lifecycle wrapper, retained alongside this file, held the global lock and
restored the baseline in an exit trap. The runner holds that lock itself and
records restoration in its ensure path. Its configured helpers show the exact
reload environment. Do not add an outer flock around the runner.

This initial 16-CTA invocation failed during traced-mode preparation:

```bash
ruby /home/nate/codex/flash-pr30-run/kern/trace_validation.rb --config /home/nate/codex/flash-pr30-run/evidence/schema2/config-cta16.json --output /home/nate/codex/flash-pr30-run/evidence/schema2/overhead-cta16
```

After baseline service recovery and the memory guard fix, these commands checked
rejection and completed the two-output comparison:

```bash
bash /home/nate/codex/flash-pr30-run/evidence/schema2/check-memory-guard.sh
ruby /home/nate/codex/flash-pr30-run/evidence/schema2/prepare_two_token.rb /home/nate/codex/flash-pr30-run/evidence/schema2
ruby /home/nate/codex/flash-pr30-run/kern/trace_validation.rb --config /home/nate/codex/flash-pr30-run/evidence/schema2/config-cta16-two.json --output /home/nate/codex/flash-pr30-run/evidence/schema2/overhead-cta16-two
```

Both completed matrices use engine c7d0a6e. The second uses the 99c7134 controller.
Their captured config files pin different core process identities because the
original process exited during the failed allocation. The baseline library and
cubin hashes remained unchanged.

The final recovery startup, under the validation lock and after GPU/process
checks, was:

```bash
cd /home/nate/claude/flash-sass-live/kern
FLASH_PLE_DISK=1 FLASH_KV_HOST=0 FLASH_MAXT=262144 FLASH_ENGINE_MAXT=230000 FLASH_MAXIMG=4096 FLASH_TWO_POSITION=0 FLASH_PREFILL_CHUNK=0 FLASH_WATCHDOG_S=600 FLASH_TRACE=0 FLASH_TRACE_CTAS=1 FLASH_CUBIN=die.cubin ./serve.sh core 9>&-
```

Descriptor 9 was explicitly closed in the child so the restarted service did
not retain the validation lock. The earlier replacement started by this worker
had inherited the lock; it was closed with its line-protocol `quit` command
while idle, before the corrected startup. Original weight holders and other
workers' processes remained running.

For a new run, freshly capture the authorized idle core's PID, configuration and
artifact identities using the supplied preparation script. Its identity checks
will reject stale or different state. With the current controller, first select
`lifecycle-ordinary` before `lifecycle`; the original archived wrapper predates
that newly required memory precondition.

## Analysis

For each traced slot 02, 03, 06, 07, 10 and 11 in each successful matrix, run:

```bash
ruby kern/trace_reconcile.rb --dies 6 --out REPORT.json RAW_DIRECTORY/trace-?.raw.gz RAW_DIRECTORY/hosttrace.raw.gz
gzip REPORT.json
```

The committed reports were generated using that command shape with the absolute
raw paths retained in their metadata. Positions 1–4 of CTA-0 slot 02, and
positions 1–2 of 16-CTA slot 02, were also analyzed using `--position N`.
Lifecycle single, normal, stop-single, stop-multi, disconnect, recovery,
pair-first and pair-resume were analyzed with `--allow-incomplete`.
Argument/image failures intentionally have no GPU records to reconcile.

The aggregate files were generated with:

```bash
ruby validation/issue13/schema2-c7d0a6e/commands/summarize_evidence.rb /home/nate/codex/flash-pr30-evidence/measured-c7d0a6e validation/issue13/schema2-c7d0a6e
```

Extracting all bundles into one directory and copying the committed
`reconciliation` directory there provides the same analysis input tree.
The original absolute path strings in metadata are provenance; analysis reads
the files passed to it.

The synthetic regression command was `ruby kern/trace_reconcile_test.rb`.
The final callback-window recheck reads the archived disconnect hosttrace with
`TraceChecks.validate_file` and requires
`cancellation_callback_checked == true`; its JSON result is committed.
