# Disk PLE read failures

Disk-backed PLE rows are model data. Each row must contain all 90 bytes.
`ple_read_exact` continues positive partial reads and retries `EINTR`.
Premature EOF and hard read errors preserve the token position, head, file
offset, received and expected byte counts, and errno/EOF reason. An unread tail
is not zero-filled or published as a valid row.

## Failure protocol

Before dispatch, the host stages the initial rows. Failure at this point returns
an explicit request error without launching kernels.

During persistent execution, the host publishes `UINT32_MAX` for the failing
position. Every die's PLEROWS operation checks that sentinel before consuming
the row, writes its error word at `CTL+0x1c`, and reaches the mandatory grid
barrier. The framework then ends all CTAs before dependent phases execute.
The PLE error-check descriptor flag is `0x20`; `0x08` and `0x10` belong to
paired execution and paired fusion.

A two-position program keeps its two PLE lookups as separate descriptors with
separate barriers and error checks. A failure in the first position must stop
before replaying the second position, whose rows might never be staged.

The host polls every die until all launch streams have completed. The watchdog
remains active if die 0 completes while another die is still running; it also
bounds the time since a known PLE failure. Only after all stream queries report
completion does the host synchronize and read the per-die error words.
A normal PLE failure returns engine status `-5` and invalidates the snapshot.
A watchdog/device-fault recovery is a failed validation, not evidence that the
PLE protocol completed correctly.

With bounded prefill chunks, a staged-ahead read can fail beyond the current
chunk's end. In that case every die completes that chunk normally, and the host
returns the PLE error before dispatching another chunk. No device error word is
expected for a row that no die reached. The test diagnostics record the launch
end and require every die's count to equal that earlier boundary.

Each new request clears readiness entries and device error words. Request IDs
exclude both zero and `UINT32_MAX`. A healthy request after failure recomputes
the invalidated prefix; a later matching request may reuse a new snapshot.

## Client behavior

The core sends an `error: PLE I/O ...` response rather than a normal `done`
response. The HTTP daemon carries the error detail into:

- HTTP 502 JSON for a non-streaming request.
- An SSE `engine_error` event for a streaming request, including a request that
  already emitted text.

The SSE `[DONE]` marker terminates the event transport after an error. It does
not mean inference succeeded. An affected stream must have no successful
`finish_reason` or success timing/usage payload.

## Build and CPU checks

Deploy `libengine.so` and the matching `die.cubin` together. An older cubin
does not understand the failure sentinel.

```bash
make -C kern die.cubin libengine.so libengine_pleiotest.so flashcore flashd
make -C kern test-ple-io-cpu
```

The CPU checks cover exact reads and metadata, descriptor construction, the
host's completion wait with simulated CUDA responses, socket deadlines, and
real HTTP/SSE error framing with a simulated core. These checks do not establish
real GPU termination or whole-model recovery. Measured results and their limits
are indexed in [validation/issue24](validation/issue24/README.md).

## Real target-GPU checks

Use an exclusively owned core with the real PLE table, all 48 layers, six dies,
at least 512 positions, and the matching test library/cubin. Check
`nvidia-smi` and `ps` before running; wait while another worker owns the GPU
runtime. The test programs take the socket of that owned core, and the HTTP
test starts its own loopback daemon.

```bash
perl kern/ple_io_check.pl "$OWNED_CORE_SOCKET" "$NEW_CORE_REPORT_JSON"
perl kern/ple_io_http_check.pl \
  --core "$OWNED_CORE_SOCKET" \
  --flashd "$FLASHD_BINARY" \
  --gguf "$TOKENIZER_GGUF" \
  --out "$NEW_HTTP_REPORT_DIRECTORY"
```

Run the matrix with `FLASH_TWO_POSITION=0` and `FLASH_TWO_POSITION=1`, each
with `FLASH_PREFILL_CHUNK=0` and `FLASH_PREFILL_CHUNK=64`.
Set a finite `FLASH_WATCHDOG_S` for the owned core and retain the complete core
log. Record the source revision, SHA-256 of the library/cubin/test drivers,
the engine's `config` response, GPU/process inventory, commands and exit codes.

Both runners enforce a whole-request deadline, including streams that keep
producing data. The defaults are `PLE_TEST_TIMEOUT_S=600` and
`PLE_TEST_FAILURE_BOUND_MS=60000`; reports record both bounds and measured
request/failure durations. A client timeout, watchdog recovery, or core restart
fails the matrix. Run each mode into a new report path.

The required evidence is:

| Case | Required observation |
| --- | --- |
| Full reads, multiple partial reads, EINTR | Identical generated tokens; counters show the selected read path actually executed. |
| EOF/EIO before launch | Exact failure metadata, no launched die and no successful response. |
| Failure with a matching existing snapshot | Snapshot exists and is resumed before failure; it is invalid afterward. |
| EOF/EIO during prefill | All six dies finish; both positions of a paired pass are covered. |
| EOF/EIO during decode | Valid output prefix precedes the error; no token or successful completion follows it. |
| HTTP/SSE propagation | Actual TCP requests, exact error metadata, and a nonempty text delta before the decode error. |
| Healthy reuse | Same outputs as baseline, cleared error words, cold prefix recomputation, then reuse of a new snapshot. |

The test-only `ple_io_test status` command reports cumulative launches and
per-request launch/completion/error masks, individual die counts/error words,
failure latency, resumed position, snapshot state and read-injection counters.
For six-die midrun failures within the current launch, the completion and error
masks must both be 63. A staged-ahead failure beyond a prefill chunk's end must
have completion mask 63, error mask zero, and all counts exactly at that launch
end before the failed position.
Prelaunch failures must have zero launch/completion/error masks.

Fault controls are compiled only into `libengine_pleiotest.so` using
`FLASH_PLE_IO_TEST`. Control messages use the owned core's Unix socket; inference
in the HTTP test uses real TCP HTTP and SSE. The model files remain read-only.
