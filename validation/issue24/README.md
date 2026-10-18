# Issue #24: measured validation and remaining gate

**PR #34 remains a draft. Real GPU failure termination and recovery have not
been measured for this revision.** The records below close the missing CPU,
build, transport-test, and provenance coverage; they do not substitute for
the required whole-model runs.

## Tested source and build

The exact source revision is
[`4524f06797812b6f9b35a8a158d094fe1728412c`](https://github.com/nm-z/flash-sass/commit/4524f06797812b6f9b35a8a158d094fe1728412c).
It was exported with `git archive`, extracted into an isolated directory on
archy, and built with `FLASH_REV` set to that full revision.

| Record | Result |
| --- | --- |
| [Source/export identity](review/source.json) | Source revision, archive SHA-256, execution location and explicit GPU status. |
| [Build command and toolchain](review/archy/build-meta.log) | GCC 16.2.1; CUDA 11.4.152 ptxas; 2026-10-18 00:18:34–00:19:08 UTC. |
| [Raw build output](review/archy/build.log), [exit code](review/archy/build.rc) | Production library, test library, persistent cubin, core and HTTP daemon built; exit 0. |
| [CPU regression output](review/archy/cpu.log), [exit code](review/archy/cpu.rc) | `make -C kern test-ple-io-cpu` completed; exit 0. |
| [Production injection gate](review/archy/production-gate.log), [exit code](review/archy/production-gate.rc) | The production library has no `ple_io_test status` command; exit 0. |
| [Binary, driver and dependency hashes](review/archy/sha256sums.txt) | SHA-256 of both libraries, cubin, core, HTTP daemon, Perl drivers/helpers and MaxAs files. |
| [MaxAs identity](review/archy/maxas-identity.txt) | External dependency revision plus hashes of the repository's overlaid MaxAs patches. |
| [Worker execution identity](review/local-proof.json) | engi host and process identity; commands reached archy via SSH. |

The built persistent cubin SHA-256 is
`35787c07056e4c6947bc519149f49c6f8a5fc2375fd98b155a1b1dd76c341193`.
The test library SHA-256 is
`3ad2400bc87f988a6e2065b6aecd1de5556c23b4ed32f574edb5f23045e886db`.

The build log includes assembler register-bank conflict diagnostics and C
warnings. Its successful exit establishes that these artifacts build with the
target toolchain; it does not establish their runtime correctness.

## What the CPU results establish

| Check | Measured observation | Limit |
| --- | --- | --- |
| Exact 90-byte reads | Full read: 1 call; injected partial reads: 12 calls, 11 partials; EINTR: 3 calls, 1 retry. All three produced identical bytes. | CPU reader, not model inference. |
| EOF and EIO | Both preserved position/head/offset, 17/90 received bytes and distinct EOF/errno. Unread tail remained untouched. Real truncated-file EOF and invalid-fd cases also passed. | Does not exercise GPU readiness. |
| Descriptor pairing | PLE failure flag is separate from paired control flags; the two PLE descriptors remain separate while following eligible work stays paired. Descriptor and BF16 pairing plan checks passed. | CPU descriptor construction. |
| Host completion wait | Six simulated launch streams completed before any blocking context synchronization. Delayed-peer error returned in 4.642 ms. A stuck peer reached the watchdog in 50.563 ms without blocking synchronization. | CUDA calls are mocked. These numbers are not GPU failure latencies. |
| Socket deadlines | 21 checks passed, including fragmented/NUL-containing lines, read/write bounds, and an absolute request deadline despite continuing valid lines. | Local CPU sockets. |
| HTTP/SSE propagation | Ten cases used the real `flashd.c` HTTP handler over actual loopback TCP. EOF/EIO errors were tested before tokens and after decoded text, for JSON and SSE. Exact metadata survived; failed SSE had one error and one transport `[DONE]`, without a successful finish or usage/timings. | Tokenizer and core responses are simulated. No model inference occurred. |
| EOF after a read timeout | Both JSON and SSE completed at approximately 3000 ms when the simulated core closed after the daemon's two-second read timeout. | CPU regression for stale errno handling. |
| Perl runtime drivers | Both drivers passed syntax checks and enforce absolute request bounds. | The real-GPU driver matrices have **not** been executed. |

The C host-loop and HTTP tests include the production implementation. They
replace CUDA or core/tokenizer inputs explicitly, so the scope of each result
is visible in both source and logs.

## Why the GPU gate is still open

[The GPU and process inventory](review/archy-availability.log) records another
worker's resident `flashcore` PID 1575007 on dies 0–5. Those dies had only
3–77 MiB free. The resident process uses a different library revision.
Checks with both `nvidia-smi` and `ps` were repeated during this work.
No resident core, daemon, weight holder, or another worker's process was
terminated, reloaded, or reconfigured. No GPU validation was launched.

The required exclusive six-die runtime window was unavailable. Consequently,
none of these results is claimed:

- Real prelaunch launch suppression and bounded persistent failure termination.
- Failure observation and normal exit on all six physical dies.
- Real snapshot invalidation, cold recovery and later healthy snapshot reuse.
- Whole-model output equivalence for full reads, partial reads and EINTR.
- HTTP/SSE propagation of an actual model I/O failure after decoded output.

## Prepared runtime matrix

See [PLE_IO.md](../../PLE_IO.md) for setup and commands. Run both
`FLASH_TWO_POSITION=0` and `FLASH_TWO_POSITION=1` with an owned six-die,
48-layer disk-PLE core and the matching source/library/cubin.

The Perl core driver covers positive partial reads, EINTR, unused rows after
early stop, prelaunch EOF/EIO at positions 0 and 1, a matching snapshot before
failure, decode position 160 with an exact emitted token prefix, both sides of
a paired prefill at positions 160 and 161, cold recovery and new snapshot reuse.

The Perl HTTP driver makes actual TCP requests through its own daemon. It
covers 16 EOF/EIO cases across prelaunch, both paired prefill positions and
decode, each as JSON and SSE. The decode fault position is derived from the
baseline's measured prompt length. Its SSE check requires nonempty text before
the error. Every failure is followed by cold recovery and a later warm request.

Both drivers require exact row metadata, evidence that injection actually
executed, per-die error/count readbacks and completion masks. A midrun failure
must complete with all six bits set in the completion and error masks. A
watchdog reset, socket timeout, process restart or missing status response
fails validation. Reports retain actual durations, configured bounds,
responses, control transcripts, config and driver identities.

Only after these real-GPU runs pass, their raw evidence is committed, and all
review feedback is resolved should PR #34 be marked ready.

## Historical records

The top-level [cpu.log](cpu.log) and [build-archy.log](build-archy.log) predate
this revision. They are retained as historical records; the `review/archy/`
records above identify and validate the current source. The correct evidence
directory is `validation/issue24`.
