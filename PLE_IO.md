# PLE read failures

Disk-backed PLE rows are model data. A failed row must not become a zero-filled
successful row. ple_io.h retries positive partial reads and EINTR, then reports
EOF or a system error with position, head, byte offset and bytes read/expected.

The host publishes the explicit UINT32_MAX readiness error tag for a failed
position. PLEROWS detects that tag before consuming the row and writes the die's
error word at CTL+0x1c. The PLE descriptor requires a grid barrier; the persistent
framework checks the error afterward, so all CTAs exit before later phases.
The host waits for all die streams, invalidates the prefix snapshot and returns
-5 with an error response. A prelaunch staging failure launches no kernel.
No ordinary done response is sent for these failures.

Each new request clears readiness entries and the device error word. Normal
request IDs avoid both zero and UINT32_MAX. This prevents a failed position's
generation-independent tag from poisoning a subsequent healthy request.

Deploy the matching libengine.so and die.cubin together. An older cubin does not
understand the failure tag. The production library contains no injection controls.
Only libengine_pleiotest.so, built with FLASH_PLE_IO_TEST, exposes them.

## Tests

    make -C kern ple_io_test
    kern/ple_io_test

The CPU test passed partial reads, EINTR, injected EOF/EIO, real truncated-file
EOF, invalid FD and precise failure metadata. Failed row tails remain untouched.

    make -C kern die.cubin libengine.so libengine_pleiotest.so
    python3 kern/ple_io_check.py /tmp/isolated-disk-ple-core.sock

The GPU test requires all 48 layers, the real disk PLE table and at least 192
positions. It compares successful partial/EINTR reads with a healthy baseline;
injects failures before launch and at position 160 during execution; checks that
failed responses have no done, that the snapshot is invalidated, and that the
same core recovers with injection disabled. It never modifies the model file.

Builds succeeded on archy. GPU execution and daemon error-path evidence must be
recorded before treating the full issue acceptance as complete.
