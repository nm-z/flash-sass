# Original real-GPU SSE framing failure

This is selected failure evidence from frozen source `0f8b09cad5dde8b8089c8b1c9f7065cfac43e9ae`, run at **06:57:03–06:57:11 UTC** on 2026-10-18. It is an earlier failed run, not the accepted run in the parent directory.

The CLI suite and HTTP JSON case passed, then the HTTP SSE checker exited **25** when it decoded the event's `data` field. The raw [SSE body](sse.body) contains an unprefixed newline between `timings.engine` and `timings.http`. The preceding engine protocol's terminal LF had been copied into the JSON. Plain JSON accepted the whitespace, while SSE field parsing discarded the unprefixed continuation and left incomplete JSON.

[http-pair0.log](http-pair0.log) records the parse failure. The request and response metadata are retained alongside the body. [The supervisor result](supervisor-results.json) and [queue log](queue.log) show failure and verified baseline restoration before the lease was released. Those records contain paths into the original full evidence directory; this folder intentionally contains only the selected failure artifacts.

The strengthened CPU fixture independently reproduces the parse error before the frontend fix (exit 255). The frontend now removes terminal LF/CRLF framing while preserving the JSON bytes. The accepted `8ad9d6e` run passes both JSON and SSE in scalar and paired modes.
