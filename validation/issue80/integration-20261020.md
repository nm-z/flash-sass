# Request deadline integration — 2026-10-20

Integrated master c91e46a1c60ff3ecbea89de3a2b6a6790f8c0d18 into the request-limit branch and corrected both defects identified in review.

## Behavior

One absolute monotonic deadline starts at request worker entry. Queue admission rejects an already expired request before taking the idle engine slot. HTTP reads, engine and vision socket connection attempts, partial engine writes, streaming client writes, and engine response reads use the remaining budget. Rendering/tokenization boundaries check the same deadline before proceeding to dispatch.

Engine request serialization no longer flushes a buffered FILE into an unbounded socket write. Nonblocking sends retain their partial offset, and EINTR/EAGAIN retries do not renew the deadline. AF_UNIX backlog EAGAIN retries a new connection rather than mistaking a zero SO_ERROR value for a connection.

When supported, cancellation half-closes engine input and drains the terminal timing record. Cleanup ends at the earlier of the configured drain bound or two seconds after the request deadline. Error delivery has its own one-second best-effort budget; it does not renew the request. A timed-out response records incomplete/aborted timing and never reports a successful socket decode rate.

## Fresh CPU validation on engi

- Built flashd, flashd_timing_test, and timings_test.
- Production helper tests pass, including expired idle admission, partial socket writes against a non-reading peer, and timeout response delivery after the request budget expires.
- The existing flashd_timing_test.pl suite passes all 35 top-level tests, including nested timing, streaming, half-close, legacy core, cancellation, cap, and fragmented response cases.
- make -C kern test-flashd-deadline passes all 26 HTTP/fake-core assertions. It creates its own tensor-free GGUF tokenizer and private socket/frontend processes.
- A 250,000-token prompt sent 219,264 protocol bytes before timing out at 1.004 seconds with HTTP 504. The next request waited 862.904 ms in the queue and returned HTTP 200.
- A running stream emits text, a request_timeout error, exactly one DONE marker, and aborted timing.
- A queued request expires without connecting to the engine while its predecessor is draining. Noncooperative drain completion is bounded, and a subsequent request succeeds.
- Source diff whitespace checks pass.

The existing vision filename truncation compiler warnings are unchanged. These checks use CPU fixtures; no model inference, GPU campaign, or live service deployment was needed for this functional HTTP fix. Archived issue80 logs retain their original tested identity and are not presented as results of this integration.
