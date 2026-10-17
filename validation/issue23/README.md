# Depth benchmark exit-status validation

`kern/depth.sh` saves the actual HTTP request and response and invokes the core-module Perl validator. Exit 0 means the expected answer was present and the timing record passed validation; exit 1 means a wrong answer; exit 2 means malformed, failed, or incomplete transport/protocol/timing evidence.

Run `perl kern/depth_check_test.pl` from the repository root. `perl-http.tap` records 55 passing assertions across 15 local HTTP fixtures and one transport failure. These exercise the real shell script and validator, including a wrong answer, missing answer/timings, API error, invalid interval rate, malformed JSON, HTTP 503, single-output null and legacy zero rates, no output, unknown counts, canonical success, and canonical unknown/aborted states.

The fixture does not load a model or establish recall quality. A live benchmark must preserve request IDs, checkpoint/configuration, capacity, raw response and correctly separated timing fields.
