# Depth benchmark exit-status validation

`kern/depth.sh` saves the actual HTTP request and response and invokes the core-module Perl validator. Exit 0 means the expected answer was present and the timing record passed validation; exit 1 means a wrong answer; exit 2 means malformed, failed, or incomplete transport/protocol/timing evidence. Consumers must accept throughput samples only when the command exits 0; a printed FAIL line is diagnostic evidence, not a successful sample.

Run `prove -v kern/depth_check_test.pl` from the repository root. The current suite has 203 assertions across 49 controlled local HTTP fixtures and one refused-connection transport failure. It executes the real shell harness and validator. The original `perl-http.tap` is retained as the historical 55-assertion record from merged PR #27.

## Prefill-rate follow-up for issue #23

Master `3d57d734d63aa74f167a6b5a02c616ddc018d2ff` already rejects wrong answers and JSON API errors, but still accepts inconsistent prefill throughput. A response with 100 prompt tokens, 2,000 ms, and a supplied rate of 999 tok/s prints PASS and exits 0; the correct uncached rate is 50 tok/s. The follow-up rejects that record with exit 2 and no successful measurement on stdout.

The validator checks prefill throughput against `prompt_n - cache_n` and `prompt_ms`, matching the producer in `kern/flashd.c`. An omitted legacy `cache_n` means zero cached tokens; a supplied cache count must be a nonnegative integer no greater than the prompt count. A fully cached prompt cannot advertise positive prefill throughput.

Milliseconds and rates are independently rounded by the HTTP producer to 0.1 ms and 0.01 tok/s. The validator checks that the rounding intervals can represent the same processed-token count, including short positive durations that round to zero and small positive rates that round to zero. Legacy zero-duration records may still report zero, and canonical records may report null, to indicate unavailable prefill throughput. Missing rates with a positive measured duration and uncached tokens remain invalid.

Nineteen added HTTP fixtures cover inconsistent, absent, negative, and nonnumeric prefill rates; cached and fully cached prompts; invalid cache counts; and valid and invalid rounding cases. Every protocol failure must emit no measurement on stdout. The successful sample set is checked against the expected valid cases, and the original wrong-answer, API-error, missing-answer, missing-timing, malformed-JSON, HTTP-error, single-output, unknown-count, and aborted-response checks remain covered.

The prefill follow-up suite failed 17 of 143 assertions against the original master validator and passed all 143 against the prefill fix. See [before/after results and tested source hashes](prefill-rate/results.json), [the raw counterexample](prefill-rate/counterexample.json), [master regression output](prefill-rate/master-regression.log), and [fixed regression output](prefill-rate/fixed-regression.log). All were executed on engi using CPU-only local fixtures; no GPU or model request was needed.

## Follow-up review of PR #77

Review of PR #77 at `2bea319767634af183b2a772ba693cfd8561bad8` found that its original 143 assertions passed, but the decode check still accepted an impossible rate: 10 intervals in 0.1 ms at 1 tok/s printed PASS and exited 0. Its symmetric tolerance around a rounded duration admitted rates below the entire valid rounding interval.

The prefill and decode checks now share the rounding-interval calculation. A decode sample must reconcile its interval count, duration, and rate, including positive durations or rates that round to zero. Zero decode intervals continue to permit an unavailable null/zero rate. The invalid 1 tok/s record now exits 2 and prints no measurement on stdout.

Eight added HTTP fixtures cover inconsistent low and high decode rates, both valid extremes of a duration rounded to 0.1 ms, a positive duration rounded to zero, an invalid rate at zero rounded duration, an unknown rate with positive work, and a small positive rate rounded to zero.

The review also reproduced malformed `engine` and `http` timing sections being silently ignored. Present sections now require JSON objects; omitted legacy sections remain supported. Seven fixtures cover array/null sections and complete, incomplete, and aborted HTTP state. Both malformed-section counterexamples change from PASS/exit 0 to no measurement/exit 2.

The complete suite passes all 203 assertions. Running that same suite against the previous PR validator fails 15 assertions, including successful-sample selection.

See [decode results, commands, and tested source hashes](decode-rate/results.json), [the raw counterexample](decode-rate/counterexample.json), [previous-head regression output](decode-rate/baseline-regression.log), and [fixed regression output](decode-rate/fixed-regression.log).

The fixtures do not establish model recall quality. A live benchmark must preserve request IDs, checkpoint/configuration, capacity, raw response, and correctly separated timing fields.
