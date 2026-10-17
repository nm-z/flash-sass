# Request timing contract

The engine and HTTP frontend expose schema version 1. The canonical engine object comes from `kern/timings.h`; the frontend transports that object under `timings.engine` and records its own events under `timings.http`. CLI callers opt in with `timings 1` (`timings on` is accepted as an alias). `timings 0` or `timings off` disables unsolicited records. `timings` retrieves the current session's last record. An opted-in request emits one `timings { ... }` record before its terminal `done` or error; `gen` emits it before its human-readable result. The default token protocol remains compatible.

The engine's decimal-string `request_id` matches the `request` header in `trace 0` and `hosttrace`. It identifies a run within the current loaded engine's lifetime. These bounded traces describe the latest run, so capture them before submitting another request and reject an ID mismatch. HTTP forwards that engine ID unchanged and adds its own process-and-sequence `timings.http.request_id` to correlate the response with the later `flashd_request_timing` log.

## Counts and rates

Let P be the number of prompt tokens, R the positions restored from a snapshot, and C the last observed completed-position counter. A completed position produces its next-token output. Therefore the last prompt position produces the first generated token:

| Field | Definition |
| --- | --- |
| `prompt_tokens` | P, including the final prompt position. |
| `resumed_positions` | R, restored positions that this request does not execute. |
| `positions_completed` | C, the completed-position counter; includes R. |
| `positions_executed` | max(C - R, 0). |
| `prompt_positions_processed` | max(min(C, P) - R, 0). |
| `generated_tokens` | max(C - (P - 1), 0). |
| `decode_intervals` | max(generated_tokens - 1, 0). |

A protocol request `chat 3 10 11` executes five positions and normally produces four outputs. Its three post-first-output intervals must not be reported as four decode intervals. A stop token is a generated output even if the frontend hides its text.

`chat 0` and `gen 0` still compute the last prompt position and produce one output. Zero-output prompt-only work is represented by a request interrupted before that position completes; the protocol does not expose a separate successful generation-disabled prefill operation. The legacy `done` generated count uses the same nonnegative calculation as the canonical record.

All host durations are milliseconds from `CLOCK_MONOTONIC`. `build_ms` starts at `engine_run` entry and ends at the run-start marker after setup/program construction; it includes engine image preparation, snapshot setup and trace-buffer initialization. `run_ms` begins at that marker and ends after the persistent launches and final stream synchronization; it includes PLE staging, polling and token callbacks. `first_observed_ms` ends when the engine first observes and reads back an output; it is measured from that same run-start marker. `post_first_ms = run_ms - first_observed_ms` includes completion work after the final observed output. These engine intervals are not full HTTP request TTFT. Frontend image preparation, request parsing, queueing and tokenization occur outside the engine interval and remain in the enclosing HTTP request timing.

The prompt rate is `1000 * prompt_positions_processed / first_observed_ms`. The host decode rate is `1000 * decode_intervals / post_first_ms`. Rates require final known counts, no engine error, and a positive valid denominator. With fewer than two generated outputs the decode rate is null. The legacy engine aliases (`prompt_n`, `cache_n`, `predicted_n`, `prompt_ms`, `predicted_ms`, `decode_ms`, `prompt_per_second`, `predicted_per_second`) reuse these exact canonical values.

`counts_known` states whether a completed-position observation exists. `counts_final` distinguishes a drained final count from a partial observation retained during a device failure. Unknown counts serialize as null, including their aliases; they do not become zero or reuse the previous request's values. `error_code`, `aborted` and `stop_token_observed` are separate. The stop field is null when the stop flag could not be read. A malformed run command replaces the preceding request's trace identity and timing record even when it is rejected before `engine_run`.

## Device publication

Device timestamps use die 0's `SR_GLOBALTIMER`. The raw nanosecond stamps are decimal JSON strings so consumers cannot silently round a 64-bit integer through a JavaScript number. Subtract timestamps as integers before converting their difference to milliseconds.

`device_first_start_ns` is the phase-zero start of the first generated position, `device_first_publish_ns` is that position's output publication sample, and `device_last_traced_publish_ns` is the last covered publication sample. Publication is sampled after the completed-position store and its local synchronization, so it includes the instrumentation between that store and the clock read. `device_first_publish_ms` measures the first generated position's device work; it excludes earlier prompt positions. It is not complete request TTFT.

When the trace covers all G generated outputs, G >= 2, and the publication span is positive, `device_decode_tokens_per_second = (G - 1) * 1e9 / (last_publish_ns - first_publish_ns)`. Otherwise the device decode rate is null. `device_traced_tokens` identifies the actual coverage. Host and device epochs must never be subtracted from one another, and different dies must not be assumed to share an epoch. Host observation, GPU publication, socket observation and successful response writes are distinct events.

## HTTP timing

The HTTP origin is entry to the request worker, before it reads the headers and body. The frontend records request parsing/setup, queue wait, tokenization, engine setup/submission, first and last socket token observations, first text write and its final-record construction time. These named intervals can overlap; do not sum them as if they were disjoint phases.

| HTTP field | Start and end, or calculation |
| --- | --- |
| `request_read_ms` | Worker entry to chat handler entry; includes reading HTTP headers/body. |
| `prequeue_ms` | Chat handler entry to queue admission call; includes JSON validation, template rendering and frontend image preparation. |
| `queue_ms` | Time waiting inside the FIFO queue; zero for immediate admission. |
| `tokenization_ms` | Sum of tokenizer calls for the prompt and candidate snapshot prefix. |
| `setup_ms` | Queue grant to completion of the engine command write/flush; includes tokenization, snapshot setup, image expansion and socket setup. |
| `engine_submitted_ms` | Worker entry to that command write/flush boundary. |
| `first_token_observed_ms`, `last_token_observed_ms` | Worker entry to parsing the first/last raw `tok` protocol line. |
| `first_text_send_ms`, `ttft_ms` | Worker entry to the first successful content/reasoning chunk write. |
| `first_observation_to_text_send_ms` | First raw socket token observation to that successful text write; includes decoding, hidden/empty outputs, buffering and the write. |
| `completion_record_created_ms` | Worker entry to construction of the final timing object. |
| `socket_decode_tokens_per_second` | `1000 * (raw_token_events - 1) / (last_token_observed_ms - first_token_observed_ms)`, for at least two outputs and a positive span on a complete request. |
| `cancel_drain_ms` | Half-close cancellation request to completion of the bounded attempt to read the terminal engine record; null when that operation is unavailable or unnecessary. |

For streaming responses, `first_text_send_ms` and `ttft_ms` end after the first successful content or reasoning chunk write. Role-only chunks, queue keepalives and hidden stop tokens do not count. They remain null for a non-streamed JSON response because its text has not been written when its payload is built. `response_write_complete_ms` is also null inside the response: that future event cannot truthfully be encoded in an already-built payload. The subsequent JSON `flashd_request_timing` log records when the final response write returned, along with `write_attempted`, failure and client-close flags. Its `response_write_returned_ms` is null if a disconnected stream skips the write. A successful server write does not prove that a client received or displayed the bytes.

`raw_token_events` counts protocol outputs independently of visible text, including in-flight outputs drained after cancellation. Socket decode rate uses the first/last socket observations and is null for incomplete, failed, aborted or capped responses. `complete` is false after an engine error, engine abort, frontend cap or detected disconnect. The same timing object is logged when a disconnected client cannot receive a final response. A legacy engine can reject either of the two initial option commands with its exact unknown-command response; that compatibility exception never suppresses a later run error. Without canonical timing support its engine object is null and the compatibility fields retain their documented legacy meaning.

The frontend negotiates `cancel_on_eof 1` and enables half-close cancellation only after the exact `ok: cancel_on_eof 1` acknowledgment. A capped or disconnected request then closes the command direction of the core socket and drains final tokens/timings/status for at most 60 seconds. The deadline applies to silence, continuing output and fragmented lines. `cancel_requested` and `cancel_drain_timed_out` report the frontend's action independently of the engine's actual `aborted` flag: the core can finish normally before cancellation arrives. Unsupported engines retain close-on-cancel behavior, with final canonical counts unavailable. Ordinary CLI sessions keep their input-EOF behavior unless they explicitly opt in; they can close a command pipe and continue reading output. A returned canceled response uses `finish_reason: "cancelled"`, and usage counts follow the engine's final generated count when known, even when some outputs were suppressed by the frontend cap.

If an admitted error occurs after queue-wait SSE headers have been sent, the frontend emits an error event in that stream. It does not send a second HTTP status line. Failed engine responses use an explicit error record and do not acquire a normal successful finish reason.

## Verification

The deterministic C fixtures cover count derivation, snapshot resume, no output, one output, multiple outputs, stop, abort, partial/failed observations, invalid requests, overflow-safe bounds, aliases and integer device timing. The HTTP fixture runs the actual frontend/tokenizer with a tiny generated tokenizer GGUF and a controlled Unix protocol engine. Fifteen scenarios are exercised through both JSON and SSE, for 30 transport cases, including cancellation during prompt processing, final-count draining after a frontend cap, older engines and deadlines for silent, continuously producing and fragmented canceled output. Helper checks also cover role/keepalive exclusion, successful text writes and errors after SSE admission. `trace_session_test` exercises the actual session parser and half-close callback without initializing a GPU, including timing opt-out and preservation of ordinary CLI input EOF.

```sh
make -C kern timings_test flashd_timing_test flashd
./kern/timings_test
perl kern/flashd_timing_test.pl
```

The real GPU check requires an owned core socket, this source's traced cubin, at least four traced output positions and the matching resident weight manifests. It verifies the resident engine's source revision, compares canonical integer device timestamps directly with the relevant die-0 trace rows, checks the host trace's request identity and observation/callback order, validates four-output and one-output rates, and checks snapshot resume, argument rejection and a paired prompt when enabled. For each output it retains the device start/publication, host readback observation, callback entry/return and local Unix-socket parse events.

```sh
perl kern/timings_core_check.pl --socket /tmp/owned-shadow.sock \
  --out /path/to/new-run-directory --source-sha FULL_SOURCE_SHA

perl kern/timings_http_real.pl --flashd /path/to/flashd --gguf /path/to/shard1.gguf \
  --socket /tmp/owned-shadow.sock --out /path/to/new-http-directory \
  --source-sha FULL_SOURCE_SHA
```

The host overhead report separates callback duration from readback-observation-to-callback-return and readback-observation-to-local-socket-parse times. The socket figure includes scheduling and the checker's parsing. The real HTTP checker captures the matching request's device/host traces and correlated post-write log. Its transport measurements use the frontend's own observation and successful-write events and separately report final-record-construction-to-write-return time, which includes serialization and logging. None of these subtracts host and device epochs. A difference between host and GPU interval lengths is a change in observation lag over the interval, not an absolute GPU-to-host latency. Tracing perturbs the measured kernel; these samples are not a claim about uninstrumented throughput.

Exact source, cubin, commands, raw protocol, fixtures and results are recorded under `validation/issue15/`. A fixture pass establishes schema and transport behavior; target-GPU runs and their recorded identities establish the measured device-event coverage. Neither establishes client-side network latency or a general model-performance improvement.
