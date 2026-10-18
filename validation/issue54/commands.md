# Issue 54 execution record

Implementation and CPU work ran on engi, from base commit
b97b6b94bc0aba994830396fadebe5f8acd082f7, branch
fix/profile-driven-placement. GPU captures were launched from engi over SSH to
archy. See engi-identity.json and the compressed raw captures for identities.

Before publication the branch was rebased onto then-current master
3d57d734d63aa74f167a6b5a02c616ddc018d2ff. The intervening upstream changes
touched qsa_test and draft-conflict automation, not the captured frontend or
collector source.

## CPU checks

Commands on engi:

    make -C kern flashd
    ruby kern/placement_capture_test.rb

Frontend build: exit 0. The existing flashd.c image-cache pathname snprintf
warnings were emitted; no frontend source was modified.

Capture suite: 9 checks passed, exit 0 (capture-cpu.tap).
The same suite passed on archy, exit 0. Its cancellation case requires upstream
closure in less than one second, before the fake engine's two-second deadline.

The selector suite passed 41 CPU-only checks, exit 0 (selector-cpu.tap):

    ruby kern/placement_select_test.rb
    make -C kern test-placement

The Make target passed all 50 checks, exit 0. The final offline evidence check:

    ruby kern/placement_select.rb validation/issue54/archy-baseline.json.gz \
      > validation/issue54/baseline-analysis.json

returned the expected **exit 1**, status partial_no_acceptance, no selected
candidate, and measurement_gates_passed=false. This is a measured failing
baseline, not a malformed capture or passing placement. `git diff --check`
also passed.

## Final resident baseline capture

Collector SHA256:
ed85e0e6d7d5cea686cf75b36a50b17eb95339e41a9bfc70cd057a365501b9b4

Temporary frontend SHA256:
133cdf34fb47639524e8bd7eba066f20616b68c504346b49a89f6361143e53a1

After copying the collector and the built C frontend to an isolated directory,
the following ran on archy:

    cd /home/nate/codex/flash-sass-issue54-run-20261018
    ruby placement_capture.rb \
      --flashd /home/nate/codex/flash-sass-issue54-run-20261018/flashd \
      --model /mnt/sentry-nfs/flash-next-q2/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
      --resident-pids 974018,974061,974103,988060,988067,988072,988080,1575007 \
      --output baseline-final.json --label archy-resident-baseline \
      > capture-final.log 2>&1
    issue54_rc=$?
    printf '%s\n' "$issue54_rc" > capture-final.exit
    exit "$issue54_rc"

**Observed exit code: 0.** Both the command adapter and capture-final.exit
recorded it. The resulting artifact reports status=complete, failures=[], and
12 complete native/HTTP captures.

The artifact was copied back and compressed with `gzip -n -9` as
archy-baseline.json.gz. At 00:54:53–00:55:09 UTC on 2026-10-18 it recorded:

- 10 measured requests and 2 warmups.
- Mean 35.915 tok/s; minimum 35.89, maximum 35.96.
- The same 38 native generated IDs in every measured reply.
- Maximum actq256 spread 2393.1 us, failing the <=500 us requirement.
- Mean of per-response maximum six-die phase spans: 27.981 ms.
- SM clocks 1177 MHz throughout observed telemetry; die 4 at 81–83 degrees C.
- Root links Gen3 x8, Gen3 x16, Gen2 x16.

The recorded resident PIDs and start times were unchanged in postflight.
Independent lslocks inspection after completion showed this collector no longer
held /tmp/flash-sass-engine-validation.lock. The temporary frontend was the only
service process this run started or stopped.

## Earlier attempts

The first launch failed before any socket/GPU work because archy's Ruby omitted
the csv gem (exit 1). The collector now parses the numeric telemetry and leading
PID/UUID columns directly, removing that dependency.

The initial completed run is retained as archy-baseline-initial.json.gz. It used
collector SHA256
fd77418796b098292b9524ac55f9b2259a24994a74cb5ea54a8cf0ba29ad6e46
and measured 35.923 tok/s with a maximum spread of 2388.0 us. Its local command
adapter timed out with status 124 while the remote capture waited for the shared
lock; the remote process continued, acquired the lock, completed all 12 replies,
and exited. Its shell exit code was not retained or inferred.

A tighter CPU cancellation check then identified Ruby's nil EOF return from
recv_nonblock. Handling that return directly prevents cancellation from waiting
for the socket deadline. The final collector was retested on both hosts and the
final baseline above was repeated after that correction, with its exit code
preserved. Only the final run is the baseline for subsequent candidate selection;
the older capture remains historical evidence.

## Interpretation

No alternative shares, attention map, or expert placement was loaded or tested.
Both runs reproduce the existing imbalance. The final result validates the
collection path and supplies a comparison baseline; it does not establish a
placement improvement or complete issue #54.
