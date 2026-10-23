# Perl cancellation runner checks

The cancellation runner now uses Perl and the shared process/socket helpers. Syntax validation passed. `process.tap` records ten passing subprocess checks; `socket.tap` records fifteen checks for fragmented/coalesced messages, binary bytes, partial EOF, response-size limits, read timeout with preserved partial data and a blocked large-write timeout.

The model acceptance contract is unchanged: admit an exact successful scalar reference with matching executable/cubin hashes and configuration; start a private core after GPU coordination; cancel a 61-token prompt with chunk size four; require at most one chunk to execute; verify a subsequent 17-token request against the scalar tokens and state; reject reuse of the incomplete snapshot; and cleanly terminate only that owned core. Output directories must be new.

The language-port checks did not rerun the six-die model cancellation test. Existing model records retain their original commands and identities. To validate this runner on real hardware, follow `PAIR_TESTING.md` with a coordinated six-die reservation and the exact matching scalar reference. A successful transport fixture is not a substitute for that model gate.
