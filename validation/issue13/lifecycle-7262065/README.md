# Trace lifecycle and paired-prefill validation

Executable engine source: `72620653d4bb3dc10e3cb405695550db7152bae7`.
The driver was added after that build; its SHA-256 is recorded in the manifest.

A fresh owned core imported the six existing holders on archy, ran all 48 layers,
used real disk PLE, a 512-position context, production QSA budget 512, VRAM KV,
a four-token trace window, and two-position prefill. It was terminated after testing.

Eight requests passed: normal, stop with snapshot resume, client disconnect,
oversized argument rejection, missing image rejection, healthy recovery, actual
paired prefill, and paired snapshot resume. Six die headers and the host header
agreed on request ID, status, abort/end flags and requested/completed positions.
Both invalid requests advanced the request ID and exposed unavailable traces
with that failed request's metadata rather than a prior request's trace.

The paired cases each processed seven positions, including two paired positions;
the resumed continuation matched the first request's four output IDs exactly.
Raw statehash responses are retained; the driver checks their position counters,
not a separate full-state equivalence acceptance criterion.

No device fault was deliberately injected. Immediate device-error metadata and
library recovery remain source-reviewed; these results do not retroactively
validate the older archived benchmark executable in issue #10. They are neither
an instrumentation-overhead measurement nor an all-CTA counter experiment.
