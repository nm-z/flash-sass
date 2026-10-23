# Checked CLI validation with the Perl runner

The replacement runner executed all 46 original invalid-argument cases against the same M60-built `qsa_test` and `attn_test` binaries identified in `cli-report.json`. Every case exited 2 while CUDA visibility was empty and an `LD_PRELOAD` guard made `cuInit` and `cuLaunchKernel` exit 99 if reached. No GPU execution is required for this rejection gate.

Reproduction from a built `kern` directory:

```sh
perl checked_cli_test.pl --out /new/cli-evidence
perl QsaEvidence_test.pl
```

The runner retains empty and whitespace arguments, huge integer and cumulative-capacity cases, timeouts, raw stdout/stderr, binary hashes, and nonzero exit status. The process helper uses a monotonic timeout and a private child process group. Its separate CPU fixtures exercise concurrent pipe draining, status and signal propagation, timeout failure, child working directory, and CSV quoting.

The C/CUDA implementation is unchanged by this language-only runner replacement. These negative cases establish rejection before CUDA initialization; they do not establish positive-path numerical parity.
