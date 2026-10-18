# Perl benchmark runner validation

The current paired benchmark and its tests use Perl core modules. `perl-cpu.tap` records 26 passing top-level checks, covering phase/timestamp ordering, publication-based intervals, PLE wait accounting, lifecycle failure rejection, matched configuration and launch provenance, CSV/gzip handling, subprocess output draining and exact file identities.

The generator was independently checked against the complete original 262,000-token array captured by the historical `paired-depth-trace3` job on Archy. Its first 32 IDs, complete-array canonical SHA-256 and original file SHA-256 are preserved in `prompt-fixture-identity.json`. The default CPU suite checks that golden identity, so new fixture bytes cannot silently replace the historical benchmark input.

Run:

```sh
perl kern/paired_depth_bench_test.pl
# Optional direct comparison with the original saved JSON array:
perl kern/paired_depth_bench_test.pl --prompt-fixture /path/to/original/prompt-ids.json
```

These checks do not run a model. The live runner continues to require explicitly supplied host/VRAM sockets, source/build/weight/launch records, full 48-layer QSA at budget 512, matching configurations, warm-up, at least five alternating pairs, exact paired output IDs and current successful lifecycle metadata. Historical missing lifecycle fields remain `legacy_unreported` only during saved-evidence analysis. The runner never reloads a core or weight holder.
